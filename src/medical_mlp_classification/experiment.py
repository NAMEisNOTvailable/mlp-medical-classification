"""Reproducible experiment pipeline for Pima diabetes MLP classification."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import urllib.request
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale"
DATA_FILENAME = "diabetes_scale"
TARGET_NAME = "diabetes_positive"
FEATURE_NAMES = [
    "num_pregnant",
    "plasma_glucose",
    "diastolic_bp",
    "triceps_skin_fold",
    "two_hour_serum_insulin",
    "bmi",
    "diabetes_pedigree",
    "age",
]
ENVIRONMENT_PACKAGES = [
    "tensorflow",
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "imbalanced-learn",
]


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("results")
    layers: tuple[int, ...] = (1, 3, 7, 12)
    epochs: int = 100
    patience: int = 5
    batch_size: int = 32
    random_state: int = 42
    test_size: float = 0.20
    validation_size: float = 0.30
    threshold: float = 0.50
    make_plots: bool = True


@dataclass(frozen=True)
class DatasetMetadata:
    rows: int
    features: int
    raw_label_counts: dict[str, int]
    target_counts: dict[str, int]
    positive_label_mapping: str
    data_url: str


def ensure_data_file(data_dir: Path, *, force_download: bool = False) -> Path:
    """Download the LIBSVM-format dataset when it is not present locally."""

    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / DATA_FILENAME
    if data_path.exists() and not force_download:
        return data_path

    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "mlp-medical-classification/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data_path.write_bytes(response.read())
    return data_path


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series, DatasetMetadata]:
    """Load and correctly align the sparse LIBSVM diabetes dataset.

    The LIBSVM file uses feature indices, so missing feature entries must be
    aligned by index, not positional order. The original Pima data
    has 268 diabetes-positive records. In this scaled LIBSVM version, that
    minority class is encoded as -1, so we map raw label -1 to target 1.
    """

    X_sparse, y_raw = load_svmlight_file(str(data_path), n_features=len(FEATURE_NAMES))
    X = pd.DataFrame(X_sparse.toarray(), columns=FEATURE_NAMES)

    y_raw_int = y_raw.astype(int)
    y = pd.Series((y_raw_int == -1).astype(int), name=TARGET_NAME)

    raw_counts = pd.Series(y_raw_int).value_counts().sort_index()
    target_counts = y.value_counts().sort_index()
    metadata = DatasetMetadata(
        rows=int(X.shape[0]),
        features=int(X.shape[1]),
        raw_label_counts={str(int(k)): int(v) for k, v in raw_counts.items()},
        target_counts={str(int(k)): int(v) for k, v in target_counts.items()},
        positive_label_mapping="raw LIBSVM label -1 -> diabetes_positive=1",
        data_url=DATA_URL,
    )
    return X, y, metadata


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    config: ExperimentConfig,
) -> dict[str, np.ndarray]:
    """Create stratified train/validation/test splits and fit scaling on train only."""

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=config.validation_size,
        random_state=config.random_state,
        stratify=y_train_full,
    )

    scaler = StandardScaler()
    return {
        "X_train": scaler.fit_transform(X_train),
        "X_val": scaler.transform(X_val),
        "X_test": scaler.transform(X_test),
        "y_train": y_train.to_numpy(),
        "y_val": y_val.to_numpy(),
        "y_test": y_test.to_numpy(),
    }


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    enabled: bool,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to the training split only."""

    if not enabled:
        return X_train, y_train
    sampler = SMOTE(random_state=random_state)
    return sampler.fit_resample(X_train, y_train)


def set_reproducible_seed(seed: int) -> None:
    """Set the random seeds used by Python, NumPy, and TensorFlow."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_mlp(input_dim: int, hidden_layers: int, learning_rate: float = 0.001):
    """Build the original assignment architecture with a configurable depth."""

    import tensorflow as tf

    model = tf.keras.Sequential(name=f"mlp_{hidden_layers}_hidden_layers")
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for _ in range(hidden_layers):
        model.add(tf.keras.layers.Dense(16, activation="relu"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_predictions(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    *,
    threshold: float,
    model_name: str,
    model_family: str,
    sampling: str,
    hidden_layers: int | None = None,
    epochs_run: int | None = None,
) -> dict[str, float | int | str | None]:
    """Compute threshold-aware and ranking-aware metrics."""

    y_probability = np.asarray(y_probability).reshape(-1)
    y_pred = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "model": model_name,
        "model_family": model_family,
        "sampling": sampling,
        "hidden_layers": hidden_layers,
        "epochs_run": epochs_run,
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_probability),
        "precision_positive": precision_score(y_true, y_pred, zero_division=0),
        "recall_positive": recall_score(y_true, y_pred, zero_division=0),
        "f1_positive": f1_score(y_true, y_pred, zero_division=0),
        "specificity_negative": specificity,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def run_baselines(
    splits: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> tuple[list[dict[str, float | int | str | None]], dict[str, np.ndarray]]:
    """Train simple baselines so the MLP comparison has context."""

    rows: list[dict[str, float | int | str | None]] = []
    probabilities: dict[str, np.ndarray] = {}

    for use_smote in (False, True):
        sampling = "SMOTE" if use_smote else "No SMOTE"
        X_train, y_train = apply_smote(
            splits["X_train"],
            splits["y_train"],
            enabled=use_smote,
            random_state=config.random_state,
        )
        model = LogisticRegression(max_iter=1000, random_state=config.random_state)
        model.fit(X_train, y_train)
        proba = model.predict_proba(splits["X_test"])[:, 1]
        name = f"Logistic Regression ({sampling})"
        probabilities[name] = proba
        rows.append(
            evaluate_predictions(
                splits["y_test"],
                proba,
                threshold=config.threshold,
                model_name=name,
                model_family="baseline",
                sampling=sampling,
            )
        )
    return rows, probabilities


def run_mlp_models(
    splits: dict[str, np.ndarray],
    config: ExperimentConfig,
) -> tuple[list[dict[str, float | int | str | None]], dict[str, np.ndarray]]:
    """Train the MLP model family with and without SMOTE."""

    import tensorflow as tf

    rows: list[dict[str, float | int | str | None]] = []
    probabilities: dict[str, np.ndarray] = {}

    for use_smote in (False, True):
        sampling = "SMOTE" if use_smote else "No SMOTE"
        X_train, y_train = apply_smote(
            splits["X_train"],
            splits["y_train"],
            enabled=use_smote,
            random_state=config.random_state,
        )

        for hidden_layers in config.layers:
            seed = config.random_state + hidden_layers + (1000 if use_smote else 0)
            set_reproducible_seed(seed)
            model = build_mlp(X_train.shape[1], hidden_layers)
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.patience,
                restore_best_weights=True,
            )
            history = model.fit(
                X_train,
                y_train,
                validation_data=(splits["X_val"], splits["y_val"]),
                epochs=config.epochs,
                batch_size=config.batch_size,
                callbacks=[early_stopping],
                verbose=0,
            )
            proba = model.predict(splits["X_test"], verbose=0).reshape(-1)
            name = f"MLP {hidden_layers} hidden layer{'s' if hidden_layers != 1 else ''} ({sampling})"
            probabilities[name] = proba
            rows.append(
                evaluate_predictions(
                    splits["y_test"],
                    proba,
                    threshold=config.threshold,
                    model_name=name,
                    model_family="mlp",
                    sampling=sampling,
                    hidden_layers=hidden_layers,
                    epochs_run=len(history.history["loss"]),
                )
            )
    return rows, probabilities


def save_plots(
    results: pd.DataFrame,
    probabilities: dict[str, np.ndarray],
    y_test: np.ndarray,
    output_dir: Path,
) -> None:
    """Write compact comparison plots for portfolio review."""

    output_dir.mkdir(parents=True, exist_ok=True)

    ordered = results.sort_values("auc_roc", ascending=True)
    labels = ordered["model"].tolist()
    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y_pos, ordered["auc_roc"], color="#2f6f73")
    ax.set_yticks(y_pos, labels)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("AUC-ROC")
    ax.set_title("Model comparison by AUC-ROC")
    fig.tight_layout()
    fig.savefig(output_dir / "auc_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} ({auc:.3f})", linewidth=1.4)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curves.png", dpi=160)
    plt.close(fig)


def collect_environment() -> dict[str, object]:
    """Collect runtime versions that affect reproducibility."""

    packages: dict[str, str] = {}
    for package_name in ENVIRONMENT_PACKAGES:
        try:
            packages[package_name] = version(package_name)
        except PackageNotFoundError:
            packages[package_name] = "not installed"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }


def run_experiment(config: ExperimentConfig, *, force_download: bool = False) -> pd.DataFrame:
    """Run the full experiment and write reproducible outputs."""

    data_path = ensure_data_file(config.data_dir, force_download=force_download)
    X, y, metadata = load_dataset(data_path)
    splits = split_and_scale(X, y, config)

    baseline_rows, baseline_probabilities = run_baselines(splits, config)
    mlp_rows, mlp_probabilities = run_mlp_models(splits, config)

    results = pd.DataFrame([*baseline_rows, *mlp_rows]).sort_values(
        ["auc_roc", "accuracy"],
        ascending=[False, False],
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output_dir / "model_comparison.csv", index=False)

    summary = {
        "dataset": asdict(metadata),
        "config": {
            **asdict(config),
            "data_dir": str(config.data_dir),
            "output_dir": str(config.output_dir),
            "layers": list(config.layers),
        },
        "split_sizes": {
            "train": int(len(splits["y_train"])),
            "validation": int(len(splits["y_val"])),
            "test": int(len(splits["y_test"])),
        },
        "environment": collect_environment(),
        "best_by_auc": results.iloc[0].to_dict(),
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if config.make_plots:
        save_plots(
            results,
            {**baseline_probabilities, **mlp_probabilities},
            splits["y_test"],
            config.output_dir,
        )

    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 3, 7, 12])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a minimal MLP depth and short training loop for CI smoke checks.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    layers = tuple(args.layers)
    epochs = args.epochs
    patience = args.patience
    if args.quick:
        layers = (1,)
        epochs = min(args.epochs, 3)
        patience = min(args.patience, 2)

    config = ExperimentConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        layers=layers,
        epochs=epochs,
        patience=patience,
        batch_size=args.batch_size,
        random_state=args.seed,
        threshold=args.threshold,
        make_plots=not args.no_plots,
    )
    results = run_experiment(config, force_download=args.force_download)
    display_columns = [
        "model",
        "accuracy",
        "auc_roc",
        "precision_positive",
        "recall_positive",
        "f1_positive",
        "false_negative",
        "true_positive",
    ]
    print(results[display_columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
