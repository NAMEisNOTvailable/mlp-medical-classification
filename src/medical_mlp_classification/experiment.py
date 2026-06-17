"""Reproducible experiment pipeline for Pima diabetes MLP classification."""

from __future__ import annotations

import argparse
import hashlib
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
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes_scale"
DATA_FILENAME = "diabetes_scale"
DATA_SHA256 = "0c07eb4c49e7a8ffb9c9f25095ac3022df2ca85b0dcb7d294c3ddea69f392cba"
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
METRIC_FIELDS = [
    "accuracy",
    "auc_roc",
    "precision_positive",
    "recall_positive",
    "f1_positive",
    "specificity_negative",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
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
    threshold: float | None = None
    threshold_strategy: str = "max_f1"
    make_plots: bool = True


@dataclass(frozen=True)
class DatasetMetadata:
    rows: int
    features: int
    raw_label_counts: dict[str, int]
    target_counts: dict[str, int]
    positive_label_mapping: str
    data_url: str
    data_sha256: str


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file without loading it all at once."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_data_file(data_path: Path) -> None:
    """Fail fast when the external dataset no longer matches the reviewed copy."""

    actual = file_sha256(data_path)
    if actual != DATA_SHA256:
        raise ValueError(
            f"Unexpected SHA-256 for {data_path}: {actual}. "
            f"Expected {DATA_SHA256}. Delete the file and rerun only after "
            "confirming the upstream dataset is still the intended source."
        )


def ensure_data_file(data_dir: Path, *, force_download: bool = False) -> Path:
    """Download the LIBSVM-format dataset when it is not present locally."""

    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / DATA_FILENAME
    if data_path.exists() and not force_download:
        verify_data_file(data_path)
        return data_path

    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "mlp-medical-classification/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data_path.write_bytes(response.read())
    verify_data_file(data_path)
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
        data_sha256=file_sha256(data_path),
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


def select_threshold(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    *,
    strategy: str,
) -> float:
    """Choose a decision threshold from validation predictions only."""

    if strategy != "max_f1":
        raise ValueError(f"Unsupported threshold strategy: {strategy}")

    y_probability = np.asarray(y_probability).reshape(-1)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_probability)))

    best_threshold = 0.5
    best_score = (-1.0, -1.0, -1.0)
    for threshold in candidates:
        y_pred = (y_probability >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        score = (f1, recall, specificity)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def metric_values(metrics: dict[str, float | int | str | None]) -> dict[str, float | int]:
    """Keep only numeric metric fields from an evaluation row."""

    return {field: metrics[field] for field in METRIC_FIELDS}


def evaluate_candidate(
    *,
    y_validation: np.ndarray,
    validation_probability: np.ndarray,
    y_test: np.ndarray,
    test_probability: np.ndarray,
    config: ExperimentConfig,
    model_name: str,
    model_family: str,
    sampling: str,
    hidden_layers: int | None = None,
    epochs_run: int | None = None,
) -> dict[str, float | int | str | None]:
    """Evaluate one trained model using validation-only threshold selection."""

    if config.threshold is None:
        threshold = select_threshold(
            y_validation,
            validation_probability,
            strategy=config.threshold_strategy,
        )
        threshold_strategy = config.threshold_strategy
    else:
        threshold = float(config.threshold)
        threshold_strategy = "fixed"

    validation_metrics = evaluate_predictions(
        y_validation,
        validation_probability,
        threshold=threshold,
        model_name=model_name,
        model_family=model_family,
        sampling=sampling,
        hidden_layers=hidden_layers,
        epochs_run=epochs_run,
    )
    test_metrics = evaluate_predictions(
        y_test,
        test_probability,
        threshold=threshold,
        model_name=model_name,
        model_family=model_family,
        sampling=sampling,
        hidden_layers=hidden_layers,
        epochs_run=epochs_run,
    )

    row: dict[str, float | int | str | None] = {
        "model": model_name,
        "model_family": model_family,
        "sampling": sampling,
        "hidden_layers": hidden_layers,
        "epochs_run": epochs_run,
        "selection_threshold": threshold,
        "threshold_strategy": threshold_strategy,
    }
    row.update(
        {
            f"validation_{key}": value
            for key, value in metric_values(validation_metrics).items()
        }
    )
    row.update(
        {f"test_{key}": value for key, value in metric_values(test_metrics).items()}
    )

    # Backward-compatible aliases: these are final test metrics.
    row.update(metric_values(test_metrics))
    return row


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
        validation_proba = model.predict_proba(splits["X_val"])[:, 1]
        test_proba = model.predict_proba(splits["X_test"])[:, 1]
        name = f"Logistic Regression ({sampling})"
        probabilities[name] = test_proba
        rows.append(
            evaluate_candidate(
                y_validation=splits["y_val"],
                validation_probability=validation_proba,
                y_test=splits["y_test"],
                test_probability=test_proba,
                config=config,
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
            validation_proba = model.predict(splits["X_val"], verbose=0).reshape(-1)
            test_proba = model.predict(splits["X_test"], verbose=0).reshape(-1)
            name = f"MLP {hidden_layers} hidden layer{'s' if hidden_layers != 1 else ''} ({sampling})"
            probabilities[name] = test_proba
            rows.append(
                evaluate_candidate(
                    y_validation=splits["y_val"],
                    validation_probability=validation_proba,
                    y_test=splits["y_test"],
                    test_probability=test_proba,
                    config=config,
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

    ordered = results.sort_values("validation_auc_roc", ascending=True)
    labels = ordered["model"].tolist()
    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y_pos, ordered["validation_auc_roc"], color="#2f6f73")
    ax.set_yticks(y_pos, labels)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Validation AUC-ROC")
    ax.set_title("Model selection by validation AUC-ROC")
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

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, proba in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        average_precision = average_precision_score(y_test, proba)
        ax.plot(
            recall,
            precision,
            label=f"{name} (AP {average_precision:.3f})",
            linewidth=1.4,
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves")
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curves.png", dpi=160)
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
        ["validation_auc_roc", "validation_f1_positive", "test_auc_roc"],
        ascending=[False, False, False],
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.output_dir / "model_comparison.csv", index=False)

    selected_by_validation = results.iloc[0].to_dict()
    best_by_test = results.sort_values(
        ["test_auc_roc", "test_accuracy"],
        ascending=[False, False],
    ).iloc[0].to_dict()

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
        "model_selection": {
            "selection_split": "validation",
            "selection_metric": "validation_auc_roc",
            "threshold_strategy": config.threshold_strategy if config.threshold is None else "fixed",
        },
        "selected_by_validation_auc": selected_by_validation,
        "best_by_test_auc_for_audit": best_by_test,
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Use a fixed decision threshold. Omit to select the threshold on validation F1.",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=["max_f1"],
        default="max_f1",
        help="Validation-only threshold strategy used when --threshold is omitted.",
    )
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
        threshold_strategy=args.threshold_strategy,
        make_plots=not args.no_plots,
    )
    results = run_experiment(config, force_download=args.force_download)
    display_columns = [
        "model",
        "validation_auc_roc",
        "selection_threshold",
        "auc_roc",
        "recall_positive",
        "f1_positive",
        "false_negative",
        "true_positive",
    ]
    print(results[display_columns].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
