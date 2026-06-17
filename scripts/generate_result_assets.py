from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _short_name(name: str) -> str:
    if name.startswith("MLP 3 hidden layers"):
        return "Selected\nMLP 3 no SMOTE"
    if name.startswith("MLP 1 hidden layer"):
        return "Audit best test\nMLP 1 + SMOTE"
    if name.startswith("Logistic Regression"):
        return "Baseline\nLogistic"
    return name


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _baseline_from_rows(rows: list[dict[str, str]]) -> dict[str, float | str]:
    baseline_rows = [row for row in rows if row["model"] == "Logistic Regression (No SMOTE)"]
    if not baseline_rows:
        baseline_rows = [row for row in rows if row["model"].startswith("Logistic Regression")]
    if not baseline_rows:
        raise ValueError("No logistic-regression baseline row found")
    row = baseline_rows[0]
    return {
        "model": row["model"],
        "validation_auc_roc": float(row["validation_auc_roc"]),
        "test_auc_roc": float(row["test_auc_roc"]),
        "test_recall_positive": float(row["test_recall_positive"]),
        "test_f1_positive": float(row["test_f1_positive"]),
    }


def generate_overview_chart(summary_path: Path, metrics_path: Path, output_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = _load_csv_rows(metrics_path)
    selected = summary["selected_by_validation_auc"]
    audit_best = summary["best_by_test_auc_for_audit"]
    baseline = _baseline_from_rows(rows)
    models = [selected, audit_best, baseline]

    labels = [_short_name(str(model["model"])) for model in models]
    validation_auc = [float(model["validation_auc_roc"]) for model in models]
    test_auc = [float(model["test_auc_roc"]) for model in models]
    test_recall = [float(model["test_recall_positive"]) for model in models]
    test_f1 = [float(model["test_f1_positive"]) for model in models]

    fig, (auc_axis, test_axis) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=160)
    x_positions = list(range(len(labels)))
    width = 0.34

    auc_axis.bar([x - width / 2 for x in x_positions], validation_auc, width, label="Validation AUC", color="#2563eb")
    auc_axis.bar([x + width / 2 for x in x_positions], test_auc, width, label="Held-out test AUC", color="#0f766e")
    auc_axis.set_xticks(x_positions)
    auc_axis.set_xticklabels(labels)
    auc_axis.set_ylim(0.72, 0.88)
    auc_axis.set_title("Selection metric vs final audit")
    auc_axis.set_ylabel("AUC-ROC")
    auc_axis.grid(axis="y", linestyle=":", alpha=0.35)
    auc_axis.legend(loc="lower left", fontsize=8)
    for x, value in zip([x - width / 2 for x in x_positions], validation_auc):
        auc_axis.text(x, value + 0.004, f"{value:.3f}", ha="center", fontsize=8)
    for x, value in zip([x + width / 2 for x in x_positions], test_auc):
        auc_axis.text(x, value + 0.004, f"{value:.3f}", ha="center", fontsize=8)

    test_axis.bar([x - width / 2 for x in x_positions], test_recall, width, label="Test recall", color="#f97316")
    test_axis.bar([x + width / 2 for x in x_positions], test_f1, width, label="Test F1", color="#64748b")
    test_axis.set_xticks(x_positions)
    test_axis.set_xticklabels(labels)
    test_axis.set_ylim(0.50, 0.90)
    test_axis.set_title("Positive-class test metrics")
    test_axis.set_ylabel("Score")
    test_axis.grid(axis="y", linestyle=":", alpha=0.35)
    test_axis.legend(loc="lower left", fontsize=8)
    for x, value in zip([x - width / 2 for x in x_positions], test_recall):
        test_axis.text(x, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)
    for x, value in zip([x + width / 2 for x in x_positions], test_f1):
        test_axis.text(x, value + 0.008, f"{value:.3f}", ha="center", fontsize=8)

    fig.suptitle("Medical MLP selected run overview", fontsize=12, fontweight="bold")
    fig.text(
        0.5,
        0.015,
        "Model selection uses validation AUC; test metrics are reported after selection. Single train/validation/test split.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate portfolio result assets.")
    parser.add_argument("--summary", type=Path, default=Path("results/summary.json"))
    parser.add_argument("--metrics", type=Path, default=Path("results/model_comparison.csv"))
    parser.add_argument("--output", type=Path, default=Path("assets/model_selection_overview.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_overview_chart(args.summary, args.metrics, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
