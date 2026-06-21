# Results

Generated experiment outputs are written here by:

```bash
python scripts/run_experiment.py
```

The committed CSV and JSON files are small review artifacts for inspecting the
reproduced metrics without rerunning TensorFlow immediately.

`model_comparison.csv` is sorted by validation AUC-ROC. The unprefixed metric
columns are final held-out test metrics kept for backwards-readable summaries;
the same values are also available with the explicit `test_` prefix. Validation
metrics and selected thresholds are included so model selection can be reviewed
without using the test split as the selector.

The plot artifacts include validation AUC comparison, held-out test ROC curves,
and held-out test precision-recall curves.
