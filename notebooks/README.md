# Notebooks

This folder contains the review notebook for the medical MLP classification
experiment.

The reusable experiment logic lives in `src/medical_mlp_classification/`.
Run `python scripts/run_experiment.py` from the repository root to regenerate
the metrics and plots used by the notebook.

Use the editable package install from the repository root for core experiment
reproduction:

```bash
python -m pip install -e ".[dev]"
```

Install notebook extras only when opening or editing the notebook interactively:

```bash
python -m pip install -e ".[notebook]"
```
