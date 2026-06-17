# Portfolio Summary

This repository is best presented as a reproducible tabular machine-learning portfolio project. It uses a medical-style diabetes classification task to demonstrate model comparison, imbalance handling, validation-based model selection, and metric-aware reporting.

## What To Highlight

- TensorFlow/Keras MLP comparison across 1, 3, 7, and 12 hidden layers.
- Logistic regression baseline kept close to the neural-network results.
- SMOTE applied only to the training split.
- Validation AUC used for model selection, with held-out test metrics reported afterwards.
- Per-model decision threshold selected on validation F1.
- Positive-class recall, F1, specificity, and confusion-matrix counts reported alongside AUC.
- Data download protected by a pinned SHA-256 checksum.
- Package metadata, console script, pytest coverage, and Python 3.10/3.11 CI.

## What Not To Overclaim

- This is not a deployable clinical model.
- The Pima dataset is small and demographically narrow.
- The headline result is from one train/validation/test split, not repeated cross-validation.
- Logistic regression remains close to the MLPs, which is an important result rather than a weakness to hide.

## Interview Talking Points

- Why model selection should use validation metrics rather than the test set.
- Why medical-style classification should discuss recall, specificity, and threshold choice.
- Why SMOTE belongs inside the training split only.
- Why a small tabular dataset can favor simpler baselines over deeper neural networks.
