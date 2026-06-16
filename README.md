# Medical MLP Classification

Binary medical classification project using multilayer perceptron models, imbalance handling, and metric-aware evaluation on the Pima diabetes prediction task.

## Project Snapshot

| Area | Summary |
| --- | --- |
| Task | Diabetes outcome classification |
| Model family | Multilayer perceptrons with different depths |
| Imbalance method | SMOTE |
| Best reported result | AUC-ROC 0.8105 for the best SMOTE model |
| Main artefact | [`notebooks/medical_mlp_classification.ipynb`](notebooks/medical_mlp_classification.ipynb) |

## What This Demonstrates

- Built a preprocessing and classification workflow for tabular medical data.
- Compared MLP models with 1, 3, 7, and 12 hidden layers.
- Applied imputation, standardisation, Adam optimisation, early stopping, and SMOTE.
- Evaluated models using AUC-ROC, accuracy, precision, recall, F1, and confusion matrices.
- Treated false negatives and class imbalance as evaluation risks instead of relying on accuracy alone.

## Evaluation Focus

Medical classification should be reviewed through risk-aware metrics:

| Metric | Why It Matters |
| --- | --- |
| Recall | Helps identify missed positive cases |
| Precision | Helps understand false-alarm cost |
| F1 | Balances precision and recall |
| AUC-ROC | Measures ranking quality across thresholds |
| Confusion matrix | Shows error types directly |

## Repository Structure

```text
notebooks/   Main experiment notebook
README.md    Portfolio overview and result summary
```

## Skills Shown

- Tabular machine-learning workflow
- Imbalanced classification
- MLP architecture comparison
- SMOTE and preprocessing pipelines
- Metric-aware model evaluation
- Clear experiment documentation

## Status

Academic portfolio project. The repository is organised so reviewers can understand the modelling objective and evaluation approach before opening the notebook.
