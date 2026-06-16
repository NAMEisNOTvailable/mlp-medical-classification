# Deep Learning MLP - Medical Classification

Machine learning coursework project for binary medical classification using multilayer perceptron models.

## Portfolio Summary

This project explores the Pima diabetes prediction task with a focus on preprocessing, imbalanced classification, model depth, and evaluation quality.

Key work:

- Built MLP classification pipelines for diabetes prediction.
- Compared models with **1, 3, 7, and 12 hidden layers**.
- Tested training with and without **SMOTE** for class imbalance handling.
- Used mean imputation, `StandardScaler`, Adam optimisation, and early stopping.
- Evaluated models with AUC-ROC, accuracy, precision, recall, F1, and confusion matrices.
- Found that the best SMOTE model reached **AUC-ROC 0.8105**, while deeper 12-layer models showed overfitting risk.

## Skills Demonstrated

- Python machine learning workflow
- TensorFlow/Keras or scikit-learn style experimentation
- Imbalanced classification and SMOTE
- Model comparison and overfitting analysis
- Medical classification evaluation metrics
- Clear experiment documentation

## Why This Matters

For health-related classification tasks, accuracy alone can be misleading. This project emphasises recall, precision, AUC-ROC, and false-negative awareness so the model is assessed through a risk-aware lens rather than a single headline score.

## Status

Academic portfolio project. README added to make the repository easier for recruiters and reviewers to understand quickly.