# Experiment Notes

This note records the main data and modelling choices behind the project.

## Data Handling

- The dataset is stored in LIBSVM sparse format, so feature values are loaded by
  feature index.
- The raw label `-1` is treated as the diabetes-positive class and mapped to
  `diabetes_positive=1`.
- The raw label `+1` is mapped to `diabetes_positive=0`.
- The external dataset is downloaded into `data/diabetes_scale` when the
  experiment script runs.

## Modelling Setup

- The data is split into train, validation, and test sets with stratification.
- `StandardScaler` is fitted on the training split and reused for validation and
  test data.
- SMOTE is applied only to the training split.
- Logistic regression provides a simple baseline.
- The MLP comparison uses 1, 3, 7, and 12 hidden layers with 16 ReLU units per
  hidden layer.

## Result Summary

The best run in the committed result set is `MLP 1 hidden layer (SMOTE)` with
AUC-ROC `0.8339`, positive recall `0.7963`, and positive F1 `0.7049`.

The logistic-regression baseline is close to the best MLP result, which is a
useful signal for this small tabular dataset. The deeper MLPs do not improve the
test metrics consistently.
