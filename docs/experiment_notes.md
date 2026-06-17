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
- The downloaded file is checked against a pinned SHA-256 digest before use.

## Modelling Setup

- The data is split into train, validation, and test sets with stratification.
- `StandardScaler` is fitted on the training split and reused for validation and
  test data.
- SMOTE is applied only to the training split.
- Logistic regression provides a simple baseline.
- The MLP comparison uses 1, 3, 7, and 12 hidden layers with 16 ReLU units per
  hidden layer.
- Each model's decision threshold is selected on validation F1, then applied to
  the held-out test split.
- Models are selected by validation AUC-ROC. Test metrics are reported after
  selection rather than used to choose the model.

## Result Summary

The single-split validation-selected run in the committed result set is
`MLP 3 hidden layers (No SMOTE)` with validation AUC-ROC `0.8471` for this
train/validation/test split. Its held-out test metrics are AUC-ROC `0.8113`,
positive recall `0.8519`, and positive F1 `0.6866`.

For audit, `MLP 1 hidden layer (SMOTE)` has the highest test AUC-ROC in this
run at `0.8339`, but it is not treated as the selected model because model
selection is based on validation results. The logistic-regression baseline is
close to the best MLP result, which is a useful signal for this small tabular
dataset. The deeper MLPs do not improve the test metrics consistently.
