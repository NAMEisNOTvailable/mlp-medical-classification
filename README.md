# Medical MLP Classification

This project explores diabetes classification on the scaled Pima Indians
Diabetes dataset. It compares logistic regression with multilayer perceptrons
of different depths, then measures how SMOTE changes recall, F1, and AUC-ROC
for the diabetes-positive class.

## Project Snapshot

| Area | Summary |
| --- | --- |
| Task | Diabetes-positive classification |
| Dataset | LIBSVM `diabetes_scale`, 768 rows, 8 scaled features |
| Positive target | Raw LIBSVM label `-1` mapped to `diabetes_positive=1` |
| Models | Logistic regression baseline; MLPs with 1, 3, 7, and 12 hidden layers |
| Imbalance handling | SMOTE on the training split |
| Best result | MLP 1 hidden layer + SMOTE, AUC-ROC 0.8339 |
| Main command | `python scripts/run_experiment.py` |

## Method

The experiment loads the LIBSVM dataset with sparse feature indices preserved,
maps the minority diabetes class to the positive target, and uses stratified
train, validation, and test splits. Feature scaling is fitted on the training
split, and SMOTE is applied only to training data.

The model comparison includes logistic regression as a baseline and four MLP
depths: 1, 3, 7, and 12 hidden layers. Each model is evaluated with accuracy,
AUC-ROC, positive-class precision, positive-class recall, F1, specificity, and
confusion-matrix counts.

## Results

These results were generated with `python scripts/run_experiment.py`.

| Model | Sampling | Accuracy | AUC-ROC | Positive recall | Positive F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| MLP 1 hidden layer | SMOTE | 0.7662 | 0.8339 | 0.7963 | 0.7049 |
| Logistic Regression | No SMOTE | 0.7273 | 0.8267 | 0.5185 | 0.5714 |
| MLP 1 hidden layer | No SMOTE | 0.7338 | 0.8263 | 0.5741 | 0.6019 |
| Logistic Regression | SMOTE | 0.7273 | 0.8230 | 0.6852 | 0.6379 |
| MLP 3 hidden layers | No SMOTE | 0.7273 | 0.8113 | 0.5926 | 0.6038 |
| MLP 7 hidden layers | SMOTE | 0.7208 | 0.8074 | 0.7037 | 0.6387 |
| MLP 3 hidden layers | SMOTE | 0.7013 | 0.7917 | 0.6481 | 0.6034 |
| MLP 7 hidden layers | No SMOTE | 0.7338 | 0.7857 | 0.6111 | 0.6168 |
| MLP 12 hidden layers | No SMOTE | 0.7208 | 0.7800 | 0.6296 | 0.6126 |
| MLP 12 hidden layers | SMOTE | 0.6818 | 0.7615 | 0.6111 | 0.5739 |

Full metrics are stored in [`results/model_comparison.csv`](results/model_comparison.csv).
Run metadata is stored in [`results/summary.json`](results/summary.json).

The strongest run is the 1-hidden-layer MLP with SMOTE. Logistic regression is
close behind, which is expected for a small tabular dataset with only eight
features. The deeper MLPs do not improve the result consistently.

## Reproduce

Install the core experiment dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

For the exact Windows CPU environment used to verify the committed results, use
the lock file:

```bash
.\.venv\Scripts\python -m pip install -r requirements-lock.txt
```

If Windows reports a TensorFlow long-path installation error, create the virtual
environment in a shorter path:

```bash
python -m venv C:\venvs\mlp-medical
C:\venvs\mlp-medical\Scripts\python.exe -m pip install -r requirements.txt
```

Run a short CPU check:

```bash
python scripts/run_experiment.py --quick --no-plots --output-dir results/smoke
```

Run the full experiment:

```bash
python scripts/run_experiment.py
```

## Repository Structure

```text
data/                         Data source note; downloaded raw data is ignored
notebooks/                    Notebook-facing report entry point
results/                      Reproduced metrics and plots
scripts/run_experiment.py     CLI entry point
src/medical_mlp_classification/
                              Reusable experiment code
tests/                        Data-loading regression tests
```

## Notebook

The notebook is a compact report view for the generated results. Install the
optional notebook dependencies only when opening it interactively:

```bash
python -m pip install -r requirements-notebook.txt
jupyter notebook notebooks/medical_mlp_classification.ipynb
```

## Background

This repository is based on a university deep-learning assignment. The project
version here keeps the original comparison of MLP depths and SMOTE, while using
a reproducible script-based workflow and the LIBSVM label mapping documented in
the dataset notes.

## Limitations

This is a small academic experiment, not a clinical model. The Pima dataset is
demographically narrow and does not provide the evidence needed for medical
deployment. The value of this project is the preprocessing, class-imbalance
handling, model comparison, and evaluation workflow.

## License and Data

Original project code and documentation are licensed under the MIT License. The
Pima diabetes dataset is external to this repository; follow the source dataset
terms when downloading or reusing it.
