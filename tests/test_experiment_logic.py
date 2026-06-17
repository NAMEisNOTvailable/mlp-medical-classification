from __future__ import annotations

import numpy as np
import pytest

from medical_mlp_classification.experiment import (
    ExperimentConfig,
    apply_smote,
    evaluate_candidate,
    file_sha256,
    select_threshold,
    verify_data_file,
)


def test_select_threshold_maximizes_validation_f1():
    y_validation = np.array([0, 0, 1, 1])
    validation_probability = np.array([0.2, 0.4, 0.35, 0.9])

    threshold = select_threshold(
        y_validation,
        validation_probability,
        strategy="max_f1",
    )

    assert threshold == pytest.approx(0.35)


def test_evaluate_candidate_uses_validation_threshold_for_test_metrics():
    config = ExperimentConfig(threshold=None, threshold_strategy="max_f1")
    y_validation = np.array([0, 0, 1, 1])
    validation_probability = np.array([0.2, 0.4, 0.35, 0.9])
    y_test = np.array([0, 1, 1, 0])
    test_probability = np.array([0.1, 0.3, 0.8, 0.6])

    row = evaluate_candidate(
        y_validation=y_validation,
        validation_probability=validation_probability,
        y_test=y_test,
        test_probability=test_probability,
        config=config,
        model_name="example",
        model_family="test",
        sampling="No SMOTE",
    )

    assert row["selection_threshold"] == pytest.approx(0.35)
    assert row["threshold_strategy"] == "max_f1"
    assert row["validation_f1_positive"] == pytest.approx(0.8)
    assert row["test_false_negative"] == 1
    assert row["false_negative"] == row["test_false_negative"]


def test_apply_smote_balances_training_labels():
    X_train = np.array([[float(i), float(i % 3)] for i in range(18)])
    y_train = np.array([0] * 12 + [1] * 6)

    X_resampled, y_resampled = apply_smote(
        X_train,
        y_train,
        enabled=True,
        random_state=42,
    )

    assert X_resampled.shape[0] == 24
    assert np.bincount(y_resampled).tolist() == [12, 12]


def test_data_file_checksum_failure_is_explicit(tmp_path):
    data_path = tmp_path / "diabetes_scale"
    data_path.write_text("not the reviewed dataset", encoding="utf-8")

    assert file_sha256(data_path)
    with pytest.raises(ValueError, match="Unexpected SHA-256"):
        verify_data_file(data_path)
