from __future__ import annotations

from medical_mlp_classification.experiment import FEATURE_NAMES, TARGET_NAME, load_dataset


def test_load_dataset_aligns_sparse_feature_indices(tmp_path):
    data_path = tmp_path / "sample.libsvm"
    data_path.write_text(
        "\n".join(
            [
                "-1 1:0.1 2:0.2 3:0.3 4:0.4 5:0.5 6:0.6 7:0.7 8:0.8",
                "+1 1:-0.1 2:-0.2 4:-0.4 5:-0.5 6:-0.6 7:-0.7 8:-0.8",
            ]
        ),
        encoding="utf-8",
    )

    X, y, metadata = load_dataset(data_path)

    assert X.shape == (2, len(FEATURE_NAMES))
    assert list(X.columns) == FEATURE_NAMES
    assert X.loc[1, "diastolic_bp"] == 0.0
    assert y.name == TARGET_NAME
    assert y.tolist() == [1, 0]
    assert metadata.positive_label_mapping == "raw LIBSVM label -1 -> diabetes_positive=1"
