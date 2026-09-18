import json

from src.utils.evaluation import (
    load_all_metrics, available_experiments,
    compute_confusion_matrix, get_misclassified_examples,
)
from src.utils.constants import LABELS, HF_MODEL_IDS


def test_load_all_metrics_reads_metrics_directory(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    for exp, metrics in [("baseline", {"test_accuracy": 0.9, "test_macro_f1": 0.7}),
                          ("bt", {"test_accuracy": 0.91, "test_macro_f1": 0.75})]:
        with open(metrics_dir / f"metrics_{exp}.json", "w") as f:
            json.dump(metrics, f)

    result = load_all_metrics(tmp_path)

    assert set(result["experiment"]) == {"baseline", "bt"}
    assert "accuracy" in result.columns
    assert "macro_f1" in result.columns


def test_load_all_metrics_skips_missing_experiments(tmp_path):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    with open(metrics_dir / "metrics_combined.json", "w") as f:
        json.dump({"test_accuracy": 0.93}, f)

    result = load_all_metrics(tmp_path)
    assert list(result["experiment"]) == ["combined"]


def test_load_all_metrics_raises_when_nothing_found(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_all_metrics(tmp_path)


def test_available_experiments_only_lists_downloaded_models(tmp_path):
    result = available_experiments(tmp_path)

    assert "bt" in result
    assert "combined" in result
    assert "baseline" in result

def test_compute_confusion_matrix_shape():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]

    cm = compute_confusion_matrix(y_true, y_pred)

    assert cm.shape == (len(LABELS), len(LABELS))
    assert cm.sum() == len(y_true)


def test_get_misclassified_examples_filters_correctly():
    id2label = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
    texts = ["câu đúng", "câu sai", "câu đúng nữa"]
    y_true = [0, 1, 2]
    y_pred = [0, 0, 2]

    errors = get_misclassified_examples(texts, y_true, y_pred, id2label)

    assert len(errors) == 1
    assert errors.iloc[0]["text"] == "câu sai"
    assert errors.iloc[0]["true_label"] == "OFFENSIVE"
    assert errors.iloc[0]["predicted_label"] == "CLEAN"


def test_get_misclassified_examples_respects_n_limit():
    id2label = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
    texts = [f"câu {i}" for i in range(10)]
    y_true = [1] * 10
    y_pred = [0] * 10

    errors = get_misclassified_examples(texts, y_true, y_pred, id2label, n=3)

    assert len(errors) == 3
