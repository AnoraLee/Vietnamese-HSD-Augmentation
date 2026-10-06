from __future__ import annotations

import json

import pytest

from src.utils.constants import EXPERIMENT_ORDER, HF_MODEL_IDS, LABELS
from src.utils.evaluation import (
    _metrics_to_row,
    available_experiments,
    compute_confusion_matrix,
    get_misclassified_examples,
    load_all_metrics,
)

def test_load_all_metrics_reads_local_directory(fake_metrics_dir):
    metrics_dir, used = fake_metrics_dir
    df = load_all_metrics(metrics_dir)

    assert set(df["experiment"]) == set(used)
    assert "accuracy" in df.columns
    assert "macro_f1" in df.columns
    assert "weighted_f1" in df.columns


def test_load_all_metrics_skips_missing_experiments(tmp_path):
    """Chỉ có 1 file → chỉ trả 1 dòng, không raise."""
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    exp = EXPERIMENT_ORDER[0]
    with (metrics_dir / f"metrics_{exp}.json").open("w") as f:
        json.dump({"test_accuracy": 0.93}, f)

    df = load_all_metrics(metrics_dir)
    assert list(df["experiment"]) == [exp]


def test_load_all_metrics_raises_when_nothing_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_all_metrics(tmp_path)


def test_metrics_to_row_strips_only_prefix():
    """`test_hate_f1` → `hate_f1`; `test_hate_test_f1` → `hate_test_f1` (không phải `hate_f1`)."""
    row = _metrics_to_row("exp", {
        "test_accuracy": 0.9,
        "test_hate_test_f1": 0.8,  
        "loss": 0.1,            
    })
    assert row["experiment"] == "exp"
    assert row["accuracy"] == 0.9
    assert row["hate_test_f1"] == 0.8
    assert "loss" not in row


def test_available_experiments_returns_full_order():
    """Vì có sanity check ở constants, mọi experiment đều có model ID."""
    assert available_experiments() == EXPERIMENT_ORDER
    for exp in available_experiments():
        assert exp in HF_MODEL_IDS


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

    assert len(get_misclassified_examples(texts, y_true, y_pred, id2label, n=3)) == 3


def test_get_misclassified_examples_n_zero_returns_empty():
    """Regression: `n=0` trả empty, không trả all (bug `if n else ...` cũ)."""
    id2label = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
    texts = ["a", "b"]
    y_true = [1, 1]
    y_pred = [0, 0]

    assert len(get_misclassified_examples(texts, y_true, y_pred, id2label, n=0)) == 0