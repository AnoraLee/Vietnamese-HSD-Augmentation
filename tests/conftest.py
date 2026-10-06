from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Xóa các env var có thể làm test flaky."""
    for key in ("HF_TOKEN", "HSD_PROJECT_DIR", "MODEL_EXPERIMENT", "LOADED_EXPERIMENTS"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def project_dir() -> Path:
    return PROJECT_DIR


@pytest.fixture
def fake_metrics_dir(tmp_path):
    """Tạo folder metrics/ với 2 experiment đầu trong EXPERIMENT_ORDER."""
    from src.utils.constants import EXPERIMENT_ORDER

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()

    used = EXPERIMENT_ORDER[:2]
    for i, exp in enumerate(used):
        payload = {
            "test_accuracy": 0.9 + i * 0.01,
            "test_macro_f1": 0.7 + i * 0.05,
            "test_weighted_f1": 0.72 + i * 0.05,
            "test_hate_f1": 0.6 + i * 0.03,
        }
        with (metrics_dir / f"metrics_{exp}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    return metrics_dir, used


class FakeInferenceService:
    """Thay thế HSDInferenceService trong test API.

    Có đủ method/property mà `api/main.py` gọi:
        device, max_length, preprocessing_description,
        predict, predict_batch, warm_up, explain_with_shap.
    """

    device = "cpu"
    max_length = 128
    preprocessing_description = "Fake preprocessing (test only)"

    def __init__(self, label: str = "CLEAN") -> None:
        self.label = label

    def _payload(self, text: str) -> dict:
        from src.utils.constants import LABELS, label2id

        pid = label2id[self.label]
        probs = {l: 0.02 for l in LABELS}
        probs[self.label] = 0.94
        return {
            "text": text,
            "text_cleaned": "câu_đã_phân_từ",
            "label": self.label,
            "predicted_id": pid,
            "confidence": probs[self.label],
            "probabilities": probs,
            "latency_ms": 5.5,
            "token_importance": [],
        }

    def predict(self, text: str) -> dict:
        return self._payload(text)

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self._payload(t) for t in texts]

    def warm_up(self) -> None:
        return None

    def explain_with_shap(self, text: str, max_evals: int = 80, predicted_id: int | None = None):
        return [
            {"token": "câu", "score": 0.5},
            {"token": "kiểm_thử", "score": -0.2},
        ]


@pytest.fixture
def fake_service():
    return FakeInferenceService()