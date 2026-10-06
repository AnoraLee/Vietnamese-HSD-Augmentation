from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from src.utils.constants import HF_DEFAULT_EXPERIMENT, LABELS


@pytest.fixture
def client(monkeypatch, fake_service):
    from src.utils.constants import EXPERIMENT_ORDER

    second = next(e for e in EXPERIMENT_ORDER if e != HF_DEFAULT_EXPERIMENT)
    services = {HF_DEFAULT_EXPERIMENT: fake_service, second: fake_service}
    monkeypatch.setattr(api_main, "_inference_services", services)

    return TestClient(api_main.app)


@pytest.fixture
def second_experiment():
    from src.utils.constants import EXPERIMENT_ORDER

    return next(e for e in EXPERIMENT_ORDER if e != HF_DEFAULT_EXPERIMENT)

def test_health_returns_runtime_contract(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["device"] == "cpu"
    assert body["model"] == HF_DEFAULT_EXPERIMENT
    assert set(body["available_models"]) >= {HF_DEFAULT_EXPERIMENT}


def test_metadata_exposes_labels_and_preprocessing(client):
    r = client.get("/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["labels"] == LABELS
    assert body["max_length"] == 128
    assert body["preprocessing"]
    assert body["model"] == HF_DEFAULT_EXPERIMENT

def test_predict_returns_standardized_payload(client):
    r = client.post("/predict", json={"text": "câu kiểm thử"})
    assert r.status_code == 200
    body = r.json()
    assert body["text_cleaned"] == "câu_đã_phân_từ"
    assert body["label"] in LABELS
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["model_used"] == HF_DEFAULT_EXPERIMENT
    assert "predicted_id" in body
    assert set(body["probabilities"]) == set(LABELS)


def test_predict_uses_requested_model(client, second_experiment):
    r = client.post("/predict", json={"text": "abc", "model": second_experiment})
    assert r.status_code == 200
    assert r.json()["model_used"] == second_experiment


def test_predict_rejects_blank_text(client):
    """Pydantic `strip_whitespace=True` → '   ' thành '' → fail min_length."""
    r = client.post("/predict", json={"text": "   "})
    assert r.status_code == 422


def test_predict_rejects_empty_text(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_rejects_unknown_model(client):
    r = client.post("/predict", json={"text": "abc", "model": "không_tồn_tại"})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert detail["code"] == "unknown_model"
    assert "Available" in detail["message"]


def test_predict_returns_503_when_service_not_loaded(monkeypatch):
    monkeypatch.setattr(api_main, "_inference_services", {})
    client = TestClient(api_main.app)
    r = client.post("/predict", json={"text": "abc"})
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "model_unavailable"

def test_predict_batch_returns_results(client):
    r = client.post("/predict/batch", json={"texts": ["a", "b", "c"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3
    assert body["model_used"] == HF_DEFAULT_EXPERIMENT
    assert body["total_latency_ms"] >= 0


def test_predict_batch_rejects_empty_list(client):
    r = client.post("/predict/batch", json={"texts": []})
    assert r.status_code == 422


def test_predict_batch_rejects_blank_element(client):
    r = client.post("/predict/batch", json={"texts": ["ok", "   "]})
    assert r.status_code == 422

def test_explain_returns_token_scores(client):
    r = client.post("/explain", json={"text": "câu kiểm thử", "max_evals": 20})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in LABELS
    assert body["model_used"] == HF_DEFAULT_EXPERIMENT
    assert body["method"] == "shap"
    assert isinstance(body["token_scores"], list)


def test_explain_rejects_too_small_max_evals(client):
    r = client.post("/explain", json={"text": "abc", "max_evals": 5})
    assert r.status_code == 422


def test_explain_rejects_blank_text(client):
    r = client.post("/explain", json={"text": "   "})
    assert r.status_code == 422