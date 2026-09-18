from fastapi.testclient import TestClient

import api.main as api_main


class FakeInferenceService:
    device = "cpu"
    max_length = 128

    def predict(self, text: str) -> dict:
        return {
            "text": text,
            "text_cleaned": "câu_đã_phân_từ",
            "label": "CLEAN",
            "confidence": 0.9,
            "probabilities": {"CLEAN": 0.9, "OFFENSIVE": 0.08, "HATE": 0.02},
            "latency_ms": 5.5,
        }


def client_with_fake_service(monkeypatch) -> TestClient:
    monkeypatch.setattr(api_main, "_inference_service", FakeInferenceService())
    return TestClient(api_main.app)


def test_health_and_metadata_expose_runtime_contract(monkeypatch):
    client = client_with_fake_service(monkeypatch)

    health = client.get("/health")
    metadata = client.get("/metadata")

    assert health.status_code == 200
    assert health.json()["device"] == "cpu"
    assert metadata.status_code == 200
    assert metadata.json()["labels"] == ["CLEAN", "OFFENSIVE", "HATE"]


def test_predict_returns_standardized_payload_and_cors_header(monkeypatch):
    client = client_with_fake_service(monkeypatch)

    response = client.post(
        "/predict",
        json={"text": "câu kiểm thử"},
    )

    assert response.status_code == 200
    assert response.json()["text_cleaned"] == "câu_đã_phân_từ"

def test_predict_rejects_whitespace_only_text(monkeypatch):
    client = client_with_fake_service(monkeypatch)

    response = client.post("/predict", json={"text": "   "})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_text"
