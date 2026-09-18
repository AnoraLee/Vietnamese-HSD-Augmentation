import pytest
from pydantic import ValidationError

from api.schemas import PredictRequest, PredictResponse


def test_predict_request_rejects_blank_text():
    with pytest.raises(ValidationError):
        PredictRequest(text="")


def test_predict_response_contains_standardized_output():
    response = PredictResponse(
        text="câu gốc",
        text_cleaned="câu_gốc",
        label="CLEAN",
        confidence=0.91,
        probabilities={"CLEAN": 0.91, "OFFENSIVE": 0.07, "HATE": 0.02},
        latency_ms=12.34,
        model_used="combined",
    )

    assert response.text_cleaned == "câu_gốc"
    assert response.probabilities["CLEAN"] == 0.91
