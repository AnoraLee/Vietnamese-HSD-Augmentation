from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.utils.constants import LABELS
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ExplainRequest,
    ExplainResponse,
    PredictRequest,
    PredictResponse,
)

def test_predict_request_rejects_empty_string():
    with pytest.raises(ValidationError):
        PredictRequest(text="")


def test_predict_request_rejects_whitespace_only():
    """strip_whitespace=True → '   ' thành '' → fail min_length=1."""
    with pytest.raises(ValidationError):
        PredictRequest(text="   ")


def test_predict_request_strips_and_accepts():
    req = PredictRequest(text="  xin chào  ")
    assert req.text == "xin chào"
    assert req.model is None


def test_predict_request_rejects_too_long():
    with pytest.raises(ValidationError):
        PredictRequest(text="a" * 2001)


def test_predict_request_accepts_model_key():
    req = PredictRequest(text="abc", model="phobert_baseline")
    assert req.model == "phobert_baseline"


def test_predict_response_contains_standardized_output():
    response = PredictResponse(
        text="câu gốc",
        text_cleaned="câu_gốc",
        label="CLEAN",
        predicted_id=0,
        confidence=0.91,
        probabilities={"CLEAN": 0.91, "OFFENSIVE": 0.07, "HATE": 0.02},
        latency_ms=12.34,
        model_used="phobert_combined",
    )
    assert response.text_cleaned == "câu_gốc"
    assert response.probabilities["CLEAN"] == 0.91
    assert response.predicted_id == 0


def test_predict_response_requires_predicted_id():
    with pytest.raises(ValidationError):
        PredictResponse(
            text="x", text_cleaned="x", label="CLEAN",
            confidence=0.9,
            probabilities={"CLEAN": 1.0, "OFFENSIVE": 0.0, "HATE": 0.0},
            latency_ms=1.0, model_used="phobert_combined",
        )


def test_batch_predict_request_rejects_empty_list():
    with pytest.raises(ValidationError):
        BatchPredictRequest(texts=[])


def test_batch_predict_request_rejects_blank_element():
    with pytest.raises(ValidationError):
        BatchPredictRequest(texts=["ok", "   "])


def test_batch_predict_request_rejects_too_many():
    with pytest.raises(ValidationError):
        BatchPredictRequest(texts=["a"] * 101)


def test_batch_predict_request_extra_field_forbidden():
    """`extra='forbid'` -- typo field phải bị từ chối."""
    with pytest.raises(ValidationError):
        BatchPredictRequest(texts=["a"], txts=["b"])

def test_explain_request_defaults_max_evals_none():
    req = ExplainRequest(text="abc")
    assert req.max_evals is None


def test_explain_request_rejects_max_evals_below_minimum():
    with pytest.raises(ValidationError):
        ExplainRequest(text="abc", max_evals=5)


def test_explain_request_rejects_max_evals_above_maximum():
    with pytest.raises(ValidationError):
        ExplainRequest(text="abc", max_evals=2000)


def test_explain_response_defaults_method_to_shap():
    resp = ExplainResponse(
        label="CLEAN",
        model_used="phobert_combined",
        token_scores=[{"token": "abc", "score": 0.5}],
        latency_ms=10.0,
    )
    assert resp.method == "shap"