from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Text = Annotated[str, StringConstraints(min_length=1, max_length=2000, strip_whitespace=True)]


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class PredictRequest(BaseModel):
    text: Text = Field(..., description="Vietnamese text to classify")
    model: str | None = Field(
        default=None,
        description=(
            "Experiment key to run. Falls back to the server's default "
            "MODEL_EXPERIMENT when omitted. Use /metadata to see available keys."
        ),
    )


class TokenImportance(BaseModel):
    token: str
    score: float


class PredictResponse(BaseModel):
    text: str
    text_cleaned: str
    label: str
    predicted_id: int
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    model_used: str
    # token_importance đã xóa -- SHAP là endpoint riêng (/explain).


class ExplainRequest(BaseModel):
    text: Text
    model: str | None = Field(default=None, description="Same keys as PredictRequest.")
    max_evals: int | None = Field(
        default=None,
        ge=20,
        le=1000,
        description=(
            "SHAP budget (số forward pass tối đa). "
            "Bỏ trống → backend tự chọn theo device. "
            "Khuyến nghị: 50 (CPU yếu), 100 (CPU khá), 200-300 (GPU)."
        ),
    )


class ExplainResponse(BaseModel):
    label: str
    method: str = "shap"
    model_used: str
    token_scores: list[TokenImportance]
    latency_ms: float


class BatchPredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    texts: list[Text] = Field(..., min_length=1, max_length=100)
    model: str | None = None


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    model_used: str
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    available_models: list[str] = Field(default_factory=list)


class ModelMetadataResponse(BaseModel):
    model: str
    device: str
    labels: list[str]
    max_length: int
    preprocessing: str
    available_models: list[str] = Field(default_factory=list)