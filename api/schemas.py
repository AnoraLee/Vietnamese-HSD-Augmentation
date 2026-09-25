from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Vietnamese text to classify")
    model: str | None = Field(
        default=None,
        description="Experiment key to run (baseline / bt / eda / llm / combined). "
        "Falls back to the server's default MODEL_EXPERIMENT when omitted.",
    )


class TokenImportance(BaseModel):
    token: str
    score: float


class PredictResponse(BaseModel):
    text: str
    text_cleaned: str
    label: str  # CLEAN / OFFENSIVE / HATE
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    model_used: str
    token_importance: list[TokenImportance] = Field(default_factory=list)


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    model: str | None = Field(default=None, description="Same experiment keys as PredictRequest.")
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
    token_scores: list[TokenImportance]
    latency_ms: float


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    model: str | None = None


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


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