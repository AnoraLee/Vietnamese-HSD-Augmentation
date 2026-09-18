from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Vietnamese text to classify")


class PredictResponse(BaseModel):
    text: str
    text_cleaned: str
    label: str  # CLEAN / OFFENSIVE / HATE
    confidence: float
    probabilities: dict[str, float]
    latency_ms: float
    model_used: str


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


class ModelMetadataResponse(BaseModel):
    model: str
    device: str
    labels: list[str]
    max_length: int
    preprocessing: str
