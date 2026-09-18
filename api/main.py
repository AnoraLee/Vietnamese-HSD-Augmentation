import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictRequest,
    PredictResponse,
)
from src.services.inference import HSDInferenceService 
from src.utils.constants import LABELS, HF_MODEL_IDS

MODEL_EXPERIMENT = os.getenv("MODEL_EXPERIMENT", "combined")

DEFAULT_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
logger = logging.getLogger(__name__)


def cors_origins_from_environment() -> list[str]:
    """Read allowed browser origins without hard-coding a future deployment URL."""
    configured_origins = os.getenv("CORS_ORIGINS")
    if not configured_origins:
        return DEFAULT_CORS_ORIGINS
    return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]

app = FastAPI(
    title="Vietnamese Hate Speech Detection API",
    description=f"Serving the '{MODEL_EXPERIMENT}' PhoBERT model.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_from_environment(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

_inference_service: HSDInferenceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _inference_service
    if MODEL_EXPERIMENT not in HF_MODEL_IDS:
        raise RuntimeError(
            f"Unknown MODEL_EXPERIMENT={MODEL_EXPERIMENT!r}. "
            f"Expected one of {list(HF_MODEL_IDS)}."
        )

    hf_model_id = HF_MODEL_IDS[MODEL_EXPERIMENT]
    _inference_service = HSDInferenceService(hf_model_id)
    print(f"Loaded experiment from Hugging Face: {hf_model_id}")
    
    yield 

app = FastAPI(
    title="Vietnamese Hate Speech Detection API",
    description=f"Serving the '{MODEL_EXPERIMENT}' PhoBERT model.",
    lifespan=lifespan,
)


def _predict_one(text: str) -> PredictResponse:
    if _inference_service is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": "Model is not loaded yet."},
        )
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_text", "message": "Text must contain non-whitespace characters."},
        )

    try:
        result = _inference_service.predict(text)
    except PreprocessingSetupError as exc:
        logger.exception("Preprocessing runtime is unavailable.")
        raise HTTPException(
            status_code=503,
            detail={"code": "preprocessing_unavailable", "message": str(exc)},
        ) from exc
    except RuntimeError as exc:
        logger.exception("Model inference failed.")
        raise HTTPException(
            status_code=500,
            detail={"code": "inference_failed", "message": "Inference could not be completed."},
        ) from exc

    return PredictResponse(**result, model_used=MODEL_EXPERIMENT)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _inference_service is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": "Model is not loaded yet."},
        )
    return HealthResponse(status="ok", model=MODEL_EXPERIMENT, device=_inference_service.device)


@app.get("/metadata", response_model=ModelMetadataResponse)
def metadata() -> ModelMetadataResponse:
    if _inference_service is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": "Model is not loaded yet."},
        )
    return ModelMetadataResponse(
        model=MODEL_EXPERIMENT,
        device=_inference_service.device,
        labels=LABELS,
        max_length=_inference_service.max_length,
        preprocessing="VnCoreNLP word segmentation with teencode normalization",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return _predict_one(request.text)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    return BatchPredictResponse(results=[_predict_one(text) for text in request.texts])
