import os
import sys
import logging
from pathlib import Path
from time import perf_counter
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictRequest,
    PredictResponse,
)
from src.services.inference import HSDInferenceService
from src.utils.constants import LABELS, HF_MODEL_IDS, EXPERIMENT_ORDER

# Default experiment used when a request doesn't specify one -- keeps the
# API backward-compatible with callers that only send {"text": "..."}.
MODEL_EXPERIMENT = os.getenv("MODEL_EXPERIMENT", "combined")

DEFAULT_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
logger = logging.getLogger(__name__)


def cors_origins_from_environment() -> list[str]:
    """Read allowed browser origins without hard-coding a future deployment URL."""
    configured_origins = os.getenv("CORS_ORIGINS")
    if not configured_origins:
        return DEFAULT_CORS_ORIGINS
    return [origin.strip() for origin in configured_origins.split(",") if origin.strip()]


# Keyed by experiment name (see EXPERIMENT_ORDER) -- populated once at
# startup. Running locally, so loading all checkpoints up front is fine;
# revisit with lazy load/unload if this ever needs to run under a RAM cap.
_inference_services: dict[str, HSDInferenceService] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_EXPERIMENT not in HF_MODEL_IDS:
        raise RuntimeError(
            f"Unknown MODEL_EXPERIMENT={MODEL_EXPERIMENT!r}. "
            f"Expected one of {list(HF_MODEL_IDS)}."
        )

    for experiment_name in [MODEL_EXPERIMENT]:
        hf_model_id = HF_MODEL_IDS[experiment_name]
        print(f"[{experiment_name}] loading from Hugging Face: {hf_model_id}")
        service = HSDInferenceService(hf_model_id)
        service.warm_up()
        if experiment_name == MODEL_EXPERIMENT:
            service.explain_with_shap("không độc hại", max_evals=20, predicted_id=0)
        _inference_services[experiment_name] = service
        print(f"[{experiment_name}] ready.")

    print(
        f"Loaded {len(_inference_services)} model(s): {list(_inference_services)}. "
        f"Default: {MODEL_EXPERIMENT!r}"
    )

    yield

    _inference_services.clear()


# Khởi tạo FastAPI duy nhất 1 lần
app = FastAPI(
    title="Vietnamese Hate Speech Detection API",
    description=f"Serving all PhoBERT experiments: {', '.join(EXPERIMENT_ORDER)}.",
    lifespan=lifespan,
)

# Thêm Middleware ngay sau khi khởi tạo app
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_from_environment(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def _resolve_service(model_key: str | None) -> tuple[str, HSDInferenceService]:
    """Pick the requested experiment, falling back to MODEL_EXPERIMENT.
    Raises a clear 400 (not a crash) if the caller asks for an unknown key."""
    resolved_key = model_key or MODEL_EXPERIMENT
    if resolved_key not in _inference_services:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "unknown_model",
                "message": f"Unknown model {resolved_key!r}. Expected one of {EXPERIMENT_ORDER}.",
            },
        )
    return resolved_key, _inference_services[resolved_key]


def _predict_one(text: str, model_key: str | None = None) -> PredictResponse:
    if not _inference_services:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": "Models are not loaded yet."},
        )
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_text", "message": "Text must contain non-whitespace characters."},
        )

    resolved_key, service = _resolve_service(model_key)

    try:
        result = service.predict(text)
    except RuntimeError as exc:
        logger.exception("Model inference failed.")
        raise HTTPException(
            status_code=500,
            detail={"code": "inference_failed", "message": "Inference could not be completed."},
        ) from exc

    return PredictResponse(**result, model_used=resolved_key)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if not _inference_services:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": "Models are not loaded yet."},
        )
    default_service = _inference_services[MODEL_EXPERIMENT]
    return HealthResponse(
        status="ok",
        model=MODEL_EXPERIMENT,
        device=default_service.device,
        available_models=list(_inference_services),
    )


@app.get("/metadata", response_model=ModelMetadataResponse)
def metadata() -> ModelMetadataResponse:
    if not _inference_services:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": "Models are not loaded yet."},
        )
    default_service = _inference_services[MODEL_EXPERIMENT]
    return ModelMetadataResponse(
        model=MODEL_EXPERIMENT,
        device=default_service.device,
        labels=LABELS,
        max_length=default_service.max_length,
        preprocessing="Underthesea word segmentation with teencode normalization",
        available_models=list(_inference_services),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return _predict_one(request.text, request.model)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(request: BatchPredictRequest):
    return BatchPredictResponse(
        results=[_predict_one(text, request.model) for text in request.texts]
    )

def _default_max_evals(device: str) -> int:
    """SHAP on CPU with PhoBERT-base is roughly 15-30x slower than on GPU.
    Pick a budget that keeps response time tolerable on the actual device."""
    return 200 if device.startswith("cuda") else 80


@app.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest):
    """SHAP-based explanation -- opt-in only.

    Runs hundreds of forward passes, so never call this from /predict.
    The frontend triggers it explicitly via a disclosure button, and
    toggles it back off without re-fetching (results are cached client-side).
    """
    if not _inference_services:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": "Models are not loaded yet."},
        )
    if not request.text.strip():
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_text", "message": "Text must contain non-whitespace characters."},
        )

    started_at = perf_counter()
    resolved_key, service = _resolve_service(request.model)

    max_evals = request.max_evals or _default_max_evals(service.device)

    try:
        prediction = service.predict(request.text)
        predicted_id = LABELS.index(prediction["label"])
        token_scores = service.explain_with_shap(
            request.text,
            max_evals=max_evals,
            predicted_id=predicted_id,
        )
    except RuntimeError as exc:
        logger.exception("SHAP explanation failed.")
        raise HTTPException(
            status_code=500,
            detail={"code": "explain_failed", "message": str(exc) or "Explanation could not be completed."},
        ) from exc

    return ExplainResponse(
        label=prediction["label"],
        token_scores=token_scores,
        latency_ms=round((perf_counter() - started_at) * 1000, 2),
    )