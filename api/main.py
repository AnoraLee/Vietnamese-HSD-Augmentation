import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from api.evaluation_routes import router as evaluation_router

from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    ModelMetadataResponse,
    PredictRequest,
    PredictResponse,
)
from src.services.inference import HSDInferenceService
from src.utils.constants import EXPERIMENT_ORDER, HF_DEFAULT_EXPERIMENT, HF_MODEL_IDS, LABELS

# Không hardcode "combined" nữa -- lấy từ config (default: phobert_combined).
MODEL_EXPERIMENT = os.getenv("MODEL_EXPERIMENT", HF_DEFAULT_EXPERIMENT)
_LOADED_ENV = os.getenv("LOADED_EXPERIMENTS", "").strip()

DEFAULT_CORS_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
logger = logging.getLogger(__name__)

# Keyed by experiment name -- populated once at startup.
_inference_services: dict[str, HSDInferenceService] = {}


def cors_origins_from_environment() -> list[str]:
    configured = os.getenv("CORS_ORIGINS")
    if not configured:
        return DEFAULT_CORS_ORIGINS
    return [o.strip() for o in configured.split(",") if o.strip()]


def _experiments_to_load() -> list[str]:
    """Chọn experiment để load. Mặc định: chỉ MODEL_EXPERIMENT.

    Muốn load nhiều model (đổi giữa các experiment qua `model` field):
        LOADED_EXPERIMENTS=phobert_baseline,phobert_combined
    """
    if _LOADED_ENV:
        requested = [e.strip() for e in _LOADED_ENV.split(",") if e.strip()]
        unknown = [e for e in requested if e not in HF_MODEL_IDS]
        if unknown:
            raise RuntimeError(f"Unknown experiments in LOADED_EXPERIMENTS: {unknown}")
    else:
        requested = [MODEL_EXPERIMENT]

    if MODEL_EXPERIMENT not in requested:
        requested.append(MODEL_EXPERIMENT)
    return requested


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_EXPERIMENT not in HF_MODEL_IDS:
        raise RuntimeError(
            f"Unknown MODEL_EXPERIMENT={MODEL_EXPERIMENT!r}. "
            f"Expected one of {list(HF_MODEL_IDS)}."
        )

    experiments = _experiments_to_load()
    failed: list[str] = []

    for experiment_name in experiments:
        hf_model_id = HF_MODEL_IDS[experiment_name]
        logger.info("[%s] loading from Hugging Face: %s", experiment_name, hf_model_id)
        try:
            service = HSDInferenceService(hf_model_id)
            service.warm_up()
            # Warm up SHAP only for the default model -- frontend hits it first.
            if experiment_name == MODEL_EXPERIMENT:
                service.explain_with_shap("không độc hại", max_evals=20, predicted_id=0)
            _inference_services[experiment_name] = service
            logger.info("[%s] ready.", experiment_name)
        except Exception:
            logger.exception("Failed to load experiment %r", experiment_name)
            failed.append(experiment_name)

    if MODEL_EXPERIMENT not in _inference_services:
        raise RuntimeError(f"Default MODEL_EXPERIMENT={MODEL_EXPERIMENT!r} failed to load.")
    if failed:
        logger.warning("Failed to load experiments: %s", failed)

    logger.info(
        "Loaded %d model(s): %s. Default: %r",
        len(_inference_services), list(_inference_services), MODEL_EXPERIMENT,
    )

    yield
    _inference_services.clear()


app = FastAPI(
    title="Vietnamese Hate Speech Detection API",
    description=f"Serving experiments: {', '.join(EXPERIMENT_ORDER)}.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins_from_environment(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

app.include_router(evaluation_router)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_services() -> None:
    if not _inference_services:
        raise HTTPException(
            status_code=503,
            detail={"code": "model_unavailable", "message": "Models are not loaded yet."},
        )


def _resolve_service(model_key: str | None) -> tuple[str, HSDInferenceService]:
    resolved_key = model_key or MODEL_EXPERIMENT
    if resolved_key not in _inference_services:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "unknown_model",
                "message": (
                    f"Unknown or unavailable model {resolved_key!r}. "
                    f"Available: {list(_inference_services)}."
                ),
            },
        )
    return resolved_key, _inference_services[resolved_key]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "name": "Vietnamese Hate Speech Detection API",
        "version": "1.0",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "/health",
            "/metadata",
            "/predict",
            "/predict/batch",
            "/explain",
            "/evaluation/metrics",
        ],
    }

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    _require_services()
    default_service = _inference_services[MODEL_EXPERIMENT]
    return HealthResponse(
        status="ok",
        model=MODEL_EXPERIMENT,
        device=default_service.device,
        available_models=list(_inference_services),
    )


@app.get("/metadata", response_model=ModelMetadataResponse)
def metadata() -> ModelMetadataResponse:
    _require_services()
    default_service = _inference_services[MODEL_EXPERIMENT]
    return ModelMetadataResponse(
        model=MODEL_EXPERIMENT,
        device=default_service.device,
        labels=LABELS,
        max_length=default_service.max_length,
        preprocessing=default_service.preprocessing_description,
        available_models=list(_inference_services),
    )


@app.post("/predict", response_model=PredictResponse,
          responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def predict(request: PredictRequest):
    _require_services()
    resolved_key, service = _resolve_service(request.model)

    try:
        result = service.predict(request.text)
    except Exception as exc:
        logger.exception("Model inference failed.")
        raise HTTPException(
            status_code=500,
            detail={"code": "inference_failed", "message": "Inference could not be completed."},
        ) from exc

    return PredictResponse(**result, model_used=resolved_key)


@app.post("/predict/batch", response_model=BatchPredictResponse,
          responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def predict_batch(request: BatchPredictRequest):
    _require_services()
    resolved_key, service = _resolve_service(request.model)

    started_at = perf_counter()
    try:
        results = service.predict_batch(request.texts)
    except Exception as exc:
        logger.exception("Batch inference failed.")
        raise HTTPException(
            status_code=500,
            detail={"code": "inference_failed", "message": "Batch inference could not be completed."},
        ) from exc

    return BatchPredictResponse(
        results=[PredictResponse(**r, model_used=resolved_key) for r in results],
        model_used=resolved_key,
        total_latency_ms=round((perf_counter() - started_at) * 1000, 2),
    )


def _default_max_evals(device: str) -> int:
    """SHAP on CPU with PhoBERT-base is ~15-30x slower than on GPU."""
    return 200 if device.startswith("cuda") else 80


@app.post("/explain", response_model=ExplainResponse,
          responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
def explain(request: ExplainRequest):
    """SHAP-based explanation -- opt-in only.

    Runs hundreds of forward passes; never call from /predict.
    """
    _require_services()
    started_at = perf_counter()
    resolved_key, service = _resolve_service(request.model)
    max_evals = request.max_evals or _default_max_evals(service.device)

    try:
        prediction = service.predict(request.text)
        token_scores = service.explain_with_shap(
            request.text,
            max_evals=max_evals,
            predicted_id=prediction["predicted_id"],
        )
    except Exception as exc:
        logger.exception("SHAP explanation failed.")
        raise HTTPException(
            status_code=500,
            detail={"code": "explain_failed", "message": "Explanation could not be completed."},
        ) from exc

    return ExplainResponse(
        label=prediction["label"],
        model_used=resolved_key,
        token_scores=token_scores,
        latency_ms=round((perf_counter() - started_at) * 1000, 2),
    )