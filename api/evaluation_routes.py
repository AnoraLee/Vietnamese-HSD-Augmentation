"""Read precomputed evaluation metrics từ results/metrics/ và expose qua API."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.utils.constants import EXPERIMENT_ORDER, METRICS_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation", tags=["evaluation"])

# HF Trainer tự thêm prefix "test_" vào eval metrics -- strip + rename.
_FIELD_MAP: dict[str, str] = {
    "loss": "loss",
    "accuracy": "accuracy",
    "macro_f1": "macroF1",
    "weighted_f1": "weightedF1",
    "hate_f1": "hateF1",
}


def _metrics_dir_candidates() -> list[Path]:
    """Ưu tiên results/metrics/, fallback results/ nếu layout khác."""
    return [METRICS_DIR, RESULTS_DIR]


def _read_metrics_file(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _normalize_row(experiment: str, raw: dict) -> dict:
    """Chuẩn hóa metrics JSON (có thể có/không prefix test_) về camelCase cho FE."""
    row: dict = {"experiment": experiment}
    for src_key, dst_key in _FIELD_MAP.items():
        value = raw.get(f"test_{src_key}", raw.get(src_key))
        if isinstance(value, (int, float)):
            row[dst_key] = float(value)
    return row


@router.get("/metrics")
def get_evaluation_metrics() -> list[dict]:
    """List metrics cho mọi experiment có file metrics_<exp>.json.

    Shape: [{"experiment": "phobert_baseline", "accuracy": 0.84, ...}, ...]
    Sorted theo EXPERIMENT_ORDER để frontend hiển thị ổn định.
    """
    by_experiment: dict[str, dict] = {}

    for base in _metrics_dir_candidates():
        if not base.exists():
            continue
        for path in sorted(base.glob("metrics_*.json")):
            experiment = path.stem.removeprefix("metrics_")
            if experiment in by_experiment:
                continue
            try:
                raw = _read_metrics_file(path)
            except Exception:
                logger.exception("Failed to read %s", path)
                continue
            by_experiment[experiment] = _normalize_row(experiment, raw)

    if not by_experiment:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "no_metrics",
                "message": (
                    "No metrics_<experiment>.json found under "
                    f"{[str(p) for p in _metrics_dir_candidates()]}."
                ),
            },
        )

    order_index = {name: i for i, name in enumerate(EXPERIMENT_ORDER)}
    return sorted(
        by_experiment.values(),
        key=lambda row: order_index.get(row["experiment"], len(order_index)),
    )