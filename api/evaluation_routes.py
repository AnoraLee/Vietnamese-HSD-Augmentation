"""Read precomputed evaluation metrics từ results/metrics/ và expose qua API."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.utils.constants import (
    ERROR_ANALYSIS_DIR,
    EXPERIMENT_ORDER,
    METRICS_DIR,
    RESULTS_DIR,
)

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


CONFUSION_FILENAME = "confusion_summary.csv"


def _confusion_path_candidates() -> list[Path]:
    """Ưu tiên results/error_analysis/, fallback results/ nếu layout khác."""
    return [ERROR_ANALYSIS_DIR / CONFUSION_FILENAME, RESULTS_DIR / CONFUSION_FILENAME]


@router.get("/errors")
def get_evaluation_errors() -> dict[str, list[dict]]:
    """Confusion counts theo từng experiment, đọc từ confusion_summary.csv.

    Shape: {"phobert_baseline": [{"from": "CLEAN", "to": "OFFENSIVE", "count": 333}, ...]}
    Mỗi list sort giảm dần theo count để frontend lấy phần tử đầu làm lỗi nổi bật.
    Số dòng mỗi experiment không cố định -- model không mắc một loại lỗi nào đó
    thì CSV không có dòng tương ứng.
    """
    path = next((p for p in _confusion_path_candidates() if p.exists()), None)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "no_confusion_summary",
                "message": (
                    f"No {CONFUSION_FILENAME} found under "
                    f"{[str(p.parent) for p in _confusion_path_candidates()]}."
                ),
            },
        )

    by_experiment: dict[str, list[dict]] = {}
    try:
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                experiment = (row.get("experiment") or "").strip()
                true_label = (row.get("true_label") or "").strip()
                predicted_as = (row.get("predicted_as") or "").strip()
                if not (experiment and true_label and predicted_as):
                    continue
                try:
                    count = int(float(row.get("count") or 0))
                except (TypeError, ValueError):
                    continue
                if count <= 0:
                    continue
                by_experiment.setdefault(experiment, []).append(
                    {"from": true_label, "to": predicted_as, "count": count}
                )
    except OSError:
        logger.exception("Failed to read %s", path)
        raise HTTPException(
            status_code=500,
            detail={"code": "confusion_unreadable", "message": f"Cannot read {path.name}."},
        ) from None

    for rows in by_experiment.values():
        rows.sort(key=lambda item: item["count"], reverse=True)
    return by_experiment
