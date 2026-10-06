from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts._common import setup_logging
from src.utils.constants import (
    EXPERIMENT_ORDER,
    FIGURES_DIR,
    LABELS,
    METRICS_DIR,
    ensure_output_dirs,
)


def _load_metrics(exp: str) -> dict | None:
    path = METRICS_DIR / f"metrics_{exp}.json"
    if not path.exists():
        print(f"[skip] không có {path.name}")
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _plot_one(exp: str, metrics: dict) -> Path:
    cm = np.array(metrics["test_confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{exp}\nmacro_f1={metrics['test_macro_f1']:.3f}")

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = FIGURES_DIR / f"confusion_{exp}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    setup_logging()
    ensure_output_dirs()

    if args.all:
        targets = list(EXPERIMENT_ORDER)
    elif args.experiment:
        targets = [args.experiment]
    else:
        print("[ERROR] cần --experiment hoặc --all.")
        return 1

    plotted = 0
    for exp in targets:
        metrics = _load_metrics(exp)
        if not metrics:
            continue
        out = _plot_one(exp, metrics)
        print(f"[saved] {out}")
        plotted += 1

    print(f"[done] vẽ {plotted}/{len(targets)} confusion matrix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())