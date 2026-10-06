from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.utils.constants import (
    EXPERIMENT_ORDER,
    FIGURES_DIR,
    HF_RESULTS_REPO,
    METRICS_DIR,
    RESULTS_DIR,
    ensure_output_dirs,
)
from src.utils.evaluation import load_all_metrics

METRIC_COLS = ["accuracy", "macro_f1", "weighted_f1", "hate_f1"]


def _has_local_metrics() -> bool:
    return METRICS_DIR.is_dir() and any(METRICS_DIR.glob("metrics_*.json"))


def _resolve_source(source: str) -> str | Path:
    """Trả về Path (local) hoặc str (HF repo id) cho load_all_metrics()."""
    if source == "local":
        if not _has_local_metrics():
            raise FileNotFoundError(f"Không có metrics local trong {METRICS_DIR}.")
        return METRICS_DIR
    if source == "hf":
        return HF_RESULTS_REPO
    # auto
    if _has_local_metrics():
        print(f"[source] auto → local ({METRICS_DIR})")
        return METRICS_DIR
    print(f"[source] auto → HF ({HF_RESULTS_REPO})")
    return HF_RESULTS_REPO


def _ordered_experiments(df):
    """Sắp xếp theo EXPERIMENT_ORDER, giữ nguyên các experiment lạ ở cuối."""
    order_map = {name: i for i, name in enumerate(EXPERIMENT_ORDER)}
    return df.assign(
        _order=df["experiment"].map(lambda e: order_map.get(e, len(order_map)))
    ).sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _plot(df, metric_cols: list[str], fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    df.set_index("experiment")[metric_cols].plot(kind="bar", ax=ax, width=0.8)

    ax.set_ylabel("score")
    ax.set_xlabel("")
    ax.set_title("Ablation experiments comparison")
    ax.set_ylim(0, 1.0)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=False,
    )
    ax.tick_params(axis="x", rotation=30)

    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["auto", "local", "hf"],
        default="auto",
        help="Nguồn metrics (mặc định: auto).",
    )
    parser.add_argument(
        "--sort-by-score",
        action="store_true",
        help="Sắp xếp theo metric đầu tiên thay vì EXPERIMENT_ORDER.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token nếu repo private (mặc định đọc env HF_TOKEN).",
    )
    args = parser.parse_args()

    ensure_output_dirs()

    source = _resolve_source(args.source)
    summary_df = load_all_metrics(source, token=args.token)

    if summary_df.empty or "experiment" not in summary_df.columns:
        print("Không có experiment nào để tổng hợp.")
        return 1

    metric_cols = [c for c in METRIC_COLS if c in summary_df.columns]
    if not metric_cols:
        print(f"Không có cột metric nào trong {METRIC_COLS}; có: {list(summary_df.columns)}")
        return 1

    if args.sort_by_score:
        summary_df = summary_df.sort_values(metric_cols[0], ascending=False)
    else:
        summary_df = _ordered_experiments(summary_df)

    summary_path = RESULTS_DIR / "experiments_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Đã lưu bảng: {summary_path}")
    print(summary_df.to_string(index=False))

    fig_path = FIGURES_DIR / "metrics_comparison.png"
    _plot(summary_df, metric_cols, fig_path)
    print(f"Đã lưu biểu đồ: {fig_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())