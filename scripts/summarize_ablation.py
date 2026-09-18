import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from src.utils.evaluation import load_all_metrics

RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = load_all_metrics(RESULTS_DIR)
    summary_path = RESULTS_DIR / "experiments_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved table: {summary_path}")
    print(summary_df.to_string(index=False))

    metric_cols = [c for c in ["accuracy", "macro_f1", "weighted_f1", "hate_f1"] if c in summary_df.columns]
    ax = summary_df.set_index("experiment")[metric_cols].plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("score")
    ax.set_title("PhoBERT experiments comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()

    fig_path = FIGURES_DIR / "metrics_comparison.png"
    plt.savefig(fig_path, dpi=150)
    print(f"Saved chart: {fig_path}")


if __name__ == "__main__":
    main()