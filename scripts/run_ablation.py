from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts._common import parse_experiments, setup_logging
from src.utils.config import load_config, resolve_path
from src.utils.constants import METRICS_DIR

AUGMENTATION_MAP = {
    "baseline": [],
    "bt": ["data/augmented/aug_bt.csv"],
    "eda": ["data/augmented/aug_eda.csv"],
    "llm": ["data/augmented/aug_gemma.csv", "data/augmented/aug_qwen.csv"],
    "combined": [
        "data/augmented/aug_bt.csv",
        "data/augmented/aug_eda.csv",
        "data/augmented/aug_gemma.csv",
        "data/augmented/aug_qwen.csv",
    ],
}


def _augmentation_files_for(experiment: str) -> list[str]:
    """Lấy theo suffix sau dấu '_' cuối cùng. Vd: 'phobert_bt' → 'bt'."""
    suffix = experiment.rsplit("_", 1)[-1]
    return AUGMENTATION_MAP.get(suffix, [])


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", default=None,
                        help="CSV experiment override. Mặc định đọc experiments.required.")
    parser.add_argument("--force", action="store_true",
                        help="Chạy lại cả experiment đã có metrics.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--push", action="store_true",
                        help="Push model + metrics lên HF sau mỗi experiment.")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()

    if args.experiments:
        experiments = parse_experiments(args.experiments)
    else:
        experiments = list(cfg.get("experiments", {}).get("required", []))

    if not experiments:
        print("[ERROR] không có experiment nào để chạy.")
        return 1

    print(f"[plan] sẽ chạy {len(experiments)} experiment: {experiments}")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    failed: list[str] = []
    for exp in experiments:
        metrics_path = METRICS_DIR / f"metrics_{exp}.json"
        if metrics_path.exists() and not args.force:
            print(f"\n[skip] {exp} đã có metrics ({metrics_path.name}). Dùng --force để chạy lại.")
            continue

        print(f"\n{'=' * 60}\n[run] {exp}\n{'=' * 60}")

        # --- Train ---
        if not args.skip_train:
            train_cmd = [
                sys.executable, str(PROJECT_DIR / "scripts" / "train.py"),
                "--experiment", exp,
            ]
            if args.push:
                train_cmd.append("--push-to-hub")

            if _run(train_cmd) != 0:
                failed.append(exp)
                if not args.continue_on_error:
                    return 1
                continue

        # --- Evaluate ---
        if not args.skip_eval:
            eval_cmd = [
                sys.executable, str(PROJECT_DIR / "scripts" / "evaluate.py"),
                "--experiment", exp,
            ]
            if _run(eval_cmd) != 0:
                failed.append(exp)
                if not args.continue_on_error:
                    return 1
                continue

        # --- Push metrics ---
        if args.push:
            push_cmd = [
                sys.executable, str(PROJECT_DIR / "scripts" / "push_metrics.py"),
                "--experiment", exp,
            ]
            _run(push_cmd)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"[done] {len(failed)} experiment fail: {failed}")
        return 1
    print("[done] tất cả experiment hoàn tất.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())