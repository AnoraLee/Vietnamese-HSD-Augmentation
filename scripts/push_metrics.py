from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts._common import get_hf_token, setup_logging
from src.utils.constants import (
    EXPERIMENT_ORDER,
    HF_RESULTS_REPO,
    METRICS_DIR,
)


def _push_one(experiment: str, repo_id: str, token: str | None) -> bool:
    from huggingface_hub import HfApi

    local = METRICS_DIR / f"metrics_{experiment}.json"
    if not local.exists():
        print(f"[skip] {local.name} không tồn tại.")
        return False

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(local),
        path_in_repo=f"metrics/metrics_{experiment}.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"update metrics for {experiment}",
    )
    print(f"[pushed] {experiment} → {repo_id}/metrics/metrics_{experiment}.json")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--all", action="store_true", help="Push mọi experiment có file local.")
    parser.add_argument("--repo", default=HF_RESULTS_REPO)
    args = parser.parse_args()

    setup_logging()
    token = get_hf_token()
    if not token:
        print("[ERROR] cần HF_TOKEN trong env.")
        return 1

    if args.all:
        targets = list(EXPERIMENT_ORDER)
    elif args.experiment:
        targets = [args.experiment]
    else:
        print("[ERROR] phải có --experiment hoặc --all.")
        return 1

    pushed = 0
    for exp in targets:
        if _push_one(exp, args.repo, token):
            pushed += 1

    print(f"[done] pushed {pushed}/{len(targets)} file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())