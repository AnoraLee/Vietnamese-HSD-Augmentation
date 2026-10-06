from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts._common import get_hf_token, hf_login_from_env, setup_logging
from src.utils.config import load_config, resolve_path
from src.utils.constants import label2id


def _load_split(
    repo: str,
    config: str,
    split: str,
    token: str | None,
) -> pd.DataFrame:
    from datasets import load_dataset

    print(f"[load] {repo} | config={config} | split={split}")
    ds = load_dataset(repo, config, split=split, token=token)
    df = ds.to_pandas()
    return df


def _normalize(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Đảm bảo có `text`, `text_raw`, `label` (int)."""
    df = df.dropna(subset=[text_col]).reset_index(drop=True)

    if text_col != "text":
        df = df.rename(columns={text_col: "text"})

    if "label" in df.columns and df["label"].dtype == object:
        unknown = set(df["label"]) - set(label2id)
        if unknown:
            raise ValueError(f"Nhãn lạ trong dataset: {unknown}")
        df["label"] = df["label"].map(label2id)

    if "text_raw" not in df.columns:
        df["text_raw"] = (
            df["text"].astype(str)
            .str.replace("_", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Ghi đè file đã tồn tại.")
    args = parser.parse_args()

    setup_logging()
    hf_login_from_env()

    cfg = load_config()
    hf = cfg["huggingface"]
    repo = hf["dataset_repo_id"]
    ds_config = hf["default_dataset_config"]
    splits = hf["split_names"]
    token = get_hf_token()

    processed_dir = resolve_path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split_name, hf_split in splits.items():
        out_path = processed_dir / f"{hf_split}.csv"
        if out_path.exists() and not args.force:
            print(f"[skip] {out_path.name} đã tồn tại (dùng --force để ghi đè).")
            continue

        df = _load_split(repo, ds_config, hf_split, token)
        df = _normalize(df, cfg["data"]["text_column"])
        df.to_csv(out_path, index=False)

        label_counts = df["label"].value_counts().sort_index().to_dict()
        print(f"[saved] {out_path} | {len(df)} dòng | labels={label_counts}")

    print("[done] download hoàn tất.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())