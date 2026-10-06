from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts._common import setup_logging
from src.utils.config import load_config, resolve_path
from src.utils.constants import LABELS, label2id


def _summarize_split(name: str, df: pd.DataFrame) -> dict:
    counts = df["label"].value_counts().sort_index().to_dict()
    id2label = {i: l for l, i in label2id.items()}
    return {
        "split": name,
        "rows": len(df),
        "unique_texts": df["text"].nunique(),
        "labels": {id2label[i]: counts.get(i, 0) for i in range(len(LABELS))},
        "avg_tokens": round(df["text"].str.split().str.len().mean(), 2),
        "max_tokens": int(df["text"].str.split().str.len().max()),
    }


def _check_leak(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame) -> None:
    train_texts = set(train["text"])
    dev_texts = set(dev["text"])
    test_texts = set(test["text"])

    train_dev = train_texts & dev_texts
    train_test = train_texts & test_texts
    dev_test = dev_texts & test_texts

    if train_dev:
        print(f"[WARN] train/dev trùng {len(train_dev)} câu.")
    if train_test:
        print(f"[WARN] train/test trùng {len(train_test)} câu.")
    if dev_test:
        print(f"[WARN] dev/test trùng {len(dev_test)} câu.")
    if not (train_dev or train_test or dev_test):
        print("[leak] không có câu trùng giữa các split.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    setup_logging()

    cfg = load_config()
    files = {
        "train": cfg["data"]["train_file"],
        "dev": cfg["data"]["dev_file"],
        "test": cfg["data"]["test_file"],
    }

    dfs: dict[str, pd.DataFrame] = {}
    for name, rel in files.items():
        path = resolve_path(rel)
        if not path.exists():
            print(f"[ERROR] thiếu file {path} — chạy download_data.py trước.")
            return 1
        dfs[name] = pd.read_csv(path)

    print("\n=== Tổng quan split ===")
    for name, df in dfs.items():
        print(_summarize_split(name, df))

    print("\n=== Kiểm tra leak ===")
    _check_leak(dfs["train"], dfs["dev"], dfs["test"])

    print("\n=== Kiểm tra nhãn ===")
    expected = set(range(len(LABELS)))
    ok = True
    for name, df in dfs.items():
        present = set(df["label"].unique())
        missing = expected - present
        if missing:
            print(f"[ERROR] split {name} thiếu nhãn {missing}")
            ok = False
    if ok:
        print("[labels] mọi split đều có đủ nhãn.")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())