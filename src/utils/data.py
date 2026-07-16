from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset, DatasetDict, load_dataset


def load_config(path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = ensure_parent(path)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8")


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found: {path}")
    return pd.read_csv(path)


def save_data(df: pd.DataFrame, path: str, index: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)
    print(f"Saved {len(df)} rows to {path}")


@dataclass(frozen=True)
class HFDatasetConfig:
    repo_id: str
    config_name: str
    train_split: str = "train"
    dev_split: str = "dev"
    test_split: str = "test"


def hf_dataset_config(config: dict, config_name: str | None = None) -> HFDatasetConfig:
    hf_cfg = config["huggingface"]
    splits = hf_cfg.get("split_names", {})
    return HFDatasetConfig(
        repo_id=hf_cfg["dataset_repo_id"],
        config_name=config_name or hf_cfg["baseline_config"],
        train_split=splits.get("train", "train"),
        dev_split=splits.get("dev", "validation"),
        test_split=splits.get("test", "test"),
    )


def load_hf_splits(config: dict, config_name: str | None = None) -> dict[str, pd.DataFrame]:
    hf = hf_dataset_config(config, config_name)
    dataset = load_dataset(hf.repo_id, hf.config_name)
    return {
        "train": dataset[hf.train_split].to_pandas(),
        "dev": dataset[hf.dev_split].to_pandas(),
        "test": dataset[hf.test_split].to_pandas(),
    }


def push_hf_splits(
    config: dict,
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    config_name: str,
    private: bool | None = None,
    token: str | None = None,
) -> None:
    hf_cfg = config["huggingface"]
    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train.reset_index(drop=True), preserve_index=False),
            "validation": Dataset.from_pandas(dev.reset_index(drop=True), preserve_index=False),
            "test": Dataset.from_pandas(test.reset_index(drop=True), preserve_index=False),
        }
    )
    dataset.push_to_hub(
        hf_cfg["dataset_repo_id"],
        config_name=config_name,
        private=hf_cfg["private"] if private is None else private,
        token=token,
    )

def load_hf_dataset(repo_id: str, split: str = 'train'):
    from datasets import load_dataset
    ds = load_dataset(repo_id, split=split)
    return ds.to_pandas()

def push_to_hf(df, repo_id: str, config_name: str = 'default'):
    from datasets import Dataset, Features, Value
    
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].astype(str)
    
    features = Features({
        col: Value('string') for col in df_copy.columns
    })
    
    dataset = Dataset.from_pandas(df_copy.reset_index(drop=True), features=features)
    dataset.push_to_hub(repo_id, config_name=config_name)
    print(f"Uploaded to {repo_id} (config: {config_name})")