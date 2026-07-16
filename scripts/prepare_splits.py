from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import (
    load_config,
    push_hf_splits,
    write_table,
    load_data,
    save_data
)
from src.utils.preprocess import clean_hsd_frame, label_distribution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reproducible train/dev/test splits.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True, help="CSV/parquet with text and label columns.")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--dev-size", type=float, default=0.15)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hf-config", default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_cfg = config["data"]
    seed = int(config["project"]["seed"])

    path = Path(args.input)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df = clean_hsd_frame(df, data_cfg["text_column"], data_cfg["label_column"])

    train_dev, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=seed,
        stratify=df["label"],
    )
    dev_ratio = args.dev_size / (1.0 - args.test_size)
    train, dev = train_test_split(
        train_dev,
        test_size=dev_ratio,
        random_state=seed,
        stratify=train_dev["label"],
    )

    write_table(train, data_cfg["train_file"])
    write_table(dev, data_cfg["dev_file"])
    write_table(test, data_cfg["test_file"])

    summary = pd.concat(
        {
            "train": label_distribution(train),
            "dev": label_distribution(dev),
            "test": label_distribution(test),
        },
        names=["split"],
    ).reset_index(level=0)
    write_table(summary, Path(config["paths"]["processed_dir"]) / "label_distribution.csv")
    print(summary.to_string(index=False))

    if args.push_to_hub:
        push_hf_splits(
            config,
            train=train,
            dev=dev,
            test=test,
            config_name=args.hf_config,
            token=os.getenv("HF_TOKEN"),
        )
        print(f"Pushed dataset config '{args.hf_config}' to {config['huggingface']['dataset_repo_id']}.")


if __name__ == "__main__":
    main()