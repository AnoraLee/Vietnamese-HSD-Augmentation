from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from transformers import Trainer, TrainingArguments

from src.models.training import HSDDataset, build_model_bundle
from src.utils.hf import load_hf_splits
from src.utils.io import load_config, read_table, set_seed, write_json
from src.utils.metrics import metrics_for_trainer
from src.utils.preprocess import clean_hsd_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PhoBERT/XLM-R HSD classifier.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--augmented-file", default=None)
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--hf-config", default="baseline")
    parser.add_argument("--push-model-to-hub", action="store_true")
    parser.add_argument("--model-repo-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(int(config["project"]["seed"]))
    train_cfg = config["training"]

    if args.source == "hf":
        splits = load_hf_splits(config, args.hf_config)
        train = clean_hsd_frame(splits["train"])
        dev = clean_hsd_frame(splits["dev"])
    else:
        train = clean_hsd_frame(read_table(config["data"]["train_file"]))
        dev = clean_hsd_frame(read_table(config["data"]["dev_file"]))
    if args.augmented_file:
        augmented = clean_hsd_frame(read_table(args.augmented_file))
        train = pd.concat([train, augmented], ignore_index=True).drop_duplicates(["text", "label"])

    bundle = build_model_bundle(train_cfg["model_name"])
    train_dataset = HSDDataset(train, bundle.tokenizer, train_cfg["max_length"])
    dev_dataset = HSDDataset(dev, bundle.tokenizer, train_cfg["max_length"])

    output_dir = Path(config["paths"]["saved_models_dir"]) / args.experiment
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(train_cfg["learning_rate"]),
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        per_device_eval_batch_size=int(train_cfg["batch_size"]),
        num_train_epochs=float(train_cfg["epochs"]),
        weight_decay=float(train_cfg["weight_decay"]),
        warmup_ratio=float(train_cfg["warmup_ratio"]),
        evaluation_strategy=train_cfg["evaluation_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=bool(train_cfg["load_best_model_at_end"]),
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        report_to=[],
    )
    trainer = Trainer(
        model=bundle.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=bundle.tokenizer,
        compute_metrics=metrics_for_trainer,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    bundle.tokenizer.save_pretrained(str(output_dir))
    write_json(metrics, Path(config["paths"]["metrics_dir"]) / f"{args.experiment}_dev_metrics.json")
    if args.push_model_to_hub:
        repo_id = args.model_repo_id or f"{config['huggingface']['model_repo_prefix']}-{args.experiment}"
        bundle.model.push_to_hub(
            repo_id,
            private=config["huggingface"]["private"],
            token=os.getenv("HF_TOKEN"),
        )
        bundle.tokenizer.push_to_hub(
            repo_id,
            private=config["huggingface"]["private"],
            token=os.getenv("HF_TOKEN"),
        )
        print(f"Pushed model to {repo_id}.")
    print(metrics)


if __name__ == "__main__":
    main()
