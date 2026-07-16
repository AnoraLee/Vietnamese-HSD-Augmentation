from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer

from src.models.training import HSDDataset, ID_TO_LABEL
from src.utils.hf import load_hf_splits
from src.utils.io import load_config, read_table, write_json, write_table
from src.utils.metrics import compute_classification_metrics, confusion_matrix_frame
from src.utils.preprocess import clean_hsd_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained HSD model on the test set.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--source", choices=["local", "hf"], default="local")
    parser.add_argument("--hf-config", default="baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.source == "hf":
        test = clean_hsd_frame(load_hf_splits(config, args.hf_config)["test"])
    else:
        test = clean_hsd_frame(read_table(config["data"]["test_file"]))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    dataset = HSDDataset(test, tokenizer, config["training"]["max_length"])
    trainer = Trainer(model=model, tokenizer=tokenizer)
    output = trainer.predict(dataset)
    pred_ids = np.argmax(output.predictions, axis=-1).tolist()
    true_ids = output.label_ids.tolist()

    metrics = compute_classification_metrics(true_ids, pred_ids)
    metrics_path = Path(config["paths"]["metrics_dir"]) / f"{args.experiment}_test_metrics.json"
    write_json(metrics, metrics_path)

    predictions = test.copy()
    predictions["prediction"] = [ID_TO_LABEL[idx] for idx in pred_ids]
    predictions["is_correct"] = predictions["label"] == predictions["prediction"]
    write_table(predictions, Path(config["paths"]["error_analysis_dir"]) / f"{args.experiment}_predictions.csv")
    write_table(
        confusion_matrix_frame(metrics["confusion_matrix"]).reset_index(names="true_label"),
        Path(config["paths"]["metrics_dir"]) / f"{args.experiment}_confusion_matrix.csv",
    )
    print(f"Wrote metrics to {metrics_path}")


if __name__ == "__main__":
    main()
