from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts._common import (
    get_hf_token,
    hf_login_from_env,
    save_json,
    set_seed,
    setup_logging,
)
from src.utils.config import load_config, resolve_path
from src.utils.constants import (
    LABELS,
    METRICS_DIR,
    TRAINING_MAX_LENGTH,
    TRAINING_OUTPUT_DIR_TEMPLATE,
    TRAINING_SEED,
    HF_MODEL_IDS,
    label2id,
    needs_slow_tokenizer,
)


def _resolve_checkpoint(experiment: str, override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    return resolve_path(TRAINING_OUTPUT_DIR_TEMPLATE.format(experiment=experiment))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="Đường dẫn checkpoint. Mặc định: models/<exp>_phobert.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    setup_logging()
    set_seed(TRAINING_SEED)
    hf_login_from_env()

    cfg = load_config()
    ckpt = _resolve_checkpoint(args.experiment, args.checkpoint)
    if not ckpt.exists():
        print(f"[ERROR] checkpoint không tồn tại: {ckpt}")
        return 1

    test_path = resolve_path(cfg["data"]["test_file"])
    test_df = pd.read_csv(test_path).dropna(subset=["text"]).reset_index(drop=True)
    if test_df["label"].dtype == object:
        test_df["label"] = test_df["label"].map(label2id)
    print(f"[data] test={len(test_df)} dòng từ {test_path.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    token = get_hf_token()
    model_id = str(ckpt)
    use_fast = not needs_slow_tokenizer(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast, token=token)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, token=token).to(device)
    model.eval()

    preds: list[int] = []
    probs_all: list[list[float]] = []
    for i in range(0, len(test_df), args.batch_size):
        batch = test_df["text"].iloc[i:i + args.batch_size].tolist()
        encoded = tokenizer(
            batch, truncation=True, padding=True,
            max_length=TRAINING_MAX_LENGTH, return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        p = torch.softmax(logits, dim=-1).cpu().numpy()
        preds.extend(p.argmax(axis=-1).tolist())
        probs_all.extend(p.tolist())

    y_true = test_df["label"].tolist()

    per_class = precision_recall_fscore_support(y_true, preds, labels=list(range(len(LABELS))))
    macro = f1_score(y_true, preds, average="macro")
    weighted = f1_score(y_true, preds, average="weighted")
    hate_id = label2id["HATE"]

    metrics = {
        "experiment": args.experiment,
        "test_accuracy": float(accuracy_score(y_true, preds)),
        "test_macro_f1": float(macro),
        "test_weighted_f1": float(weighted),
        "test_hate_f1": float(per_class[2][hate_id]),
        "test_per_class": {
            label: {
                "precision": float(per_class[0][i]),
                "recall": float(per_class[1][i]),
                "f1": float(per_class[2][i]),
                "support": int(per_class[3][i]),
            }
            for i, label in enumerate(LABELS)
        },
        "test_confusion_matrix": confusion_matrix(
            y_true, preds, labels=list(range(len(LABELS)))
        ).tolist(),
    }

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"metrics_{args.experiment}.json"
    save_json(metrics_path, metrics)
    print(f"[saved] {metrics_path}")
    print(f"[metrics] accuracy={metrics['test_accuracy']:.4f} "
          f"macro_f1={metrics['test_macro_f1']:.4f} "
          f"hate_f1={metrics['test_hate_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())