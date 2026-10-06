from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts._common import (
    get_hf_token,
    hf_login_from_env,
    set_seed,
    setup_logging,
)
from src.utils.config import load_config, resolve_path
from src.utils.constants import (
    HF_MODEL_REPO_PREFIX,
    HF_TOKEN_ENV,
    LABELS,
    TRAINING_BATCH_SIZE,
    TRAINING_BF16,
    TRAINING_EPOCHS,
    TRAINING_FP16,
    TRAINING_LEARNING_RATE,
    TRAINING_MAX_LENGTH,
    TRAINING_METRIC_FOR_BEST_MODEL,
    TRAINING_MODEL_NAME,
    TRAINING_OUTPUT_DIR_TEMPLATE,
    TRAINING_SAVE_TOTAL_LIMIT,
    TRAINING_SEED,
    TRAINING_WARMUP_RATIO,
    TRAINING_WEIGHT_DECAY,
    id2label,
    label2id,
    needs_slow_tokenizer,
)


def _build_dataset(df: pd.DataFrame):
    from datasets import Dataset
    return Dataset.from_pandas(df, preserve_index=False)


def _tokenize(batch, tokenizer, max_length: int):
    return tokenizer(batch["text"], truncation=True, max_length=max_length)


def _compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "weighted_f1": f1_score(labels, preds, average="weighted"),
        "hate_f1": f1_score(labels, preds, average=None, labels=[label2id["HATE"]])[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, help="Tên experiment (e.g. phobert_bt).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume từ checkpoint mới nhất nếu có.")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push model lên HF sau khi train.")
    args = parser.parse_args()

    setup_logging()
    set_seed(TRAINING_SEED)
    hf_login_from_env()

    cfg = load_config()
    token = get_hf_token()

    # --- ĐỒNG BỘ VỚI HUGGING FACE CLOUD ---
    # Tự động ánh xạ từ tên experiment sang tên subset trên HF
    # Ví dụ: "phobert_bt" -> "bt", sau đó -> "bt_segmented"
    exp_suffix = args.experiment.split("_")[-1]
    subset_name = f"{exp_suffix}_segmented"
    hf_dataset_repo = "AnoraLee/tdtu_vietnamese_hsd_final"

    print(f"[data] Đang tải subset '{subset_name}' từ kho '{hf_dataset_repo}'...")
    ds = load_dataset(hf_dataset_repo, subset_name, token=token)
    
    train_df = ds["tdtu_train"].to_pandas()
    dev_df = ds["tdtu_dev"].to_pandas()

    model_name = TRAINING_MODEL_NAME
    # Kiểm tra xem đang chạy PhoBERT hay ViSoBERT
    is_phobert = "phobert" in args.experiment.lower() or "phobert" in model_name.lower()
    
    if not is_phobert:
        # ViSoBERT ưu tiên dùng raw_text (văn bản thô không gạch dưới)
        train_df["text"] = train_df["text_raw"]
        dev_df["text"] = dev_df["text_raw"]

    train_df = train_df.dropna(subset=["text", "label"]).reset_index(drop=True)
    dev_df = dev_df.dropna(subset=["text", "label"]).reset_index(drop=True)

    print(f"[data] train={len(train_df)} | dev={len(dev_df)}")

    use_fast = not needs_slow_tokenizer(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast, token=token)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
        token=token,
    )

    train_ds = _build_dataset(train_df).map(
        lambda b: _tokenize(b, tokenizer, TRAINING_MAX_LENGTH),
        batched=True, remove_columns=["text"],
    )
    dev_ds = _build_dataset(dev_df).map(
        lambda b: _tokenize(b, tokenizer, TRAINING_MAX_LENGTH),
        batched=True, remove_columns=["text"],
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    output_dir = resolve_path(TRAINING_OUTPUT_DIR_TEMPLATE.format(experiment=args.experiment))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {output_dir}")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=TRAINING_LEARNING_RATE,
        per_device_train_batch_size=TRAINING_BATCH_SIZE,
        per_device_eval_batch_size=TRAINING_BATCH_SIZE * 2,
        num_train_epochs=TRAINING_EPOCHS,
        weight_decay=TRAINING_WEIGHT_DECAY,
        warmup_ratio=TRAINING_WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=TRAINING_SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model=TRAINING_METRIC_FOR_BEST_MODEL,
        greater_is_better=True,
        logging_steps=50,
        fp16=TRAINING_FP16 and torch.cuda.is_available(),
        bf16=TRAINING_BF16 and torch.cuda.is_available(),
        report_to=["none"],
        push_to_hub=args.push_to_hub,
        hub_model_id=(
            f"{HF_MODEL_REPO_PREFIX}-{args.experiment}" if args.push_to_hub else None
        ),
        hub_token=token if args.push_to_hub else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if TRAINING_EPOCHS > 2 else None,
    )

    resume_from = None
    if args.resume:
        ckpts = sorted(output_dir.glob("checkpoint-*"))
        if ckpts:
            resume_from = str(ckpts[-1])
            print(f"[resume] từ {resume_from}")
        else:
            print("[resume] không có checkpoint, train từ đầu.")

    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save metrics từ best checkpoint.
    metrics = trainer.evaluate()
    from scripts._common import save_json
    metrics_path = output_dir / "train_metrics.json"
    save_json(metrics_path, metrics)
    print(f"[saved] {metrics_path}")
    print(f"[metrics] {metrics}")

    if args.push_to_hub:
        trainer.push_to_hub()
        print(f"[hub] pushed → {training_args.hub_model_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())