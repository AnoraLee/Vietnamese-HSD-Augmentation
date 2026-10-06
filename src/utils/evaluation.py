from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.constants import (
    EXPERIMENT_ORDER,
    HF_DATASET_REPO,
    HF_DEFAULT_DATASET_CONFIG,
    HF_MODEL_IDS,
    HF_RESULTS_REPO,
    HF_SPLIT_NAMES,
    LABELS,
    needs_slow_tokenizer,
)

def _metrics_to_row(experiment: str, metrics: dict) -> dict:
    """Bỏ prefix `test_` mà HF Trainer tự thêm vào eval metrics.

    Dùng `removeprefix` (không phải `.replace`) để chỉ chạm lần xuất hiện
    đầu tiên -- `test_hate_test_f1` → `hate_test_f1` (không phải `hate_f1`).
    """
    row: dict = {"experiment": experiment}
    row.update(
        {k.removeprefix("test_"): v for k, v in metrics.items() if k.startswith("test_")}
    )
    return row


def _read_metrics_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_metrics_local(metrics_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for experiment in EXPERIMENT_ORDER:
        local = metrics_dir / f"metrics_{experiment}.json"
        if not local.exists():
            print(f"  (skip) {local.name} not found")
            continue
        rows.append(_metrics_to_row(experiment, _read_metrics_json(local)))
    if not rows:
        raise FileNotFoundError(f"No metrics_<experiment>.json found under {metrics_dir}.")
    return pd.DataFrame(rows)


def _load_metrics_hf(repo_id: str, token: str | None) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    rows: list[dict] = []
    for experiment in EXPERIMENT_ORDER:
        filename = f"metrics/metrics_{experiment}.json"
        try:
            local_path = hf_hub_download(
                repo_id=repo_id, filename=filename,
                repo_type="dataset", token=token,
            )
        except EntryNotFoundError:
            print(f"  (skip) {filename} not found on HF")
            continue
        rows.append(_metrics_to_row(experiment, _read_metrics_json(Path(local_path))))

    if not rows:
        raise FileNotFoundError(
            f"No metrics found in HF repo '{repo_id}'. "
            f"Expected 'metrics/metrics_<exp>.json'."
        )
    return pd.DataFrame(rows)


def load_all_metrics(
    source: str | Path | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    if isinstance(source, Path):
        return _load_metrics_local(source)
    return _load_metrics_hf(source or HF_RESULTS_REPO, token)

def load_test_set(
    dataset_repo: str | None = None,
    config: str | None = None,
    split: str | None = None,
    token: str | None = None,
) -> pd.DataFrame:
    """Load test split dùng chung cho mọi experiment.

    Default lấy từ config.yaml -- chỉ override khi biết rõ tại sao.
    """
    from datasets import load_dataset

    repo = dataset_repo or HF_DATASET_REPO
    cfg_name = config or HF_DEFAULT_DATASET_CONFIG
    split_name = split or HF_SPLIT_NAMES["test"]

    ds = load_dataset(repo, cfg_name, split=split_name, token=token)
    df = ds.to_pandas().dropna(subset=["text"]).reset_index(drop=True)

    if "text_raw" not in df.columns:
        df["text_raw"] = (
            df["text"].astype(str)
            .str.replace("_", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
    return df


def available_experiments() -> list[str]:
    """Experiments có trong EXPERIMENT_ORDER và có model ID."""
    return [e for e in EXPERIMENT_ORDER if e in HF_MODEL_IDS]

def load_model_and_tokenizer(
    experiment: str,
    device: str = "cpu",
    token: str | None = None,
):
    """Load (model, tokenizer) cho 1 experiment từ HF Hub.

    Dùng `constants.needs_slow_tokenizer` -- 1 nguồn logic duy nhất cho
    quyết định fast/slow tokenizer, không lặp ở đây.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if experiment not in HF_MODEL_IDS:
        raise KeyError(
            f"Unknown experiment {experiment!r}. Available: {list(HF_MODEL_IDS)}"
        )

    hf_model_id = HF_MODEL_IDS[experiment]
    use_fast = not needs_slow_tokenizer(hf_model_id)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=use_fast, token=token)
    model = (
        AutoModelForSequenceClassification
        .from_pretrained(hf_model_id, token=token)
        .to(device)
    )
    model.eval()
    return model, tokenizer

def predict_texts(
    model,
    tokenizer,
    texts: list[str],
    device: str = "cpu",
    max_length: int = 128,
    batch_size: int = 32,
) -> list[int]:
    """Trả về predicted label indices cho `texts`."""
    import torch

    if not texts:
        return []

    all_preds: list[int] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**encoded).logits
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return all_preds

def compute_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))


def get_misclassified_examples(
    texts: list[str],
    y_true: list[int],
    y_pred: list[int],
    id2label: dict[int, str],
    n: int | None = None,
) -> pd.DataFrame:
    """Các dòng model đoán sai, với nhãn đọc được.

    `n=None` → tất cả; `n=0` → rỗng; `n>0` → n dòng đầu.
    """
    df = pd.DataFrame({
        "text": texts,
        "true_label": [id2label[i] for i in y_true],
        "predicted_label": [id2label[i] for i in y_pred],
    })
    errors = df[df["true_label"] != df["predicted_label"]].reset_index(drop=True)
    return errors.head(n) if n is not None else errors