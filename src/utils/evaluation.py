from pathlib import Path

import pandas as pd

from src.utils.constants import LABELS, HF_MODEL_IDS, EXPERIMENT_ORDER


def load_all_metrics(results_dir: Path) -> pd.DataFrame:
    """Read ``results/metrics/metrics_<experiment>.json`` files.

    ``results_dir`` is the project results directory. Missing experiments are
    skipped so an incomplete local download can still be summarized.
    """
    rows = []
    for experiment in EXPERIMENT_ORDER:
        metrics_path = results_dir / "metrics" / f"metrics_{experiment}.json"
        if not metrics_path.exists():
            print(f"  (skip) {metrics_path} not found yet")
            continue
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)
        row = {"experiment": experiment}
        row.update({k.replace("test_", ""): v for k, v in metrics.items() if k.startswith("test_")})
        rows.append(row)

    if not rows:
        raise FileNotFoundError(
            f"No metrics_<experiment>.json found under {results_dir / 'metrics'}. "
            "Generate or download the experiment metrics first."
        )
    return pd.DataFrame(rows)


def available_experiments(models_dir: Path) -> list[str]:
    """Which experiments actually have a saved model downloaded locally."""
    return list(HF_MODEL_IDS.keys())


def load_model_and_tokenizer(models_dir: Path, experiment: str, device: str = "cpu"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    hf_model_id = HF_MODEL_IDS[experiment]
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(hf_model_id).to(device)
    model.eval()
    return model, tokenizer


def predict_texts(model, tokenizer, texts, device: str = "cpu", max_length: int = 128, batch_size: int = 32):
    """Run inference over a list of texts, return predicted label indices."""
    import torch

    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return all_preds


def compute_confusion_matrix(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    return confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))


def get_misclassified_examples(texts, y_true, y_pred, label2id_inv, n: int | None = None) -> pd.DataFrame:
    """Return rows where prediction != ground truth, with human-readable labels."""
    df = pd.DataFrame({
        "text": texts,
        "true_label": [label2id_inv[i] for i in y_true],
        "predicted_label": [label2id_inv[i] for i in y_pred],
    })
    errors = df[df["true_label"] != df["predicted_label"]].reset_index(drop=True)
    return errors.head(n) if n else errors
