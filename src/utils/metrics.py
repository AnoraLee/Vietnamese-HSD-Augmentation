from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]


def compute_classification_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    hate_idx = LABEL_NAMES.index("HATE")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "hate_f1": per_class["HATE"]["f1-score"],
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(LABEL_NAMES)))
        ).tolist(),
        "hate_support": int(np.sum(np.array(y_true) == hate_idx)),
    }


def metrics_for_trainer(eval_pred: Any) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics = compute_classification_metrics(labels.tolist(), preds.tolist())
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "hate_f1": float(metrics["hate_f1"]),
    }


def confusion_matrix_frame(matrix: list[list[int]]) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=LABEL_NAMES, columns=LABEL_NAMES)


def compute_metrics(y_true, y_pred):
    if isinstance(y_true, pd.Series):
        y_true = y_true.tolist()
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.tolist()
    return compute_classification_metrics(y_true, y_pred)


def print_metrics(metrics, title="Metrics"):
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"  Accuracy   : {metrics['accuracy']:.4f}")
    print(f"  Macro F1   : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"  Hate F1    : {metrics['hate_f1']:.4f}")
    print(f"{'='*50}\n")