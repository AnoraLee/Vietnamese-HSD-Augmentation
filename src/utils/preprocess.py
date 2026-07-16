from __future__ import annotations

import re
import unicodedata

import pandas as pd


VALID_LABELS = ("CLEAN", "OFFENSIVE", "HATE")


def normalize_text(text: str) -> str:
    text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_label(label: str) -> str:
    value = str(label).strip().upper()
    aliases = {
        "0": "CLEAN",
        "1": "OFFENSIVE",
        "2": "HATE",
        "NORMAL": "CLEAN",
        "CLEAN": "CLEAN",
        "OFFENSIVE": "OFFENSIVE",
        "HATE": "HATE",
        "HATE SPEECH": "HATE",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported label: {label!r}")
    return aliases[value]


def clean_hsd_frame(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    required = {text_column, label_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df[[text_column, label_column]].copy()
    out.columns = ["text", "label"]
    out["text"] = out["text"].map(normalize_text)
    out["label"] = out["label"].map(normalize_label)
    out = out[out["text"].str.len() > 0]
    if drop_duplicates:
        out = out.drop_duplicates(subset=["text", "label"])
    return out.reset_index(drop=True)


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["label"].value_counts().reindex(VALID_LABELS, fill_value=0)
    pct = (counts / max(len(df), 1)).round(4)
    return pd.DataFrame({"label": counts.index, "count": counts.values, "ratio": pct.values})

preprocess_text = normalize_text