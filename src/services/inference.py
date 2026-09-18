"""Canonical PhoBERT inference service used by every application entry point."""

from __future__ import annotations

from time import perf_counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.config import load_config
from src.utils.constants import LABELS
from src.utils.preprocess import TextPreprocessor, get_text_preprocessor


class HSDInferenceService:
    """Load one checkpoint and return a stable, UI-agnostic prediction payload."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        max_length: int | None = None,
        preprocessor: TextPreprocessor | None = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length or load_config()["training"]["max_length"]
        self.preprocessor = preprocessor or get_text_preprocessor()

        print(f"Loading model on {self.device}: {self.model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Preprocess text, run inference, and return JSON-serializable values."""
        started_at = perf_counter()
        text_cleaned = self.preprocessor.clean_text(text)
        encoded = self.tokenizer(
            text_cleaned,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**encoded).logits, dim=-1)[0]

        probability_values = [float(value) for value in probabilities.cpu().tolist()]
        predicted_id = int(probabilities.argmax())
        return {
            "text": text,
            "text_cleaned": text_cleaned,
            "label": LABELS[predicted_id],
            "confidence": probability_values[predicted_id],
            "probabilities": dict(zip(LABELS, probability_values, strict=True)),
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        }