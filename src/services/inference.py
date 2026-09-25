"""Canonical PhoBERT inference service used by every application entry point."""

from __future__ import annotations

from time import perf_counter

import numpy as np
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
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
        ).to(self.device)
        self.model.eval()
        self._shap_explainer = None
        self._shap_cache: dict[tuple[str, int, int], list[dict]] = {}

    def warm_up(self) -> None:
        """Initialize the model device before the first user request."""
        encoded = self.tokenizer(
            "không độc hại",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            self.model(**encoded)

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
            outputs = self.model(**encoded)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        probability_values = [float(value) for value in probabilities.cpu().tolist()]
        predicted_id = int(probabilities.argmax())

        return {
            "text": text,
            "text_cleaned": text_cleaned,
            "label": LABELS[predicted_id],
            "confidence": probability_values[predicted_id],
            "probabilities": dict(zip(LABELS, probability_values, strict=True)),
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
            "token_importance": [],
        }

    def _token_importance(self, text_cleaned: str, encoded, attentions) -> list[dict]:
        """Approximate per-word importance from the model's own attention.

        Method: last transformer layer, attention heads averaged, taking the
        row for the CLS/BOS position (index 0) -- i.e. "how much did each
        subword contribute to the representation the classifier head reads."
        This is a heuristic, not a formally validated attribution method
        (unlike LIME/Integrated Gradients); report it as "attention-based
        visualization," not as a rigorous explainability claim.

        `use_fast=False` means there's no automatic subword->word offset
        map, so word boundaries are recovered by re-tokenizing each
        whitespace-split word on its own and counting pieces. Because the
        input is already word-segmented (compound words joined by "_"),
        PhoBERT's BPE vocabulary is built to respect those boundaries, so
        this recovers the true split in the large majority of cases -- but
        it is still an approximation, not a guaranteed exact alignment.
        """
        words = text_cleaned.split()
        if not words or not attentions:
            return []

        # attentions: tuple of (num_layers) tensors, each [batch, heads, seq, seq]
        last_layer_attention = attentions[-1][0]              # -> [heads, seq, seq]
        cls_attention = last_layer_attention.mean(dim=0)[0]   # avg heads -> [seq]; row 0 = CLS/BOS

        total_tokens = encoded["input_ids"][0].shape[0]
        piece_counts = [max(len(self.tokenizer.tokenize(word)), 1) for word in words]

        cursor = 1  # skip the leading BOS/CLS special token
        last_valid_index = total_tokens - 1  # reserve the final slot for EOS
        scores: list[float] = []

        for count in piece_counts:
            if cursor >= last_valid_index:
                scores.append(0.0)  # word fell outside max_length after truncation
                continue
            end = min(cursor + count, last_valid_index)
            span = cls_attention[cursor:end]
            scores.append(float(span.sum()) if span.numel() else 0.0)
            cursor = end

        return [{"token": word, "score": score} for word, score in zip(words, scores)]

    def _predict_proba_for_shap(self, masked_texts) -> np.ndarray:
        encoded = self.tokenizer(
            list(masked_texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**encoded).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def explain_with_shap(
        self,
        text: str,
        max_evals: int = 80,
        predicted_id: int | None = None,
    ) -> list[dict]:
        import logging

        import shap

        logger = logging.getLogger(__name__)

        text_cleaned = self.preprocessor.clean_text(text)
        if not text_cleaned.strip():
            return []

        cache_key = (text_cleaned, max_evals, predicted_id if predicted_id is not None else -1)
        cached_scores = self._shap_cache.get(cache_key)
        if cached_scores is not None:
            return cached_scores

        if self._shap_explainer is None:
            masker = shap.maskers.Text(tokenizer=r"\s+")
            self._shap_explainer = shap.Explainer(
                self._predict_proba_for_shap,
                masker,
                algorithm="partition",
                output_names=LABELS,
                seed=7,
            )

        try:
            shap_values = self._shap_explainer([text_cleaned], max_evals=max_evals)
        except Exception as exc:
            logger.exception("SHAP explainer failed.")
            raise RuntimeError(f"SHAP explainer failed: {exc}") from exc

        if predicted_id is None:
            predicted_id = int(np.argmax(self._predict_proba_for_shap([text_cleaned])[0]))
        sv = shap_values[0, :, predicted_id]

        token_scores = [
            {"token": str(token).strip(), "score": float(score)}
            for token, score in zip(sv.data, sv.values)
            if str(token).strip()
        ]
        self._shap_cache[cache_key] = token_scores
        if len(self._shap_cache) > 32:
            self._shap_cache.pop(next(iter(self._shap_cache)))
        return token_scores