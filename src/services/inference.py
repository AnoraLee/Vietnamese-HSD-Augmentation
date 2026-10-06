from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict
from time import perf_counter
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.constants import (
    HF_TOKEN_ENV,
    LABELS,
    TRAINING_MAX_LENGTH,
    needs_slow_tokenizer,
)
from src.utils.preprocess import TextPreprocessor, get_text_preprocessor

logger = logging.getLogger(__name__)

_WARMUP_TEXT = "không độc hại"

_SHAP_CACHE_SIZE = 32


class HSDInferenceService:
    """Load model 1 lần, serve predictions + SHAP explanations."""

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        max_length: int | None = None,
        preprocessor: TextPreprocessor | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length or TRAINING_MAX_LENGTH
        self.preprocessor = preprocessor or get_text_preprocessor()
        self._hf_token = hf_token or os.environ.get(HF_TOKEN_ENV)
        self._slow_tokenizer = needs_slow_tokenizer(self.model_name_or_path)
        self._shap_explainer: Any | None = None
        self._shap_cache: OrderedDict[tuple, list[dict]] = OrderedDict()
        self._cache_lock = threading.Lock()

        logger.info("Loading model on %s: %s", self.device, self.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            use_fast=not self._slow_tokenizer,
            token=self._hf_token,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path,
            token=self._hf_token,
        ).to(self.device)
        self.model.eval()

    @property
    def preprocessing_description(self) -> str:
        """Mô tả pipeline cho endpoint /metadata."""
        if self._slow_tokenizer:
            return "Underthesea word segmentation + teencode normalization"
        return "Raw text + teencode normalization (no word segmentation)"

    def _prepare_input(self, text: str) -> str:
        """Trả về string chính xác sẽ được đưa vào tokenizer.

        PhoBERT train trên text đã word-segment (compound words nối bằng
        '_'); ViSoBERT/XLM-R train trên raw text. Cả hai đều qua cùng
        bước cleanup nhẹ để behavior user thấy được nhất quán.
        """
        if self._slow_tokenizer:
            return self.preprocessor.clean_text(text)
        return self.preprocessor.normalize_teencode(
            self.preprocessor.remove_repeated_chars(self.preprocessor.text_key(text))
        )

    def predict(self, text: str) -> dict:
        started_at = perf_counter()
        text_input = self._prepare_input(text)

        encoded = self.tokenizer(
            text_input,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            probabilities = torch.softmax(self.model(**encoded).logits, dim=-1)[0]

        probability_values = [float(v) for v in probabilities.cpu().tolist()]
        predicted_id = int(probabilities.argmax())

        return {
            "text": text,
            "text_cleaned": text_input,
            "label": LABELS[predicted_id],
            "predicted_id": predicted_id,
            "confidence": probability_values[predicted_id],
            "probabilities": dict(zip(LABELS, probability_values, strict=True)),
            "latency_ms": round((perf_counter() - started_at) * 1000, 2),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Batched forward pass -- 1 pass cho cả batch.

        Nhanh hơn ~20-50x so với gọi predict() trong vòng lặp cho batch 100
        vì tận dụng được GPU parallelism.
        """
        if not texts:
            return []

        started_at = perf_counter()
        cleaned = [self._prepare_input(t) for t in texts]

        encoded = self.tokenizer(
            cleaned,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            probs = torch.softmax(self.model(**encoded).logits, dim=-1).cpu()

        per_item_ms = round((perf_counter() - started_at) * 1000 / len(texts), 2)

        results: list[dict] = []
        for i, text in enumerate(texts):
            p = probs[i].tolist()
            pid = int(probs[i].argmax())
            results.append({
                "text": text,
                "text_cleaned": cleaned[i],
                "label": LABELS[pid],
                "predicted_id": pid,
                "confidence": p[pid],
                "probabilities": dict(zip(LABELS, p, strict=True)),
                "latency_ms": per_item_ms,
                "token_importance": [],
            })
        return results

    def warm_up(self) -> None:
        """1 dummy forward pass để CUDA context + kernel cache nóng trước
        khi request thật đầu tiên đập vào endpoint."""
        encoded = self.tokenizer(
            _WARMUP_TEXT,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            self.model(**encoded)

    def _predict_proba_for_shap(self, masked_texts) -> np.ndarray:
        """Callable cho shap.Explainer -- phải trả [n_samples, n_classes]."""
        encoded = self.tokenizer(
            [str(t) for t in masked_texts],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = self.model(**encoded).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()

    def _get_shap_explainer(self):
        """Build explainer 1 lần / service instance."""
        if self._shap_explainer is None:
            import shap

            masker = shap.maskers.Text(tokenizer=r"\s+")
            self._shap_explainer = shap.Explainer(
                self._predict_proba_for_shap,
                masker,
                algorithm="partition",
                output_names=LABELS,
                seed=7,
            )
        return self._shap_explainer

    def _cache_get(self, key: tuple) -> list[dict] | None:
        """LRU lookup -- an toàn với multi-thread FastAPI."""
        with self._cache_lock:
            if key in self._shap_cache:
                self._shap_cache.move_to_end(key)
                return self._shap_cache[key]
        return None

    def _cache_put(self, key: tuple, value: list[dict]) -> None:
        with self._cache_lock:
            self._shap_cache[key] = value
            self._shap_cache.move_to_end(key)
            while len(self._shap_cache) > _SHAP_CACHE_SIZE:
                self._shap_cache.popitem(last=False)

    def explain_with_shap(
        self,
        text: str,
        max_evals: int = 80,
        predicted_id: int | None = None,
    ) -> list[dict]:
        """Trả về SHAP score theo token cho class đã chỉ định.

        Preprocessing ủy thác cho `_prepare_input`, nên explanation luôn
        giải thích CÙNG input mà predict() sẽ dùng.
        """
        import shap  # noqa: F401 -- validate dependency sớm

        text_cleaned = self._prepare_input(text)
        if not text_cleaned.strip():
            return []

        cache_key = (
            text_cleaned,
            max_evals,
            predicted_id if predicted_id is not None else -1,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        explainer = self._get_shap_explainer()

        try:
            shap_values = explainer([text_cleaned], max_evals=max_evals)
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

        self._cache_put(cache_key, token_scores)
        return token_scores