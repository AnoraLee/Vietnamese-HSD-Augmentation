"""Shared Vietnamese text preprocessing for training and inference.

The model was trained with word segmentation. Every inference entry
point must use this module instead of tokenizing raw text independently.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from underthesea import word_tokenize

TOXIC_TEENCODE_MAP = {
    r"\bko\b": "không", r"\bhok\b": "không", r"\bdc\b": "được", r"\bđc\b": "được",
    r"\bj\b": "gì", r"\bbt\b": "bình thường", r"\btrc\b": "trước", r"\bnhg\b": "nhưng",
    r"\bthg\b": "thằng",
    r"\bdm\b": "địt mẹ", r"\bđm\b": "địt mẹ", r"\bdkm\b": "địt con mẹ", r"\bđkm\b": "địt con mẹ",
    r"\bvkl\b": "vãi lồn", r"\bvcl\b": "vãi lồn", r"\bvl\b": "vãi lồn", r"\bkl\b": "cái lồn",
    r"\bcc\b": "cục cứt", r"\bcđm\b": "cộng đồng mạng", r"\bml\b": "mặt lồn",
    r"\bđjt\b": "địt", r"\bdjt\b": "địt", r"\bdit\b": "địt",
    r"\bloz\b": "lồn", r"\blon\b": "lồn",
    r"\bcac\b": "cặc", r"\bcặk\b": "cặc", r"\bđb\b": "đầu buồi",
    r"\bcút\b": "cút", r"\bđĩ\b": "đĩ", r"\bphò\b": "phò",
}


class TextPreprocessor:
    """Normalize teencode and segment Vietnamese text with Underthesea."""

    def __init__(self, vncorenlp_dir: str | Path | None = None):
        # Giữ lại tham số vncorenlp_dir để tương thích ngược (backward compatibility)
        # với các file khác đang gọi TextPreprocessor(vncorenlp_dir=...)
        pass

    @staticmethod
    def text_key(text: str) -> str:
        """Normalize Unicode, lowercase, and collapse whitespace."""
        text = unicodedata.normalize("NFKC", str(text)).strip().lower()
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def normalize_teencode(text: str) -> str:
        """Expand selected Vietnamese teencode and toxic abbreviations."""
        for pattern, replacement in TOXIC_TEENCODE_MAP.items():
            text = re.sub(pattern, replacement, str(text), flags=re.IGNORECASE)
        return text

    def segment_text(self, text: str) -> str:
        """Segment text and surface runtime failures."""
        try:
            return word_tokenize(str(text), format="text")
        except Exception as exc:
            raise RuntimeError("Underthesea word segmentation failed.") from exc

    def clean_text(self, text: str) -> str:
        """Apply the canonical preprocessing pipeline used before PhoBERT inference."""
        return self.segment_text(self.normalize_teencode(self.text_key(text)))


@lru_cache(maxsize=1)
def get_text_preprocessor() -> TextPreprocessor:
    """Return the shared preprocessor used by all application entry points."""
    return TextPreprocessor()