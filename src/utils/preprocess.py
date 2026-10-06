from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from underthesea import word_tokenize

TOXIC_TEENCODE_MAP: dict[str, str] = {
    r"\bko\b": "không", r"\bhok\b": "không", r"\bdc\b": "được", r"\bđc\b": "được",
    r"\bj\b": "gì", r"\bbt\b": "bình thường", r"\btrc\b": "trước", r"\bnhg\b": "nhưng",
    r"\bthg\b": "thằng", r"\bdm\b": "địt mẹ", r"\bđm\b": "địt mẹ",
    r"\bdkm\b": "địt con mẹ",
    r"\bvkl\b": "vãi lồn", r"\bvcl\b": "vãi lồn", r"\bvl\b": "vãi lồn",
    r"\bkl\b": "cái lồn",
    r"\bcc\b": "cục cứt", r"\bcđm\b": "cộng đồng mạng", r"\bml\b": "mặt lồn",
    r"\bđjt\b": "địt", r"\bdjt\b": "địt", r"\bdit\b": "địt",
    r"\bloz\b": "lồn", r"\blon\b": "lồn",
    r"\bcac\b": "cặc", r"\bcặk\b": "cặc", r"\bđb\b": "đầu buồi",
}

FORCE_COMPOUND_TERMS: tuple[str, ...] = (
    "địt_mẹ", "địt_con_mẹ", "vãi_lồn", "cái_lồn", "cục_cứt", "mặt_lồn",
    "đầu_buồi", "thằng_chó", "con_chó", "phản_động", "lũ_ngu", "óc_chó",
)

_COMPOUND_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (
        re.compile(
            r"\b" + r"\s+".join(re.escape(w) for w in term.split("_")) + r"\b",
            flags=re.IGNORECASE,
        ),
        term,
    )
    for term in FORCE_COMPOUND_TERMS
)

class TextPreprocessor:

    @staticmethod
    def text_key(text: str) -> str:
        text = unicodedata.normalize("NFKC", str(text)).strip().lower()
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def remove_repeated_chars(text: str) -> str:
        return re.sub(r"(.)\1{2,}", r"\1", text)

    @staticmethod
    def normalize_teencode(text: str) -> str:
        """Áp TOXIC_TEENCODE_MAP (word-boundary, case-insensitive)."""
        for pattern, replacement in TOXIC_TEENCODE_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def pre_segment_toxic(text: str) -> str:
        for pattern, replacement in _COMPOUND_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    def segment_text(self, text: str) -> str:
        try:
            return word_tokenize(text, format="text")
        except Exception as exc:
            raise RuntimeError("Underthesea word segmentation failed.") from exc

    def clean_text(self, text: str) -> str:
        text = self.text_key(text)
        text = self.remove_repeated_chars(text)
        text = self.normalize_teencode(text)
        text = self.pre_segment_toxic(text)
        text = self.segment_text(text)
        return text


@lru_cache(maxsize=1)
def get_text_preprocessor() -> TextPreprocessor:
    return TextPreprocessor()