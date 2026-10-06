from __future__ import annotations

import pytest

from src.utils.preprocess import TextPreprocessor, get_text_preprocessor


@pytest.fixture
def pre():
    return get_text_preprocessor()


def test_get_text_preprocessor_is_singleton():
    assert get_text_preprocessor() is get_text_preprocessor()


def test_text_key_normalizes_and_lowercases(pre):
    assert pre.text_key("  HELLO   World  ") == "hello world"
    assert pre.text_key("e\u0301") == "é"


def test_remove_repeated_chars_keeps_two():
    """'nguuuu' → 'ngu'; 'xoong' giữ nguyên vì chỉ 2 ký tự trùng."""
    p = TextPreprocessor()
    assert p.remove_repeated_chars("đẹppp") == "đẹp"
    assert p.remove_repeated_chars("nguuuu") == "ngu"
    assert p.remove_repeated_chars("xoong") == "xoong"


def test_normalize_teencode_maps_common_tokens(pre):
    out = pre.normalize_teencode("thằng này ngu vl")
    assert "vãi lồn" in out
    assert "vl" not in out.split()


def test_pre_segment_toxic_glues_compound(pre):
    out = pre.pre_segment_toxic("thằng chó này")
    assert "thằng_chó" in out


def test_pre_segment_toxic_handles_multiple_spaces(pre):
    """Regex cho phép nhiều khoảng trắng giữa các từ trong cụm."""
    out = pre.pre_segment_toxic("thằng   chó này")
    assert "thằng_chó" in out


def test_pre_segment_toxic_does_not_match_substring(pre):
    """'thằng chóai' (không tồn tại) không bị dính thành 'thằng_chóai'."""
    out = pre.pre_segment_toxic("thằng chóai")
    assert "thằng_chóai" not in out


def test_clean_text_end_to_end(pre):
    raw = "HỌC KỲ ko ổn VL!!!"
    cleaned = pre.clean_text(raw)
    assert isinstance(cleaned, str)
    assert cleaned  # không rỗng
    assert cleaned == cleaned.lower()