from types import SimpleNamespace

import pytest

from src.utils import preprocess
from src.utils.preprocess import PreprocessingSetupError, TextPreprocessor


class FakeSegmenter:
    def word_segment(self, text):
        return [text.replace("học kỳ", "học_kỳ")]


def test_clean_text_normalizes_teencode_and_segments():
    preprocessor = object.__new__(TextPreprocessor)
    preprocessor.rdrsegmenter = FakeSegmenter()

    assert preprocessor.clean_text("  HỌC KỲ ko ổn VL ") == "học_kỳ không ổn vãi lồn"


def test_missing_vncorenlp_assets_raise_actionable_error(tmp_path):
    with pytest.raises(PreprocessingSetupError, match="setup_vncorenlp.py"):
        TextPreprocessor(tmp_path)


def test_configure_java_home_derives_value_from_java_runtime(monkeypatch, tmp_path):
    java_home = tmp_path / "jdk"
    java_binary = java_home / "bin" / preprocess._java_executable_name()
    java_binary.parent.mkdir(parents=True)
    java_binary.touch()

    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(preprocess.shutil, "which", lambda _: str(java_binary))
    monkeypatch.setattr(
        preprocess.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stderr=f"    java.home = {java_home}\n"),
    )

    assert preprocess.configure_java_home() == java_home
    assert preprocess.os.environ["JAVA_HOME"] == str(java_home)
