"""Shared Vietnamese text preprocessing for training and inference.

The model was trained with VnCoreNLP word segmentation. Every inference entry
point must use this module instead of tokenizing raw text independently.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import unicodedata
from functools import lru_cache
from pathlib import Path

from src.utils.config import load_config

try:
    import py_vncorenlp
except ImportError:
    py_vncorenlp = None


class PreprocessingSetupError(RuntimeError):
    """Raised when the local VnCoreNLP runtime is not ready for inference."""


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


def default_vncorenlp_dir() -> Path:
    """Resolve the configured VnCoreNLP asset directory under the project."""
    project_dir = Path(__file__).resolve().parents[2]
    configured_path = Path(load_config()["preprocessing"]["vncorenlp_dir"])
    return configured_path if configured_path.is_absolute() else project_dir / configured_path


def _java_executable_name() -> str:
    return "java.exe" if os.name == "nt" else "java"


def configure_java_home() -> Path:
    """Ensure ``JAVA_HOME`` is available before pyjnius initializes.

    The value is respected when already valid. Otherwise, it is derived from
    the Java runtime discovered on PATH, which avoids a machine-specific path.
    """
    configured_home = os.getenv("JAVA_HOME")
    if configured_home:
        java_home = Path(configured_home)
        if (java_home / "bin" / _java_executable_name()).is_file():
            return java_home

    java_executable = shutil.which("java")
    if java_executable is None:
        raise PreprocessingSetupError(
            "Java was not found on PATH. Install Java 8 or newer and restart the terminal."
        )

    try:
        runtime_details = subprocess.run(
            [java_executable, "-XshowSettings:properties", "-version"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except OSError as exc:
        raise PreprocessingSetupError("Java could not be executed from PATH.") from exc

    match = re.search(r"^\s*java\.home\s*=\s*(.+?)\s*$", runtime_details.stderr, re.MULTILINE)
    if match is None:
        raise PreprocessingSetupError(
            "Could not derive JAVA_HOME from the Java runtime. "
            "Set JAVA_HOME to your JDK directory, then restart the terminal."
        )

    java_home = Path(match.group(1))
    if not (java_home / "bin" / _java_executable_name()).is_file():
        raise PreprocessingSetupError(
            f"Java reported an invalid home directory: {java_home}. "
            "Set JAVA_HOME to a JDK directory containing bin/java."
        )

    os.environ["JAVA_HOME"] = str(java_home)
    return java_home


def validate_vncorenlp_setup(vncorenlp_dir: Path) -> None:
    """Validate required local runtime assets without downloading or mutating them."""
    jar_path = vncorenlp_dir / "VnCoreNLP-1.2.jar"
    models_dir = vncorenlp_dir / "models"

    if not jar_path.is_file() or not models_dir.is_dir():
        raise PreprocessingSetupError(
            f"VnCoreNLP assets are incomplete at {vncorenlp_dir}. "
            "Run `python scripts/setup_vncorenlp.py` from the project root."
        )
    if py_vncorenlp is None:
        raise PreprocessingSetupError(
            "The `py_vncorenlp` package is not installed. "
            "Run `python -m pip install -r requirements.txt`."
        )
    configure_java_home()


class VnCoreNLPSingleton:
    """Keep one segmenter process per VnCoreNLP asset directory."""

    _instances: dict[Path, object] = {}

    @classmethod
    def get_instance(cls, vncorenlp_dir: Path):
        if vncorenlp_dir not in cls._instances:
            cls._instances[vncorenlp_dir] = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"], save_dir=str(vncorenlp_dir)
            )
        return cls._instances[vncorenlp_dir]


class TextPreprocessor:
    """Normalize teencode and segment Vietnamese text with VnCoreNLP."""

    def __init__(self, vncorenlp_dir: str | Path | None = None):
        self.vncorenlp_dir = Path(vncorenlp_dir) if vncorenlp_dir else default_vncorenlp_dir()
        self.vncorenlp_dir = self.vncorenlp_dir.resolve()
        validate_vncorenlp_setup(self.vncorenlp_dir)
        self.rdrsegmenter = VnCoreNLPSingleton.get_instance(self.vncorenlp_dir)

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
        """Segment text and surface runtime failures instead of silently changing input."""
        try:
            return " ".join(self.rdrsegmenter.word_segment(str(text)))
        except Exception as exc:
            raise RuntimeError("VnCoreNLP word segmentation failed.") from exc

    def clean_text(self, text: str) -> str:
        """Apply the canonical preprocessing pipeline used before PhoBERT inference."""
        return self.segment_text(self.normalize_teencode(self.text_key(text)))


@lru_cache(maxsize=1)
def get_text_preprocessor() -> TextPreprocessor:
    """Return the shared preprocessor used by all application entry points."""
    return TextPreprocessor()
