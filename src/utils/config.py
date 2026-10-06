from __future__ import annotations

import copy
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _detect_project_dir() -> Path:
    env = os.environ.get("HSD_PROJECT_DIR")
    if env:
        return Path(env).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]
    if (repo_root / "configs").is_dir():
        return repo_root

    colab_drive = Path("/content/drive/MyDrive/Hate_Speech_Detection")
    if colab_drive.exists():
        return colab_drive.resolve()

    return Path.cwd().resolve()


PROJECT_DIR: Path = _detect_project_dir()
DEFAULT_CONFIG_PATH: Path = PROJECT_DIR / "configs" / "config.yaml"


@lru_cache(maxsize=None)
def _load_raw(config_path: Path) -> dict[str, Any]:
    """Parse YAML once per path. Callers never touch this directly."""
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Set HSD_PROJECT_DIR hoặc đặt configs/config.yaml ở repo root."
        )
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping, got {type(data).__name__}.")
    return data


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    path = Path(config_path).expanduser().resolve() if config_path else DEFAULT_CONFIG_PATH
    return copy.deepcopy(_load_raw(path))


def resolve_path(relative: str | Path, base: Path | None = None) -> Path:
    p = Path(relative)
    if p.is_absolute():
        return p
    return (base or PROJECT_DIR) / p