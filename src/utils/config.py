"""Load configs/config.yaml into a plain dict, with helper accessors for
the paths/settings used across src/ and scripts/.

Usage:
    from src.utils.config import load_config
    config = load_config()
    config["huggingface"]["dataset_repo_id"]
    config["paths"]["models_dir"]
"""
from pathlib import Path
from functools import lru_cache

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Read configs/config.yaml once and cache it. Pass a different
    config_path for tests or alternate configs (bypasses the cache)."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str, project_dir: Path) -> Path:
    """Turn a config-relative path (e.g. "data/augmented/aug_bt.csv")
    into an absolute Path under the given project root."""
    return project_dir / relative_path
