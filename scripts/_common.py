from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.utils.constants import (
    HF_TOKEN_ENV,
    TRAINING_SEED,
)


def setup_logging(level: int = logging.INFO) -> None:
    """Format log thống nhất giữa các script."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def set_seed(seed: int = TRAINING_SEED) -> None:
    """Set seed cho Python / NumPy / PyTorch. Không fail nếu thiếu torch."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_hf_token() -> str | None:
    """Đọc HF token từ env. Trả None nếu không có (repo public)."""
    return os.environ.get(HF_TOKEN_ENV)


def hf_login_from_env() -> None:
    """Login Hugging Face nếu có token trong env. Silent nếu không."""
    token = get_hf_token()
    if not token:
        return
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)


def parse_experiments(csv_value: str | None) -> list[str]:
    """'a,b,c' -> ['a', 'b', 'c']. None/rỗng -> []."""
    if not csv_value:
        return []
    return [e.strip() for e in csv_value.split(",") if e.strip()]


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Ghi JSON với indent=2, đảm bảo parent dir tồn tại."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)