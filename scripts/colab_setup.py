from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from scripts._common import setup_logging

logger = setup_logging.__self__ if False else None


def _mount_drive_if_colab() -> None:
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        print("[drive] đã mount sẵn.")
        return
    try:
        from google.colab import drive
    except ImportError:
        print("[drive] không phải Colab, bỏ qua.")
        return
    drive.mount("/content/drive")


def _install_requirements(requirements: Path) -> None:
    """Cài requirements.txt (không pin version — pip sẽ skip cái đã có)."""
    if not requirements.exists():
        print(f"[pip] không có {requirements}, bỏ qua.")
        return
    print(f"[pip] installing {requirements.name}...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "--no-warn-script-location", "-r", str(requirements)],
        check=True,
    )


def _login_hf_if_token() -> None:
    import os
    from huggingface_hub import login

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[hf] không có HF_TOKEN trong env — bỏ qua login.")
        return
    login(token=token, add_to_git_credential=False)
    print("[hf] đã login.")


def _verify_project_layout() -> None:
    required = ["configs/config.yaml", "src/utils/config.py", "src/utils/constants.py"]
    missing = [p for p in required if not (PROJECT_DIR / p).exists()]
    if missing:
        raise SystemExit(
            f"[layout] thiếu file bắt buộc trong {PROJECT_DIR}: {missing}\n"
            f"Kiểm tra PROJECT_DIR hoặc clone lại repo."
        )
    print(f"[layout] PROJECT_DIR = {PROJECT_DIR}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pip", action="store_true", help="Bỏ qua pip install.")
    parser.add_argument("--skip-hf", action="store_true", help="Bỏ qua HF login.")
    args = parser.parse_args()

    _mount_drive_if_colab()
    _verify_project_layout()

    if not args.skip_pip:
        _install_requirements(PROJECT_DIR / "requirements.txt")

    if not args.skip_hf:
        _login_hf_if_token()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.utils.config import load_config
    from src.utils.constants import EXPERIMENT_ORDER, HF_DEFAULT_EXPERIMENT

    cfg = load_config()
    print(f"[config] project.name = {cfg['project']['name']}")
    print(f"[config] default experiment = {HF_DEFAULT_EXPERIMENT}")
    print(f"[config] experiments = {EXPERIMENT_ORDER}")
    print("[done] setup hoàn tất.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())