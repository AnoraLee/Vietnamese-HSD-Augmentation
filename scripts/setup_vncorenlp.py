"""Explicitly download VnCoreNLP assets for a new local environment.

This script is intentionally separate from API startup so serving never causes
a network download or modifies the project directory unexpectedly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import py_vncorenlp  # noqa: E402

from src.utils.preprocess import default_vncorenlp_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download local VnCoreNLP assets.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_vncorenlp_dir(),
        help="Directory that will contain VnCoreNLP-1.2.jar and models/.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    jar_path = output_dir / "VnCoreNLP-1.2.jar"
    models_dir = output_dir / "models"

    if jar_path.is_file() and models_dir.is_dir():
        print(f"VnCoreNLP assets are already available at {output_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading VnCoreNLP assets to {output_dir}...")
    py_vncorenlp.download_model(save_dir=str(output_dir))
    print("Download complete. Ensure Java 8+ is available on PATH before starting the API.")


if __name__ == "__main__":
    main()
