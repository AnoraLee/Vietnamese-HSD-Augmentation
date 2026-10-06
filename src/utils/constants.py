from __future__ import annotations

from pathlib import Path

from src.utils.config import PROJECT_DIR, load_config, resolve_path

_cfg = load_config()

LABELS: list[str] = list(_cfg["project"]["label_names"])
if not LABELS:
    raise ValueError("project.label_names must not be empty.")

label2id: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
id2label: dict[int, str] = {i: label for label, i in label2id.items()}

_paths = _cfg.get("paths", {})

DATA_DIR: Path = resolve_path(_paths.get("processed_dir", "data/processed"))
MODELS_DIR: Path = resolve_path(_paths.get("models_dir", "models"))
RESULTS_DIR: Path = resolve_path(_paths.get("results_dir", "results"))
FIGURES_DIR: Path = resolve_path(_paths.get("figures_dir", "results/figures"))
METRICS_DIR: Path = resolve_path(_paths.get("metrics_dir", "results/metrics"))
ERROR_ANALYSIS_DIR: Path = resolve_path(_paths.get("error_analysis_dir", "results/error_analysis"))


def ensure_output_dirs() -> None:
    """Tạo output dirs theo nhu cầu.

    KHÔNG chạy ở import time -- import constants từ test hoặc API không
    được phép touch disk. Script nào ghi file thì gọi hàm này 1 lần.
    """
    for d in (RESULTS_DIR, FIGURES_DIR, METRICS_DIR, ERROR_ANALYSIS_DIR):
        d.mkdir(parents=True, exist_ok=True)

_hf = _cfg["huggingface"]

HF_DATASET_REPO: str = _hf["dataset_repo_id"]
HF_RESULTS_REPO: str = _hf["results_repo_id"]
HF_DEFAULT_DATASET_CONFIG: str = _hf.get("default_dataset_config", "default")
HF_SPLIT_NAMES: dict[str, str] = dict(_hf["split_names"])
HF_TOKEN_ENV: str = _hf.get("token_env", "HF_TOKEN")
HF_SLOW_TOKENIZER_MARKERS: tuple[str, ...] = tuple(
    _hf.get("slow_tokenizer_models", ["phobert"])
)

HF_MODEL_IDS: dict[str, str] = dict(_hf["models"])
EXPERIMENT_ORDER: list[str] = list(_hf["experiment_order"])
HF_DEFAULT_EXPERIMENT: str = _hf.get("default_experiment", EXPERIMENT_ORDER[0])

HF_MODEL_REPO_PREFIX: str = _hf.get("model_repo_prefix", "")


def needs_slow_tokenizer(model_id: str) -> bool:
    lowered = model_id.lower()
    return any(marker in lowered for marker in HF_SLOW_TOKENIZER_MARKERS)

_unknown_in_order = [e for e in EXPERIMENT_ORDER if e not in HF_MODEL_IDS]
if _unknown_in_order:
    raise ValueError(
        f"huggingface.experiment_order references unknown experiments: "
        f"{_unknown_in_order}. Add them under huggingface.models."
    )

_missing_from_order = [e for e in HF_MODEL_IDS if e not in EXPERIMENT_ORDER]
if _missing_from_order:
    raise ValueError(
        f"huggingface.models has entries not listed in experiment_order: "
        f"{_missing_from_order}. Add them to experiment_order."
    )

if HF_DEFAULT_EXPERIMENT not in HF_MODEL_IDS:
    raise ValueError(
        f"huggingface.default_experiment={HF_DEFAULT_EXPERIMENT!r} "
        f"không có trong huggingface.models."
    )

_training = _cfg.get("training", {})

TRAINING_MODEL_NAME: str = _training.get("model_name", "vinai/phobert-base")
TRAINING_MAX_LENGTH: int = int(_training.get("max_length", 128))
TRAINING_BATCH_SIZE: int = int(_training.get("batch_size", 16))
TRAINING_LEARNING_RATE: float = float(_training.get("learning_rate", 2e-5))
TRAINING_EPOCHS: int = int(_training.get("epochs", 3))
TRAINING_WEIGHT_DECAY: float = float(_training.get("weight_decay", 0.01))
TRAINING_WARMUP_RATIO: float = float(_training.get("warmup_ratio", 0.06))
TRAINING_FP16: bool = bool(_training.get("fp16", False))
TRAINING_BF16: bool = bool(_training.get("bf16", False))
TRAINING_SAVE_TOTAL_LIMIT: int = int(_training.get("save_total_limit", 2))
TRAINING_METRIC_FOR_BEST_MODEL: str = _training.get("metric_for_best_model", "macro_f1")
TRAINING_OUTPUT_DIR_TEMPLATE: str = _training.get(
    "output_dir_template", "models/{experiment}_phobert"
)

TRAINING_SEED: int = int(_cfg["project"].get("seed", 42))