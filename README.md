# Vietnamese Hate Speech Detection Augmentation

This repository is a reproducible research pipeline for studying whether data augmentation improves Vietnamese Hate Speech Detection (HSD). The current implementation focuses on a solid research deliverable by the end of August 2026: fixed data splits, BT/EDA/LLM augmentation outputs, PhoBERT training, ablation metrics, and error analysis.

Frontend, FastAPI deployment, Docker, and full MLflow tracking are intentionally postponed until the research pipeline is complete.

The recommended workflow is:

- GitHub stores code, notebooks, configs, and docs.
- Hugging Face Datasets stores processed and augmented dataset versions.
- Hugging Face Models stores final trained checkpoints.
- Colab provides GPU compute.
- Google Drive is optional backup, not the primary source of truth.

## Current Status

Implemented:

- Reproducible config in `configs/config.yaml`
- Data split preparation script
- OOP augmentation interface with Back Translation, EDA, and controlled LLM prompt logging
- PhoBERT/XLM-R compatible training and evaluation scripts
- Metrics export for Accuracy, Macro F1, Weighted F1, HATE F1, confusion matrix, and predictions
- Dataset/model card drafts in `docs/`

In progress:

- Final processed train/dev/test datasets
- Actual large-scale augmentation runs
- Baseline and ablation experiment results

Planned after the research result is stable:

- Statistical significance testing
- MLflow tracking
- Web demo and deployment

## Project Structure

```text
Vietnamese-HSD-Augmentation/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── augmented/
├── docs/
│   ├── dataset_card.md
│   └── model_card.md
├── notebooks/
│   ├── 01_data_collection.ipynb
│   └── 02_back_translation.ipynb
├── results/
│   ├── error_analysis/
│   ├── figures/
│   └── metrics/
├── scripts/
│   ├── prepare_splits.py
│   ├── run_augmentation.py
│   ├── train_classifier.py
│   ├── evaluate_model.py
│   └── summarize_ablation.py
├── src/
│   ├── augmentation/
│   ├── models/
│   └── utils/
├── saved_models/
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Use Python 3.10+ if possible. GPU is strongly recommended for transformer training.

## Data Contract

The training pipeline expects CSV files with:

- `text`: Vietnamese text
- `label`: one of `CLEAN`, `OFFENSIVE`, `HATE`

Default paths are configured in `configs/config.yaml`:

- `data/processed/train.csv`
- `data/processed/dev.csv`
- `data/processed/test.csv`

If VLSP or another private dataset is used, keep the raw file outside Git and document the local path in `configs/config.yaml`.

## Hugging Face Workflow

Set a token in Colab before pushing or loading private repos:

```python
import os
from google.colab import userdata

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

Default Hub repos are configured in `configs/config.yaml`:

- Dataset repo: `HoaiAn001/vietnamese-hsd-combined`
- Model repo prefix: `HoaiAn001/vietnamese-hsd`

Use dataset configs to version each experiment:

```text
baseline
bt
eda
llm
combined
```

Keep the dataset private while it contains VLSP or any data that cannot be redistributed.

## Run Pipeline

Create fixed splits from one normalized CSV:

```bash
python scripts/prepare_splits.py --input data/raw/combined.csv
```

Create fixed splits and push them to Hugging Face:

```bash
python scripts/prepare_splits.py --input data/raw/combined.csv --push-to-hub --hf-config baseline
```

Run augmentation:

```bash
python scripts/run_augmentation.py --method eda
python scripts/run_augmentation.py --method bt
python scripts/run_augmentation.py --method llm
```

Run augmentation from the Hugging Face baseline config and push a new dataset config:

```bash
python scripts/run_augmentation.py --source hf --source-hf-config baseline --method bt --push-to-hub --target-hf-config bt
python scripts/run_augmentation.py --source hf --source-hf-config baseline --method eda --push-to-hub --target-hf-config eda
```

LLM augmentation writes prompts to `data/augmented/llm_prompts.jsonl`. Fill generations externally, then ingest them:

```bash
python scripts/run_augmentation.py --method llm --llm-generated data/augmented/llm_generated.csv
```

Train baseline:

```bash
python scripts/train_classifier.py --experiment baseline
```

Train from a Hugging Face dataset config on Colab:

```bash
python scripts/train_classifier.py --source hf --hf-config baseline --experiment baseline
python scripts/train_classifier.py --source hf --hf-config bt --experiment bt_only
```

Train and push the checkpoint to Hugging Face Models:

```bash
python scripts/train_classifier.py --source hf --hf-config bt --experiment bt_only --push-model-to-hub
```

Train with one augmentation set:

```bash
python scripts/train_classifier.py --experiment eda_only --augmented-file data/augmented/eda_only.csv
```

Evaluate:

```bash
python scripts/evaluate_model.py --model-dir saved_models/baseline --experiment baseline
```

Evaluate using a Hugging Face dataset config:

```bash
python scripts/evaluate_model.py --source hf --hf-config baseline --model-dir saved_models/baseline --experiment baseline
```

Summarize ablation:

```bash
python scripts/summarize_ablation.py
```

## Required Experiments

Minimum experiments for the August 2026 deliverable:

- `baseline`
- `bt_only`
- `eda_only`
- `llm_only`
- `combined`

Primary comparison metrics:

- Macro F1
- HATE F1
- Per-class F1
- Confusion matrix

## Notes

- Notebook code is useful for exploration, but reusable logic should live in `src/` and `scripts/`.
- Generated data, model checkpoints, and private datasets are intentionally excluded from Git.
- The project should prioritize reproducible research results before deployment work.
