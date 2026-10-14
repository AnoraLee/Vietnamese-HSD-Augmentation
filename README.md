# HSD Studio — Vietnamese Hate Speech Detection

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#quick-start)
[![Models](https://img.shields.io/badge/models-PhoBERT%20%7C%20ViSoBERT-orange)](#the-seven-experiments)
[![Backend](https://img.shields.io/badge/backend-FastAPI-009688)](#api-reference)
[![Frontend](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61DAFB)](#frontend)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

A transformer-based classification suite for Vietnamese hate speech (`CLEAN` / `OFFENSIVE` / `HATE`), built around one central question: **which data-augmentation strategy actually improves hate speech detection in Vietnamese social media texts?**

The repository contains the end-to-end research pipeline across PhoBERT and ViSoBERT backbones, a high-performance FastAPI service supporting on-demand SHAP explainability (XAI), and a responsive React web studio for interactive side-by-side model evaluation and error analysis.

**Live demo:** https://vietnamese-hsd-augmentation-xx46.vercel.app/  
**Backend (Hugging Face Space):** https://huggingface.co/spaces/BonTori/vietnamese_tdtu_hsd_detectToxic  
*(Direct API Endpoint: `https://bontori-vietnamese-tdtu-hsd-detecttoxic.hf.space`)*

> **Note:** The backend on Hugging Face Spaces runs on CPU and may take 30–60 seconds to spin up from cold sleep on initial requests.

---

## Contents

- [Results](#results)
- [Quick start](#quick-start)
- [Docker](#docker)
- [Deployment](#deployment)
- [Project structure](#project-structure)
- [The seven experiments](#the-seven-experiments)
- [Research pipeline](#research-pipeline)
- [API reference](#api-reference)
- [Frontend](#frontend)
- [Data](#data)
- [Installation](#installation)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Acknowledgments](#acknowledgments)
- [Authors](#authors)
- [License](#license)

---

## Results

All experiments are evaluated on the exact same frozen test set (**6,576 samples**; the development set has **2,650 samples**). Only the training set and model backbone change between experiments.

| Experiment | Backbone | Accuracy | Macro F1 | Weighted F1 | HATE F1 |
| ---------- | -------- | -------- | -------- | ----------- | ------- |
| `phobert_baseline` | PhoBERT | 84.4% | 60.2% | 84.7% | 53.8% |
| `phobert_bt` | PhoBERT | 83.0% | 44.5% | 79.8% | 42.7% |
| `phobert_eda` | PhoBERT | 78.4% | 58.3% | 81.3% | 56.5% |
| `phobert_llm` | PhoBERT | 83.5% | 60.9% | 84.5% | 56.1% |
| `phobert_combined` | PhoBERT | 78.0% | 57.3% | 80.7% | 58.0% |
| `visobert_baseline` | ViSoBERT | **84.5%** | **61.5%** | **85.1%** | 55.5% |
| `visobert_combined` | ViSoBERT | 77.9% | 57.0% | 80.5% | **58.7%** |

### Key Findings
1. **The Class Imbalance Trade-off:** The corpus is heavily imbalanced (`CLEAN` accounts for the vast majority). Standard Accuracy and Weighted F1 predominantly reflect performance on clean sentences. For content moderation, **HATE F1** is the metric that matters most.
2. **Value of Data Augmentation:** Combining multiple augmentation sources (`phobert_combined` and `visobert_combined`) yields the highest HATE F1 scores (**58.0%** and **58.7%**, up by +4.2% and +4.9% over PhoBERT baseline), successfully trading off a small percentage of accuracy on clean samples to capture significantly more genuine toxicity.
3. **Architecture Comparison:** ViSoBERT demonstrates superior representations on informal Vietnamese social media abbreviations and slang, with `visobert_baseline` achieving the best overall Macro F1 (61.5%) and Weighted F1 (85.1%).
4. **Back-translation Semantic Drift:** Round-trip translation (vi $\rightarrow$ en $\rightarrow$ vi) frequently softened aggressive terms and colloquial slurs, causing `phobert_bt` to struggle on minority classes.

Consolidated metrics are stored in `results/experiments_summary.csv` and per-experiment metrics are in `results/metrics/metrics_<experiment>.json`. Run `python scripts/summarize_ablation.py` to regenerate the comparison table and chart under `results/figures/`.

---

## Quick start

**Requirements:** Python 3.10+ and Node.js (v18+).

### 1. Backend Service

The backend loads the model specified by `MODEL_EXPERIMENT` (default: `phobert_combined`). To preload multiple models for instant switching in the web UI, set `LOADED_EXPERIMENTS` (comma-separated list).

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Set default model (or specify LOADED_EXPERIMENTS="phobert_baseline,phobert_combined,visobert_baseline")
$env:MODEL_EXPERIMENT = "phobert_combined"

# Start FastAPI server
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export MODEL_EXPERIMENT=phobert_combined
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Frontend Studio

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The status indicator will switch to **online** once connected to `http://127.0.0.1:8000/health`.

---

## Docker

Run the containerized backend using Docker Compose:

```bash
docker compose up --build
```

- **Service:** `backend` (container: `hsd-backend`)
- **Port:** `8000:8000`
- **Volume:** `hf_cache` mounted at `/app/.cache/huggingface` to persist downloaded Hugging Face weights across container restarts.
- **Healthcheck:** Queries `http://localhost:8000/health` automatically.

---

## Deployment

The system is deployed as two decoupled components:

- **Backend (Hugging Face Spaces):** Containerized FastAPI app running at `https://huggingface.co/spaces/BonTori/vietnamese_tdtu_hsd_detectToxic` (`CORS_ORIGINS` configured to allow the Vercel domain).
- **Frontend (Vercel):** React SPA hosted at `https://vietnamese-hsd-augmentation-xx46.vercel.app/` with `VITE_API_BASE_URL` pointing to the Hugging Face Space URL.

---

## Project structure

```
Vietnamese-HSD-Augmentation/
├── api/
│   ├── main.py                  # FastAPI application entrypoint (api.main:app)
│   ├── schemas.py               # Pydantic request & response schemas
│   └── evaluation_routes.py     # Evaluation & metrics endpoints (/evaluation/*)
├── configs/
│   └── config.yaml              # Central experiment, model, and path configuration
├── data/                        # Not committed
│   ├── raw/                     # Raw vihsd_*.csv, voz_*.csv, tdtu_*.csv
│   ├── processed/               # Frozen datasets: tdtu_train.csv, tdtu_dev.csv, tdtu_test.csv
│   └── augmented/               # aug_bt.csv, aug_eda.csv, aug_llm.csv, aug_combined.csv
├── frontend/                    # React + Vite application
│   ├── public/                  # Static assets (logo-mark.svg)
│   └── src/                     # App.jsx, styles, components
├── models/                      # Local model checkpoints (not committed)
├── notebooks/
│   ├── 1_data_exploration/      # Data collection from HF & YouTube
│   ├── 2_preprocessing/         # Text normalization, cleaning, word segmentation
│   ├── 3_augmentation/          # BT, EDA, LLM, and Combined augmentation pipelines
│   ├── 4_training/              # PhoBERT & ViSoBERT fine-tuning notebooks
│   └── 5_evaluation/            # Error analysis & metrics visualization
├── results/
│   ├── metrics/                 # metrics_<experiment>.json
│   ├── figures/                 # Confusion matrices, F1 comparison bar charts
│   ├── error_analysis/          # Misclassified sample CSVs & confusion summaries
│   └── experiments_summary.csv  # Consolidated ablation summary table
├── scripts/                     # Reusable CLI scripts
│   ├── check_dataset.py         # Integrity and distribution validation
│   ├── evaluate.py              # Model evaluation runner
│   ├── run_ablation.py          # Automated ablation runner
│   ├── run_augmentation.py      # Augmentation pipeline runner
│   ├── summarize_ablation.py    # Generates summary CSV & comparison plots
│   └── train.py                 # Training script
├── src/
│   ├── models/classifier.py     # Model architectures & wrappers
│   ├── services/inference.py    # PhoBERT/ViSoBERT inference service + SHAP XAI
│   └── utils/                   # Config, constants, preprocessing, evaluation helpers
├── tests/                       # Pytest test suite (API, schemas, preprocessing)
├── Dockerfile                   # Multi-stage production build
├── docker-compose.yml           # Compose specification for backend service
├── requirements.txt             # Project dependencies
└── LICENSE                      # Apache-2.0 License
```

---

## The seven experiments

| Experiment Key | Architecture | Description / Training Data |
| -------------- | ------------ | --------------------------- |
| `phobert_baseline` | PhoBERT-base | Fine-tuned on the original unaugmented training set |
| `phobert_bt` | PhoBERT-base | Baseline + Back-translation augmentation (vi $\rightarrow$ en $\rightarrow$ vi) |
| `phobert_eda` | PhoBERT-base | Baseline + Easy Data Augmentation (swap, insertion, deletion) |
| `phobert_llm` | PhoBERT-base | Baseline + Synthetic samples generated via LLMs (Gemma, Qwen) |
| `phobert_combined` | PhoBERT-base | Merged BT + EDA + LLM augmentations (deduplicated & filtered) |
| `visobert_baseline` | ViSoBERT | Fine-tuned on unaugmented data using ViSoBERT |
| `visobert_combined` | ViSoBERT | ViSoBERT fine-tuned on the combined augmented training set |

These keys are used consistently across the entire repository:
- Checkpoints: Hosted on Hugging Face Hub under `BonTori/` and `AnoraLee/` (mapped in `configs/config.yaml` and `src/utils/constants.py`).
- Metrics: `results/metrics/metrics_<experiment>.json`.
- API: Accepted in the `"model"` field of all prediction and explanation endpoints.

---

## Research pipeline

| Stage | Directory | Description |
| :---: | --------- | ----------- |
| **1** | `notebooks/1_data_exploration/` | Collect raw datasets from ViHSD, VOZ forums, and YouTube comments. |
| **2** | `notebooks/2_preprocessing/` | Unicode NFC normalization, teencode translation, Underthesea word segmentation, deduplication, and removal of cross-split contamination. Produces frozen `tdtu_train.csv`, `tdtu_dev.csv`, and `tdtu_test.csv`. |
| **3** | `notebooks/3_augmentation/` | Generate augmented samples independently (BT, EDA, LLM), filter for semantic drift and leakage, and synthesize `aug_combined.csv`. |
| **4** | `notebooks/4_training/` | Fine-tune PhoBERT and ViSoBERT checkpoints across all experimental setups. |
| **5** | `notebooks/5_evaluation/` | Evaluate on frozen test data, compute confusion matrices, and export misclassification reports. |

---

## API reference

### `POST /predict`

Classifies input Vietnamese text into `CLEAN`, `OFFENSIVE`, or `HATE`.

**Request:**
```json
{
  "text": "câu cần phân loại",
  "model": "phobert_combined"
}
```
*Note: `model` is optional and falls back to the server's default `MODEL_EXPERIMENT`.*

**Response:**
```json
{
  "text": "câu cần phân loại",
  "text_cleaned": "câu cần phân_loại",
  "label": "CLEAN",
  "predicted_id": 0,
  "confidence": 0.94,
  "probabilities": {
    "CLEAN": 0.94,
    "OFFENSIVE": 0.04,
    "HATE": 0.02
  },
  "latency_ms": 42.1,
  "model_used": "phobert_combined"
}
```

### `POST /explain`

Computes token-level attribution scores using **SHAP** (Shapley Additive Explanations). Run on-demand for model interpretability.

**Request:**
```json
{
  "text": "câu cần phân loại",
  "model": "phobert_combined",
  "max_evals": 100
}
```
*`max_evals` is optional (20–1000, defaults to 200 on GPU, 80 on CPU).*

**Response:**
```json
{
  "label": "OFFENSIVE",
  "method": "shap",
  "model_used": "phobert_combined",
  "token_scores": [
    { "token": "rất", "score": -0.12 },
    { "token": "tệ", "score": 0.48 }
  ],
  "latency_ms": 1420.5
}
```
*Positive scores indicate contribution towards the predicted class, while negative scores indicate pushback against it.*

### `POST /predict/batch`

Batched inference accepting 1 to 100 sentences in a single request:

```json
{
  "texts": ["câu thứ nhất", "câu thứ hai"],
  "model": "phobert_combined"
}
```

### `GET /health`

Health check endpoint returning active server state:

```json
{
  "status": "ok",
  "model": "phobert_combined",
  "device": "cpu",
  "available_models": ["phobert_combined", "phobert_baseline", "visobert_baseline"]
}
```

### `GET /metadata`

Exposes model metadata, preprocessing configuration, class labels, and loaded checkpoints.

### `GET /evaluation/metrics`

Returns benchmark metrics for all evaluated experiments (used by the frontend evaluation dashboard).

---

## Frontend

Built with **React + Vite** in `frontend/`. It communicates with the backend via `VITE_API_BASE_URL` (defaults to `http://127.0.0.1:8000` in development).

Features:
- **Interactive Classification Studio:** Real-time text analysis with confidence gauges and tokenized output.
- **Model Switching:** Switch dynamically between any preloaded PhoBERT and ViSoBERT checkpoints.
- **Explainability (XAI):** On-demand SHAP waterfall/token contribution visualizer.
- **Analysis History:** LocalStorage session history keeping the last 30 queries.
- **Benchmark & Error Analysis Dashboard:** Interactive metrics comparison table, rank badges, and confusion breakdown.
- **Dark / Light theme:** Fully responsive for desktop and mobile devices.

To point to a remote backend, copy `frontend/.env.example` to `frontend/.env.local` and set `VITE_API_BASE_URL=https://bontori-vietnamese-tdtu-hsd-detecttoxic.hf.space`.

---

## Data

The dataset combines the ViHSD benchmark with domain-specific social media text gathered from VOZ forums and YouTube comments.

- Raw data and intermediate augmented files are stored locally under `data/` and are not committed to git.
- Processed splits are configured in `configs/config.yaml`:
  - `data/processed/tdtu_train.csv` (86,719 samples)
  - `data/processed/tdtu_dev.csv` (2,650 samples)
  - `data/processed/tdtu_test.csv` (6,576 samples)

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Running Tests

Execute the automated test suite using `pytest`:

```bash
pytest tests/ -v
```

---

## Troubleshooting

- **`output_attentions=True` issues:** `src/services/inference.py` loads models with eager attention when needed to maintain compatibility with varying `transformers` releases.
- **`POST /explain` latency:** SHAP evaluation runs multiple forward passes over masked inputs. On CPU, keep `max_evals` between 20 and 80 for reasonable response times (~1–3 seconds).
- **CORS Errors:** Ensure your frontend URL is included in the backend's `CORS_ORIGINS` environment variable (comma-separated list).

---

## Known limitations

- **Social Media Domain Shift:** Fast-evolving youth slang, sarcasm, and intentionally obfuscated profanity remain challenging for automated classifiers.
- **SHAP Latency:** Explanations require compute budgets proportional to input sentence length.
- **Research Prototype:** Designed for academic evaluation and benchmarking of data augmentation strategies; human oversight is recommended for production content moderation.

---

## Acknowledgments

This research builds upon the following open-source resources and foundational works:
- [PhoBERT](https://github.com/VinAIResearch/PhoBERT) (VinAI Research)
- [ViSoBERT](https://github.com/VinAIResearch/ViSoBERT) (VinAI Research)
- [Underthesea](https://github.com/undertheseanlp/underthesea) (Vietnamese NLP Toolkit)
- [SHAP](https://github.com/shap-project/shap) (Shapley Additive Explanations)
- ViHSD (Vietnamese Hate Speech Detection Dataset)

---

## Authors

Faculty of Information Technology, **Ton Duc Thang University (TDTU)**.

---

## License

Code is distributed under the [Apache License 2.0](LICENSE). Datasets and pre-trained checkpoints are subject to their respective upstream licenses.