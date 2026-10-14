# HSD Studio — Vietnamese Hate Speech Detection

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#quick-start)
[![Models](https://img.shields.io/badge/models-PhoBERT%20%7C%20ViSoBERT-orange)](#experiments--results)
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

- [Quick start](#quick-start)
- [Docker](#docker)
- [Deployment](#deployment)
- [Project structure](#project-structure)
- [Experiments & Results](#experiments--results)
- [Research pipeline](#research-pipeline)
- [API reference](#api-reference)
- [Frontend](#frontend)
- [Data](#data)
- [Testing](#testing)
- [Authors & License](#authors--license)

---

## Quick start

**Requirements:** Python 3.10+ and Node.js (v18+).

### 1. Backend

Set up the Python virtual environment, install requirements, and run the FastAPI server:

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Set default model (or specify LOADED_EXPERIMENTS="phobert_baseline,phobert_combined,visobert_baseline")
$env:MODEL_EXPERIMENT = "phobert_combined"

# Start server
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

### 2. Frontend

In a separate terminal, install dependencies and launch the Vite development server:

```bash
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The status pill will turn green once connected to `http://127.0.0.1:8000/health`.

---

## Docker

Run the containerized backend using Docker Compose:

```bash
docker compose up --build
```

- **Service:** `backend` (container: `hsd-backend`)
- **Port:** `8000:8000`
- **Volume:** `hf_cache` mounted at `/app/.cache/huggingface` to persist downloaded model weights across container restarts.
- **Healthcheck:** Evaluates `http://localhost:8000/health` automatically every 30s.

---

## Deployment

The application is deployed across two decoupled hosting services:

- **Backend (Hugging Face Spaces):** Containerized FastAPI app running at `https://huggingface.co/spaces/BonTori/vietnamese_tdtu_hsd_detectToxic` (`CORS_ORIGINS` configured to permit the Vercel domain).
- **Frontend (Vercel):** React SPA hosted at `https://vietnamese-hsd-augmentation-xx46.vercel.app/` with `VITE_API_BASE_URL` pointing to the Hugging Face Space URL.

---

## Project structure

```
Vietnamese-HSD-Augmentation/
├── api/                         # FastAPI backend & inference endpoints
│   ├── main.py
│   ├── schemas.py
│   └── evaluation_routes.py
├── configs/                     # Experiment & model configuration
│   └── config.yaml
├── data/                        # Datasets (not committed)
│   ├── processed/               # tdtu_train.csv, tdtu_dev.csv, tdtu_test.csv
│   ├── augmented/               # aug_combined.csv, aug_bt.csv, ...
│   └── raw/                     # Raw vihsd, voz, youtube dumps
├── frontend/                    # React + Vite web studio
│   ├── public/
│   └── src/
├── models/                      # Fine-tuned model checkpoints
├── notebooks/                   # Jupyter research & exploration notebooks
│   ├── 1_data_exploration/
│   ├── 2_preprocessing/
│   ├── 3_augmentation/
│   ├── 4_training/
│   └── 5_evaluation/
├── results/                     # Experiment metrics, figures, and error analysis
│   ├── metrics/                 # metrics_phobert_combined.json, ...
│   ├── figures/                 # Confusion matrices & comparison plots
│   ├── error_analysis/          # Misclassified samples & confusion summaries
│   └── experiments_summary.csv
├── scripts/                     # Reusable CLI utility scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── run_ablation.py
│   └── ...
├── src/                         # Shared core modules
│   ├── models/                  # Classifier architectures
│   ├── services/                # Inference & SHAP service
│   └── utils/                   # Preprocessing, config, metrics
├── tests/                       # Automated test suite
│   ├── test_api.py
│   ├── test_schemas.py
│   └── ...
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── LICENSE
```

---

## Experiments & Results

All seven experiments are evaluated on the exact same frozen test set (**6,576 samples**; the development set has **2,650 samples**).

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>Experiment Key</th>
      <th>Data Augmentation Strategy</th>
      <th>Accuracy</th>
      <th>Macro F1</th>
      <th>Weighted F1</th>
      <th>HATE F1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" align="center"><strong>PhoBERT-base</strong></td>
      <td><code>phobert_baseline</code></td>
      <td>Original unaugmented training set</td>
      <td>84.4%</td>
      <td>60.2%</td>
      <td>84.7%</td>
      <td>53.8%</td>
    </tr>
    <tr>
      <td><code>phobert_bt</code></td>
      <td>+ Back-translation (vi → en → vi)</td>
      <td>83.0%</td>
      <td>44.5%</td>
      <td>79.8%</td>
      <td>42.7%</td>
    </tr>
    <tr>
      <td><code>phobert_eda</code></td>
      <td>+ Easy Data Augmentation (swap, insert, delete)</td>
      <td>78.4%</td>
      <td>58.3%</td>
      <td>81.3%</td>
      <td>56.5%</td>
    </tr>
    <tr>
      <td><code>phobert_llm</code></td>
      <td>+ Synthetic samples via LLMs (Gemma, Qwen)</td>
      <td>83.5%</td>
      <td>60.9%</td>
      <td>84.5%</td>
      <td>56.1%</td>
    </tr>
    <tr>
      <td><code>phobert_combined</code></td>
      <td>Merged BT + EDA + LLM augmentations (deduplicated)</td>
      <td>78.0%</td>
      <td>57.3%</td>
      <td>80.7%</td>
      <td>58.0%</td>
    </tr>
    <tr>
      <td rowspan="2" align="center"><strong>ViSoBERT</strong></td>
      <td><code>visobert_baseline</code></td>
      <td>Original unaugmented training set</td>
      <td><strong>84.5%</strong></td>
      <td><strong>61.5%</strong></td>
      <td><strong>85.1%</strong></td>
      <td>55.5%</td>
    </tr>
    <tr>
      <td><code>visobert_combined</code></td>
      <td>Combined augmented training set</td>
      <td>77.9%</td>
      <td>57.0%</td>
      <td>80.5%</td>
      <td><strong>58.7%</strong></td>
    </tr>
  </tbody>
</table>

### Key Findings
1. **The Class Imbalance Trade-off:** The corpus is heavily imbalanced (`CLEAN` accounts for the vast majority). Standard Accuracy and Weighted F1 predominantly reflect performance on clean sentences. For content moderation, **HATE F1** is the critical metric.
2. **Value of Data Augmentation:** Combining multiple augmentation sources (`phobert_combined` and `visobert_combined`) yields the highest HATE F1 scores (**58.0%** and **58.7%**, up by +4.2% and +4.9% over PhoBERT baseline), successfully trading off a small percentage of accuracy on clean samples to capture significantly more genuine toxicity.
3. **Architecture Comparison:** ViSoBERT demonstrates superior representations on informal Vietnamese social media abbreviations and slang, with `visobert_baseline` achieving the best overall Macro F1 (61.5%) and Weighted F1 (85.1%).
4. **Back-translation Semantic Drift:** Round-trip translation (vi → en → vi) frequently softened aggressive terms and colloquial slurs, causing `phobert_bt` to struggle on minority classes.

Consolidated metrics are stored in `results/experiments_summary.csv` and per-experiment metrics are in `results/metrics/metrics_<experiment>.json`. Run `python scripts/summarize_ablation.py` to regenerate the comparison table and chart under `results/figures/`.

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

Health check endpoint returning active server state, default model, compute device, and all loaded checkpoints:

```json
{
  "status": "ok",
  "model": "phobert_combined",
  "device": "cpu",
  "available_models": ["phobert_baseline", "phobert_bt", "phobert_eda", "phobert_llm", "phobert_combined", "visobert_baseline", "visobert_combined"]
}
```

### `GET /metadata`

Exposes model parameters, preprocessing pipeline description, label set, and available checkpoints:

```json
{
  "model": "phobert_combined",
  "device": "cpu",
  "labels": ["CLEAN", "OFFENSIVE", "HATE"],
  "max_length": 128,
  "preprocessing": "Underthesea word segmentation + teencode normalization",
  "available_models": ["phobert_baseline", "phobert_bt", "phobert_eda", "phobert_llm", "phobert_combined", "visobert_baseline", "visobert_combined"]
}
```

### `GET /evaluation/metrics`

Returns benchmark metrics for all evaluated experiments (consumed by the web frontend's evaluation dashboard):

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

The benchmark dataset consolidates the ViHSD corpus with domain-specific comments collected from VOZ forums and YouTube.

- **Frozen Splits:** `tdtu_train.csv` (86,719 samples), `tdtu_dev.csv` (2,650 samples), `tdtu_test.csv` (6,576 samples) located in `data/processed/`.
- **Augmentation Data:** Augmented subsets generated via Back-translation, EDA, and LLM synthesis are stored under `data/augmented/`.

---

## Testing

Run the automated test suite with pytest:

```bash
pytest tests/ -v
```

---

## Authors & Acknowledgments

Faculty of Information Technology, **Ton Duc Thang University (TDTU)**.

This project is built upon open-source foundations: [PhoBERT](https://github.com/VinAIResearch/PhoBERT), [ViSoBERT](https://github.com/VinAIResearch/ViSoBERT), [Underthesea](https://github.com/undertheseanlp/underthesea), [SHAP](https://github.com/shap-project/shap), and the ViHSD corpus.

---

## License

Code is distributed under the [Apache License 2.0](LICENSE).