# Vietnamese Hate Speech Detection

A PhoBERT-based classifier for Vietnamese hate speech (`CLEAN` / `OFFENSIVE` /
`HATE`), built to compare five data-augmentation strategies side by side:
back-translation, EDA, LLM-generated synthetic data, their combination, and a
no-augmentation baseline. The repo includes the research notebooks, a FastAPI
serving layer that loads all five checkpoints at once, and a React demo app
for trying them interactively.

## Quick start

Requirements: Python 3.10+, Node.js, and Java 8+ (for VnCoreNLP-related
tooling used during preprocessing — see [Installation](#installation) if
`java -version` doesn't resolve).

**1. Backend** — loads all five PhoBERT checkpoints from Hugging Face Hub on
startup, so the first run takes a few minutes and noticeably more RAM than a
single-model server.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

$env:MODEL_EXPERIMENT = "combined"   # default experiment when a request omits "model"
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

**2. Frontend** — in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The status badge should read **API sẵn sàng**;
if it doesn't, check `http://127.0.0.1:8000/health` first — the error there
is usually more specific than what reaches the browser.

## Project structure

```text
Vietnamese-HSD-Augmentation/
├── api/                         # FastAPI application and request schemas
├── configs/config.yaml          # Shared project configuration
├── data/
│   ├── raw/                     # Source datasets (not committed)
│   ├── processed/               # train.csv, dev.csv, test.csv (not committed)
│   └── augmented/               # Augmentation outputs (not committed)
├── frontend/                    # React demo app
├── models/                      # Local PhoBERT checkpoints (not committed)
├── notebooks/
│   ├── 1_data_exploration/
│   ├── 2_preprocessing/
│   ├── 3_augmentation/
│   ├── 4_model_training/
│   └── 5_evaluation/
├── results/
│   ├── metrics/                 # metrics_<experiment>.json
│   ├── figures/
│   └── error_analysis/
├── scripts/summarize_ablation.py
├── src/
│   ├── services/inference.py    # Shared PhoBERT inference service
│   └── utils/                   # Configuration, preprocessing and evaluation
├── tests/
└── vncorenlp/                   # Local Java runtime assets (not committed)
```

## The five experiments

| Key        | What it is                                      |
| ---------- | ------------------------------------------------ |
| `baseline` | PhoBERT fine-tuned on the unaugmented training set |
| `bt`       | + back-translation augmentation                  |
| `eda`      | + Easy Data Augmentation                          |
| `llm`      | + LLM-generated synthetic samples                 |
| `combined` | All three augmentation sources merged             |

These keys are used consistently across the repo:

- Metric files: `results/metrics/metrics_<experiment>.json`
- Checkpoint directories: `models/<experiment>_phobert/`
- The `model` field accepted by `POST /predict` (see [API reference](#api-reference))

`scripts/summarize_ablation.py` reads whichever metric files are present and
writes a comparison table and chart under `results/`.

## API reference

All five checkpoints are loaded once at startup and kept in memory, so
switching experiments between requests costs no extra load time.

**`POST /predict`**

```json
{ "text": "câu cần phân loại", "model": "combined" }
```

`model` is optional — omit it to use whatever `MODEL_EXPERIMENT` was set to
at startup. Response:

```json
{
  "text": "câu cần phân loại",
  "text_cleaned": "câu cần phân_loại",
  "label": "CLEAN",
  "confidence": 0.94,
  "probabilities": { "CLEAN": 0.94, "OFFENSIVE": 0.04, "HATE": 0.02 },
  "latency_ms": 42.1,
  "model_used": "combined",
  "token_importance": [{ "token": "câu", "score": 0.02 }, { "token": "cần", "score": 0.05 }]
}
```

`text_cleaned` is the Underthesea-segmented input (compound words joined by
`_`) actually fed to the model. `token_importance` is an attention-based
saliency score per word — the last transformer layer's attention from the
CLS token, averaged across heads. Treat it as a visualization aid, not a
rigorous attribution method (it isn't LIME or Integrated Gradients).

**`POST /predict/batch`** — same shape, `{"texts": [...], "model": "..."}` in,
`{"results": [...]}` out.

**`GET /health`** — `{"status", "model", "device", "available_models"}`.

**`GET /metadata`** — labels, max sequence length, preprocessing description,
and `available_models` (the five experiment keys currently loaded).

CORS allows `http://localhost:5173` and `http://127.0.0.1:5173` by default.
Set `CORS_ORIGINS` (comma-separated) before deploying anywhere else.

## React frontend

Lives in `frontend/`, talks to the API via `VITE_API_BASE_URL` (defaults to
`http://127.0.0.1:8000`, so local development needs no configuration). Lets
you switch between the five experiments per request and see which words the
model attended to most.

For a different backend URL, copy `.env.example` to `.env.local`, set
`VITE_API_BASE_URL`, then restart `npm run dev`.

## Data contract

Processed CSVs use `text` (Vietnamese input) and `label` (`CLEAN` /
`OFFENSIVE` / `HATE`). Default split locations, set in `configs/config.yaml`:
`data/processed/{train,dev,test}.csv`.

Raw data, processed data, augmentation outputs, checkpoints, VnCoreNLP assets
and generated figures are local artifacts and are not committed to Git.

## Installation

VnCoreNLP-related preprocessing tooling requires Java 8+ plus the VnCoreNLP
`.jar` and its `models/` directory locally — also not committed to Git.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

python -m pip install -r requirements.txt              # FastAPI service only
python -m pip install -r requirements-research.txt      # + notebooks, training, evaluation
python -m pip install -r requirements-dev.txt           # + tests
```

Set up the local VnCoreNLP assets once, after installing dependencies. API
startup never downloads assets automatically.

```powershell
python scripts/setup_vncorenlp.py
java -version
```

If `java -version` isn't found, install Java 8+ and add its `bin` directory
to `PATH`. The configured asset location is `preprocessing.vncorenlp_dir` in
`configs/config.yaml`.

The app derives `JAVA_HOME` from whatever Java is on `PATH`. If Java
discovery is unreliable on a given Windows machine, set it explicitly before
starting the API:

```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-21"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

Dependencies are deliberately unpinned while the report is still being
finalized, to avoid forcing resolver conflicts mid-writeup. Once the demo
environment is verified, snapshot it separately —
`python -m pip freeze > requirements-lock.txt` — rather than replacing the
source requirement files with that output.

## Evaluation

Primary metrics: Macro F1, HATE F1, per-class F1, and the confusion matrix.
Reusable implementation lives in `src/`; notebooks under `notebooks/` are kept
for exploration, data preparation, training, and analysis — not for anything
imported at runtime.