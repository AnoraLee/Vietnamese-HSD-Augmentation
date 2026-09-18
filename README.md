# Vietnamese Hate Speech Detection

Vietnamese hate-speech detection with PhoBERT. The repository contains the
research artifacts used to compare data-augmentation experiments and a local
FastAPI/Streamlit serving layer for model demonstration.

## Project structure

```text
Vietnamese-HSD-Augmentation/
├── api/                         # FastAPI application and request schemas
├── configs/config.yaml          # Shared project configuration
├── data/
│   ├── raw/                     # Source datasets (not committed)
│   ├── processed/               # train.csv, dev.csv, test.csv (not committed)
│   └── augmented/               # Augmentation outputs (not committed)
├── frontend/                    # demonstration app
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
│   ├── models/classifier.py     # Local PhoBERT inference wrapper
│   └── utils/                   # Configuration, preprocessing and evaluation
├── tests/
└── vncorenlp/                   # Local Java runtime assets (not committed)
```

## Experiment contract

The supported experiments are `baseline`, `bt`, `eda`, `llm`, and `combined`.

Each metric file follows this convention:

```text
results/metrics/metrics_<experiment>.json
```

For example, `results/metrics/metrics_combined.json`. The ablation summary
script reads all available metric files and writes the comparison table and
chart under `results/`.

Model checkpoints use this convention:

```text
models/<experiment>_phobert/
```

For example, the combined model is expected at `models/combined_phobert/`.

## Data contract

Processed CSV files use the columns below:

- `text`: Vietnamese input text
- `label`: one of `CLEAN`, `OFFENSIVE`, or `HATE`

The default split locations are configured in `configs/config.yaml`:
`data/processed/train.csv`, `data/processed/dev.csv`, and
`data/processed/test.csv`.

Raw datasets, processed data, augmentation outputs, checkpoints, VnCoreNLP
assets and generated figures are local artifacts. They are deliberately not
committed to Git.

## Installation

Use Python 3.10 or newer. VnCoreNLP also requires Java 8 or newer and the
VnCoreNLP `.jar` file plus its `models/` directory on the local machine. The
runtime assets are intentionally not stored in Git.

Create and activate an isolated environment, then install the dependency set
that matches the task:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# FastAPI inference service only
python -m pip install -r requirements.txt

# Notebooks, training, evaluation and temporary Streamlit demo
python -m pip install -r requirements-research.txt

# Development and tests; includes the research dependencies
python -m pip install -r requirements-dev.txt
```

Prepare the local VnCoreNLP assets once after installing the runtime
dependencies. This is an explicit setup action; API startup never downloads
assets automatically.

```powershell
python scripts/setup_vncorenlp.py
java -version
```

If `java -version` is not found, install Java 8+ and add its `bin` directory to
`PATH` before running the API. The configured asset location is
`preprocessing.vncorenlp_dir` in `configs/config.yaml`.

The application derives `JAVA_HOME` from the Java executable on `PATH` for
pyjnius. If Java discovery is unusual on a Windows machine, set it explicitly
for the current PowerShell session before starting FastAPI:

```powershell
$env:JAVA_HOME = "C:\Program Files\Java\jdk-21"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

Dependency versions are deliberately not pinned during the report-delivery
period to avoid forcing an unnecessary resolver conflict. Once the demo
environment is verified, record its resolved packages separately with
`python -m pip freeze > requirements-lock.txt`; do not replace the unpinned
source requirement files with that snapshot.

## Current entry points

Summarize available experiment metrics:

```bash
python scripts/summarize_ablation.py
```

Run the API after placing a supported checkpoint under `models/`:

```bash
uvicorn api.main:app --reload
```

`MODEL_EXPERIMENT` selects the checkpoint and defaults to `combined`.

```bash
$env:MODEL_EXPERIMENT = "combined"  # PowerShell
uvicorn api.main:app --reload
```

`POST /predict` receives `{"text": "..."}` and returns the original text,
the VnCoreNLP-preprocessed text, predicted label, confidence, probabilities
for all three labels, inference latency, and the selected experiment. Both the
API and Streamlit use the same inference service and preprocessing pipeline.

`GET /health` reports whether the model is ready; `GET /metadata` provides the
selected model, labels, max length and preprocessing description for clients.
The local React development server is allowed by default at
`http://localhost:5173`. Set `CORS_ORIGINS` to a comma-separated allowlist
before deploying elsewhere.

## React frontend

The React/Vite client lives in `frontend/` and uses `VITE_API_BASE_URL` to
find FastAPI. It defaults to `http://127.0.0.1:8000`, so no configuration is
needed for local development.

```powershell
cd frontend
npm install
npm run dev
```

Open the local URL printed by Vite, normally `http://127.0.0.1:5173`. For a
different backend URL, copy `.env.example` to `.env.local`, update
`VITE_API_BASE_URL`, then restart the Vite server.

## Run the local application

Start the backend first from the project root. `combined` is the default model;
change `MODEL_EXPERIMENT` only when the matching checkpoint exists under
`models/`.

```powershell
.\.venv\Scripts\Activate.ps1
$env:MODEL_EXPERIMENT = "combined"
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

In a second PowerShell terminal, start the React client:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173`. The status badge should show **API sẵn sàng**.
If it does not, first open `http://127.0.0.1:8000/health` to read the backend
error before troubleshooting the browser client.

The temporary Streamlit demo can be run with:

```bash
streamlit run app/app.py
```

## Evaluation priorities

Primary research metrics are Macro F1, HATE F1, per-class F1 and the confusion
matrix. Reusable implementation belongs in `src/`; notebooks are retained for
exploration, data preparation, training and analysis.
