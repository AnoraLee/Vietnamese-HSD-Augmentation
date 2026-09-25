# Vietnamese Hate Speech Detection

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Model](https://img.shields.io/badge/model-PhoBERT-orange)
![Framework](https://img.shields.io/badge/backend-FastAPI-009688)
![Frontend](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61DAFB)

A PhoBERT-based classifier for Vietnamese hate speech (`CLEAN` / `OFFENSIVE` /
`HATE`), built around one central question: **which data-augmentation
strategy actually helps a Vietnamese hate-speech classifier generalize?**
The repo holds the research notebooks that produced the answer, a FastAPI
serving layer that keeps all five trained checkpoints in memory for instant
side-by-side comparison, and a React app for trying them interactively —
including two independent ways to see *why* the model made a call.

## Contents

- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [The five experiments](#the-five-experiments)
- [API reference](#api-reference)
- [React frontend](#react-frontend)
- [Data contract](#data-contract)
- [Installation](#installation)
- [Troubleshooting](#troubleshooting)
- [Evaluation](#evaluation)
- [Results](#results)
- [Known limitations](#known-limitations)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Quick start

Requirements: Python 3.10+ and Node.js. Preprocessing runs on
[underthesea](https://github.com/undertheeye/underthesea) — pure Python, no
JVM or external runtime.

**1. Backend** — loads all five PhoBERT checkpoints from Hugging Face Hub on
startup, so the first run takes a few minutes and noticeably more RAM than a
single-model server. Every request can then pick any of the five instantly,
with no restart.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

$env:MODEL_EXPERIMENT = "combined"   # default when a request omits "model"
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
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
├── api/                         # Request/response schemas
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
│   ├── services/inference.py    # Shared PhoBERT inference + explanation service
│   └── utils/                   # Configuration, preprocessing and evaluation
├── main.py                      # FastAPI application
└── tests/
```

## The five experiments

| Key        | What it is                                          |
| ---------- | ---------------------------------------------------- |
| `baseline` | PhoBERT fine-tuned on the unaugmented training set   |
| `bt`       | + back-translation augmentation                      |
| `eda`      | + Easy Data Augmentation                              |
| `llm`      | + LLM-generated synthetic samples                     |
| `combined` | All three augmentation sources merged                 |

These keys are used consistently everywhere in the repo:

- Metric files: `results/metrics/metrics_<experiment>.json`
- Checkpoint directories: `models/<experiment>_phobert/`
- HF Hub source repos: `src/utils/constants.py`
- The `model` field accepted by every prediction/explanation endpoint below

`scripts/summarize_ablation.py` reads whichever metric files are present and
writes a consolidated comparison table and chart under `results/`.

## API reference

All five checkpoints are loaded once at startup and kept in memory, so
switching experiments between requests costs no extra load time.

### `POST /predict`

```json
{ "text": "câu cần phân loại", "model": "combined" }
```

`model` is optional — omit it to use whatever `MODEL_EXPERIMENT` was set to
at startup.

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

`text_cleaned` is the underthesea-segmented input (compound words joined by
`_`) actually fed to the model. `token_importance` is an **attention-based**
saliency score per word — fast (one forward pass), always included, but a
heuristic rather than a validated attribution method. See `/explain` for a
slower, theoretically grounded alternative.

### `POST /explain`

```json
{ "text": "câu cần phân loại", "model": "combined", "max_evals": 100 }
```

Runs a **SHAP** (Shapley value) explanation instead of attention — hundreds
of forward passes on masked variants of the input, so this is meant to be
triggered explicitly (a "explain in detail" action in the UI), never called
on every prediction. `max_evals` is optional; the server picks a
device-appropriate default (`200` on GPU, `80` on CPU) when omitted.

```json
{
  "label": "OFFENSIVE",
  "method": "shap",
  "latency_ms": 6980.4,
  "token_scores": [
    { "token": "hay", "score": -0.35 },
    { "token": "lồn", "score": 0.38 }
  ]
}
```

Positive scores push toward the predicted label; negative scores push away
from it. Unlike `token_importance`, these values satisfy SHAP's local
accuracy guarantee — they sum to the gap between the model's output and its
baseline expectation for this input.

### `POST /predict/batch`

Same shape as `/predict`, batched: `{"texts": [...], "model": "..."}` in,
`{"results": [...]}` out.

### `GET /health`

```json
{ "status": "ok", "model": "combined", "device": "cpu", "available_models": ["baseline", "bt", "eda", "llm", "combined"] }
```

### `GET /metadata`

Labels, max sequence length, preprocessing description, and
`available_models`.

CORS allows `http://localhost:5173` and `http://127.0.0.1:5173` by default.
Set `CORS_ORIGINS` (comma-separated) before deploying anywhere else.

## React frontend

Lives in `frontend/`, talks to the API via `VITE_API_BASE_URL` (defaults to
`http://127.0.0.1:8000`, so local development needs no configuration). Lets
you switch between the five experiments per request, see which words the
model attended to (instant), and request a SHAP explanation on demand
(slower, shown behind an explicit action so the wait is never a surprise).

For a different backend URL, copy `.env.example` to `.env.local`, set
`VITE_API_BASE_URL`, then restart `npm run dev`.

## Data contract

Processed CSVs use `text` (Vietnamese input) and `label` (`CLEAN` /
`OFFENSIVE` / `HATE`). Default split locations, set in `configs/config.yaml`:
`data/processed/{train,dev,test}.csv`.

Raw data, processed data, augmentation outputs, checkpoints, and generated
figures are local artifacts and are not committed to Git.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

python -m pip install -r requirements.txt              # FastAPI service only
python -m pip install -r requirements-research.txt      # + notebooks, training, evaluation
python -m pip install -r requirements-dev.txt           # + tests
```

No JVM setup or external runtime is required — `underthesea` installs like
any other Python package and needs no post-install asset download.

Dependencies are deliberately unpinned while the report is still being
finalized, to avoid forcing resolver conflicts mid-writeup. Once the demo
environment is verified, snapshot it separately —
`python -m pip freeze > requirements-lock.txt` — rather than replacing the
source requirement files with that output.

## Troubleshooting

Two failure modes worth knowing before you hit them yourself:

**`output_attentions=True` raises `IndexError: tuple index out of range`.**
Recent `transformers` versions default to the `sdpa` attention
implementation, which silently returns no attention weights at all — the
tuple `outputs.attentions` comes back empty. `src/services/inference.py`
loads every model with `attn_implementation="eager"` specifically to avoid
this; if you see this error, something is loading a model without that
argument.

**`POST /explain` never returns.** SHAP's `"partition"` algorithm builds a
hierarchical clustering over tokens before it can run, and with a custom
whitespace-based text masker that step can fail to converge — the explainer
keeps requesting more evaluations indefinitely. `explain_with_shap` uses
`algorithm="permutation"` instead, which is strictly bounded by `max_evals`
and always terminates. If you're extending this method, keep it on
`"permutation"` unless you've verified `"partition"` terminates for your
masker.

## Evaluation

Primary metrics: Macro F1, HATE F1, per-class F1, and the confusion matrix.
Reusable implementation lives in `src/`; notebooks under `notebooks/` are kept
for exploration, data preparation, training, and analysis — not for anything
imported at runtime.

## Results

Per-experiment metrics live in `results/metrics/metrics_<experiment>.json`;
run `python scripts/summarize_ablation.py` for a consolidated comparison
table and chart across whichever experiments have been trained so far. Full
discussion of the numbers belongs in the thesis report, not duplicated here —
this repo is the source of truth for how they were produced.

## Known limitations

- **Preprocessing mismatch risk:** training data was originally segmented
  with a different tokenizer (VnCoreNLP) before the project standardized on
  `underthesea`. If any checkpoint under `models/` was fine-tuned on the
  older segmentation, re-validate its dev/test metrics against
  underthesea-segmented input before trusting them at face value.
- **Back-translation augmentation** (`bt`) has a measured semantic-drift rate
  — a meaningful share of candidate sentences were dropped for drifting from
  the original meaning or softening the toxic content during round-trip
  translation. See the filtering pipeline in
  `notebooks/3_augmentation/bt_augmentation.ipynb` for the exact filters and
  thresholds applied.
- **`token_importance`** (the always-on, attention-based signal) is a
  heuristic, not a validated attribution method — useful for a quick visual
  read, not for claims about causal feature importance. `/explain` (SHAP)
  exists specifically to give a more rigorous alternative when it matters.
- **Word-to-subword alignment** for `token_importance` is approximate: the
  tokenizer runs with `use_fast=False`, so there's no exact offset mapping,
  and word boundaries are recovered by re-tokenizing each word in isolation.
- **SHAP latency scales with `max_evals`** and runs on CPU by default unless
  CUDA is available — expect single-digit seconds to a minute per call, not
  suitable for high-throughput use without further optimization.

## Acknowledgments

Built on [PhoBERT](https://github.com/VinAIResearch/PhoBERT) (VinAI
Research), [underthesea](https://github.com/undertheeye/underthesea) for
Vietnamese word segmentation, [Helsinki-NLP/opus-mt](https://huggingface.co/Helsinki-NLP)
for back-translation, and [SHAP](https://github.com/shap/shap) for
explanation. Trained checkpoints are hosted across the team's Hugging Face
accounts — see `src/utils/constants.py` for the exact repo IDs behind each
experiment key.

## License

Not yet decided for this repository. Treat all checkpoints and data as
private/academic-use only until a license file is added.