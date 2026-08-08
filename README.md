# LeakDB Hanoi — Leak Detector

A small FastAPI website around the Random Forest model trained in
[`leakdb_hanoi_pipeline_3.py`](leakdb_hanoi_pipeline_3.py) for leak detection
on the LeakDB Hanoi benchmark network.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000, download the synthetic sample CSV, and upload it
to see predictions. To use it with real data, upload a CSV with 66 sensor
columns (pressures + flows, same order as training) and at least 48 rows
(24h at 30-min intervals) — one prediction is returned per 48-row sliding
window (stride 6).

## Getting the model

`rf_model.joblib` and `rf_model_compressed.joblib` are **not** committed to
this repo (see `.gitignore`) — GitHub rejects files over 100MB, and both
files exceed that (976MB / 384MB).

- **Right after training**: the app finds `rf_model_compressed.joblib` (or
  `rf_model.joblib`) directly in the repo root and loads it — no extra setup.
- **On a fresh checkout / anywhere else**: host the compressed model on
  [Hugging Face Hub](https://huggingface.co) (free, built for large ML
  files) and let the app download it at startup:

  ```bash
  pip install huggingface_hub
  huggingface-cli login
  python scripts/upload_model_to_hf.py --repo-id <your-username>/leakdb-hanoi-rf
  export HF_REPO_ID=<your-username>/leakdb-hanoi-rf
  uvicorn app.main:app --reload --port 8000
  ```

  The download is cached under `~/.cache/huggingface`, so it only happens
  once per machine. It's already uploaded to a **private** repo at
  `rockandrock/leakdb-hanoi-rf` (set as the default `HF_REPO_ID` in
  `app/model_loader.py`), so any machine logged in as `rockandrock` via
  `huggingface-cli login` can pull it without setting `HF_REPO_ID` at all —
  though a local `.joblib` file, if present, is still preferred over
  downloading.

## Note on scikit-learn version

The model was trained with scikit-learn 1.6.1. Loading it with a newer
scikit-learn prints an `InconsistentVersionWarning` but has worked fine in
testing. If you ever see different predictions than expected, try
`pip install scikit-learn==1.6.1` (may require an older Python — 1.6.1 has
no wheels for very new Python versions) to match exactly.

## Layout

- `app/features.py` — sliding-window feature extraction, ported verbatim
  from the training pipeline (must stay in sync to avoid train/serve skew).
- `app/model_loader.py` — loads the model from a local file or HF Hub.
- `app/main.py` — FastAPI app: `/predict`, `/health`, `/sample-csv`.
- `app/static/index.html` — upload UI.
- `scripts/upload_model_to_hf.py` — one-time model upload helper.
