"""Loads the trained RF model: local file first, else Hugging Face Hub.

Set HF_REPO_ID (e.g. "your-username/leakdb-hanoi-rf") once the model has been
uploaded via scripts/upload_model_to_hf.py. Until then, the app falls back to
a local rf_model_compressed.joblib / rf_model.joblib in the repo root.
"""

import os

import joblib

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_CANDIDATES = ["rf_model_compressed.joblib", "rf_model.joblib"]

HF_REPO_ID = os.environ.get("HF_REPO_ID", "rockandrock/leakdb-hanoi-rf")
HF_FILENAME = os.environ.get("HF_FILENAME", "rf_model_compressed.joblib")

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model

    for name in LOCAL_CANDIDATES:
        path = os.path.join(REPO_ROOT, name)
        if os.path.exists(path):
            _model = joblib.load(path)
            return _model

    if not HF_REPO_ID:
        raise RuntimeError(
            "No local model file found and HF_REPO_ID is not set. "
            "Either place rf_model_compressed.joblib in the repo root, or "
            "upload it to Hugging Face Hub (see scripts/upload_model_to_hf.py) "
            "and set the HF_REPO_ID environment variable."
        )

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    _model = joblib.load(path)
    return _model


def is_loaded():
    return _model is not None
