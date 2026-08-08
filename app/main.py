import io
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app import model_loader
from app.features import N_SENSORS, WINDOW_SIZE, extract_features

app = FastAPI(title="LeakDB Hanoi Leak Detector")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.on_event("startup")
def _load_model_on_startup():
    try:
        model_loader.load_model()
    except RuntimeError as exc:
        # Defer the error to request time so the static UI still loads and
        # /health can report the problem instead of crashing the process.
        print(f"[startup] model not loaded yet: {exc}")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model_loader.is_loaded()}


@app.get("/sample-csv", response_class=PlainTextResponse)
def sample_csv():
    """Synthetic demo data (NOT real LeakDB readings) so the UI has something
    to try immediately: a plausible-looking 60-row x 66-column window with a
    simulated pressure drop partway through, to exercise both classes."""
    rng = np.random.default_rng(42)
    n_rows = 60
    baseline = rng.normal(loc=50.0, scale=2.0, size=(n_rows, N_SENSORS))
    leak_start = 30
    baseline[leak_start:, : N_SENSORS // 2] -= np.linspace(
        0, 8, n_rows - leak_start
    )[:, None]

    columns = [f"sensor_{i}" for i in range(N_SENSORS)]
    df = pd.DataFrame(baseline, columns=columns)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        model = model_loader.load_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if df.shape[1] != N_SENSORS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Expected {N_SENSORS} sensor columns, got {df.shape[1]}. "
                "Columns must be raw pressure/flow readings, one per sensor, "
                "in the same order used during training."
            ),
        )

    if df.shape[0] < WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Need at least {WINDOW_SIZE} rows (one {WINDOW_SIZE}-step "
                f"window = 24h at 30-min intervals), got {df.shape[0]}."
            ),
        )

    X = df.to_numpy(dtype=np.float64)
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise HTTPException(status_code=400, detail="CSV contains NaN/Inf values.")

    feats, ends = extract_features(X)
    probs = model.predict_proba(feats)[:, 1]
    preds = (probs >= 0.5).astype(int)

    windows = [
        {
            "window_index": i,
            "start_row": int(end - WINDOW_SIZE),
            "end_row": int(end),
            "leak_probability": round(float(p), 4),
            "prediction": "leak" if pred == 1 else "no_leak",
        }
        for i, (end, p, pred) in enumerate(zip(ends, probs, preds))
    ]

    return {
        "n_windows": len(windows),
        "any_leak_detected": bool(preds.any()),
        "max_leak_probability": round(float(probs.max()), 4),
        "windows": windows,
    }


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
