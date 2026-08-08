"""Feature engineering for the LeakDB Hanoi Random Forest model.

Ported verbatim from leakdb_hanoi_pipeline_3.py so that served predictions use
exactly the same features the model was trained on. Any change here must be
mirrored in the training pipeline (or vice versa) to avoid train/serve skew.
"""

import numpy as np
from scipy.stats import skew as scipy_skew

WINDOW_SIZE = 48
WINDOW_STRIDE = 6
TIMESTEP_MIN = 30
N_SENSORS = 66
N_STATS = 7
N_FEATURES = N_SENSORS * N_STATS + 4

_T_AXIS = np.arange(WINDOW_SIZE, dtype=np.float64)
_T_MEAN = _T_AXIS.mean()
_T_VAR = np.mean((_T_AXIS - _T_MEAN) ** 2)

FEATURE_NAMES = [
    f"S{sensor_id}_{stat}"
    for sensor_id in range(N_SENSORS)
    for stat in ["mean", "std", "min", "max", "slope", "delta", "skew"]
] + ["hour_sin", "hour_cos", "day_sin", "day_cos"]


def extract_features(X):
    """Vectorized sliding-window feature extraction.

    Args:
        X: (T, n_sensors) array of raw sensor readings (pressures + flows).

    Returns:
        (n_windows, N_FEATURES) float32 array, or None if T < WINDOW_SIZE.
        window_ends: the row index (exclusive) each window ends at.
    """
    T = X.shape[0]
    if T < WINDOW_SIZE:
        return None, None

    ends = np.arange(WINDOW_SIZE, T + 1, WINDOW_STRIDE)
    n_windows = len(ends)
    n_sensors = X.shape[1]
    n_feats = n_sensors * N_STATS + 4

    all_feats = np.empty((n_windows, n_feats), dtype=np.float32)

    for i, end in enumerate(ends):
        w = X[end - WINDOW_SIZE:end]

        means = w.mean(axis=0)
        stds = w.std(axis=0)
        mins = w.min(axis=0)
        maxs = w.max(axis=0)
        deltas = w[-1] - w[0]
        slopes = np.dot(_T_AXIS - _T_MEAN, w - means) / _T_VAR
        safe = stds > 1e-10
        sk = np.zeros(n_sensors, dtype=np.float32)
        if safe.any():
            sk[safe] = scipy_skew(w[:, safe], axis=0)

        all_feats[i, :n_sensors * N_STATS] = np.column_stack(
            [means, stds, mins, maxs, slopes, deltas, sk]
        ).ravel()

        minutes = (end - 1) * TIMESTEP_MIN
        hour = (minutes / 60) % 24
        day = (minutes / 1440) % 7
        all_feats[i, -4] = np.sin(2 * np.pi * hour / 24)
        all_feats[i, -3] = np.cos(2 * np.pi * hour / 24)
        all_feats[i, -2] = np.sin(2 * np.pi * day / 7)
        all_feats[i, -1] = np.cos(2 * np.pi * day / 7)

    all_feats = np.nan_to_num(all_feats, nan=0.0, posinf=0.0, neginf=0.0)
    return all_feats, ends
