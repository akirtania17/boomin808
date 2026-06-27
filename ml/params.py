"""Parameter normalization helpers (numpy only, no torch).

Kept separate from model.py so the synth and dataset code can normalize parameters
without importing torch.
"""

from __future__ import annotations

import numpy as np

from config import N_PARAMS, PARAM_SPEC


def normalize_params(params) -> np.ndarray:
    """Map real synth params (sequence in PARAM_SPEC order) to [0, 1]."""
    vals = np.asarray(params, dtype=np.float64).reshape(-1, N_PARAMS)
    out = np.empty_like(vals)
    for i, (_, lo, hi) in enumerate(PARAM_SPEC):
        out[:, i] = (vals[:, i] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def denormalize_params(norm_params) -> dict:
    """Map a [0, 1] vector back to a dict of real synth parameters."""
    vals = np.asarray(norm_params, dtype=np.float64).reshape(-1)
    return {name: float(lo + vals[i] * (hi - lo)) for i, (name, lo, hi) in enumerate(PARAM_SPEC)}
