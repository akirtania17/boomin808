"""Synthetic dataset.

There is no curated corpus of (song bassline -> ideal 808) pairs shipped with this
project, so training data is generated: sample random synth parameters, render each
one-shot, then extract its features. The model learns to invert that mapping
(features -> parameters). Swap build_dataset for a real labeled set when you have one;
the training loop does not need to change.
"""

from __future__ import annotations

import numpy as np

from config import PARAM_SPEC, SAMPLE_RATE
from features import extract_features
from params import normalize_params
from synth import render_one_shot


def sample_params(rng: np.random.Generator) -> np.ndarray:
    """Draw one random parameter vector uniformly inside each PARAM_SPEC range."""
    return np.array([rng.uniform(lo, hi) for _, lo, hi in PARAM_SPEC], dtype=np.float64)


def build_dataset(n: int = 2000, seed: int = 0, sr: int = SAMPLE_RATE):
    """Return (features [n, F] float32, norm_params [n, P] float32).

    Features are raw (un-normalized) here; the trainer normalizes them so the same
    normalization can be reused at inference time.
    """
    rng = np.random.default_rng(seed)
    feats = []
    params = []
    for _ in range(n):
        p = sample_params(rng)
        wav = render_one_shot(p, sr=sr)
        feats.append(extract_features(wav, sr=sr))
        params.append(p)
    feats = np.stack(feats).astype(np.float32)
    norm_p = normalize_params(np.stack(params))
    return feats, norm_p
