"""Acoustic feature extraction.

Turns a waveform into the small conditioning vector the model is trained on:
estimated fundamental, spectral centroid (brightness), and decay time. This is
deterministic DSP. When you analyze a song's bassline these are the three numbers
that describe the target tone the generator should aim for.
"""

from __future__ import annotations

import numpy as np

from config import FEATURE_SPEC, SAMPLE_RATE


def estimate_f0(x: np.ndarray, sr: int = SAMPLE_RATE, fmin: float = 20.0, fmax: float = 250.0) -> float:
    """Fundamental estimate: magnitude peak of the rFFT inside the bass band."""
    n = len(x)
    if n == 0:
        return fmin
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(band) or float(np.max(spec[band])) <= 0.0:
        return fmin
    band_idx = np.where(band)[0]
    peak = band_idx[int(np.argmax(spec[band]))]
    return float(freqs[peak])


def spectral_centroid(x: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Amplitude-weighted mean frequency. Higher means brighter."""
    n = len(x)
    if n == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    total = float(np.sum(spec))
    if total <= 1e-9:
        return 0.0
    return float(np.sum(freqs * spec) / total)


def decay_time(x: np.ndarray, sr: int = SAMPLE_RATE, frame: int = 1024, hop: int = 256) -> float:
    """Time in seconds for the windowed RMS to fall to 10 percent of its peak."""
    n = len(x)
    if n < frame:
        return 0.02
    starts = range(0, n - frame, hop)
    rms = np.array([np.sqrt(np.mean(x[s:s + frame] ** 2)) for s in starts])
    if rms.size == 0 or float(np.max(rms)) <= 1e-9:
        return 0.02
    peak_idx = int(np.argmax(rms))
    threshold = rms[peak_idx] * 0.1
    below = np.where(rms[peak_idx:] <= threshold)[0]
    if below.size == 0:
        return float(n / sr)
    frames_to_decay = below[0]
    return float((frames_to_decay * hop) / sr)


def extract_features(x: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return the conditioning vector in FEATURE_SPEC order."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.array(
        [estimate_f0(x, sr), spectral_centroid(x, sr), decay_time(x, sr)],
        dtype=np.float32,
    )


def normalize_features(feats: np.ndarray) -> np.ndarray:
    out = np.array(feats, dtype=np.float32).reshape(-1, len(FEATURE_SPEC))
    for i, (_, lo, hi) in enumerate(FEATURE_SPEC):
        out[:, i] = (out[:, i] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)
