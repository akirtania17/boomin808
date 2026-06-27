"""Deterministic 808 one-shot renderer.

This is plain DSP, not learned. Given a small parameter vector it renders a real
808-style one-shot: a sine body whose pitch glides down into the fundamental, an
exponential amplitude decay, tanh saturation for harmonic content, and a short
onset click. The same parameters always render the same waveform, which is what
lets the model learn a clean features-to-parameters mapping.
"""

from __future__ import annotations

import numpy as np

from config import PARAM_SPEC, SAMPLE_RATE, ONE_SHOT_SECONDS


def _unpack(params):
    """Accept either a dict or a sequence in PARAM_SPEC order."""
    if isinstance(params, dict):
        return [float(params[name]) for name, _, _ in PARAM_SPEC]
    return [float(v) for v in params]


def render_one_shot(params, sr: int = SAMPLE_RATE, length_s: float = ONE_SHOT_SECONDS) -> np.ndarray:
    """Render a mono float32 one-shot in [-1, 1].

    params: dict keyed by PARAM_SPEC names, or a sequence in that order.
    """
    f0, glide_octaves, glide_time, amp_decay, drive, click_gain = _unpack(params)

    n = int(round(sr * length_s))
    t = np.arange(n, dtype=np.float64) / sr

    # Pitch envelope: start glide_octaves above f0, slide exponentially down to f0.
    f_start = f0 * (2.0 ** glide_octaves)
    glide = np.exp(-t / max(glide_time, 1e-4))
    inst_freq = f0 + (f_start - f0) * glide

    # Integrate instantaneous frequency to get phase.
    phase = 2.0 * np.pi * np.cumsum(inst_freq) / sr
    osc = np.sin(phase)

    # Saturation adds the harmonics that give an 808 its weight when driven.
    osc = np.tanh((1.0 + 4.0 * drive) * osc)

    # Amplitude envelope.
    amp = np.exp(-t / max(amp_decay, 1e-4))
    body = osc * amp

    # Onset click: a short, fast-decaying high-frequency burst (deterministic).
    click_len = int(round(0.006 * sr))
    click = np.zeros(n, dtype=np.float64)
    if click_len > 1:
        ct = np.arange(click_len, dtype=np.float64) / sr
        click[:click_len] = np.sin(2.0 * np.pi * 1800.0 * ct) * np.exp(-ct / 0.0015)

    out = body + click_gain * click

    peak = float(np.max(np.abs(out)))
    if peak > 1e-9:
        out = out / peak
    return out.astype(np.float32)
