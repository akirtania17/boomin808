"""Shared config for the Boomin 808 one-shot model.

Everything the synth, feature extractor, dataset, trainer and inference share lives
here so there is a single source of truth for sample rate, one-shot length, and the
parameter / feature ranges used for normalization.
"""

from __future__ import annotations

SAMPLE_RATE = 44100
ONE_SHOT_SECONDS = 0.8

# Synthesis parameters the model predicts, in fixed order.
# Each entry is (name, low, high) used for [0, 1] normalization.
PARAM_SPEC = [
    ("f0_hz", 30.0, 120.0),        # fundamental frequency of the body
    ("glide_octaves", 0.0, 2.0),   # how far above f0 the pitch starts before sliding down
    ("glide_time", 0.005, 0.08),   # seconds for the pitch glide to settle
    ("amp_decay", 0.10, 1.20),     # amplitude decay time constant in seconds
    ("drive", 0.0, 1.0),           # saturation amount (adds harmonics)
    ("click_gain", 0.0, 0.5),      # level of the onset transient
]

# Acoustic features used as the conditioning vector, in fixed order.
# (name, low, high) for [0, 1] normalization.
FEATURE_SPEC = [
    ("f0_hz", 20.0, 250.0),        # estimated fundamental
    ("centroid_hz", 40.0, 4000.0), # spectral centroid (brightness)
    ("decay_time", 0.02, 1.20),    # time to fall to 10 percent of peak RMS
]

PARAM_NAMES = [p[0] for p in PARAM_SPEC]
FEATURE_NAMES = [f[0] for f in FEATURE_SPEC]
N_PARAMS = len(PARAM_SPEC)
N_FEATURES = len(FEATURE_SPEC)

CHECKPOINT_PATH = "ml/checkpoints/oneshot808.pt"
