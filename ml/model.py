"""The learned model.

OneShot808Net maps a conditioning feature vector (target f0, brightness, decay) to
the synthesis parameters that render a matching 808 one-shot. It is a small MLP with
a sigmoid output, so every prediction is a normalized parameter vector in [0, 1] that
denormalize_params (in params.py) turns back into real synth values.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config import N_FEATURES, N_PARAMS

# Re-exported for convenience so callers can do `from model import denormalize_params`.
from params import denormalize_params, normalize_params  # noqa: F401


class OneShot808Net(nn.Module):
    def __init__(self, n_features: int = N_FEATURES, n_params: int = N_PARAMS, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_params),
            nn.Sigmoid(),  # outputs normalized params in [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
