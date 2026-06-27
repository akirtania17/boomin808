"""End-to-end training.

Builds the synthetic dataset, trains OneShot808Net to predict synth parameters from
features, and saves a checkpoint (weights plus the feature normalization stats). Runs
on CPU in well under a minute with the defaults.

Usage:
    python -m ml.train                 # from the Boomin808 directory
    python ml/train.py --n 4000 --epochs 400
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn

from config import CHECKPOINT_PATH, FEATURE_SPEC, PARAM_SPEC
from data import build_dataset
from model import OneShot808Net


def normalize_features_with(feats: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    return np.clip((feats - lows) / (highs - lows), 0.0, 1.0).astype(np.float32)


def train(n: int, epochs: int, lr: float, seed: int, out_path: str, batch_size: int = 64) -> None:
    torch.manual_seed(seed)
    print(f"Building synthetic dataset: {n} one-shots...")
    feats, norm_params = build_dataset(n=n, seed=seed)

    lows = np.array([lo for _, lo, _ in FEATURE_SPEC], dtype=np.float32)
    highs = np.array([hi for _, _, hi in FEATURE_SPEC], dtype=np.float32)
    x = torch.from_numpy(normalize_features_with(feats, lows, highs))
    y = torch.from_numpy(norm_params)

    n_val = max(1, int(0.1 * len(x)))
    x_train, y_train = x[:-n_val], y[:-n_val]
    x_val, y_val = x[-n_val:], y[-n_val:]

    model = OneShot808Net()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Mean baseline: MSE if we always predict the training-set mean. Beating this
    # is what tells us the model actually learned the mapping rather than collapsing.
    mean_pred = y_train.mean(dim=0, keepdim=True)
    baseline_mse = loss_fn(mean_pred.expand_as(y_val), y_val).item()

    n_train = len(x_train)
    print(f"Training for {epochs} epochs on {n_train} samples (batch {batch_size})...")
    print(f"  mean-prediction baseline val_mse: {baseline_mse:.5f}")
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(x_train[idx]), y_train[idx])
            loss.backward()
            opt.step()
        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(x_val), y_val).item()
                # Per-parameter val MSE so the breakdown of what is learnable is visible.
                per = ((model(x_val) - y_val) ** 2).mean(dim=0).tolist()
            per_str = " ".join(f"{name}={per[i]:.3f}" for i, (name, *_ ) in enumerate(PARAM_SPEC))
            print(f"  epoch {epoch:4d}  train_mse {loss.item():.5f}  val_mse {val_loss:.5f}  [{per_str}]")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_lows": lows,
            "feature_highs": highs,
        },
        out_path,
    )
    print(f"Saved checkpoint to {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Boomin 808 one-shot model.")
    ap.add_argument("--n", type=int, default=2000, help="number of synthetic one-shots")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=CHECKPOINT_PATH)
    args = ap.parse_args()
    train(args.n, args.epochs, args.lr, args.seed, args.out)


if __name__ == "__main__":
    main()
