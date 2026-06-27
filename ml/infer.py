"""Inference: generate a real 808 one-shot.

Two ways to drive it:
  1. Analyze a reference bassline (a .wav) and generate a one-shot tailored to it:
         python ml/infer.py --input path/to/bassline.wav --out examples/oneshot.wav
  2. Target a tone directly with explicit conditioning:
         python ml/infer.py --f0 50 --centroid 350 --decay 0.6 --out examples/oneshot.wav

If no checkpoint exists yet it trains a quick one first, so this always produces audio.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from scipy.io import wavfile

from config import CHECKPOINT_PATH, FEATURE_SPEC, ONE_SHOT_SECONDS, SAMPLE_RATE
from features import extract_features
from model import OneShot808Net
from params import denormalize_params
from synth import render_one_shot


def _load_wav_mono(path: str):
    sr, data = wavfile.read(path)
    data = np.asarray(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    # Normalize common integer PCM formats to [-1, 1].
    if np.issubdtype(np.asarray(wavfile.read(path)[1]).dtype, np.integer):
        max_abs = float(np.max(np.abs(data))) or 1.0
        data = data / max_abs
    return sr, data


def _ensure_checkpoint(path: str) -> None:
    if os.path.exists(path):
        return
    print(f"No checkpoint at {path}. Training a quick model first...")
    from train import train  # local import to avoid a hard dependency at import time

    train(n=1500, epochs=250, lr=1e-2, seed=0, out_path=path)


def load_model(path: str):
    _ensure_checkpoint(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = OneShot808Net()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    lows = np.asarray(ckpt["feature_lows"], dtype=np.float32)
    highs = np.asarray(ckpt["feature_highs"], dtype=np.float32)
    return model, lows, highs


def conditioning_from_args(args):
    """Return a raw feature vector in FEATURE_SPEC order."""
    if args.input:
        sr, x = _load_wav_mono(args.input)
        feats = extract_features(x, sr=sr)
        print(f"Analyzed {args.input}: f0={feats[0]:.1f} Hz, centroid={feats[1]:.0f} Hz, decay={feats[2]:.3f} s")
        return feats
    feats = np.array([args.f0, args.centroid, args.decay], dtype=np.float32)
    print(f"Target tone: f0={feats[0]:.1f} Hz, centroid={feats[1]:.0f} Hz, decay={feats[2]:.3f} s")
    return feats


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate an 808 one-shot.")
    ap.add_argument("--input", type=str, default=None, help="reference bassline .wav to match")
    ap.add_argument("--f0", type=float, default=50.0, help="target fundamental (used when no --input)")
    ap.add_argument("--centroid", type=float, default=350.0, help="target brightness Hz (no --input)")
    ap.add_argument("--decay", type=float, default=0.6, help="target decay seconds (no --input)")
    ap.add_argument("--out", type=str, default="examples/oneshot.wav")
    ap.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    args = ap.parse_args()

    model, lows, highs = load_model(args.checkpoint)

    feats = conditioning_from_args(args)
    norm_feats = np.clip((feats - lows) / (highs - lows), 0.0, 1.0).astype(np.float32)

    with torch.no_grad():
        norm_params = model(torch.from_numpy(norm_feats).reshape(1, -1)).numpy().reshape(-1)
    params = denormalize_params(norm_params)
    print("Predicted synth parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:.4f}")

    wav = render_one_shot(params, sr=SAMPLE_RATE, length_s=ONE_SHOT_SECONDS)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pcm = np.int16(np.clip(wav, -1.0, 1.0) * 32767)
    wavfile.write(args.out, SAMPLE_RATE, pcm)
    print(f"Wrote one-shot to {args.out} ({len(wav) / SAMPLE_RATE:.2f} s @ {SAMPLE_RATE} Hz)")


if __name__ == "__main__":
    main()
