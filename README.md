# Boomin808

Analyze a bassline and generate an 808 one-shot tailored to it, plus an interactive 808 tone-shaping tool.

## Overview

Boomin808 has two parts:

1. `ml/` - a small PyTorch model that takes a target tone (estimated from a reference bassline, or specified directly) and predicts the synthesis parameters for an 808 one-shot, then renders it to a real `.wav`.
2. `Boomin.py` - a desktop GUI that loads a `.wav` and shapes its low end through a distortion, compressor, EQ, and limiter chain (built on Pedalboard). This is the hands-on tone tool.

The ML side is the part that "analyzes a bassline and makes a one-shot for it." The synth and the audio analysis are deterministic DSP. The model is the learned step that maps a target tone to synth settings.

## How it works (the ML one-shot generator)

The pipeline is analysis by synthesis:

1. Synth (`ml/synth.py`). A deterministic 808 renderer. Given six parameters (fundamental, pitch-glide depth, glide time, amplitude decay, drive, click level) it renders a one-shot: a sine body whose pitch glides down into the fundamental, an exponential amplitude decay, tanh saturation for harmonics, and a short onset click. Same parameters in, same waveform out.

2. Features (`ml/features.py`). A deterministic feature extractor. From any waveform it computes a three-number conditioning vector: estimated fundamental (FFT peak in the bass band), spectral centroid (brightness), and decay time (time for the windowed RMS to fall to 10 percent of peak). When you analyze a song's bassline, these three numbers describe the tone you want.

3. Dataset (`ml/data.py`). Since there is no shipped corpus of (bassline -> ideal 808) pairs, training data is generated: sample random synth parameters, render each one-shot, then extract its features. This gives clean (features -> parameters) pairs.

4. Model (`ml/model.py`). `OneShot808Net` is a small MLP (3 -> 64 -> 64 -> 6, sigmoid output) that maps the feature vector to normalized synth parameters. It learns to invert the synth: given a target tone, output the parameters that produce it.

5. Inference (`ml/infer.py`). Analyze a reference `.wav` (or pass a target tone directly), run the model, denormalize the predicted parameters, render the one-shot, and write it to disk.

## What is actually trained vs scaffolding

This matters, so it is stated plainly:

- Trained: `OneShot808Net` genuinely trains end to end on the synthetic dataset and learns the recoverable structure. On held-out validation it beats the mean-prediction baseline (val MSE about 0.084 for the mean baseline, about 0.056 after training). The fundamental is recovered almost exactly (per-parameter validation MSE about 0.001) and amplitude decay is recovered well (about 0.035 vs 0.083 baseline).
- Partially identifiable: glide depth, glide time, drive, and click level are only weakly recoverable from the current three features, because several different settings produce nearly the same three numbers. The model predicts a reasonable average for these. Enriching the conditioning (for example an attack-energy feature and a harmonic-ratio feature) would make them identifiable. That is left as future work.
- Scaffolding: training on a real, curated set of song-and-bass pairs. The training loop does not change for that; you only swap `build_dataset` in `ml/data.py` for a loader over your labeled audio.

No pretrained weights ship with the repo. Training is fast on CPU, so you generate the checkpoint yourself.

## Tech stack

- ML: PyTorch, NumPy, SciPy (SciPy only for reading and writing `.wav`).
- Tone tool: Pedalboard, customtkinter, Pillow, requests.

## Setup

```
pip install -r requirements.txt
```

PyTorch, NumPy, and SciPy are enough for the ML side. The tone tool also needs Pedalboard and customtkinter.

## Usage

Train the model (writes `ml/checkpoints/oneshot808.pt`):

```
python -m ml.train
# or tune it:
python ml/train.py --n 2000 --epochs 200
```

Generate a one-shot from a target tone:

```
python ml/infer.py --f0 50 --centroid 350 --decay 0.6 --out examples/oneshot.wav
```

Generate a one-shot tailored to a reference bassline:

```
python ml/infer.py --input path/to/bassline.wav --out examples/matched.wav
```

If no checkpoint exists yet, `infer.py` trains a quick one first, so it always produces audio.

Run the interactive tone tool:

```
python Boomin.py
```

## Project structure

```
Boomin808/
  Boomin.py            interactive 808 tone-shaping GUI (Pedalboard effect chain)
  ml/
    config.py          sample rate, one-shot length, parameter and feature ranges
    synth.py           deterministic 808 renderer
    features.py        f0 / centroid / decay extraction
    params.py          parameter normalization helpers
    model.py           OneShot808Net (the MLP)
    data.py            synthetic dataset builder
    train.py           end-to-end training
    infer.py           generate a one-shot wav
  examples/            output folder for rendered one-shots
  requirements.txt
```

## Limitations and notes

- The model trains on synthetic 808s, so it learns the synthetic mapping. It is honest about which parameters it can and cannot recover from the current features (see above).
- The feature set is intentionally small (three numbers). Richer conditioning would make more parameters identifiable.
- `Boomin.py` is a separate DSP tool. It shapes existing audio and does not use the model.
- Output is a single mono one-shot at 44.1 kHz. It is not a full sampler or instrument.
