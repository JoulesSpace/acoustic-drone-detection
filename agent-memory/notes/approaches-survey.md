---
title: Detection approaches survey
type: note
date: 2026-06-01
tags: [detection, benchmark, reference]
---

# Detection approaches survey (researched 2026-06-01)

The drone acoustic signature is highly structured: a **blade-pass fundamental
(BPF) + harmonic stack**, energy concentrated below ~2.5 kHz with tonal content
up to ~10–16 kHz. Classical DSP is genuinely competitive for *binary detection*
at moderate SNR; ML (CNN) holds the robustness ceiling at distance/wind.

All approaches must emit a **confidence score in [0,1]** so we can compare via
ROC/PR, not just one threshold.

## Approaches to benchmark (ranked by value-for-effort, MCU-portable first)

1. **Harmonic-comb / HPS** — exploit BPF+harmonics directly; HPS downsample-and-
   multiply or comb correlation scanning f0. Cheap, `no_std`, likely strongest
   *classical* detector. Watch octave errors. → strongest physics/compute ratio.
2. **MFCC + logistic regression / linear SVM** — sweet spot accuracy vs cost;
   dot-product inference is MCU-deployable; LR gives calibrated probabilities.
   Literature: ~0.95+ F1 on clean data. Needs training + shipped mean/var.
3. **Spectral-feature gate** — centroid, flatness, rolloff, band-energy ratio,
   spectral entropy. Cheapest; great as a low-power gate and as features for (2).
   Drone = tonal → low flatness/entropy.
4. **Spectral template + cosine similarity** — the owner's hunch: average drone
   magnitude spectra → template, score by cosine sim, score = confidence.
   Include as simplest baseline; expect it to trail (1)–(3) under noise/model
   variation (rewards envelope, ignores harmonic structure). Quantifies what the
   harmonic structure in (1) actually buys.
5. **Cepstrum / autocorrelation periodicity** — cepstral peak at harmonic
   spacing = compact tonality/periodicity confidence; complements HPS (cross-
   check to kill octave errors).
6. **Mel-spectrogram + small CNN** — heavyweight reference *oracle* (accuracy
   ceiling), not the MCU target. Likely a later Python addition, not the Rust wave.
7. *(stretch)* **Feature-fusion shallow model** — MFCC + spectral + harmonic/
   cepstral features into one small model; fused beats any single feature.

## Published SOTA (indicative, NOT a common split — don't rank across rows)
- Al-Emadi DroneAudioDataset: CNN 92.94%, CRNN 92.22%, RNN 57.16% (STFT).
- General mel-CNN: ~95% acc, F1 ≈ 0.955; **~70% at 100 m**, big hit > 54 dB wind.
- MFCC+cubic-SVM ≈ 96.7%; MFCC+kNN ≈ 0.97 F1. Multiclass CNN ~98.7%.

## Evaluation conventions
- Headline metric **F1**; counter-UAS emphasizes **recall** (a miss is costly).
- Calibrate scores; compare with **ROC-AUC + PR-AUC + Brier**, not one threshold.
- Split by recording/drone/environment (avoid leakage); stratify by distance/SNR.
- Report a **cost vs quality** view (inference ms vs F1) — that's the
  insightface-style ensemble lens the owner asked for.

## Our plan
Implement #1–#5 in Rust against the `drone-bench` harness (one approach per file,
each emits a [0,1] confidence). #6 CNN oracle = later Python. See the benchmark
foundation in [`crates/drone-bench`](../../crates/drone-bench).
