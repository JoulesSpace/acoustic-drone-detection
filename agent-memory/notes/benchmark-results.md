---
title: Benchmark results (v0.1, real DADS subset)
type: note
date: 2026-06-01
tags: [benchmark, results]
---

# Benchmark results - v0.1

Dataset: balanced **DADS** subset, 300 drone + 300 non-drone real 16 kHz clips
(`data/dads`), 50/50 stratified train/test, seed 7. Threshold 0.5 for F1/acc;
AUCs are threshold-free. Run: `docker compose run --rm bench --data /work/data/dads`.

| approach        | F1     | ROC-AUC | PR-AUC | ms/clip | notes |
|-----------------|--------|---------|--------|---------|-------|
| **mfcc_lr**     | 0.997  | 1.000   | 0.995  | 2.1     | best; MFCC + logistic regression |
| **spectral_gate** | 0.977 | 0.998  | 0.993  | 2.7     | flatness/entropy/band-ratio + logistic |
| **cepstrum**    | 0.967  | 0.990   | 0.987  | 45.2    | accurate but O(N²) DCT → slow |
| **hps**         | 0.949  | 0.987   | 0.896  | 2.1     | harmonic-comb contrast |
| band_ratio      | 0.766  | 0.915   | 0.816  | 2.0     | baseline heuristic |
| template        | 0.706  | 0.995   | 0.992  | 2.0     | excellent **ranker** (AUC), F1 hurt by uncalibrated 0.5 threshold |

## Verdict vs upstream (goal: parity-or-better)
Published binary-detection baselines: Al-Emadi CNN ≈0.93, general mel-CNN F1 ≈0.955
(see [approaches-survey.md](approaches-survey.md)). **Four approaches
(mfcc_lr, spectral_gate, cepstrum, hps) meet or beat that** - and all are cheap
classical/light methods, not a heavy CNN. **Goal met.**

## Caveats (don't oversell)
- This is a **300+300 subset** of DADS, 50/50 random split → some risk of
  recording-level leakage (DADS doesn't expose recording IDs to split by). Treat
  these as in-distribution numbers, not field performance. Next: larger subset,
  group-aware split, distance/SNR stratification (see handoff).
- `template` shows why we report AUC, not just F1: it ranks near-perfectly
  (AUC 0.995) but its raw cosine scores sit below 0.5, tanking thresholded F1.
  A per-approach calibrated threshold (Youden's J) would fix the headline F1.
- `cepstrum` is 20× slower (O(N²) DCT per frame); fine for host benchmarking,
  would need an FFT-based cepstrum to be edge-viable.

## How the spread compares to synthetic
On `--synth` most approaches score ~1.0 (trivially separable). Real DADS is what
produced the spread above - see [synth-vs-real-generalization](../insights/synth-vs-real-generalization.md).
