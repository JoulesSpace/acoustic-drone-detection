---
title: Suite results — all task heads (v0.2)
type: note
date: 2026-06-01
tags: [benchmark, results, doa, id, freq, robustness]
---

# Suite results — all task heads

Companion to [benchmark-results](benchmark-results.md) (detection detail). All
numbers are reproducible via the commands in the README / `benchmarks/README.md`.

## Detection (12 approaches, real DADS 300+300, 50/50)
Best `feature_fusion` F1 1.000 / ROC-AUC 1.000; `mfcc_lr` & `fusion` F1 0.997.
8 of 12 meet/beat published CNN baselines (≈0.93–0.955 F1). All 90–2400×
real-time on desktop. Calibrated `F1*` shows even `template` is a strong ranker
(0.706→0.986). Full table in the README and `benchmark-results.md`.

## Direction of arrival — `drone-doa` (simulated, 4-mic ULA @0.043 m, 120 Hz src)
| SNR | MAE | RMSE (±80°) | RMSE within ±60° |
|-----|-----|-----|-----|
| 20 dB | 1.22° | 2.25° | **0.88°** |
| 10 dB | 2.82° | 4.10° | 2.60° |
| 0 dB | 9.02° | 11.60° | 10.70° |
GCC-PHAT with **coherence-gated** PHAT whitening (classic PHAT flattened the
peak for a narrowband drone — key fix). `no_std` core. Endfire (±70/80°) carries
most of the error (steep sin θ mapping). Simulated, not real multi-mic — the
honest caveat.

## Type ID — `drone-id` (real Al-Emadi multiclass, balanced 600/class)
accuracy 0.861, **macro-F1 0.860**. Per-class F1: bebop 0.893, membo 0.829,
unknown 0.860. Confusion mostly membo↔unknown. Linear softmax on pooled MFCC
mean/std — an MLP head would likely do better. Synth 4-class = 1.000.

## Blade-pass frequency — `drone-freq`
Synth (known f0 80–250 Hz): MAE ~1 Hz (0.88 @20 dB, 1.98 @0 dB), **0% octave
error** at all SNRs. Real DADS drones: median f0 ≈ **231 Hz** (IQR 227–234),
secondary cluster ~120 Hz. RPM = f0/blades (context only; f0 is what's measured).

## Robustness — `drone-bench --snr` sweep (clean/20/10/0/−10 dB)
Learned/cepstral methods (mfcc_lr, mfcc_mlp, gtcc_lr, spectral_gate, cepstrum,
envelope_periodicity) hold ROC-AUC **>0.95 down to −10 dB**. Naive baselines
collapse (`band_ratio` 0.30, `template` 0.43 at −10 dB) — quantifies why the
feature/learned methods are worth it. Curves: `benchmarks/plots/robustness_*.png`.

## Caveats (carry into review)
- DADS subset numbers are in-distribution; no recording-level grouping (DADS
  exposes none) → possible leakage. DoA & some synth numbers are simulation.
- `drone-id` real numbers are on a class-balanced subset (unknown is 16× larger).
- Next: group-aware splits, larger subsets, real multi-mic for DoA, MLP/CNN
  heads for ID, distance/SNR regression (DroneAudioSet has labels).
