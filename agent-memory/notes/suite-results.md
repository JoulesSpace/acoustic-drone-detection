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

## ⭐ Cross-dataset reality check (the honest headline) — `xeval`
Train on DADS, test on **Al-Emadi drones + ESC-50 hard negatives**. In-dist
ROC-AUC ~1.0 **collapses** cross-dataset — confirming the in-dist numbers were
largely recording-fit (leakage), not drone recognition:

| approach | cross-dataset ROC-AUC | in-dist ROC-AUC |
|---|---|---|
| envelope_periodicity | **0.872** | 0.984 |
| hps | 0.852 | 0.992 |
| fusion | 0.848 | 0.999 |
| feature_fusion | 0.813 | 1.000 |
| cepstrum | 0.699 | 0.993 |
| mfcc_lr | 0.685 | 1.000 |
| gtcc_lr / mfcc_mlp / spectral_gate | 0.61–0.63 | ~0.99 |
| template / spectrogram_template | ~0.49 (chance) | ~0.99 |
| band_ratio | 0.233 (inverted!) | 0.938 |

**Takeaway:** the methods keyed to *physical structure* (rotor periodicity /
harmonics — envelope, hps, fusion, feature_fusion) generalize; learned spectral
templates overfit to the training mic/recording. This is the differentiator: we
**measure** generalization honestly, which almost no OSS/paper does.
**Still optimistic** — Al-Emadi & ESC-50 are IN the DADS merge ([[dads-is-a-merge-superset]]),
so a truly held-out set (DroneAudioset) would likely score lower. Follow-up open.

### Augmentation did NOT help (honest negative — `robust`)
Retraining `mfcc_lr`/`feature_fusion`/`gtcc_lr`/`mfcc_mlp` on DADS augmented with
noise/SNR/gain/time-shift + ESC-50 hard-negative mix-in **widened** the
cross-dataset gap (mean ΔAUC **−0.28**; helped 0/4), while in-dist stayed ~0.99.
Coherent explanation: because Al-Emadi/ESC-50 are *inside* DADS, the "plain"
cross-dataset AUC is itself inflated by shared lineage — perturbing training away
from those exact recordings erodes the *leaked* advantage faster than it builds
real invariance. Reinforces: physics methods generalize, learned-template data
volume of this kind does not manufacture recognition. **The real fix is a truly
held-out set + features with built-in invariance, not more of this augmentation.**

### Hard-negative false-positive rate (threshold 0.5, top generalizers)
- `fusion`: 2% overall FPR (most robust); slight chainsaw bleed.
- `hps`: **chainsaw 0.82** (strong harmonic stack fools the comb).
- `envelope_periodicity`: engine 0.32, helicopter 0.32 (shared rotor AM).
- `feature_fusion`: rain 0.32 (broadband).
- Wind / sea / fire / dog / clapping ~never trigger.
→ Real confounders are **other rotary/harmonic machines** (chainsaw, engine,
helicopter), exactly as the literature warns.

## Sample-rate & bit-depth limits — `ratesweep`
- **Sample rate:** detection is essentially **flat from 8 kHz upward** for strong
  detectors (mfcc_lr, feature_fusion, spectral_gate ~0.99–1.0). `hps`/`band_ratio`
  wobble (harmonic resolution suffers as the fixed-1024 frame coarsens at higher
  rates; 8 kHz also discards >4 kHz content). A cheap 8 kHz mic costs almost
  nothing for the strong detectors.
- **Bit depth:** remarkably robust — ROC-AUC holds **down to 4-bit**. Only `hps`
  PR-AUC degrades at 6/4-bit (quantization muddies the harmonic peaks it
  multiplies). Edge ADCs are fine.
- Caveat: DADS is 16 kHz native, so >8 kHz points are informational; frame fixed
  at 1024.

## Caveats (carry into review)
- DADS subset numbers are in-distribution; no recording-level grouping (DADS
  exposes none) → possible leakage. DoA & some synth numbers are simulation.
- `drone-id` real numbers are on a class-balanced subset (unknown is 16× larger).
- Next: group-aware splits, larger subsets, real multi-mic for DoA, MLP/CNN
  heads for ID, distance/SNR regression (DroneAudioSet has labels).
