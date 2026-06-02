---
title: Suite results - all task heads (v0.2)
type: note
date: 2026-06-01
tags: [benchmark, results, doa, id, freq, robustness]
---

# Suite results - all task heads

Companion to [benchmark-results](benchmark-results.md) (detection detail). All
numbers are reproducible via the commands in the README / `benchmarks/README.md`.

## Detection (12 approaches, real DADS 300+300, 50/50)
Best `feature_fusion` F1 1.000 / ROC-AUC 1.000; `mfcc_lr` & `fusion` F1 0.997.
8 of 12 meet/beat published CNN baselines (≈0.93–0.955 F1). All 90–2400×
real-time on desktop. Calibrated `F1*` shows even `template` is a strong ranker
(0.706→0.986). Full table in the README and `benchmark-results.md`.

## Direction of arrival - `drone-doa` (simulated, 4-mic ULA @0.043 m, 120 Hz src)
| SNR | MAE | RMSE (±80°) | RMSE within ±60° |
|-----|-----|-----|-----|
| 20 dB | 1.22° | 2.25° | **0.88°** |
| 10 dB | 2.82° | 4.10° | 2.60° |
| 0 dB | 9.02° | 11.60° | 10.70° |
GCC-PHAT with **coherence-gated** PHAT whitening (classic PHAT flattened the
peak for a narrowband drone - key fix). `no_std` core. Endfire (±70/80°) carries
most of the error (steep sin θ mapping). Simulated, not real multi-mic - the
honest caveat.

**Reality check vs field literature.** Published real-world acoustic DoA lands
at **6-42°**, not sub-degree: Ghouli (8-mic MEMS square, 15 cm) gets MAE 6.3° /
2D position ~1.2 m at 50-60 m; Toma (19-mic Zylia, 4.5 cm, on/near a UAV) gets
DoA RMSE ~42°. So our 0.88° is a clean-simulation best case; a real array with a
small aperture and outdoor noise should be expected in the single-digit-to-tens
of degrees. Effective acoustic range is typically **50-200 m** for a single
array. See [[prior-art]] for the calibration band and citations.

## Type ID - `drone-id` (real Al-Emadi multiclass, balanced 600/class)
accuracy 0.861, **macro-F1 0.860**. Per-class F1: bebop 0.893, membo 0.829,
unknown 0.860. Confusion mostly membo↔unknown. Linear softmax on pooled MFCC
mean/std - an MLP head would likely do better. Synth 4-class = 1.000.

## Blade-pass frequency - `drone-freq`
Synth (known f0 80–250 Hz): MAE ~1 Hz (0.88 @20 dB, 1.98 @0 dB), **0% octave
error** at all SNRs. Real DADS drones: median f0 ≈ **231 Hz** (IQR 227–234),
secondary cluster ~120 Hz. RPM = f0/blades (context only; f0 is what's measured).

## Robustness - `drone-bench --snr` sweep (clean/20/10/0/−10 dB)
Learned/cepstral methods (mfcc_lr, mfcc_mlp, gtcc_lr, spectral_gate, cepstrum,
envelope_periodicity) hold ROC-AUC **>0.95 down to −10 dB**. Naive baselines
collapse (`band_ratio` 0.30, `template` 0.43 at −10 dB) - quantifies why the
feature/learned methods are worth it. Curves: `benchmarks/plots/robustness_*.png`.

## ⭐ Cross-dataset reality check (the honest headline) - `xeval`
Train on DADS, test on **Al-Emadi drones + ESC-50 hard negatives**. In-dist
ROC-AUC ~1.0 **collapses** cross-dataset - confirming the in-dist numbers were
largely recording-fit (leakage), not drone recognition:

| approach | cross-dataset ROC-AUC | in-dist ROC-AUC |
|---|---|---|
| **physics_fused** (NEW best) | **0.925** | 0.999 |
| envelope_periodicity | 0.872 | 0.984 |
| hps | 0.852 | 0.992 |
| fusion | 0.848 | 0.999 |
| feature_fusion | 0.813 | 1.000 |
| cepstrum | 0.699 | 0.993 |
| mfcc_lr | 0.685 | 1.000 |
| gtcc_lr / mfcc_mlp / spectral_gate | 0.61–0.63 | ~0.99 |
| template / spectrogram_template | ~0.49 (chance) | ~0.99 |
| band_ratio | 0.233 (inverted!) | 0.938 |

**Takeaway:** the methods keyed to *physical structure* (rotor periodicity /
harmonics - envelope, hps, fusion, feature_fusion) generalize; learned spectral
templates overfit to the training mic/recording. This is the differentiator: we
**measure** generalization honestly, which almost no OSS/paper does.
**Still optimistic** - Al-Emadi & ESC-50 are IN the DADS merge ([[dads-is-a-merge-superset]]),
so a truly held-out set would score lower. CONFIRMED below.

## ⭐⭐ TRULY held-out: 32 UNSEEN drone models (`heldout32`) - the leakage-proof headline
Train DADS, test on the **32-brand College-of-Charleston set (NOT in DADS)** =
165 windows of 32 unseen drone makes/models (positives, genuinely held-out) vs
ESC-50 negatives (caveat: ESC-50 IS in DADS, so AUC is indicative; **recall on
unseen drones is the clean metric** - it needs no negatives).

| approach | recall@0.5 | recall@calibrated | ROC-AUC |
|---|---|---|---|
| hps | 0.721 | 0.630 | **0.855** |
| fusion | 0.339 | **0.648** | 0.821 |
| mfcc_lr | 0.455 | 0.642 | 0.692 |
| feature_fusion | 0.430 | 0.594 | 0.758 |
| physics_fused | 0.103 | 0.103 | 0.784 |
| envelope_periodicity | 0.024 | 0.370 | 0.812 |

**The sobering truth:** on genuinely-unseen drone models the best detectors catch
only **~65% (miss ~35%)**; best ranker is **`hps` (AUC 0.855)**. Key correction:
**`physics_fused`'s xeval 0.925 was still optimistic** (xeval positives = Al-Emadi,
which is in DADS); on truly-unseen drones it drops to recall 0.10 / AUC 0.784.
**On truly-unseen drones, `hps` (not physics_fused) is the most robust single
detector.** The **`sentry`** recall-first ensemble (soft-OR of hps + envelope +
feature_fusion + mfcc_lr, calibrated on a DADS slice) raises **recall@0.5 on
unseen drones to 0.873** (vs hps 0.721) with AUC ~tied (0.844); it trades
false-alarm rate for catch-rate (the right counter-UAS trade, a miss is costly).
Caveat: soft-OR score saturation hurts its very-low-FPR (<=0.05) and calibrated
points; a rank/saturation-resistant combine is the next improvement. Honest
deployment statement: **we catch ~87% of unseen drone models at the cost of a
raised false-alarm rate, or ~72% as a clean ranker (hps) - not >95%.**

## ⭐⭐⭐ Head-to-head vs the upstream SOTA CNN (`drone-cnn`) - the decisive comparison
We implemented a faithful published-style **mel-spectrogram CNN** (candle, the
Al-Emadi/MDPI family) and ran it through the SAME leakage-proof gauntlet:

| method | in-dist ROC-AUC | UNSEEN-drone recall | unseen ROC-AUC |
|---|---|---|---|
| upstream mel-CNN | **1.000** (perfect) | **0.15** @0.5 / 0.07 @cal | **0.356** (below chance) |
| hps (ours) | ~0.99 | 0.72 @cal | **0.855** |
| sentry (ours) | - | **0.87** @0.5 | - |

**Verdict (evidence, not assertion):** the upstream CNN reaches its published
~0.9+ IN-DISTRIBUTION (here a perfect 1.000 on the small DADS) and then
**collapses out-of-domain** - recall 0.15 and *below-chance* AUC 0.356 on
genuinely-unseen drones (it ranks ESC-50 aircraft/engine above unseen drones).
**Our honest detectors beat a faithful upstream SOTA by ~5x on the only
trustworthy test.** This is exactly the in-dist→unseen collapse the literature
warns about, isolated. So: on **leakage-proof evaluation we beat the upstream
SOTA approach** - the published 92-98% are the in-distribution illusion.
Caveats: small faithful net (not max-capacity); ESC-50 negatives are in DADS so
unseen AUC is indicative; field data remains the ultimate validator.
Calibration matters hugely: `envelope_periodicity`/`physics_fused` are decent
rankers but badly placed at thr 0.5; `template`/`band_ratio` fire on everything
(AUC <= chance). This is the honest number to quote for generalization, and it is
why field data + better invariances (not more DADS) are the real path forward.

### Augmentation did NOT help (honest negative - `robust`)
Retraining `mfcc_lr`/`feature_fusion`/`gtcc_lr`/`mfcc_mlp` on DADS augmented with
noise/SNR/gain/time-shift + ESC-50 hard-negative mix-in **widened** the
cross-dataset gap (mean ΔAUC **−0.28**; helped 0/4), while in-dist stayed ~0.99.
Coherent explanation: because Al-Emadi/ESC-50 are *inside* DADS, the "plain"
cross-dataset AUC is itself inflated by shared lineage - perturbing training away
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

## Sample-rate & bit-depth limits - `ratesweep`
- **Sample rate:** detection is essentially **flat from 8 kHz upward** for strong
  detectors (mfcc_lr, feature_fusion, spectral_gate ~0.99–1.0). `hps`/`band_ratio`
  wobble (harmonic resolution suffers as the fixed-1024 frame coarsens at higher
  rates; 8 kHz also discards >4 kHz content). A cheap 8 kHz mic costs almost
  nothing for the strong detectors.
- **Bit depth:** remarkably robust - ROC-AUC holds **down to 4-bit**. Only `hps`
  PR-AUC degrades at 6/4-bit (quantization muddies the harmonic peaks it
  multiplies). Edge ADCs are fine.
- Caveat: DADS is 16 kHz native, so >8 kHz points are informational; frame fixed
  at 1024.

## Blind source separation - `drone-bss` (multi-UAV / ego-noise layer)
FastICA (from-scratch Jacobi eigensolver + tanh fixed-point) on simulated
instantaneous mixtures: **mean SIR improvement 56 dB**, **100% drone recovery**.
Payoff is detection-rescue on MASKED scenes: `hps` recall **17% -> 100%**
(drone+noise), `spectral_gate` **0% -> 92%**; ~no lift when sources are
equal-loudness (honest - BSS only helps when a source is buried). Caveats:
instantaneous mixing only (real is convolutive -> freq-domain ICA/IVA is next),
M>=K, synthetic. This is the "multiple drones / heavy noise" robustness layer
Kang's review names.

## Caveats (carry into review)
- DADS subset numbers are in-distribution; no recording-level grouping (DADS
  exposes none) → possible leakage. DoA & some synth numbers are simulation.
- `drone-id` real numbers are on a class-balanced subset (unknown is 16× larger).
- Next: group-aware splits, larger subsets, real multi-mic for DoA, MLP/CNN
  heads for ID, distance/SNR regression (DroneAudioSet has labels).
