---
title: Honest limitations & critical self-assessment (post first pass)
type: note
date: 2026-06-01
tags: [limitations, honesty, review, critical]
---

# Honest limitations & critical self-assessment

Read this BEFORE quoting any headline number. The infra is solid; the
*evaluation* is not yet trustworthy.

## The big one: dataset leakage → inflated numbers — NOW CONFIRMED
DADS clips are short (0.5–7 s) fragments cut from shared source recordings. Our
random clip-level split places near-duplicates in train AND test, so models learn
"which recording" more than "droneness." **The `xeval` cross-dataset test
confirmed this**: in-distribution ROC-AUC ~1.0 collapses to **0.49–0.87**
cross-dataset (best `envelope_periodicity` 0.872; `mfcc_lr` 0.685; `template`
~0.49; `band_ratio` 0.233 inverted). See [suite-results](suite-results.md#-cross-dataset-reality-check-the-honest-headline--xeval).

**So:** the headline ~1.0 numbers are recording-fit, NOT generalization. Honest
generalization is **~0.85 ROC-AUC at best** (the physics/periodicity methods).
Do NOT claim we "beat CNN baselines" — different datasets/splits, not
apples-to-apples. And even 0.85 is optimistic: `xeval` tested on Al-Emadi/ESC-50
which are *inside* the DADS merge ([dads-is-a-merge-superset](../insights/dads-is-a-merge-superset.md)) —
a truly held-out set (DroneAudioset) is the next test and will likely be lower.

## What is and isn't true (as of first pass)
- **Datasets:** used ~1.x real (DADS for detection, Al-Emadi for type-ID).
  Researched ~7. NOT "12 datasets" (12 = approaches). No cross-dataset eval.
- **Vendors/types:** only 3 Al-Emadi types, macro-F1 0.86. Not broad vendors.
- **Live/real-time from a mic:** NOT built. We proved the compute budget offline
  (90–2400× RT) but there is no cpal capture / alert demo yet.
- **Edge (esp32 / Android):** only `--no-default-features` *hygiene* builds. No
  cross-compile, no firmware, no Android. Strong detectors still std-only.
- **Hard negatives (wind turbine / car / airplane / helicopter):** UNTESTED.
  Likely our weakest point — aircraft/props share harmonic+broadband structure.
- **Multichannel:** only `drone-doa`, and simulated. Detection/ID/freq are mono.
- **Sample rate:** frame is hard-fixed at 1024, tuned for 16 kHz. Higher rates
  coarsen low-freq resolution (47 Hz/bin @48 kHz) → hurts blade-pass f0 unless
  frame scales. Not yet configurable.
- **Bit depth / codecs:** 16/24/32-bit PCM + float WAV (hound), downmix to f32.
  No mp3/opus decode. Compression effects on fine properties untested.
- **Speed↔accuracy:** only a cost-vs-quality scatter; no Pareto frontier / model
  tiers / FLOP estimates / latency-budget analysis.

## Priorities to fix (in rough order)
1. **Leakage-honest eval:** cross-dataset (DADS↔Al-Emadi), group-aware k-fold,
   and a **hard-negative** suite (ESC-50/UrbanSound8K aircraft/engine/wind). This
   reframes every headline number — do it first.
2. **Live demo:** `drone-live` (cpal) — enumerate input devices + capabilities
   (the "what mic/hardware do we have" probe) and a real-time listen+alert loop.
3. **Speed↔accuracy Pareto + model tiers** (tiny-edge / balanced / max-accuracy)
   with per-tier model cards (features, params, latency, FLOPs).
4. **Sample-rate/bit-depth robustness sweep** (resample DADS to 8k/16k/44.1k,
   quantize) to characterize the rate/bitrate limits empirically.
5. **Edge port proof:** a real riscv/xtensa cross-build of the tiny tier.
6. **Wider vendors/properties:** 32-brand dataset; distance/SNR regression.

See the multi-task map in [architecture](architecture.md) and prior numbers in
[suite-results](suite-results.md) (now to be read with the caveats above).
