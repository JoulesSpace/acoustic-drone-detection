---
title: Honest limitations & critical self-assessment (post first pass)
type: note
date: 2026-06-01
tags: [limitations, honesty, review, critical]
---

# Honest limitations & critical self-assessment

Read this BEFORE quoting any headline number. The infra is solid; the
*evaluation* is not yet trustworthy.

## The big one: probable dataset leakage → inflated numbers
DADS clips are short (0.5–7 s) fragments very likely cut from shared source
recordings. Our random clip-level 50/50 split almost certainly places
near-duplicate clips in train AND test, so models partly learn "which recording"
rather than "droneness." That is why detection F1/ROC-AUC sit at ~1.0 — **treat
those as optimistic in-distribution numbers, not generalization.** The honest
test is **cross-dataset** (train on one source, test on a different one) and
**group-aware** splits. Until that's run, do not claim we "beat CNN baselines"
(different datasets/splits — not apples-to-apples).

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
