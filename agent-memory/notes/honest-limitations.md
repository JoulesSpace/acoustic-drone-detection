---
title: Honest limitations & critical self-assessment (post first pass)
type: note
date: 2026-06-01
tags: [limitations, honesty, review, critical]
---

# Honest limitations & critical self-assessment

Read this BEFORE quoting any headline number. The infra is solid; the
*evaluation* is not yet trustworthy.

## The big one: dataset leakage → inflated numbers - NOW CONFIRMED
DADS clips are short (0.5-7 s) fragments cut from shared source recordings. Our
random clip-level split places near-duplicates in train AND test, so models learn
"which recording" more than "droneness." **The `xeval` cross-dataset test
confirmed this**: in-distribution ROC-AUC ~1.0 collapses to **0.49-0.87**
cross-dataset (best `envelope_periodicity` 0.872; `mfcc_lr` 0.685; `template`
~0.49; `band_ratio` 0.233 inverted). See [suite-results](suite-results.md#-cross-dataset-reality-check-the-honest-headline--xeval).

**So:** the headline ~1.0 numbers are recording-fit, NOT generalization. And even
the xeval cross-dataset 0.925 (`physics_fused`) was still optimistic, because
xeval's Al-Emadi positives are inside DADS. The **leakage-proof** test
(`heldout32`: 32 UNSEEN drone models, not in DADS) gives the real number:
**best recall on unseen drones ~0.65 (we miss ~35%); best ROC-AUC 0.855 (`hps`)**,
and `physics_fused` drops to recall 0.10 / AUC 0.784. On truly-unseen drones,
**`hps` is the most robust, not physics_fused.** That is the honest generalization
ceiling today.
We now HAVE the apples-to-apples evidence: a faithful upstream mel-CNN
(`drone-cnn`) run through the same leakage-proof `heldout32` gauntlet scores
perfect in-distribution (1.000) but **collapses to recall 0.15 / AUC 0.356 on
unseen drones**, while ours hold 0.72-0.87. So on TRUSTWORTHY (leakage-proof)
evaluation we DO beat the upstream SOTA approach - the published 92-98% are the
in-distribution illusion. (Stated as "on honest eval", with the caveats: small
faithful net, leaky negatives, field data still the ultimate validator.) And even 0.85 is optimistic: `xeval` tested on Al-Emadi/ESC-50
which are *inside* the DADS merge ([dads-is-a-merge-superset](../insights/dads-is-a-merge-superset.md)) -
a truly held-out set (DroneAudioset) is the next test and will likely be lower.

## Update (v0.4): which first-pass gaps are now CLOSED
The list below was the *first-pass* assessment. Since then, built + committed:
- **Live mic + hardware probe:** `drone-live` (`devices` + `listen`). DONE.
- **Edge:** `drone-edge` no_std cross-builds for riscv32imc; **`drone-firmware`**
  is a real esp32-C6 esp-hal firmware that links (riscv32imac, ~66 KB .text) and
  runs the detector (I2S mic read still a stub). DONE (flashing needs hardware).
- **Hard negatives:** `xeval` measured per-confounder FP rates (chainsaw/engine/
  helicopter are the real confusers). DONE.
- **Speed-accuracy Pareto + tiers + model cards; rate/bit-depth sweep.** DONE.
- **Vendors:** `drone-vendor` real 32-brand macro-F1 0.93 (within-recording). DONE.
- **Cross-dataset honest eval:** `xeval` + `physics_fused` (0.925). DONE.
- **Field tooling:** `drone-live record` + `fieldeval` + `FIELD_PROTOCOL.md`. DONE.

Still open (mostly needs the human/hardware): the actual **field recording**, a
real **I2S mic** in firmware + flashing a C6, **Android** build, configurable FFT
frame size, distance/SNR regression, multi-recording-per-brand vendor data.

## What is and isn't true (as of first pass)
- **Datasets:** used ~1.x real (DADS for detection, Al-Emadi for type-ID).
  Researched ~7. NOT "12 datasets" (12 = approaches). No cross-dataset eval.
- **Vendors/types:** only 3 Al-Emadi types, macro-F1 0.86. Not broad vendors.
- **Live/real-time from a mic:** NOT built. We proved the compute budget offline
  (90-2400× RT) but there is no cpal capture / alert demo yet.
- **Edge (esp32 / Android):** only `--no-default-features` *hygiene* builds. No
  cross-compile, no firmware, no Android. Strong detectors still std-only.
- **Hard negatives (wind turbine / car / airplane / helicopter):** UNTESTED.
  Likely our weakest point - aircraft/props share harmonic+broadband structure.
- **Multichannel:** only `drone-doa`, and simulated. Detection/ID/freq are mono.
  Field literature puts real acoustic DoA at **6-42°** (Ghouli 6.3°, Toma 42°),
  so our simulated 0.88° is best-case; range is typically **50-200 m** per array.
- **Sample rate:** frame is hard-fixed at 1024, tuned for 16 kHz. Higher rates
  coarsen low-freq resolution (47 Hz/bin @48 kHz) → hurts blade-pass f0 unless
  frame scales. Not yet configurable.
- **Bit depth / codecs:** 16/24/32-bit PCM + float WAV (hound), downmix to f32.
  No mp3/opus decode. Compression effects on fine properties untested.
- **Speed↔accuracy:** only a cost-vs-quality scatter; no Pareto frontier / model
  tiers / FLOP estimates / latency-budget analysis.

## Why a clean held-out test is genuinely hard (and what to do)
DADS **absorbed most of the public field**: its positives merge Al-Emadi, DREGON,
DroneNoise, AUDROK, fault-data; its negatives merge UrbanSound8K, ESC-50, TUT.
So almost every easily-downloadable drone/env-sound set is *already inside DADS*
→ no clean held-out. Worse, **DroneAudioset is NOT drone-vs-not** - it's recorded
*on* a drone (constant ego-noise) for detecting human sounds in SAR, so it can't
serve as held-out drone positives/negatives. The best remaining held-out
**positive** source is the **32-brand set** (arXiv 2509.04715, not in DADS);
held-out **negatives** are the harder gap (most env-sound sets are in DADS).
**Honest conclusion:** trustworthy generalization numbers ultimately need
**freshly recorded field data** (the owner has a real drone) - public data alone
can't fully prove it. State capability claims as *engineering* (only Rust
multi-tier edge-proven honest-eval suite) until field data exists, not as
absolute "beats all upstreams accuracy."

## Priorities to fix (in rough order)
1. **Leakage-honest eval:** cross-dataset (DADS↔Al-Emadi), group-aware k-fold,
   and a **hard-negative** suite (ESC-50/UrbanSound8K aircraft/engine/wind). This
   reframes every headline number - do it first.
2. **Live demo:** `drone-live` (cpal) - enumerate input devices + capabilities
   (the "what mic/hardware do we have" probe) and a real-time listen+alert loop.
3. **Speed↔accuracy Pareto + model tiers** (tiny-edge / balanced / max-accuracy)
   with per-tier model cards (features, params, latency, FLOPs).
4. **Sample-rate/bit-depth robustness sweep** (resample DADS to 8k/16k/44.1k,
   quantize) to characterize the rate/bitrate limits empirically.
5. **Edge port proof:** a real riscv/xtensa cross-build of the tiny tier.
6. **Wider vendors/properties:** 32-brand dataset; distance/SNR regression.

See the multi-task map in [architecture](architecture.md) and prior numbers in
[suite-results](suite-results.md) (now to be read with the caveats above).
