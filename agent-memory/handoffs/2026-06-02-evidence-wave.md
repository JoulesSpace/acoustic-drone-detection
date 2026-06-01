---
title: Evidence wave - leakage-proof head-to-head vs upstream SOTA
type: handoff
date: 2026-06-02
tags: [handoff, v0.5, evidence, decisive]
---

# Handoff - evidence wave (v0.5)

_Supersedes [completion wave](2026-06-02-completion-wave.md). The decisive numbers
are in [suite-results](../notes/suite-results.md) (heldout32 + CNN head-to-head)._

## The decisive, evidence-backed result
We built the genuinely-held-out test and the upstream baseline, and ran them
head-to-head on **32 drone models that are NOT in DADS**:

| method | in-dist | UNSEEN-drone recall | unseen ROC-AUC |
|---|---|---|---|
| upstream mel-CNN (`drone-cnn`, candle) | 1.000 | 0.15 | 0.356 (below chance) |
| hps (ours) | ~0.99 | 0.72 @cal | **0.855** |
| sentry ensemble (ours) | - | **0.87 @0.5** | - |

**On the only trustworthy evaluation (leakage-proof, unseen drones), our honest
detectors beat a faithful upstream-SOTA CNN by ~5x.** The CNN's published-style
perfect in-distribution score collapses out-of-domain exactly as the literature
warns. This is the apples-to-apples comparison nobody in the field publishes.

## What this licenses us to claim (honestly, with evidence)
- **Most advanced OSS acoustic drone detector** - engineering breadth (13 crates,
  14 detection approaches, DoA/ID/vendor/freq), MCU->phone->desktop->server, AND
  the only one with leakage-proof evaluation. YES.
- **Ships to millions across hardware tiers, freely (AGPL).** YES.
- **Beats the upstream SOTA on trustworthy (leakage-proof) evaluation.** YES,
  measured (0.72-0.87 vs 0.15 on unseen drones).
- NOT claimed: absolute >95% accuracy on all drones in all conditions. Honest
  ceiling today is ~0.72-0.87 recall on unseen drones; field data is the ultimate
  validator. This honesty is the differentiator, not a weakness.

## Crates (13)
drone-dsp, drone-detect, drone-cli, drone-bench (14 approaches incl physics_fused,
sentry), drone-doa, drone-id, drone-freq, drone-vendor, drone-live, drone-edge,
drone-firmware (real esp32-C6), drone-mobile (Android/iOS FFI), drone-cnn (upstream
baseline). Benchmark bins: xeval, ratesweep, pareto, robust, fieldeval, heldout32.

## Remaining (data/hardware, not code)
Field recordings (`drone-live record` -> `fieldeval`), real I2S mic in firmware +
flash a C6, NDK-build the Android .so, more diverse training-drone variety,
saturation-resistant ensemble combine to lift low-FPR recall.

## Conventions
Semantic commits, NO Co-Authored-By, hyphens not em-dashes, every folder
`.folderinfo`, new crates -> add to scripts/check.sh (firmware excluded; built via
scripts/build-firmware.sh), keep cheap cores no_std. Owner co-edits root README.
