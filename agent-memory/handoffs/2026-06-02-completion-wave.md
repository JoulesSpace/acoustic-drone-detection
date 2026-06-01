---
title: Completion wave - vendor, real firmware, field pipeline, physics_fused
type: handoff
date: 2026-06-02
tags: [handoff, v0.4, completion]
---

# Handoff - completion wave (v0.4)

_Supersedes [hardening wave](2026-06-01-hardening-wave.md). Read
[honest-limitations](../notes/honest-limitations.md) and
[suite-results](../notes/suite-results.md)._

## What this wave closed (all committed; `docker compose run --rm dev` green, 11 crates)
- **physics_fused** detector: cross-dataset ROC-AUC **0.925** (new best honest
  generalizer; physics-only features, smallest generalization gap 0.074).
- **drone-vendor**: multi-brand recognition. Real 32-brand College-of-Charleston
  set (via the drone-visualization repo) macro-F1 **0.93** (within-recording
  caveat); synthetic 12-brand 1.0.
- **drone-firmware**: REAL esp32-C6 firmware (esp-hal 1.1), links for
  riscv32imac, `.text` ~66 KB, runs the real `drone-edge` detector + LED/serial
  alert. I2S mic read is the one marked stub. Build: `scripts/build-firmware.sh`.
- **Field pipeline**: `drone-live record` (labelled capture to data/field) +
  `fieldeval` bin (train DADS, test field) + `FIELD_PROTOCOL.md`. The honest
  held-out number is now ONE recording session + one command away.
- Signal-chain **infographic** (9 panels) in the README; em-dash banned repo-wide
  (use hyphens).

## The standing goal: where we honestly are
Engineering gaps an agent can close are CLOSED: detection (12 approaches +
physics_fused), DoA, type-ID, vendor-ID, blade-pass freq, robustness/SNR,
rate/bit-depth limits, speed-accuracy tiers, live mic + hardware probe, edge
lib + real firmware, honest cross-dataset eval + hard-negative confusion, field
tooling, wide prior-art.

**The one irreducible remaining step is physical:** record a held-out set with
the real drone (`drone-live record`), then `cargo run -r --bin fieldeval`. Until
that field number exists, claim leadership on ENGINEERING + HONEST EVALUATION
(no other OSS is Rust, multi-tier, edge-proven, with measured cross-dataset
generalization), NOT on absolute "beats all upstreams accuracy."

## Next (mostly needs the human or more hardware)
1. Record field data -> run `fieldeval` -> the real generalization number.
2. Wire a real I2S MEMS-mic read into `drone-firmware` (replace the stub) + flash a C6.
3. Android build of `drone-live`/detector (cpal/Oboe + JNI).
4. Distance/SNR regression head; multi-recording-per-brand vendor data.
5. Configurable FFT frame size (decouple from 16 kHz / 1024).

## Conventions
Semantic commits, NO Co-Authored-By, every folder `.folderinfo`, hyphens not
em-dashes, new crates -> path-dep drone-dsp/drone-bench + add to
`scripts/check.sh` (firmware excluded; built via scripts/build-firmware.sh),
keep cheap cores no_std. Owner co-edits root README - don't clobber.
