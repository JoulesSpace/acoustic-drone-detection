---
title: Hardening wave — honesty, live, edge, tiers, limits
type: handoff
date: 2026-06-01
tags: [handoff, v0.3, honesty, edge, live]
---

# Handoff — hardening wave (v0.3)

_Supersedes [multi-task suite](2026-06-01-multitask-suite.md). Read
[honest-limitations](../notes/honest-limitations.md) FIRST, then
[suite-results](../notes/suite-results.md) and [prior-art](../reference/prior-art.md)._

## What this wave added (all committed; `docker compose run --rm dev` green, 9 crates)
- **Honesty:** `xeval` proved the in-dist ~1.0 numbers are **leakage-inflated** —
  cross-dataset ROC-AUC collapses to **0.49–0.87**, physics methods (envelope/
  hps/fusion/feature_fusion) generalize, learned templates don't. Hard-negative
  FP rates measured (chainsaw fools hps 0.82; engine/heli fool envelope 0.32).
- **Live + hardware:** `drone-live` (cpal) — `devices` probe (this laptop = one
  Intel Smart Sound 2-ch 48 kHz array) + `listen` real-time alert loop.
- **Edge:** `drone-edge` `no_std` detector **cross-builds for riscv32imc &
  Cortex-M4F**, ~17–27 KB flash / ~0 static RAM. In CI. Xtensa (esp32-S3) needs
  the esp-rs fork — documented.
- **Tiers + speed↔accuracy:** `pareto` frontier (band_ratio→hps→mfcc_lr→
  feature_fusion) + `benchmarks/MODEL_CARDS.md` (tiny-edge / balanced / max).
- **Limits:** `ratesweep` — accuracy flat from 8 kHz up; robust to 4-bit for
  strong detectors. Cheap mics are fine.
- **Prior-art:** wide survey in `reference/prior-art.md` — OSS field is shallow
  (Batear ESP32 + Python notebooks); no Rust, no honest cross-dataset, no
  multi-tier. Our differentiator is exactly that.

## Honest standing (do NOT overclaim)
Real generalization is **~0.85 ROC-AUC at best** (physics methods), not ~1.0; and
even that is optimistic until a truly held-out set (DroneAudioset) is tested.
"Most advanced OSS / beats all upstreams" is NOT yet earned — it requires the
held-out number + a generalization improvement. That's the active task (#19).

## In flight
- **Augmentation agent (#19):** noise/SNR/gain/time-shift + hard-negative mix-in
  to retrain and (hopefully) narrow the cross-dataset gap; best-effort held-out
  DroneAudioset. Integrate its `robust.rs` + numbers when it lands.

## Next steps toward the goal
1. Truly held-out DroneAudioset (MIT, not in DADS) eval → the real number.
2. If augmentation helps, ship the augmented strong detectors as the default.
3. 32-brand vendor recognition head (breadth).
4. Group-aware split for DADS if any grouping signal can be derived.
5. README polish for "ship to millions" (the OWNER is actively editing root
   README — coordinate; I keep capabilities in `benchmarks/README.md` + memory).
6. Real esp32-C6 firmware demo + Android (cpal/Oboe) build.

## Conventions
Semantic commits, **no Co-Authored-By**, every folder `.folderinfo`, new crates →
path-dep drone-dsp/drone-bench + add to `scripts/check.sh`, keep cheap cores
`no_std`. Owner co-edits root README — don't clobber it; prefer `git checkout
main -- crates` then per-file integration for agents.
