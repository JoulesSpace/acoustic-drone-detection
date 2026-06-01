---
title: Synthetic scores lie; validate on real data
type: insight
date: 2026-06-01
tags: [benchmark, detection, gotcha]
---

# Synthetic scores lie — validate on real data

The `--synth` dataset is trivially separable: every approach scored ROC-AUC
≈0.97–1.00 on it, including ones that were **broken on real audio**.

The sharpest example: the **HPS** approach scored ROC-AUC **0.972 on synth** but
**0.082 on real DADS** — i.e. *inverted* (it ranked non-drones above drones).
Two real-data facts the synthetic generator didn't capture caused it:

1. Real multirotor blade-pass fundamentals sit ~**200–260 Hz**, not the ~100 Hz
   the synth used; and DADS *negatives* carry strong sub-100 Hz rumble. The
   80–260 Hz search band was partly keying on the negatives' rumble.
2. Normalizing harmonic-comb energy by **total** magnitude penalized real drones
   (their broadband motor hiss floods the off-comb total) while rewarding tonal
   negatives — inverting the score.

**Fix:** a *level-invariant local* harmonic-to-background contrast (each tooth
vs. its inter-harmonic valleys, ≥2 well-formed teeth, absolute energy gate) at a
corrected ~100–330 Hz f0 band, plus a mild high-frequency motor-whine bonus.
Result: real ROC-AUC **0.082 → 0.987**.

**Lessons:**
- Synthetic data validates *plumbing*, never *generalization*. Always confirm on
  real recordings before trusting an approach.
- Prefer **level-invariant, local-contrast** features over fraction-of-total
  ones — the latter invert under broadband content.
- Report **AUC** (threshold-free) alongside F1: `template` looked weak on F1
  (uncalibrated threshold) but is a near-perfect ranker (AUC 0.995).

See [benchmark-results](../notes/benchmark-results.md).
