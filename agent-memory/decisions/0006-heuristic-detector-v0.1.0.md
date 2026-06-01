---
title: Heuristic detector for v0.1.0 (not ML)
type: decision
date: 2026-06-01
status: accepted
tags: [detection, dsp]
---

# 0006 - Heuristic detector for v0.1.0 (not ML)

**Decision:** Detection = band-energy ratio in ~100–4000 Hz **plus** a
dominant-tone-in-band test, threshold 0.5.

**Why:** Transparent, cheap enough for a microcontroller, and gives a measurable
baseline to beat. No labelled data needed (constraints: one real drone, an
afternoon, 50€ budget). A synthetic harmonic-stack generator (`drone synth`)
exercises the whole pipeline without recordings.

**Verified:** discriminates synthetic drone from a plain tone - see
[insight](../insights/synthetic-signal-discriminates.md). The reasoning behind
the band/tonal approach is in [notes](../notes/detection-thesis.md).

**Revisit when:** real data is wired in; expect to add temporal smoothing,
harmonic-spacing features, and SNR normalization (and eventually ML).
