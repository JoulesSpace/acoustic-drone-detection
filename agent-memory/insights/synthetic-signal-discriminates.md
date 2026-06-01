---
title: Synthetic signal discriminates cleanly
type: insight
date: 2026-06-01
tags: [detection, verification]
---

# The pipeline discriminates synthetic drone from a plain tone

**Verified 2026-06-01**, host and container:

| Input (`drone synth`)            | Result                                   |
|----------------------------------|------------------------------------------|
| synthetic drone, f = 120 Hz      | **100%** drone frames, dominant ~125 Hz, band ratio ~0.995 → `DRONE PRESENT` |
| `--plain` 60 Hz sine             | **0%** drone frames → `no drone`         |

**Why it matters:** the FFT → features → detector path is wired correctly and
the v0.1.0 heuristic separates the harmonic-band case from an out-of-band tone.

**Caveat:** this is *synthetic* audio (controllable stand-in, not a recording).
It validates the plumbing, **not** real-world robustness. The 60 Hz negative
control only proves out-of-band rejection - it is not a hard case. Real noisy
audio is the actual test; see the handoff's next steps.

Generator details: [notes/detection-thesis.md](../notes/detection-thesis.md).
