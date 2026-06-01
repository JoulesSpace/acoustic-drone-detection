---
title: DADS is a merge superset — pick held-out sets carefully
type: insight
date: 2026-06-01
tags: [datasets, evaluation, leakage, gotcha]
---

# DADS is a merge superset — cross-dataset tests must avoid its members

The DADS dataset (`geronimobasso/drone-audio-detection-samples`) we benchmark on
is a **merge/superset of other datasets**, not a primary recording set:
- **Positives** merge: Al-Emadi DroneAudioDataset, SPCup19 Egonoise, DREGON,
  DroneNoise DB, AUDROK, drone-fault (Yi 2023).
- **Negatives** merge: UrbanSound8K, TUT Acoustic Scenes 2017, ESC-50, DNC.

**Consequence for evaluation:** training on DADS and "testing" on Al-Emadi or
ESC-50 is **NOT held-out** — those samples are (partly) already in DADS. Such a
"cross-dataset" result is optimistic (overlap leakage on top of the within-DADS
recording-level leakage).

**A truly disjoint cross-dataset test must use a source NOT in the merge:**
- **DroneAudioset** (`ahlab-drone-project`, MIT, multi-mic, low-SNR) — positives.
- **32-class brand set** (arXiv 2509.04715) — positives + type/brand.
- **RWDA** (IEEE-DataPort, multi-ch) — positives + DoA.
- For negatives genuinely outside DADS: AudioSet aircraft/helicopter subsets, or
  freshly recorded confounders.

The `xeval` agent was (mistakenly) pointed at Al-Emadi+ESC-50 before this was
known — its hard-negative *confusion* analysis is still useful (which confounders
fool which detectors), but its "cross-dataset generalization" numbers should be
read as **upper bounds**, not held-out. Follow-up: redo with DroneAudioset.

See [[honest-limitations]] and [prior-art](../reference/prior-art.md).
