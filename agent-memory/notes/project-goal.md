---
title: Project goal (v0.x north star)
type: note
date: 2026-06-01
tags: [goal, benchmark]
---

# Project goal

**Implement multiple acoustic drone-detection methods (per the README scope),
benchmark them apples-to-apples, and reach parity-or-better vs upstream/published
results.**

Concretely:
- ≥5 distinct approaches implemented against the `drone-bench` harness, each
  emitting a calibratable confidence in [0,1].
- Benchmarked on a real dataset (DADS subset, see [datasets](datasets.md)) with
  ROC-AUC / PR-AUC / F1 + a cost-vs-quality view (inference ms vs F1).
- Target: match or beat published F1 on the comparable binary-detection task
  (Al-Emadi CNN ≈ 0.93; general mel-CNN F1 ≈ 0.955) - at least for the
  classical/light methods at moderate SNR, and document where they fall short
  (distance/wind) vs the ML ceiling.
- Auto-generated matplotlib plots in `benchmarks/plots/` from result JSON.

"Upstream" = the published numbers in [approaches-survey.md](approaches-survey.md)
and the Al-Emadi baselines; parity means comparable F1 on a comparable split
(beware leakage - split by recording/drone/environment).
