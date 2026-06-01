---
title: Multi-task suite — detection + DoA + ID + freq + robustness
type: handoff
date: 2026-06-01
tags: [handoff, suite, v0.2]
---

# Handoff — multi-task suite (v0.2)

_Supersedes [the benchmark-wave handoff](2026-06-01-benchmark-wave.md). Read
[architecture](../notes/architecture.md) for the big picture and
[suite-results](../notes/suite-results.md) for all numbers._

## State (DONE & verified — `docker compose run --rm dev` is green across 7 crates)

The repo is now an insightface-style multi-task suite over a shared DSP backbone:

- **Detection** — 12 approaches in `drone-bench`, best F1 1.000, 8/12 beat CNN
  baselines, all real-time. Calibrated-F1 + k-fold + SNR + ×real-time reporting.
- **Direction of arrival** — `drone-doa`: GCC-PHAT + ULA → azimuth, RMSE 0.88°
  @20 dB. `no_std` core.
- **Type ID** — `drone-id`: multiclass (Al-Emadi), macro-F1 0.86 + confusion.
- **Property inference** — `drone-freq`: blade-pass f0/RPM, synth MAE ~1 Hz.
- **Robustness** — SNR-sweep degradation curves; learned methods hold to −10 dB.

Plots in `benchmarks/plots/` (metrics, ROC, PR, cost-vs-quality, robustness_*).
Data: `docker compose run --rm data`; each head runnable per the README.

## Lessons captured this session
- [Synthetic scores lie; validate on real data](../insights/synth-vs-real-generalization.md)
- [Commit before dispatching tree-mutating agents](../insights/commit-before-dispatching-tree-mutating-agents.md)
- **Agent benches write a stray crate-local `benchmarks/` when run from the crate
  dir** — strip `crates/*/benchmarks` on integration (results belong at repo root;
  the root `/benchmarks/results/*` gitignore doesn't cover nested copies).
- **clap rejects `--snr -10`** (parsed as a flag); use `--snr=-10`.

## Next steps (for the reviewer / next session)
1. **Eval hardening:** group-aware splits (avoid leakage), larger DADS subset,
   k-fold by default for headline numbers, distance/SNR-stratified reports.
2. **Stronger heads:** MLP/CNN for `drone-id`; a Python mel-CNN oracle for the
   classical-vs-ML ceiling; calibrated thresholds baked into `drone-detect`.
3. **Real multi-mic** for `drone-doa` (DREGON / a home array); planar array for
   elevation; replace simulation-only numbers.
4. **New properties:** distance/SNR regression (DroneAudioSet has labels), rotor
   count, motor-health — more "inferrable params" heads.
5. **Edge port:** lower the winning light detectors + DoA core onto esp32/riscv;
   prove a cross-build; replace `cepstrum` O(N²) DCT with an FFT-based cepstrum.
6. **System design** doc: mic count/geometry/sample-rate/range trade-offs.

## Conventions reminder
Semantic commits, **no Co-Authored-By**, every folder has `.folderinfo`, keep
this store + `CLAUDE.md` current. New crates: path-dep on `drone-dsp`/`drone-bench`,
add to `scripts/check.sh`, keep cheap cores `no_std`.
