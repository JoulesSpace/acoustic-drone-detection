---
title: Benchmark wave - six approaches, goal met
type: handoff
date: 2026-06-01
tags: [handoff, benchmark, v0.1]
---

# Handoff - benchmark wave (six approaches, goal met)

_Supersedes [the v0.1.0 scaffold handoff](2026-06-01-v0.1.0-scaffold.md)._

## State (DONE & verified)

Six detection approaches implemented against the `drone-bench` harness and
benchmarked on a real **DADS** subset (300+300 clips). **Four beat/match the
published upstream baseline** (CNN ≈0.93-0.955 F1) - goal met. Full table:
[benchmark-results](../notes/benchmark-results.md). Headline: `mfcc_lr` F1 1.00,
`spectral_gate` 0.98, `cepstrum` 0.97, `hps` 0.95, all cheap classical/light
methods.

- Approaches: `crates/drone-bench/src/approaches/{band_ratio,template,hps,spectral_gate,cepstrum,mfcc_lr}.rs`.
- Data: `docker compose run --rm data --per-class 300` → `data/dads` (gitignored).
- Run: `docker compose run --rm bench --data /work/data/dads` then `… plot`.
- Plots committed in `benchmarks/plots/` (metrics, ROC, PR, cost-vs-quality).
- `ALL CHECKS PASSED` (fmt, clippy -D warnings, tests incl. hps/cepstrum units,
  no_std build).

## Lessons captured (read these)
- [Synthetic scores lie; validate on real data](../insights/synth-vs-real-generalization.md)
  - HPS was AUC 0.972 on synth but 0.082 (inverted) on real; fixed to 0.987.
- [Commit integrations before dispatching tree-mutating agents](../insights/commit-before-dispatching-tree-mutating-agents.md)
  - an agent's git revert clobbered uncommitted integrations.

## Next steps
1. **Stronger evaluation.** Larger DADS subset; group-aware split (avoid
   recording leakage); distance/SNR stratification; per-approach threshold
   calibration (Youden's J) so headline F1 reflects the ranker quality (esp.
   `template`).
2. **Ensemble.** Mesh approaches (insightface-style): logistic stack over the
   six confidences; expect a robustness gain over any single method.
3. **Edge port.** Move the winning light methods (`spectral_gate`, `mfcc_lr`,
   `hps`) into the `no_std` crates and prove a riscv/xtensa cross-build. Replace
   `cepstrum`'s O(N²) DCT with an FFT-based cepstrum before it's edge-viable.
4. **CNN oracle (optional).** A Python mel-CNN reference to quantify the
   classical-vs-ML ceiling on this data.
5. **Multiclass / DoA** per the README scope (drone type; multi-mic).

## Conventions reminder
Semantic commits, **no Co-Authored-By trailer**, every folder has `.folderinfo`,
keep this store + `CLAUDE.md` current. See [`../../CLAUDE.md`](../../CLAUDE.md).
