# Benchmarks

Apples-to-apples comparison of acoustic drone-detection approaches. Each
approach (see `crates/drone-bench/src/approaches/`) turns a clip into a
confidence in `[0, 1]`; the harness scores them on a held-out test split and
emits metrics + curves, which the Python plotter renders.

## Run it (docker-first)

```bash
# 1. (optional) fetch a real dataset subset into ./data/dads
docker compose run --rm data --per-class 300

# 2a. benchmark on the real dataset …
docker compose run --rm bench --data /work/data/dads

# 2b. … or on synthetic data (no download needed)
docker compose run --rm bench --synth --n 300

# 3. render plots from the result JSON
docker compose run --rm plot
```

`scripts/bench.sh [args]` does steps 2+3 in one go (defaults to synthetic).

## What you get

- `results/<approach>.json` — per-approach metrics (accuracy, precision, recall,
  F1, ROC-AUC, PR-AUC, Brier), per-clip scores, and ROC/PR curve points.
  Git-ignored (regenerated); only the folder marker is tracked.
- `plots/` (tracked, so they render on GitHub):
  - `metrics_bar.png` — accuracy/precision/recall/F1 per approach
  - `roc.png` — ROC curves overlaid (AUC in legend)
  - `pr.png` — Precision-Recall curves
  - `cost_quality.png` — inference cost (ms/clip, log x) vs F1 — the
    cost-vs-quality view; upper-left is best.

## Approaches

| name            | idea                                              | trains? |
|-----------------|---------------------------------------------------|---------|
| `band_ratio`    | baseline: mean band-energy ratio                  | no      |
| `template`      | cosine similarity to averaged drone spectrum      | yes     |
| `hps`           | harmonic product spectrum / harmonic-comb         | —       |
| `spectral_gate` | flatness/entropy/band-ratio/centroid + logistic   | yes     |
| `cepstrum`      | cepstral / autocorrelation periodicity            | —       |
| `mfcc_lr`       | MFCC features + logistic regression               | yes     |

Methodology, SOTA targets, and leakage caveats: see
[`agent-memory/notes/approaches-survey.md`](../agent-memory/notes/approaches-survey.md)
and [`project-goal.md`](../agent-memory/notes/project-goal.md).

> **Note on synthetic data:** `--synth` is highly separable, so most approaches
> score near-perfectly on it — it validates the plumbing only. Real performance
> and the spread between approaches show up on the DADS dataset (`--data`).
