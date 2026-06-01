# Benchmarks

Apples-to-apples evaluation of the acoustic drone-detection suite. Each detection
approach (see `crates/drone-bench/src/approaches/`) turns a clip into a confidence
in `[0, 1]`; the harness scores them, computes metrics, and emits JSON that the
Python/matplotlib plotters render. Other task heads (DoA, type-ID, frequency) have
their own crates and benchmarks.

## Run it (docker-first)

```bash
# fetch a real dataset subset into ./data/dads
docker compose run --rm data --per-class 300

# benchmark all 12 detectors on real data (add --kfold 5, --snr 0, --threshold ...)
docker compose run --rm bench --data /work/data/dads
docker compose run --rm plot                     # → plots/{metrics_bar,roc,pr,cost_quality}.png
```

`scripts/bench.sh [args]` does bench+plot in one go (defaults to synthetic).

## Benchmark binaries (`crates/drone-bench/src/bin/`)

Run any via the dev container, e.g.
`docker compose run --rm --entrypoint bash dev -c "cargo run -r --manifest-path crates/drone-bench/Cargo.toml --bin <name> -- --data /work/data/dads"`:

| bin | what it answers | plot |
|---|---|---|
| `drone-bench` (main) | per-approach metrics on one split | metrics_bar / roc / pr / cost_quality |
| `xeval` | **honest cross-dataset** (train DADS → test Al-Emadi + ESC-50) + per-confounder false-positive rate | — (JSON) |
| `pareto` | **speed↔accuracy frontier** + hardware-tier assignment | `pareto.py` → pareto.png |
| `ratesweep` | accuracy vs **sample rate** {8k–44.1k} and **bit depth** {4–16} | `ratesweep.py` → ratesweep.png |
| `robust` | does training-time **augmentation** narrow the cross-dataset gap | — (JSON) |

Robustness SNR sweep: `bench --snr <dB> --out-dir results/snr_<dB>` for several
levels, then `robustness.py` → robustness_{roc,f1}.png.

## ⚠ Read this before quoting numbers

In-distribution DADS numbers are **leakage-inflated** (DADS is a *merge superset*
of short clips; a random split puts near-duplicates in train+test). The honest
signal is **cross-dataset** (`xeval`): in-dist ROC-AUC ~1.0 **collapses to
0.49–0.87**. Best generalizers are the *physics* methods (`envelope_periodicity`
0.87, `hps` 0.85, `fusion` 0.85, `feature_fusion` 0.81); learned templates fall to
chance. Even that is optimistic (Al-Emadi/ESC-50 are inside DADS). Full honesty
writeup: [`agent-memory/notes/honest-limitations.md`](../agent-memory/notes/honest-limitations.md)
and [`suite-results.md`](../agent-memory/notes/suite-results.md).

## Hardware tiers

`pareto` assigns each detector to a tier; see
[`MODEL_CARDS.md`](MODEL_CARDS.md):
- **tiny-edge** (esp32-class MCU): `band_ratio`, `hps`, `spectral_gate` — and the
  `drone-edge` crate cross-compiles the rule detector to riscv32imc (~17–27 KB).
- **balanced** (phone / Pi): `mfcc_lr`, `gtcc_lr`, `mfcc_mlp`, `cepstrum`,
  `envelope_periodicity`, `template`, `spectrogram_template`.
- **max-accuracy** (server): `feature_fusion`, `fusion`.

## The 12 detection approaches

| name | idea | tier |
|---|---|---|
| `band_ratio` | baseline: mean band-energy ratio | tiny-edge |
| `hps` | harmonic product spectrum / comb | tiny-edge |
| `spectral_gate` | flatness/entropy/band-ratio + logistic | tiny-edge |
| `template` | cosine to averaged drone spectrum | balanced |
| `spectrogram_template` | 2D spectro-temporal template | balanced |
| `cepstrum` | cepstral / autocorrelation periodicity | balanced |
| `envelope_periodicity` | amplitude-modulation spectrum | balanced |
| `mfcc_lr` | MFCC + logistic regression | balanced |
| `gtcc_lr` | gammatone cepstral coeffs + logistic | balanced |
| `mfcc_mlp` | MFCC + small MLP | balanced |
| `feature_fusion` | fused MFCC+spectral+harmonic+cepstral + logistic | max-accuracy |
| `fusion` | logistic stack (ensemble) over the classics | max-accuracy |

> **Synthetic data lies:** `--synth` is trivially separable (~1.0 for everyone) —
> it validates plumbing only. Real spread shows on DADS, and real *generalization*
> only shows cross-dataset (`xeval`).
