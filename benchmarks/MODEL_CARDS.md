# Hardware-tier model cards

"Different detections for different hardware capabilities." Each of the 12
benchmarked approaches is mapped to one of three hardware tiers by its
*structural* cost - what it must store and compute per clip - not just its
measured wall-clock time. The numbers below come from
`crates/drone-bench/src/bin/pareto.rs` over the **DADS** dataset (600 clips,
300 drone / 300 non-drone), single stratified 50/50 split, `score()` timed 3×
per clip on the host CPU. Regenerate with:

```
cargo run --release --manifest-path crates/drone-bench/Cargo.toml --bin pareto
python benchmarks/pareto.py            # -> benchmarks/plots/pareto.png
```

> **Leakage caveat.** Accuracy here is from a single random split of one dataset
> (DADS), with positives and negatives drawn from related recordings. Clips from
> the same source/session can land on both sides of the split, so these
> ROC-AUC / F1 figures are an **optimistic upper bound**, not field accuracy.
> Near-1.0 AUCs in particular reflect how separable this dataset is, not how the
> detector will behave on unseen drones, mics, and backgrounds. Treat the
> *ordering* and the *speed* numbers as the trustworthy signal; discount the
> absolute accuracy. K-fold (`--kfold 5`) tightens the estimate but does not
> remove source-level leakage.

`latency_us/frame` is microseconds per 1024-sample analysis frame
(`drone_dsp::FRAME_SIZE`); `xRT` is how many times faster than real time one
stream runs on this machine (higher is better). `dim` is the rough feature
dimension, used as a relative compute-cost proxy.

---

## tiny-edge - esp32-class MCU

- **Intended hardware:** microcontroller, ~16 kHz mono mic, KB of RAM, no
  FPU-heavy / matrix workload. The detector is a handful of scalar statistics
  over a magnitude spectrum - there is no learned weight matrix to store and no
  per-frame ML, so it fits where everything else does not.
- **Approaches:** `band_ratio`, `spectral_gate`, `hps`.
- **Feature set:** band energy ratios (`band_ratio`), a 5-element spectral
  summary fed to a tiny learned gate (`spectral_gate`), and a
  harmonic-product-spectrum / comb score (`hps`).
- **Approx. params / feature-dim:** 1–5 features; effectively no model state
  beyond a threshold (`spectral_gate` carries a 5-weight logistic gate).
- **Measured latency & real-time factor (release build, host CPU):**
  - `band_ratio` - ~25 us/frame, ~2500× RT.
  - `hps` - ~26 us/frame, ~2460× RT.
  - `spectral_gate` - ~33 us/frame, ~1950× RT.
- **Expected accuracy (leakage caveat applies):** ROC-AUC 0.94 (`band_ratio`) to
  0.99 (`hps`, `spectral_gate`); best-F1 0.92–0.99. `hps` is on the Pareto
  frontier's "good enough, near-free" shoulder - strong harmonic structure is
  the cheapest reliable drone cue.
- **Use when…** you are on bare metal / an MCU, power and RAM are the binding
  constraints, and a modest accuracy hit is acceptable. Start with `hps`; fall
  back to `band_ratio` only if FFT cost itself is too high.

## balanced - phone / Raspberry Pi / Android

- **Intended hardware:** a real CPU with an FPU and MB-to-GB of RAM, but a
  power / thermal budget - a phone, Raspberry Pi, or Android device. Affords a
  learned linear / small-MLP head over a modest cepstral or spectral feature, or
  a single template / patch correlation.
- **Approaches:** `mfcc_lr`, `gtcc_lr`, `mfcc_mlp`, `cepstrum`,
  `envelope_periodicity`, `template`, `spectrogram_template`.
- **Feature set:** 13 MFCC mean+std → logistic regression (`mfcc_lr`); the same
  with a gammatone filterbank (`gtcc_lr`); 27 features → a 24-unit MLP
  (`mfcc_mlp`); quefrency-peak + comb-energy blend (`cepstrum`); AM modulation
  strength of the amplitude envelope (`envelope_periodicity`); cosine similarity
  to a 512-bin mean log-spectrum template (`template`); and a 24-mel × 16-time
  log-mel patch correlation (`spectrogram_template`).
- **Approx. params / feature-dim:** 27-dim feature for the cepstral models
  (`mfcc_mlp` adds ~700 MLP weights: 27×24 + 24); 34 for the comb/envelope
  pair's effective 2-dim score; 512-dim template; 384-dim spectrogram patch.
- **Measured latency & real-time factor (release build, host CPU):**
  - `mfcc_lr` - ~26 us/frame, ~2440× RT (fastest learned model).
  - `template` - ~25 us/frame, ~2580× RT.
  - `mfcc_mlp` - ~27 us/frame, ~2390× RT.
  - `spectrogram_template` - ~30 us/frame, ~2170× RT.
  - `gtcc_lr` - ~36 us/frame, ~1770× RT.
  - `cepstrum` - ~740 us/frame, ~87× RT (autocorrelation-heavy).
  - `envelope_periodicity` - ~1066 us/frame, ~60× RT (long envelope analysis).
- **Expected accuracy (leakage caveat applies):** ROC-AUC 0.98–1.00; best-F1
  0.97–0.99. `mfcc_lr` is the headline balanced choice - it sits on the Pareto
  frontier as the fastest model that still reaches ~1.0 AUC on this split.
- **Use when…** you have a phone / Pi-class CPU and want the best accuracy per
  millisecond. Default to `mfcc_lr`; reach for `mfcc_mlp` or `gtcc_lr` only if a
  small accuracy gain justifies the extra compute. Avoid `cepstrum` /
  `envelope_periodicity` on this tier unless their specific cue is needed - they
  are ~30× slower for no accuracy advantage here.

## max-accuracy - server / workstation

- **Intended hardware:** server or workstation CPU, accuracy-first, latency and
  memory effectively unconstrained for a single stream.
- **Approaches:** `feature_fusion`, `fusion`.
- **Feature set:** `feature_fusion` concatenates 13 MFCCs (mean+std) with 8
  hand-engineered spectral/harmonic extras → a 34-dim logistic regression.
  `fusion` is a stacked ensemble: it runs the 6 base detectors
  (`band_ratio`, `template`, `hps`, `spectral_gate`, `cepstrum`, `mfcc_lr`) and
  learns a meta-model over their outputs.
- **Approx. params / feature-dim:** `feature_fusion` 34-dim feature + 35
  logistic weights; `fusion` stacks 6 base models plus a 6-input meta-head, so
  its cost is roughly the sum of its members.
- **Measured latency & real-time factor (release build, host CPU):**
  - `feature_fusion` - ~46 us/frame, ~1410× RT.
  - `fusion` - ~908 us/frame, ~71× RT (pays for all 6 base detectors).
- **Expected accuracy (leakage caveat applies):** ROC-AUC ~1.00; best-F1 ~0.99.
  `feature_fusion` is the top-accuracy point **and** is on the Pareto frontier -
  nothing is both faster and more accurate. `fusion` matches its accuracy but is
  ~20× slower, so it is dominated.
- **Use when…** accuracy is paramount, you are on server hardware, and you want
  the most robust decision (fusing complementary cues guards against any single
  feature's failure mode). Prefer `feature_fusion`: it gives the same top
  accuracy as `fusion` at a fraction of the cost. Reserve `fusion` for when
  ensemble diversity / per-detector introspection is itself the goal.

---

## Pareto frontier (this dataset)

The speed/accuracy non-dominated set - no other approach is both cheaper *and*
more accurate - sorted cheapest → most accurate, is:

| approach | tier | cost rank | us/frame (measured) | ROC-AUC | best-F1 |
| --- | --- | --- | --- | --- | --- |
| `band_ratio` | tiny-edge | 10 | ~25 | ~0.938 | ~0.917 |
| `hps` | tiny-edge | 12 | ~26 | ~0.992 | ~0.983 |
| `mfcc_lr` | balanced | 20 | ~26 | ~1.000 | ~0.990 |
| `feature_fusion` | max-accuracy | 40 | ~46 | ~1.000 | ~0.997 |

Everything else is dominated by one of these four. **Determinism note:** the
frontier is computed on a fixed per-approach *cost rank* (a relative-cost proxy
assigned from each algorithm, mirroring the measured speed ordering), not on the
raw stopwatch latency - measured latency jitters run-to-run, so using it for
frontier *membership* would make the flagged set non-deterministic for the cheap,
speed-tied cluster. The reported / plotted `latency_us_per_frame` is still the
measured value; only the dominance test uses the deterministic proxy, so the
frontier set is identical on every run for fixed accuracy.

Practical reading, one pick per tier: `band_ratio`/`hps` on tiny-edge (near-free,
`hps` for accuracy or `band_ratio` for the absolute floor), `mfcc_lr` as the
balanced workhorse (cheap learned head at ~1.0 AUC), and `feature_fusion` on
max-accuracy when you can afford it (top accuracy). See
`benchmarks/plots/pareto.png` for the full picture and
`benchmarks/results/pareto.json` for the raw numbers.
