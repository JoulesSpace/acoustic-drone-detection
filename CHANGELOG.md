# Changelog

All notable changes to this project are documented here. The format loosely
follows [Keep a Changelog](https://keepachangelog.com/); versions follow semver.

## [Unreleased]

### Added
- **DSP backbone** (`drone-dsp`): `no_std` Hann windowing, real FFT (`microfft`),
  magnitude spectrum, and spectral features; float math via `libm` so it builds
  bare-metal.
- **Detection** (`drone-detect`, `drone-bench`): a transparent `no_std`
  energy-in-band heuristic plus an evaluation harness with a pluggable
  `Approach` trait, dataset loader (CSV / synth, stratified split, k-fold, SNR
  augmentation), and calibrated metrics (F1, calibrated-F1, ROC-AUC, PR-AUC,
  Brier, real-time factor). 12+ detection approaches benchmarked head-to-head.
- **Direction of arrival** (`drone-doa`): GCC-PHAT TDOA on a uniform linear array
  with a propagation simulator and an angular-error benchmark (`no_std` core).
- **Type / vendor ID** (`drone-id`, `drone-vendor`): multiclass drone-type and
  multi-brand recognition (MFCC + multinomial logistic) with per-class F1.
- **Blade-pass frequency / RPM** (`drone-freq`): HPS + cepstrum + autocorrelation
  fusion.
- **Range** (`drone-range`): distance estimation with an air-absorption tilt
  feature (`no_std` core).
- **Blind source separation** (`drone-bss`): FastICA for multi-UAV / ego-noise,
  with detection-rescue on masked scenes.
- **Edge** (`drone-edge`, `drone-firmware`, `drone-mobile`, `drone-live`):
  training-free `no_std` detector that cross-builds to `riscv32imc`, real
  esp32-C6 firmware, a C ABI / JNI FFI for Android/iOS, and a `cpal` live-mic
  runner (device probe, listen / alert, record).
- **Honest baseline** (`drone-cnn`): an upstream mel-CNN for a leakage-proof
  head-to-head against the classical detectors.
- **Tooling:** Docker-first workflow (runtime / dev / bench / plot / data
  services), `scripts/check.sh` correctness oracle, `.folderinfo` lint, signal-
  chain infographic, benchmark plots (ROC, PR, cost-vs-quality, robustness),
  GitHub Actions CI, Makefile, CITATION.cff, CONTRIBUTING.md, DATA_SOURCES.md.

### Results
- **Detection (in-distribution, DADS subset):** `feature_fusion` reaches F1 1.000
  / ROC-AUC 1.000; 8 of 12 approaches beat the CNN baselines, all 90-2400x
  real-time. These numbers are optimistic (likely recording-level leakage on a
  random clip split); the trustworthy cross-dataset / hard-negative evaluations
  are the stated priority.
- **Unseen-drone (leakage-proof head-to-head):** the upstream mel-CNN is perfect
  in-distribution (1.0) but collapses on unseen drones (recall 0.15, AUC 0.356);
  our recall-first ensemble holds recall 0.87 at the same operating point.
- **Direction of arrival:** RMSE 0.88 deg at 20 dB (4-mic ULA, simulated),
  2.8 deg at 10 dB.
- **Type ID:** macro-F1 0.86 on Al-Emadi multiclass (linear softmax).
- **Blade-pass frequency:** synthetic f0 MAE ~1 Hz, 0% octave error; real DADS
  drones cluster around 230 Hz.
- **Blind source separation:** 56 dB SIR, rescuing masked detection (HPS recall
  17 -> 100 percent).
- **Range:** simulated MAE 12.6 m; the air-absorption tilt feature cuts error
  from 27 m to 13 m.
- **Robustness:** learned methods hold ROC-AUC above 0.95 down to -10 dB and are
  robust to 4-bit quantization and to sample rates from 8 kHz up.

### Notes
- **No cargo workspace yet** (deliberate): crates use path deps. See
  `agent-memory/decisions/0002-crates-no-workspace.md`.
- License: **AGPL-3.0-or-later** (strong copyleft, network-use clause).
