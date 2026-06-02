---
title: Multi-task architecture (insightface-style)
type: note
date: 2026-06-01
tags: [architecture, design]
---

# Multi-task architecture

The repo is structured like a task-suite (à la insightface: one shared core,
many task "heads", a common benchmark/eval harness), not a single model.

## Shared infrastructure
- **`drone-dsp`** - the `no_std` DSP core every head reuses: windowing, real FFT
  (microfft), magnitude spectrum, spectral features, `bin_to_hz`/`hz_to_bin`.
  This is the "shared backbone."
- **`drone-bench`** - shared eval harness: the `Approach` trait, dataset loader
  (`Sample`, CSV manifest, synth generator, stratified split, k-fold, SNR
  augmentation), metrics (F1/calibrated-F1/ROC-AUC/PR-AUC/Brier + real-time
  factor), and JSON output. Other heads reuse `drone_bench::util::spectra` and
  `drone_bench::dataset`.

## Task heads
- **Detection** (binary drone / no-drone) - `drone-detect` + the 12 approaches in
  `drone-bench/src/approaches/` (template, band_ratio, hps, spectral_gate,
  cepstrum, mfcc_lr, mfcc_mlp, gtcc_lr, feature_fusion, spectrogram_template,
  envelope_periodicity, fusion-ensemble). Beats published CNN baselines; see
  [benchmark-results](benchmark-results.md).
- **Direction of Arrival** - `drone-doa`: GCC-PHAT TDOA + ULA geometry → azimuth,
  with a propagation simulator and an angular-error benchmark.
- **Type ID** (multiclass) - `drone-id`: MFCC + multinomial logistic over drone
  types; per-class F1 + confusion matrix.
- **Property inference** - `drone-freq`: blade-pass fundamental / RPM estimation
  (HPS + cepstrum + autocorrelation fusion), benchmarked for f0 accuracy.
- **Robustness** - SNR-sweep degradation curves over all detectors
  (`benchmarks/robustness.py` + `--snr`), the "stress test earns its keep" axis.
- **Vendor/brand ID** - `drone-vendor`: MFCC + softmax over many brands (real
  32-brand macro-F1 0.93, synthetic 12-brand).
- **Distance** - `drone-range`: air-absorption-tilt + level features -> ridge
  regression / range bins (sim MAE 12.6 m; hardest head; no_std core).
- **Blind source separation** - `drone-bss`: FastICA preprocessing layer that
  rescues detection on masked multi-UAV / ego-noise scenes (56 dB SIR;
  hps recall 17->100% when masked).
- **Upstream baseline** - `drone-cnn`: faithful mel-CNN (candle); perfect in-dist,
  collapses on unseen drones (the honest head-to-head we win).

## Deployment spectrum (same detector, four tiers)
- **MCU** - `drone-edge` (no_std lib) + `drone-firmware` (real esp32-C6 esp-hal,
  links riscv32imac, ~66 KB).
- **Phone** - `drone-mobile` (C ABI / JNI cdylib, Android/iOS via cargo-ndk).
- **Desktop** - `drone-live` (cpal mic capture + hardware probe + alert).
- **Server** - `drone-bench` harness + all the analysis bins.
- **Field data loop** - `drone-live record` -> `fieldeval` -> the honest held-out
  number (see `FIELD_PROTOCOL.md`).

## Conventions that make it cohere
- Every head emits **machine-readable JSON** into `benchmarks/results/`, consumed
  by Python/matplotlib plotters in `benchmarks/`.
- Cheap, deployable methods stay `no_std`-portable (the edge goal); heavy/host
  concerns live in the `*-bench`/bin layers.
- Each head is benchmarked on real data where it exists, with a synthetic
  fallback so the whole suite always runs end-to-end (docker-first).

## Why this shape
It lets many approaches/properties be added independently (one file or one crate
each) and compared apples-to-apples on shared infra - the same reason
insightface shares a backbone across detection/recognition/alignment/attributes.
See decision [0007](../decisions/0007-multi-task-suite.md).
