# Acoustic Drone Detection

[![CI](https://github.com/JoulesSpace/acoustic-drone-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/JoulesSpace/acoustic-drone-detection/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Rust 2021](https://img.shields.io/badge/rust-2021-orange.svg)](https://www.rust-lang.org/)
[![no_std core](https://img.shields.io/badge/core-no__std-green.svg)](crates/drone-dsp)
[![edge: esp32 / riscv32](https://img.shields.io/badge/edge-esp32%20%2F%20riscv32-8a2be2.svg)](crates/drone-edge)

**A Rust DSP and machine-learning toolbox for detecting drones from sound.** A
shared `no_std` signal-processing backbone feeds many task heads - detection,
direction-of-arrival, type and vendor ID, blade-pass / RPM, range, and source
separation - each benchmarked head-to-head on real data and ready to lower onto
esp32-class edge hardware. Built with an emphasis on **honest evaluation**:
leakage-proof, cross-dataset, and hard-negative tests over the optimistic
in-distribution numbers most of the field reports.

![Acoustic drone detection - signal chain & physics](assets/signal_chain.png)

<sub>The full signal chain, with this project's real parameters: rotor acoustics → pressure wave → sampling (Nyquist) → ADC quantization → STFT blade-pass comb → detection pipeline → hardware tiers. Regenerate with `docker compose run --rm --entrypoint python plot scripts/infographic.py` ([`scripts/infographic.py`](scripts/infographic.py)).</sub>

**State of the project:** detection works. We benchmarked a range of detection
algorithms and models across hardware tiers and bundled everything into an
extensible Rust crate toolbox, driven from a CLI and lowerable onto esp32-S3 or
other edge hardware. Heads for drone situational attributes (`distance`,
`elevation`, `speed`) and hardware attributes (`type`, `vendor`, `rotor_count`,
`drone_health`, `drone_weight`) are wired in and ready to be performance-tuned
and extended.

**Notable mention** worth checking out: [batear-io/batear](https://github.com/batear-io/batear) for simple drone detection on esp32-S3+mic.

## TOC

1. [Scope](#1-scope)
2. [On Audio and Sound](#2-on-audio-and-sound-human-reasoning) (human reasoning)
3. [Problem Layers](#3-problem-layers-human-reasoning) (human reasoning)
4. [Possible Approaches](#4-possible-approaches-human-reasoning) (human reasoning)
5. [Constraints of v0.1.0](#5-constraints-of-this-projects-first-iteration-v010) (human reasoning)
6. [Implementation](#6-implementation-v010)
7. [Contributing](#7-contributing)
8. [License](#8-license)

## 1. Scope

Design an acoustic drone detection pipeline, and validate the ideas in
simulation. The questions this project answers:

- Detection. What makes a drone signature distinguishable from background sound? What features or representations would you feed a model, and how would you know it's actually working?
- Direction of Arrival. If you used multiple microphones, how would you estimate where the drone is? What does array geometry buy you, and what are the trade-offs?
- Robustness. Real deployments are noisy - literally. Wind, rain, overlapping sources, varying drone types. How would you stress-test your approach? This is where simulation earns its keep: you control what you throw at it.
- System Design. What would a real deployment look like? How many microphones, in what configuration, at what sample rate? What detection range would you expect and why? What are the fundamental physical limits?

## 2. On Audio and Sound (Human reasoning)

**Sound** is the pressure of a material (mostly air) fluctuating over time. Sound travels at `355 m/s` in air at 40 degrees Celsius and `343 m/s` in air at 20 degrees Celsius, so environmental conditions are relevant to detection.

**Microphones** come in different types, based on the physical quantity that changes with pressure:
  + magnetic (dynamic) microphones measure the vibration of air as it moves the microphone head up and down; they are omnidirectional
  + laser microphones measure the vibration (pressure) of an object the laser points at, interferometrically; they are unidirectional
  + other types exist: the gyroscopes in phones can sample audio at a low rate, and smart materials like piezo crystals turn pressure on a surface into electrical impulses

**Sample frequency** is how many times per second a microphone measures this pressure at one point.
  + old phones like a Nokia sampled around 8k measurements per second
  + a common rate today is `44.1 kHz`, which most microphones and hardware support
  + higher rates exist for studio or specialised equipment

**Audio** is how the sound is stored by digital devices:
  + an ADC (Analog-Digital-Converter) converts electrical pulses from a microphone to a binary signal / value
  + the resolution of this conversion depends on the ADC: older ones gave `8 bit`, newer ones `24 bit`, `32 bit`, or better
  + audio is usually stored compressed, since a raw `.wav` file (a pickled array of samples) takes a lot of storage
  + audio compression is lossy: humans do not need the whole spectrum to, for example, recognize a voice. Many codecs exist (e.g. those implemented in the `ffmpeg` project); modern ones include `.mp3`, `.m4a`, and `.opus` (used by WhatsApp). These differences matter for data quality: a lossy codec can erase the cues needed to predict finer properties of a drone or the situation.

## 3. Problem Layers (Human reasoning)

Acoustic drone detection is not one algorithm; it is a stack of engineering
problems, each constraining the ones around it. **Detection, localization,
tracking, and type-ID are not separate layers** - they are different *outputs*
of one inference layer asked of the same signal.

1. **Requirements & metrics** - defines what "good" means: required outputs,
   detection range, environments, threat model, and the error-cost asymmetry. A
   missed drone usually costs more than a false alarm, so detection leans toward
   recall - but the operational metric is *false-alarms-per-hour at a required
   recall*, not raw false-positive rate (drones are rare, so base rate makes a
   flat FPR misleading).
2. **Sensors** - mic technology (MEMS, condenser, piezoelectric, fiber-optic),
   count, geometry, directionality, frequency/dynamic range, weatherproofing.
   Sets what is physically observable.
3. **Signal acquisition (front-end)** - what the hardware already did to the
   signal before you see it: ADC characteristics, automatic gain control,
   on-device denoise, beamforming, codec, sample rate / bit depth. Commercial
   mics and phones alter the signal in ways that can erase drone cues.
4. **Compute** - where inference runs (MCU, edge node, server, cloud), with its
   latency, power, and memory budgets. Directly limits algorithm complexity.
5. **Data** - datasets and their diversity (drones, distances, weather,
   terrain), labeling, augmentation, synthetic data, and field/continual
   collection. Usually the dominant uncertainty; cross-environment
   generalization is the hard part.
6. **Signal processing / representation** - denoise, framing/windowing, and the
   representation itself: FFT/STFT, mel, MFCC, cepstral, harmonic features.
   Blind source separation (ICA/IVA) lives here when several drones or strong
   interferers must be separated.
7. **Inference** - one chain, several outputs: detection, type, direction-of-
   arrival, tracking, RPM / distance / health. The design choice is which
   outputs are realistically achievable from the available signal.
8. **System architecture** - beyond a single node: multi-node ("swarm")
   cooperation and synchronization, plus cross-modal fusion (RF, radar, EO/IR,
   lidar). Acoustic often serves as the cheap, non-line-of-sight gate that
   *cues* heavier sensors.
9. **Validation & robustness** - performance vs SNR, weather, unseen drones, and
   hard negatives; the dominant failure modes; and honest cross-dataset
   evaluation.
10. **Deployment & operations** - power, enclosure, networking/outages,
    monitoring, and lifecycle (drift detection, field-data pipeline,
    retraining), plus security (evasion, spoofing, jamming, decoys) and
    cost/scalability per node.

This repo currently lives mostly in layers 6-9 (signal processing through
validation), with `drone-edge` / `drone-live` reaching into 3-4 (acquisition
and edge compute). Real sensor front-ends and full deployment/operations are the
open ends.

## 4. Possible Approaches (Human reasoning)

There is no single "drone-detection algorithm". We benchmark several methods
against the same data and compare accuracy, robustness, compute, and
deployability - the goal is the *simplest approach that stays reliable under
realistic conditions*, not the most complex model.

- **Detection from frequency structure** - propellers produce harmonic peaks.
  `audio -> FFT/STFT -> representation -> detector`. Detectors range from band
  energy, harmonic-product-spectrum, cepstrum, and spectrogram templates to
  MFCC + classifier (logistic / random forest / SVM), CNNs on spectrograms, and
  audio foundation models (e.g. YAMNet). Open question: do cheap DSP methods
  suffice before reaching for ML? (`drone-bench` benchmarks 12 such approaches.)
- **Detection from periodicity** - rotor rotation makes drone audio strongly
  periodic, so autocorrelation, cepstrum, amplitude-modulation, and blade-pass
  estimation can detect even when individual frequency peaks are buried in
  noise. (`drone-freq`, `envelope_periodicity`.)
- **Drone attribute estimation** - beyond yes/no: `type`, `rotor_count`, `rpm`,
  `direction`, `elevation_angle`, `distance`, `speed`, `motor_health`,
  `rotor_damage`. Some are correlated and estimable jointly; distance-range
  classification is shown feasible in the literature. (`drone-id`, `drone-freq`.)
- **Direction of arrival & localization** - multiple synchronized mics enable
  TDOA (GCC-PHAT), beamforming, and super-resolution (MUSIC / ESPRIT) plus
  triangulation. Questions: mic count, spacing (`d < c / 2*f_max` to avoid
  spatial aliasing), achievable angular accuracy, and degradation with noise.
  (`drone-doa`, simulated; real-world DoA is ~6-42 deg, not sub-degree.)
- **Robustness** - wind, rain, traffic, construction, aircraft, birds, multiple
  drones, unseen models. `drone + background + synthetic noise = test scenario`,
  scored as a function of SNR. The real confusers are other rotary/harmonic
  machines (chainsaw, engine, helicopter). (`drone-bench --snr`, `xeval`, plus a
  hard-negative suite.) Software noise-cancellation can help where the noise is
  periodic or white (subtract the predictable part, "rausrechnen"); and it is
  worth talking to people who detect unique events in noisy real-time data (e.g.
  [hydrop-systems](https://hydrop-systems.com/), [kinemic](https://kinemic.com/de/)).
- **Sensor design** - single mic (cheap, detection only) vs array (direction,
  beamforming, better SNR) vs directional / fiber-optic / laser mics (higher
  SNR, smaller search space). Mixed setups are future work.
- **Hardware exploration** - ESP32-S3 / -P4, Arduino / AVR8-class, embedded
  Linux, Jetson, and FPGA (we have one, ~45k LUTs). Map the accuracy / latency /
  power / cost tradeoff, and the field enclosure (weatherproofing, and depending
  on the threat, EMP/laser hardening). (`drone-edge` is a `no_std` cross-build
  proof.)
- **Data strategy** - public sets (DADS, Al-Emadi, Kaggle, HuggingFace; e.g.
  [saraalemadi/DroneAudioDataset](https://share.google/3r4LoZTEbmyATlB56) and the
  [Drone Sound Detection set](https://share.google/rMNhLehvEraoAqpfG)),
  self-recorded flights, and augmentation (added noise, SNR levels, codec
  degradation, sample-rate reduction, reverberation). The key uncertainty is how
  well a model trained on one dataset transfers to entirely different
  environments and drones.
- **Multi-sensor fusion** - acoustic + RF + radar + EO/IR + thermal. Acoustic is
  the low-power, non-line-of-sight modality that cues the others (fielded by
  Fraunhofer IDMT; RF-assisted variants by Toma et al.).
- **Validation** - the same metrics for every approach: precision, recall, F1,
  ROC-AUC, PR-AUC, false-alarm rate (detection); angular and position error
  (localization); latency, power, memory, range (deployment).

## 5. Constraints of this projects first iteration (v0.1.0)

- Only one real drone available for testing.
- Limited hardware: esp32-S3, -C6, and -P4 modules, plus a few Arduino boards (notably the 4 GB-RAM one).
- Hardly any specialised microphones on hand (one camera mic, otherwise phone and laptop mics).
- Limited AI budget of 50 EUR (the Claude weekly limit).
- Limited dev time: a single afternoon for v0.1.0.

## 6. Implementation (v0.1.0)

Built in Rust for fast, typed iteration on real DSP - and so the core can later
be lowered onto edge hardware (esp32 xtensa / riscv). It's structured as a
multi-task suite (insightface-style: one shared DSP backbone + a common eval
harness + many task "heads"). Crates live under `crates/` (no workspace yet, by
design):

- **`drone-dsp`** - `no_std` DSP backbone reused by every head: Hann windowing,
  real FFT (`microfft`), magnitude spectrum, spectral features. Math via `libm`,
  so it builds bare-metal.
- **`drone-detect`** - `no_std` heuristic detector (energy-in-band + dominant
  tone). The transparent baseline.
- **`drone-cli`** - host binary `drone`: `synth` a test signal, `analyze` WAVs.
- **`drone-bench`** - shared eval harness: pluggable `Approach` trait, dataset
  loader (CSV/synth, stratified split, k-fold, SNR augmentation), metrics
  (F1, calibrated-F1, ROC-AUC, PR-AUC, Brier, real-time factor), JSON output.
  Hosts **12** detection approaches.
- **`drone-doa`** - direction-of-arrival: GCC-PHAT TDOA + ULA geometry → azimuth,
  with a propagation simulator and an angular-error benchmark (`no_std` core).
- **`drone-id`** - multiclass drone-**type** recognition (MFCC + multinomial
  logistic) with per-class F1 + confusion matrix.
- **`drone-freq`** - blade-pass-frequency / RPM estimation (HPS + cepstrum +
  autocorrelation fusion) - an inferrable drone property.

### Capabilities at a glance (real-data results)

| task | crate | headline result | notes |
|---|---|---|---|
| **Detection** (drone vs not) | `drone-bench` | best **F1 1.000 / ROC-AUC 1.000** (`feature_fusion`); 8/12 beat CNN baselines | all run 90-2400× real-time |
| **Direction of arrival** | `drone-doa` | **RMSE 0.88°** @20 dB (±60°), 2.8° @10 dB | 4-mic ULA, simulated |
| **Type ID** (bebop/membo/unknown) | `drone-id` | **macro-F1 0.86** on Al-Emadi multiclass | linear softmax; honest |
| **Blade-pass freq / RPM** | `drone-freq` | synth **f0 MAE ~1 Hz, 0% octave error** | real DADS drones cluster ~230 Hz |
| **Robustness** | `drone-bench --snr` | learned methods hold ROC-AUC >0.95 to **−10 dB**; naive baselines collapse | see `benchmarks/plots/robustness_*.png` |

### Detection approaches & benchmark

Twelve approaches are implemented and benchmarked head-to-head (each emits a
confidence in `[0,1]`, so they're comparable via ROC/PR). On a real
[DADS](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples)
subset (300 + 300 clips, 50/50 split); `F1*` = best-threshold (calibrated) F1,
`×RT` = times faster than real time:

| approach | F1 | F1* | ROC-AUC | ×RT |
|---|---|---|---|---|
| `feature_fusion` - fused MFCC+spectral+harmonic+cepstral + logistic | **1.000** | 1.000 | 1.000 | 1300× |
| `mfcc_lr` - MFCC + logistic regression | 0.997 | 0.997 | 1.000 | 2300× |
| `fusion` - logistic stack (ensemble) over the classics | 0.997 | 1.000 | 1.000 | 90× |
| `mfcc_mlp` - MFCC + small MLP | 0.987 | 0.993 | 1.000 | 2400× |
| `gtcc_lr` - gammatone cepstral coeffs + logistic | 0.987 | 0.990 | 1.000 | 1900× |
| `spectral_gate` - flatness/entropy/band-ratio + logistic | 0.977 | 0.986 | 0.998 | 1900× |
| `cepstrum` - cepstral / autocorrelation periodicity | 0.967 | 0.977 | 0.990 | 110× |
| `envelope_periodicity` - AM modulation spectrum | 0.966 | 0.987 | 0.991 | 95× |
| `hps` - harmonic-product-spectrum / comb | 0.949 | 0.967 | 0.987 | 2150× |
| `spectrogram_template` - 2D spectro-temporal template | 0.925 | 0.974 | 0.980 | 2300× |
| `band_ratio` - baseline heuristic | 0.766 | 0.921 | 0.915 | 2450× |
| `template` - cosine vs. averaged drone spectrum | 0.706 | 0.986 | 0.995 | 2370× |

Cheap classical/light methods score very high here - no GPU, all real-time on a
desktop. **⚠ Honesty caveat:** these are *in-distribution* numbers on a random
clip-level split of one dataset (DADS), which very likely has recording-level
**leakage** (short clips from shared source recordings landing in both train and
test), so they are optimistic and **not** an apples-to-apples win over published
CNN baselines. The trustworthy tests - **cross-dataset** and **hard-negative**
(aircraft / car / wind) evaluation - are the current priority; see
[`agent-memory/notes/honest-limitations.md`](agent-memory/notes/honest-limitations.md).
Plots (ROC, PR, cost-vs-quality, robustness) live in
[`benchmarks/plots/`](benchmarks/); methodology in
[`benchmarks/README.md`](benchmarks/README.md).

### Benchmark plots

<table>
<tr>
<td width="50%"><img src="benchmarks/plots/roc.png" alt="ROC curves for all approaches" width="100%"><br><sub><b>ROC</b> &middot; in-distribution, all 12 approaches (labelled with ROC-AUC)</sub></td>
<td width="50%"><img src="benchmarks/plots/robustness_roc.png" alt="ROC-AUC vs SNR" width="100%"><br><sub><b>Robustness</b> &middot; ROC-AUC vs SNR; learned methods hold above 0.95 to -10 dB</sub></td>
</tr>
<tr>
<td width="50%"><img src="benchmarks/plots/cost_quality.png" alt="Inference cost vs F1" width="100%"><br><sub><b>Cost vs quality</b> &middot; inference ms/clip (log x) vs F1</sub></td>
<td width="50%"><img src="benchmarks/plots/ratesweep.png" alt="Sample-rate and bit-depth sweep" width="100%"><br><sub><b>Rate / bit-depth</b> &middot; detection flat from 8 kHz up, robust to 4-bit</sub></td>
</tr>
</table>

Regenerate any of these with `docker compose run --rm plot` (see
[`benchmarks/README.md`](benchmarks/README.md)); the physics + signal-chain
poster above is `scripts/infographic.py`.

### Quick start (Docker-first)

```bash
# Generate a synthetic drone signal and analyze it (writes into ./data)
docker compose run --rm detector synth   --out /data/test.wav --fundamental 120
docker compose run --rm detector analyze --input /data/test.wav

# Fetch a real dataset subset, benchmark all 12 detectors, and plot
docker compose run --rm data --per-class 300
docker compose run --rm bench --data /work/data/dads      # add --kfold 5 or --snr 0
docker compose run --rm plot

# Other task heads (run inside the dev toolchain container)
docker compose run --rm --entrypoint bash dev -c \
  "cargo run -r --manifest-path crates/drone-doa/Cargo.toml --bin doa-bench"   # direction of arrival
docker compose run --rm --entrypoint bash dev -c \
  "cargo run -r --manifest-path crates/drone-id/Cargo.toml -- --synth"          # drone-type ID
docker compose run --rm --entrypoint bash dev -c \
  "cargo run -r --manifest-path crates/drone-freq/Cargo.toml -- --data /work/data/dads"  # blade-pass freq

# Run the full check suite: folderinfo lint, fmt, clippy -D warnings, tests, no_std builds
docker compose run --rm dev
```

> On Git Bash (Windows) prefix docker commands with `MSYS_NO_PATHCONV=1`, or use
> PowerShell - otherwise `/data/...` args get rewritten. See `CLAUDE.md`.

Agent working memory - decisions, insights, domain notes, and session handoffs -
is tracked in [`agent-memory/`](agent-memory/MEMORY.md).

## 7. Contributing

Contributions are welcome. The workflow is:

```
fork -> branch [name]/feat|fix-[short-name] -> PR -> address feedback -> merge
```

Everything builds and tests in Docker (no host Rust toolchain needed); run the
full correctness oracle with `docker compose run --rm dev` before opening a PR.
See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, conventions (no em-dashes,
`no_std`-clean core, a `.folderinfo` per directory), and how the check suite
runs in CI.

## 8. License

This project is licensed under the **GNU Affero General Public License v3.0**
(AGPL-3.0-or-later) - strong copyleft with a network-use clause, so anyone who
runs a modified version, including over a network, must share their source.

```
Acoustic Drone Detection
Copyright (C) 2026 Julia Yukovich

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version. See the LICENSE file for the full text.
```

Datasets retain their original upstream licenses and are not redistributed here;
see [DATA_SOURCES.md](DATA_SOURCES.md). If you use this work, citation metadata
is in [CITATION.cff](CITATION.cff).