# Acoustic-Drone-Detection

Detecting drones accoustically.

## Scope

Design an acoustic drone detection pipeline, and validate your ideas in simulation.

Questions we answer with this porject:

- Detection. What makes a drone signature distinguishable from background sound? What features or representations would you feed a model, and how would you know it's actually working?
- Direction of Arrival. If you used multiple microphones, how would you estimate where the drone is? What does array geometry buy you, and what are the trade-offs?
- Robustness. Real deployments are noisy — literally. Wind, rain, overlapping sources, varying drone types. How would you stress-test your approach? This is where simulation earns its keep: you control what you throw at it.
- System Design. What would a real deployment look like? How many microphones, in what configuration, at what sample rate? What detection range would you expect and why? What are the fundamental physical limits?

## On Audio and Sound (Human reasoning)

**Sound** is the pressure of a material (mostly air) fluctuating over time. Sound has a speed of `355 m/s` m air at 40 degrees celsius and a speed of `343m/s` in air at 20 degrees celsius. So environmental conditions are relevant to detection.

**Microphones** exist in different types, based on different physical quantities that changes with pressure:
  + magnetic microphones measure the vibration of air by it moving the microphone head up and down, they are omnidirectional
  + laser microphones measure the vibration (pressure) of an object that the laser points at, interferometrically, they are unidirectional
  + other types of microphone exist, for example gyroscopes of phones can sample audio at a low amount. also smart materials like piezzo-crystals can be used to turn pressure on a surface to electrical impulses

**Sample frequency** is how many times per second microphones measure this pressure in one point.
  + old phones like a nokia have around 8k measurements per second
  + common sampling rate is `44.1kHZ` which most microphones and hardwares can do today
  + there exist better sampling rates for studio or specialised hardware equipment

**Audio** is how the sound is stored by digital devices:
  + an ADC (Analog-Digital-Converter) converts electrical pulses from a microphone to a binary signal / value
  + the qualitative resolution off this conversion depends on the adc, older ones got `8 bit`, newer ones `24 bit`, `32 bit` or even better quality
  + audio is usually stored compressedly, as storing it as a raw `.wav` file / pickled numpy array takes too much storage.
  + audio compression is lossy, as humans dont need to head all the spectrum to detect voice for example. different audio codecs exist, for example implemented in the `ffmpeg project` (c++). modern codecs include `.mp3`, `.m4a`, `.opus` (whatsapp). it is important to consider these differences for data quality also, as a lossy codec might ruin predictions of more precise properties of a drone or situation.

## Possible Approaches (Human reasoning)

- Detection:
  + Audio is sampled at a rate `f`, keep in mind [Nyquist–Shannon sampling theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem) -> fft / short time fourrier transform -> frequencies histogram which should be characteristic (like for guitars / pianos / fridges) ; drone audio may also be assumed sort of periodic
  + Drone Audio Dataset -> kaggle, gh, huggingfacce (saraalemadi/DroneAudioDataset, GitHub https://share.google/3r4LoZTEbmyATlB56 ; Audio | Drone Sound Detection https://share.google/rMNhLehvEraoAqpfG)
  + Multi-Dataset found on Kaggle that combines multiple drone datasets
  + Broader audio classifier by Google YAMNet (Open-Source)
  + Drone params possibly estimatable (some of which correlated): `drone.type`, `drone.rotor_size`, `drone.distance`, `drone.height`, `drone.speed`,`drone.accelleration`, `drone.type`, `drone.rotor_damage`, `drone.direction`, `drone.elevation_angle`, `drone.motor_health`, `drone.obstacles_inbetween`
- Direction of Arrival
  + Multiple Microphones, at best high sample rate and some distance between them 
  + Triangulation possible
  + Audio Interferometry / Interference of the audio signal
- Robustness
  + ask people from own network who detect unique events in noisy real time data, possibly https://hydrop-systems.com/ or https://kinemic.com/de/
  + detect other events and do software based "noise canceling" in the data, as most noise is cancelable if periodic or just plain white noise or so "rausrrechnen"
  + possibly have a directional mic / laser mic that is more precise and unidirectional and based on the "noisy" mics the rough direction could be estimated
  + speed of sound may vary a bit depending on conditions
- System Design
  + important params are: environmental noise in deployment, other counter-engineering in-field ; as well as the specific dimensions of the hardware, and limitations like `microphone_count`, `microphone_count`, `sample_freq`, `microphone_positions` relative to each other, ...
  + enclosure for durability needed against weather, depending on where its used also against emp, laser or similar
  + edge hardware / is it an `avr8` or `xtensa` esp32 or something like an intel edge ai thing?
  + for maxium performance of audio processing, a fpga or asic chip might be needed to handle the full bit-width and high sample rates at once (we got one at home to test possibly later, has 45k look up tables)

## Constraints of this projects first iteration (v0.1.0)

- Only one real drone for testing
- Limited hardware: esp32 s3, c6, p4 modules, and a ffew arduino boards notably the Q 4gb ram one
- Hardly any specialised microphones here in our appartment (only one camera attached, rest phone and laptop ones)
- Limited AI Budget of 50€ (claude weekly limit)
- Limited dev time, only one afternoon time for v0.1.0

## Implementation (v0.1.0)

Built in Rust for fast, typed iteration on real DSP — and so the core can later
be lowered onto edge hardware (esp32 xtensa / riscv). Crates live under
`crates/` (no workspace yet, by design):

- **`drone-dsp`** — `no_std`-friendly DSP core: Hann windowing, real FFT
  (`microfft`), magnitude spectrum, and spectral features. All math via `libm`,
  so it builds bare-metal.
- **`drone-detect`** — `no_std`-friendly heuristic detector: energy-in-band
  ratio plus a dominant-tone-in-band test. A transparent baseline to beat.
- **`drone-cli`** — host binary `drone` that can `synth` a test signal and
  `analyze` WAV files frame-by-frame.
- **`drone-bench`** — benchmark harness: a pluggable `Approach` trait, dataset
  loader, metrics (F1 / ROC-AUC / PR-AUC / Brier), and JSON output for plotting.

### Detection approaches & benchmark

Six approaches are implemented and benchmarked head-to-head (each emits a
confidence in `[0,1]`, so they're comparable via ROC/PR). On a real
[DADS](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples)
subset (300 + 300 clips, 50/50 split):

| approach | F1 | ROC-AUC | ms/clip |
|---|---|---|---|
| `mfcc_lr` — MFCC + logistic regression | **0.997** | 1.000 | 2.1 |
| `spectral_gate` — flatness/entropy/band-ratio + logistic | 0.977 | 0.998 | 2.7 |
| `cepstrum` — cepstral / autocorrelation periodicity | 0.967 | 0.990 | 45 |
| `hps` — harmonic-product-spectrum / comb | 0.949 | 0.987 | 2.1 |
| `band_ratio` — baseline heuristic | 0.766 | 0.915 | 2.0 |
| `template` — cosine vs. averaged drone spectrum | 0.706 | 0.995 | 2.0 |

Four cheap classical/light methods **meet or beat** published CNN baselines
(≈0.93–0.955 F1) on this binary task. Plots (ROC, PR, cost-vs-quality) live in
[`benchmarks/plots/`](benchmarks/); methodology and caveats (leakage, threshold
calibration) in [`benchmarks/README.md`](benchmarks/README.md). These are
in-distribution subset numbers — see the caveats before quoting them.

### Quick start (Docker-first)

```bash
# Generate a synthetic drone signal and analyze it (writes into ./data)
docker compose run --rm detector synth   --out /data/test.wav --fundamental 120
docker compose run --rm detector analyze --input /data/test.wav

# Fetch a real dataset subset, benchmark all approaches, and plot
docker compose run --rm data --per-class 300
docker compose run --rm bench --data /work/data/dads
docker compose run --rm plot

# Run the full check suite: folderinfo lint, fmt, clippy -D warnings, tests, no_std build
docker compose run --rm dev
```

> On Git Bash (Windows) prefix docker commands with `MSYS_NO_PATHCONV=1`, or use
> PowerShell — otherwise `/data/...` args get rewritten. See `CLAUDE.md`.

Agent working memory — decisions, insights, domain notes, and session handoffs —
is tracked in [`agent-memory/`](agent-memory/MEMORY.md).

## Contributing

Welcome! fork -> branch `[name]/feat|fix-[feat/fix-name]` -> pr -> fix feedback -> get merged

## License

Use in the open only.

> what is the license that makes people need to open source if they modify or use it?

Google says AGPLv3.

let license = "AGPLv3";
