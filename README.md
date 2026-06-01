# Acoustic-Drone-Detection

Detecting drones accoustically.

## Scope

Design an acoustic drone detection pipeline, and validate your ideas in simulation.

Questions we answer with this porject:

- Detection. What makes a drone signature distinguishable from background sound? What features or representations would you feed a model, and how would you know it's actually working?
- Direction of Arrival. If you used multiple microphones, how would you estimate where the drone is? What does array geometry buy you, and what are the trade-offs?
- Robustness. Real deployments are noisy — literally. Wind, rain, overlapping sources, varying drone types. How would you stress-test your approach? This is where simulation earns its keep: you control what you throw at it.
- System Design. What would a real deployment look like? How many microphones, in what configuration, at what sample rate? What detection range would you expect and why? What are the fundamental physical limits?

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
- System Design
  + important params are: environmental noise in deployment, other counter-engineering in-field ; as well as the specific dimensions of the hardware, and limitations like `microphone_count`, `microphone_count`, `sample_freq`, `microphone_positions` relative to each other, ...
  + enclosure for durability needed against weather, depending on where its used also against emp, laser or similar
  + edge hardware / is it an `avr8` or `xtensa` esp32 or something like an intel edge ai thing?

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

### Quick start (Docker-first)

```bash
# Generate a synthetic drone signal and analyze it (writes into ./data)
docker compose run --rm detector synth   --out /data/test.wav --fundamental 120
docker compose run --rm detector analyze --input /data/test.wav

# Run the full check suite: fmt, clippy -D warnings, tests, no_std build
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
