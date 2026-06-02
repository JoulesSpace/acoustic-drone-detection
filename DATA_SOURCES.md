# Data sources

This project trains and benchmarks on **openly-licensed, third-party datasets**.
No proprietary data is redistributed here. Each dataset keeps its own upstream
license; respect those terms (several are research / non-commercial use only).

Stream a balanced subset of the primary dataset into `./data/` with:

```bash
docker compose run --rm data --per-class 300     # writes data/dads/ + labels.csv
```

## 1. Detection - DADS (primary)

- **What:** `geronimobasso/drone-audio-detection-samples` on HuggingFace.
  180,320 clips (163,591 drone / 16,729 no-drone), WAV, mono, 16 kHz, 16-bit,
  6.81 GB. Binary labels (`1` = drone, `0` = no-drone).
- **License:** MIT, no authentication required.
- **Used for:** the head-to-head detection benchmark (`drone-bench`). The class
  imbalance and size mean we stream and cap a balanced subset rather than
  downloading all 6.81 GB; see [`scripts/download_dads.py`](scripts/download_dads.py).
- **Link:** https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples

## 2. Type ID - Al-Emadi DroneAudioDataset

- **What:** `saraalemadi/DroneAudioDataset` (GitHub). `Binary_Drone_Audio/`
  (drone vs unknown) and `Multiclass_Drone_Audio/` (drone **type**: bebop,
  mambo, ...). Negatives are sourced from ESC-50, speech, and silence.
- **License:** research use only.
- **Used for:** multiclass drone-type recognition (`drone-id`, macro-F1 0.86).
- **Link:** https://github.com/saraalemadi/DroneAudioDataset

## 3. Hard negatives / confounders

- **ESC-50** - 2000 environmental clips, 50 classes (CC BY-NC).
  https://github.com/karolpiczak/ESC-50 - the canonical "unknown" source, and a
  supply of hard negatives (engines, machinery) for the robustness suite.
- **UrbanSound8K** - urban sound classes (engine idling, jackhammer) used as
  hard rotary/harmonic negatives against the drone hum.

## 4. Heavier / specialized (optional, later work)

- **DroneAudioSet** `ahlab-drone-project/DroneAudioSet` (HuggingFace, MIT,
  42.6 GB) - rich SNR / throttle labels and multi-mic (up to 8-channel) audio
  for SNR-robustness and array work; stream it.
- **DREGON** (Inria, 8-channel, 44.1 kHz) - localization / ego-noise; manual zip
  links, academic use.

## Notes on honest evaluation

- **Resample to a common 16 kHz** rate (sources mix 16 k / 44.1 k / variable).
- **Commercial-safe licenses:** only DADS and DroneAudioSet (MIT) are clearly
  commercial-friendly; ESC-50 / UrbanSound8K / Al-Emadi / DREGON are
  non-commercial or academic-use.
- **Avoid leakage:** split by recording / drone / environment, not by random
  clip shuffle. A random clip-level split on DADS leaks shared source
  recordings across train and test and inflates results; see
  [`agent-memory/notes/honest-limitations.md`](agent-memory/notes/honest-limitations.md)
  and the cross-dataset / hard-negative evaluations.
