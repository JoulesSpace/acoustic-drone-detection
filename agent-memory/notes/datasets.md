---
title: Drone audio datasets
type: note
date: 2026-06-01
tags: [datasets, benchmark, reference]
---

# Drone audio datasets (researched 2026-06-01)

Ranked by ease of automated, auth-free download.

## Primary (positives + binary baseline)

### DADS - `geronimobasso/drone-audio-detection-samples` ⭐ use this first
- HuggingFace, **MIT**, **no auth**. 180,320 clips (163,591 drone / 16,729 no-drone), 6.81 GB.
- **WAV, mono, 16 kHz, 16-bit** - matches our `FRAME_SIZE=1024` design directly.
- Binary labels: `label` 1 = drone, 0 = no-drone. Schema: `audio`, `audioduration (s)`, `label`.
- Download: `from datasets import load_dataset; load_dataset("geronimobasso/drone-audio-detection-samples")`
  or `git clone https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples`.
- 6.81 GB is large → **download a subset** (stream + cap per class). Our script does this.

### DroneAudioDataset - Al-Emadi (canonical academic baseline)
- `git clone https://github.com/saraalemadi/DroneAudioDataset.git` (no auth, ~tens MB).
- `Binary_Drone_Audio/` (drone vs unknown) and `Multiclass_Drone_Audio/` (drone **type**:
  bebop, mambo, …). Negatives sourced from ESC-50 + speech + silence. Research-use only.

## Negatives / hard confounders
- **ESC-50** - `git clone https://github.com/karolpiczak/ESC-50.git` (2000 clips, 50 env classes,
  5 s, 44.1 kHz, CC BY-NC). Easiest negatives; exact source of Al-Emadi's "unknown".
- **UrbanSound8K** - via `soundata` (engine_idling/jackhammer = hard negatives vs drone hum).

## Heavy / specialized (optional, later)
- **DroneAudioSet** `ahlab-drone-project/DroneAudioSet` (HF, MIT, 42.6 GB, rich SNR/throttle
  labels, multi-mic incl. 8-ch) - for SNR-robustness work; stream it.
- **DREGON** (Inria, 8-ch 44.1 kHz, localization/egonoise) - manual zip links, academic-use.

## Gotchas
- Sample rates differ (16k vs 44.1k vs variable) → **resample to 16 kHz** common rate.
- Licenses: only DADS / DroneAudioSet (MIT) and DroneDetectionThesis (CC0) are
  commercial-safe; ESC-50 / UrbanSound8K / Al-Emadi / DREGON are non-commercial/academic.
- **Avoid leakage:** split by recording/drone/environment, not random clip shuffle.

See download tooling in [`scripts/`](../../scripts) and approach plan in
[approaches-survey.md](approaches-survey.md).
