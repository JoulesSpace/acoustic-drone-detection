---
title: Prior-art landscape — acoustic drone detection
type: reference
date: 2026-06-01
tags: [reference, research, datasets, sota, competitors]
---

# Acoustic drone detection — landscape & positioning

Researched 2026-06-01. `[V]` = verified against a primary/strong source this
session; `[~]` = uncertain / single weak source / verify before publishing.

## TL;DR positioning
OSS acoustic drone detection is **shallow & fragmented**: a few low-star Python
notebooks, one MATLAB coursework repo, and one breakout C/ESP32 DSP detector
(**Batear**, ~235★). Datasets are scattered (16 kHz→150 kHz, 1→64 ch, binary→
32-class, mixed/unstated licenses). **Almost nobody publishes honest
cross-dataset generalization** — most report >95% in-distribution that collapses
out-of-domain. **No fast, typed, edge-portable, multi-tier OSS suite with
reproducible cross-dataset eval exists.** That's our gap to own.

## Open-source projects
- **Batear** (github.com/batear-io/batear) — C, **DSP** (Goertzel filters at
  rotor harmonics) on **ESP32-S3 + ICS-43434 MEMS mic (~$15)**. ~235★, active.
  The only real fielded edge acoustic detector — pure DSP because NN models that
  fit a Pi don't fit an ESP32. No formal perf numbers. `[V]`
- **SudarshanChakra** (kbhujbal) — Python, mel-spectrogram→CNN, recall-optimized.
  ~39★. `[V]`
- **DroneAudioDataset** (saraalemadi) — dataset only, ~112★/36 forks. `[V]`
- **uav-audio-detection** (seven-up-purdue) — Jupyter, stale ~2018, ~7★. `[V]`
- **Acoustic-Drone-Detection-System** (shani-pinhas) — MATLAB + Arduino,
  coursework. `[V]`
- **drone-visualization** (mackenzie-jane) — viz/tool for the 32-class set. `[V]`
- GitHub `drone-detection` topic is dominated by **vision (YOLO)** and **RF**;
  only ~2/68 are genuinely acoustic. **No Rust.** None ship a typed lib + CLI +
  reproducible benchmark. `[V]`

## Papers / SOTA (mostly in-domain; honest cross-condition is rare)
- "Empirical Study…" (arXiv 1701.05779): RNN **F1 0.80** > CNN 0.64 > GMM 0.52,
  240 ms input, real urban. `[V]`
- Al-Emadi IWCMC 2019: CRNN ≈ CNN but ~49% faster. `[V]`
- MDPI Sensors 21(15):4953 (2021): GAN-augmented DL. `[V]`
- MDPI Drones 9(6):389 (2025): **ResNet10_CBAM F1 94.3%**, big gains at SNR
  −20/−25/−30 dB. `[~]` (paywalled)
- RF+Acoustic LSTM fusion (PMC11054550): ~91% acc at −10 dB. `[V]`
- MEMS+ML (Acta Acustica 2026): precision 92.3%/recall 86.7% controlled →
  88.5%/83.0% outdoor (≈4 pt FP rise outdoors = confusion is the real failure
  mode). `[V]`
- AUDRON (arXiv 2512.20407): fused signatures for type recognition. `[~]`

## Commercial
- **Squarehead Discovair G2+**: 128-mic array + cam; range **180 m (DJI S1000),
  120 m (Mavic Pro), 90 m (Spark)**; near-zero FP via ML. `[V]`
- **Hall Lidar UDL-64** (2026), **Robin Radar ELVIRA** (acoustic as fusion).
- Class typical best-case **300–500 m**, degrades hard in noise. `[V]`

## Datasets (beyond ours)
- **32-class brand set** (arXiv 2509.04715): 3,200 clips/16,000 s, 32 brand/model
  classes (DJI×15, Syma×6, …); EfficientNet 96.31%. SR/channels/license **[~]**.
- **DroneAudioset** (arXiv 2510.15383, ahlab): **23.5 h**, 17 mics (two 8-ch
  circular arrays 25/50 cm + center), 60 Hz–20 kHz, **SNR −57.2..−2.5 dB**,
  **MIT**. Great for DoA + low-SNR. `[V]`
- **RWDA** (IEEE-DataPort): **32-ch @48 kHz, 64-ch @150 kHz**, DJI Air 3S/Mini,
  alt 5–120 m, urban-train/mountain-test; login required. `[V]`
- **AudioSet** aircraft/helicopter/propeller/aircraft-engine subsets for hard
  negatives (verify /m/ IDs in ontology.json). `[~]`
- DREGON: **8-ch @44.1 kHz**, UAV-embedded, Vicon DoA ground truth; academic-only.

## Hard negatives (most confusable)
Helicopters, prop airplanes, motorbikes, lawnmowers/garden equipment,
construction, RC vehicles, HVAC/engine hum, wind, birdsong, insects. All share
broadband rotor/engine noise + harmonic blade tones; drones differ by **higher
BPF + rapidly varying RPM** (smeared at low SNR/wind). Literature separates via
harmonic/BPF structure, realistic-negative augmentation, explicit hard-negative
classes, and attention/feature-fusion nets.

## Edge / real-time
- Batear: Goertzel on ESP32-S3, off-grid, in SRAM. `[V]`
- Sound detection ~50 ms latency achievable. `[V]`
- ESP32 TinyML speech (analog): int8 quant −37% RAM/−27% ROM. `[V]`
- **No published TinyML drone *classifier* with RAM/FLOPs/latency disclosed** —
  open territory. `[V]`

## Opportunities a Rust suite can own
1. Fast, typed, memory-safe core; deterministic latency; `no_std`/edge path.
2. **Honest cross-dataset eval** (train X → test Y, report the drop) — nobody does.
3. **Multi-tier** ESP32→phone→server (Tier-0 BPF gate → Tier-1 light quantized →
   Tier-2 heavy) — unclaimed in OSS.
4. Published **edge ML RAM/FLOPs/latency** numbers.
5. Curated **hard-negative pack** + confusion matrices.
6. **Dataset harmonization** loader across the 16 kHz↔150 kHz / 1↔64-ch zoo.
7. DoA on **commodity small arrays** (bridge Batear ↔ Squarehead).
8. Clear license + dataset-license provenance.

**⚠ Eval-design consequence:** DADS is a *merge superset* containing Al-Emadi,
DREGON, ESC-50, UrbanSound8K, etc. → testing on those is NOT held-out. See
[[dads-is-a-merge-superset]]. A truly disjoint cross-dataset test needs
DroneAudioset (MIT) or the 32-class set, which are NOT in DADS.
