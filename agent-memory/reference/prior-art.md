---
title: Prior-art landscape - acoustic drone detection
type: reference
date: 2026-06-01
tags: [reference, research, datasets, sota, competitors]
---

# Acoustic drone detection - landscape & positioning

Researched 2026-06-01. `[V]` = verified against a primary/strong source this
session; `[~]` = uncertain / single weak source / verify before publishing.

## TL;DR positioning
OSS acoustic drone detection is **shallow & fragmented**: a few low-star Python
notebooks, one MATLAB coursework repo, and one breakout C/ESP32 DSP detector
(**Batear**, ~235★). Datasets are scattered (16 kHz→150 kHz, 1→64 ch, binary→
32-class, mixed/unstated licenses). **Almost nobody publishes honest
cross-dataset generalization** - most report >95% in-distribution that collapses
out-of-domain. **No fast, typed, edge-portable, multi-tier OSS suite with
reproducible cross-dataset eval exists.** That's our gap to own.

## Open-source projects
- **Batear** (github.com/batear-io/batear) - C, **DSP** (Goertzel filters at
  rotor harmonics) on **ESP32-S3 + ICS-43434 MEMS mic (~$15)**. ~235★, active.
  The only real fielded edge acoustic detector - pure DSP because NN models that
  fit a Pi don't fit an ESP32. No formal perf numbers. `[V]`
- **SudarshanChakra** (kbhujbal) - Python, mel-spectrogram→CNN, recall-optimized.
  ~39★. `[V]`
- **DroneAudioDataset** (saraalemadi) - dataset only, ~112★/36 forks. `[V]`
- **uav-audio-detection** (seven-up-purdue) - Jupyter, stale ~2018, ~7★. `[V]`
- **Acoustic-Drone-Detection-System** (shani-pinhas) - MATLAB + Arduino,
  coursework. `[V]`
- **drone-visualization** (mackenzie-jane) - viz/tool for the 32-class set. `[V]`
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
- **Ghouli (Acta Acustica 2026)** - MEMS (INMP441) 8-mic square array (15 cm),
  ESP32-S3 I2S acquisition + Raspberry Pi 4, Random Forest on MFCC + TDOA/GCC
  localization. precision 92.3%/recall 86.7% controlled → 88.5%/83.0% outdoor;
  real-world DoA **MAE 6.3 deg**, 2D position median **1.2 m**, range **50-60 m**
  (DJI Phantom), latency ~150 ms. Honest split (no same-recording leakage) but
  single drone, no cross-dataset. `[V]`
- **Paszkowski & Gola (Adv. Sci. Tech. Res. J. 2024)** - 4 condenser mics +
  PCM1864 ADC + BeagleBone Black; 48 kHz low-passed at 6.5 kHz and decimated to
  ~16 kHz; spectrogram → small CNN (1.66 M params, 516 images). ~97% on the
  anechoic reference but only **0.757 on the real park drone** (Mavic 2 Pro) - a
  textbook in-distribution-to-field collapse on tiny single-drone data; also
  feeds lossy JPEG spectrograms to the CNN. `[V]`
- **Kang et al. (AIP Advances 15, 120701, 2025)** - comprehensive review; its
  taxonomy (acquisition → preprocessing → BSS → features → recognition →
  localization) mirrors our problem-layers decomposition (README §3) and treats
  detection/localization as outputs of one chain. In-distribution benchmark on a
  Bebop/Mambo+ESC set (16 kHz, 8:2): CNN-LSTM best ~96.9%, CRNN ~96%, Transformer
  ~96.4% (slowest, 6.84 s), classical ML ~88-92% but <1 s. Adds **blind source
  separation** (ICA/FastICA/IVA) as a multi-UAV layer we lacked. (Note: body text
  vs Table III MLP figure is inconsistent: 97.87 vs 91.87.) `[V]`
- **Toma et al. (NATO STO-MP-IST-190)** - RF-assisted acoustic from a flying UAV:
  19-ch Zylia spherical MEMS (4.5 cm, 69 dB SNR, 48 kHz/24-bit) + 4-stage CNN
  fusing acoustic-covariance phase with RF RSS for joint recognition + DoA +
  distance. Recognition acc **0.957**; localization weak (**DoA RMSE ~42 deg**,
  distance ~3 m, semi-simulated RF). Small array → poor spatial resolution
  motivates RF cueing. `[V]`
- AUDRON (arXiv 2512.20407): fused signatures for type recognition. `[~]`

## Commercial
- **Squarehead Discovair G2+**: 128-mic array + cam; range **180 m (DJI S1000),
  120 m (Mavic Pro), 90 m (Spark)**; near-zero FP via ML. `[V]`
- **Hall Lidar UDL-64** (2026), **Robin Radar ELVIRA** (acoustic as fusion).
- **Fraunhofer IDMT** (Oldenburg, 2025): fielded acoustic detection that "hears
  around corners" (non-LOS, forested/built-up), combinable with radar/camera/
  lidar; **50-200 m**, 360 deg, 1 s resolution, battery-autonomous; *cues* other
  sensors after acoustic contact (projects AMBOS / ALADDIN). `[V]`
- Class typical best-case **300-500 m**, degrades hard in noise. `[V]`

**Real-world calibration band (acoustic, single array).** Effective range is
typically **50-200 m** (Ghouli 50-60, Fraunhofer 50-200, most cited works
10-100; the Benyamin tetrahedral array at 27 cm spacing is the 600 m outlier).
Real-world **DoA accuracy is 6-42 deg** depending on aperture (Ghouli 6.3, Toma
42), i.e. our simulated `drone-doa` sub-degree numbers are best-case, not
field-typical.

## Datasets (beyond ours)
- **32-class brand set** (arXiv 2509.04715): 3,200 clips/16,000 s, 32 brand/model
  classes (DJI×15, Syma×6, …); EfficientNet 96.31%. SR/channels/license **[~]**.
- **DroneAudioset** (arXiv 2510.15383, ahlab): **23.5 h**, 17 mics (two 8-ch
  circular arrays 25/50 cm + center), 60 Hz-20 kHz, **SNR −57.2..−2.5 dB**,
  **MIT**. Great for DoA + low-SNR. `[V]`
- **RWDA** (IEEE-DataPort): **32-ch @48 kHz, 64-ch @150 kHz**, DJI Air 3S/Mini,
  alt 5-120 m, urban-train/mountain-test; login required. `[V]`
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
- **No published TinyML drone *classifier* with RAM/FLOPs/latency disclosed** -
  open territory. `[V]`

## Opportunities a Rust suite can own
1. Fast, typed, memory-safe core; deterministic latency; `no_std`/edge path.
2. **Honest cross-dataset eval** (train X → test Y, report the drop) - nobody does.
3. **Multi-tier** ESP32→phone→server (Tier-0 BPF gate → Tier-1 light quantized →
   Tier-2 heavy) - unclaimed in OSS.
4. Published **edge ML RAM/FLOPs/latency** numbers.
5. Curated **hard-negative pack** + confusion matrices.
6. **Dataset harmonization** loader across the 16 kHz↔150 kHz / 1↔64-ch zoo.
7. DoA on **commodity small arrays** (bridge Batear ↔ Squarehead).
8. Clear license + dataset-license provenance.

**⚠ Eval-design consequence:** DADS is a *merge superset* containing Al-Emadi,
DREGON, ESC-50, UrbanSound8K, etc. → testing on those is NOT held-out. See
[[dads-is-a-merge-superset]]. A truly disjoint cross-dataset test needs
DroneAudioset (MIT) or the 32-class set, which are NOT in DADS.
