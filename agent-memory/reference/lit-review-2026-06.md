---
title: Literature corroboration (4-paper read, Jun 2026)
type: reference
date: 2026-06-02
tags: [reference, research, validation, sota]
---

# Literature corroboration - 4 papers (Kang review + 3 corroborating)

External validation of our architecture, honest-eval thesis, and calibration band.

## Kang et al. 2025 (AIP Advances review) - the centerpiece
- Layering: **acquisition (sensors) -> preprocessing -> blind source separation
  -> feature extraction -> recognition -> localization -> challenges**. This
  matches our shared-backbone + heads decomposition, and treats
  detection/localization as OUTPUTS of one chain (our "inference = outputs"
  reframe). Strong independent validation.
- NEW layer we lacked: **Blind Source Separation (ICA / FastICA / IVA)** between
  preprocessing and features, for multi-UAV separation + heavy-noise demixing.
- Their single-dataset (Bebop/Mambo + ESC, 16 kHz/16-bit, 8:2) ranking:
  classical MLP ~92% / SVM ~89% / KNN ~88% / HMM ~79% (all sub-1 s); deep
  CNN-LSTM ~96.9% (best), CRNN ~96%, Transformer ~96.4% (slowest, 6.84 s), plain
  RNN ~89%. -> corroborates OUR cost_quality story (classical within a few points
  of deep, far faster). In-distribution only; their "future challenges" = our
  gaps (short-range data, no swarm data, no standards, generalization unproven).
- Sensor taxonomy: capacitive (sensitive/low-noise), piezoelectric
  (passive/rugged), **fiber-optic (very high SNR, EMI-immune, long-range, costly
  demod; ties to the README laser-mic idea + DAS 3D tracking)**, MEMS
  (tiny/cheap/mass). Enriches our Sensors layer.
- Near-field vs far-field criterion: `L` vs `D^2/lambda` (array physics nugget).
- **Distance regression IS feasible:** cited 5-50 m dataset, GRU classifies
  distance bins at 94-98% (10/15 m intervals). Validates our distance ambition.
- CITE-WITH-CARE: body says MLP 97.87% but Table III says 91.87% - cite the
  table, flag the discrepancy.

## Calibration band (anchors our honest claims)
- Effective range: Paszkowski 3 m (lab), Ghouli 50-60 m, Fraunhofer 50-200 m,
  Kang's cited 10-100 m, Benyamin tetrahedral (27 cm spacing) 600 m / 99.5%
  (high outlier). **Our ~100 m geometric estimate sits in the realistic middle.**
- Real DoA: Ghouli **6.3 deg**, Toma **42 deg** RMSE (19-mic Zylia 4.5 cm,
  semi-simulated). **Our simulated drone-doa 0.88 deg is best-case;** real small
  arrays are single-digit-to-tens of degrees. (We already say this.)
- Sample rate: Paszkowski records 48 kHz -> LP 6.5 kHz -> decimate /3 to ~16 kHz
  before the CNN -> **independent support for our 16 kHz pipeline.**

## Tiered-confirmation / cueing pattern (2 real citations)
Fraunhofer IDMT and Toma both use acoustic as the cheap, always-on, low-power,
non-line-of-sight **gate that wakes/cues** radar/camera/lidar after contact;
Toma uses RF to refine acoustic localization. This is exactly our acoustic-Tier-0
-> confirm-with-heavier-modality architecture (see MODEL_CARDS tiers). Fraunhofer
press: 50-200 m / 1 s / 360 deg / battery-autonomous (cite as signal, not a
measurement).

## Honesty tells (what NOT to copy)
- **Paszkowski: 0.997 reference -> 0.757 field** (anechoic-chamber training, 516
  images from 10 min of one drone, JPEG-compressed spectrograms fed to the CNN).
  The in-dist->field collapse we keep flagging, independently. Failure mode, not
  a method.
- Toma: recognition 0.957 solid, DoA 42 deg not yet (honest re: small arrays).
- **None of the four does cross-dataset generalization.** Our `heldout32` +
  CNN head-to-head (we beat upstream CNN 5x on unseen drones) remains the
  differentiator the field literature does NOT claim.

## Actionable for us
1. **BSS layer** (FastICA/IVA) for multi-UAV / heavy noise -> new `drone-bss`
   capability + multi-UAV-mixture benchmark (ties to the "multiple drones"
   robustness stress test).
2. Distance regression head (validated feasible) - data-gated.
3. Sensors-layer enrichment (fiber-optic, near/far-field) in docs/infographic.
