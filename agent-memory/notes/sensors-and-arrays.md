---
title: Sensors & arrays - taxonomy, near/far-field, cueing
type: note
date: 2026-06-02
tags: [sensors, arrays, doa, system-design, reference]
---

# Sensors & arrays (the acquisition layer)

Domain knowledge for the Sensors / array layer, distilled from the Kang 2025
review + the corroborating papers (see [lit-review-2026-06](../reference/lit-review-2026-06.md)).
This is the layer BELOW our DSP backbone - it sets what the rest of the suite
can ever see.

## Microphone taxonomy (real tradeoffs)
- **MEMS** (e.g. INMP441, ICS-43434): tiny, cheap (~$1-5), mass-producible, I2S
  digital out. The default for edge/phone/our `drone-firmware`. ~60-70 dB SNR.
- **Capacitive / condenser** (+ an ADC like PCM1864): sensitive, low self-noise,
  studio-grade; the quality tier (Paszkowski's stack: condenser + PCM1864 +
  BeagleBone).
- **Piezoelectric**: passive, rugged, no bias needed; good for harsh/outdoor,
  lower fidelity.
- **Fiber-optic / DAS**: extremely high sensitivity and SNR, immune to EMI,
  long-range; expensive demodulation. Ties to the README's **laser-mic** idea and
  to distributed-acoustic-sensing **3D tracking**. The exotic high-end.

Concrete acquisition stacks seen in the literature: INMP441 (Ghouli);
19-ch Zylia spherical MEMS, 4.5 cm, 69 dB SNR, 48 kHz/24-bit (Toma); condenser +
PCM1864 ADC + BeagleBone Black (Paszkowski); 8-ch arrays at 96 kHz/32-bit
(UaVirBASE). Our pipeline assumes 16 kHz mono (Paszkowski independently decimates
48 -> 16 kHz before the CNN), which is the right floor for blade-pass + low-kHz
motor content.

## Array geometry: near-field vs far-field
- Criterion: a source at distance `L` is **far-field** when `L >> D^2 / lambda`
  (D = array aperture, lambda = wavelength), else **near-field**. Far-field =>
  plane-wave assumption holds (what `drone-doa` GCC-PHAT + ULA assumes); near-field
  needs spherical-wavefront models.
- Aperture vs resolution: bigger aperture `D` -> finer angular resolution but
  spatial-aliasing limit at `d > lambda/2` (inter-mic spacing). Small arrays
  (4.5 cm, Toma) give poor DoA (42 deg RMSE); larger/structured arrays do better
  (Ghouli 6.3 deg; Benyamin 27 cm tetrahedral -> 600 m). Our `drone-doa` sim
  0.88 deg is best-case; real small-array is single-to-tens of degrees.

## Tiered confirmation / cueing (system design)
Acoustic is the cheap, always-on, low-power, **non-line-of-sight** gate that
detects + roughly localizes, then **wakes/cues** heavier modalities
(radar/camera/lidar) for confirmation and fine localization; RF can refine the
acoustic fix (Toma). Two real citations: Fraunhofer IDMT ("hears around corners,
wakes the other sensors"; 50-200 m / 1 s / 360 deg / battery-autonomous) and
Toma (RF-assisted). This is exactly the acoustic-Tier-0 -> confirm-with-heavier
architecture our hardware tiers ([`benchmarks/MODEL_CARDS.md`]) imply: don't make
a single acoustic threshold carry the whole false-alarm burden.

## Implications for the suite
- Our edge tier (`drone-edge`/`drone-firmware`) targets the MEMS reality; the
  quality/array tiers (condenser+ADC, multi-mic) are where DoA + distance get
  better.
- `drone-doa` should state its plane-wave (far-field) assumption and the
  near-field caveat for close drones.
- A multi-mic DoA on a commodity small array (INMP441 x4) bridges hobby (Batear)
  and commercial (Squarehead 128-mic) - an open OSS niche.
