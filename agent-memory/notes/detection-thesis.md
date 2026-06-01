---
title: Detection thesis
type: note
date: 2026-06-01
tags: [detection, domain]
---

# Detection thesis - why a drone is acoustically distinguishable

A multirotor's sound is **quasi-periodic**: each motor spins at a roughly steady
rate, and the blades produce a blade-pass fundamental (tens to low hundreds of
Hz) plus a stack of harmonics extending into the low kHz. Background noise
(wind, traffic, speech) is broadband or has different harmonic structure.

So the signal concentrates energy in a **band** (~100–4000 Hz here) and in
**tonal peaks**. The v0.1.0 detector ([0006](../decisions/0006-heuristic-detector-v0.1.0.md)):

> `band_energy / total_energy >= 0.5` **and** the dominant bin sits in the band.

Cheap, interpretable, runs on a microcontroller.

## Synthetic test signal (`drone synth`)

Fundamental `f` with 6 harmonics (amplitude `0.5/k`), a slow 8 Hz amplitude
modulation (rotor wobble), and light broadband noise (xorshift32 PRNG,
deterministic). A `--plain` sine is the negative control. This is a controllable
stand-in, **not** a real drone - it tests the plumbing, not robustness. Result:
[insight](../insights/synthetic-signal-discriminates.md).

## What real robustness will need

Temporal smoothing across frames; an explicit harmonic-spacing feature; SNR
normalization; stress tests against wind/noise mixes and overlapping sources.
Datasets to wire in: saraalemadi/DroneAudioDataset, Kaggle multi-datasets,
YAMNet as a broad baseline (see `README.md`).
