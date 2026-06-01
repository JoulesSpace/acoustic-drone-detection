---
title: DSP conventions
type: note
date: 2026-06-01
tags: [dsp, fft, reference]
---

# DSP conventions

The math the code relies on, written down so it isn't re-derived.

## Frame / FFT

- **Frame size:** 1024 samples (`drone_dsp::FRAME_SIZE`). At 16 kHz → 64 ms
  window, ~15.6 Hz/bin.
- **Real FFT:** `microfft::real::rfft_1024` → `FRAME_SIZE/2 = 512` complex bins,
  DC .. just below Nyquist. (Nyquist packing gotcha:
  [insight](../insights/microfft-nyquist-packing.md).)
- **Window:** periodic Hann, applied in place before the FFT, computed on the
  fly via `libm::cosf` (no static table → embedded-friendly).
- **Overlap:** the host `analyze` loop uses 50% overlap (hop = 512).

## Frequency math

- `bin_to_hz(bin, sr) = bin * sr / FRAME_SIZE`
- `hz_to_bin(hz, sr) = round(hz * FRAME_SIZE / sr)`, clamped to `[0, 511]`.

## Spectral features (`drone-dsp`)

- `dominant_bin` — strongest bin, ignoring DC.
- `total_energy` — Σ magnitude² over all bins (power proxy).
- `band_energy(lo, hi)` — Σ magnitude² within a Hz band.
- `spectral_centroid` — energy-weighted mean frequency (brightness); 0 on
  silence.

## Open DSP work

- Fixed FFT size; no configurable resolution yet.
- Single-frame features only — no temporal smoothing, harmonic-spacing, or SNR
  normalization yet.
- No multi-channel / direction-of-arrival yet (needs ≥2 mics + array geometry).
