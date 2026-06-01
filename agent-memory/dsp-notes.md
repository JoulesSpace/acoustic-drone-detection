# DSP & domain notes

Signal-processing knowledge and gotchas for this project. The math the code
relies on, written down so it isn't re-derived each time.

## Frame / FFT conventions

- **Frame size:** 1024 samples (`drone_dsp::FRAME_SIZE`). At 16 kHz that's a
  64 ms window with ~15.6 Hz/bin resolution.
- **Real FFT:** `microfft::real::rfft_1024` returns `FRAME_SIZE/2 = 512` complex
  bins covering DC .. just below Nyquist.
- **Nyquist packing gotcha:** microfft stores the real DC term in `bins[0].re`
  and the real Nyquist term in `bins[0].im`. We expose `|DC|` as `spectrum[0]`
  and **drop the Nyquist magnitude**. Fine for drone work (energy is well below
  Nyquist); revisit if you ever need the top bin.
- **Window:** periodic Hann, applied in place before the FFT, computed on the
  fly via `libm::cosf` (no static table → embedded-friendly).
- **Overlap:** the host `analyze` loop uses 50% overlap (hop = 512).

## Frequency math

- `bin_to_hz(bin, sr) = bin * sr / FRAME_SIZE`
- `hz_to_bin(hz, sr) = round(hz * FRAME_SIZE / sr)`, clamped to `[0, 511]`.

## Why a drone is distinguishable (the detection thesis)

A multirotor's sound is **quasi-periodic**: each motor spins at a roughly steady
rate, and the blades produce a blade-pass fundamental (tens to low hundreds of
Hz) plus a stack of harmonics extending into the low kHz. Background noise
(wind, traffic, speech) is broadband or has different harmonic structure. So:

- Energy concentrates in a **band** (~100–4000 Hz here) and in **tonal peaks**.
- v0.1.0 detector = `band_energy / total_energy >= 0.5` **and** the dominant
  bin sits inside the band. Cheap, interpretable, runs on a microcontroller.

## Synthetic test signal (`drone synth`)

Generates fundamental `f` with 6 harmonics (amplitude `0.5/k`), a slow 8 Hz
amplitude modulation (rotor wobble), and light broadband noise (xorshift32 PRNG,
deterministic). This is *not* a real drone — it's a controllable stand-in so the
pipeline is testable without recordings. A `--plain` sine is the negative
control.

**Verified 2026-06-01:** synth drone (f=120 Hz) → 100% drone frames, dominant
~125 Hz, band ratio ~0.995. Plain 60 Hz tone → 0% drone frames. Discrimination
works on synthetic data.

## Known limitations / next steps for the DSP

- Fixed FFT size; no configurable resolution yet.
- Detector is a single-frame heuristic — no temporal smoothing, no harmonic
  spacing check, no SNR normalization. Real noisy audio will need those.
- No direction-of-arrival yet (needs multi-channel input + array geometry).
- No real dataset wired in (saraalemadi/DroneAudioDataset etc. — see README).
