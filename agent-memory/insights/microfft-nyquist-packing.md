---
title: microfft packs Nyquist into bin 0
type: insight
date: 2026-06-01
tags: [dsp, fft, gotcha]
---

# microfft packs the Nyquist term into bin 0

`microfft::real::rfft_1024` returns `FRAME_SIZE/2 = 512` complex bins covering
DC up to just below Nyquist. To fit the full real-FFT result into that many
bins it **packs the two purely-real terms together**: `bins[0].re` holds DC and
`bins[0].im` holds the **Nyquist** term (not an imaginary DC component).

**What we do:** expose `|DC|` as `spectrum[0]` (`fabsf(bins[0].re)`) and **drop
the Nyquist magnitude**. Drone energy lives well below Nyquist, so this is fine.

**If you ever need the top bin:** recover Nyquist magnitude as
`fabsf(bins[0].im)` and extend the spectrum to `NUM_BINS + 1`.

See [`crates/drone-dsp/src/fft.rs`](../../crates/drone-dsp/src/fft.rs) and
decision [0004](../decisions/0004-microfft-and-libm.md).
