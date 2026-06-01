---
title: microfft + libm for FFT and math
type: decision
date: 2026-06-01
status: accepted
tags: [dsp, fft, edge, dependencies]
---

# 0004 — microfft + libm for the FFT/math

**Decision:** Use `microfft` (pure-Rust, `no_std`, fixed power-of-two real FFT)
and route all float math through `libm`.

**Why:** `rustfft` pulls in `std`/alloc and is overkill for fixed-size embedded
frames. `microfft::real::rfft_1024` matches our 1024-sample frame exactly.
`libm` gives `sqrtf`/`cosf` without `std`, so **one code path serves host and
bare-metal**.

**Trade-off:** FFT size is compile-time fixed (acceptable for v0.1.0).

**Gotcha it created:** microfft packs the Nyquist term into bin 0 — see
[insight](../insights/microfft-nyquist-packing.md).
