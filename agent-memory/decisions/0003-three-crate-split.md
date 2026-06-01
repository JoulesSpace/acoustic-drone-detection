---
title: Three-crate split (dsp / detect / cli)
type: decision
date: 2026-06-01
status: accepted
tags: [structure, edge]
---

# 0003 — Three-crate split

**Decision:**
- `drone-dsp` — `no_std`-friendly DSP core (windowing, FFT, spectral features).
- `drone-detect` — `no_std`-friendly heuristic detector on top of the core.
- `drone-cli` — `std` host binary (`drone`) for synth + WAV analysis.

**Why:** Keep the edge-deployable math (`drone-dsp`, `drone-detect`) free of
`std` so it can be reused in esp32/riscv firmware, while quarantining all
host-only concerns (file IO, the CLI, the over-the-whole-signal loop) in the
binary.

**Related:** [0001](0001-rust-implementation-language.md),
[0004](0004-microfft-and-libm.md).
