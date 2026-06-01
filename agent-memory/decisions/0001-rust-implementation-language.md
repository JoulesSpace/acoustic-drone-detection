---
title: Rust as the implementation language
type: decision
date: 2026-06-01
status: accepted
tags: [language, edge]
---

# 0001 — Rust as the implementation language

**Decision:** Implement the detection pipeline in Rust rather than Python.

**Why:** Fast, typed iteration on real, performant DSP algorithms — and the same
code can lower onto microcontrollers. Avoids the Python-stub-then-rewrite
detour: we write the performant version once.

**Consequences:** Edge-deployable core (see
[0004](0004-microfft-and-libm.md)); steeper authoring cost than NumPy, paid back
by not rewriting for hardware.
