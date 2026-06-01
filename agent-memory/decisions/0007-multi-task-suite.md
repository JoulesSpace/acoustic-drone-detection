---
title: Multi-task suite over shared DSP/eval infra
type: decision
date: 2026-06-01
status: accepted
tags: [architecture, scope]
---

# 0007 - Multi-task suite over shared infra (insightface-style)

**Decision:** Grow the repo as a *suite of task heads* over shared
infrastructure rather than a single detector: `drone-dsp` (DSP backbone) and
`drone-bench` (eval harness) are shared; detection, direction-of-arrival
(`drone-doa`), type ID (`drone-id`), and property inference (`drone-freq`) are
separate heads, each benchmarked and emitting JSON for the common plotters.

**Why:** The README scope spans detection, DoA, robustness, and system design -
plus many inferrable drone properties. A shared-backbone/many-heads layout (the
insightface model) lets each capability be added and benchmarked independently
on common infra, and matches the edge goal (the cheap heads stay `no_std`).

**Consequences:** New crates depend on `drone-dsp`/`drone-bench` by path (still
no workspace - [0002](0002-crates-no-workspace.md)). Each head owns its own
metric where the binary-classification metrics don't fit (angular error for DoA,
macro-F1/confusion for ID, f0 MAE for frequency).

**Map & rationale:** [architecture note](../notes/architecture.md).
