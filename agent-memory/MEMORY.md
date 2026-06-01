# MEMORY — agent memory index

This is the **entry point to the project's agent memory**. It *is* the Claude
memory for this repo: tracked in git, read first each session, kept current as
work happens. Start here, then follow the links.

How this store is organized and maintained: [README.md](README.md).

## 🔭 Start here
- [Latest handoff — v0.1.0 scaffold](handoffs/2026-06-01-v0.1.0-scaffold.md) — current state & the next concrete steps

## 📐 Decisions (ADRs)
- [0001 — Rust as the implementation language](decisions/0001-rust-implementation-language.md) — typed, fast DSP that can lower to edge
- [0002 — Crates in `crates/`, no workspace yet](decisions/0002-crates-no-workspace.md) — path deps, by owner request
- [0003 — Three-crate split](decisions/0003-three-crate-split.md) — dsp / detect / cli boundaries
- [0004 — microfft + libm for FFT/math](decisions/0004-microfft-and-libm.md) — one no_std code path, host + bare-metal
- [0005 — Docker-first, two compose services](decisions/0005-docker-first-two-services.md) — detector + dev oracle
- [0006 — Heuristic detector for v0.1.0](decisions/0006-heuristic-detector-v0.1.0.md) — transparent baseline, no ML

## 💡 Insights (gotchas & learnings)
- [Git Bash mangles `/data` docker args](insights/msys-path-mangling.md) — use PowerShell or `MSYS_NO_PATHCONV=1`
- [microfft packs Nyquist into bin 0](insights/microfft-nyquist-packing.md) — we expose `|DC|`, drop Nyquist
- [Slim Rust image lacks rustfmt/clippy](insights/slim-rust-missing-rustfmt-clippy.md) — why `Dockerfile.dev` exists
- [Synthetic signal discriminates cleanly](insights/synthetic-signal-discriminates.md) — verified pipeline result

## 📓 Notes (domain knowledge)
- [DSP conventions](notes/dsp-conventions.md) — frame size, FFT, windowing, frequency math
- [Detection thesis](notes/detection-thesis.md) — why a drone is acoustically distinguishable

## 🗂 Handoffs (session log, newest first)
- [2026-06-01 — v0.1.0 scaffold](handoffs/2026-06-01-v0.1.0-scaffold.md)
