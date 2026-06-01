# CLAUDE.md — operating guide for this repository

This file is **mine to maintain**. Whenever something here goes stale — a path
moves, a command changes, a decision is reversed — I update it in the same
change that caused the drift. A stale CLAUDE.md is a bug.

## What this project is

Acoustic drone detection. Detect drones from sound: FFT/STFT → spectral
features → detection, plus direction-of-arrival and robustness work later. See
`README.md` for the problem framing and `agent-memory/` for the running record
of how and why things are built.

## Hard rules (do not violate)

1. **No "assisted by Claude" / `Co-Authored-By` trailers in commits.** The user
   owns this code. Commit as the configured git author, nothing more.
2. **Semantic / conventional commits.** `type(scope): summary`
   (`feat`, `fix`, `docs`, `build`, `chore`, `refactor`, `test`, `perf`).
   Commit in logical, in-between chunks rather than one giant commit.
3. **Maintain `agent-memory/`.** It is the tracked memory directory: decisions,
   DSP/domain notes, and handoffs live there. Update it as work happens, not
   after. See `agent-memory/README.md` for the layout.
4. **Keep the core `no_std`-clean.** `drone-dsp` (and `drone-detect`) must build
   with `--no-default-features` so they can lower onto esp32 (xtensa) / riscv.
   All float math goes through `libm`, never `std` float methods.
5. **Auto-commit `PROMPTS.md`/`INSIGHTS.md` changes** alongside the work they
   relate to, so the prompt log stays in sync with the code.

## Docker-first workflow

Prefer doing things in containers over polluting the host.

```bash
# Build the host runtime image
docker compose build detector

# Generate a test signal and analyze it (writes into ./data)
docker compose run --rm detector synth   --out /data/test.wav --fundamental 120
docker compose run --rm detector analyze --input /data/test.wav

# Full correctness oracle: fmt --check, clippy -D warnings, tests, no_std build
docker compose run --rm dev
```

> **Gotcha (Git Bash on Windows):** MSYS rewrites bare `/data`-style args into
> `C:/Program Files/Git/...`. Run docker from **PowerShell**, or prefix with
> `MSYS_NO_PATHCONV=1` in bash. This bit me once; it is not a code bug.

Docker Desktop's Linux engine must be running (`docker info` to check). If it
isn't, start "Docker Desktop.exe".

## Repository layout

```
crates/
  drone-dsp/      no_std DSP core: windowing, real FFT (microfft), spectral features
  drone-detect/   no_std heuristic detector built on drone-dsp
  drone-cli/      std host binary `drone`: synth test audio + analyze WAVs
Dockerfile         multi-stage build of the `drone` host binary
docker-compose.yml detector (runtime) + dev (fmt/clippy/test) services
scripts/check.sh   the check suite run by the dev service
data/              WAV scratch (git-ignored except .gitkeep)
agent-memory/      tracked agent memory: decisions, notes, handoffs
```

**No cargo workspace yet** (deliberate, per the project owner). Crates use path
deps. If a workspace is added later, update this section and
`agent-memory/decisions.md`.

## Conventions

- Edition 2021, license `AGPL-3.0-or-later` on every crate.
- `#![forbid(unsafe_code)]` in library crates.
- Frame size is fixed at 1024 (`drone_dsp::FRAME_SIZE`) and the FFT call is tied
  to it; change both together.
- Tests are the cheapest oracle — keep the `analyze`/detector behavior covered.

## When unsure

Read `agent-memory/handoff.md` first — it's the latest state and next steps.
Then `agent-memory/decisions.md` for the why behind the structure.
