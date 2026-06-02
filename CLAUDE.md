# CLAUDE.md - operating guide for this repository

This file is **mine to maintain**. Whenever something here goes stale - a path
moves, a command changes, a decision is reversed - I update it in the same
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
3. **Maintain `agent-memory/`.** It is the tracked, layered Claude memory:
   `decisions/` (ADRs), `insights/` (gotchas), `notes/` (domain knowledge), and
   `handoffs/` (dated session state), all indexed by `agent-memory/MEMORY.md`.
   Update it as work happens, not after - and update the index in the same
   change. See `agent-memory/README.md` for the format.
4. **Keep the core `no_std`-clean.** `drone-dsp` (and `drone-detect`) must build
   with `--no-default-features` so they can lower onto esp32 (xtensa) / riscv.
   All float math goes through `libm`, never `std` float methods.
5. **Auto-commit `PROMPTS.md`/`INSIGHTS.md` changes** alongside the work they
   relate to, so the prompt log stays in sync with the code.
6. **Every folder carries a `.folderinfo`.** Each tracked directory in this repo
   has a `.folderinfo` file: a one-line plain-text description of what lives
   there and what it's for (e.g. `agent memories for future sessions or
   reference`). When you create a new folder, create its `.folderinfo` in the
   same change. `scripts/folderinfo.sh` lints this and runs as part of the check
   suite, so a missing one fails CI. (Gitignored scratch like `workspace/` is
   exempt; `data/` keeps a tracked `.folderinfo` even though its contents are
   ignored.)
7. **No em-dashes (`-`).** Never write the em-dash character anywhere: prose,
   commit messages, code comments, doc strings, or generated figures. Use a
   spaced hyphen `-` as a separator, or rewrite the sentence. (En-dashes in
   numeric ranges like `90-2400x` are fine, but a plain hyphen is preferred.)

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
  drone-dsp/       no_std DSP backbone: windowing, real FFT (microfft), spectral features
  drone-detect/    no_std heuristic detector (energy-in-band baseline)
  drone-cli/       std host binary `drone`: synth test audio + analyze WAVs
  drone-bench/     eval harness: Approach trait, dataset, metrics; 14 detection approaches
                   + analysis bins (xeval, heldout32, pareto, ratesweep, robust, fieldeval)
  drone-doa/       direction of arrival (GCC-PHAT + ULA), no_std core
  drone-id/        multiclass drone-type recognition
  drone-vendor/    multi-brand / vendor recognition
  drone-freq/      blade-pass frequency / RPM estimation
  drone-range/     distance estimation (no_std core)
  drone-bss/       blind source separation (FastICA) for multi-UAV / ego-noise
  drone-cnn/       upstream mel-CNN baseline (candle) for the honest head-to-head
  drone-edge/      no_std training-free detector (cross-builds to riscv32imc)
  drone-firmware/  real esp32-C6 firmware running drone-edge (scripts/build-firmware.sh)
  drone-live/      cpal live mic: device probe + listen/alert + record
  drone-mobile/    C ABI / JNI FFI over drone-edge (Android/iOS via cargo-ndk)
docker/            Dockerfiles: runtime, dev toolchain (.dev), plot, data
docker-compose.yml services: detector, dev, bench, plot, data
.github/workflows/ ci.yml - runs scripts/check.sh on push / PR (mirrors `dev`)
Makefile           convenience wrappers over docker compose (build/check/bench/...)
scripts/           checks, dataset download, folderinfo + firmware build, infographic
benchmarks/        results (JSON, gitignored) + matplotlib plots + MODEL_CARDS.md
assets/            generated infographic (signal_chain.png)
data/              datasets - git-ignored contents (.gitkeep + .folderinfo tracked)
workspace/         gitignored scratch for cloning/inspecting upstream repos
agent-memory/      tracked agent memory; see agent-memory/MEMORY.md
```

**No cargo workspace yet** (deliberate, per the project owner). Crates use path
deps. If a workspace is added later, update this section and
`agent-memory/decisions/0002-crates-no-workspace.md`.

## Conventions

- Edition 2021, license `AGPL-3.0-or-later` on every crate.
- `#![forbid(unsafe_code)]` in library crates.
- Frame size is fixed at 1024 (`drone_dsp::FRAME_SIZE`) and the FFT call is tied
  to it; change both together.
- Tests are the cheapest oracle - keep the `analyze`/detector behavior covered.

## When unsure

Read `agent-memory/MEMORY.md` first - it indexes everything and links the latest
handoff (current state + next steps), the decisions, and the insights.
