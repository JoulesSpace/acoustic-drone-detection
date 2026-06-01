# Handoff — latest state & next steps

_Last updated: 2026-06-01_

## State (v0.1.0 scaffold — DONE & verified)

The repo is now a Rust crate workshop, docker-first, edge-ready at the core.

- **`crates/drone-dsp`** — `no_std`-friendly DSP core: Hann window, real FFT via
  `microfft`, magnitude spectrum, spectral features (centroid, band/total
  energy, dominant bin, bin↔Hz). 5 unit tests.
- **`crates/drone-detect`** — `no_std`-friendly heuristic detector
  (band-ratio + dominant-in-band). 3 unit tests.
- **`crates/drone-cli`** — `std` binary `drone` with `synth` and `analyze`
  subcommands.
- **Docker**: `Dockerfile` (host binary), `Dockerfile.dev` (toolchain w/
  rustfmt+clippy), `docker-compose.yml` (`detector` + `dev`), `scripts/check.sh`.

### Verified on 2026-06-01
- Host: all 8 tests pass; `drone synth` + `drone analyze` discriminate synthetic
  drone (100% frames) from a 60 Hz tone (0%).
- Docker: image builds; `docker compose run --rm detector` synth+analyze works
  end-to-end through the `./data` mount.
- `docker compose run --rm dev bash scripts/check.sh` → **ALL CHECKS PASSED**
  (fmt, clippy `-D warnings`, tests, `--no-default-features` no_std build).

## Gotchas learned

- **Git Bash path mangling:** running `docker compose run ... /data/x.wav` from
  MSYS bash rewrites `/data/...` into `C:/Program Files/Git/...`. Use PowerShell
  or `MSYS_NO_PATHCONV=1`. Cost me a debugging detour; not a code bug.
- The slim Rust image has **no rustfmt/clippy** → `Dockerfile.dev` adds them.

## Next steps (pick up here)

1. **Real data.** Wire in a dataset (saraalemadi/DroneAudioDataset or a Kaggle
   multi-dataset) and add an `analyze`-over-a-folder mode + a tiny labelled
   eval (precision/recall) to replace gut-feel with numbers.
2. **Robustness.** Temporal smoothing across frames; harmonic-spacing feature;
   SNR normalization. Stress-test against wind/noise mixes.
3. **Edge build proof.** Add a real cross-target build (riscv32imc or esp32
   xtensa via espup) for `drone-dsp` to prove the no_std core actually links
   bare-metal — currently only `--no-default-features` host build is checked.
4. **DoA.** Multi-channel input + array geometry (needs ≥2 mics).

## Conventions reminder

Semantic commits, **no Co-Authored-By trailer**, keep `agent-memory/` and
`CLAUDE.md` current. See `../CLAUDE.md`.
