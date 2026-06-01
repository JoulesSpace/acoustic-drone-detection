# Decisions

Lightweight architecture decision records. Newest at the bottom.

## 2026-06-01 — Rust as the implementation language

**Decision:** Implement the detection pipeline in Rust rather than Python.

**Why:** Fast, typed iteration on real performant DSP algorithms; the same code
can lower onto microcontrollers. Avoids the Python-stub-then-rewrite detour.

## 2026-06-01 — Crates in `crates/`, no workspace (yet)

**Decision:** Put each crate under `crates/` with its own `Cargo.toml` and use
path dependencies. Do **not** create a root workspace yet.

**Why:** Explicit owner request ("no need to set up a workspace quite yet").
Path deps still resolve fine without a workspace; the Docker build copies the
whole `crates/` tree so siblings are visible. Revisit if cross-crate builds or
shared lockfiles become painful — and update `CLAUDE.md` if so.

## 2026-06-01 — Three-crate split

**Decision:**
- `drone-dsp` — `no_std`-friendly DSP core (windowing, FFT, spectral features).
- `drone-detect` — `no_std`-friendly heuristic detector on top of the core.
- `drone-cli` — `std` host binary (`drone`) for synth + WAV analysis.

**Why:** Keep the edge-deployable math (`drone-dsp`, `drone-detect`) free of
`std` so it can be reused in esp32/riscv firmware, while quarantining all
host-only concerns (file IO, CLI, the over-the-whole-signal loop) in the binary.

## 2026-06-01 — microfft + libm for the FFT/math

**Decision:** Use `microfft` (pure-Rust, `no_std`, fixed power-of-two real FFT)
and route all float math through `libm`.

**Why:** `rustfft` pulls in `std`/alloc and is overkill for fixed-size embedded
frames. `microfft::real::rfft_1024` matches our 1024-sample frame exactly.
`libm` gives `sqrtf`/`cosf` without `std`, so one code path serves host and
bare-metal. Trade-off: FFT size is compile-time fixed (acceptable for v0.1.0).

**Gotcha recorded:** microfft packs the real Nyquist term into bin 0's `im`
field; we expose `|DC|` as `spectrum[0]` and drop Nyquist. See `dsp-notes.md`.

## 2026-06-01 — Docker-first, two compose services

**Decision:** `Dockerfile` (multi-stage host binary) + `Dockerfile.dev`
(toolchain with rustfmt/clippy) driven by `docker-compose.yml` services
`detector` and `dev`. `scripts/check.sh` is the correctness oracle.

**Why:** Owner prefers working in containers over the host. The slim Rust image
omits rustfmt/clippy (our oracles), hence the dedicated dev image so they aren't
re-downloaded every run.

## 2026-06-01 — Heuristic detector for v0.1.0 (not ML)

**Decision:** Detection = band-energy ratio in ~100–4000 Hz + dominant-tone-in-
band test, threshold 0.5.

**Why:** Transparent, cheap enough for a microcontroller, and gives a measurable
baseline to beat. No labelled data needed (constraint: one real drone, an
afternoon, 50€ budget). Synthetic harmonic-stack generator (`drone synth`) lets
us exercise the whole pipeline without recordings.
