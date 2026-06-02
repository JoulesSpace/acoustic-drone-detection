# Contributing

Thanks for your interest in improving acoustic-drone-detection.

## Setup

Everything runs in containers (Docker Desktop's Linux engine must be running),
so you do not need a host Rust toolchain to build, test, or benchmark.

```bash
docker compose build detector          # build the runtime image
docker compose run --rm dev            # fmt + clippy -D warnings + tests + no_std builds
```

If you prefer a native toolchain, install stable Rust with `rustfmt`, `clippy`,
and the `riscv32imc-unknown-none-elf` target, then run `bash scripts/check.sh`.

## Before opening a PR

```bash
docker compose run --rm dev            # the full correctness oracle (CI runs this)
bash scripts/folderinfo.sh             # every directory needs a one-line .folderinfo
```

`scripts/check.sh` (what `dev` runs) does, per crate: `cargo fmt --check`,
`cargo clippy --all-targets -- -D warnings`, `cargo test`, and the `no_std` /
`riscv32imc` cross-builds that keep the core lowerable to edge hardware.

## Conventions

- **No em-dashes** anywhere (prose, comments, docstrings, figures) - use a
  spaced hyphen `-` or rewrite the sentence.
- **Keep the core `no_std`-clean.** `drone-dsp` and `drone-detect` must build
  with `--no-default-features`; all float math goes through `libm`.
- **Every directory carries a one-line `.folderinfo`** (CI lints this).
- **Report what you verified**, not what you assume: numbers in docs must come
  from running the benchmarks; flag caveats in
  [`agent-memory/notes/honest-limitations.md`](agent-memory/notes/honest-limitations.md).
- New task heads go in their own crate under `crates/`; reuse the `drone-dsp`
  backbone and the `drone-bench` eval harness rather than re-implementing DSP.
- Conventional commit messages (`feat:`, `fix:`, `docs:`, `chore:`, ...), in
  logical chunks rather than one giant commit.

## Branching

```
fork -> branch [name]/feat|fix-[short-name] -> PR -> address feedback -> merge
```

## Where things live

See [`CLAUDE.md`](CLAUDE.md) for the repository layout and operating rules, and
[`agent-memory/MEMORY.md`](agent-memory/MEMORY.md) for the design record (ADRs,
insights, domain notes, and session handoffs).
