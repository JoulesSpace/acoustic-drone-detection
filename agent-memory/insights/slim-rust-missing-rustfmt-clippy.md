---
title: Slim Rust image lacks rustfmt/clippy
type: insight
date: 2026-06-01
tags: [docker, tooling, gotcha]
---

# The slim Rust image has no rustfmt/clippy

**Symptom:** `docker compose run --rm dev` failed at the first fmt step:
`error: 'cargo-fmt' is not installed for the toolchain '1.92.0-...'`.

**Cause:** `rust:1.92-slim-bookworm` ships a minimal toolchain without the
`rustfmt` and `clippy` components — and those are our correctness oracles.

**Fix:** dedicated [`Dockerfile.dev`](../../Dockerfile.dev) that does
`rustup component add rustfmt clippy` once, baked into the image, instead of
re-downloading them on every `docker compose run --rm dev`.

See decision [0005](../decisions/0005-docker-first-two-services.md).
