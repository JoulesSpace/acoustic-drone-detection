---
title: Docker-first with two compose services
type: decision
date: 2026-06-01
status: accepted
tags: [docker, tooling, ci]
---

# 0005 - Docker-first, two compose services

**Decision:** `Dockerfile` (multi-stage host binary) + `Dockerfile.dev`
(toolchain with rustfmt/clippy) driven by `docker-compose.yml` services
`detector` and `dev`. `scripts/check.sh` is the correctness oracle.

**Why:** Owner prefers working in containers over the host. Splitting `dev` from
`detector` keeps the shipped runtime image tiny while giving a fat toolchain
image for the checks.

**Gotcha it created:** the slim Rust image omits rustfmt/clippy - see
[insight](../insights/slim-rust-missing-rustfmt-clippy.md).
