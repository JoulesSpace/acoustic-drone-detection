---
title: Crates in crates/, no workspace yet
type: decision
date: 2026-06-01
status: accepted
tags: [structure, cargo]
---

# 0002 - Crates in `crates/`, no workspace (yet)

**Decision:** Put each crate under `crates/` with its own `Cargo.toml` and use
path dependencies. Do **not** create a root cargo workspace yet.

**Why:** Explicit owner request ("no need to set up a workspace quite yet").
Path deps still resolve without a workspace; the Docker build copies the whole
`crates/` tree so siblings are visible.

**Revisit when:** cross-crate builds or a shared lockfile become painful. If a
workspace is added, update [`CLAUDE.md`](../../CLAUDE.md) and supersede this ADR.
