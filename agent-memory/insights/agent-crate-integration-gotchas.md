---
title: Integration gotchas for agent-built crates
type: insight
date: 2026-06-01
tags: [agents, workflow, gotcha]
---

# Integration gotchas for agent-built crates

When integrating whole crates built by subagents (in worktrees) into main:

- **Stray crate-local `benchmarks/`.** Agents that run their bench binary from
  inside the crate directory write `crates/<crate>/benchmarks/results/*.json`.
  The root `.gitignore` only anchors `/benchmarks/results/*`, so these nested
  copies are NOT ignored and sneak into commits - and the new untracked dirs
  fail the `.folderinfo` lint. **On integration, `rm -rf crates/*/benchmarks`**
  (results belong at the repo-root `benchmarks/`). I also re-create them by
  running a crate's bench via `cd crates/<c>` during verification - run from the
  repo root instead, or clean up after.
- **Add the new crate to `scripts/check.sh`** (`CRATES=(...)`) and, if it has a
  `no_std` core, add a `--no-default-features` build line.
- **clap rejects negative option values**: `--snr -10` is parsed as a flag →
  use `--snr=-10`.
- Keep the agent's worktree until the crate is integrated AND committed (it's the
  recovery copy). See [[commit-before-dispatching-tree-mutating-agents]].
