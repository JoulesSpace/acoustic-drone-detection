---
title: Commit integrations before dispatching tree-mutating agents
type: insight
date: 2026-06-01
tags: [agents, workflow, gotcha, git]
---

# Commit integrations before dispatching agents that run git in the shared tree

When fixing `hps`, I dispatched an agent to work **in the main tree** (so it had
the gitignored `data/`). That agent ran `cargo fmt` (whole-crate) and then
"reverted the incidental reformatting of other files" with a git checkout/restore.

Because my freshly-integrated `spectral_gate.rs`, `mfcc_lr.rs`, and `cepstrum.rs`
were **uncommitted** at that moment, the agent's git revert restored them to
their last committed state — the **stubs** — silently clobbering ~900 lines of
integrated work. I had also already removed the source worktrees, so I had to
reconstruct the three files from the agents' return messages.

**How to apply:**
- **Commit (or stash) integrated work before dispatching any agent that may run
  `git checkout/restore/stash/fmt` in the same working tree.** Uncommitted work
  is the thing most easily lost.
- Prefer giving file-mutating agents an **isolated worktree**, and **keep the
  worktree until you've integrated AND committed** its file (don't `git worktree
  remove` early — it's your only recovery copy).
- Have agents **return the full file content** in their final message too; it's a
  second recovery path (note: it arrives HTML-escaped — un-escape `&amp; &lt; &gt;`).
- Tell tree-mutating agents explicitly: do **not** run repo-wide `cargo fmt` or
  any `git` command that touches files other than your own.

Related: [[synth-vs-real-generalization]].
