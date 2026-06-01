# agent-memory

This directory **is the agent's (Claude's) memory** for this repository. It is
tracked in git on purpose: decisions, insights, domain notes, and session
handoffs live here so any future session - or human - picks up with full context
instead of re-deriving it.

[**MEMORY.md**](MEMORY.md) is the index and the entry point. Read it first.

## Layout

```
agent-memory/
  MEMORY.md      ← index of everything; start here
  README.md      ← this file (how the memory works)
  handoffs/      ← dated session handoffs: state + next steps (newest = current)
  decisions/     ← architecture decision records (ADRs), one per file, numbered
  insights/      ← discrete gotchas / learnings, one idea per file
  notes/         ← durable domain knowledge (DSP conventions, detection theory)
```

## Entry format

Every entry starts with YAML frontmatter so it is greppable and self-describing:

```yaml
---
title: Short human title
type: decision | insight | handoff | note
date: 2026-06-01
status: accepted        # decisions only: accepted | superseded
tags: [dsp, docker]
---
```

Then the body. Cross-link entries with relative markdown links (they render on
GitHub) and keep the link in [MEMORY.md](MEMORY.md) up to date.

## Rules of use

- Write things down **as you learn them**, not after the fact.
- One idea per file. Prefer small, dated, specific entries over vague summaries.
- **Never rewrite history to hide a reversal.** When a decision changes, add a
  new ADR and set the old one's `status: superseded` with a pointer. The trail
  is the value.
- Record what was **actually verified** vs. assumed.
- When you add/rename/remove a file, update [MEMORY.md](MEMORY.md) in the same
  change. A stale index is a bug.
- End-of-session: drop a new handoff in `handoffs/` and relink it as "latest" in
  MEMORY.md.

See [`../CLAUDE.md`](../CLAUDE.md) for the repo-wide operating rules.
