# agent-memory

This directory **is the agent's memory** for this repository. It is tracked in
git on purpose: insights, decisions, domain notes, and handoffs live here so
that any future session (or human) can pick up with full context instead of
re-deriving it.

## Files

- **`handoff.md`** — the latest state and the next concrete steps. Read this
  first. Update it at the end of any substantial chunk of work.
- **`decisions.md`** — append-only-ish log of architectural decisions and the
  reasoning behind them (lightweight ADRs).
- **`dsp-notes.md`** — signal-processing and domain knowledge: conventions,
  gotchas, and the math behind the features.

## How to use it

- Write things down *as you learn them*, not after the fact.
- Prefer small, dated, specific entries over vague summaries.
- When a decision is reversed, don't delete the old entry — add a new one that
  supersedes it and say why. The trail is the value.
- Keep it honest: record what was actually verified vs. assumed.

See `../CLAUDE.md` for the operating rules that govern this repo.
