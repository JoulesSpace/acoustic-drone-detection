---
title: Git Bash mangles /data docker args
type: insight
date: 2026-06-01
tags: [docker, windows, gotcha]
---

# Git Bash mangles `/data`-style docker args

**Symptom:** `docker compose run --rm detector synth --out /data/x.wav` failed
with `error: No such file or directory (os error 2)`, while `--entrypoint sh -c
"drone synth --out /data/x.wav"` worked fine.

**Cause:** MSYS (Git Bash on Windows) auto-converts bare Unix-looking arguments
into Windows paths - `/data` became `C:/Program Files/Git/data`. Inside a quoted
`sh -c "..."` string the conversion doesn't happen, which is why that path
worked and masked the issue.

**Fix:** Run docker from **PowerShell**, or prefix the bash command with
`MSYS_NO_PATHCONV=1`.

**Lesson:** This is a shell artifact, **not a code bug**. Don't go hunting in the
Rust when only the bash-launched docker invocations fail but `sh -c` ones pass.
