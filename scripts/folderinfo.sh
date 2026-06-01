#!/usr/bin/env bash
# Lint: every tracked directory must contain a `.folderinfo` file - a one-line
# plain-text description of what lives there. Enforces CLAUDE.md hard-rule #6.
# Gitignored scratch (workspace/) is exempt.
set -euo pipefail

all_dirs() {
  echo "."
  git ls-files | while IFS= read -r f; do
    d=$(dirname "$f")
    while [ "$d" != "." ]; do
      echo "$d"
      d=$(dirname "$d")
    done
  done
}

missing=0
while IFS= read -r d; do
  case "$d" in
    workspace | workspace/*) continue ;;
  esac
  if [ ! -f "$d/.folderinfo" ]; then
    echo "MISSING: $d/.folderinfo"
    missing=1
  fi
done < <(all_dirs | sort -u)

if [ "$missing" -ne 0 ]; then
  echo "folderinfo lint FAILED - add a one-line .folderinfo to each dir listed above."
  exit 1
fi
echo "folderinfo lint OK"
