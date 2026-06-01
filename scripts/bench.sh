#!/usr/bin/env bash
# Run the benchmark and render plots in one go (docker-first).
# Any args are forwarded to drone-bench, e.g.:
#   scripts/bench.sh --data /work/data/dads
# Defaults to synthetic data when no args are given.
set -euo pipefail

# On Git Bash, stop MSYS from rewriting /work-style paths.
export MSYS_NO_PATHCONV=1

if [ "$#" -eq 0 ]; then
  set -- --synth --n 300
fi

echo ">> benchmarking ($*)"
docker compose run --rm bench "$@"

echo ">> plotting"
docker compose run --rm plot

echo ">> done — see benchmarks/plots/"
