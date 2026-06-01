#!/usr/bin/env bash
# Correctness oracles for the crates. Runs inside the `dev` compose service
# (Linux + Rust toolchain). Treats clippy warnings as errors.
set -euo pipefail

echo "== folderinfo lint =="
bash scripts/folderinfo.sh

CRATES=(drone-dsp drone-detect drone-cli)

for c in "${CRATES[@]}"; do
  echo "== $c: fmt =="
  ( cd "crates/$c" && cargo fmt --check )
  echo "== $c: clippy =="
  ( cd "crates/$c" && cargo clippy --all-targets -- -D warnings )
  echo "== $c: test =="
  ( cd "crates/$c" && cargo test )
done

# Verify the core stays no_std-clean (build with std feature removed). This is
# the property that lets it lower onto esp32/riscv firmware later.
echo "== drone-dsp: no_std build =="
( cd crates/drone-dsp && cargo build --no-default-features )

echo "ALL CHECKS PASSED"
