#!/usr/bin/env bash
# Correctness oracles for the crates. Runs inside the `dev` compose service
# (Linux + Rust toolchain). Treats clippy warnings as errors.
set -euo pipefail

echo "== folderinfo lint =="
bash scripts/folderinfo.sh

CRATES=(drone-dsp drone-detect drone-cli drone-bench drone-freq drone-id drone-doa drone-live drone-edge drone-vendor drone-mobile drone-cnn drone-bss)

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

echo "== drone-doa: no_std core build =="
( cd crates/drone-doa && cargo build --no-default-features )

echo "== drone-edge: riscv32imc bare-metal cross-build =="
( cd crates/drone-edge && cargo build --release --target riscv32imc-unknown-none-elf )

# Note: crates/drone-firmware (esp32-C6, esp-hal) is intentionally NOT in CRATES.
# It targets bare-metal riscv32imac with build-std and must build from its own
# dir; build it with scripts/build-firmware.sh (needs the target + rust-src +
# network for the esp-hal crates).

echo "ALL CHECKS PASSED"
