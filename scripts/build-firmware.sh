#!/usr/bin/env bash
# Build the esp32-C6 firmware (crates/drone-firmware).
#
# It targets bare-metal RISC-V (riscv32imac-unknown-none-elf) via esp-hal and
# uses build-std, so it is NOT part of the standard `dev` check loop. It also
# MUST be built from the crate directory: the crate's `.cargo/config.toml` pins
# the target and linker args, and cargo reads `.cargo/config.toml` relative to
# the invocation cwd (not the manifest path).
#
# Needs: the riscv32imac target, the rust-src component, and network access for
# the esp-hal crates. Run on the host (or a toolchain image that has them).
set -euo pipefail

rustup target add riscv32imac-unknown-none-elf
rustup component add rust-src

cd crates/drone-firmware
cargo build --release

elf="target/riscv32imac-unknown-none-elf/release/drone-firmware"
echo "firmware ELF: crates/drone-firmware/$elf"
echo "flash a connected esp32-C6 with:  espflash flash --monitor $elf"
