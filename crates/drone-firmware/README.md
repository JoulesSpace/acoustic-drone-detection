# drone-firmware

Real `#![no_std] #![no_main]` firmware for the **esp32-C6** (RISC-V, RV32IMAC,
target `riscv32imac-unknown-none-elf`). It is the project's deployment proof: it
runs the **real** edge detector (`drone_edge::EdgeDetector`, the no_std /
alloc-free spectral-rule scorer) over `drone_dsp::FRAME_SIZE` (1024) sample
audio frames and, on a latched drone alert, lights an LED GPIO and prints a log
line over the USB-Serial-JTAG console.

esp32-C6 is RISC-V, so this links on **mainline** stable Rust - no xtensa fork,
unlike the esp32-S3. Built on the esp-rs `esp-hal` 1.x stack.

## Layout

```
crates/drone-firmware/
  Cargo.toml            esp-hal/esp-backtrace/esp-println/esp-bootloader-esp-idf + drone-edge/drone-dsp
  .cargo/config.toml    pins target = riscv32imac-unknown-none-elf + esp linker args (-Tlinkall.x)
  src/main.rs           #[main] entry: init, EdgeDetector, frame loop, LED + println alert
```

## Build

The esp linker args and target live in `./.cargo/config.toml`, which cargo
discovers **relative to the current directory**. Build from inside the crate:

```bash
rustup target add riscv32imac-unknown-none-elf
rustup component add rust-src           # config uses build-std
cd crates/drone-firmware
cargo build --release
```

> Gotcha: `cargo build --manifest-path crates/drone-firmware/Cargo.toml` run
> from the repo root does NOT pick up this crate's `.cargo/config.toml` (cargo
> walks up from the invocation cwd, not the manifest), so it tries to build for
> the host and fails. Always build from inside the crate directory, or pass
> `--target riscv32imac-unknown-none-elf` plus the linker rustflags yourself.

Output ELF:
`target/riscv32imac-unknown-none-elf/release/drone-firmware`.

## Size (release, esp-hal 1.1.1, opt-level "s", fat LTO)

`llvm-size -A` on the ELF:

| section   | bytes  | note                     |
|-----------|--------|--------------------------|
| `.text`   | 67,480 | code (~66 KB)            |
| `.rodata` | 20,052 | read-only data (~20 KB)  |
| `.data`   | 1,140  | initialised RAM data     |
| `.bss`    | 516    | zero-init RAM            |

(The on-disk ELF is ~1.6 MB because it keeps `debug_info`; the flashed image is
the section bytes above plus the bootloader/partition headers.)

## Flash + monitor

`.cargo/config.toml` sets the runner to `espflash flash --monitor`, so with an
esp32-C6 attached over USB:

```bash
cargo install espflash          # one-time, host tool
cd crates/drone-firmware
cargo run --release             # builds, flashes, opens the serial monitor
```

Or flash an already-built ELF directly:

```bash
espflash flash --monitor target/riscv32imac-unknown-none-elf/release/drone-firmware
```

Save a raw merged image (no board attached) and inspect its size:

```bash
espflash save-image --chip esp32c6 --merge \
  target/riscv32imac-unknown-none-elf/release/drone-firmware drone-fw.bin
```

## Wiring (esp32-C6-DevKitC-1 / -M-1 defaults)

- Alert LED -> `GPIO8` (external LED + resistor, or an on-board user LED),
  active-high.
- Serial -> on-board USB-Serial-JTAG; `esp-println` logs land there at the
  `ESP_LOG=info` level set in `.cargo/config.toml`.
- I2S MEMS mic (e.g. INMP441 / ICS-43434) -> **not yet wired** (see caveat).

## Behaviour

The detector path is fully real. To exercise it on silicon without a mic, the
loop alternates a synthetic harmonic (drone-like) signature with broadband
noise; you should see the LED latch on and an `ALERT: drone detected` line
during the harmonic phases, then `clear:` during the noise phases.

## The one caveat: I2S mic read is stubbed

`read_mic_frame` in `src/main.rs` is the single TODO. It currently synthesises
frames; the real next step is an `esp_hal::i2s::master` read from a MEMS mic
(std/Philips mode @ 16 kHz, DMA `FRAME_SIZE` samples, convert 24-bit words to
`f32`). Everything downstream of that frame - window, FFT, spectral features,
the EMA+hold alert logic - is the real `drone-edge` code, unchanged from the
host build.
