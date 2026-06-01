# drone-mobile

A stable C ABI over the `drone-edge` detector so the same DSP that runs on the
desktop and the esp32 firmware also runs in a phone app. Completes the hardware
spectrum: **MCU (`drone-firmware`) -> phone (`drone-mobile`) -> desktop
(`drone-live`) -> server (`drone-bench`).**

## C API

```c
size_t drone_mobile_frame_size(void);                 // block size to feed
float  drone_mobile_score(const float* buf, size_t len, uint32_t sr); // [0,1], <0 on null
DroneDetector* drone_mobile_new(uint32_t sample_rate);
int    drone_mobile_push(DroneDetector*, const float* buf, size_t len); // 1=alert,0,-1=err
float  drone_mobile_confidence(const DroneDetector*);
void   drone_mobile_free(DroneDetector*);
```

Audio is mono `f32` in `[-1, 1]`. Feed blocks of `drone_mobile_frame_size()`
samples (resample the mic stream to 16 kHz first, as `drone-live` does).

## Build a host shared library (sanity)

```bash
cargo build --release --manifest-path crates/drone-mobile/Cargo.toml
# -> target/release/{libdrone_mobile.so|.dylib|drone_mobile.dll}
```

## Build for Android (.so per ABI) with cargo-ndk

```bash
cargo install cargo-ndk
rustup target add aarch64-linux-android armv7-linux-androideabi \
                  x86_64-linux-android i686-linux-android
# needs the Android NDK (set ANDROID_NDK_HOME)
cargo ndk -t arm64-v8a -t armeabi-v7a -o ./jniLibs \
  build --release --manifest-path crates/drone-mobile/Cargo.toml
```

Load from Kotlin via `System.loadLibrary("drone_mobile")` and JNI `external fun`
declarations, or generate a binding with `uniffi`/`cbindgen`. iOS: add
`aarch64-apple-ios` and link the `staticlib`.

> The Rust side builds cleanly today; producing the actual `.so` needs the NDK
> toolchain on the build host (the analogue of needing `espflash` + a board for
> `drone-firmware`).
