//! Core DSP for acoustic drone detection.
//!
//! This crate is deliberately `no_std`-friendly so it can be lowered onto edge
//! targets (esp32 xtensa, riscv32). It operates on a single fixed-size audio
//! frame at a time, which is exactly how you would feed it from a ring buffer
//! on a microcontroller. Higher-level "process a whole signal" loops live in
//! the host-side crates (`drone-cli`).
//!
//! All float math goes through [`libm`] so nothing here depends on `std`.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

pub mod features;
pub mod fft;
pub mod window;

/// Number of real samples per analysis frame.
///
/// 1024 is a pragmatic v0.1.0 choice: at 16 kHz it gives ~15.6 Hz/bin and a
/// 64 ms frame, which resolves drone blade-pass / motor tones well while still
/// fitting comfortably in microcontroller RAM. The FFT routine is tied to this
/// size (see [`fft`]); change both together.
pub const FRAME_SIZE: usize = 1024;

/// Number of usable magnitude bins (DC .. just below Nyquist).
///
/// A real FFT of `FRAME_SIZE` samples yields `FRAME_SIZE / 2` complex bins.
pub const NUM_BINS: usize = FRAME_SIZE / 2;

/// A single frame of mono audio samples in `[-1.0, 1.0]`.
pub type Frame = [f32; FRAME_SIZE];

/// A magnitude spectrum: linear magnitudes per frequency bin.
pub type Spectrum = [f32; NUM_BINS];

pub use features::{
    band_energy, bin_to_hz, dominant_bin, hz_to_bin, spectral_centroid, total_energy,
};
pub use fft::magnitude_spectrum;
pub use window::hann_in_place;
