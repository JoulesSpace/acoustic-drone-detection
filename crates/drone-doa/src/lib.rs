//! Acoustic Direction-of-Arrival (DoA) estimation for a drone source.
//!
//! The pipeline is the classic small-array recipe:
//!
//! 1. [`geometry`] — a uniform linear array (ULA) of `M` microphones spaced `d`
//!    metres apart. A far-field plane wave arriving from azimuth `θ` reaches
//!    adjacent mics with a time difference of arrival (TDOA) of `d·sin(θ)/c`.
//! 2. [`gcc_phat`] — estimate the TDOA between a mic pair via the
//!    generalized cross-correlation with phase transform (GCC-PHAT), including
//!    parabolic sub-sample peak interpolation.
//! 3. [`azimuth`] — turn the pairwise TDOAs into a single azimuth estimate by a
//!    least-squares fit over all adjacent ULA pairs, clamped to the
//!    unambiguous range set by the spacing.
//! 4. [`sim`] — a propagation simulator that produces `M` noisy channels with
//!    correct fractional inter-mic delays, so the whole thing can be benchmarked
//!    without multi-mic recordings.
//!
//! The estimation core is deliberately `no_std`-friendly (all float math goes
//! through [`libm`], the FFT through [`microfft`]) so it can lower onto the same
//! edge targets as `drone-dsp`. The simulator's noise generator and the
//! benchmark binary are host-side conveniences.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod azimuth;
pub mod gcc_phat;
pub mod geometry;
pub mod sim;

pub use azimuth::{estimate_azimuth, AzimuthEstimate};
pub use gcc_phat::{gcc_phat_tdoa, GccConfig};
pub use geometry::{UlaGeometry, SPEED_OF_SOUND};
pub use sim::{simulate_array, DroneSource, SimConfig};

/// FFT size used by the GCC-PHAT core, fixed so the `no_std` `microfft` path can
/// pick a concrete transform.
///
/// 2048 gives a `[-1024, +1024]` sample lag window. At 16 kHz that is ±64 ms,
/// orders of magnitude more than the largest physically possible inter-mic delay
/// for any sane array (a 1 m baseline is only ~140 samples), so the correlation
/// peak is never wrapped. Picking the smallest size that comfortably covers the
/// frame keeps the transform cheap on edge targets.
pub const GCC_FFT_SIZE: usize = 2048;

/// The largest analysis block (per channel) the GCC-PHAT core accepts.
///
/// Equal to [`GCC_FFT_SIZE`]; longer signals should be processed in blocks and
/// the resulting TDOAs averaged (the benchmark does exactly this).
pub const MAX_BLOCK: usize = GCC_FFT_SIZE;
