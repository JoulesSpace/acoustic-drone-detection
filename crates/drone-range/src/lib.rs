//! Acoustic distance (range) estimation for a drone source.
//!
//! Distance-from-audio is feasible but setup-dependent. The literature spans a
//! wide band: a ground-based GRU reaches 94-98% on 5-50 m bins (Kang), while a
//! drone-to-drone setup with strong ego-noise gets only 61-78% audio-only (Kim
//! 2023). The honest takeaway is that **distance is the hardest of our property
//! heads**, because real range is confounded by source loudness (a louder or
//! closer-looking drone looks the same) and by the environment.
//!
//! The most range-*specific* cue is **not raw loudness** but the spectral
//! **tilt** introduced by frequency-dependent air absorption: high frequencies
//! attenuate faster with distance than low ones, so the spectrum darkens with
//! range in a way a level change alone cannot mimic. This crate is built around
//! that idea.
//!
//! Pipeline:
//!
//! 1. [`features`] - clip-level features: overall level/energy, spectral
//!    centroid, spectral rolloff (85% / 95%), a high-vs-low band-energy ratio
//!    (the air-absorption tilt proxy), spectral slope, and a few MFCCs.
//! 2. [`regress`] - ridge (L2-regularized linear) regression that predicts the
//!    range in metres (mode (a)).
//! 3. [`classify`] - multinomial logistic regression that classifies a clip into
//!    range bins (mode (b), e.g. 10/15 m intervals like Kang).
//! 4. [`sim`] - a physics range simulator (spherical spreading,
//!    frequency-dependent air absorption, SNR falling with range) so the whole
//!    thing can be validated without distance-labeled recordings, analogous to
//!    how `drone-doa` is simulation-validated.
//!
//! The estimation core is deliberately `no_std`-friendly (all float math goes
//! through [`libm`]) so it can lower onto the same edge targets as `drone-dsp`.
//! The simulator's noise generator is `std`-free too; only the `range-bench`
//! binary needs `std`.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod classify;
pub mod features;
pub mod regress;
pub mod sim;

pub use classify::{BinModel, RangeBins};
pub use features::{clip_features, ClipFeatures, FeatureSet, N_FEATURES};
pub use regress::RidgeModel;
pub use sim::{simulate_clip, AbsorptionModel, RangeSimConfig, SourceConfig};

/// Sample rate (Hz) the range pipeline is tuned for.
///
/// 16 kHz matches the rest of the project (`drone-dsp::FRAME_SIZE` at 16 kHz is
/// a 64 ms / ~15.6 Hz-per-bin frame). Air absorption only becomes a strong cue
/// above a few kHz, so the Nyquist of 8 kHz still leaves useful tilt headroom.
pub const SAMPLE_RATE: u32 = 16_000;
