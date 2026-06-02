//! Blind Source Separation (BSS) for acoustic drone detection.
//!
//! When several drones fly together, or a drone is buried under broadband wind
//! noise or a tonal interferer, a single microphone hears an additive *mixture*
//! of sources. The harmonic-comb signature a detector keys on is smeared across
//! that mixture, so detection degrades. With `M` microphones observing `M`
//! sources we can try to *unmix* them blindly - without knowing the sources or
//! the mixing - and then run detection on the recovered drone component instead
//! of the raw channel. That is the layer this crate adds.
//!
//! # What is implemented
//!
//! * [`fastica`] - the standard FastICA pipeline: mean-centering, PCA whitening
//!   (covariance eigendecomposition via a from-scratch [`eig`] Jacobi symmetric
//!   eigensolver), then the fixed-point iteration with the `tanh` nonlinearity
//!   and symmetric decorrelation. Returns the unmixing matrix and the separated
//!   sources for the general `M`-mic, `M`-source case (2x2 is handled cleanly as
//!   a special case of the general path).
//! * [`mix`] - a mixing simulator: synthesize drone-like harmonic sources at
//!   chosen fundamentals (plus optional broadband noise / a tonal interferer),
//!   then mix them through a random invertible `M x M` matrix into `M` channels.
//! * [`metrics`] - separation-quality metrics: **SIR improvement (dB)** and
//!   correlation to the ground-truth sources, with ICA's permutation/scale
//!   ambiguity resolved by best-match assignment.
//!
//! # Modeling caveat (important)
//!
//! This crate models **instantaneous** mixing: each channel is a weighted sum of
//! the sources at the *same* time instant, `x = A s`. Real acoustic mixing is
//! **convolutive** - each source reaches each mic through a different,
//! frequency-dependent room/air impulse response, so `x[n] = sum_k a_k * s_k`.
//! Instantaneous ICA is the standard, well-understood first cut (and is exactly
//! right for closely-spaced mics in the far field with negligible inter-mic
//! delay); the convolutive case (frequency-domain ICA / IVA per STFT bin) is the
//! documented next step and is noted but not implemented here.
//!
//! All randomness flows through a single seeded PRNG ([`rng`]) so every run -
//! source synthesis, mixing matrix, and the FastICA weight initialization - is
//! bit-for-bit reproducible.

#![forbid(unsafe_code)]

pub mod eig;
pub mod fastica;
pub mod metrics;
pub mod mix;
pub mod rng;

pub use fastica::{fastica, FastIcaConfig, FastIcaResult};
pub use metrics::{
    best_match, drone_separation_quality, sir_db, sir_improvement_db, SeparationQuality,
};
pub use mix::{mix_sources, scene, DroneSource, ExtraSource, MixConfig, Mixture};
pub use rng::Rng;
