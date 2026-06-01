//! Spectral-feature gate.
//!
//! TODO(approach-wave): implement. Combine cheap scalar spectral descriptors —
//! spectral flatness, spectral entropy, centroid, rolloff, band-energy ratio —
//! into a tonality/drone-likeness confidence in [0,1]. Drone audio is tonal:
//! LOW flatness and LOW entropy, energy biased to the BPF band. Can be a
//! hand-tuned rule or a tiny logistic combination fit in `fit`. Use
//! `crate::util::spectra` and `drone_dsp` feature helpers.

use crate::Approach;

#[derive(Default)]
pub struct SpectralGate;

impl SpectralGate {
    pub fn new() -> Self {
        Self
    }
}

impl Approach for SpectralGate {
    fn name(&self) -> &str {
        "spectral_gate"
    }
    fn description(&self) -> &str {
        "Spectral-feature gate: flatness/entropy/centroid/band-ratio (stub)"
    }
    fn score(&self, _samples: &[f32], _sample_rate: u32) -> f32 {
        0.5 // TODO(approach-wave): replace with real spectral-feature scoring
    }
}
