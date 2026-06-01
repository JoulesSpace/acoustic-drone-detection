//! Cepstrum / autocorrelation periodicity detector.
//!
//! TODO(approach-wave): implement. The harmonic stack is periodic in frequency;
//! the cepstrum (IFFT of the log-magnitude spectrum) shows a peak at the
//! quefrency of the harmonic spacing — a compact tonality/periodicity readout.
//! Alternatively use time-domain autocorrelation peaking at the fundamental
//! period. Confidence = normalized cepstral/autocorr peak height in [0,1].
//! Complements HPS (different failure modes; can cross-check octave errors).
//! Use `crate::util::spectra` for spectra; you may need an inverse FFT (a real
//! IFFT can be done via the forward FFT with conjugation, or compute the
//! cepstrum directly from the log-magnitude vector).

use crate::Approach;

#[derive(Default)]
pub struct Cepstrum;

impl Cepstrum {
    pub fn new() -> Self {
        Self
    }
}

impl Approach for Cepstrum {
    fn name(&self) -> &str {
        "cepstrum"
    }
    fn description(&self) -> &str {
        "Cepstral / autocorrelation periodicity detector (stub)"
    }
    fn score(&self, _samples: &[f32], _sample_rate: u32) -> f32 {
        0.5 // TODO(approach-wave): replace with real periodicity scoring
    }
}
