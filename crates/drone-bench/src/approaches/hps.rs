//! Harmonic Product Spectrum / harmonic-comb matcher.
//!
//! TODO(approach-wave): implement. Exploit the drone blade-pass fundamental
//! (BPF) + harmonic stack. Either HPS (downsample the magnitude spectrum by
//! 2,3,4,… and multiply — energy reinforces at the true f0) or a harmonic-comb
//! correlation scanning f0 over the plausible BPF range. Confidence should
//! reflect peak height/sharpness and the number of consistent harmonics, mapped
//! to [0,1]. Use `crate::util::spectra` for per-frame magnitude spectra and
//! `drone_dsp::bin_to_hz` for frequency mapping. Watch octave errors.

use crate::Approach;

#[derive(Default)]
pub struct Hps;

impl Hps {
    pub fn new() -> Self {
        Self
    }
}

impl Approach for Hps {
    fn name(&self) -> &str {
        "hps"
    }
    fn description(&self) -> &str {
        "Harmonic Product Spectrum / harmonic-comb matcher (stub)"
    }
    fn score(&self, _samples: &[f32], _sample_rate: u32) -> f32 {
        0.5 // TODO(approach-wave): replace with real harmonic scoring
    }
}
