//! Amplitude-envelope modulation detector.
//!
//! TODO(approach-wave-2): implement. A multirotor modulates loudness at the
//! blade-pass rate. Extract the amplitude envelope (e.g. per-frame energy over
//! time, or rectified-and-smoothed signal), take its modulation spectrum (FFT of
//! the envelope), and look for a strong peak in the ~5-100 Hz blade/rotor
//! modulation band. A cue distinct from the spectral-comb methods. Confidence in
//! [0,1] from the normalized modulation-peak strength.

use crate::Approach;

#[derive(Default)]
pub struct EnvelopePeriodicity;

impl EnvelopePeriodicity {
    pub fn new() -> Self {
        Self
    }
}

impl Approach for EnvelopePeriodicity {
    fn name(&self) -> &str {
        "envelope_periodicity"
    }
    fn description(&self) -> &str {
        "Amplitude-envelope modulation-spectrum detector (stub)"
    }
    fn score(&self, _samples: &[f32], _sample_rate: u32) -> f32 {
        0.5
    }
}
