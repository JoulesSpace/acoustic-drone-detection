//! Baseline approach: the existing `drone-detect` band-energy heuristic.
//!
//! Confidence = mean band-energy ratio across frames. This is the simplest
//! real detector and the floor every other approach should beat.

use crate::{util, Approach};
use drone_detect::Detector;

#[derive(Default)]
pub struct BandRatio;

impl BandRatio {
    pub fn new() -> Self {
        Self
    }
}

impl Approach for BandRatio {
    fn name(&self) -> &str {
        "band_ratio"
    }
    fn description(&self) -> &str {
        "Baseline: mean band-energy ratio (drone-detect heuristic)"
    }
    fn score(&self, samples: &[f32], sample_rate: u32) -> f32 {
        let det = Detector::new(sample_rate);
        let frames = util::spectra(samples);
        if frames.is_empty() {
            return 0.0;
        }
        let sum: f32 = frames.iter().map(|sp| det.analyze(sp).band_ratio).sum();
        (sum / frames.len() as f32).clamp(0.0, 1.0)
    }
}
