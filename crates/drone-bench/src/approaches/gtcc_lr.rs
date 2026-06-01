//! GTCC (gammatone cepstral coefficients) + logistic regression.
//!
//! TODO(approach-wave-2): implement. Like `mfcc_lr` but with a gammatone
//! filterbank (ERB-spaced) instead of mel; the literature reports GTCC sometimes
//! beats MFCC for drone audio. Pool to a clip feature, standardize, train
//! logistic regression in `fit`, return sigmoid in [0,1].

use crate::Approach;

#[derive(Default)]
pub struct GtccLr;

impl GtccLr {
    pub fn new() -> Self {
        Self
    }
}

impl Approach for GtccLr {
    fn name(&self) -> &str {
        "gtcc_lr"
    }
    fn description(&self) -> &str {
        "Gammatone cepstral coefficients + logistic regression (stub)"
    }
    fn score(&self, _samples: &[f32], _sample_rate: u32) -> f32 {
        0.5
    }
}
