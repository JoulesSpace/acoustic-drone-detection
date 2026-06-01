//! MFCC + logistic-regression classifier.
//!
//! TODO(approach-wave): implement. Extract MFCCs per frame (mel filterbank over
//! the magnitude spectrum → log → DCT-II, ~13-20 coeffs), pool to a clip-level
//! feature vector (mean/std over frames), standardize (store mean/var from
//! `fit`), and train a logistic-regression classifier in `fit` (gradient
//! descent is fine, no ML crate needed). `score` returns the sigmoid
//! probability in [0,1] — naturally calibrated confidence. This is expected to
//! be one of the strongest approaches. Use `crate::util::spectra` for spectra.

use crate::dataset::Sample;
use crate::Approach;

#[derive(Default)]
pub struct MfccLr {
    // TODO(approach-wave): weights, bias, feature mean/std.
}

impl MfccLr {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Approach for MfccLr {
    fn name(&self) -> &str {
        "mfcc_lr"
    }
    fn description(&self) -> &str {
        "MFCC features + logistic regression (stub)"
    }
    fn fit(&mut self, _train: &[Sample]) {
        // TODO(approach-wave): extract MFCCs, standardize, train logistic regression.
    }
    fn score(&self, _samples: &[f32], _sample_rate: u32) -> f32 {
        0.5 // TODO(approach-wave): replace with sigmoid(w·x + b)
    }
}
