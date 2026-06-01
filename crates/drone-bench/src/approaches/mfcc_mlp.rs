//! MFCC + small multi-layer perceptron.
//!
//! TODO(approach-wave-2): implement. Nonlinear upgrade of `mfcc_lr` — same MFCC
//! clip features, but a 1-hidden-layer MLP (ReLU/tanh) trained by backprop in
//! `fit`. `score` returns the sigmoid output in [0,1].

use crate::Approach;

#[derive(Default)]
pub struct MfccMlp;

impl MfccMlp {
    pub fn new() -> Self {
        Self
    }
}

impl Approach for MfccMlp {
    fn name(&self) -> &str {
        "mfcc_mlp"
    }
    fn description(&self) -> &str {
        "MFCC features + small MLP (stub)"
    }
    fn score(&self, _samples: &[f32], _sample_rate: u32) -> f32 {
        0.5
    }
}
