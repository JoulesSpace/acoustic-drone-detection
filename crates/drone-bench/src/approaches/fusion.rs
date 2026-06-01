//! Ensemble / stacking meta-detector (insightface-style mesh).
//!
//! TODO(approach-wave-2): implement. Stack the classic base approaches
//! (`crate::approaches::stackable()`): in `fit`, split the train set, fit each
//! base on part A, collect their scores on part B as meta-features, train a
//! logistic meta-model on those (avoids leakage), then refit each base on the
//! full train set for final scoring. `score` queries each base for its
//! confidence and feeds the vector through the meta-model. Expected to be the
//! most robust single result. Return the meta sigmoid in [0,1].

use crate::Approach;

#[derive(Default)]
pub struct Fusion;

impl Fusion {
    pub fn new() -> Self {
        Self
    }
}

impl Approach for Fusion {
    fn name(&self) -> &str {
        "fusion"
    }
    fn description(&self) -> &str {
        "Ensemble: logistic stack over the classic approaches' confidences (stub)"
    }
    fn score(&self, _samples: &[f32], _sample_rate: u32) -> f32 {
        0.5
    }
}
