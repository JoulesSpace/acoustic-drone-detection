//! Ensemble / stacking meta-detector (insightface-style mesh).
//!
//! Stacks the six classic base approaches from [`crate::approaches::stackable`]
//! (band_ratio, template, hps, spectral_gate, cepstrum, mfcc_lr) under a
//! logistic-regression meta-model.
//!
//! Training (`fit`):
//! 1. Deterministically split the train set into part A (~70%) and part B
//!    (~30%) with an index/stride-based stratified split (no RNG), keeping both
//!    classes in each part.
//! 2. Fit a fresh set of bases on part A.
//! 3. For each clip in part B, build a meta-feature vector of the bases'
//!    confidences. Training the meta-model on B (held out from the bases) avoids
//!    the leakage that fitting the meta-model on the bases' own training data
//!    would cause.
//! 4. Fit a logistic-regression meta-model on B's meta-features vs labels by
//!    batch gradient descent with L2, deterministic zero init.
//! 5. Refit a fresh set of bases on the FULL train set and store them, so final
//!    scoring uses bases trained on all available data.
//!
//! `score` queries each stored base for its confidence, then returns the meta
//! sigmoid `sigmoid(w·x + b)` clamped to `[0, 1]`. Unfit → 0.5.

use crate::dataset::Sample;
use crate::Approach;

/// Gradient-descent iterations for the meta-model.
const ITERS: usize = 4000;
/// Learning rate for the meta-model.
const LR: f32 = 2.0;
/// L2 regularization strength for the meta-model.
const L2: f32 = 1e-4;
/// Fraction of the train set used to fit the bases (part A); the rest (part B)
/// trains the meta-model.
const PART_A_STRIDE: usize = 10;
/// Items with `index % PART_A_STRIDE < PART_A_CUT` go to part A (~70%).
const PART_A_CUT: usize = 7;

/// Logistic-regression meta-model stacking the classic base approaches.
pub struct Fusion {
    /// Base approaches, fitted on the full train set, used at scoring time.
    bases: Vec<Box<dyn Approach>>,
    /// Meta-model weights, one per base.
    weights: Vec<f32>,
    /// Meta-model bias.
    bias: f32,
    /// Whether `fit` has completed successfully.
    fitted: bool,
}

impl Default for Fusion {
    fn default() -> Self {
        Self {
            bases: Vec::new(),
            weights: Vec::new(),
            bias: 0.0,
            fitted: false,
        }
    }
}

impl Fusion {
    pub fn new() -> Self {
        Self::default()
    }

    /// Meta-feature vector for a clip: each base's confidence in order.
    fn meta_features(bases: &[Box<dyn Approach>], samples: &[f32], sample_rate: u32) -> Vec<f32> {
        bases
            .iter()
            .map(|b| b.score(samples, sample_rate).clamp(0.0, 1.0))
            .collect()
    }
}

impl Approach for Fusion {
    fn name(&self) -> &str {
        "fusion"
    }

    fn description(&self) -> &str {
        "Ensemble: logistic stack over the classic approaches' confidences"
    }

    fn fit(&mut self, train: &[Sample]) {
        if train.is_empty() {
            return;
        }

        // --- Stratified, deterministic A/B split (no RNG) ---------------------
        // Within each class, walk samples in order and assign by a fixed stride
        // pattern so both A and B retain both classes when present.
        let mut a: Vec<Sample> = Vec::new();
        let mut b: Vec<Sample> = Vec::new();
        for class in [0u8, 1u8] {
            for (rank, s) in train.iter().filter(|s| s.label == class).enumerate() {
                if rank % PART_A_STRIDE < PART_A_CUT {
                    a.push(s.clone());
                } else {
                    b.push(s.clone());
                }
            }
        }
        // Guard against a degenerate split (e.g. a class with too few samples):
        // fall back to using the full train set for both stages.
        if a.is_empty() || b.is_empty() {
            a = train.to_vec();
            b = train.to_vec();
        }

        // --- Fit bases on part A, collect meta-features on part B -------------
        let mut bases_a = crate::approaches::stackable();
        for base in bases_a.iter_mut() {
            base.fit(&a);
        }
        let n_bases = bases_a.len();

        let x: Vec<Vec<f32>> = b
            .iter()
            .map(|s| Self::meta_features(&bases_a, &s.samples, s.sample_rate))
            .collect();
        let labels: Vec<f32> = b.iter().map(|s| s.label as f32).collect();
        let n = x.len() as f32;

        // --- Train logistic meta-model on B (deterministic zero init) --------
        let mut w = vec![0.0f32; n_bases];
        let mut bias = 0.0f32;
        for _ in 0..ITERS {
            let mut grad_w = vec![0.0f32; n_bases];
            let mut grad_b = 0.0f32;
            for (xi, &yi) in x.iter().zip(labels.iter()) {
                let mut z = bias;
                for (wj, &xj) in w.iter().zip(xi.iter()) {
                    z += wj * xj;
                }
                let err = sigmoid(z) - yi;
                for (g, &xj) in grad_w.iter_mut().zip(xi.iter()) {
                    *g += err * xj;
                }
                grad_b += err;
            }
            for (wj, gj) in w.iter_mut().zip(grad_w.iter()) {
                let grad = gj / n + L2 * *wj;
                *wj -= LR * grad;
            }
            bias -= LR * (grad_b / n);
        }

        // --- Refit fresh bases on the FULL train set for final scoring -------
        let mut bases_full = crate::approaches::stackable();
        for base in bases_full.iter_mut() {
            base.fit(train);
        }

        self.bases = bases_full;
        self.weights = w;
        self.bias = bias;
        self.fitted = true;
    }

    fn score(&self, samples: &[f32], sample_rate: u32) -> f32 {
        if !self.fitted || self.bases.is_empty() {
            return 0.5;
        }
        let x = Self::meta_features(&self.bases, samples, sample_rate);
        let mut z = self.bias;
        for (w, xi) in self.weights.iter().zip(x.iter()) {
            z += w * xi;
        }
        sigmoid(z).clamp(0.0, 1.0)
    }
}

/// Numerically stable logistic sigmoid.
#[inline]
fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}
