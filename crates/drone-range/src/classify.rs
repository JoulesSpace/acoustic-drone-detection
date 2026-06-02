//! Multinomial logistic regression over range bins (mode (b)).
//!
//! Continuous range is quantized into fixed-width bins (e.g. 10 or 15 m, like
//! Kang's 5-50 m bins), and a softmax classifier predicts the bin. Training is
//! deterministic batch gradient descent on the cross-entropy loss with L2
//! regularization, zero-initialized weights, no RNG - so no ML crate is needed
//! and runs are reproducible.

use alloc::vec;
use alloc::vec::Vec;

use libm::{expf, sqrtf};

/// A fixed-width binning of range into `[0, max)` plus an overflow top bin.
#[derive(Debug, Clone)]
pub struct RangeBins {
    /// Bin width in metres.
    pub width_m: f32,
    /// Number of bins (the last bin absorbs everything `>= (n-1)*width`).
    pub n_bins: usize,
}

impl RangeBins {
    /// Bins of `width_m` covering `[0, max_m)`, with a final overflow bin.
    pub fn new(width_m: f32, max_m: f32) -> Self {
        let n = libm::ceilf(max_m / width_m) as usize;
        Self {
            width_m,
            n_bins: n.max(1),
        }
    }

    /// Bin index for a range in metres (clamped to the top overflow bin).
    pub fn index_of(&self, range_m: f32) -> usize {
        let raw = libm::floorf(range_m / self.width_m) as isize;
        if raw < 0 {
            0
        } else {
            (raw as usize).min(self.n_bins - 1)
        }
    }

    /// Inclusive-lo / exclusive-hi label for a bin, in metres (`hi = inf` for
    /// the overflow bin).
    pub fn bounds(&self, bin: usize) -> (f32, f32) {
        let lo = bin as f32 * self.width_m;
        let hi = if bin + 1 == self.n_bins {
            f32::INFINITY
        } else {
            (bin + 1) as f32 * self.width_m
        };
        (lo, hi)
    }
}

/// Gradient-descent iterations.
const ITERS: usize = 800;
/// Learning rate.
const LR: f32 = 0.3;
/// L2 regularization strength.
const L2: f32 = 1e-3;

/// A fitted multinomial-logistic range-bin classifier.
#[derive(Debug, Clone)]
pub struct BinModel {
    /// Weight matrix: `n_bins` rows of `d` weights each.
    weights: Vec<Vec<f32>>,
    /// Per-class bias.
    bias: Vec<f32>,
    /// Per-feature train mean.
    feat_mean: Vec<f32>,
    /// Per-feature train std.
    feat_std: Vec<f32>,
    /// Number of classes (bins).
    n_bins: usize,
    /// Whether `fit` ran.
    fitted: bool,
}

impl BinModel {
    /// Fit on raw feature rows `x` and integer bin labels `labels` with `n_bins`
    /// classes. Features are standardized internally with train stats.
    pub fn fit(x: &[Vec<f32>], labels: &[usize], n_bins: usize) -> Self {
        if x.is_empty() || x[0].is_empty() || x.len() != labels.len() || n_bins == 0 {
            return Self {
                weights: Vec::new(),
                bias: Vec::new(),
                feat_mean: Vec::new(),
                feat_std: Vec::new(),
                n_bins: n_bins.max(1),
                fitted: false,
            };
        }
        let d = x[0].len();
        let n = x.len();

        // Standardization stats.
        let mut feat_mean = vec![0.0_f32; d];
        for row in x {
            for (m, &v) in feat_mean.iter_mut().zip(row.iter()) {
                *m += v;
            }
        }
        for m in feat_mean.iter_mut() {
            *m /= n as f32;
        }
        let mut feat_std = vec![0.0_f32; d];
        for row in x {
            for (s, (&v, &m)) in feat_std.iter_mut().zip(row.iter().zip(feat_mean.iter())) {
                let dv = v - m;
                *s += dv * dv;
            }
        }
        for s in feat_std.iter_mut() {
            let val = sqrtf(*s / n as f32);
            *s = if val > 1e-6 { val } else { 1.0 };
        }

        let xs: Vec<Vec<f32>> = x
            .iter()
            .map(|row| standardize(row, &feat_mean, &feat_std))
            .collect();

        let mut weights = vec![vec![0.0_f32; d]; n_bins];
        let mut bias = vec![0.0_f32; n_bins];

        for _ in 0..ITERS {
            let mut grad_w = vec![vec![0.0_f32; d]; n_bins];
            let mut grad_b = vec![0.0_f32; n_bins];
            for (row, &lbl) in xs.iter().zip(labels.iter()) {
                let probs = softmax_logits(&weights, &bias, row);
                for k in 0..n_bins {
                    let target = if k == lbl { 1.0 } else { 0.0 };
                    let err = probs[k] - target;
                    grad_b[k] += err;
                    for (g, &xj) in grad_w[k].iter_mut().zip(row.iter()) {
                        *g += err * xj;
                    }
                }
            }
            let inv_n = 1.0 / n as f32;
            for k in 0..n_bins {
                for (wj, gj) in weights[k].iter_mut().zip(grad_w[k].iter()) {
                    let grad = gj * inv_n + L2 * *wj;
                    *wj -= LR * grad;
                }
                bias[k] -= LR * (grad_b[k] * inv_n);
            }
        }

        Self {
            weights,
            bias,
            feat_mean,
            feat_std,
            n_bins,
            fitted: true,
        }
    }

    /// Predict the most likely bin index for a raw feature vector.
    pub fn predict(&self, raw: &[f32]) -> usize {
        if !self.fitted {
            return 0;
        }
        let xs = standardize(raw, &self.feat_mean, &self.feat_std);
        let probs = softmax_logits(&self.weights, &self.bias, &xs);
        let mut best = 0;
        for k in 1..self.n_bins {
            if probs[k] > probs[best] {
                best = k;
            }
        }
        best
    }

    /// Full posterior over bins for a raw feature vector.
    pub fn predict_proba(&self, raw: &[f32]) -> Vec<f32> {
        if !self.fitted {
            return vec![1.0 / self.n_bins as f32; self.n_bins];
        }
        let xs = standardize(raw, &self.feat_mean, &self.feat_std);
        softmax_logits(&self.weights, &self.bias, &xs)
    }
}

/// Standardize a raw vector with stored stats.
fn standardize(raw: &[f32], mean: &[f32], std: &[f32]) -> Vec<f32> {
    raw.iter()
        .zip(mean.iter().zip(std.iter()))
        .map(|(&v, (&m, &s))| (v - m) / s)
        .collect()
}

/// Numerically stable softmax of the class logits `w_k · x + b_k`.
fn softmax_logits(weights: &[Vec<f32>], bias: &[f32], x: &[f32]) -> Vec<f32> {
    let k = bias.len();
    let mut logits = vec![0.0_f32; k];
    let mut max_logit = f32::NEG_INFINITY;
    for c in 0..k {
        let mut z = bias[c];
        for (wj, &xj) in weights[c].iter().zip(x.iter()) {
            z += wj * xj;
        }
        logits[c] = z;
        if z > max_logit {
            max_logit = z;
        }
    }
    let mut sum = 0.0_f32;
    for z in logits.iter_mut() {
        *z = expf(*z - max_logit);
        sum += *z;
    }
    if sum > 0.0 {
        for z in logits.iter_mut() {
            *z /= sum;
        }
    }
    logits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bins_index_and_overflow() {
        let b = RangeBins::new(10.0, 50.0); // bins: [0,10),[10,20),...,[40,50)
        assert_eq!(b.n_bins, 5);
        assert_eq!(b.index_of(0.0), 0);
        assert_eq!(b.index_of(15.0), 1);
        assert_eq!(b.index_of(49.0), 4);
        assert_eq!(b.index_of(1000.0), 4); // overflow clamps to top bin
    }

    #[test]
    fn separable_classes_are_learned() {
        // Two well-separated 1-D clusters -> perfect classification.
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..10 {
            x.push(vec![i as f32 * 0.1]); // ~0..0.9 -> class 0
            y.push(0usize);
            x.push(vec![5.0 + i as f32 * 0.1]); // ~5..5.9 -> class 1
            y.push(1usize);
        }
        let model = BinModel::fit(&x, &y, 2);
        assert_eq!(model.predict(&[0.2]), 0);
        assert_eq!(model.predict(&[5.5]), 1);
    }

    #[test]
    fn proba_sums_to_one() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let y = vec![0usize, 0, 1, 1];
        let model = BinModel::fit(&x, &y, 2);
        let p = model.predict_proba(&[1.5]);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-4);
    }

    #[test]
    fn unfitted_predicts_zero_bin() {
        let model = BinModel::fit(&[], &[], 3);
        assert_eq!(model.predict(&[1.0]), 0);
    }
}
