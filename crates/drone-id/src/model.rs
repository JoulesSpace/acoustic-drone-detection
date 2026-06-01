//! Multinomial logistic regression (softmax) for drone-type recognition.
//!
//! Clip MFCC features ([`crate::features`]) are standardized with the train-set
//! mean/std (stored on the model), then a `K`-way softmax classifier is fit by
//! deterministic batch gradient descent on the cross-entropy loss with L2
//! regularization. There is no RNG and no ML crate: weights start at zero, so a
//! given dataset always trains to the same model.

use crate::features::{clip_features, N_FEAT};
use drone_bench::dataset::Sample;

/// Gradient-descent iterations.
const ITERS: usize = 800;
/// Learning rate.
const LR: f32 = 0.5;
/// L2 regularization strength (on weights, not biases).
const L2: f32 = 1e-3;

/// A fitted (or default) softmax classifier over `K` classes of `N_FEAT`-dim
/// standardized MFCC features.
pub struct SoftmaxClassifier {
    n_classes: usize,
    /// Row-major weights: `weights[c * N_FEAT + j]` for class `c`, feature `j`.
    weights: Vec<f32>,
    /// Per-class bias.
    bias: Vec<f32>,
    /// Per-feature mean from the train set (for standardization).
    feat_mean: [f32; N_FEAT],
    /// Per-feature standard deviation from the train set.
    feat_std: [f32; N_FEAT],
    fitted: bool,
}

impl SoftmaxClassifier {
    /// A fresh, unfitted classifier for `n_classes` classes.
    pub fn new(n_classes: usize) -> Self {
        Self {
            n_classes,
            weights: vec![0.0; n_classes * N_FEAT],
            bias: vec![0.0; n_classes],
            feat_mean: [0.0; N_FEAT],
            feat_std: [1.0; N_FEAT],
            fitted: false,
        }
    }

    /// Number of classes the model predicts over.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Fit on labelled training clips. Labels are class ids `0..n_classes`.
    pub fn fit(&mut self, train: &[Sample]) {
        let feats: Vec<[f32; N_FEAT]> = train
            .iter()
            .map(|s| clip_features(&s.samples, s.sample_rate))
            .collect();
        let labels: Vec<usize> = train.iter().map(|s| s.label as usize).collect();
        if feats.is_empty() {
            return;
        }

        // Standardization statistics over the train set.
        let n = feats.len() as f32;
        let mut mean = [0.0f32; N_FEAT];
        for f in &feats {
            for (m, &v) in mean.iter_mut().zip(f.iter()) {
                *m += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n;
        }
        let mut var = [0.0f32; N_FEAT];
        for f in &feats {
            for (i, vacc) in var.iter_mut().enumerate() {
                let d = f[i] - mean[i];
                *vacc += d * d;
            }
        }
        let mut std = [1.0f32; N_FEAT];
        for (s, &v) in std.iter_mut().zip(var.iter()) {
            let val = (v / n).sqrt();
            *s = if val > 1e-6 { val } else { 1.0 };
        }
        self.feat_mean = mean;
        self.feat_std = std;

        let x: Vec<[f32; N_FEAT]> = feats.iter().map(|f| self.standardize(f)).collect();

        let k = self.n_classes;
        let mut w = vec![0.0f32; k * N_FEAT];
        let mut b = vec![0.0f32; k];
        for _ in 0..ITERS {
            let mut grad_w = vec![0.0f32; k * N_FEAT];
            let mut grad_b = vec![0.0f32; k];
            for (xi, &yi) in x.iter().zip(labels.iter()) {
                let p = softmax_probs(&w, &b, xi, k);
                for c in 0..k {
                    // dL/dz_c = p_c - 1{y == c}
                    let err = p[c] - if c == yi { 1.0 } else { 0.0 };
                    let base = c * N_FEAT;
                    for (j, &xj) in xi.iter().enumerate() {
                        grad_w[base + j] += err * xj;
                    }
                    grad_b[c] += err;
                }
            }
            for c in 0..k {
                let base = c * N_FEAT;
                for j in 0..N_FEAT {
                    let idx = base + j;
                    let grad = grad_w[idx] / n + L2 * w[idx];
                    w[idx] -= LR * grad;
                }
                b[c] -= LR * (grad_b[c] / n);
            }
        }
        self.weights = w;
        self.bias = b;
        self.fitted = true;
    }

    /// Predict the class probability vector for a clip (length `n_classes`).
    /// Returns a uniform distribution if the model is unfitted.
    pub fn predict_proba(&self, samples: &[f32], sample_rate: u32) -> Vec<f32> {
        let k = self.n_classes;
        if !self.fitted {
            return vec![1.0 / k as f32; k];
        }
        let raw = clip_features(samples, sample_rate);
        let x = self.standardize(&raw);
        softmax_probs(&self.weights, &self.bias, &x, k)
    }

    /// Predict the argmax class id for a clip.
    pub fn predict(&self, samples: &[f32], sample_rate: u32) -> usize {
        let p = self.predict_proba(samples, sample_rate);
        argmax(&p)
    }

    /// Standardize a raw feature vector with the stored train statistics.
    fn standardize(&self, raw: &[f32; N_FEAT]) -> [f32; N_FEAT] {
        let mut out = [0.0; N_FEAT];
        for (i, o) in out.iter_mut().enumerate() {
            *o = (raw[i] - self.feat_mean[i]) / self.feat_std[i];
        }
        out
    }
}

/// Numerically stable softmax over the `K` class logits `z_c = w_c . x + b_c`.
fn softmax_probs(w: &[f32], b: &[f32], x: &[f32; N_FEAT], k: usize) -> Vec<f32> {
    let mut z = vec![0.0f32; k];
    for (c, zc) in z.iter_mut().enumerate() {
        let base = c * N_FEAT;
        let mut acc = b[c];
        for (j, &xj) in x.iter().enumerate() {
            acc += w[base + j] * xj;
        }
        *zc = acc;
    }
    let zmax = z.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for zc in z.iter_mut() {
        *zc = (*zc - zmax).exp();
        sum += *zc;
    }
    if sum > 0.0 {
        for zc in z.iter_mut() {
            *zc /= sum;
        }
    }
    z
}

/// Index of the maximum element (first on ties).
pub fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    let mut bestv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bestv {
            bestv = x;
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::MultiDataset;

    #[test]
    fn probs_are_a_distribution() {
        let ds = MultiDataset::synth(10, 16_000, 3);
        let mut clf = SoftmaxClassifier::new(ds.n_classes());
        clf.fit(&ds.samples);
        let p = clf.predict_proba(&ds.samples[0].samples, ds.samples[0].sample_rate);
        assert_eq!(p.len(), 4);
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "probs sum to {s}");
        assert!(p.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn learns_synthetic_train_set() {
        // On the easy synthetic set the model should fit the training data well.
        let ds = MultiDataset::synth(15, 16_000, 9);
        let mut clf = SoftmaxClassifier::new(ds.n_classes());
        clf.fit(&ds.samples);
        let correct = ds
            .samples
            .iter()
            .filter(|s| clf.predict(&s.samples, s.sample_rate) == s.label as usize)
            .count();
        let acc = correct as f32 / ds.samples.len() as f32;
        assert!(acc > 0.9, "train accuracy too low: {acc}");
    }

    #[test]
    fn unfitted_is_uniform() {
        let clf = SoftmaxClassifier::new(4);
        let p = clf.predict_proba(&[0.1, 0.2, 0.3], 16_000);
        assert!(p.iter().all(|&x| (x - 0.25).abs() < 1e-6));
    }
}
