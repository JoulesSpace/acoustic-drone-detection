//! Multinomial logistic regression (softmax) for multi-vendor recognition.
//!
//! Clip features ([`crate::features`]) are standardized with the train-set
//! mean/std (stored on the model), then a `K`-way softmax classifier is fit by
//! deterministic batch gradient descent on the cross-entropy loss with L2
//! regularization. There is no RNG and no ML crate: weights start at zero, so a
//! given dataset always trains to the same model - the bit-for-bit determinism
//! the repo asks for.
//!
//! Unlike the fixed-width `drone-id` head, the feature dimension here is chosen
//! at runtime by the spectral-feature toggle, so weights and statistics are
//! `Vec`-backed and sized from [`crate::features::feature_len`].

use crate::features::{clip_features, feature_len, N_FEAT_MAX};
use drone_bench::dataset::Sample;

/// Gradient-descent iterations.
const ITERS: usize = 900;
/// Learning rate.
const LR: f32 = 0.5;
/// L2 regularization strength (on weights, not biases).
const L2: f32 = 1e-3;

/// A fitted (or default) softmax classifier over `K` classes of `n_feat`-dim
/// standardized clip features.
pub struct SoftmaxClassifier {
    n_classes: usize,
    /// Active feature dimension (depends on the spectral toggle).
    n_feat: usize,
    /// Whether the spectral/harmonic block is included in the features.
    use_spectral: bool,
    /// Row-major weights: `weights[c * n_feat + j]` for class `c`, feature `j`.
    weights: Vec<f32>,
    /// Per-class bias.
    bias: Vec<f32>,
    /// Per-feature mean from the train set (for standardization).
    feat_mean: Vec<f32>,
    /// Per-feature standard deviation from the train set.
    feat_std: Vec<f32>,
    fitted: bool,
}

impl SoftmaxClassifier {
    /// A fresh, unfitted classifier for `n_classes` classes. `use_spectral`
    /// selects whether the spectral/harmonic descriptors join the MFCC block.
    pub fn new(n_classes: usize, use_spectral: bool) -> Self {
        let n_feat = feature_len(use_spectral);
        Self {
            n_classes,
            n_feat,
            use_spectral,
            weights: vec![0.0; n_classes * n_feat],
            bias: vec![0.0; n_classes],
            feat_mean: vec![0.0; n_feat],
            feat_std: vec![1.0; n_feat],
            fitted: false,
        }
    }

    /// Number of classes the model predicts over.
    pub fn n_classes(&self) -> usize {
        self.n_classes
    }

    /// Active feature dimension.
    pub fn n_feat(&self) -> usize {
        self.n_feat
    }

    /// Extract the active-length feature slice for a clip.
    fn raw_features(&self, samples: &[f32], sample_rate: u32) -> Vec<f32> {
        let full: [f32; N_FEAT_MAX] = clip_features(samples, sample_rate, self.use_spectral);
        full[..self.n_feat].to_vec()
    }

    /// Fit on labelled training clips. Labels are class ids `0..n_classes`.
    pub fn fit(&mut self, train: &[Sample]) {
        let feats: Vec<Vec<f32>> = train
            .iter()
            .map(|s| self.raw_features(&s.samples, s.sample_rate))
            .collect();
        let labels: Vec<usize> = train.iter().map(|s| s.label as usize).collect();
        if feats.is_empty() {
            return;
        }

        // Standardization statistics over the train set.
        let n = feats.len() as f32;
        let nf = self.n_feat;
        let mut mean = vec![0.0f32; nf];
        for f in &feats {
            for (m, &v) in mean.iter_mut().zip(f.iter()) {
                *m += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n;
        }
        let mut var = vec![0.0f32; nf];
        for f in &feats {
            for (i, vacc) in var.iter_mut().enumerate() {
                let d = f[i] - mean[i];
                *vacc += d * d;
            }
        }
        let mut std = vec![1.0f32; nf];
        for (s, &v) in std.iter_mut().zip(var.iter()) {
            let val = (v / n).sqrt();
            *s = if val > 1e-6 { val } else { 1.0 };
        }
        self.feat_mean = mean;
        self.feat_std = std;

        let x: Vec<Vec<f32>> = feats.iter().map(|f| self.standardize(f)).collect();

        let k = self.n_classes;
        let mut w = vec![0.0f32; k * nf];
        let mut b = vec![0.0f32; k];
        for _ in 0..ITERS {
            let mut grad_w = vec![0.0f32; k * nf];
            let mut grad_b = vec![0.0f32; k];
            for (xi, &yi) in x.iter().zip(labels.iter()) {
                let p = softmax_probs(&w, &b, xi, k, nf);
                for c in 0..k {
                    // dL/dz_c = p_c - 1{y == c}
                    let err = p[c] - if c == yi { 1.0 } else { 0.0 };
                    let base = c * nf;
                    for (j, &xj) in xi.iter().enumerate() {
                        grad_w[base + j] += err * xj;
                    }
                    grad_b[c] += err;
                }
            }
            for c in 0..k {
                let base = c * nf;
                for j in 0..nf {
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
        let raw = self.raw_features(samples, sample_rate);
        let x = self.standardize(&raw);
        softmax_probs(&self.weights, &self.bias, &x, k, self.n_feat)
    }

    /// Predict the argmax class id for a clip.
    pub fn predict(&self, samples: &[f32], sample_rate: u32) -> usize {
        let p = self.predict_proba(samples, sample_rate);
        argmax(&p)
    }

    /// Standardize a raw feature vector with the stored train statistics.
    fn standardize(&self, raw: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.n_feat];
        for (i, o) in out.iter_mut().enumerate() {
            *o = (raw[i] - self.feat_mean[i]) / self.feat_std[i];
        }
        out
    }
}

/// Numerically stable softmax over the `K` class logits `z_c = w_c . x + b_c`.
fn softmax_probs(w: &[f32], b: &[f32], x: &[f32], k: usize, nf: usize) -> Vec<f32> {
    let mut z = vec![0.0f32; k];
    for (c, zc) in z.iter_mut().enumerate() {
        let base = c * nf;
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
    use crate::data::VendorDataset;

    #[test]
    fn probs_are_a_distribution() {
        let ds = VendorDataset::synth(8, 16_000, 3);
        let mut clf = SoftmaxClassifier::new(ds.n_classes(), true);
        clf.fit(&ds.samples);
        let p = clf.predict_proba(&ds.samples[0].samples, ds.samples[0].sample_rate);
        assert_eq!(p.len(), ds.n_classes());
        let s: f32 = p.iter().sum();
        assert!((s - 1.0).abs() < 1e-3, "probs sum to {s}");
        assert!(p.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn learns_synthetic_train_set() {
        let ds = VendorDataset::synth(12, 16_000, 9);
        let mut clf = SoftmaxClassifier::new(ds.n_classes(), true);
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
    fn deterministic_fit() {
        let ds = VendorDataset::synth(6, 16_000, 4);
        let mut a = SoftmaxClassifier::new(ds.n_classes(), true);
        let mut b = SoftmaxClassifier::new(ds.n_classes(), true);
        a.fit(&ds.samples);
        b.fit(&ds.samples);
        let pa = a.predict_proba(&ds.samples[0].samples, ds.samples[0].sample_rate);
        let pb = b.predict_proba(&ds.samples[0].samples, ds.samples[0].sample_rate);
        assert_eq!(pa, pb);
    }

    #[test]
    fn unfitted_is_uniform() {
        let clf = SoftmaxClassifier::new(5, false);
        let p = clf.predict_proba(&[0.1, 0.2, 0.3], 16_000);
        assert!(p.iter().all(|&x| (x - 0.2).abs() < 1e-6));
    }
}
