//! MFCC + logistic-regression classifier.
//!
//! Front-end: each frame's 512-bin Hann magnitude spectrum is passed through a
//! mel filterbank (`N_MELS` triangular filters spanning ~0..sr/2), log-compressed,
//! then a DCT-II yields `N_MFCC` cepstral coefficients. Per clip we pool the
//! frame-wise MFCCs into mean and std vectors and append the mean log-energy, for
//! a fixed-length clip feature vector.
//!
//! Training (`fit`): features are standardized using the train-set mean/std
//! (stored on the struct), then a logistic-regression classifier is fit by
//! batch gradient descent with L2 regularization. `score` standardizes the clip
//! feature with the stored statistics and returns `sigmoid(w·x + b)`, a
//! naturally calibrated probability in `[0, 1]`.
//!
//! Everything is deterministic (zero-initialized weights, no RNG) and pure Rust.

use crate::dataset::Sample;
use crate::util::spectra;
use crate::Approach;
use drone_dsp::{bin_to_hz, NUM_BINS};

/// Number of mel filterbank channels.
const N_MELS: usize = 26;
/// Number of MFCC coefficients kept (including c0).
const N_MFCC: usize = 13;
/// Feature vector length: mean + std of each MFCC, plus mean log-energy.
const N_FEAT: usize = 2 * N_MFCC + 1;

/// Gradient-descent iterations.
const ITERS: usize = 600;
/// Learning rate.
const LR: f32 = 0.5;
/// L2 regularization strength.
const L2: f32 = 1e-3;

pub struct MfccLr {
    /// Logistic-regression weights, one per standardized feature.
    weights: [f32; N_FEAT],
    /// Bias term.
    bias: f32,
    /// Per-feature mean from the train set (for standardization).
    feat_mean: [f32; N_FEAT],
    /// Per-feature standard deviation from the train set.
    feat_std: [f32; N_FEAT],
    /// Whether `fit` has been called.
    fitted: bool,
}

impl Default for MfccLr {
    fn default() -> Self {
        Self {
            weights: [0.0; N_FEAT],
            bias: 0.0,
            feat_mean: [0.0; N_FEAT],
            feat_std: [1.0; N_FEAT],
            fitted: false,
        }
    }
}

impl MfccLr {
    pub fn new() -> Self {
        Self::default()
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

impl Approach for MfccLr {
    fn name(&self) -> &str {
        "mfcc_lr"
    }

    fn description(&self) -> &str {
        "MFCC features (mean/std pooled) + logistic regression"
    }

    fn fit(&mut self, train: &[Sample]) {
        // Extract raw clip features for every training sample.
        let feats: Vec<[f32; N_FEAT]> = train
            .iter()
            .map(|s| clip_features(&s.samples, s.sample_rate))
            .collect();
        let labels: Vec<f32> = train.iter().map(|s| s.label as f32).collect();
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

        // Standardize all training features once.
        let x: Vec<[f32; N_FEAT]> = feats.iter().map(|f| self.standardize(f)).collect();

        // Batch gradient descent on the logistic loss with L2 on weights.
        let mut w = [0.0f32; N_FEAT];
        let mut b = 0.0f32;
        for _ in 0..ITERS {
            let mut grad_w = [0.0f32; N_FEAT];
            let mut grad_b = 0.0f32;
            for (xi, &yi) in x.iter().zip(labels.iter()) {
                let mut z = b;
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
            b -= LR * (grad_b / n);
        }
        self.weights = w;
        self.bias = b;
        self.fitted = true;
    }

    fn score(&self, samples: &[f32], sample_rate: u32) -> f32 {
        if !self.fitted {
            return 0.5;
        }
        let raw = clip_features(samples, sample_rate);
        let x = self.standardize(&raw);
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

/// Compute the clip-level MFCC feature vector: mean and std of each MFCC across
/// frames, plus the mean log filterbank energy. Returns a zero vector for empty
/// or silent input so callers degrade gracefully.
fn clip_features(samples: &[f32], sample_rate: u32) -> [f32; N_FEAT] {
    let mut out = [0.0f32; N_FEAT];
    let frames = spectra(samples);
    if frames.is_empty() {
        return out;
    }

    let fb = mel_filterbank(sample_rate);
    let dct = dct_matrix();

    // Accumulate frame-wise MFCCs into running mean / mean-of-squares.
    let mut sum = [0.0f32; N_MFCC];
    let mut sum_sq = [0.0f32; N_MFCC];
    let mut sum_log_energy = 0.0f32;
    let mut count = 0.0f32;

    for sp in &frames {
        // Mel energies via the triangular filterbank.
        let mut log_mel = [0.0f32; N_MELS];
        for (m, filt) in fb.iter().enumerate() {
            let mut e = 0.0f32;
            for &(bin, w) in filt {
                // Use power (magnitude squared), standard for MFCCs.
                e += w * sp[bin] * sp[bin];
            }
            // log of energy with a floor to avoid -inf on silence.
            log_mel[m] = (e + 1e-10).ln();
        }

        // DCT-II of the log-mel energies to get cepstral coefficients.
        for k in 0..N_MFCC {
            let mut c = 0.0f32;
            for m in 0..N_MELS {
                c += dct[k][m] * log_mel[m];
            }
            sum[k] += c;
            sum_sq[k] += c * c;
        }

        // Mean log-energy across mel channels for this frame.
        let mut frame_log_energy = 0.0f32;
        for &v in &log_mel {
            frame_log_energy += v;
        }
        sum_log_energy += frame_log_energy / N_MELS as f32;
        count += 1.0;
    }

    for k in 0..N_MFCC {
        let mean = sum[k] / count;
        let var = (sum_sq[k] / count - mean * mean).max(0.0);
        out[k] = mean;
        out[N_MFCC + k] = var.sqrt();
    }
    out[N_FEAT - 1] = sum_log_energy / count;
    out
}

/// Build a mel filterbank as a list of `(bin, weight)` pairs per channel.
///
/// `N_MELS` triangular filters are spaced equally on the mel scale between 0 Hz
/// and the Nyquist frequency, then mapped back to the linear FFT bins.
fn mel_filterbank(sample_rate: u32) -> Vec<Vec<(usize, f32)>> {
    let f_max = sample_rate as f32 / 2.0;
    let mel_max = hz_to_mel(f_max);

    // `N_MELS + 2` mel points define the edges/centers of the filters.
    let n_points = N_MELS + 2;
    let mut centers_hz = [0.0f32; N_MELS + 2];
    for (i, c) in centers_hz.iter_mut().enumerate() {
        let mel = mel_max * i as f32 / (n_points - 1) as f32;
        *c = mel_to_hz(mel);
    }

    let mut fb: Vec<Vec<(usize, f32)>> = Vec::with_capacity(N_MELS);
    for m in 0..N_MELS {
        let lo = centers_hz[m];
        let ctr = centers_hz[m + 1];
        let hi = centers_hz[m + 2];
        let mut filt = Vec::new();
        for bin in 0..NUM_BINS {
            let f = bin_to_hz(bin, sample_rate);
            let w = if f >= lo && f <= ctr && ctr > lo {
                (f - lo) / (ctr - lo)
            } else if f > ctr && f <= hi && hi > ctr {
                (hi - f) / (hi - ctr)
            } else {
                0.0
            };
            if w > 0.0 {
                filt.push((bin, w));
            }
        }
        fb.push(filt);
    }
    fb
}

/// Precompute the DCT-II basis matrix (`N_MFCC` x `N_MELS`).
fn dct_matrix() -> [[f32; N_MELS]; N_MFCC] {
    let mut dct = [[0.0f32; N_MELS]; N_MFCC];
    let scale = (2.0f32 / N_MELS as f32).sqrt();
    for (k, row) in dct.iter_mut().enumerate() {
        for (m, val) in row.iter_mut().enumerate() {
            *val =
                scale * (core::f32::consts::PI / N_MELS as f32 * (m as f32 + 0.5) * k as f32).cos();
        }
    }
    dct
}

/// Hz → mel (Slaney/HTK-style 2595*log10(1+f/700)).
#[inline]
fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

/// Mel → Hz (inverse of [`hz_to_mel`]).
#[inline]
fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10.0f32.powf(m / 2595.0) - 1.0)
}
