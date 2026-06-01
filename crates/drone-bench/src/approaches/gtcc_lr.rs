//! GTCC (gammatone cepstral coefficients) + logistic-regression classifier.
//!
//! Front-end: each frame's 512-bin Hann magnitude spectrum is passed through a
//! gammatone filterbank (`N_GAMMA` filters whose center frequencies are spaced
//! equally on the ERB scale between ~50 Hz and the Nyquist), log-compressed,
//! then a DCT-II yields `N_GTCC` cepstral coefficients. This mirrors MFCC but
//! swaps the mel triangular bank for an ERB-spaced gammatone-shaped bank, which
//! the literature reports can beat MFCC on drone audio. Per clip we pool the
//! frame-wise GTCCs into mean and std vectors and append the mean log-energy,
//! for a fixed-length clip feature vector.
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

/// Number of gammatone filterbank channels.
const N_GAMMA: usize = 26;
/// Number of GTCC coefficients kept (including c0).
const N_GTCC: usize = 13;
/// Feature vector length: mean + std of each GTCC, plus mean log-energy.
const N_FEAT: usize = 2 * N_GTCC + 1;

/// Lowest gammatone center frequency (Hz).
const F_LOW: f32 = 50.0;

/// Gradient-descent iterations.
const ITERS: usize = 600;
/// Learning rate.
const LR: f32 = 0.5;
/// L2 regularization strength.
const L2: f32 = 1e-3;

pub struct GtccLr {
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

impl Default for GtccLr {
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

impl GtccLr {
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

impl Approach for GtccLr {
    fn name(&self) -> &str {
        "gtcc_lr"
    }

    fn description(&self) -> &str {
        "Gammatone cepstral coefficients (mean/std pooled) + logistic regression"
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

/// Compute the clip-level GTCC feature vector: mean and std of each GTCC across
/// frames, plus the mean log filterbank energy. Returns a zero vector for empty
/// or silent input so callers degrade gracefully.
fn clip_features(samples: &[f32], sample_rate: u32) -> [f32; N_FEAT] {
    let mut out = [0.0f32; N_FEAT];
    let frames = spectra(samples);
    if frames.is_empty() {
        return out;
    }

    let fb = gammatone_filterbank(sample_rate);
    let dct = dct_matrix();

    // Accumulate frame-wise GTCCs into running mean / mean-of-squares.
    let mut sum = [0.0f32; N_GTCC];
    let mut sum_sq = [0.0f32; N_GTCC];
    let mut sum_log_energy = 0.0f32;
    let mut count = 0.0f32;

    for sp in &frames {
        // Gammatone energies via the filterbank.
        let mut log_gamma = [0.0f32; N_GAMMA];
        for (m, filt) in fb.iter().enumerate() {
            let mut e = 0.0f32;
            for &(bin, w) in filt {
                // Use power (magnitude squared), as in standard cepstral pipelines.
                e += w * sp[bin] * sp[bin];
            }
            // log of energy with a floor to avoid -inf on silence.
            log_gamma[m] = (e + 1e-10).ln();
        }

        // DCT-II of the log gammatone energies to get cepstral coefficients.
        for k in 0..N_GTCC {
            let mut c = 0.0f32;
            for m in 0..N_GAMMA {
                c += dct[k][m] * log_gamma[m];
            }
            sum[k] += c;
            sum_sq[k] += c * c;
        }

        // Mean log-energy across gammatone channels for this frame.
        let mut frame_log_energy = 0.0f32;
        for &v in &log_gamma {
            frame_log_energy += v;
        }
        sum_log_energy += frame_log_energy / N_GAMMA as f32;
        count += 1.0;
    }

    for k in 0..N_GTCC {
        let mean = sum[k] / count;
        let var = (sum_sq[k] / count - mean * mean).max(0.0);
        out[k] = mean;
        out[N_GTCC + k] = var.sqrt();
    }
    out[N_FEAT - 1] = sum_log_energy / count;
    out
}

/// Build a gammatone filterbank as a list of `(bin, weight)` pairs per channel.
///
/// `N_GAMMA` filters have center frequencies spaced equally on the ERB scale
/// between [`F_LOW`] and the Nyquist frequency. Each filter's magnitude response
/// is approximated over the FFT bins by a 4th-order gammatone magnitude shape
/// whose bandwidth is the ERB at its center frequency. Tiny weights are dropped
/// to keep the bank sparse.
fn gammatone_filterbank(sample_rate: u32) -> Vec<Vec<(usize, f32)>> {
    let f_max = sample_rate as f32 / 2.0;
    let f_low = F_LOW.min(f_max * 0.5);

    // Equally spaced center frequencies in ERB-number units.
    let erb_lo = hz_to_erb_number(f_low);
    let erb_hi = hz_to_erb_number(f_max);
    let mut centers_hz = [0.0f32; N_GAMMA];
    for (i, c) in centers_hz.iter_mut().enumerate() {
        let e = erb_lo + (erb_hi - erb_lo) * (i as f32 + 1.0) / (N_GAMMA as f32 + 1.0);
        *c = erb_number_to_hz(e);
    }

    let mut fb: Vec<Vec<(usize, f32)>> = Vec::with_capacity(N_GAMMA);
    for &ctr in centers_hz.iter() {
        let erb = erb_hz(ctr);
        // 4th-order gammatone equivalent bandwidth parameter `b`. The relation
        // ERB = b * 1.0186... / (2*pi) * (something); we use the standard
        // b = 1.019 * 2*pi * ERB for the gammatone envelope, which makes the
        // magnitude half-power width track the ERB.
        let b = 1.019 * core::f32::consts::PI * erb;
        let mut filt = Vec::new();
        for bin in 0..NUM_BINS {
            let f = bin_to_hz(bin, sample_rate);
            // 4th-order gammatone magnitude response (normalized to 1 at center):
            //   |H(f)| = (1 + ((f - fc)/b)^2)^(-n/2), with n = 4.
            let r = (f - ctr) / b;
            let w = (1.0 + r * r).powi(-2);
            if w > 1e-3 {
                filt.push((bin, w));
            }
        }
        // Guarantee at least one bin so a degenerate filter never divides by
        // nothing downstream: snap to the nearest bin if the bank dropped all.
        if filt.is_empty() {
            let mut best_bin = 0usize;
            let mut best_d = f32::INFINITY;
            for bin in 0..NUM_BINS {
                let d = (bin_to_hz(bin, sample_rate) - ctr).abs();
                if d < best_d {
                    best_d = d;
                    best_bin = bin;
                }
            }
            filt.push((best_bin, 1.0));
        }
        fb.push(filt);
    }
    fb
}

/// Precompute the DCT-II basis matrix (`N_GTCC` x `N_GAMMA`).
fn dct_matrix() -> [[f32; N_GAMMA]; N_GTCC] {
    let mut dct = [[0.0f32; N_GAMMA]; N_GTCC];
    let scale = (2.0f32 / N_GAMMA as f32).sqrt();
    for (k, row) in dct.iter_mut().enumerate() {
        for (m, val) in row.iter_mut().enumerate() {
            *val = scale
                * (core::f32::consts::PI / N_GAMMA as f32 * (m as f32 + 0.5) * k as f32).cos();
        }
    }
    dct
}

/// Equivalent Rectangular Bandwidth (Hz) at frequency `f` (Glasberg & Moore).
#[inline]
fn erb_hz(f: f32) -> f32 {
    24.7 * (4.37 * f / 1000.0 + 1.0)
}

/// Hz → ERB-number (the ERB-rate scale), Glasberg & Moore 1990.
#[inline]
fn hz_to_erb_number(f: f32) -> f32 {
    21.4 * (4.37 * f / 1000.0 + 1.0).log10()
}

/// ERB-number → Hz (inverse of [`hz_to_erb_number`]).
#[inline]
fn erb_number_to_hz(e: f32) -> f32 {
    (10.0f32.powf(e / 21.4) - 1.0) * 1000.0 / 4.37
}
