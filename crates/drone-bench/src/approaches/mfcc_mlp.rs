//! MFCC + small multi-layer perceptron (nonlinear classifier).
//!
//! Front-end (identical in spirit to `mfcc_lr`): each frame's 512-bin Hann
//! magnitude spectrum is passed through a mel filterbank (`N_MELS` triangular
//! filters spanning ~0..sr/2, `mel = 2595*log10(1+f/700)`), log-compressed,
//! then a DCT-II yields `N_MFCC` cepstral coefficients. Per clip the frame-wise
//! MFCCs are pooled into mean and std vectors plus the mean log-energy, giving a
//! fixed-length feature vector. Features are standardized with the train-set
//! mean/std (stored on the struct).
//!
//! Model: a 1-hidden-layer multi-layer perceptron with `H_HIDDEN` tanh units and
//! a sigmoid output, trained by full-batch backprop with L2 weight decay. The
//! nonlinearity lets it carve decision boundaries the linear `mfcc_lr` cannot,
//! pushing real-data ROC-AUC up.
//!
//! Everything is deterministic: weights are initialized from a fixed-seed
//! xorshift generator purely to break symmetry (no run-to-run RNG). `score`
//! returns the sigmoid output clamped to `[0, 1]`, or `0.5` if `fit` was never
//! called.

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
/// Hidden-layer width.
const H_HIDDEN: usize = 24;

/// Training epochs (full-batch).
const EPOCHS: usize = 500;
/// Learning rate.
const LR: f32 = 0.08;
/// L2 weight-decay strength.
const L2: f32 = 1e-4;
/// Fixed seed for deterministic, symmetry-breaking weight init.
const INIT_SEED: u32 = 0x9E37_79B9;

pub struct MfccMlp {
    /// Hidden-layer weights, `[H_HIDDEN][N_FEAT]`.
    w1: Vec<[f32; N_FEAT]>,
    /// Hidden-layer biases, `[H_HIDDEN]`.
    b1: Vec<f32>,
    /// Output weights, `[H_HIDDEN]`.
    w2: Vec<f32>,
    /// Output bias.
    b2: f32,
    /// Per-feature mean from the train set (for standardization).
    feat_mean: [f32; N_FEAT],
    /// Per-feature standard deviation from the train set.
    feat_std: [f32; N_FEAT],
    /// Whether `fit` has been called.
    fitted: bool,
}

impl Default for MfccMlp {
    fn default() -> Self {
        Self {
            w1: vec![[0.0; N_FEAT]; H_HIDDEN],
            b1: vec![0.0; H_HIDDEN],
            w2: vec![0.0; H_HIDDEN],
            b2: 0.0,
            feat_mean: [0.0; N_FEAT],
            feat_std: [1.0; N_FEAT],
            fitted: false,
        }
    }
}

impl MfccMlp {
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

    /// Forward pass: returns (hidden activations, output probability).
    fn forward(&self, x: &[f32; N_FEAT]) -> ([f32; H_HIDDEN], f32) {
        let mut h = [0.0f32; H_HIDDEN];
        for (j, hj) in h.iter_mut().enumerate() {
            let mut z = self.b1[j];
            for (w, xi) in self.w1[j].iter().zip(x.iter()) {
                z += w * xi;
            }
            *hj = z.tanh();
        }
        let mut z2 = self.b2;
        for (w, hj) in self.w2.iter().zip(h.iter()) {
            z2 += w * hj;
        }
        (h, sigmoid(z2))
    }
}

impl Approach for MfccMlp {
    fn name(&self) -> &str {
        "mfcc_mlp"
    }

    fn description(&self) -> &str {
        "MFCC features (mean/std pooled) + 1-hidden-layer MLP (tanh)"
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

        // Deterministic symmetry-breaking init via a fixed-seed xorshift.
        // Scaled by ~1/sqrt(fan_in) for stable tanh activations.
        let mut rng = INIT_SEED;
        let scale1 = (1.0 / N_FEAT as f32).sqrt();
        let scale2 = (1.0 / H_HIDDEN as f32).sqrt();
        let mut w1 = vec![[0.0f32; N_FEAT]; H_HIDDEN];
        let mut b1 = vec![0.0f32; H_HIDDEN];
        let mut w2 = vec![0.0f32; H_HIDDEN];
        for row in w1.iter_mut() {
            for wij in row.iter_mut() {
                *wij = next_uniform(&mut rng) * scale1;
            }
        }
        for wj in w2.iter_mut() {
            *wj = next_uniform(&mut rng) * scale2;
        }
        let mut b2 = 0.0f32;

        // Full-batch gradient descent on the binary cross-entropy loss.
        for _ in 0..EPOCHS {
            let mut g_w1 = vec![[0.0f32; N_FEAT]; H_HIDDEN];
            let mut g_b1 = [0.0f32; H_HIDDEN];
            let mut g_w2 = [0.0f32; H_HIDDEN];
            let mut g_b2 = 0.0f32;

            for (xi, &yi) in x.iter().zip(labels.iter()) {
                // Forward.
                let mut h = [0.0f32; H_HIDDEN];
                for (j, hj) in h.iter_mut().enumerate() {
                    let mut z = b1[j];
                    for (w, v) in w1[j].iter().zip(xi.iter()) {
                        z += w * v;
                    }
                    *hj = z.tanh();
                }
                let mut z2 = b2;
                for (w, hj) in w2.iter().zip(h.iter()) {
                    z2 += w * hj;
                }
                let p = sigmoid(z2);

                // Backward. dL/dz2 = p - y for BCE + sigmoid.
                let dz2 = p - yi;
                g_b2 += dz2;
                for (g, &hj) in g_w2.iter_mut().zip(h.iter()) {
                    *g += dz2 * hj;
                }
                // Backprop into hidden: d/dz_j = dz2 * w2_j * (1 - h_j^2).
                for j in 0..H_HIDDEN {
                    let dh = dz2 * w2[j];
                    let dz1 = dh * (1.0 - h[j] * h[j]);
                    g_b1[j] += dz1;
                    for (g, &v) in g_w1[j].iter_mut().zip(xi.iter()) {
                        *g += dz1 * v;
                    }
                }
            }

            // Parameter update with L2 decay on weights (not biases).
            for j in 0..H_HIDDEN {
                for k in 0..N_FEAT {
                    let grad = g_w1[j][k] / n + L2 * w1[j][k];
                    w1[j][k] -= LR * grad;
                }
                b1[j] -= LR * (g_b1[j] / n);
                let grad2 = g_w2[j] / n + L2 * w2[j];
                w2[j] -= LR * grad2;
            }
            b2 -= LR * (g_b2 / n);
        }

        self.w1 = w1;
        self.b1 = b1;
        self.w2 = w2;
        self.b2 = b2;
        self.fitted = true;
    }

    fn score(&self, samples: &[f32], sample_rate: u32) -> f32 {
        if !self.fitted {
            return 0.5;
        }
        let raw = clip_features(samples, sample_rate);
        let x = self.standardize(&raw);
        let (_, p) = self.forward(&x);
        p.clamp(0.0, 1.0)
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

/// Advance an xorshift32 state and return a deterministic value in `(-1, 1)`.
#[inline]
fn next_uniform(state: &mut u32) -> f32 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    // Map to (-1, 1).
    (x as f32 / u32::MAX as f32) * 2.0 - 1.0
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
