//! Spectral-feature gate.
//!
//! Cheap, interpretable scalar spectral descriptors combined into a
//! drone-likeness confidence in `[0, 1]`. Drone audio is *tonal*: its energy
//! concentrates in a few harmonic peaks inside the band-pass band (~100-4000
//! Hz), which shows up as LOW spectral flatness, LOW spectral entropy, a HIGH
//! in-band energy ratio, and a comparatively LOW centroid/rolloff. Broadband
//! noise is flat and high-entropy; out-of-band hum and bright single tones miss
//! the band-ratio / centroid profile.
//!
//! Rather than hand-tune a threshold, we self-calibrate: in [`fit`] we extract a
//! small feature vector per clip, standardize it (storing the train-split
//! mean/std), and train a tiny logistic-regression classifier by batch gradient
//! descent (no ML crate). `score` returns the sigmoid probability, which is a
//! naturally calibrated confidence in `[0, 1]`. If `fit` was never called (or
//! the train split was degenerate) we fall back to a sensible hand-designed
//! monotonic rule over the same features.

use crate::dataset::Sample;
use crate::util::spectra;
use crate::Approach;
use drone_dsp::{band_energy, spectral_centroid, total_energy, Spectrum, NUM_BINS};

/// Number of scalar spectral features per clip.
const N_FEATURES: usize = 5;

/// Band of interest for drone tonals (matches the BPF front-end intent).
const BAND_LO_HZ: f32 = 100.0;
const BAND_HI_HZ: f32 = 4000.0;

/// Rolloff fraction (share of spectral energy below the rolloff frequency).
const ROLLOFF_FRAC: f32 = 0.85;

/// Small constant to keep logs/divisions finite.
const EPS: f32 = 1e-10;

pub struct SpectralGate {
    /// Per-feature mean from the train split (standardization).
    mean: [f32; N_FEATURES],
    /// Per-feature std from the train split (standardization).
    std: [f32; N_FEATURES],
    /// Logistic-regression weights over the standardized features.
    weights: [f32; N_FEATURES],
    /// Logistic-regression bias.
    bias: f32,
    /// Whether `fit` successfully trained a model.
    trained: bool,
}

impl Default for SpectralGate {
    fn default() -> Self {
        Self {
            mean: [0.0; N_FEATURES],
            std: [1.0; N_FEATURES],
            weights: [0.0; N_FEATURES],
            bias: 0.0,
            trained: false,
        }
    }
}

impl SpectralGate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Mean per-clip feature vector, or `None` if the clip is effectively
    /// silent (so the caller can return a confident 0.0).
    ///
    /// Features (all clamped to be finite):
    ///   0. spectral flatness   - geo_mean / arith_mean of magnitudes (low = tonal)
    ///   1. spectral entropy    - normalized Shannon entropy of the power spectrum (low = peaky)
    ///   2. band-energy ratio   - in-band power / total power (high for drones)
    ///   3. spectral centroid   - energy-weighted mean frequency, in Hz
    ///   4. spectral rolloff    - frequency below which `ROLLOFF_FRAC` of power sits, in Hz
    fn features(samples: &[f32], sample_rate: u32) -> Option<[f32; N_FEATURES]> {
        let frames = spectra(samples);
        if frames.is_empty() {
            return None;
        }

        let mut acc = [0.0_f32; N_FEATURES];
        let mut n_active = 0u32;

        for spec in &frames {
            let total = total_energy(spec);
            // Skip near-silent frames so padding / quiet gaps don't dominate.
            if total <= EPS {
                continue;
            }
            n_active += 1;

            acc[0] += flatness(spec);
            acc[1] += entropy(spec);
            acc[2] +=
                (band_energy(spec, BAND_LO_HZ, BAND_HI_HZ, sample_rate) / total).clamp(0.0, 1.0);
            acc[3] += spectral_centroid(spec, sample_rate);
            acc[4] += rolloff(spec, sample_rate, ROLLOFF_FRAC);
        }

        if n_active == 0 {
            return None;
        }
        let inv = 1.0 / n_active as f32;
        let mut feats = [0.0_f32; N_FEATURES];
        for (f, a) in feats.iter_mut().zip(acc.iter()) {
            *f = a * inv;
            if !f.is_finite() {
                *f = 0.0;
            }
        }
        Some(feats)
    }

    /// Hand-designed fallback when no trained model is available.
    ///
    /// Monotonic in the right directions (high band-ratio + low flatness + low
    /// entropy => high confidence) squashed through a logistic.
    fn rule_score(feats: &[f32; N_FEATURES]) -> f32 {
        let flatness = feats[0];
        let entropy = feats[1];
        let band_ratio = feats[2];
        // Weighted vote, all terms in roughly [0, 1].
        let z = -4.0 + 5.0 * band_ratio + 3.0 * (1.0 - flatness) + 3.0 * (1.0 - entropy);
        sigmoid(z)
    }
}

impl Approach for SpectralGate {
    fn name(&self) -> &str {
        "spectral_gate"
    }

    fn description(&self) -> &str {
        "Spectral-feature gate: flatness/entropy/band-ratio/centroid/rolloff + logistic regression"
    }

    fn fit(&mut self, train: &[Sample]) {
        // Collect features + labels, skipping silent clips.
        let mut xs: Vec<[f32; N_FEATURES]> = Vec::with_capacity(train.len());
        let mut ys: Vec<f32> = Vec::with_capacity(train.len());
        for s in train {
            if let Some(f) = Self::features(&s.samples, s.sample_rate) {
                xs.push(f);
                ys.push(if s.label == 1 { 1.0 } else { 0.0 });
            }
        }

        // Need both classes present to learn a meaningful boundary.
        let n_pos = ys.iter().filter(|&&y| y > 0.5).count();
        if xs.len() < 4 || n_pos == 0 || n_pos == ys.len() {
            self.trained = false;
            return;
        }

        // Standardization stats over the train split.
        let n = xs.len() as f32;
        let mut mean = [0.0_f32; N_FEATURES];
        for x in &xs {
            for (m, v) in mean.iter_mut().zip(x.iter()) {
                *m += *v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n;
        }
        let mut var = [0.0_f32; N_FEATURES];
        for x in &xs {
            for (vr, (v, m)) in var.iter_mut().zip(x.iter().zip(mean.iter())) {
                let d = v - m;
                *vr += d * d;
            }
        }
        let mut std = [1.0_f32; N_FEATURES];
        for (s, vr) in std.iter_mut().zip(var.iter()) {
            let sd = (vr / n).sqrt();
            *s = if sd > 1e-6 { sd } else { 1.0 };
        }
        self.mean = mean;
        self.std = std;

        // Standardize in place.
        for x in xs.iter_mut() {
            for (v, (m, sd)) in x.iter_mut().zip(self.mean.iter().zip(self.std.iter())) {
                *v = (*v - m) / sd;
            }
        }

        // Batch gradient descent on logistic loss with light L2 regularization.
        let mut w = [0.0_f32; N_FEATURES];
        let mut b = 0.0_f32;
        let lr = 0.3_f32;
        let l2 = 1e-3_f32;
        let epochs = 600;
        for _ in 0..epochs {
            let mut grad_w = [0.0_f32; N_FEATURES];
            let mut grad_b = 0.0_f32;
            for (x, &y) in xs.iter().zip(ys.iter()) {
                let mut z = b;
                for (wi, xi) in w.iter().zip(x.iter()) {
                    z += wi * xi;
                }
                let err = sigmoid(z) - y; // dL/dz for logistic loss
                for (g, xi) in grad_w.iter_mut().zip(x.iter()) {
                    *g += err * xi;
                }
                grad_b += err;
            }
            let inv = 1.0 / n;
            for (wi, g) in w.iter_mut().zip(grad_w.iter()) {
                *wi -= lr * (g * inv + l2 * *wi);
            }
            b -= lr * grad_b * inv;
        }

        if w.iter().all(|v| v.is_finite()) && b.is_finite() {
            self.weights = w;
            self.bias = b;
            self.trained = true;
        } else {
            self.trained = false;
        }
    }

    fn score(&self, samples: &[f32], sample_rate: u32) -> f32 {
        let feats = match Self::features(samples, sample_rate) {
            Some(f) => f,
            None => return 0.0, // silence => no drone
        };

        let s = if self.trained {
            let mut z = self.bias;
            for (i, &f) in feats.iter().enumerate() {
                let std = self.std[i].max(1e-6);
                z += self.weights[i] * (f - self.mean[i]) / std;
            }
            sigmoid(z)
        } else {
            Self::rule_score(&feats)
        };

        if s.is_finite() {
            s.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Spectral flatness: geometric mean / arithmetic mean of magnitudes, in
/// `[0, 1]`. ~1 for flat (noise-like) spectra, low for peaky/tonal ones.
/// The geometric mean is computed in the log domain for numerical stability.
fn flatness(spec: &Spectrum) -> f32 {
    let mut log_sum = 0.0_f32;
    let mut arith = 0.0_f32;
    for &m in spec.iter() {
        let v = m + EPS;
        log_sum += v.ln();
        arith += v;
    }
    let n = NUM_BINS as f32;
    let geo = (log_sum / n).exp();
    let arith_mean = arith / n;
    if arith_mean > EPS {
        (geo / arith_mean).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Normalized Shannon spectral entropy of the power spectrum, in `[0, 1]`.
/// ~1 for uniform (flat) spectra, low for energy concentrated in few bins.
fn entropy(spec: &Spectrum) -> f32 {
    let mut power = [0.0_f32; NUM_BINS];
    let mut sum = 0.0_f32;
    for (p, &m) in power.iter_mut().zip(spec.iter()) {
        *p = m * m;
        sum += *p;
    }
    if sum <= EPS {
        return 1.0;
    }
    let mut h = 0.0_f32;
    for &p in power.iter() {
        let prob = p / sum;
        if prob > EPS {
            h -= prob * prob.ln();
        }
    }
    let max_h = (NUM_BINS as f32).ln();
    if max_h > 0.0 {
        (h / max_h).clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// Spectral rolloff: the frequency (Hz) below which `frac` of the total
/// spectral power lies. Higher for broadband/bright content.
fn rolloff(spec: &Spectrum, sample_rate: u32, frac: f32) -> f32 {
    let total = total_energy(spec);
    if total <= EPS {
        return 0.0;
    }
    let target = frac * total;
    let mut acc = 0.0_f32;
    for (i, &m) in spec.iter().enumerate() {
        acc += m * m;
        if acc >= target {
            return drone_dsp::bin_to_hz(i, sample_rate);
        }
    }
    drone_dsp::bin_to_hz(NUM_BINS - 1, sample_rate)
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
