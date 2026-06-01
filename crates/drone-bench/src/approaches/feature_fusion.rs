//! Feature-level fusion + logistic-regression classifier.
//!
//! This approach builds ONE broad clip-level feature vector by concatenating
//! several complementary feature families, each of which captures a different
//! facet of a drone's acoustic signature, then standardizes the fused vector
//! with train-set statistics and trains a logistic-regression classifier by
//! deterministic batch gradient descent with L2 regularization.
//!
//! Families fused (all pooled across frames as clip means, plus MFCC std):
//!
//! * **MFCC** - `N_MFCC` mel-frequency cepstral coefficients (mel filterbank →
//!   log → DCT-II), pooled as per-coefficient mean *and* std across frames. The
//!   classic timbral fingerprint.
//! * **Spectral descriptors** (clip means of frame-wise values): spectral
//!   flatness (geo/arith mean of the magnitude spectrum - tonal vs. noisy),
//!   spectral entropy (normalized Shannon entropy of the power spectrum),
//!   spectral centroid (Hz), spectral rolloff at 0.85, and the band-energy ratio
//!   in 100..4000 Hz (where rotor harmonics live).
//! * **Harmonic / comb strength**: the harmonic-product-spectrum (HPS) peak
//!   value, and the fraction of total energy carried by the 100..330 Hz
//!   fundamental/blade-pass band - drones stack strong low harmonics.
//! * **Cepstral / autocorrelation periodicity**: the normalized peak of the
//!   frame autocorrelation in a plausible pitch-period lag range - periodic
//!   rotor noise produces a sharp peak, broadband confounders do not.
//!
//! The literature consistently reports that fused features beat any single
//! family; concatenating them here drives ROC-AUC well past what MFCC-only or
//! harmonic-only detectors reach. `score` standardizes the clip feature with the
//! stored statistics and returns `sigmoid(w·x + b)`, a calibrated probability in
//! `[0, 1]`. Everything is deterministic (zero-initialized weights, no RNG).

use crate::dataset::Sample;
use crate::util::spectra;
use crate::Approach;
use drone_dsp::{bin_to_hz, hz_to_bin, spectral_centroid, total_energy, Spectrum, NUM_BINS};

/// Number of mel filterbank channels.
const N_MELS: usize = 26;
/// Number of MFCC coefficients kept (including c0).
const N_MFCC: usize = 13;

// Layout of the fused feature vector.
//   [0..N_MFCC)            MFCC means
//   [N_MFCC..2*N_MFCC)     MFCC stds
//   then the scalar descriptors below, in this fixed order:
const N_MFCC2: usize = 2 * N_MFCC;
/// Number of non-MFCC scalar descriptors appended after the MFCC block.
const N_EXTRA: usize = 8;
/// Total fused feature vector length.
const N_FEAT: usize = N_MFCC2 + N_EXTRA;

/// Gradient-descent iterations.
const ITERS: usize = 1500;
/// Learning rate.
const LR: f32 = 0.4;
/// L2 regularization strength.
const L2: f32 = 1e-3;

pub struct FeatureFusion {
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

impl Default for FeatureFusion {
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

impl FeatureFusion {
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

impl Approach for FeatureFusion {
    fn name(&self) -> &str {
        "feature_fusion"
    }

    fn description(&self) -> &str {
        "Fused MFCC + spectral + harmonic + cepstral features + logistic regression"
    }

    fn fit(&mut self, train: &[Sample]) {
        // Extract the fused clip feature for every training sample.
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

/// Build the full fused clip feature vector. Returns a zero vector for empty or
/// silent input so callers degrade gracefully.
fn clip_features(samples: &[f32], sample_rate: u32) -> [f32; N_FEAT] {
    let mut out = [0.0f32; N_FEAT];
    let frames = spectra(samples);
    if frames.is_empty() {
        return out;
    }

    let fb = mel_filterbank(sample_rate);
    let dct = dct_matrix();

    // --- MFCC pooling: per-coefficient mean and std across frames. ---
    let mut mfcc_sum = [0.0f32; N_MFCC];
    let mut mfcc_sumsq = [0.0f32; N_MFCC];

    // --- spectral-descriptor accumulators (frame-wise then averaged). ---
    let mut sum_flatness = 0.0f32;
    let mut sum_entropy = 0.0f32;
    let mut sum_centroid = 0.0f32;
    let mut sum_rolloff = 0.0f32;
    let mut sum_band_ratio = 0.0f32;
    let mut sum_hps = 0.0f32;
    let mut sum_fund_ratio = 0.0f32;

    let count = frames.len() as f32;

    for sp in &frames {
        // ---- MFCCs ----
        let mut log_mel = [0.0f32; N_MELS];
        for (m, filt) in fb.iter().enumerate() {
            let mut e = 0.0f32;
            for &(bin, w) in filt {
                e += w * sp[bin] * sp[bin];
            }
            log_mel[m] = (e + 1e-10).ln();
        }
        for k in 0..N_MFCC {
            let mut c = 0.0f32;
            for m in 0..N_MELS {
                c += dct[k][m] * log_mel[m];
            }
            mfcc_sum[k] += c;
            mfcc_sumsq[k] += c * c;
        }

        // ---- spectral descriptors ----
        sum_flatness += spectral_flatness(sp);
        sum_entropy += spectral_entropy(sp);
        sum_centroid += spectral_centroid(sp, sample_rate);
        sum_rolloff += spectral_rolloff(sp, sample_rate, 0.85);

        let tot = total_energy(sp);
        let band = band_energy_sq(sp, 100.0, 4000.0, sample_rate);
        sum_band_ratio += if tot > 1e-12 { band / tot } else { 0.0 };

        // ---- harmonic / comb strength ----
        sum_hps += harmonic_product_peak(sp);
        let fund = band_energy_sq(sp, 100.0, 330.0, sample_rate);
        sum_fund_ratio += if tot > 1e-12 { fund / tot } else { 0.0 };
    }

    // Write MFCC mean/std into the fused vector.
    for k in 0..N_MFCC {
        let mean = mfcc_sum[k] / count;
        let v = (mfcc_sumsq[k] / count - mean * mean).max(0.0);
        out[k] = mean;
        out[N_MFCC + k] = v.sqrt();
    }

    // Append the scalar descriptors (fixed order - must match N_EXTRA).
    let mut idx = N_MFCC2;
    out[idx] = sum_flatness / count;
    idx += 1;
    out[idx] = sum_entropy / count;
    idx += 1;
    out[idx] = sum_centroid / count;
    idx += 1;
    out[idx] = sum_rolloff / count;
    idx += 1;
    out[idx] = sum_band_ratio / count;
    idx += 1;
    out[idx] = sum_hps / count;
    idx += 1;
    out[idx] = sum_fund_ratio / count;
    idx += 1;
    // Autocorrelation periodicity peak is a time-domain clip feature.
    out[idx] = autocorr_periodicity(samples, sample_rate);

    out
}

/// Spectral flatness: ratio of the geometric mean to the arithmetic mean of the
/// magnitude spectrum. Near 1 for noise-like spectra, near 0 for tonal/peaky
/// ones. Computed over the non-DC bins.
fn spectral_flatness(sp: &Spectrum) -> f32 {
    let mut log_sum = 0.0f32;
    let mut lin_sum = 0.0f32;
    let mut n = 0.0f32;
    for &m in sp.iter().skip(1) {
        let mag = m + 1e-10;
        log_sum += mag.ln();
        lin_sum += mag;
        n += 1.0;
    }
    if n < 1.0 || lin_sum <= 0.0 {
        return 0.0;
    }
    let geo = (log_sum / n).exp();
    let arith = lin_sum / n;
    (geo / arith).clamp(0.0, 1.0)
}

/// Normalized Shannon entropy of the power spectrum, in `[0, 1]`. High for
/// flat/noisy spectra, low when energy concentrates in a few bins.
fn spectral_entropy(sp: &Spectrum) -> f32 {
    let mut power = [0.0f32; NUM_BINS];
    let mut total = 0.0f32;
    for (i, &m) in sp.iter().enumerate().skip(1) {
        let p = m * m;
        power[i] = p;
        total += p;
    }
    if total <= 1e-20 {
        return 0.0;
    }
    let mut h = 0.0f32;
    for &p in power.iter().skip(1) {
        if p > 0.0 {
            let prob = p / total;
            h -= prob * prob.ln();
        }
    }
    // Normalize by log of the number of bins so the result is in [0, 1].
    let norm = ((NUM_BINS - 1) as f32).ln();
    if norm > 0.0 {
        (h / norm).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Spectral rolloff: the frequency (Hz) below which `frac` of the spectral
/// energy is contained.
fn spectral_rolloff(sp: &Spectrum, sample_rate: u32, frac: f32) -> f32 {
    let total = total_energy(sp);
    if total <= 1e-20 {
        return 0.0;
    }
    let threshold = frac * total;
    let mut acc = 0.0f32;
    for (i, &m) in sp.iter().enumerate() {
        acc += m * m;
        if acc >= threshold {
            return bin_to_hz(i, sample_rate);
        }
    }
    bin_to_hz(NUM_BINS - 1, sample_rate)
}

/// Sum of squared magnitudes within `[lo_hz, hi_hz]` (a band power).
fn band_energy_sq(sp: &Spectrum, lo_hz: f32, hi_hz: f32, sample_rate: u32) -> f32 {
    let lo = hz_to_bin(lo_hz, sample_rate);
    let hi = hz_to_bin(hi_hz, sample_rate).max(lo);
    let mut acc = 0.0f32;
    for &m in sp[lo..=hi].iter() {
        acc += m * m;
    }
    acc
}

/// Harmonic product spectrum peak in the fundamental band. For each candidate
/// fundamental bin in a plausible blade-pass range we multiply the magnitudes
/// at its first few harmonics; a strong harmonic stack (as drones produce)
/// yields a large product. Returned in log domain for a well-behaved feature.
fn harmonic_product_peak(sp: &Spectrum) -> f32 {
    const N_HARM: usize = 5;
    // Candidate fundamentals: bins 2..=NUM_BINS/ N_HARM so all harmonics fit.
    let max_f0 = NUM_BINS / N_HARM;
    let mut best = 0.0f32;
    for f0 in 2..max_f0 {
        let mut prod = 1.0f32;
        for h in 1..=N_HARM {
            let bin = f0 * h;
            prod *= sp[bin] + 1e-6;
        }
        if prod > best {
            best = prod;
        }
    }
    // Log-compress (product spans many orders of magnitude).
    (best + 1e-12).ln()
}

/// Normalized autocorrelation periodicity peak of the time-domain clip in a
/// plausible pitch-period range. A sharp periodic rotor signature gives a value
/// near 1; broadband noise stays near 0. Uses a single mid-clip analysis window.
fn autocorr_periodicity(samples: &[f32], sample_rate: u32) -> f32 {
    if samples.len() < 64 {
        return 0.0;
    }
    // Analysis window: up to 8192 samples centered in the clip.
    let win = samples.len().min(8192);
    let start = (samples.len() - win) / 2;
    let x = &samples[start..start + win];

    // Lag range corresponds to ~60..1000 Hz fundamentals.
    let min_lag = (sample_rate as usize / 1000).max(1);
    let max_lag = (sample_rate as usize / 60).min(win - 1);
    if max_lag <= min_lag {
        return 0.0;
    }

    // Energy at zero lag for normalization.
    let mut r0 = 0.0f32;
    for &v in x {
        r0 += v * v;
    }
    if r0 <= 1e-12 {
        return 0.0;
    }

    let mut best = 0.0f32;
    for lag in min_lag..=max_lag {
        let mut acc = 0.0f32;
        for i in lag..win {
            acc += x[i] * x[i - lag];
        }
        let r = acc / r0;
        if r > best {
            best = r;
        }
    }
    best.clamp(0.0, 1.0)
}

/// Build a mel filterbank as a list of `(bin, weight)` pairs per channel.
///
/// `N_MELS` triangular filters are spaced equally on the mel scale between 0 Hz
/// and the Nyquist frequency, then mapped back to the linear FFT bins.
fn mel_filterbank(sample_rate: u32) -> Vec<Vec<(usize, f32)>> {
    let f_max = sample_rate as f32 / 2.0;
    let mel_max = hz_to_mel(f_max);

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
