//! Physics-fused drone detector - engineered for CROSS-DATASET generalization.
//!
//! Our leakage-honest `xeval` (train DADS → test Al-Emadi drones + ESC-50 hard
//! negatives) exposed a sharp split between two kinds of detector:
//!
//! * **Physical / structural cues generalize.** Methods that key on the *physics*
//!   of a multirotor - a regularly-spaced harmonic comb at the blade-pass
//!   fundamental, periodic amplitude modulation from the rotors, tonality of the
//!   spectrum - keep most of their in-distribution ROC-AUC when moved to a
//!   completely different corpus (different mics, rooms, drones, noise).
//!   `envelope_periodicity` (0.872), `hps` (0.852), and the fusions (0.848 /
//!   0.813) all held up.
//!
//! * **Learned spectral / MFCC templates overfit the recording.** `mfcc_lr`
//!   (0.685) and the averaged-spectrum `template` (~0.49) collapsed cross-dataset:
//!   they memorize the *timbre of the training recordings* (mic colour, room,
//!   specific motor whine), which does not transfer. Training augmentation made
//!   it WORSE - it erodes the recording lineage the template was leaning on.
//!
//! `physics_fused` acts on that finding directly: it fuses **only** the
//! physics/structure features and *deliberately excludes* the two things that
//! overfit - raw MFCC-mean templates and cosine-to-an-averaged-spectrum. The hope
//! (borne out empirically) is that physics features transfer where raw spectral
//! templates do not, so a small classifier over them generalizes better than any
//! single physical cue alone.
//!
//! ## Feature vector (all timbre-invariant, level-invariant where possible)
//!  0. **Harmonic-comb contrast** - HPS-guided, blade-pass band ~80-330 Hz. Each
//!     predicted harmonic is scored against its *local* inter-harmonic background
//!     (a contrast, so absolute level / broadband motor hiss cannot inflate or
//!     invert it), robustly aggregated across frames.
//!  1. **Consistent harmonic count** - how many comb teeth stand clearly above
//!     their local floor (a regular *stack*, not one lone tone).
//!  2. **Cepstral periodicity peak** - peak-to-baseline of the real cepstrum in
//!     the blade-pass quefrency band (harmonic comb is periodic in frequency).
//!  3. **Autocorrelation periodicity peak** - time-domain period strength.
//!  4. **Envelope AM strength** - fraction of envelope variance in the single
//!     strongest modulation line, 5-200 Hz (rotor / blade-pass AM).
//!  5. **Harmonic-to-noise ratio** - comb-band energy vs. off-comb residual.
//!  6. **Spectral flatness** (clip mean) - tonal vs. noise-like.
//!  7. **Spectral entropy** (clip mean) - tonality (concentrated vs. flat).
//!  8. **Fundamental-band energy ratio** - fraction of energy in 80-330 Hz.
//!
//! These are standardized with train-set statistics and fed to a deterministic
//! L2-regularized logistic regression (zero-init, batch gradient descent, no
//! RNG). A logistic over physics features transfers because the *decision* is
//! "how drone-like is the structure", not "does this match a training recording".
//!
//! `score` returns a calibrated probability in `[0, 1]`. Silence / sub-frame
//! input scores `0.0`. Everything is finite and deterministic.

use std::f32::consts::PI;

use drone_dsp::{bin_to_hz, hz_to_bin, FRAME_SIZE, NUM_BINS};

use crate::approach::Approach;
use crate::dataset::Sample;
use crate::util::spectra;

/// Number of fused physics features.
const N_FEAT: usize = 9;

/// Blade-pass fundamental search band, Hz. Covers real DADS/Al-Emadi multirotors
/// (~200-260 Hz) and lower synthetic fundamentals (~110-120 Hz) while excluding
/// the sub-80 Hz urban/wind rumble that dominates the negatives.
const BPF_LO_HZ: f32 = 80.0;
const BPF_HI_HZ: f32 = 330.0;

/// HPS factors used to pick the candidate fundamental.
const HPS_R: usize = 5;
/// Harmonics scored in the comb-contrast feature.
const COMB_HARMONICS: usize = 10;
/// Tolerance half-width (bins) around each predicted harmonic.
const HARM_HALF_WIDTH: usize = 1;

/// Envelope (decimated) sample rate target, Hz.
const ENV_RATE_HZ: f32 = 1000.0;
/// Rotor amplitude-modulation band, Hz.
const MOD_LO_HZ: f32 = 5.0;
const MOD_HI_HZ: f32 = 200.0;
/// Envelope-smoothing window, ms (AM demodulation low-pass, cascaded twice).
const SMOOTH_MS: f32 = 8.0;

/// Gradient-descent iterations.
const ITERS: usize = 2000;
/// Learning rate.
const LR: f32 = 0.5;
/// L2 regularization strength. Slightly stronger than `feature_fusion` so the
/// classifier leans on the broad physics consensus rather than fitting any one
/// feature's training-set quirk - which is what we want for generalization.
const L2: f32 = 3e-3;

/// Physics-only fused detector.
pub struct PhysicsFused {
    weights: [f32; N_FEAT],
    bias: f32,
    feat_mean: [f32; N_FEAT],
    feat_std: [f32; N_FEAT],
    fitted: bool,
}

impl Default for PhysicsFused {
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

impl PhysicsFused {
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

impl Approach for PhysicsFused {
    fn name(&self) -> &str {
        "physics_fused"
    }

    fn description(&self) -> &str {
        "Physics-only fusion (harmonic comb, cepstral/ACF periodicity, rotor AM, \
         HNR, tonality) + logistic regression - engineered for cross-dataset transfer"
    }

    fn fit(&mut self, train: &[Sample]) {
        let feats: Vec<[f32; N_FEAT]> = train
            .iter()
            .map(|s| physics_features(&s.samples, s.sample_rate))
            .collect();
        let labels: Vec<f32> = train.iter().map(|s| s.label as f32).collect();
        if feats.is_empty() {
            return;
        }

        // --- Standardization statistics over the train set. ---
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

        // Standardize once.
        let x: Vec<[f32; N_FEAT]> = feats.iter().map(|f| self.standardize(f)).collect();

        // --- Deterministic L2-regularized logistic regression. ---
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
        // Silence / sub-frame guard.
        let energy: f32 = samples.iter().map(|&x| x * x).sum();
        if samples.len() < FRAME_SIZE || !energy.is_finite() || energy <= 1e-6 {
            return 0.0;
        }
        let raw = physics_features(samples, sample_rate);
        let x = self.standardize(&raw);
        let mut z = self.bias;
        for (w, xi) in self.weights.iter().zip(x.iter()) {
            z += w * xi;
        }
        let s = sigmoid(z);
        if s.is_finite() {
            s.clamp(0.0, 1.0)
        } else {
            0.0
        }
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

// ----------------------------------------------------------------------------
// Feature extraction - physics / structure only.
// ----------------------------------------------------------------------------

/// Build the full physics feature vector for a clip. Returns a zero vector for
/// empty / silent input so callers degrade gracefully.
fn physics_features(samples: &[f32], sample_rate: u32) -> [f32; N_FEAT] {
    let mut out = [0.0f32; N_FEAT];
    let frames = spectra(samples);
    if frames.is_empty() {
        return out;
    }

    let lo_bin = hz_to_bin(BPF_LO_HZ, sample_rate).max(1);
    let hi_bin = hz_to_bin(BPF_HI_HZ, sample_rate).min(NUM_BINS - 1);

    // Per-frame harmonic structure (comb contrast, tooth count, HNR), plus
    // tonality descriptors.
    let mut comb_vals: Vec<f32> = Vec::with_capacity(frames.len());
    let mut teeth_vals: Vec<f32> = Vec::with_capacity(frames.len());
    let mut hnr_vals: Vec<f32> = Vec::with_capacity(frames.len());
    let mut sum_flatness = 0.0f32;
    let mut sum_entropy = 0.0f32;
    let mut sum_fund_ratio = 0.0f32;
    let mut count = 0.0f32;

    for sp in &frames {
        // Skip silent frames so they neither help nor hurt the descriptors.
        let e: f32 = sp.iter().map(|&m| m * m).sum();
        if e <= 1e-12 {
            continue;
        }
        count += 1.0;

        if lo_bin < hi_bin {
            let (contrast, teeth, hnr) = frame_harmonic_stats(sp, lo_bin, hi_bin, sample_rate);
            comb_vals.push(contrast);
            teeth_vals.push(teeth);
            hnr_vals.push(hnr);
        }

        sum_flatness += spectral_flatness(sp);
        sum_entropy += spectral_entropy(sp);

        let total: f32 = sp.iter().map(|&m| m * m).sum();
        let fund = band_energy_sq(sp, BPF_LO_HZ, BPF_HI_HZ, sample_rate);
        sum_fund_ratio += if total > 1e-12 { fund / total } else { 0.0 };
    }

    if count < 1.0 {
        return out;
    }

    // Robust aggregates for the per-frame harmonic stats: drone presence is
    // intermittent, so reward the strongest frames with a high quantile rather
    // than the mean (which a quiet frame would drag down).
    out[0] = quantile(&mut comb_vals, 0.8);
    out[1] = quantile(&mut teeth_vals, 0.8);
    out[2] = cepstral_periodicity(&frames, sample_rate);
    out[3] = autocorr_periodicity(samples, sample_rate);
    out[4] = envelope_am_strength(samples, sample_rate);
    out[5] = quantile(&mut hnr_vals, 0.8);
    out[6] = sum_flatness / count;
    out[7] = sum_entropy / count;
    out[8] = sum_fund_ratio / count;

    for v in out.iter_mut() {
        if !v.is_finite() {
            *v = 0.0;
        }
    }
    out
}

/// Per-frame harmonic structure: `(comb_contrast, tooth_count, harmonic_to_noise)`.
///
/// HPS picks the candidate fundamental (with an octave-error guard), then each
/// predicted harmonic is scored against its *local* inter-harmonic background.
/// A regular stack has every tooth standing above its local floor regardless of
/// absolute level or broadband motor hiss - exactly the recording-invariant cue
/// that transfers cross-dataset.
fn frame_harmonic_stats(
    spec: &[f32; NUM_BINS],
    lo_bin: usize,
    hi_bin: usize,
    sample_rate: u32,
) -> (f32, f32, f32) {
    let total: f32 = spec.iter().sum();
    if total <= f32::EPSILON {
        return (0.0, 0.0, 0.0);
    }

    // --- Harmonic Product Spectrum over the candidate-f0 range. ---
    let mut best_bin = lo_bin;
    let mut best_hps = -1.0_f32;
    for b in lo_bin..=hi_bin {
        let mut prod = 1.0_f32;
        for r in 1..=HPS_R {
            let idx = b * r;
            if idx >= NUM_BINS {
                prod *= f32::EPSILON;
                continue;
            }
            prod *= spec[idx] + f32::EPSILON;
        }
        if prod > best_hps {
            best_hps = prod;
            best_bin = b;
        }
    }

    // Octave-error guard: try f0 and f0/2, keep the stronger comb.
    let cand_a = best_bin;
    let cand_b = best_bin / 2;
    let a = comb_stats(spec, cand_a, sample_rate);
    let b = if cand_b >= 1 && bin_to_hz(cand_b, sample_rate) >= 0.5 * BPF_LO_HZ {
        comb_stats(spec, cand_b, sample_rate)
    } else {
        (0.0, 0.0, 0.0)
    };
    if a.0 >= b.0 {
        a
    } else {
        b
    }
}

/// Comb statistics for fundamental bin `f0_bin`:
/// `(contrast, tooth_count, harmonic_to_noise_ratio)`.
///
/// `contrast` rewards both the number of well-formed teeth and their average
/// prominence above the local background. `harmonic_to_noise` is the ratio of
/// on-comb energy to the off-comb (inter-harmonic) residual - high for a clean
/// rotor stack, low for broadband noise.
fn comb_stats(spec: &[f32; NUM_BINS], f0_bin: usize, sample_rate: u32) -> (f32, f32, f32) {
    if f0_bin == 0 {
        return (0.0, 0.0, 0.0);
    }
    let max_hz = 5000.0_f32.min(bin_to_hz(NUM_BINS - 1, sample_rate));
    let max_bin = hz_to_bin(max_hz, sample_rate);

    let spec_max = spec.iter().copied().fold(0.0_f32, f32::max);
    let energy_floor = 0.02 * spec_max;

    let mut contrast_sum = 0.0_f32;
    let mut teeth = 0u32;
    let mut on_comb = 0.0_f32;
    let mut off_comb = 0.0_f32;

    for h in 1..=COMB_HARMONICS {
        let center = f0_bin * h;
        if center > max_bin || center >= NUM_BINS - 1 {
            break;
        }
        let lo = center.saturating_sub(HARM_HALF_WIDTH);
        let hi = (center + HARM_HALF_WIDTH).min(NUM_BINS - 1);
        let mut peak = 0.0_f32;
        for &m in &spec[lo..=hi] {
            if m > peak {
                peak = m;
            }
        }
        let half = (f0_bin / 2).max(1);
        let bg = background_at(spec, center, half);
        on_comb += peak * peak;
        off_comb += bg * bg;

        let c = peak / (bg + f32::EPSILON);
        if c > 2.0 && peak > energy_floor {
            contrast_sum += (c - 2.0).min(8.0);
            teeth += 1;
        }
    }

    if teeth < 2 {
        // A lone peak is not a harmonic stack.
        return (0.0, 0.0, 0.0);
    }
    let avg = contrast_sum / teeth as f32;
    let contrast = (teeth as f32).min(8.0) * avg;
    // Harmonic-to-noise ratio in dB-ish log domain, clamped to a sane range.
    let hnr = (on_comb / (off_comb + f32::EPSILON) + 1.0)
        .ln()
        .clamp(0.0, 8.0);
    (contrast, teeth as f32, hnr)
}

/// Local off-comb background magnitude around bin `center`: the quieter of the
/// two trough regions `offset` bins below/above the harmonic.
fn background_at(spec: &[f32; NUM_BINS], center: usize, offset: usize) -> f32 {
    let mean_around = |c: usize| -> f32 {
        let lo = c.saturating_sub(1);
        let hi = (c + 1).min(NUM_BINS - 1);
        let n = (hi - lo + 1) as f32;
        spec[lo..=hi].iter().sum::<f32>() / n
    };
    let below = center.saturating_sub(offset).max(1);
    let above = (center + offset).min(NUM_BINS - 1);
    mean_around(below).min(mean_around(above))
}

/// Real-cepstrum periodicity peak in the blade-pass quefrency band, robustly
/// aggregated across frames. The harmonic comb is periodic *in frequency* with
/// spacing `f0`, so the real cepstrum peaks at the matching quefrency.
fn cepstral_periodicity(frames: &[[f32; NUM_BINS]], sample_rate: u32) -> f32 {
    if frames.is_empty() {
        return 0.0;
    }
    let df = sample_rate as f32 / FRAME_SIZE as f32;
    let q_lo = (df * NUM_BINS as f32 / BPF_HI_HZ).floor() as usize;
    let q_hi = (df * NUM_BINS as f32 / BPF_LO_HZ).ceil() as usize;
    let q_lo = q_lo.max(4);
    let q_hi = q_hi.min(NUM_BINS - 1);
    if q_lo >= q_hi {
        return 0.0;
    }

    let n = NUM_BINS;
    let mut logmag = vec![0.0_f32; n];
    let margin = ((q_hi - q_lo) / 4).max(8);
    let eval_lo = q_lo.saturating_sub(margin).max(1);
    let eval_hi = (q_hi + margin).min(n - 1);
    let mut coeffs = vec![0.0_f32; eval_hi - eval_lo + 1];

    let mut peaks: Vec<f32> = Vec::with_capacity(frames.len());
    for spec in frames {
        let e: f32 = spec.iter().map(|&m| m * m).sum();
        if e <= 1e-9 {
            continue;
        }
        for i in 0..n {
            logmag[i] = (1.0 + spec[i]).ln();
        }
        let mean = logmag.iter().sum::<f32>() / n as f32;
        for v in logmag.iter_mut() {
            *v -= mean;
        }
        for (slot, q) in (eval_lo..=eval_hi).enumerate() {
            coeffs[slot] = dct_coeff(&logmag, q, n).abs();
        }
        let mut peak = 0.0_f32;
        for (slot, q) in (eval_lo..=eval_hi).enumerate() {
            if (q_lo..=q_hi).contains(&q) && coeffs[slot] > peak {
                peak = coeffs[slot];
            }
        }
        let baseline = coeffs.iter().sum::<f32>() / coeffs.len() as f32 + 1e-9;
        let ratio = peak / baseline;
        peaks.push(((ratio - 1.0) / 5.0).clamp(0.0, 1.0));
    }
    robust_upper_mean(&mut peaks)
}

/// One DCT-II coefficient `c[q] = sum_n x[n] cos(pi (n+0.5) q / N)`.
#[inline]
fn dct_coeff(x: &[f32], q: usize, n: usize) -> f32 {
    let factor = PI * q as f32 / n as f32;
    let mut acc = 0.0_f32;
    for (i, &xi) in x.iter().enumerate() {
        acc += xi * (factor * (i as f32 + 0.5)).cos();
    }
    acc
}

/// Per-frame normalized autocorrelation peak in the drone-fundamental lag band,
/// aggregated as the mean of the upper half of frames. `r(lag)/r(0)` is ~1 for a
/// perfectly periodic signal and ~0 for white noise.
fn autocorr_periodicity(samples: &[f32], sample_rate: u32) -> f32 {
    if samples.len() < FRAME_SIZE {
        return 0.0;
    }
    let hop = FRAME_SIZE / 2;
    let min_lag = (sample_rate as f32 / BPF_HI_HZ).floor() as usize;
    let max_lag = (sample_rate as f32 / BPF_LO_HZ).ceil() as usize;
    let max_lag = max_lag.min(FRAME_SIZE - 1);
    if min_lag < 1 || min_lag >= max_lag {
        return 0.0;
    }

    let mut peaks: Vec<f32> = Vec::new();
    let mut start = 0usize;
    while start + FRAME_SIZE <= samples.len() {
        let frame = &samples[start..start + FRAME_SIZE];
        start += hop;
        let r0: f32 = frame.iter().map(|&x| x * x).sum();
        if r0 <= 1e-6 {
            continue;
        }
        let mut best = 0.0_f32;
        for lag in min_lag..=max_lag {
            let mut acc = 0.0_f32;
            for i in 0..(FRAME_SIZE - lag) {
                acc += frame[i] * frame[i + lag];
            }
            let norm = acc / r0;
            if norm > best {
                best = norm;
            }
        }
        peaks.push(best.clamp(0.0, 1.0));
    }
    robust_upper_mean(&mut peaks)
}

/// Rotor amplitude-modulation strength: fraction of the envelope's total
/// variance concentrated in the single strongest modulation line (5-200 Hz).
/// Periodic blade-pass AM dumps most of its envelope variance into one line;
/// noise and steady tones spread it flat. Level- and timbre-invariant.
fn envelope_am_strength(samples: &[f32], sample_rate: u32) -> f32 {
    if sample_rate == 0 || samples.len() < FRAME_SIZE {
        return 0.0;
    }
    let energy: f32 = samples.iter().map(|&x| x * x).sum();
    if !energy.is_finite() || energy <= 1e-6 {
        return 0.0;
    }

    let smooth_n = ((SMOOTH_MS * 1e-3) * sample_rate as f32).round() as usize;
    let smooth_n = smooth_n.max(1);
    let smoothed = moving_average(&moving_average_abs(samples, smooth_n), smooth_n);

    let decim = (sample_rate as f32 / ENV_RATE_HZ).round() as usize;
    let decim = decim.max(1);
    let fe = sample_rate as f32 / decim as f32;
    let env: Vec<f32> = smoothed.iter().step_by(decim).copied().collect();
    let m = env.len();
    if m < 32 {
        return 0.0;
    }

    let mean = env.iter().sum::<f32>() / m as f32;
    if !mean.is_finite() || mean <= 1e-9 {
        return 0.0;
    }
    let mut win = vec![0.0_f32; m];
    let mut total_power = 0.0_f32;
    for (i, &e) in env.iter().enumerate() {
        let w = hann(i, m);
        let v = (e / mean - 1.0) * w;
        win[i] = v;
        total_power += v * v;
    }
    if !total_power.is_finite() || total_power <= 1e-12 {
        return 0.0;
    }

    let df = fe / m as f32;
    let nyq = fe * 0.5;
    let hi = MOD_HI_HZ.min(nyq);
    if hi <= MOD_LO_HZ {
        return 0.0;
    }
    let k_lo = (MOD_LO_HZ / df).floor().max(1.0) as usize;
    let k_hi = ((hi / df).ceil() as usize).min(m / 2);
    if k_lo >= k_hi {
        return 0.0;
    }

    let two_pi_over_m = 2.0 * PI / m as f32;
    let mut peak = 0.0_f32;
    for k in k_lo..=k_hi {
        let mut re = 0.0_f32;
        let mut im = 0.0_f32;
        let wk = two_pi_over_m * k as f32;
        for (nn, &x) in win.iter().enumerate() {
            let ang = wk * nn as f32;
            re += x * ang.cos();
            im -= x * ang.sin();
        }
        let p = re * re + im * im;
        if p > peak {
            peak = p;
        }
    }
    let strength = (peak / (m as f32 * total_power)).clamp(0.0, 1.0);
    if strength.is_finite() {
        strength
    } else {
        0.0
    }
}

/// Spectral flatness: geometric/arithmetic mean of the magnitude spectrum.
/// Near 1 for noise-like spectra, near 0 for tonal ones. Over non-DC bins.
fn spectral_flatness(sp: &[f32; NUM_BINS]) -> f32 {
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

/// Normalized Shannon entropy of the power spectrum, in `[0, 1]`. High for flat
/// spectra, low when energy concentrates in a few bins.
fn spectral_entropy(sp: &[f32; NUM_BINS]) -> f32 {
    let mut total = 0.0f32;
    for &m in sp.iter().skip(1) {
        total += m * m;
    }
    if total <= 1e-20 {
        return 0.0;
    }
    let mut h = 0.0f32;
    for &m in sp.iter().skip(1) {
        let p = m * m;
        if p > 0.0 {
            let prob = p / total;
            h -= prob * prob.ln();
        }
    }
    let norm = ((NUM_BINS - 1) as f32).ln();
    if norm > 0.0 {
        (h / norm).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Sum of squared magnitudes within `[lo_hz, hi_hz]`.
fn band_energy_sq(sp: &[f32; NUM_BINS], lo_hz: f32, hi_hz: f32, sample_rate: u32) -> f32 {
    let lo = hz_to_bin(lo_hz, sample_rate);
    let hi = hz_to_bin(hi_hz, sample_rate).max(lo).min(NUM_BINS - 1);
    let mut acc = 0.0f32;
    for &m in sp[lo..=hi].iter() {
        acc += m * m;
    }
    acc
}

/// Mean of the upper half of a per-frame score list (robust to quiet frames).
fn robust_upper_mean(peaks: &mut [f32]) -> f32 {
    if peaks.is_empty() {
        return 0.0;
    }
    peaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let start = peaks.len() / 2;
    let upper = &peaks[start..];
    upper.iter().sum::<f32>() / upper.len() as f32
}

/// `frac`-quantile of a list (sorts in place). Empty -> 0.0.
fn quantile(v: &mut [f32], frac: f32) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((v.len() as f32 * frac) as usize).min(v.len() - 1);
    v[idx]
}

/// Full-wave rectify then centred moving-average smooth (AM demodulation).
fn moving_average_abs(samples: &[f32], win: usize) -> Vec<f32> {
    centred_moving_average(samples, win, true)
}

/// Centred moving average (no rectification).
fn moving_average(samples: &[f32], win: usize) -> Vec<f32> {
    centred_moving_average(samples, win, false)
}

/// Centred moving average over `win` samples, optionally rectifying first.
/// O(n) via a prefix sum.
fn centred_moving_average(samples: &[f32], win: usize, rectify: bool) -> Vec<f32> {
    let n = samples.len();
    let mut out = vec![0.0_f32; n];
    let val = |x: f32| if rectify { x.abs() } else { x };
    if win <= 1 {
        for (o, &x) in out.iter_mut().zip(samples) {
            *o = val(x);
        }
        return out;
    }
    let mut prefix = vec![0.0_f32; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + val(samples[i]);
    }
    let half = win / 2;
    for (i, o) in out.iter_mut().enumerate() {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let cnt = (hi - lo) as f32;
        *o = (prefix[hi] - prefix[lo]) / cnt;
    }
    out
}

/// Hann window value at index `i` of a length-`m` window.
#[inline]
fn hann(i: usize, m: usize) -> f32 {
    if m <= 1 {
        return 1.0;
    }
    let x = PI * i as f32 / (m - 1) as f32;
    let s = x.sin();
    s * s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn am_harmonic(f0: f32, am_hz: f32, sr: u32, secs: f32) -> Vec<f32> {
        let n = (secs * sr as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let am = 1.0 + 0.4 * (2.0 * PI * am_hz * t).sin();
                let mut v = 0.0;
                for h in 1..=6 {
                    v += (0.5 / h as f32) * (2.0 * PI * f0 * h as f32 * t).sin();
                }
                0.6 * am * v
            })
            .collect()
    }

    fn pseudo_noise(n: usize, seed: u32) -> Vec<f32> {
        let mut x = seed.max(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn pure_tone(f: f32, sr: u32, secs: f32) -> Vec<f32> {
        let n = (secs * sr as f32) as usize;
        (0..n)
            .map(|i| 0.6 * (2.0 * PI * f * i as f32 / sr as f32).sin())
            .collect()
    }

    /// A simple labelled train set lets us exercise `fit` deterministically.
    fn train_set(sr: u32) -> Vec<Sample> {
        let mut out = Vec::new();
        for (k, f0) in [110.0, 130.0, 150.0, 170.0].iter().enumerate() {
            out.push(Sample {
                id: format!("pos{k}"),
                samples: am_harmonic(*f0, 30.0 + k as f32 * 5.0, sr, 1.0),
                sample_rate: sr,
                label: 1,
            });
        }
        for k in 0..4 {
            let neg = if k % 2 == 0 {
                pseudo_noise(sr as usize, 7 + k)
            } else {
                pure_tone(2000.0 + k as f32 * 300.0, sr, 1.0)
            };
            out.push(Sample {
                id: format!("neg{k}"),
                samples: neg,
                sample_rate: sr,
                label: 0,
            });
        }
        out
    }

    #[test]
    fn features_are_finite() {
        let sr = 16_000;
        let f = physics_features(&am_harmonic(120.0, 25.0, sr, 1.0), sr);
        assert!(f.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn unfitted_scores_half() {
        let det = PhysicsFused::new();
        assert_eq!(det.score(&[0.1, -0.1, 0.2, 0.0, 0.1], 16_000), 0.5);
    }

    #[test]
    fn fitted_drone_beats_noise_and_tone() {
        let sr = 16_000;
        let mut det = PhysicsFused::new();
        det.fit(&train_set(sr));

        let drone = det.score(&am_harmonic(140.0, 35.0, sr, 1.0), sr);
        let noise = det.score(&pseudo_noise(sr as usize, 99), sr);
        let tone = det.score(&pure_tone(3000.0, sr, 1.0), sr);

        assert!((0.0..=1.0).contains(&drone));
        assert!((0.0..=1.0).contains(&noise));
        assert!((0.0..=1.0).contains(&tone));
        assert!(drone > noise, "drone {drone} should beat noise {noise}");
        assert!(drone > tone, "drone {drone} should beat tone {tone}");
    }

    #[test]
    fn silence_is_zero_after_fit() {
        let sr = 16_000;
        let mut det = PhysicsFused::new();
        det.fit(&train_set(sr));
        assert_eq!(det.score(&vec![0.0; sr as usize], sr), 0.0);
    }

    #[test]
    fn deterministic() {
        let sr = 16_000;
        let mut a = PhysicsFused::new();
        let mut b = PhysicsFused::new();
        a.fit(&train_set(sr));
        b.fit(&train_set(sr));
        let clip = am_harmonic(120.0, 25.0, sr, 1.0);
        assert_eq!(a.score(&clip, sr), b.score(&clip, sr));
    }
}
