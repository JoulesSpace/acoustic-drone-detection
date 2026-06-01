//! Cepstrum / autocorrelation periodicity drone detector.
//!
//! A multirotor's acoustic signature is a *harmonic stack*: a blade-pass
//! fundamental and a comb of equally-spaced overtones. Two equivalent views of
//! that regularity drive this detector:
//!
//! 1. **Time-domain autocorrelation.** A periodic waveform correlates strongly
//!    with a lag-shifted copy of itself at the period `lag = sample_rate / f0`.
//!    The normalised autocorrelation peak (excluding lag 0) in the lag band for
//!    plausible drone fundamentals is a clean, robust periodicity score.
//!
//! 2. **Real cepstrum.** The harmonic comb is *periodic in frequency* with
//!    spacing `f0`. The inverse transform of the log-magnitude spectrum (the
//!    real cepstrum) therefore shows a peak at the quefrency matching that
//!    spacing. We compute it with a direct cosine transform (DCT-II) over the
//!    512 log-magnitudes — O(N^2) per frame, which is fine at benchmark scale.
//!
//! Both cues respond to *regular harmonic structure* and reject white noise,
//! single tones, and slow hum. We aggregate per-frame scores robustly and map
//! the result into `[0, 1]` with a logistic. Silence scores `0.0`.

use std::f32::consts::PI;

use drone_dsp::{FRAME_SIZE, NUM_BINS};

use crate::approach::Approach;
use crate::dataset::Sample;
use crate::util::spectra;

/// Cepstrum / autocorrelation periodicity detector.
pub struct Cepstrum {
    /// Logistic centre — periodicity scores above this lean "drone".
    center: f32,
    /// Logistic steepness.
    scale: f32,
}

impl Cepstrum {
    /// Construct with reasonable defaults; [`Approach::fit`] refines `center`.
    pub fn new() -> Self {
        Self {
            center: 0.45,
            scale: 14.0,
        }
    }

    /// Raw periodicity strength of a signal in roughly `[0, 1]`, *before* the
    /// logistic squashing. This is what `fit` calibrates against and what
    /// `score` feeds through the logistic.
    fn periodicity(&self, samples: &[f32], sample_rate: u32) -> f32 {
        // --- Cue 1: time-domain autocorrelation peak, per frame. ---------
        let acf = autocorr_strength(samples, sample_rate);

        // --- Cue 2: cepstral peak from the log-magnitude spectra. --------
        let cep = cepstral_strength(samples, sample_rate);

        // Both cues live in [0, 1] and respond to the same underlying
        // harmonic regularity, but fail differently: autocorrelation can be
        // fooled by a single tone whose period-multiple lands in band, while
        // the cepstral comb test cannot. Averaging them cancels those
        // independent failure modes — an equal blend separates the classes
        // best across seeds.
        0.5 * acf + 0.5 * cep
    }
}

impl Default for Cepstrum {
    fn default() -> Self {
        Self::new()
    }
}

impl Approach for Cepstrum {
    fn name(&self) -> &str {
        "cepstrum"
    }

    fn description(&self) -> &str {
        "harmonic periodicity via autocorrelation + real cepstrum peak"
    }

    fn fit(&mut self, train: &[Sample]) {
        // Calibrate the logistic centre to the midpoint between the mean
        // positive and mean negative raw periodicity, so the decision boundary
        // sits where the classes separate.
        let mut pos_sum = 0.0_f32;
        let mut pos_n = 0usize;
        let mut neg_sum = 0.0_f32;
        let mut neg_n = 0usize;

        for s in train {
            let p = self.periodicity(&s.samples, s.sample_rate);
            if s.label == 1 {
                pos_sum += p;
                pos_n += 1;
            } else {
                neg_sum += p;
                neg_n += 1;
            }
        }

        if pos_n > 0 && neg_n > 0 {
            let pos_mean = pos_sum / pos_n as f32;
            let neg_mean = neg_sum / neg_n as f32;
            let mid = 0.5 * (pos_mean + neg_mean);
            if mid.is_finite() {
                self.center = mid;
            }
            // Set steepness so the class means land near the logistic's
            // saturating ends. Guard against a degenerate (zero) gap.
            let gap = (pos_mean - neg_mean).abs().max(1e-3);
            self.scale = (4.0 / gap).clamp(4.0, 40.0);
        }
    }

    fn score(&self, samples: &[f32], sample_rate: u32) -> f32 {
        // Guard silence (and sub-frame input): no signal, no drone.
        let energy: f32 = samples.iter().map(|&x| x * x).sum();
        if samples.len() < FRAME_SIZE || energy <= 1e-6 {
            return 0.0;
        }

        let p = self.periodicity(samples, sample_rate);
        // Logistic squashing into [0, 1].
        let z = self.scale * (p - self.center);
        let s = 1.0 / (1.0 + (-z).exp());
        if s.is_finite() {
            s.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Energy of a frame (sum of squares). Used to skip silence.
fn frame_energy(frame: &[f32]) -> f32 {
    frame.iter().map(|&x| x * x).sum()
}

/// Per-frame normalised autocorrelation peak in the drone-fundamental lag band,
/// aggregated as a trimmed/robust mean over frames.
///
/// For a fundamental `f0`, the period in samples is `lag = sample_rate / f0`.
/// We search `f0` in `[80, 250] Hz`. The normalised peak is `r(lag) / r(0)`,
/// which is `1.0` for a perfectly periodic signal and near `0` for white noise.
fn autocorr_strength(samples: &[f32], sample_rate: u32) -> f32 {
    if samples.len() < FRAME_SIZE {
        return 0.0;
    }
    let hop = FRAME_SIZE / 2;

    // Lag search range from the fundamental band.
    let min_lag = (sample_rate as f32 / 250.0).floor() as usize;
    let max_lag = (sample_rate as f32 / 80.0).ceil() as usize;
    let max_lag = max_lag.min(FRAME_SIZE - 1);
    if min_lag < 1 || min_lag >= max_lag {
        return 0.0;
    }

    let mut peaks: Vec<f32> = Vec::new();
    let mut start = 0usize;
    while start + FRAME_SIZE <= samples.len() {
        let frame = &samples[start..start + FRAME_SIZE];
        start += hop;

        let r0 = frame_energy(frame);
        // Skip near-silent frames so they neither help nor hurt.
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

    robust_aggregate(&mut peaks)
}

/// Per-frame real-cepstrum peak in the plausible-BPF quefrency band, aggregated
/// robustly across frames.
///
/// We take `log(1 + magnitude)` over the 512-bin spectrum (mean-removed to drop
/// the spectral tilt / envelope term), run a DCT-II to get real cepstral
/// coefficients, then measure the strongest peak in a quefrency band that
/// corresponds to harmonic spacings of ~80-250 Hz. The peak is normalised by
/// the mean cepstral magnitude over a slightly wider band (its local baseline);
/// this peak-to-mean ratio is large for a regular harmonic comb and ~1 for
/// noise or a flat spectrum, and is mapped into `[0, 1]`.
fn cepstral_strength(samples: &[f32], sample_rate: u32) -> f32 {
    let frames = spectra(samples);
    if frames.is_empty() {
        return 0.0;
    }

    // Quefrency index `q` of the DCT corresponds to a harmonic spacing of
    // roughly `df * NUM_BINS / q` Hz, where `df = sample_rate / FRAME_SIZE` is
    // the bin width. Map the 80-250 Hz spacing band to a `q` range via
    // `q = df * NUM_BINS / spacing_hz`.
    let df = sample_rate as f32 / FRAME_SIZE as f32;
    let q_lo = (df * NUM_BINS as f32 / 250.0).floor() as usize;
    let q_hi = (df * NUM_BINS as f32 / 80.0).ceil() as usize;
    let q_lo = q_lo.max(4); // skip tilt / envelope
    let q_hi = q_hi.min(NUM_BINS - 1);
    if q_lo >= q_hi {
        return 0.0;
    }

    let n = NUM_BINS;
    let mut peaks: Vec<f32> = Vec::with_capacity(frames.len());

    // Reusable scratch buffers.
    let mut logmag = vec![0.0_f32; n];
    // We evaluate cepstral coefficients across the search band plus a small
    // margin so a peak-to-local-baseline ratio is well defined at the edges.
    let margin = ((q_hi - q_lo) / 4).max(8);
    let eval_lo = q_lo.saturating_sub(margin).max(1);
    let eval_hi = (q_hi + margin).min(n - 1);
    let mut coeffs = vec![0.0_f32; eval_hi - eval_lo + 1];

    for spec in &frames {
        // Skip silent frames.
        let e: f32 = spec.iter().map(|&m| m * m).sum();
        if e <= 1e-9 {
            continue;
        }
        for i in 0..n {
            logmag[i] = (1.0 + spec[i]).ln();
        }
        // Mean-remove so the q=0 (DC / envelope) term doesn't leak in.
        let mean = logmag.iter().sum::<f32>() / n as f32;
        for v in logmag.iter_mut() {
            *v -= mean;
        }

        // Cepstral magnitudes over the evaluation band.
        for (slot, q) in (eval_lo..=eval_hi).enumerate() {
            coeffs[slot] = dct_coeff(&logmag, q, n).abs();
        }

        // Peak inside the *search* band, and the mean cepstral magnitude over
        // the whole evaluation band as a local baseline. The peak-to-mean
        // ratio is the classic cepstral pitch-strength: large for a regular
        // harmonic comb, ~1 for noise (no preferred quefrency).
        let mut peak = 0.0_f32;
        for (slot, q) in (eval_lo..=eval_hi).enumerate() {
            if (q_lo..=q_hi).contains(&q) && coeffs[slot] > peak {
                peak = coeffs[slot];
            }
        }
        let baseline = coeffs.iter().sum::<f32>() / coeffs.len() as f32 + 1e-9;
        // Peak-to-mean ratio: >= 0, ~1 for a flat/noisy cepstrum. Map the
        // useful ~1..6 range into [0, 1].
        let ratio = peak / baseline;
        let strength = ((ratio - 1.0) / 5.0).clamp(0.0, 1.0);
        peaks.push(strength);
    }

    robust_aggregate(&mut peaks)
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

/// Robust aggregation of per-frame periodicity peaks: the mean of the upper
/// half (frames where the source is actually periodic), which is resistant to
/// frames where noise momentarily dominates. Empty input -> 0.0.
fn robust_aggregate(peaks: &mut [f32]) -> f32 {
    if peaks.is_empty() {
        return 0.0;
    }
    peaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let start = peaks.len() / 2; // upper half
    let upper = &peaks[start..];
    let sum: f32 = upper.iter().sum();
    sum / upper.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI as TPI;

    fn harmonic(f0: f32, sr: u32, secs: f32) -> Vec<f32> {
        let n = (secs * sr as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                (1..=6)
                    .map(|k| (0.6 / k as f32) * (2.0 * TPI * f0 * k as f32 * t).sin())
                    .sum()
            })
            .collect()
    }

    #[test]
    fn harmonic_scores_higher_than_noise() {
        let sr = 16_000;
        let det = Cepstrum::new();
        let h = harmonic(140.0, sr, 1.0);

        // Deterministic pseudo-noise.
        let mut x = 123u32;
        let noise: Vec<f32> = (0..sr)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        let sh = det.score(&h, sr);
        let sn = det.score(&noise, sr);
        assert!(sh > sn, "harmonic {sh} should beat noise {sn}");
        assert!((0.0..=1.0).contains(&sh));
        assert!((0.0..=1.0).contains(&sn));
    }

    #[test]
    fn silence_is_zero() {
        let det = Cepstrum::new();
        let s = det.score(&vec![0.0; 16_000], 16_000);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn too_short_is_finite() {
        let det = Cepstrum::new();
        let s = det.score(&[0.1, -0.1, 0.2], 16_000);
        assert!(s.is_finite() && (0.0..=1.0).contains(&s));
    }
}
