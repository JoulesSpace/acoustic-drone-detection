//! Harmonic Product Spectrum / harmonic-comb drone detector.
//!
//! A multirotor's acoustic signature is a blade-pass fundamental (BPF) with a
//! stack of integer harmonics, plus a broad high-frequency motor whine. White
//! noise, an isolated low hum, or a single bright tone all *lack* that
//! regularly-spaced harmonic comb. This approach measures how strongly a
//! harmonic stack is present.
//!
//! For each frame we:
//!  1. Build a Harmonic Product Spectrum (HPS) by multiplying the magnitude
//!     spectrum with `R-1` downsampled copies of itself. Where a true
//!     fundamental sits, every decimated copy has energy, so the product spikes;
//!     spurious peaks get suppressed. The HPS peak inside the BPF range is the
//!     candidate f0.
//!  2. Score that candidate with a *harmonic-comb contrast*: each harmonic
//!     `f0, 2f0, 3f0, ...` is compared against its *local* off-comb background
//!     (the inter-harmonic valleys), and we reward many teeth each standing well
//!     above their local floor. This keys on the regular spacing itself and is
//!     invariant to absolute level and to broadband off-comb energy — unlike a
//!     fraction-of-total measure, which on real recordings is *inverted* by the
//!     motor's broadband hiss (see the bug note in `comb_contrast`).
//!
//! The comb contrast is multiplied by a mild high-frequency motor-whine bonus
//! (real drones radiate broadband 2.5-7 kHz hiss that environmental negatives
//! lack). Per-frame values are aggregated with a robust high-quantile (drone
//! presence is intermittent across a clip), then squashed into `[0, 1]` with a
//! logistic. The mapping is unsupervised but `fit` calibrates the logistic
//! centre from the training scores so the 0.5 decision boundary lands sensibly.

use drone_dsp::{bin_to_hz, hz_to_bin, NUM_BINS};

use crate::approach::Approach;
use crate::dataset::Sample;
use crate::util::spectra;

/// Plausible blade-pass fundamental range, Hz.
///
/// On real multirotor recordings (DADS) the dominant blade/motor fundamental
/// sits around 200-260 Hz, not the textbook ~100 Hz; the strong sub-100 Hz
/// energy in this dataset belongs to the *negatives* (urban/wind rumble). We
/// search ~100-330 Hz so we cover both the real drones and the lower synthetic
/// fundamentals (110-120 Hz) while excluding the negatives' rumble band.
const BPF_LO_HZ: f32 = 100.0;
const BPF_HI_HZ: f32 = 330.0;
/// Number of HPS factors (fundamental + harmonics 2..=R) used to pick f0.
const R: usize = 5;
/// Number of harmonics scored in the comb-contrast feature.
const COMB_HARMONICS: usize = 10;
/// Half-width (in bins) of the window summed around each predicted harmonic, to
/// tolerate inexact f0 and spectral leakage.
const HARM_HALF_WIDTH: usize = 1;
/// Motor-whine band, Hz. Real multirotors radiate a broad high-frequency motor
/// hiss here that environmental negatives largely lack; used as a mild bonus.
const MOTOR_LO_HZ: f32 = 2500.0;
const MOTOR_HI_HZ: f32 = 7000.0;

/// Harmonic Product Spectrum / harmonic-comb detector.
pub struct Hps {
    /// Logistic centre: comb ratio mapped to confidence 0.5. Calibrated by
    /// `fit`; the default is a reasonable prior for the synthetic data.
    center: f32,
    /// Logistic steepness.
    scale: f32,
}

impl Hps {
    pub fn new() -> Self {
        // Default prior for the harmonic-comb-contrast scale: a clean stack
        // scores tens-to-hundreds, noise/lone tones score ~0, so the 0.5
        // boundary sits in the valley around 20. `fit` refines this per dataset.
        Self {
            center: 20.0,
            scale: 0.12,
        }
    }

    /// Aggregate harmonic-comb strength for a whole clip into a raw, unbounded
    /// score (higher = more harmonic). Returns 0.0 for silence/too-short input.
    fn raw_strength(&self, samples: &[f32], sample_rate: u32) -> f32 {
        let frames = spectra(samples);
        if frames.is_empty() {
            return 0.0;
        }

        let lo_bin = hz_to_bin(BPF_LO_HZ, sample_rate).max(1);
        let hi_bin = hz_to_bin(BPF_HI_HZ, sample_rate).min(NUM_BINS - 1);
        if lo_bin >= hi_bin {
            return 0.0;
        }

        let mut per_frame: Vec<f32> = Vec::with_capacity(frames.len());
        let mut motor: Vec<f32> = Vec::with_capacity(frames.len());
        for spec in &frames {
            per_frame.push(frame_comb_contrast(spec, lo_bin, hi_bin, sample_rate));
            motor.push(motor_band_ratio(spec, sample_rate));
        }

        // Robust aggregate: ~80th percentile of per-frame comb contrasts. Drone
        // energy is intermittent, so we reward clips whose best frames look
        // strongly harmonic without letting a single outlier dominate.
        per_frame.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        motor.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q = |v: &[f32], frac: f32| v[((v.len() as f32 * frac) as usize).min(v.len() - 1)];
        let comb = q(&per_frame, 0.8);
        // Median motor-band fraction: a steady broadband motor whine is present
        // across most drone frames, so the median (not a tail quantile) captures
        // it without being fooled by a single bright transient in a negative.
        let motor_med = q(&motor, 0.5);

        // Combine the (level-invariant) harmonic comb contrast with a mild
        // motor-whine bonus. The comb dominates; the bonus only nudges genuine
        // multirotor recordings (broadband 2.5-7 kHz hiss) upward. Synthetic
        // harmonic clips have little motor-band energy and still score on comb.
        comb * (1.0 + 4.0 * motor_med)
    }
}

impl Default for Hps {
    fn default() -> Self {
        Self::new()
    }
}

/// HPS-guided harmonic-comb contrast for one magnitude spectrum.
///
/// Finds the candidate fundamental via a Harmonic Product Spectrum, then scores
/// it with a *local* harmonic-to-background contrast (guarding against the
/// classic octave error by also trying half the candidate f0). Higher = more
/// clearly a regularly-spaced harmonic stack.
fn frame_comb_contrast(
    spec: &[f32; NUM_BINS],
    lo_bin: usize,
    hi_bin: usize,
    sample_rate: u32,
) -> f32 {
    let total: f32 = spec.iter().sum();
    if total <= f32::EPSILON {
        return 0.0;
    }

    // --- 1. Harmonic Product Spectrum over the candidate f0 range. ---
    // hps[b] = prod_{r=1..=R} spec[b * r], for b*r < NUM_BINS.
    let mut best_bin = lo_bin;
    let mut best_hps = -1.0_f32;
    for b in lo_bin..=hi_bin {
        let mut prod = 1.0_f32;
        for r in 1..=R {
            let idx = b * r;
            if idx >= NUM_BINS {
                // Past Nyquist: treat missing harmonic as a small floor so a
                // genuine fundamental is not zeroed by a single absent bin.
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

    // --- 2. Comb contrast for the candidate, with an octave-error guard. ---
    // Real drones sometimes make 2*f0 the strongest HPS peak; checking f0/2 as
    // an alternate candidate recovers the true comb when that happens.
    let cand_a = best_bin;
    let cand_b = best_bin / 2;
    let c_a = comb_contrast(spec, cand_a, sample_rate);
    let c_b = if cand_b >= 1 && bin_to_hz(cand_b, sample_rate) >= 0.5 * BPF_LO_HZ {
        comb_contrast(spec, cand_b, sample_rate)
    } else {
        0.0
    };
    c_a.max(c_b)
}

/// Harmonic-comb *contrast* for fundamental bin `f0_bin`.
///
/// For each harmonic `h*f0` we compare the on-comb peak against the *local*
/// off-comb background (the spectrum halfway to the neighbouring harmonics).
/// A regularly-spaced stack has every tooth standing well above its local
/// background → high contrast, regardless of the clip's absolute level or how
/// much broadband energy it has elsewhere. This is the key fix: the old feature
/// normalised by *total* magnitude, which penalised real drones (whose motors
/// add lots of off-comb broadband energy) and rewarded tonal negatives. A local
/// contrast keys on the *regular spacing* itself, so a single bright tone (one
/// tooth, no stack) or broadband noise (no teeth above local background) both
/// score low while a true comb scores high.
fn comb_contrast(spec: &[f32; NUM_BINS], f0_bin: usize, sample_rate: u32) -> f32 {
    if f0_bin == 0 {
        return 0.0;
    }
    // Drone harmonic energy fades well before Nyquist; cap the comb at ~5 kHz.
    let max_hz = 5000.0_f32.min(bin_to_hz(NUM_BINS - 1, sample_rate));
    let max_bin = hz_to_bin(max_hz, sample_rate);

    // Absolute energy gate: a tooth must carry real energy, not just sit above a
    // near-zero local floor. Without this, the quiet inter-bin regions of a pure
    // tone's spectrum produce huge but meaningless local-contrast ratios (FFT
    // numerical noise / leakage tails), faking a comb. We require each tooth to
    // exceed a small fraction of the spectrum's strongest bin.
    let spec_max = spec.iter().copied().fold(0.0_f32, f32::max);
    let energy_floor = 0.02 * spec_max;

    let mut contrast_sum = 0.0_f32;
    let mut teeth = 0u32;
    for h in 1..=COMB_HARMONICS {
        let center = f0_bin * h;
        if center > max_bin || center >= NUM_BINS - 1 {
            break;
        }
        // On-comb peak in a small tolerance window.
        let lo = center.saturating_sub(HARM_HALF_WIDTH);
        let hi = (center + HARM_HALF_WIDTH).min(NUM_BINS - 1);
        let mut peak = 0.0_f32;
        for &m in &spec[lo..=hi] {
            if m > peak {
                peak = m;
            }
        }
        // Local off-comb background: the troughs roughly midway to the adjacent
        // harmonics (±f0/2), where a genuine comb should have little energy.
        let half = (f0_bin / 2).max(1);
        let bg = background_at(spec, center, half);
        // Contrast of this tooth above its local floor.
        let c = peak / (bg + f32::EPSILON);
        if c > 2.0 && peak > energy_floor {
            // Only well-formed teeth (clearly above local background) count, and
            // we accumulate a log-ish, saturating contribution so one huge tooth
            // can't masquerade as a full stack.
            contrast_sum += (c - 2.0).min(8.0);
            teeth += 1;
        }
    }
    if teeth < 2 {
        // A lone peak is not a harmonic stack.
        return 0.0;
    }
    // Reward both the number of well-formed teeth and their average prominence.
    let avg = contrast_sum / teeth as f32;
    (teeth as f32).min(8.0) * avg
}

/// Local off-comb background magnitude around bin `center`: the minimum of the
/// two trough regions sitting `offset` bins below and above the harmonic. Using
/// the min (the quieter shoulder) makes the contrast demanding — both shoulders
/// must be low for a tooth to count, which is exactly the inter-harmonic valley
/// signature of a real comb.
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

/// Fraction of total magnitude in the high-frequency motor-whine band.
///
/// Real multirotors radiate a broad 2.5-7 kHz motor hiss; most environmental /
/// urban negatives in DADS roll off below that. Used only as a mild bonus on top
/// of the comb contrast, so it helps real drones without being load-bearing for
/// the (HF-poor) synthetic clips.
fn motor_band_ratio(spec: &[f32; NUM_BINS], sample_rate: u32) -> f32 {
    let total: f32 = spec.iter().sum();
    if total <= f32::EPSILON {
        return 0.0;
    }
    let lo = hz_to_bin(MOTOR_LO_HZ, sample_rate).min(NUM_BINS - 1);
    let hi = hz_to_bin(MOTOR_HI_HZ, sample_rate).min(NUM_BINS - 1);
    if lo >= hi {
        return 0.0;
    }
    let band: f32 = spec[lo..=hi].iter().sum();
    (band / total).clamp(0.0, 1.0)
}

impl Approach for Hps {
    fn name(&self) -> &str {
        "hps"
    }

    fn description(&self) -> &str {
        "Harmonic Product Spectrum + harmonic-comb ratio over the blade-pass fundamental band"
    }

    fn fit(&mut self, train: &[Sample]) {
        // Calibrate the logistic centre to the valley between the per-class
        // median raw strengths (falls back to the default prior if a class is
        // missing). Unsupervised scoring would also work; this just centres the
        // 0.5 boundary where the classes separate.
        let mut pos: Vec<f32> = Vec::new();
        let mut neg: Vec<f32> = Vec::new();
        for s in train {
            let v = self.raw_strength(&s.samples, s.sample_rate);
            if s.label == 1 {
                pos.push(v);
            } else {
                neg.push(v);
            }
        }
        if pos.is_empty() || neg.is_empty() {
            return;
        }
        // Use class medians (robust to outliers) and put the logistic centre in
        // the valley between them. Steepness scales with the inter-median gap so
        // the confidence transitions across that valley.
        pos.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        neg.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = |xs: &[f32]| xs[xs.len() / 2];
        let mp = median(&pos);
        let mn = median(&neg);
        let center = 0.5 * (mp + mn);
        let spread = (mp - mn).abs().max(0.25);
        let scale = (4.0 / spread).clamp(0.5, 8.0);
        if center.is_finite() && center > 0.0 {
            self.center = center;
            self.scale = scale;
        }
    }

    fn score(&self, samples: &[f32], sample_rate: u32) -> f32 {
        let raw = self.raw_strength(samples, sample_rate);
        if !raw.is_finite() || raw <= 0.0 {
            return 0.0;
        }
        // Logistic squashing into [0, 1].
        let z = self.scale * (raw - self.center);
        let conf = 1.0 / (1.0 + (-z).exp());
        if conf.is_finite() {
            conf.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn harmonic_clip(f0: f32, sr: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|n| {
                let t = n as f32 / sr as f32;
                let mut s = 0.0;
                for h in 1..=6 {
                    s += (0.6 / h as f32) * (2.0 * PI * f0 * h as f32 * t).sin();
                }
                s * 0.8
            })
            .collect()
    }

    fn noise_clip(len: usize) -> Vec<f32> {
        let mut state = 0x2545_F491u32;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 1.6 - 0.8
            })
            .collect()
    }

    #[test]
    fn harmonic_scores_higher_than_noise() {
        let sr = 16_000;
        let len = sr as usize / 2;
        let hps = Hps::new();
        let drone = hps.score(&harmonic_clip(120.0, sr, len), sr);
        let noise = hps.score(&noise_clip(len), sr);
        assert!(drone > noise, "drone {drone} should exceed noise {noise}");
        assert!((0.0..=1.0).contains(&drone));
        assert!((0.0..=1.0).contains(&noise));
    }

    #[test]
    fn silence_and_empty_score_zero() {
        let hps = Hps::new();
        assert_eq!(hps.score(&[], 16_000), 0.0);
        assert_eq!(hps.score(&vec![0.0; 16_000], 16_000), 0.0);
    }

    #[test]
    fn bright_tone_is_not_harmonic() {
        let sr = 16_000;
        let len = sr as usize / 2;
        let tone: Vec<f32> = (0..len)
            .map(|n| 0.8 * (2.0 * PI * 3000.0 * n as f32 / sr as f32).sin())
            .collect();
        let hps = Hps::new();
        let drone = hps.score(&harmonic_clip(110.0, sr, len), sr);
        let tone_s = hps.score(&tone, sr);
        assert!(drone > tone_s, "drone {drone} should exceed tone {tone_s}");
    }
}
