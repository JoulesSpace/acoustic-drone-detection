//! Amplitude-envelope modulation-spectrum drone detector.
//!
//! A multirotor does more than emit a harmonic stack - it *modulates its
//! loudness* periodically. Each blade sweeping past the airframe (and small
//! per-revolution thrust ripples) imprints a slow amplitude modulation (AM) on
//! the radiated sound at the blade-pass / rotor rate, typically a few tens of
//! Hz up to a couple hundred. That periodic AM is a cue *orthogonal* to the
//! spectral-comb / harmonic detectors: it lives in the temporal **envelope**,
//! not in the fine spectral structure, so it survives even when the carrier
//! harmonics are smeared by reverberation or wind, and it rejects steady tones
//! and broadband noise (both of which have a flat, featureless envelope
//! spectrum).
//!
//! Pipeline:
//!   1. **Envelope.** Full-wave rectify (`abs`) the signal and smooth with a
//!      short moving average (cascaded twice, ~8 ms each) - a crude but
//!      effective AM demodulator whose cascade gives a steep enough roll-off to
//!      double as the decimation anti-alias filter.
//!   2. **Downsample.** Decimate the smoothed envelope to an envelope rate
//!      `fe ~= 1 kHz` (one sample per ~1 ms). The modulation band of interest
//!      (5-200 Hz) is far below the Nyquist of that rate.
//!   3. **Modulation spectrum.** Convert the envelope to a *modulation index*
//!      (divide by its mean, subtract 1), window it, and take a direct DFT
//!      periodogram over the 5-200 Hz band.
//!   4. **Confidence.** The raw strength is the **peak modulation-band power /
//!      total envelope-spectrum power** - the fraction of envelope variation in
//!      the single strongest periodic line. It is mapped through a logistic into
//!      `[0, 1]`; [`Approach::fit`] calibrates the logistic centre/steepness
//!      from the train-set score distribution.
//!
//! Silence and sub-window clips score `0.0`. Everything is deterministic.

use drone_dsp::FRAME_SIZE;

use crate::approach::Approach;
use crate::dataset::Sample;

/// Target envelope (decimated) sample rate in Hz. One envelope sample per ~1 ms.
const ENV_RATE_HZ: f32 = 1000.0;

/// Modulation band of interest (blade-pass / rotor AM), in Hz.
const MOD_LO_HZ: f32 = 5.0;
const MOD_HI_HZ: f32 = 200.0;

/// Envelope-smoothing window, in milliseconds (AM demodulation low-pass).
/// Applied twice for a steeper roll-off - a moving average alone has fat
/// side-lobes, which would let carrier energy alias into the modulation band
/// after decimation; cascading two of them suppresses that cleanly.
const SMOOTH_MS: f32 = 8.0;

/// Amplitude-envelope modulation detector.
pub struct EnvelopePeriodicity {
    /// Logistic centre - raw modulation strengths above this lean "drone".
    center: f32,
    /// Logistic steepness.
    scale: f32,
}

impl EnvelopePeriodicity {
    /// Construct with sensible defaults; [`Approach::fit`] refines the logistic.
    pub fn new() -> Self {
        Self {
            center: 0.4,
            scale: 10.0,
        }
    }

    /// Raw modulation strength in roughly `[0, 1]`, *before* logistic squashing.
    /// This is what `fit` calibrates against and what `score` feeds through the
    /// logistic. Returns `0.0` for silence / too-short input.
    fn modulation_strength(&self, samples: &[f32], sample_rate: u32) -> f32 {
        if sample_rate == 0 || samples.len() < FRAME_SIZE {
            return 0.0;
        }
        let energy: f32 = samples.iter().map(|&x| x * x).sum();
        if !energy.is_finite() || energy <= 1e-6 {
            return 0.0;
        }

        // --- 1. Rectified + smoothed amplitude envelope. -----------------
        // Moving-average window length in source samples (>= 1). Cascade two
        // passes for a steeper anti-alias roll-off ahead of decimation.
        let smooth_n = ((SMOOTH_MS * 1e-3) * sample_rate as f32).round() as usize;
        let smooth_n = smooth_n.max(1);
        let smoothed = moving_average(&moving_average_abs(samples, smooth_n), smooth_n);

        // --- 2. Decimate to the envelope rate ~ENV_RATE_HZ. --------------
        // Integer decimation factor; the prior smoothing is the anti-alias.
        let decim = (sample_rate as f32 / ENV_RATE_HZ).round() as usize;
        let decim = decim.max(1);
        let fe = sample_rate as f32 / decim as f32; // actual envelope rate
        let env: Vec<f32> = smoothed.iter().step_by(decim).copied().collect();
        let m = env.len();
        // Need enough envelope samples to resolve a few cycles of MOD_LO_HZ.
        if m < 32 {
            return 0.0;
        }

        // --- 3. Modulation index, mean-remove, window. -------------------
        // Normalise the envelope by its mean so the feature is a *modulation
        // index* (fractional loudness swing), independent of absolute level.
        let mean = env.iter().sum::<f32>() / m as f32;
        if !mean.is_finite() || mean <= 1e-9 {
            return 0.0;
        }
        // Hann window; accumulate the total windowed power (Parseval reference
        // for the whole envelope spectrum).
        let mut win = vec![0.0_f32; m];
        let mut total_power = 0.0_f32;
        for (i, &e) in env.iter().enumerate() {
            let w = hann(i, m);
            let v = (e / mean - 1.0) * w; // mean-removed modulation index
            win[i] = v;
            total_power += v * v;
        }
        // `total_power` is the time-domain energy of the windowed, mean-removed
        // modulation-index signal; by Parseval it is proportional to the total
        // power summed over the whole modulation spectrum. It is the
        // denominator that makes the peak a *fraction of total* envelope
        // variation.
        if !total_power.is_finite() || total_power <= 1e-12 {
            return 0.0;
        }

        // --- 4. Modulation-band periodogram via a direct DFT. ------------
        // Frequency resolution of the periodogram: df = fe / m.
        let df = fe / m as f32;
        let nyq = fe * 0.5;
        let hi = MOD_HI_HZ.min(nyq);
        if hi <= MOD_LO_HZ {
            return 0.0;
        }
        let k_lo = (MOD_LO_HZ / df).floor().max(1.0) as usize;
        let k_hi = (hi / df).ceil() as usize;
        let k_hi = k_hi.min(m / 2);
        if k_lo >= k_hi {
            return 0.0;
        }

        // Strongest single line in the modulation band: |X_k|^2 peak.
        let two_pi_over_m = 2.0 * core::f32::consts::PI / m as f32;
        let mut peak = 0.0_f32;
        for k in k_lo..=k_hi {
            let mut re = 0.0_f32;
            let mut im = 0.0_f32;
            let wk = two_pi_over_m * k as f32;
            for (n, &x) in win.iter().enumerate() {
                let ang = wk * n as f32;
                re += x * cos_f32(ang);
                im -= x * sin_f32(ang);
            }
            let p = re * re + im * im;
            if p > peak {
                peak = p;
            }
        }

        // --- 5. Peak-power / total-power. --------------------------------
        // The fraction of the envelope's total variation concentrated in the
        // single strongest modulation line. Periodic blade-pass AM dumps most
        // of its envelope variance into one (or a few) lines, so this fraction
        // is high; noise and steady tones spread their (small) envelope
        // variance flat across the spectrum, so it is low. `|X_k|^2` and
        // `total_power` differ by the constant Parseval factor `m`, which we
        // divide out to land the ratio in a clean ~[0, 1) range.
        let strength = (peak / (m as f32 * total_power)).clamp(0.0, 1.0);
        if strength.is_finite() {
            strength
        } else {
            0.0
        }
    }
}

impl Default for EnvelopePeriodicity {
    fn default() -> Self {
        Self::new()
    }
}

impl Approach for EnvelopePeriodicity {
    fn name(&self) -> &str {
        "envelope_periodicity"
    }

    fn description(&self) -> &str {
        "amplitude-envelope modulation-spectrum periodicity (blade-pass AM)"
    }

    fn fit(&mut self, train: &[Sample]) {
        // Calibrate the logistic centre to the midpoint between the mean
        // positive and mean negative raw modulation strength, and the steepness
        // from the class gap - so the decision boundary lands where the classes
        // separate on *this* data.
        let mut pos_sum = 0.0_f32;
        let mut pos_n = 0usize;
        let mut neg_sum = 0.0_f32;
        let mut neg_n = 0usize;

        for s in train {
            let p = self.modulation_strength(&s.samples, s.sample_rate);
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
            // Place the class means near the logistic's saturating ends; guard
            // a degenerate (near-zero) gap so `scale` stays finite & sane.
            let gap = (pos_mean - neg_mean).abs().max(1e-3);
            let scale = 4.0 / gap;
            if scale.is_finite() {
                self.scale = scale.clamp(4.0, 40.0);
            }
        }
    }

    fn score(&self, samples: &[f32], sample_rate: u32) -> f32 {
        let p = self.modulation_strength(samples, sample_rate);
        if p <= 0.0 {
            // Includes the silence / too-short guards inside modulation_strength.
            return 0.0;
        }
        let z = self.scale * (p - self.center);
        let s = 1.0 / (1.0 + (-z).exp());
        if s.is_finite() {
            s.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Full-wave rectify (`abs`) then smooth with a centred moving average of length
/// `win` samples. A crude but effective amplitude-modulation demodulator: the
/// `abs` shifts the AM down to baseband, the average low-passes away the carrier.
fn moving_average_abs(samples: &[f32], win: usize) -> Vec<f32> {
    centred_moving_average(samples, win, true)
}

/// Centred moving average of a signal (no rectification), length `win`.
fn moving_average(samples: &[f32], win: usize) -> Vec<f32> {
    centred_moving_average(samples, win, false)
}

/// Centred moving average over a window of `win` samples, optionally rectifying
/// each input sample with `abs` first. Implemented with a prefix sum, so O(n).
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
    // Running sum over a sliding window via a prefix-sum.
    let mut prefix = vec![0.0_f32; n + 1];
    for i in 0..n {
        prefix[i + 1] = prefix[i] + val(samples[i]);
    }
    let half = win / 2;
    for (i, o) in out.iter_mut().enumerate() {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(n);
        let count = (hi - lo) as f32;
        *o = (prefix[hi] - prefix[lo]) / count;
    }
    out
}

/// Hann window value at index `i` of a length-`m` window.
#[inline]
fn hann(i: usize, m: usize) -> f32 {
    if m <= 1 {
        return 1.0;
    }
    let x = core::f32::consts::PI * i as f32 / (m - 1) as f32;
    let s = sin_f32(x);
    s * s // sin^2 = 0.5*(1 - cos(2x)), the Hann window
}

/// `cos` via std (host crate, no `no_std` constraint here).
#[inline]
fn cos_f32(x: f32) -> f32 {
    x.cos()
}

/// `sin` via std.
#[inline]
fn sin_f32(x: f32) -> f32 {
    x.sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// A harmonic carrier whose amplitude is modulated at `am_hz` - a drone-like
    /// signal (blade-pass AM on a rotor harmonic stack).
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

    /// A single pure sine has a perfectly constant amplitude envelope, so its
    /// mean-removed modulation index is ~0 - the canonical "no AM" negative.
    fn pure_tone(f: f32, sr: u32, secs: f32) -> Vec<f32> {
        let n = (secs * sr as f32) as usize;
        (0..n)
            .map(|i| 0.6 * (2.0 * PI * f * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn modulated_beats_steady_and_noise() {
        let sr = 16_000;
        let det = EnvelopePeriodicity::new();

        let modulated = am_harmonic(150.0, 40.0, sr, 1.0);
        // A steady tone (flat envelope) and broadband noise (flat envelope
        // spectrum) are both "no periodic AM" negatives.
        let tone = pure_tone(440.0, sr, 1.0);
        let noise = pseudo_noise(sr as usize, 7);

        let sm = det.modulation_strength(&modulated, sr);
        let st = det.modulation_strength(&tone, sr);
        let sn = det.modulation_strength(&noise, sr);

        assert!(sm > st, "modulated {sm} should beat steady tone {st}");
        assert!(sm > sn, "modulated {sm} should beat noise {sn}");
    }

    #[test]
    fn silence_is_zero() {
        let det = EnvelopePeriodicity::new();
        assert_eq!(det.score(&vec![0.0; 16_000], 16_000), 0.0);
    }

    #[test]
    fn too_short_is_finite_zero() {
        let det = EnvelopePeriodicity::new();
        let s = det.score(&[0.1, -0.1, 0.2], 16_000);
        assert!(s.is_finite() && (0.0..=1.0).contains(&s));
        assert_eq!(s, 0.0);
    }

    #[test]
    fn score_is_in_unit_interval() {
        let sr = 16_000;
        let det = EnvelopePeriodicity::new();
        let modulated = am_harmonic(120.0, 25.0, sr, 1.0);
        let s = det.score(&modulated, sr);
        assert!(s.is_finite() && (0.0..=1.0).contains(&s));
    }
}
