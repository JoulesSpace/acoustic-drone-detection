//! Bare-metal, training-free acoustic drone detector.
//!
//! This crate is the *edge-deployment proof* for the project: it is `#![no_std]`,
//! `alloc`-free, and depends on [`drone_dsp`] with `default-features = false`, so
//! it cross-compiles for esp32-class microcontrollers (RISC-V `riscv32imc` and
//! Cortex-M `thumbv7em`). All float math goes through [`libm`]; nothing here
//! touches `std`.
//!
//! It works on **one** [`drone_dsp::FRAME_SIZE`]-sample frame at a time, which is
//! exactly how firmware would feed it from a DMA ring buffer. There is no model
//! and no training: the drone-likeness score is a fixed monotonic rule over three
//! cheap spectral features - spectral flatness, spectral entropy, and the
//! 100-4000 Hz band-energy ratio - squashed through a logistic. This is a direct
//! port of the *rule* (untrained) path of `drone-bench`'s `spectral_gate`
//! approach, which is the natural tiny-edge detector because it needs no learned
//! weights.
//!
//! Two layers of API:
//!
//! * [`drone_confidence`] - stateless: window + FFT + features + rule → score in
//!   `[0, 1]` for a single frame.
//! * [`EdgeDetector`] - stateful: EMA-smooths the per-frame confidence and raises
//!   a boolean alert once it holds above a threshold for `hold` frames, mirroring
//!   the live listener's alert logic. No heap, no `std`.

#![no_std]
#![forbid(unsafe_code)]

use drone_dsp::{
    band_energy, hann_in_place, magnitude_spectrum, total_energy, Spectrum, FRAME_SIZE, NUM_BINS,
};

/// Band of interest for drone tonals (matches the band-pass front-end intent).
const BAND_LO_HZ: f32 = 100.0;
/// Upper edge of the drone band.
const BAND_HI_HZ: f32 = 4000.0;

/// Small constant to keep logs/divisions finite.
const EPS: f32 = 1e-10;

/// Drone-likeness confidence in `[0, 1]` for a single audio frame.
///
/// `frame` is a block of `FRAME_SIZE` mono samples in roughly `[-1.0, 1.0]`. The
/// frame is copied internally (the FFT consumes its input), so the caller's
/// buffer is left untouched.
///
/// The score is a fixed, training-free rule: tonal/harmonic in-band content
/// (low flatness, low entropy, high band-energy ratio) scores high; broadband
/// noise scores low; silence scores ~0.
pub fn drone_confidence(frame: &[f32; FRAME_SIZE], sample_rate: u32) -> f32 {
    // Window into a scratch buffer; the FFT works in place on it.
    let mut scratch = *frame;
    hann_in_place(&mut scratch);
    let spectrum = magnitude_spectrum(&mut scratch);

    let total = total_energy(&spectrum);
    if total <= EPS {
        return 0.0; // silence => no drone
    }

    let flat = flatness(&spectrum);
    let ent = entropy(&spectrum);
    let band_ratio = clamp01(band_energy(&spectrum, BAND_LO_HZ, BAND_HI_HZ, sample_rate) / total);

    let s = rule_score(flat, ent, band_ratio);
    if s.is_finite() {
        clamp01(s)
    } else {
        0.0
    }
}

/// Hand-designed, monotonic drone-likeness rule, squashed through a logistic.
///
/// High band-ratio + low flatness + low entropy => high confidence. These are
/// the exact coefficients from `drone-bench`'s `spectral_gate::rule_score`.
#[inline]
fn rule_score(flatness: f32, entropy: f32, band_ratio: f32) -> f32 {
    let z = -4.0 + 5.0 * band_ratio + 3.0 * (1.0 - flatness) + 3.0 * (1.0 - entropy);
    sigmoid(z)
}

/// Spectral flatness: geometric mean / arithmetic mean of magnitudes, in
/// `[0, 1]`. ~1 for flat (noise-like) spectra, low for peaky/tonal ones. The
/// geometric mean is computed in the log domain for numerical stability.
fn flatness(spec: &Spectrum) -> f32 {
    let mut log_sum = 0.0_f32;
    let mut arith = 0.0_f32;
    for &m in spec.iter() {
        let v = m + EPS;
        log_sum += libm::logf(v);
        arith += v;
    }
    let n = NUM_BINS as f32;
    let geo = libm::expf(log_sum / n);
    let arith_mean = arith / n;
    if arith_mean > EPS {
        clamp01(geo / arith_mean)
    } else {
        1.0
    }
}

/// Normalized Shannon spectral entropy of the power spectrum, in `[0, 1]`.
/// ~1 for uniform (flat) spectra, low for energy concentrated in few bins.
fn entropy(spec: &Spectrum) -> f32 {
    let mut sum = 0.0_f32;
    for &m in spec.iter() {
        sum += m * m;
    }
    if sum <= EPS {
        return 1.0;
    }
    let mut h = 0.0_f32;
    for &m in spec.iter() {
        let prob = (m * m) / sum;
        if prob > EPS {
            h -= prob * libm::logf(prob);
        }
    }
    let max_h = libm::logf(NUM_BINS as f32);
    if max_h > 0.0 {
        clamp01(h / max_h)
    } else {
        1.0
    }
}

/// Numerically stable logistic sigmoid.
#[inline]
fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + libm::expf(-z))
    } else {
        let e = libm::expf(z);
        e / (1.0 + e)
    }
}

/// Clamp to `[0, 1]`. `f32::clamp` is a `core` method, so this stays `no_std`.
/// Inputs here are always finite (NaN is filtered upstream), so the documented
/// `clamp` NaN/`min > max` caveats do not apply.
#[inline]
fn clamp01(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Stateful, `no_std` drone alerter for firmware.
///
/// Feed it consecutive frames; it EMA-smooths the per-frame [`drone_confidence`]
/// and raises a latching boolean alert once the smoothed confidence holds at or
/// above `threshold` for `hold` consecutive frames. A single below-threshold
/// frame clears the alert and resets the counter. This mirrors the EMA + hold
/// logic in the host-side live listener, but with no heap and no `std`.
#[derive(Debug, Clone)]
pub struct EdgeDetector {
    sample_rate: u32,
    /// EMA smoothing factor in `(0, 1]`; smaller = smoother / slower.
    alpha: f32,
    /// Smoothed confidence above which a frame counts toward the hold.
    threshold: f32,
    /// Consecutive over-threshold frames required to latch the alert.
    hold: u32,
    ema: f32,
    primed: bool,
    over_count: u32,
    alerting: bool,
}

impl EdgeDetector {
    /// Create a detector.
    ///
    /// * `sample_rate` - sample rate of the incoming frames, in Hz.
    /// * `alpha` - EMA factor in `(0, 1]` (clamped); the live default is `0.4`.
    /// * `threshold` - alert threshold on the smoothed confidence in `[0, 1]`.
    /// * `hold` - consecutive over-threshold frames required to latch (≥ 1).
    pub fn new(sample_rate: u32, alpha: f32, threshold: f32, hold: u32) -> Self {
        Self {
            sample_rate,
            alpha: if alpha > 0.0 { clamp01(alpha) } else { 1.0 },
            threshold,
            hold: hold.max(1),
            ema: 0.0,
            primed: false,
            over_count: 0,
            alerting: false,
        }
    }

    /// A sensible firmware default: `alpha = 0.4`, `threshold = 0.5`, `hold = 3`,
    /// matching the live listener's defaults.
    pub fn with_defaults(sample_rate: u32) -> Self {
        Self::new(sample_rate, 0.4, 0.5, 3)
    }

    /// Feed one frame; returns the current latched alert state.
    ///
    /// Updates the EMA, the hold counter, and the latched alert, then returns
    /// `true` while the detector is alerting.
    pub fn push_frame(&mut self, frame: &[f32; FRAME_SIZE]) -> bool {
        let conf = drone_confidence(frame, self.sample_rate);
        self.ema = if self.primed {
            self.alpha * conf + (1.0 - self.alpha) * self.ema
        } else {
            self.primed = true;
            conf
        };

        if self.ema >= self.threshold {
            self.over_count = self.over_count.saturating_add(1);
            if self.over_count >= self.hold {
                self.alerting = true;
            }
        } else {
            self.over_count = 0;
            self.alerting = false;
        }
        self.alerting
    }

    /// The current EMA-smoothed confidence in `[0, 1]`.
    pub fn confidence(&self) -> f32 {
        self.ema
    }

    /// Whether the detector is currently latched into an alert.
    pub fn is_alerting(&self) -> bool {
        self.alerting
    }

    /// Reset all internal state to "just constructed".
    pub fn reset(&mut self) {
        self.ema = 0.0;
        self.primed = false;
        self.over_count = 0;
        self.alerting = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    const SR: u32 = 16_000;

    /// A harmonic frame: a fundamental plus a few harmonics inside the drone
    /// band, like a multirotor's motor/blade-pass signature.
    fn harmonic_frame(fundamental_hz: f32) -> [f32; FRAME_SIZE] {
        let mut f = [0.0_f32; FRAME_SIZE];
        for (i, s) in f.iter_mut().enumerate() {
            let t = i as f32 / SR as f32;
            let mut v = 0.0_f32;
            for h in 1..=5 {
                let amp = 1.0 / h as f32;
                v += amp * libm::sinf(2.0 * PI * fundamental_hz * h as f32 * t);
            }
            *s = 0.3 * v;
        }
        f
    }

    /// A noise-ish frame via a cheap deterministic LCG (no std rng).
    fn noise_frame(seed: u32) -> [f32; FRAME_SIZE] {
        let mut state = seed | 1;
        let mut f = [0.0_f32; FRAME_SIZE];
        for s in f.iter_mut() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let u = (state >> 8) as f32 / (1u32 << 24) as f32; // [0,1)
            *s = 2.0 * u - 1.0;
        }
        f
    }

    #[test]
    fn silence_scores_near_zero() {
        let frame = [0.0_f32; FRAME_SIZE];
        let c = drone_confidence(&frame, SR);
        assert!(c < 1e-3, "silence should score ~0, got {c}");
    }

    #[test]
    fn harmonic_scores_higher_than_noise() {
        let harmonic = drone_confidence(&harmonic_frame(150.0), SR);
        let noise = drone_confidence(&noise_frame(12345), SR);
        assert!(
            harmonic > noise,
            "harmonic ({harmonic}) should exceed noise ({noise})"
        );
        assert!(
            harmonic > 0.5,
            "harmonic should be a confident drone: {harmonic}"
        );
        assert!(noise < 0.5, "broadband noise should be low: {noise}");
    }

    #[test]
    fn confidence_is_bounded() {
        for f in [harmonic_frame(120.0), noise_frame(7), [0.0_f32; FRAME_SIZE]] {
            let c = drone_confidence(&f, SR);
            assert!((0.0..=1.0).contains(&c), "out of range: {c}");
        }
    }

    #[test]
    fn detector_latches_after_hold_then_clears() {
        let mut det = EdgeDetector::new(SR, 1.0, 0.5, 3); // alpha=1 => no smoothing lag
        let drone = harmonic_frame(150.0);
        let quiet = [0.0_f32; FRAME_SIZE];

        // First two strong frames: counting up, not yet latched.
        assert!(!det.push_frame(&drone));
        assert!(!det.push_frame(&drone));
        // Third strong frame meets hold => alert.
        assert!(det.push_frame(&drone));
        assert!(det.is_alerting());

        // A silent frame clears the alert and resets the counter.
        assert!(!det.push_frame(&quiet));
        assert!(!det.is_alerting());
    }

    #[test]
    fn reset_clears_state() {
        let mut det = EdgeDetector::with_defaults(SR);
        let drone = harmonic_frame(150.0);
        for _ in 0..5 {
            det.push_frame(&drone);
        }
        det.reset();
        assert_eq!(det.confidence(), 0.0);
        assert!(!det.is_alerting());
    }
}
