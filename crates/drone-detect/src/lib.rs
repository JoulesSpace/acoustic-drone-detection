//! Heuristic, per-frame acoustic drone detector.
//!
//! v0.1.0 is intentionally a transparent baseline rather than a learned model:
//! it asks "how much of the frame's energy sits in the drone band, and is the
//! dominant tone inside it?". This is cheap enough to run on a microcontroller
//! and gives us a measurable yardstick to beat with anything fancier later.
//!
//! Like [`drone_dsp`], this is `no_std`-friendly: it works on one
//! [`drone_dsp::Spectrum`] at a time and pulls in no allocation.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]

use drone_dsp::{band_energy, bin_to_hz, dominant_bin, total_energy, Spectrum};

/// Outcome of analysing a single frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// Whether this frame is flagged as drone-like.
    pub is_drone: bool,
    /// `band_energy / total_energy` in `[0, 1]`.
    pub band_ratio: f32,
    /// Frequency (Hz) of the strongest bin.
    pub dominant_hz: f32,
    /// Confidence in `[0, 1]`, derived from how far the band ratio clears the
    /// threshold.
    pub confidence: f32,
}

/// Tunable detector configuration.
#[derive(Debug, Clone, Copy)]
pub struct Detector {
    /// Sample rate of the audio the spectra came from.
    pub sample_rate: u32,
    /// Lower edge of the drone band, Hz.
    pub band_lo_hz: f32,
    /// Upper edge of the drone band, Hz.
    pub band_hi_hz: f32,
    /// Minimum band-energy ratio to flag a frame as drone-like.
    pub ratio_threshold: f32,
}

impl Detector {
    /// A reasonable default for small multirotors: most blade-pass and motor
    /// harmonic energy lands between ~100 Hz and a few kHz.
    pub fn new(sample_rate: u32) -> Self {
        Self {
            sample_rate,
            band_lo_hz: 100.0,
            band_hi_hz: 4000.0,
            ratio_threshold: 0.5,
        }
    }

    /// Analyse one magnitude spectrum.
    pub fn analyze(&self, spectrum: &Spectrum) -> Detection {
        let total = total_energy(spectrum);
        let band = band_energy(spectrum, self.band_lo_hz, self.band_hi_hz, self.sample_rate);
        // Guard against the silent-frame divide-by-zero.
        let band_ratio = if total > f32::EPSILON {
            band / total
        } else {
            0.0
        };

        let dom_bin = dominant_bin(spectrum);
        let dominant_hz = bin_to_hz(dom_bin, self.sample_rate);
        let dom_in_band = dominant_hz >= self.band_lo_hz && dominant_hz <= self.band_hi_hz;

        let is_drone = band_ratio >= self.ratio_threshold && dom_in_band;

        // Linear confidence: 0 at the threshold, 1 when the band holds all energy.
        let span = (1.0 - self.ratio_threshold).max(f32::EPSILON);
        let confidence = ((band_ratio - self.ratio_threshold) / span).clamp(0.0, 1.0);

        Detection {
            is_drone,
            band_ratio,
            dominant_hz,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use drone_dsp::{hz_to_bin, NUM_BINS};

    #[test]
    fn energy_in_band_flags_drone() {
        let sr = 16_000;
        let det = Detector::new(sr);
        let mut spec = [0.0_f32; NUM_BINS];
        // Put a strong tone at 800 Hz, well inside the band.
        spec[hz_to_bin(800.0, sr)] = 5.0;
        let d = det.analyze(&spec);
        assert!(d.is_drone);
        assert!(d.band_ratio > 0.9);
        assert!((d.dominant_hz - 800.0).abs() < 20.0);
    }

    #[test]
    fn out_of_band_tone_is_not_a_drone() {
        let sr = 16_000;
        let det = Detector::new(sr);
        let mut spec = [0.0_f32; NUM_BINS];
        // 60 Hz hum, below the band.
        spec[hz_to_bin(60.0, sr)] = 5.0;
        let d = det.analyze(&spec);
        assert!(!d.is_drone);
    }

    #[test]
    fn silence_does_not_panic_or_flag() {
        let det = Detector::new(16_000);
        let spec = [0.0_f32; NUM_BINS];
        let d = det.analyze(&spec);
        assert!(!d.is_drone);
        assert_eq!(d.band_ratio, 0.0);
    }
}
