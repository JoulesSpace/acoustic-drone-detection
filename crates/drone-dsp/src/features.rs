//! Spectral features derived from a magnitude spectrum.
//!
//! These are the cheap, interpretable quantities a drone detector reasons over:
//! where the energy sits in frequency, how concentrated it is, and how much of
//! it falls inside a band of interest.

use crate::{Spectrum, FRAME_SIZE, NUM_BINS};

/// Centre frequency of magnitude bin `bin`, in Hz.
#[inline]
pub fn bin_to_hz(bin: usize, sample_rate: u32) -> f32 {
    bin as f32 * sample_rate as f32 / FRAME_SIZE as f32
}

/// Nearest magnitude bin to frequency `hz`, clamped to the valid range.
#[inline]
pub fn hz_to_bin(hz: f32, sample_rate: u32) -> usize {
    let raw = hz * FRAME_SIZE as f32 / sample_rate as f32;
    let rounded = libm::roundf(raw);
    if rounded <= 0.0 {
        0
    } else if rounded as usize >= NUM_BINS {
        NUM_BINS - 1
    } else {
        rounded as usize
    }
}

/// Index of the strongest bin, ignoring DC (bin 0).
pub fn dominant_bin(spectrum: &Spectrum) -> usize {
    let mut peak = 1usize;
    for i in 2..NUM_BINS {
        if spectrum[i] > spectrum[peak] {
            peak = i;
        }
    }
    peak
}

/// Sum of squared magnitudes over the whole spectrum (a power proxy).
pub fn total_energy(spectrum: &Spectrum) -> f32 {
    let mut acc = 0.0_f32;
    for &m in spectrum.iter() {
        acc += m * m;
    }
    acc
}

/// Sum of squared magnitudes within `[lo_hz, hi_hz]`.
pub fn band_energy(spectrum: &Spectrum, lo_hz: f32, hi_hz: f32, sample_rate: u32) -> f32 {
    let lo = hz_to_bin(lo_hz, sample_rate);
    let hi = hz_to_bin(hi_hz, sample_rate).max(lo);
    let mut acc = 0.0_f32;
    for &m in spectrum[lo..=hi].iter() {
        acc += m * m;
    }
    acc
}

/// Spectral centroid in Hz - the energy-weighted mean frequency.
///
/// A higher centroid means brighter/higher-pitched content. Returns 0.0 for a
/// silent frame.
pub fn spectral_centroid(spectrum: &Spectrum, sample_rate: u32) -> f32 {
    let mut weighted = 0.0_f32;
    let mut total = 0.0_f32;
    for (i, &m) in spectrum.iter().enumerate() {
        weighted += bin_to_hz(i, sample_rate) * m;
        total += m;
    }
    if total > 0.0 {
        weighted / total
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_hz_roundtrip() {
        let sr = 16_000;
        assert_eq!(hz_to_bin(bin_to_hz(100, sr), sr), 100);
    }

    #[test]
    fn band_energy_is_subset_of_total() {
        let mut spec = [0.0_f32; NUM_BINS];
        spec[50] = 2.0;
        spec[400] = 1.0;
        let total = total_energy(&spec);
        let band = band_energy(&spec, 0.0, bin_to_hz(60, 16_000), 16_000);
        assert!(band <= total);
        assert!(band > 0.0);
    }

    #[test]
    fn centroid_of_single_peak_is_that_bin() {
        let mut spec = [0.0_f32; NUM_BINS];
        spec[100] = 1.0;
        let c = spectral_centroid(&spec, 16_000);
        assert!((c - bin_to_hz(100, 16_000)).abs() < 1e-3);
    }
}
