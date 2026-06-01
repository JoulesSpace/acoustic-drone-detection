//! Cepstral pitch cue.
//!
//! The real cepstrum is the inverse transform of the log-magnitude spectrum.
//! A harmonic stack with spacing `f0` produces a "ripple" in the log spectrum
//! whose period maps to a peak at quefrency `1/f0` seconds. Finding that peak
//! gives a robust f0 estimate that is largely immune to a missing or weak
//! fundamental (a classic failure mode of raw spectral-peak picking).
//!
//! We compute the cepstrum only over the quefrency range that corresponds to
//! the f0 band of interest, evaluating the inverse cosine transform directly.
//! That range is small (a few tens of quefrency bins), so this is cheap and
//! avoids pulling in a second FFT size.

use core::f32::consts::PI;
use drone_dsp::{Spectrum, NUM_BINS};

/// Estimate f0 from one frame's magnitude spectrum via the real cepstrum.
///
/// Returns `(f0_hz, confidence)` where confidence is the cepstral peak height
/// normalized by the local cepstral energy, clamped to `[0, 1]`. Returns `None`
/// if the spectrum is empty or no peak is found in the band.
pub fn cepstral_f0(spectrum: &Spectrum, sr: u32, f_lo: f32, f_hi: f32) -> Option<(f32, f32)> {
    let sr_f = sr as f32;

    // Log-magnitude spectrum (the cepstrum's input). A small floor avoids
    // -inf on empty bins and tames the dynamic range.
    let log_mag: Vec<f32> = spectrum.iter().map(|&m| (m + 1e-6).ln()).collect();

    // Quefrency (in samples) corresponds to a period; f0 = sr / quefrency.
    // The frame that produced `spectrum` had FRAME_SIZE samples, so quefrency
    // here is in units of those samples.
    let q_min = (sr_f / f_hi).floor() as usize; // small quefrency -> high freq
    let q_max = (sr_f / f_lo).ceil() as usize;
    if q_min < 1 || q_max <= q_min {
        return None;
    }

    // Inverse cosine transform of the real log-spectrum, evaluated per quefrency.
    let mut best_q = q_min;
    let mut best_val = f32::MIN;
    let mut ceps = vec![0.0_f32; q_max + 1];
    let scale = PI / NUM_BINS as f32;
    for (q, slot) in ceps.iter_mut().enumerate().take(q_max + 1).skip(q_min) {
        let mut acc = 0.0_f32;
        let w = scale * q as f32;
        for (k, &lm) in log_mag.iter().enumerate() {
            acc += lm * (w * k as f32).cos();
        }
        // Magnitude of the cepstral coefficient; we care about the ripple
        // strength, not its sign.
        *slot = acc.abs();
        if *slot > best_val {
            best_val = *slot;
            best_q = q;
        }
    }

    if best_val <= 0.0 {
        return None;
    }

    let q = parabolic_peak(&ceps, best_q);
    let f0 = sr_f / q;
    if !(f_lo..=f_hi).contains(&f0) {
        return None;
    }

    // Confidence: peak height relative to the mean of the searched range.
    let mean: f32 = ceps[q_min..=q_max].iter().sum::<f32>() / (q_max - q_min + 1) as f32;
    let conf = if mean > 1e-9 {
        ((best_val / mean - 1.0) / 4.0).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Some((f0, conf))
}

/// Sub-sample parabolic peak refinement around integer index `idx`.
fn parabolic_peak(data: &[f32], idx: usize) -> f32 {
    if idx == 0 || idx + 1 >= data.len() {
        return idx as f32;
    }
    let a = data[idx - 1];
    let b = data[idx];
    let c = data[idx + 1];
    let denom = a - 2.0 * b + c;
    if denom.abs() < 1e-12 {
        return idx as f32;
    }
    let delta = 0.5 * (a - c) / denom;
    idx as f32 + delta.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use drone_dsp::{hann_in_place, magnitude_spectrum, Frame, FRAME_SIZE};
    use std::f32::consts::PI as PI64;

    fn harmonic_spectrum(f0: f32, sr: u32) -> Spectrum {
        let mut frame: Frame = [0.0; FRAME_SIZE];
        for (i, s) in frame.iter_mut().enumerate() {
            let t = i as f32 / sr as f32;
            let mut v = 0.0;
            for h in 1..=6 {
                v += (1.0 / h as f32) * (2.0 * PI64 * f0 * h as f32 * t).sin();
            }
            *s = v;
        }
        hann_in_place(&mut frame);
        magnitude_spectrum(&mut frame)
    }

    #[test]
    fn recovers_harmonic_spacing() {
        let sr = 16_000;
        let spec = harmonic_spectrum(150.0, sr);
        let (f0, _c) = cepstral_f0(&spec, sr, 50.0, 400.0).unwrap();
        assert!((f0 - 150.0).abs() < 6.0, "f0 was {f0}");
    }
}
