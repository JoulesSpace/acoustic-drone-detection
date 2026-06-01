//! Harmonic Product Spectrum (HPS) pitch cue.
//!
//! HPS multiplies the magnitude spectrum by downsampled copies of itself. A
//! true fundamental lines up with its own harmonics under downsampling, so its
//! bin is reinforced while spurious peaks are suppressed. This makes HPS strong
//! at picking the fundamental even when higher harmonics dominate the raw
//! spectrum — but it is prone to picking f0/2 or 2·f0 (octave errors), which is
//! exactly why we cross-check it against the cepstrum and autocorrelation.

use drone_dsp::{hz_to_bin, Spectrum, FRAME_SIZE, NUM_BINS};

/// Number of harmonic products to fold in. 5 is a good balance: enough to
/// disambiguate the fundamental, not so many that we run off the top of the
/// usable band for high-f0 candidates.
const N_HARMONICS: usize = 5;

/// Estimate f0 from one frame's magnitude spectrum via HPS.
///
/// Returns `(f0_hz, confidence)`. Confidence is the HPS peak normalized by the
/// mean HPS value over the search band, mapped into `[0, 1]`. Returns `None`
/// if no peak is found in `[f_lo, f_hi]`.
pub fn hps_f0(spectrum: &Spectrum, sr: u32, f_lo: f32, f_hi: f32) -> Option<(f32, f32)> {
    let lo = hz_to_bin(f_lo, sr).max(1);
    let hi = hz_to_bin(f_hi, sr).min(NUM_BINS - 1);
    if hi <= lo {
        return None;
    }

    // Work in log-magnitude (sum of logs == log of product); this is numerically
    // gentler than multiplying small magnitudes and equally peak-preserving.
    let mut hps = vec![0.0_f32; hi + 1];
    for (bin, slot) in hps.iter_mut().enumerate().take(hi + 1).skip(lo) {
        let mut acc = 0.0_f32;
        let mut ok = true;
        for h in 1..=N_HARMONICS {
            let idx = bin * h;
            if idx >= NUM_BINS {
                ok = false;
                break;
            }
            acc += (spectrum[idx] + 1e-9).ln();
        }
        *slot = if ok { acc } else { f32::MIN };
    }

    let mut best_bin = lo;
    let mut best_val = f32::MIN;
    for (bin, &v) in hps.iter().enumerate().take(hi + 1).skip(lo) {
        if v > best_val {
            best_val = v;
            best_bin = bin;
        }
    }
    if best_val == f32::MIN {
        return None;
    }

    // Convert the (sub-bin) peak position to Hz directly, since `bin_to_hz`
    // only accepts integer bins.
    let frac_bin = refine_bin(&hps, best_bin);
    let f0 = frac_bin * sr as f32 / FRAME_SIZE as f32;
    if !(f_lo..=f_hi).contains(&f0) {
        return None;
    }

    // Confidence from peak prominence over the valid (finite) band.
    let finite: Vec<f32> = hps[lo..=hi]
        .iter()
        .copied()
        .filter(|v| *v > f32::MIN)
        .collect();
    let conf = if finite.len() > 1 {
        let mean = finite.iter().sum::<f32>() / finite.len() as f32;
        let spread = finite
            .iter()
            .map(|v| (v - mean).abs())
            .fold(0.0_f32, f32::max)
            .max(1e-6);
        ((best_val - mean) / spread).clamp(0.0, 1.0)
    } else {
        0.0
    };
    Some((f0, conf))
}

/// Sub-bin parabolic peak refinement, skipping sentinel (`f32::MIN`) neighbours.
fn refine_bin(hps: &[f32], bin: usize) -> f32 {
    if bin == 0 || bin + 1 >= hps.len() {
        return bin as f32;
    }
    let a = hps[bin - 1];
    let b = hps[bin];
    let c = hps[bin + 1];
    if a == f32::MIN || c == f32::MIN {
        return bin as f32;
    }
    let denom = a - 2.0 * b + c;
    if denom.abs() < 1e-12 {
        return bin as f32;
    }
    let delta = 0.5 * (a - c) / denom;
    bin as f32 + delta.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use drone_dsp::{hann_in_place, magnitude_spectrum, Frame, FRAME_SIZE};
    use std::f32::consts::PI;

    fn harmonic_spectrum(f0: f32, sr: u32, with_fundamental: bool) -> Spectrum {
        let mut frame: Frame = [0.0; FRAME_SIZE];
        for (i, s) in frame.iter_mut().enumerate() {
            let t = i as f32 / sr as f32;
            let mut v = 0.0;
            let start = if with_fundamental { 1 } else { 2 };
            for h in start..=7 {
                v += (1.0 / h as f32) * (2.0 * PI * f0 * h as f32 * t).sin();
            }
            *s = v;
        }
        hann_in_place(&mut frame);
        magnitude_spectrum(&mut frame)
    }

    #[test]
    fn recovers_fundamental() {
        let sr = 16_000;
        let spec = harmonic_spectrum(120.0, sr, true);
        let (f0, _c) = hps_f0(&spec, sr, 50.0, 400.0).unwrap();
        assert!((f0 - 120.0).abs() < 16.0, "f0 was {f0}");
    }

    #[test]
    fn recovers_fundamental_when_missing() {
        // HPS should still recover f0 even with no energy at f0 itself.
        let sr = 16_000;
        let spec = harmonic_spectrum(160.0, sr, false);
        let (f0, _c) = hps_f0(&spec, sr, 50.0, 400.0).unwrap();
        assert!((f0 - 160.0).abs() < 16.0, "f0 was {f0}");
    }
}
