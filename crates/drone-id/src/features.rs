//! MFCC clip-feature front-end for drone-type recognition.
//!
//! Each frame's Hann magnitude spectrum (from [`drone_bench::util::spectra`]) is
//! pushed through a mel triangular filterbank spanning `0..sr/2`, log-compressed,
//! and DCT-II'd into `N_MFCC` cepstral coefficients. Per clip we pool the
//! frame-wise MFCCs into a mean and std vector and append the mean log-energy,
//! yielding a fixed-length [`N_FEAT`] clip feature vector.
//!
//! The math mirrors `drone-bench`'s `mfcc_lr` approach so the recognition head
//! and the detection head share the same notion of "MFCC". It is pure Rust and
//! fully deterministic.

use drone_bench::util::spectra;
use drone_dsp::{bin_to_hz, NUM_BINS};

/// Number of mel filterbank channels.
pub const N_MELS: usize = 26;
/// Number of MFCC coefficients kept (including c0).
pub const N_MFCC: usize = 13;
/// Clip feature length: mean + std of each MFCC, plus the mean log-energy.
pub const N_FEAT: usize = 2 * N_MFCC + 1;

/// Compute the clip-level MFCC feature vector: mean and std of each MFCC across
/// frames, plus the mean log filterbank energy. Returns a zero vector for empty
/// or silent input so callers degrade gracefully.
pub fn clip_features(samples: &[f32], sample_rate: u32) -> [f32; N_FEAT] {
    let mut out = [0.0f32; N_FEAT];
    let frames = spectra(samples);
    if frames.is_empty() {
        return out;
    }

    let fb = mel_filterbank(sample_rate);
    let dct = dct_matrix();

    let mut sum = [0.0f32; N_MFCC];
    let mut sum_sq = [0.0f32; N_MFCC];
    let mut sum_log_energy = 0.0f32;
    let mut count = 0.0f32;

    for sp in &frames {
        // Mel energies via the triangular filterbank (power spectrum).
        let mut log_mel = [0.0f32; N_MELS];
        for (m, filt) in fb.iter().enumerate() {
            let mut e = 0.0f32;
            for &(bin, w) in filt {
                e += w * sp[bin] * sp[bin];
            }
            log_mel[m] = (e + 1e-10).ln();
        }

        // DCT-II of the log-mel energies to get cepstral coefficients.
        for (k, sk) in sum.iter_mut().enumerate() {
            let mut c = 0.0f32;
            for (m, &lm) in log_mel.iter().enumerate() {
                c += dct[k][m] * lm;
            }
            *sk += c;
            sum_sq[k] += c * c;
        }

        let frame_log_energy: f32 = log_mel.iter().sum::<f32>() / N_MELS as f32;
        sum_log_energy += frame_log_energy;
        count += 1.0;
    }

    for k in 0..N_MFCC {
        let mean = sum[k] / count;
        let var = (sum_sq[k] / count - mean * mean).max(0.0);
        out[k] = mean;
        out[N_MFCC + k] = var.sqrt();
    }
    out[N_FEAT - 1] = sum_log_energy / count;
    out
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

/// Hz -> mel (HTK-style `2595*log10(1+f/700)`).
#[inline]
fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

/// Mel -> Hz (inverse of [`hz_to_mel`]).
#[inline]
fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10.0f32.powf(m / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_zero_vector() {
        let f = clip_features(&[], 16_000);
        assert_eq!(f, [0.0f32; N_FEAT]);
    }

    #[test]
    fn distinct_tones_give_distinct_features() {
        let sr = 16_000;
        let n = sr as usize;
        let tone = |hz: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (2.0 * core::f32::consts::PI * hz * i as f32 / sr as f32).sin())
                .collect()
        };
        let a = clip_features(&tone(200.0), sr);
        let b = clip_features(&tone(2000.0), sr);
        let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1.0, "features should differ across tones: {diff}");
        assert!(a.iter().all(|v| v.is_finite()));
    }
}
