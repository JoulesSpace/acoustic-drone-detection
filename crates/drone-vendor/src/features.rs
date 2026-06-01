//! Clip feature front-end for multi-vendor drone recognition.
//!
//! The base front-end mirrors the rest of the repo: each frame's Hann magnitude
//! spectrum (from [`drone_bench::util::spectra`]) is pushed through a mel
//! triangular filterbank spanning `0..sr/2`, log-compressed, and DCT-II'd into
//! [`N_MFCC`] cepstral coefficients. Per clip the frame-wise MFCCs are pooled
//! into mean and std vectors.
//!
//! Telling DJI from Autel from a toy Syma is a finer discrimination than "drone
//! vs not", so on top of the MFCCs we optionally append a handful of cheap,
//! interpretable **spectral / harmonic** descriptors that capture the physical
//! signature of a vendor's airframe: where the energy sits (centroid, rolloff),
//! how it splits across motor-whine bands (low/mid/high band-energy ratios),
//! how strongly the spectrum is harmonically combed (a blade-pass / rotor-count
//! proxy), and how fast the envelope is amplitude-modulated (the rotor AM rate).
//!
//! Everything is pure Rust and fully deterministic (no RNG, no ML crate).

use drone_bench::util::spectra;
use drone_dsp::{bin_to_hz, spectral_centroid, NUM_BINS};

/// Number of mel filterbank channels.
pub const N_MELS: usize = 26;
/// Number of MFCC coefficients kept (including c0).
pub const N_MFCC: usize = 13;

/// Length of the MFCC-only feature block: mean + std of each MFCC, plus the
/// mean log filterbank energy.
pub const N_MFCC_FEAT: usize = 2 * N_MFCC + 1;

/// Number of extra spectral/harmonic descriptors appended when enabled.
///
/// `[centroid_hz, rolloff85_hz, low_ratio, mid_ratio, high_ratio,
///   harmonic_strength, am_rate_hz, flatness]`.
pub const N_SPECTRAL: usize = 8;

/// Maximum clip feature length (MFCC block + spectral block). The actual length
/// in use is [`feature_len`]; entries past it are left zero so the model can
/// allocate a fixed-width buffer regardless of the toggle.
pub const N_FEAT_MAX: usize = N_MFCC_FEAT + N_SPECTRAL;

/// Feature length actually populated, given the spectral toggle.
#[inline]
pub fn feature_len(use_spectral: bool) -> usize {
    if use_spectral {
        N_FEAT_MAX
    } else {
        N_MFCC_FEAT
    }
}

/// Compute the clip-level feature vector.
///
/// The first [`N_MFCC_FEAT`] entries are the pooled MFCC mean/std + mean
/// log-energy. When `use_spectral` is true the next [`N_SPECTRAL`] entries hold
/// the spectral/harmonic descriptors; otherwise they stay zero. Returns an
/// all-zero vector for empty or silent input so callers degrade gracefully.
pub fn clip_features(samples: &[f32], sample_rate: u32, use_spectral: bool) -> [f32; N_FEAT_MAX] {
    let mut out = [0.0f32; N_FEAT_MAX];
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
    out[N_MFCC_FEAT - 1] = sum_log_energy / count;

    if use_spectral {
        let spec = spectral_descriptors(&frames, samples, sample_rate);
        out[N_MFCC_FEAT..N_MFCC_FEAT + N_SPECTRAL].copy_from_slice(&spec);
    }

    out
}

/// Compute the [`N_SPECTRAL`] spectral/harmonic descriptors from the frame
/// spectra and the raw samples.
fn spectral_descriptors(
    frames: &[drone_dsp::Spectrum],
    samples: &[f32],
    sample_rate: u32,
) -> [f32; N_SPECTRAL] {
    // Mean magnitude spectrum across frames.
    let mut mean_spec = [0.0f32; NUM_BINS];
    for sp in frames {
        for (a, &v) in mean_spec.iter_mut().zip(sp.iter()) {
            *a += v;
        }
    }
    let nf = frames.len() as f32;
    for a in mean_spec.iter_mut() {
        *a /= nf;
    }

    let nyquist = sample_rate as f32 / 2.0;
    let centroid = spectral_centroid(&mean_spec, sample_rate);

    // Spectral rolloff: frequency below which 85% of the energy lies.
    let total: f32 = mean_spec.iter().map(|m| m * m).sum();
    let mut acc = 0.0f32;
    let mut rolloff_hz = 0.0f32;
    if total > 0.0 {
        for (i, &m) in mean_spec.iter().enumerate() {
            acc += m * m;
            if acc >= 0.85 * total {
                rolloff_hz = bin_to_hz(i, sample_rate);
                break;
            }
        }
    }

    // Band-energy ratios. Motor whine, blade-pass harmonics and broadband hiss
    // sit in different parts of the band depending on the airframe; the split
    // at sr/8 and sr/3 of Nyquist gives three vendor-discriminative buckets.
    let lo = band_power(&mean_spec, 0.0, nyquist / 8.0, sample_rate);
    let mid = band_power(&mean_spec, nyquist / 8.0, nyquist / 3.0, sample_rate);
    let hi = band_power(&mean_spec, nyquist / 3.0, nyquist, sample_rate);
    let band_sum = lo + mid + hi + 1e-12;

    // Harmonic comb strength: how much energy concentrates on integer multiples
    // of the dominant low-frequency peak (a blade-pass / rotor-count proxy).
    let harmonic = harmonic_strength(&mean_spec, sample_rate);

    // Amplitude-modulation rate of the envelope (the rotor / rotor-count beat),
    // estimated from the autocorrelation of the rectified, smoothed signal.
    let am_rate = am_rate_hz(samples, sample_rate);

    // Spectral flatness (geometric mean / arithmetic mean of power), a
    // tonal-vs-noisy measure: toy quads are buzzier, big rigs more broadband.
    let flatness = spectral_flatness(&mean_spec);

    [
        centroid,
        rolloff_hz,
        lo / band_sum,
        mid / band_sum,
        hi / band_sum,
        harmonic,
        am_rate,
        flatness,
    ]
}

/// Power (sum of squared magnitude) in `[lo_hz, hi_hz]` of a magnitude spectrum.
fn band_power(spec: &[f32; NUM_BINS], lo_hz: f32, hi_hz: f32, sample_rate: u32) -> f32 {
    let mut acc = 0.0f32;
    for (i, &m) in spec.iter().enumerate() {
        let f = bin_to_hz(i, sample_rate);
        if f >= lo_hz && f < hi_hz {
            acc += m * m;
        }
    }
    acc
}

/// Fraction of total power that lands on integer multiples of the dominant
/// low-frequency peak (searched in 40..400 Hz, the blade-pass range). Higher
/// when the spectrum is strongly harmonically combed.
fn harmonic_strength(spec: &[f32; NUM_BINS], sample_rate: u32) -> f32 {
    // Dominant peak within the blade-pass band.
    let lo_bin = hz_to_bin_local(40.0, sample_rate).max(1);
    let hi_bin = hz_to_bin_local(400.0, sample_rate).min(NUM_BINS - 1);
    if hi_bin <= lo_bin {
        return 0.0;
    }
    let mut peak = lo_bin;
    for b in lo_bin..=hi_bin {
        if spec[b] > spec[peak] {
            peak = b;
        }
    }
    let f0_bin = peak.max(1);

    let total: f32 = spec.iter().map(|m| m * m).sum::<f32>() + 1e-12;
    let mut harm = 0.0f32;
    for h in 1..=8 {
        let center = f0_bin * h;
        if center >= NUM_BINS {
            break;
        }
        // Sum a +/-1 bin neighbourhood around each harmonic to be robust to the
        // bin grid not landing exactly on the harmonic.
        let lo = center.saturating_sub(1);
        let hi = (center + 1).min(NUM_BINS - 1);
        for &m in &spec[lo..=hi] {
            harm += m * m;
        }
    }
    harm / total
}

/// Estimate the dominant amplitude-modulation rate (Hz) of the signal envelope
/// via autocorrelation of the rectified, decimated waveform. This is the rotor
/// beat: it scales with rotor count and RPM and separates quads from hexes from
/// toys. Returns 0.0 when no clear period is found.
fn am_rate_hz(samples: &[f32], sample_rate: u32) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    // Decimate to ~1 kHz envelope rate to make the lag search cheap and to focus
    // on the slow AM (a few Hz to ~50 Hz), not the carrier.
    let decim = (sample_rate as usize / 1000).max(1);
    let env: Vec<f32> = samples
        .iter()
        .step_by(decim)
        .map(|&s| if s >= 0.0 { s } else { -s })
        .collect();
    let env_rate = sample_rate as f32 / decim as f32;
    if env.len() < 16 {
        return 0.0;
    }
    // Remove the DC of the envelope so autocorrelation reflects modulation.
    let mean: f32 = env.iter().sum::<f32>() / env.len() as f32;
    let centered: Vec<f32> = env.iter().map(|&v| v - mean).collect();

    // Search lags for AM in [2 Hz, 50 Hz].
    let min_lag = (env_rate / 50.0) as usize;
    let max_lag = ((env_rate / 2.0) as usize).min(centered.len() - 1);
    let min_lag = min_lag.max(1);
    if max_lag <= min_lag {
        return 0.0;
    }
    let energy: f32 = centered.iter().map(|v| v * v).sum::<f32>() + 1e-12;

    let mut best_lag = 0usize;
    let mut best_corr = 0.0f32;
    for lag in min_lag..=max_lag {
        let mut c = 0.0f32;
        for i in lag..centered.len() {
            c += centered[i] * centered[i - lag];
        }
        let norm = c / energy;
        if norm > best_corr {
            best_corr = norm;
            best_lag = lag;
        }
    }
    if best_lag == 0 || best_corr < 0.05 {
        0.0
    } else {
        env_rate / best_lag as f32
    }
}

/// Spectral flatness: geometric mean over arithmetic mean of the power
/// spectrum, in `[0, 1]`. Near 1 for white noise, near 0 for a pure tone.
fn spectral_flatness(spec: &[f32; NUM_BINS]) -> f32 {
    let mut log_sum = 0.0f32;
    let mut arith = 0.0f32;
    let n = NUM_BINS as f32;
    for &m in spec.iter() {
        let p = m * m + 1e-12;
        log_sum += p.ln();
        arith += p;
    }
    let geo = (log_sum / n).exp();
    let am = arith / n;
    if am > 0.0 {
        (geo / am).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// Local nearest-bin helper (mirrors `drone_dsp::hz_to_bin` but kept here so the
/// feature module owns its own clamping policy).
#[inline]
fn hz_to_bin_local(hz: f32, sample_rate: u32) -> usize {
    let raw = (hz * NUM_BINS as f32 * 2.0 / sample_rate as f32).round();
    if raw <= 0.0 {
        0
    } else if raw as usize >= NUM_BINS {
        NUM_BINS - 1
    } else {
        raw as usize
    }
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
    use core::f32::consts::PI;

    #[test]
    fn empty_input_is_zero_vector() {
        let f = clip_features(&[], 16_000, true);
        assert_eq!(f, [0.0f32; N_FEAT_MAX]);
    }

    #[test]
    fn spectral_block_is_zero_when_disabled() {
        let sr = 16_000;
        let n = sr as usize;
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 200.0 * i as f32 / sr as f32).sin())
            .collect();
        let f = clip_features(&tone, sr, false);
        for v in &f[N_MFCC_FEAT..N_FEAT_MAX] {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn distinct_tones_give_distinct_features() {
        let sr = 16_000;
        let n = sr as usize;
        let tone = |hz: f32| -> Vec<f32> {
            (0..n)
                .map(|i| (2.0 * PI * hz * i as f32 / sr as f32).sin())
                .collect()
        };
        let a = clip_features(&tone(200.0), sr, true);
        let b = clip_features(&tone(2000.0), sr, true);
        let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1.0, "features should differ across tones: {diff}");
        assert!(a.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn am_rate_recovers_known_modulation() {
        let sr = 16_000;
        let n = sr as usize;
        let am_hz = 12.0;
        // 200 Hz carrier amplitude-modulated at 12 Hz.
        let sig: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let am = 1.0 + 0.5 * (2.0 * PI * am_hz * t).sin();
                am * (2.0 * PI * 200.0 * t).sin()
            })
            .collect();
        let est = am_rate_hz(&sig, sr);
        assert!(
            (est - am_hz).abs() < 2.0,
            "AM rate estimate {est} should be near {am_hz}"
        );
    }

    #[test]
    fn flatness_tone_below_noise() {
        let sr = 16_000;
        let n = sr as usize;
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * PI * 500.0 * i as f32 / sr as f32).sin())
            .collect();
        let ft = clip_features(&tone, sr, true)[N_MFCC_FEAT + 7];
        // A clean tone should be far from white-noise flatness (~1).
        assert!(ft < 0.5, "tone flatness too high: {ft}");
    }
}
