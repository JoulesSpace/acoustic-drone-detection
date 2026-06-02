//! Clip-level features for distance estimation.
//!
//! The feature extractor consumes a clip already framed into per-frame
//! magnitude spectra (the host benchmark gets these from
//! [`drone_bench::util::spectra`], which is exactly the common front-end every
//! approach reuses). It pools across frames into one fixed-length vector.
//!
//! The features, in order, are:
//!
//! * `0` - **overall level**: mean log energy. The raw loudness cue; the
//!   literature warns this is confounded by source loudness.
//! * `1` - **spectral centroid** (Hz, scaled): energy-weighted mean frequency.
//! * `2` - **rolloff 85%** (Hz, scaled): frequency below which 85% of energy
//!   lies.
//! * `3` - **rolloff 95%** (Hz, scaled).
//! * `4` - **high/low band-energy log-ratio**: `log(E_high / E_low)`, the
//!   direct **air-absorption tilt** proxy. As range grows, highs fade faster
//!   than lows, so this drops - and crucially it does **not** depend on overall
//!   level.
//! * `5` - **spectral slope**: least-squares slope of log-magnitude vs
//!   frequency, another tilt measure.
//! * `6..6+N_MFCC` - the first [`N_MFCC`] MFCCs (mean-pooled across frames).
//!
//! Features `1..` are deliberately level-*invariant* so the model can learn
//! range from spectral shape rather than loudness. The [`FeatureSet`] selector
//! lets the benchmark run a **level-only vs tilt-included** ablation.

use alloc::vec::Vec;

use drone_dsp::{bin_to_hz, spectral_centroid, total_energy, Spectrum, NUM_BINS};
use libm::{logf, sqrtf};

/// Number of MFCC coefficients kept (excluding the redundant c0 level term,
/// which `level` already captures - we keep c1..c_NMFCC).
pub const N_MFCC: usize = 6;

/// Number of mel filterbank channels feeding the MFCC DCT.
const N_MELS: usize = 20;

/// Crossover frequency (Hz) splitting "low" from "high" band for the tilt ratio.
///
/// 1.5 kHz sits above the drone's dominant low harmonics but well inside the
/// band where air absorption (the `f^2` law) starts to bite, so the high band
/// carries the range-sensitive energy.
const TILT_CROSSOVER_HZ: f32 = 1_500.0;

/// Total feature-vector length: level, centroid, rolloff85, rolloff95,
/// tilt-ratio, slope, then `N_MFCC` MFCCs.
pub const N_FEATURES: usize = 6 + N_MFCC;

/// Indices (into the full vector) that constitute the **level-only** baseline.
///
/// Just the raw loudness term - the control arm of the ablation.
pub const LEVEL_ONLY_IDX: &[usize] = &[0];

/// Which subset of features a model should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureSet {
    /// Only the overall level (raw loudness). The ablation control.
    LevelOnly,
    /// Level plus the air-absorption tilt ratio and slope (no MFCCs / rolloff).
    LevelPlusTilt,
    /// All features.
    Full,
}

impl FeatureSet {
    /// The feature indices this set selects, in order.
    pub fn indices(self) -> Vec<usize> {
        match self {
            FeatureSet::LevelOnly => LEVEL_ONLY_IDX.to_vec(),
            // level(0), tilt-ratio(4), slope(5)
            FeatureSet::LevelPlusTilt => alloc::vec![0, 4, 5],
            FeatureSet::Full => (0..N_FEATURES).collect(),
        }
    }

    /// Human-readable name (for tables / JSON).
    pub fn name(self) -> &'static str {
        match self {
            FeatureSet::LevelOnly => "level-only",
            FeatureSet::LevelPlusTilt => "level+tilt",
            FeatureSet::Full => "full",
        }
    }
}

/// A computed clip feature vector plus its selectable views.
#[derive(Debug, Clone)]
pub struct ClipFeatures {
    /// The full fixed-length feature vector (length [`N_FEATURES`]).
    pub raw: [f32; N_FEATURES],
}

impl ClipFeatures {
    /// Gather the features selected by `set` into a fresh vector.
    pub fn select(&self, set: FeatureSet) -> Vec<f32> {
        set.indices().iter().map(|&i| self.raw[i]).collect()
    }
}

/// Compute clip-level features from per-frame magnitude spectra.
///
/// Returns an all-zero feature vector for empty input so callers degrade
/// gracefully. `sample_rate` is needed to map bins to Hz.
pub fn clip_features(frames: &[Spectrum], sample_rate: u32) -> ClipFeatures {
    let mut raw = [0.0_f32; N_FEATURES];
    if frames.is_empty() {
        return ClipFeatures { raw };
    }
    let nf = frames.len() as f32;

    // Mel filterbank + DCT basis are clip-independent; build once.
    let fb = mel_filterbank(sample_rate);
    let dct = dct_matrix();

    let crossover = TILT_CROSSOVER_HZ;
    let nyquist = sample_rate as f32 / 2.0;

    let mut sum_log_energy = 0.0_f32;
    let mut sum_centroid = 0.0_f32;
    let mut sum_roll85 = 0.0_f32;
    let mut sum_roll95 = 0.0_f32;
    let mut sum_tilt = 0.0_f32;
    let mut sum_slope = 0.0_f32;
    let mut sum_mfcc = [0.0_f32; N_MFCC];

    for sp in frames {
        let e = total_energy(sp);
        sum_log_energy += logf(e + 1e-10);
        sum_centroid += spectral_centroid(sp, sample_rate) / nyquist;
        sum_roll85 += rolloff(sp, sample_rate, 0.85) / nyquist;
        sum_roll95 += rolloff(sp, sample_rate, 0.95) / nyquist;
        sum_tilt += tilt_log_ratio(sp, sample_rate, crossover);
        sum_slope += spectral_slope(sp, sample_rate);

        // Mel log energies -> DCT -> MFCC (keep c1..c_NMFCC).
        let mut log_mel = [0.0_f32; N_MELS];
        for (m, filt) in fb.iter().enumerate() {
            let mut me = 0.0_f32;
            for &(bin, w) in filt {
                me += w * sp[bin] * sp[bin];
            }
            log_mel[m] = logf(me + 1e-10);
        }
        for (k, acc) in sum_mfcc.iter_mut().enumerate() {
            // DCT row k+1 (skip c0; the level feature already carries it).
            let mut c = 0.0_f32;
            for (m, &lm) in log_mel.iter().enumerate() {
                c += dct[k + 1][m] * lm;
            }
            *acc += c;
        }
    }

    raw[0] = sum_log_energy / nf;
    raw[1] = sum_centroid / nf;
    raw[2] = sum_roll85 / nf;
    raw[3] = sum_roll95 / nf;
    raw[4] = sum_tilt / nf;
    raw[5] = sum_slope / nf;
    for (k, &acc) in sum_mfcc.iter().enumerate() {
        raw[6 + k] = acc / nf;
    }
    ClipFeatures { raw }
}

/// Spectral rolloff: the frequency below which `frac` of the total magnitude
/// (L1) energy lies. Returns 0 for a silent frame.
fn rolloff(sp: &Spectrum, sample_rate: u32, frac: f32) -> f32 {
    let total: f32 = sp.iter().sum();
    if total <= 0.0 {
        return 0.0;
    }
    let target = frac * total;
    let mut acc = 0.0_f32;
    for (i, &m) in sp.iter().enumerate() {
        acc += m;
        if acc >= target {
            return bin_to_hz(i, sample_rate);
        }
    }
    bin_to_hz(NUM_BINS - 1, sample_rate)
}

/// High-vs-low band-energy log-ratio `log(E_high / E_low)` - the air-absorption
/// tilt proxy. Level-invariant: scaling the whole spectrum cancels in the ratio.
fn tilt_log_ratio(sp: &Spectrum, sample_rate: u32, crossover_hz: f32) -> f32 {
    let mut lo = 0.0_f32;
    let mut hi = 0.0_f32;
    for (i, &m) in sp.iter().enumerate() {
        let f = bin_to_hz(i, sample_rate);
        let p = m * m;
        if f < crossover_hz {
            lo += p;
        } else {
            hi += p;
        }
    }
    logf((hi + 1e-10) / (lo + 1e-10))
}

/// Least-squares slope of `log(magnitude)` vs frequency (Hz), over non-DC bins.
///
/// A steeper (more negative) slope means the highs are weaker relative to lows -
/// the same darkening air absorption produces with range. Level-invariant: an
/// overall scale shifts the intercept, not the slope.
fn spectral_slope(sp: &Spectrum, sample_rate: u32) -> f32 {
    // Linear regression of y = log(mag) on x = freq, skipping DC.
    let mut n = 0.0_f32;
    let mut sx = 0.0_f32;
    let mut sy = 0.0_f32;
    let mut sxx = 0.0_f32;
    let mut sxy = 0.0_f32;
    for (i, &m) in sp.iter().enumerate().skip(1) {
        let x = bin_to_hz(i, sample_rate);
        let y = logf(m + 1e-10);
        n += 1.0;
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-6 {
        return 0.0;
    }
    // Scale by 1000 so the slope (per-Hz, tiny) lands in a sane numeric range.
    1000.0 * (n * sxy - sx * sy) / denom
}

/// Build a mel filterbank as a list of `(bin, weight)` pairs per channel.
fn mel_filterbank(sample_rate: u32) -> Vec<Vec<(usize, f32)>> {
    let f_max = sample_rate as f32 / 2.0;
    let mel_max = hz_to_mel(f_max);
    let n_points = N_MELS + 2;
    let mut centers_hz = [0.0_f32; N_MELS + 2];
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

/// Precompute the DCT-II basis matrix (`(N_MFCC + 1)` x `N_MELS`); row 0 is c0.
fn dct_matrix() -> Vec<[f32; N_MELS]> {
    let rows = N_MFCC + 1;
    let scale = sqrtf(2.0 / N_MELS as f32);
    let mut dct = Vec::with_capacity(rows);
    for k in 0..rows {
        let mut row = [0.0_f32; N_MELS];
        for (m, val) in row.iter_mut().enumerate() {
            let ang = core::f32::consts::PI / N_MELS as f32 * (m as f32 + 0.5) * k as f32;
            *val = scale * libm::cosf(ang);
        }
        dct.push(row);
    }
    dct
}

/// Hz -> mel (HTK-style 2595*log10(1+f/700)).
#[inline]
fn hz_to_mel(f: f32) -> f32 {
    2595.0 * libm::log10f(1.0 + f / 700.0)
}

/// mel -> Hz (inverse of [`hz_to_mel`]).
#[inline]
fn mel_to_hz(m: f32) -> f32 {
    700.0 * (libm::powf(10.0, m / 2595.0) - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{simulate_clip, RangeSimConfig, SourceConfig};
    use drone_dsp::{hann_in_place, magnitude_spectrum, Frame, FRAME_SIZE};

    /// Minimal local copy of the framing front-end (the no_std core can't reach
    /// `drone_bench::util::spectra`, which is a std crate).
    fn spectra(samples: &[f32]) -> Vec<Spectrum> {
        let mut out = Vec::new();
        if samples.len() < FRAME_SIZE {
            return out;
        }
        let hop = FRAME_SIZE / 2;
        let mut start = 0;
        while start + FRAME_SIZE <= samples.len() {
            let mut frame: Frame = [0.0; FRAME_SIZE];
            frame.copy_from_slice(&samples[start..start + FRAME_SIZE]);
            hann_in_place(&mut frame);
            out.push(magnitude_spectrum(&mut frame));
            start += hop;
        }
        out
    }

    fn feats_at(range_m: f32) -> ClipFeatures {
        feats_with_noise(range_m, 0.01)
    }

    fn feats_with_noise(range_m: f32, noise_std: f32) -> ClipFeatures {
        let src = SourceConfig::default();
        let cfg = RangeSimConfig {
            range_m,
            num_samples: 16_000,
            noise_std,
            ..Default::default()
        };
        let clip = simulate_clip(&src, &cfg);
        clip_features(&spectra(&clip), 16_000)
    }

    #[test]
    fn empty_is_zero() {
        let f = clip_features(&[], 16_000);
        assert!(f.raw.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn tilt_ratio_drops_with_range() {
        // In the (near) noiseless limit the air-absorption tilt feature
        // (index 4) falls monotonically with range: highs fade faster than
        // lows, darkening the spectrum. This is the core physical range cue.
        // (With a fixed noise floor the far-field high band is eventually
        // dominated by broadband noise, which is the honest noise-limited
        // regime the benchmark quantifies - hence we test the clean cue here.)
        let near = feats_with_noise(15.0, 1e-5).raw[4];
        let far = feats_with_noise(150.0, 1e-5).raw[4];
        assert!(near > far, "tilt near {near} should exceed far {far}");
    }

    #[test]
    fn level_drops_with_range() {
        let near = feats_at(15.0).raw[0];
        let far = feats_at(150.0).raw[0];
        assert!(near > far, "level near {near} should exceed far {far}");
    }

    #[test]
    fn select_level_only_is_single_value() {
        let f = feats_at(50.0);
        assert_eq!(f.select(FeatureSet::LevelOnly).len(), 1);
        assert_eq!(f.select(FeatureSet::Full).len(), N_FEATURES);
    }

    #[test]
    fn tilt_is_level_invariant() {
        // Scaling the spectrum by 2x must not change the tilt log-ratio.
        let mut sp = [0.0_f32; NUM_BINS];
        sp[10] = 1.0;
        sp[200] = 0.5;
        let t1 = tilt_log_ratio(&sp, 16_000, TILT_CROSSOVER_HZ);
        for v in sp.iter_mut() {
            *v *= 2.0;
        }
        let t2 = tilt_log_ratio(&sp, 16_000, TILT_CROSSOVER_HZ);
        assert!((t1 - t2).abs() < 1e-4);
    }
}
