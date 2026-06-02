//! Physics range simulator: render a drone-like source as heard at distance `r`.
//!
//! We have no distance-labeled multi-range recordings, so the benchmark drives
//! the estimator with a physically-motivated simulation. Starting from a
//! harmonic stack (a multirotor's blade-pass / motor tones), we apply three
//! range-dependent effects:
//!
//! 1. **Spherical spreading.** Free-field pressure falls as `1/r`, i.e. `-6 dB`
//!    per distance doubling. Implemented as an amplitude factor
//!    `r_ref / r` (so at `r = r_ref` the source is at unit level).
//!
//! 2. **Frequency-dependent air absorption.** Atmospheric absorption attenuates
//!    each frequency by `exp(-alpha(f) * r)` nepers, where the absorption
//!    coefficient grows roughly with the square of frequency. We use a simple,
//!    documented model
//!
//!    ```text
//!    alpha(f) = ALPHA_REF_NP_PER_M * (f / F_REF_HZ)^2          [nepers / metre]
//!    ```
//!
//!    with `ALPHA_REF_NP_PER_M` calibrated so that at `F_REF_HZ` the loss is a
//!    realistic ~0.1 dB/m (a few dB at 4 kHz over ~100 m, the right order of
//!    magnitude for mid-humidity air). This `f^2` law is the standard
//!    classical-absorption term; it deliberately omits the
//!    relaxation/humidity peaks of the full ISO 9613-1 model, which is fine for
//!    a controlled tilt cue (see the crate-level honesty notes).
//!
//!    **This is the range-specific cue:** the `f^2` weighting means high
//!    harmonics fade faster than low ones, tilting the spectrum darker with
//!    range in a way a pure level change cannot reproduce.
//!
//! 3. **Additive ambient noise at a fixed absolute level.** The microphone's
//!    ambient floor does not change with range, so as the (spread + absorbed)
//!    signal gets quieter the **SNR falls with distance** - exactly the regime
//!    where audio-only range estimation gets hard.
//!
//! Everything is deterministic given the `seed`; the noise generator is a seeded
//! xorshift with a Box-Muller Gaussian so the core stays `no_std`.

use alloc::vec;
use alloc::vec::Vec;

use libm::{cosf, expf, logf, powf, sinf, sqrtf};

/// Reference distance (metres) at which the spherical-spreading factor is unity.
///
/// Choosing `1 m` makes the spreading factor simply `1 / r` for `r >= 1 m`.
pub const R_REF_M: f32 = 1.0;

/// Reference frequency (Hz) for the air-absorption law.
pub const F_REF_HZ: f32 = 1_000.0;

/// Absorption coefficient at [`F_REF_HZ`], in **nepers per metre**.
///
/// `0.0115 Np/m` is `~0.1 dB/m` at 1 kHz (1 Np = 8.686 dB). Combined with the
/// `f^2` law this gives `~1.6 dB/m` at 4 kHz, so over 100 m a 4 kHz harmonic is
/// attenuated far more than a 200 Hz fundamental - a strong, range-monotone
/// tilt. The order of magnitude matches mid-humidity classical absorption.
pub const ALPHA_REF_NP_PER_M: f32 = 0.0115;

/// A synthetic drone-like harmonic source (the clean, at-reference signal).
#[derive(Debug, Clone)]
pub struct SourceConfig {
    /// Fundamental (blade-pass / motor) frequency in Hz.
    pub fundamental_hz: f32,
    /// Relative amplitude of each harmonic, starting at the fundamental.
    pub harmonics: Vec<f32>,
    /// Source loudness multiplier at the reference distance.
    ///
    /// This is the confounder the literature warns about: a louder drone looks
    /// closer. The benchmark can jitter it to show how level alone is fooled.
    pub source_gain: f32,
}

impl Default for SourceConfig {
    /// A plausible small-multirotor signature: a ~120 Hz fundamental with a long
    /// stack of harmonics that decays smoothly but still reaches several kHz, so
    /// there is real energy in the high band for air absorption to bite on
    /// (without it the tilt cue would have nothing to act on).
    fn default() -> Self {
        // 40 harmonics: 120 Hz .. 4800 Hz. Amplitude rolls off as 1/(h+1)^0.7,
        // a gentle decay so the high harmonics are weak but non-negligible.
        let harmonics: Vec<f32> = (0..40).map(|h| libm::powf(h as f32 + 1.0, -0.7)).collect();
        Self {
            fundamental_hz: 120.0,
            harmonics,
            source_gain: 1.0,
        }
    }
}

/// The air-absorption model: `alpha(f) = alpha_ref * (f / f_ref)^2` nepers/m.
#[derive(Debug, Clone, Copy)]
pub struct AbsorptionModel {
    /// Absorption coefficient at [`AbsorptionModel::f_ref_hz`], nepers/metre.
    pub alpha_ref_np_per_m: f32,
    /// Reference frequency, Hz.
    pub f_ref_hz: f32,
}

impl Default for AbsorptionModel {
    fn default() -> Self {
        Self {
            alpha_ref_np_per_m: ALPHA_REF_NP_PER_M,
            f_ref_hz: F_REF_HZ,
        }
    }
}

impl AbsorptionModel {
    /// Absorption coefficient at frequency `f_hz`, in nepers/metre.
    #[inline]
    pub fn alpha(&self, f_hz: f32) -> f32 {
        let ratio = f_hz / self.f_ref_hz;
        self.alpha_ref_np_per_m * ratio * ratio
    }

    /// Multiplicative amplitude factor for a tone of frequency `f_hz` after
    /// travelling `r_m` metres: `exp(-alpha(f) * r)`, in `[0, 1]`.
    #[inline]
    pub fn transmission(&self, f_hz: f32, r_m: f32) -> f32 {
        expf(-self.alpha(f_hz) * r_m)
    }
}

/// Configuration for one simulated capture at a given range.
#[derive(Debug, Clone)]
pub struct RangeSimConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of mono samples to produce.
    pub num_samples: usize,
    /// True source distance in metres (clamped to `>= R_REF_M` internally).
    pub range_m: f32,
    /// Ambient noise standard deviation, as an **absolute** level (independent
    /// of range). Because the signal shrinks with range, SNR falls with range.
    pub noise_std: f32,
    /// Air-absorption model.
    pub absorption: AbsorptionModel,
    /// RNG seed (noise is deterministic given this).
    pub seed: u32,
}

impl Default for RangeSimConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            num_samples: 16_000, // ~1 s clip
            range_m: 50.0,
            noise_std: 0.01,
            absorption: AbsorptionModel::default(),
            seed: 1,
        }
    }
}

/// Render one mono clip of `src` as heard at `cfg.range_m`.
///
/// Applies spherical spreading, per-harmonic air absorption and additive ambient
/// noise, then returns the noisy samples. The clean signal is *not* peak
/// normalized after attenuation - that is the whole point, the absolute level
/// (and hence the SNR against the fixed noise floor) must carry range
/// information.
pub fn simulate_clip(src: &SourceConfig, cfg: &RangeSimConfig) -> Vec<f32> {
    let fs = cfg.sample_rate as f32;
    let r = cfg.range_m.max(R_REF_M);
    let n = cfg.num_samples;

    // Spherical spreading: 1/r amplitude (-6 dB per doubling), unity at R_REF_M.
    let spread = R_REF_M / r;

    // Per-harmonic effective amplitude after spreading + absorption.
    let mut amps = Vec::with_capacity(src.harmonics.len());
    let mut freqs = Vec::with_capacity(src.harmonics.len());
    for (h, &amp) in src.harmonics.iter().enumerate() {
        let f = src.fundamental_hz * (h as f32 + 1.0);
        // Drop harmonics above Nyquist to avoid aliasing.
        if f >= fs * 0.5 {
            continue;
        }
        let absorbed = cfg.absorption.transmission(f, r);
        amps.push(src.source_gain * amp * spread * absorbed);
        freqs.push(f);
    }

    let mut out = vec![0.0_f32; n];
    let mut rng = Xorshift::new(cfg.seed.wrapping_mul(2_654_435_761).wrapping_add(1));
    let two_pi = 2.0 * core::f32::consts::PI;
    for (i, s) in out.iter_mut().enumerate() {
        let t = i as f32 / fs;
        let mut acc = 0.0_f32;
        for (a, f) in amps.iter().zip(freqs.iter()) {
            acc += a * sinf(two_pi * f * t);
        }
        *s = acc + cfg.noise_std * rng.next_gaussian();
    }
    out
}

/// Mean signal power (sum of squared harmonic amplitudes / 2) at this range, a
/// cheap way to compute the *true* SNR for diagnostics.
pub fn signal_power(src: &SourceConfig, cfg: &RangeSimConfig) -> f32 {
    let fs = cfg.sample_rate as f32;
    let r = cfg.range_m.max(R_REF_M);
    let spread = R_REF_M / r;
    let mut p = 0.0_f32;
    for (h, &amp) in src.harmonics.iter().enumerate() {
        let f = src.fundamental_hz * (h as f32 + 1.0);
        if f >= fs * 0.5 {
            continue;
        }
        let a = src.source_gain * amp * spread * cfg.absorption.transmission(f, r);
        p += 0.5 * a * a; // mean power of a sine of amplitude a
    }
    p
}

/// True SNR in dB at this range against the fixed ambient floor.
pub fn snr_db(src: &SourceConfig, cfg: &RangeSimConfig) -> f32 {
    let sig = signal_power(src, cfg);
    let noise = cfg.noise_std * cfg.noise_std;
    if sig <= 0.0 || noise <= 0.0 {
        return f32::NEG_INFINITY;
    }
    10.0 * logf(sig / noise) / core::f32::consts::LN_10 * 1.0 // 10*log10
}

/// `x^p` re-exported through `libm` for the rare host caller that wants it.
#[inline]
#[doc(hidden)]
pub fn powf_libm(x: f32, p: f32) -> f32 {
    powf(x, p)
}

/// Tiny seeded xorshift32 with a Box-Muller Gaussian, so noise is deterministic
/// and `std`-free.
struct Xorshift {
    state: u32,
}

impl Xorshift {
    fn new(seed: u32) -> Self {
        Self { state: seed.max(1) }
    }

    #[inline]
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.state = x;
        x
    }

    /// Uniform in `(0, 1)`.
    #[inline]
    fn next_unit(&mut self) -> f32 {
        (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0)
    }

    /// Standard-normal sample via Box-Muller.
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        sqrtf(-2.0 * logf(u1)) * cosf(2.0 * core::f32::consts::PI * u2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg_at(range_m: f32) -> RangeSimConfig {
        RangeSimConfig {
            range_m,
            ..Default::default()
        }
    }

    #[test]
    fn deterministic_given_seed() {
        let src = SourceConfig::default();
        let cfg = cfg_at(50.0);
        let a = simulate_clip(&src, &cfg);
        let b = simulate_clip(&src, &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn spreading_is_minus_six_db_per_doubling() {
        // Disable absorption so we isolate spreading; compare signal power.
        let src = SourceConfig::default();
        let no_abs = AbsorptionModel {
            alpha_ref_np_per_m: 0.0,
            f_ref_hz: 1000.0,
        };
        let p1 = signal_power(
            &src,
            &RangeSimConfig {
                range_m: 10.0,
                absorption: no_abs,
                ..Default::default()
            },
        );
        let p2 = signal_power(
            &src,
            &RangeSimConfig {
                range_m: 20.0,
                absorption: no_abs,
                ..Default::default()
            },
        );
        // Doubling distance => 1/4 the power => -6.02 dB.
        let ratio_db = 10.0 * (p2 / p1).log10();
        assert!((ratio_db + 6.02).abs() < 0.1, "got {ratio_db} dB");
    }

    #[test]
    fn absorption_attenuates_high_frequencies_more() {
        let m = AbsorptionModel::default();
        let lo = m.transmission(200.0, 100.0);
        let hi = m.transmission(4000.0, 100.0);
        assert!(hi < lo, "hi {hi} should be < lo {lo}");
        // High freq should be attenuated by a large factor over 100 m.
        assert!(hi < 0.5, "4 kHz over 100 m should lose >half: {hi}");
        // Low freq barely touched.
        assert!(lo > 0.9, "200 Hz over 100 m should be near 1: {lo}");
    }

    #[test]
    fn snr_falls_with_range() {
        let src = SourceConfig::default();
        let near = snr_db(&src, &cfg_at(10.0));
        let far = snr_db(&src, &cfg_at(150.0));
        assert!(near > far, "near SNR {near} should exceed far {far}");
    }

    #[test]
    fn alpha_is_quadratic_in_frequency() {
        let m = AbsorptionModel::default();
        // Doubling f should quadruple alpha.
        let a1 = m.alpha(1000.0);
        let a2 = m.alpha(2000.0);
        assert!((a2 / a1 - 4.0).abs() < 1e-4);
    }
}
