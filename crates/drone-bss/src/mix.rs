//! A mixing simulator for blind source separation experiments.
//!
//! We synthesize a handful of statistically-independent acoustic sources -
//! drone-like harmonic stacks at chosen blade-pass fundamentals, broadband
//! noise, a tonal interferer - and mix them through a random invertible `M x M`
//! matrix into `M` observed channels. This gives us *ground truth* (the clean
//! sources and the true mixing) to score separation against.
//!
//! The drone-like sources mirror the synthesis used elsewhere in the suite
//! (`drone_bench::dataset::synth` / `drone-dsp`): a harmonic stack `f0, 2f0, ...`
//! with `~1/h`-tapered amplitudes, gentle amplitude modulation, and a small
//! broadband floor, which is what a multirotor's blade-pass tone plus motor
//! harmonics and hiss look like.
//!
//! **Mixing model.** This is *instantaneous* mixing: `x[n] = A s[n]`, a single
//! scalar gain per (source, mic) pair. Real acoustic mixing is *convolutive*
//! (each path is a different impulse response, so `x = sum_k a_k * s_k`), which
//! the crate docs flag as the next step. Instantaneous mixing is the standard
//! first-cut idealization for validating ICA, and is a good approximation for
//! closely-spaced mics in the far field where inter-mic delay is negligible.

// `random_invertible` and the mixing loop index the matrix in two dimensions;
// the index-loop form mirrors `x = A s` directly.
#![allow(clippy::needless_range_loop)]

use crate::rng::Rng;

/// A drone-like harmonic source description.
///
/// The acoustic model mirrors `drone_bench::dataset::synth`: an amplitude-
/// modulated harmonic stack (`f0, 2f0, ...` with `~1/h`-tapered amplitudes) plus
/// a small broadband floor. The floor matters for *realism* - a real multirotor
/// radiates motor hiss between the harmonic teeth, and the suite's harmonic-comb
/// detector expects that inter-harmonic background to exist. Each source draws
/// its own seeded floor, so the sources stay mutually independent (which is what
/// ICA requires).
#[derive(Clone, Debug)]
pub struct DroneSource {
    /// Blade-pass fundamental in Hz.
    pub f0: f64,
    /// Number of harmonics in the stack.
    pub harmonics: usize,
    /// Amplitude-modulation rate in Hz (rotor/blade wobble).
    pub am_hz: f64,
    /// Modulation depth in `[0, 1]`.
    pub am_depth: f64,
    /// Broadband motor-floor level relative to the harmonic stack.
    pub noise_level: f64,
}

impl DroneSource {
    /// A reasonable default multirotor source at fundamental `f0`.
    pub fn at(f0: f64) -> Self {
        Self {
            f0,
            harmonics: 8,
            am_hz: 7.0,
            am_depth: 0.25,
            noise_level: 0.10,
        }
    }

    /// Render the source to `n` samples at `sample_rate`, normalized to unit RMS.
    ///
    /// `rng` supplies the (deterministic) broadband motor floor; passing a
    /// distinct generator per source keeps the sources independent.
    pub fn render(&self, n: usize, sample_rate: u32, rng: &mut Rng) -> Vec<f64> {
        let sr = sample_rate as f64;
        let mut out = vec![0.0; n];
        for (i, s) in out.iter_mut().enumerate() {
            let t = i as f64 / sr;
            let am = 1.0 + self.am_depth * (2.0 * std::f64::consts::PI * self.am_hz * t).sin();
            let mut v = 0.0;
            for h in 1..=self.harmonics {
                let a = 0.5 / h as f64;
                v += a * (2.0 * std::f64::consts::PI * self.f0 * h as f64 * t).sin();
            }
            *s = am * v + self.noise_level * rng.gaussian();
        }
        normalize_rms(&mut out);
        out
    }
}

/// Configuration for [`mix_sources`] and [`scene`].
#[derive(Clone, Debug)]
pub struct MixConfig {
    /// Number of samples per channel.
    pub n: usize,
    /// Sample rate (Hz).
    pub sample_rate: u32,
    /// Seed controlling the mixing matrix and any noise sources.
    pub seed: u64,
    /// Drone level relative to the interferers, in dB, applied by [`scene`].
    ///
    /// `0.0` means equal-loudness sources. A *negative* value models a quiet /
    /// distant drone buried under a louder interferer - the multi-source masking
    /// regime where a single-mic detector struggles but BSS, being scale-blind,
    /// can still pull the drone out. This is the level knob that creates a real
    /// detection lift to demonstrate.
    pub drone_gain_db: f64,
}

impl Default for MixConfig {
    fn default() -> Self {
        Self {
            n: 16_000,
            sample_rate: 16_000,
            seed: 0x0005_15ED,
            drone_gain_db: 0.0,
        }
    }
}

/// One independent source plus a label of what it is, so a benchmark can find
/// "the drone" among the recovered components.
#[derive(Clone, Debug)]
pub struct LabeledSource {
    /// The clean source signal (unit RMS).
    pub signal: Vec<f64>,
    /// `true` if this source is a drone (the one detection should recover).
    pub is_drone: bool,
    /// Short human label (`drone:120Hz`, `noise`, `tone:1800Hz`).
    pub label: String,
}

/// The output of the mixing simulator: clean sources, the true mixing matrix,
/// and the observed mixture channels.
#[derive(Clone, Debug)]
pub struct Mixture {
    /// The clean source signals with labels (`K` of them; here `K == M`).
    pub sources: Vec<LabeledSource>,
    /// True mixing matrix `A` (`M x M`, row-major): `x = A s`.
    pub mixing: Vec<Vec<f64>>,
    /// Observed mixture channels (`M`, each length `n`).
    pub channels: Vec<Vec<f64>>,
    /// Sample rate (Hz).
    pub sample_rate: u32,
}

impl Mixture {
    /// Index of the (first) drone source, if any.
    pub fn drone_index(&self) -> Option<usize> {
        self.sources.iter().position(|s| s.is_drone)
    }
}

/// Build a labeled source set, then mix it.
///
/// `sources` are the clean signals (already rendered, ideally unit RMS); they
/// are mixed through a random invertible `M x M` matrix (`M == sources.len()`).
/// The matrix is drawn from a seeded generator and conditioned to be safely
/// invertible (diagonally loaded), so the instantaneous model `x = A s` is
/// genuinely separable.
///
/// # Panics
///
/// Panics if `sources` is empty or the signals differ in length.
pub fn mix_sources(sources: Vec<LabeledSource>, cfg: &MixConfig) -> Mixture {
    let m = sources.len();
    assert!(m > 0, "mix: no sources");
    let n = sources[0].signal.len();
    assert!(
        sources.iter().all(|s| s.signal.len() == n),
        "mix: ragged sources"
    );

    let mut rng = Rng::new(cfg.seed);
    let mixing = random_invertible(m, &mut rng);

    // x_c[t] = sum_k A[c][k] * s_k[t].
    let mut channels = vec![vec![0.0; n]; m];
    for c in 0..m {
        for (k, src) in sources.iter().enumerate() {
            let a = mixing[c][k];
            let sig = &src.signal;
            let ch = &mut channels[c];
            for t in 0..n {
                ch[t] += a * sig[t];
            }
        }
    }

    Mixture {
        sources,
        mixing,
        channels,
        sample_rate: cfg.sample_rate,
    }
}

/// Convenience: render a standard scene and mix it.
///
/// `drone_f0s` lists the fundamentals of the drone sources; `extra` lists
/// non-drone sources (noise / tonal interferers) to add. The total source count
/// is the mixing dimension `M`.
pub fn scene(drone_f0s: &[f64], extra: &[ExtraSource], cfg: &MixConfig) -> Mixture {
    let mut rng = Rng::new(cfg.seed ^ 0xA5A5_A5A5);
    let drone_gain = 10f64.powf(cfg.drone_gain_db / 20.0);
    let mut sources: Vec<LabeledSource> = Vec::new();
    for (k, &f0) in drone_f0s.iter().enumerate() {
        // Each drone gets its own seeded floor so the sources stay independent.
        let mut src_rng = Rng::new(cfg.seed ^ (0x0000_D00E_u64.wrapping_mul(k as u64 + 1)));
        let mut signal = DroneSource::at(f0).render(cfg.n, cfg.sample_rate, &mut src_rng);
        for v in signal.iter_mut() {
            *v *= drone_gain;
        }
        sources.push(LabeledSource {
            signal,
            is_drone: true,
            label: format!("drone:{f0:.0}Hz"),
        });
    }
    for e in extra {
        sources.push(e.render(cfg.n, cfg.sample_rate, &mut rng));
    }
    mix_sources(sources, cfg)
}

/// A non-drone interferer for [`scene`].
#[derive(Clone, Debug)]
pub enum ExtraSource {
    /// Broadband white noise.
    Noise,
    /// A pure tone at the given Hz (e.g. a mains hum or whistle).
    Tone(f64),
}

impl ExtraSource {
    fn render(&self, n: usize, sample_rate: u32, rng: &mut Rng) -> LabeledSource {
        match self {
            ExtraSource::Noise => {
                let mut sig: Vec<f64> = (0..n).map(|_| rng.gaussian()).collect();
                normalize_rms(&mut sig);
                LabeledSource {
                    signal: sig,
                    is_drone: false,
                    label: "noise".to_string(),
                }
            }
            ExtraSource::Tone(f) => {
                let sr = sample_rate as f64;
                let mut sig: Vec<f64> = (0..n)
                    .map(|i| (2.0 * std::f64::consts::PI * f * i as f64 / sr).sin())
                    .collect();
                normalize_rms(&mut sig);
                LabeledSource {
                    signal: sig,
                    is_drone: false,
                    label: format!("tone:{f:.0}Hz"),
                }
            }
        }
    }
}

/// A random `M x M` matrix conditioned to be well-invertible: entries are
/// Gaussian, then the diagonal is loaded so the matrix is diagonally dominant
/// enough to avoid the (measure-zero but numerically annoying) near-singular
/// draws. Deterministic given the RNG state.
fn random_invertible(m: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut a = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            a[i][j] = rng.gaussian();
        }
        // Diagonal loading: push the diagonal out so the matrix stays
        // comfortably invertible regardless of the random draw.
        a[i][i] += if a[i][i] >= 0.0 { 1.5 } else { -1.5 };
    }
    a
}

/// Scale a signal to unit RMS in place (no-op for a silent signal).
fn normalize_rms(x: &mut [f64]) {
    let n = x.len() as f64;
    if n == 0.0 {
        return;
    }
    let ms = x.iter().map(|v| v * v).sum::<f64>() / n;
    let rms = ms.sqrt();
    if rms > 1e-12 {
        for v in x.iter_mut() {
            *v /= rms;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drone_render_is_unit_rms() {
        let mut rng = Rng::new(5);
        let s = DroneSource::at(120.0).render(16_000, 16_000, &mut rng);
        let rms = (s.iter().map(|v| v * v).sum::<f64>() / s.len() as f64).sqrt();
        assert!((rms - 1.0).abs() < 1e-6, "rms {rms}");
    }

    #[test]
    fn mixture_has_expected_shape() {
        let cfg = MixConfig {
            n: 4000,
            ..Default::default()
        };
        let mix = scene(&[120.0], &[ExtraSource::Noise], &cfg);
        assert_eq!(mix.channels.len(), 2);
        assert_eq!(mix.channels[0].len(), 4000);
        assert_eq!(mix.drone_index(), Some(0));
    }

    #[test]
    fn mixing_is_deterministic() {
        let cfg = MixConfig {
            n: 2000,
            ..Default::default()
        };
        let a = scene(&[110.0], &[ExtraSource::Tone(1800.0)], &cfg);
        let b = scene(&[110.0], &[ExtraSource::Tone(1800.0)], &cfg);
        assert_eq!(a.mixing, b.mixing);
        assert_eq!(a.channels, b.channels);
    }
}
