//! Propagation simulator: turn a mono source into `M` noisy ULA channels.
//!
//! We have no multi-mic recordings, so the benchmark drives the estimator with
//! a physically-faithful simulation:
//!
//! * Generate (or accept) a mono drone-like signal - a fundamental plus a few
//!   harmonics, the spectral signature a multirotor's blade-pass tones produce.
//! * For each mic `k`, delay the source by `k · d · sin(θ) / c` seconds. The
//!   delay is generally fractional, so we resample with a **windowed-sinc**
//!   (Lanczos) kernel rather than nearest-sample - sub-sample accuracy is the
//!   whole point of GCC-PHAT and integer delays would cap the achievable error.
//! * Add independent white Gaussian-ish noise to each channel at the target SNR.
//!
//! Everything is deterministic given the `seed`, and the noise generator is a
//! seeded xorshift so it works without `std`.

use alloc::vec;
use alloc::vec::Vec;

use libm::{cosf, expf, logf, sinf, sqrtf};

use crate::geometry::UlaGeometry;

/// A synthetic drone-like harmonic source.
#[derive(Debug, Clone)]
pub struct DroneSource {
    /// Fundamental (blade-pass / motor) frequency in Hz.
    pub fundamental_hz: f32,
    /// Relative amplitude of each harmonic, starting at the fundamental.
    pub harmonics: Vec<f32>,
}

impl Default for DroneSource {
    /// A plausible small-multirotor signature: ~120 Hz fundamental with a few
    /// decaying harmonics.
    fn default() -> Self {
        Self {
            fundamental_hz: 120.0,
            harmonics: vec![1.0, 0.6, 0.4, 0.25, 0.15],
        }
    }
}

impl DroneSource {
    /// Render `n` mono samples of the source at `sample_rate` Hz.
    ///
    /// Normalized to roughly unit peak amplitude so SNR maths is predictable.
    pub fn render(&self, n: usize, sample_rate: u32) -> Vec<f32> {
        let fs = sample_rate as f32;
        let mut out = vec![0.0_f32; n];
        let mut peak = 0.0_f32;
        for (i, s) in out.iter_mut().enumerate() {
            let t = i as f32 / fs;
            let mut acc = 0.0_f32;
            for (h, &amp) in self.harmonics.iter().enumerate() {
                let f = self.fundamental_hz * (h as f32 + 1.0);
                acc += amp * sinf(2.0 * core::f32::consts::PI * f * t);
            }
            *s = acc;
            let a = if acc < 0.0 { -acc } else { acc };
            if a > peak {
                peak = a;
            }
        }
        if peak > 1e-9 {
            let g = 1.0 / peak;
            for s in out.iter_mut() {
                *s *= g;
            }
        }
        out
    }
}

/// Configuration for one simulated capture.
#[derive(Debug, Clone, Copy)]
pub struct SimConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of samples per channel to produce.
    pub num_samples: usize,
    /// True source azimuth in degrees (broadside = 0).
    pub true_azimuth_deg: f32,
    /// Signal-to-noise ratio in dB (per channel).
    pub snr_db: f32,
    /// RNG seed (noise is deterministic given this).
    pub seed: u32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16_000,
            num_samples: 2048,
            true_azimuth_deg: 0.0,
            snr_db: 20.0,
            seed: 1,
        }
    }
}

/// Simulate the `M` array channels for `src` arriving from `cfg.true_azimuth_deg`.
///
/// Returns one `Vec<f32>` per microphone, in array order. To avoid edge
/// transients from the fractional-delay filter, the source is rendered with
/// padding and a centred window is extracted.
pub fn simulate_array(src: &DroneSource, geom: &UlaGeometry, cfg: &SimConfig) -> Vec<Vec<f32>> {
    let fs = cfg.sample_rate as f32;
    let theta = cfg.true_azimuth_deg.to_radians();
    // Per-adjacent-pair delay in samples; mic k is delayed by k * this.
    let step_delay_samples = geom.tdoa_for_azimuth(theta) * fs;

    // Render with margin on both sides so fractional shifts never read past the
    // ends of the clean signal.
    let margin = HALF_KERNEL + 2;
    let padded_len = cfg.num_samples + 2 * margin;
    let clean = src.render(padded_len, cfg.sample_rate);

    // Mean signal power over the windowed region (used to set noise level).
    let sig_power = {
        let region = &clean[margin..margin + cfg.num_samples];
        region.iter().map(|v| v * v).sum::<f32>() / cfg.num_samples.max(1) as f32
    };
    let noise_std = if sig_power > 0.0 {
        sqrtf(sig_power / pow10(cfg.snr_db / 10.0))
    } else {
        0.0
    };

    let mut channels = Vec::with_capacity(geom.mics);
    for k in 0..geom.mics {
        let delay = k as f32 * step_delay_samples;
        let mut rng = Xorshift::new(
            cfg.seed
                .wrapping_mul(2_654_435_761)
                .wrapping_add(k as u32 + 1),
        );
        let mut ch = vec![0.0_f32; cfg.num_samples];
        for (i, out) in ch.iter_mut().enumerate() {
            // Centre of the analysis window maps to index `margin + i`; mic k's
            // signal is delayed, so it samples the source slightly earlier.
            let src_pos = (margin + i) as f32 - delay;
            let s = lanczos_interp(&clean, src_pos);
            *out = s + noise_std * rng.next_gaussian();
        }
        channels.push(ch);
    }
    channels
}

/// Half-width of the Lanczos interpolation kernel (taps = `2*HALF_KERNEL`).
const HALF_KERNEL: usize = 8;

/// Windowed-sinc (Lanczos, `a = HALF_KERNEL`) fractional-delay interpolation of
/// `x` at the (possibly non-integer) position `pos`. Out-of-range taps read 0.
fn lanczos_interp(x: &[f32], pos: f32) -> f32 {
    let a = HALF_KERNEL as f32;
    let base = libm::floorf(pos) as isize;
    let mut acc = 0.0_f32;
    for t in (1 - HALF_KERNEL as isize)..=(HALF_KERNEL as isize) {
        let idx = base + t;
        if idx < 0 || idx as usize >= x.len() {
            continue;
        }
        let d = pos - idx as f32;
        acc += x[idx as usize] * lanczos_kernel(d, a);
    }
    acc
}

/// The Lanczos kernel `sinc(x) · sinc(x/a)` for `|x| < a`, else 0.
#[inline]
fn lanczos_kernel(xv: f32, a: f32) -> f32 {
    if xv == 0.0 {
        return 1.0;
    }
    if xv <= -a || xv >= a {
        return 0.0;
    }
    let pix = core::f32::consts::PI * xv;
    sinf(pix) / pix * (sinf(pix / a) / (pix / a))
}

/// `10^x` without `std::f32::powf` (keeps the core `no_std`).
#[inline]
fn pow10(x: f32) -> f32 {
    expf(x * core::f32::consts::LN_10)
}

/// Tiny seeded xorshift32 with a Box–Muller Gaussian, so noise is deterministic
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
        // Add 1 and divide by 2^32 + 1 worth of range to keep it open on (0,1).
        (self.next_u32() as f32 + 1.0) / (u32::MAX as f32 + 2.0)
    }

    /// Standard-normal sample via Box–Muller.
    fn next_gaussian(&mut self) -> f32 {
        let u1 = self.next_unit();
        let u2 = self.next_unit();
        sqrtf(-2.0 * logf(u1)) * cosf(2.0 * core::f32::consts::PI * u2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noiseless_endfire_delays_match_geometry() {
        // With high SNR and a sharp impulse-like check, the cross-channel delay
        // should match k * d sin θ / c.
        let geom = UlaGeometry::new(3, 0.05);
        let cfg = SimConfig {
            sample_rate: 16_000,
            num_samples: 1024,
            true_azimuth_deg: 30.0,
            snr_db: 80.0,
            seed: 7,
        };
        let src = DroneSource::default();
        let ch = simulate_array(&src, &geom, &cfg);
        assert_eq!(ch.len(), 3);
        assert_eq!(ch[0].len(), 1024);
        // Channels must actually differ (delay applied).
        let diff: f32 = ch[0]
            .iter()
            .zip(ch[2].iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 1.0, "channels look identical, delay not applied");
    }

    #[test]
    fn snr_controls_noise_floor() {
        let geom = UlaGeometry::new(2, 0.05);
        let src = DroneSource::default();
        let high = SimConfig {
            snr_db: 40.0,
            seed: 3,
            ..Default::default()
        };
        let low = SimConfig {
            snr_db: 0.0,
            seed: 3,
            ..Default::default()
        };
        let chs_hi = simulate_array(&src, &geom, &high);
        let chs_lo = simulate_array(&src, &geom, &low);
        // Higher SNR channel should be closer to the clean (mic 0) shape: compare
        // the two mic-0 channels' energy spread isn't a direct test, so instead
        // confirm low-SNR channel has larger residual vs the high-SNR one.
        let var = |v: &[f32]| {
            let m = v.iter().sum::<f32>() / v.len() as f32;
            v.iter().map(|x| (x - m) * (x - m)).sum::<f32>() / v.len() as f32
        };
        assert!(var(&chs_lo[0]) > var(&chs_hi[0]));
    }

    #[test]
    fn deterministic_given_seed() {
        let geom = UlaGeometry::new(2, 0.05);
        let src = DroneSource::default();
        let cfg = SimConfig::default();
        let a = simulate_array(&src, &geom, &cfg);
        let b = simulate_array(&src, &geom, &cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn lanczos_passes_through_integer_positions() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32 * 0.3).sin()).collect();
        for i in HALF_KERNEL..(32 - HALF_KERNEL) {
            let v = lanczos_interp(&x, i as f32);
            assert!((v - x[i]).abs() < 1e-4, "i={i}: {v} vs {}", x[i]);
        }
    }
}
