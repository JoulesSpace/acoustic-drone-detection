//! Azimuth estimation from per-channel array data.
//!
//! Given the `M` channels of a ULA, we estimate one TDOA per adjacent mic pair
//! with GCC-PHAT, convert each to a per-pair `sin(θ)` observation, and combine
//! them. Because every adjacent pair shares the same baseline `d`, the
//! least-squares combination of independent `sin(θ)` estimates is just their
//! mean — which also averages down the per-pair noise by `√(M-1)`.
//!
//! We work in the `sin(θ)` domain rather than averaging angles directly: the
//! TDOA is linear in `sin(θ)`, so noise is well-behaved there, and a single
//! `asin` at the end maps back to an angle clamped to the unambiguous range.

use alloc::vec::Vec;

use crate::gcc_phat::{gcc_phat_tdoa, GccConfig};
use crate::geometry::UlaGeometry;

/// Result of an azimuth estimate, with enough detail to debug a bad fit.
#[derive(Debug, Clone)]
pub struct AzimuthEstimate {
    /// Estimated azimuth in radians, in `[-π/2, π/2]`.
    pub azimuth_rad: f32,
    /// Estimated azimuth in degrees (convenience mirror of `azimuth_rad`).
    pub azimuth_deg: f32,
    /// The averaged `sin(θ)` observation the angle was derived from.
    pub sin_theta: f32,
    /// Per-adjacent-pair TDOA estimates in samples (length `M - 1`).
    pub pair_tdoa_samples: Vec<f32>,
}

/// Estimate the source azimuth from `channels`, one `&[f32]` per microphone.
///
/// `sample_rate` is in Hz. The channels must be in array order (mic 0..M). The
/// search is constrained to physically possible lags from the geometry, which
/// is the main low-SNR robustness lever.
///
/// # Panics
/// Panics if `channels.len()` does not equal `geom.mics`, or if `geom.mics < 2`.
pub fn estimate_azimuth(
    channels: &[&[f32]],
    geom: &UlaGeometry,
    sample_rate: u32,
    cfg: &GccConfig,
) -> AzimuthEstimate {
    assert!(geom.mics >= 2, "need at least 2 mics");
    assert_eq!(
        channels.len(),
        geom.mics,
        "channel count must match geometry"
    );

    // Constrain the lag search to the array's physical limit (+1 sample slack
    // for the parabolic vertex), unless the caller already set a tighter bound.
    let phys_max_lag = libm::ceilf(geom.max_adjacent_tdoa() * sample_rate as f32) as usize + 1;
    let mut pair_cfg = *cfg;
    pair_cfg.max_lag = Some(match cfg.max_lag {
        Some(user) => user.min(phys_max_lag),
        None => phys_max_lag,
    });

    let mut pair_tdoa_samples = Vec::with_capacity(geom.mics - 1);
    let mut sin_sum = 0.0_f32;
    for k in 0..geom.mics - 1 {
        // TDOA of mic k relative to mic k+1, in samples.
        let tau = gcc_phat_tdoa(channels[k], channels[k + 1], &pair_cfg);
        pair_tdoa_samples.push(tau);
        let tau_s = tau / sample_rate as f32;
        // tau_s = (k - (k+1)) * d sin θ / c = -d sin θ / c  =>  sinθ = -tau_s c/d
        let sin_theta = -tau_s * geom.speed_of_sound / geom.spacing_m;
        sin_sum += sin_theta;
    }

    let sin_theta = (sin_sum / (geom.mics - 1) as f32).clamp(-1.0, 1.0);
    let azimuth_rad = libm::asinf(sin_theta);
    AzimuthEstimate {
        azimuth_rad,
        azimuth_deg: azimuth_rad.to_degrees(),
        sin_theta,
        pair_tdoa_samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{simulate_array, DroneSource, SimConfig};

    #[test]
    fn clean_array_recovers_azimuth() {
        let geom = UlaGeometry::new(4, 0.05);
        let sr = 16_000;
        for true_deg in [-60.0, -30.0, 0.0, 20.0, 45.0_f32] {
            let cfg = SimConfig {
                sample_rate: sr,
                num_samples: 2048,
                true_azimuth_deg: true_deg,
                snr_db: 60.0,
                seed: 42,
            };
            let src = DroneSource::default();
            let ch = simulate_array(&src, &geom, &cfg);
            let refs: Vec<&[f32]> = ch.iter().map(|c| c.as_slice()).collect();
            let est = estimate_azimuth(&refs, &geom, sr, &GccConfig::default());
            assert!(
                (est.azimuth_deg - true_deg).abs() < 3.0,
                "true {true_deg}, est {}",
                est.azimuth_deg
            );
        }
    }
}
