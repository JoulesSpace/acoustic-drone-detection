//! Uniform linear array (ULA) geometry and the far-field plane-wave model.
//!
//! Mics sit on a line, equally spaced by `d` metres, indexed `0..M` along the
//! array axis. Azimuth `θ` is measured from broadside (the perpendicular
//! bisector of the array): `θ = 0` is straight ahead, positive towards the
//! high-index end of the array. For a far-field source the wavefront is planar,
//! so the extra path to mic `k` relative to mic `0` is `k·d·sin(θ)`, giving an
//! inter-mic (adjacent-pair) TDOA of `d·sin(θ)/c`.

use libm::{asinf, sinf};

/// Speed of sound in air, metres per second (≈ 343 m/s at 20 °C).
pub const SPEED_OF_SOUND: f32 = 343.0;

/// A uniform linear array: `mics` microphones spaced `spacing_m` metres apart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UlaGeometry {
    /// Number of microphones (`M >= 2`).
    pub mics: usize,
    /// Inter-mic spacing in metres.
    pub spacing_m: f32,
    /// Speed of sound in metres per second.
    pub speed_of_sound: f32,
}

impl UlaGeometry {
    /// Build a ULA with the default speed of sound ([`SPEED_OF_SOUND`]).
    ///
    /// # Panics
    /// Panics if `mics < 2` or `spacing_m <= 0`.
    pub fn new(mics: usize, spacing_m: f32) -> Self {
        Self::with_speed(mics, spacing_m, SPEED_OF_SOUND)
    }

    /// Build a ULA with an explicit speed of sound.
    ///
    /// # Panics
    /// Panics if `mics < 2`, `spacing_m <= 0`, or `speed_of_sound <= 0`.
    pub fn with_speed(mics: usize, spacing_m: f32, speed_of_sound: f32) -> Self {
        assert!(mics >= 2, "a ULA needs at least 2 mics");
        assert!(spacing_m > 0.0, "mic spacing must be positive");
        assert!(speed_of_sound > 0.0, "speed of sound must be positive");
        Self {
            mics,
            spacing_m,
            speed_of_sound,
        }
    }

    /// Adjacent-pair TDOA (seconds) for a plane wave from azimuth `theta_rad`.
    ///
    /// Positive means the wave reaches the lower-index mic first.
    #[inline]
    pub fn tdoa_for_azimuth(&self, theta_rad: f32) -> f32 {
        self.spacing_m * sinf(theta_rad) / self.speed_of_sound
    }

    /// TDOA (seconds) between mics `i` and `j` for azimuth `theta_rad`.
    ///
    /// Defined as `t_i - t_j`, i.e. the delay of mic `i` relative to mic `j`.
    /// For the ULA this is just `(i - j)` adjacent steps.
    #[inline]
    pub fn pair_tdoa_for_azimuth(&self, i: usize, j: usize, theta_rad: f32) -> f32 {
        (i as f32 - j as f32) * self.tdoa_for_azimuth(theta_rad)
    }

    /// The largest adjacent-pair delay (seconds), at endfire (`θ = ±90°`).
    ///
    /// This is the magnitude bound on any physically valid adjacent TDOA;
    /// estimates beyond it are clamped before inversion.
    #[inline]
    pub fn max_adjacent_tdoa(&self) -> f32 {
        self.spacing_m / self.speed_of_sound
    }

    /// The maximum unambiguous frequency (Hz) for this spacing: a half-wavelength
    /// equal to the spacing, `c / (2·d)`. Above this, spatial aliasing means a
    /// single pair's phase can map to multiple azimuths.
    #[inline]
    pub fn aliasing_free_max_hz(&self) -> f32 {
        self.speed_of_sound / (2.0 * self.spacing_m)
    }

    /// Recover azimuth (radians) from an adjacent-pair TDOA (seconds).
    ///
    /// Inverts `tdoa = d·sin(θ)/c`. The argument to `asin` is clamped to
    /// `[-1, 1]` so a slightly-too-large measured delay (noise, interpolation)
    /// saturates at endfire instead of producing `NaN`.
    #[inline]
    pub fn azimuth_from_adjacent_tdoa(&self, tdoa_s: f32) -> f32 {
        let s = (tdoa_s * self.speed_of_sound / self.spacing_m).clamp(-1.0, 1.0);
        asinf(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    #[test]
    fn broadside_has_zero_delay() {
        let g = UlaGeometry::new(4, 0.05);
        assert!(g.tdoa_for_azimuth(0.0).abs() < 1e-9);
    }

    #[test]
    fn endfire_delay_is_spacing_over_c() {
        let g = UlaGeometry::new(2, 0.08);
        let t = g.tdoa_for_azimuth(PI / 2.0);
        assert!((t - 0.08 / SPEED_OF_SOUND).abs() < 1e-7);
    }

    #[test]
    fn tdoa_azimuth_roundtrip() {
        let g = UlaGeometry::new(4, 0.043);
        for deg in [-80.0, -40.0, -10.0, 0.0, 25.0, 60.0, 80.0_f32] {
            let theta = deg.to_radians();
            let t = g.tdoa_for_azimuth(theta);
            let back = g.azimuth_from_adjacent_tdoa(t);
            assert!((back - theta).abs() < 1e-4, "deg {deg}: got {} rad", back);
        }
    }

    #[test]
    fn out_of_range_tdoa_saturates_not_nans() {
        let g = UlaGeometry::new(2, 0.05);
        let huge = g.max_adjacent_tdoa() * 2.0;
        let a = g.azimuth_from_adjacent_tdoa(huge);
        assert!(a.is_finite());
        assert!((a - PI / 2.0).abs() < 1e-5);
    }
}
