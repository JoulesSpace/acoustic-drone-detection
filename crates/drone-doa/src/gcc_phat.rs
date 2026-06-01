//! GCC-PHAT time-difference-of-arrival (TDOA) estimation between two channels.
//!
//! The generalized cross-correlation with phase transform (GCC-PHAT) is the
//! workhorse for small-array TDOA. Given two windowed channels `x` and `y`:
//!
//! 1. Take the FFT of each, `X` and `Y`.
//! 2. Form the cross-power spectrum `R = X · conj(Y)`.
//! 3. Whiten it (PHAT): `R ← R / |R|`. This throws away magnitude and keeps only
//!    phase, which makes the correlation peak sharp and robust to the source
//!    spectrum / reverberation colouring.
//! 4. Inverse-transform back to the lag domain; the peak lag is the TDOA in
//!    samples. A 3-point parabolic fit around the peak gives sub-sample
//!    resolution.
//!
//! Everything here is `no_std`-friendly. `microfft` only ships a forward complex
//! FFT, so the inverse transform uses the standard conjugate trick:
//! `ifft(Z) = conj(fft(conj(Z))) / N`.

use alloc::vec;
use alloc::vec::Vec;

use libm::sqrtf;
use microfft::Complex32;

use crate::GCC_FFT_SIZE;

/// Tunable knobs for [`gcc_phat_tdoa`].
#[derive(Debug, Clone, Copy)]
pub struct GccConfig {
    /// Coherence gate, as a fraction of the strongest cross-spectrum bin.
    ///
    /// Pure PHAT divides *every* bin by its magnitude, which for a narrowband
    /// source (like a drone's harmonics) blows up the empty noise bins to the
    /// same unit weight as the few signal bins — and they then dominate the
    /// correlation, flattening the peak. We instead only whiten bins whose
    /// magnitude is at least `gate · max_magnitude` and zero the rest. With
    /// `gate = 0` this reduces to classical PHAT.
    pub gate: f32,
    /// Only search lags within `±max_lag` samples. `None` searches the full
    /// `±N/2` window. Constraining to the array's physical limit rejects spurious
    /// far-lag peaks at low SNR.
    pub max_lag: Option<usize>,
}

impl Default for GccConfig {
    fn default() -> Self {
        Self {
            gate: 0.05,
            max_lag: None,
        }
    }
}

/// Estimate the TDOA (in **samples**, fractional) of `x` relative to `y`.
///
/// A positive return value means `x` lags `y` (the feature in `x` appears later)
/// — equivalently `x[n] ≈ y[n - tdoa]`. Both inputs are zero-padded/truncated to
/// [`GCC_FFT_SIZE`]. Returns `0.0` for empty or silent input.
pub fn gcc_phat_tdoa(x: &[f32], y: &[f32], cfg: &GccConfig) -> f32 {
    let n = GCC_FFT_SIZE;

    // Pack the two real channels into the complex FFT input (zero-padded).
    let mut xf = to_complex_buf(x, n);
    let mut yf = to_complex_buf(y, n);
    fft(&mut xf);
    fft(&mut yf);

    // Cross-power spectrum R = X · conj(Y), stored into `xf`, recording each
    // bin's magnitude so we can apply the coherence-gated PHAT in a second pass.
    let mut max_mag = 0.0_f32;
    let mut mags = vec![0.0_f32; n];
    for ((rx, ry), m) in xf.iter_mut().zip(yf.iter()).zip(mags.iter_mut()) {
        let re = rx.re * ry.re + rx.im * ry.im;
        let im = rx.im * ry.re - rx.re * ry.im;
        rx.re = re;
        rx.im = im;
        let mag = sqrtf(re * re + im * im);
        *m = mag;
        if mag > max_mag {
            max_mag = mag;
        }
    }

    // Coherence-gated PHAT: whiten (phase-only) the bins that actually carry
    // cross-channel energy, and zero the rest so noise bins can't flatten the
    // correlation peak. Falls back to classical PHAT when `gate == 0`.
    let threshold = cfg.gate * max_mag;
    for (rx, &mag) in xf.iter_mut().zip(mags.iter()) {
        if mag > threshold && mag > f32::MIN_POSITIVE {
            rx.re /= mag;
            rx.im /= mag;
        } else {
            rx.re = 0.0;
            rx.im = 0.0;
        }
    }

    // Inverse transform via the conjugate trick. The real part is the
    // (circular) cross-correlation; index 0 is lag 0, the upper half holds the
    // negative lags wrapped around.
    ifft(&mut xf);
    let corr: Vec<f32> = xf.iter().map(|c| c.re).collect();

    let max_lag = cfg.max_lag.unwrap_or(n / 2).min(n / 2);
    peak_lag_parabolic(&corr, max_lag)
}

/// Find the integer lag of maximum correlation within `±max_lag`, then refine it
/// with a 3-point parabolic interpolation for sub-sample accuracy.
///
/// `corr` is the circular cross-correlation: `corr[0]` is lag 0, `corr[tau]` for
/// `tau <= max_lag` is positive lag `tau`, and `corr[N - tau]` is negative lag
/// `-tau`.
fn peak_lag_parabolic(corr: &[f32], max_lag: usize) -> f32 {
    let n = corr.len();
    if n == 0 {
        return 0.0;
    }

    // Walk the allowed lags, mapping each to its wrapped index.
    let mut best_lag: isize = 0;
    let mut best_val = corr[0];
    for lag in 0..=(max_lag as isize) {
        for &l in &[lag, -lag] {
            if l == 0 && lag != 0 {
                continue;
            }
            let idx = wrap_index(l, n);
            if corr[idx] > best_val {
                best_val = corr[idx];
                best_lag = l;
            }
        }
    }

    // Parabolic refinement using the two neighbouring lags.
    let y0 = corr[wrap_index(best_lag - 1, n)];
    let y1 = corr[wrap_index(best_lag, n)];
    let y2 = corr[wrap_index(best_lag + 1, n)];
    let denom = y0 - 2.0 * y1 + y2;
    let delta = if libm::fabsf(denom) > f32::EPSILON {
        0.5 * (y0 - y2) / denom
    } else {
        0.0
    };
    // Guard the vertex against a degenerate fit pushing it past a neighbour.
    let delta = delta.clamp(-1.0, 1.0);
    best_lag as f32 + delta
}

/// Map a (possibly negative) lag to an index into the length-`n` correlation
/// buffer, wrapping modulo `n`.
#[inline]
fn wrap_index(lag: isize, n: usize) -> usize {
    let m = n as isize;
    (((lag % m) + m) % m) as usize
}

/// Copy a real slice into a zero-padded length-`n` complex buffer.
fn to_complex_buf(x: &[f32], n: usize) -> Vec<Complex32> {
    let mut buf = vec![Complex32::new(0.0, 0.0); n];
    for (b, &v) in buf.iter_mut().zip(x.iter().take(n)) {
        b.re = v;
    }
    buf
}

/// Forward complex FFT at [`GCC_FFT_SIZE`].
fn fft(buf: &mut Vec<Complex32>) {
    debug_assert_eq!(buf.len(), GCC_FFT_SIZE);
    let arr: &mut [Complex32; GCC_FFT_SIZE] = buf
        .as_mut_slice()
        .try_into()
        .expect("buffer is GCC_FFT_SIZE long");
    // The transform is in place; the returned reference aliases `arr`.
    let _ = microfft::complex::cfft_2048(arr);
}

/// Inverse complex FFT at [`GCC_FFT_SIZE`] via the conjugate trick, normalized.
fn ifft(buf: &mut [Complex32]) {
    debug_assert_eq!(buf.len(), GCC_FFT_SIZE);
    for c in buf.iter_mut() {
        c.im = -c.im;
    }
    let arr: &mut [Complex32; GCC_FFT_SIZE] = buf.try_into().expect("buffer is GCC_FFT_SIZE long");
    let _ = microfft::complex::cfft_2048(arr);
    let inv_n = 1.0 / GCC_FFT_SIZE as f32;
    for c in arr.iter_mut() {
        c.re *= inv_n;
        c.im = -c.im * inv_n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;

    /// Build a band-limited noisy-ish test signal (sum of a few tones).
    fn test_signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32;
                0.6 * libm::sinf(2.0 * PI * 0.05 * t)
                    + 0.3 * libm::sinf(2.0 * PI * 0.11 * t + 0.7)
                    + 0.1 * libm::sinf(2.0 * PI * 0.17 * t + 1.9)
            })
            .collect()
    }

    /// Delay a signal by an integer number of samples (zeros shifted in).
    fn delay_int(x: &[f32], d: isize) -> Vec<f32> {
        let n = x.len();
        let mut y = vec![0.0; n];
        for (i, out) in y.iter_mut().enumerate() {
            let src = i as isize - d;
            if src >= 0 && (src as usize) < n {
                *out = x[src as usize];
            }
        }
        y
    }

    #[test]
    fn recovers_zero_lag() {
        let x = test_signal(512);
        let tau = gcc_phat_tdoa(&x, &x, &GccConfig::default());
        assert!(tau.abs() < 0.05, "expected ~0, got {tau}");
    }

    #[test]
    fn recovers_positive_integer_lag() {
        let x = test_signal(1024);
        // y is x delayed by 7 -> x leads y, so x relative to y is -7.
        let y = delay_int(&x, 7);
        let tau = gcc_phat_tdoa(&x, &y, &GccConfig::default());
        assert!((tau - (-7.0)).abs() < 0.2, "expected ~-7, got {tau}");
    }

    #[test]
    fn recovers_negative_integer_lag() {
        let x = test_signal(1024);
        let y = delay_int(&x, -5);
        let tau = gcc_phat_tdoa(&x, &y, &GccConfig::default());
        assert!((tau - 5.0).abs() < 0.2, "expected ~+5, got {tau}");
    }

    #[test]
    fn max_lag_constrains_search() {
        let x = test_signal(1024);
        let y = delay_int(&x, 30);
        let cfg = GccConfig {
            max_lag: Some(5),
            ..Default::default()
        };
        let tau = gcc_phat_tdoa(&x, &y, &cfg);
        // True lag is -30 but the search is clamped to ±5 (plus up to ±1 of
        // parabolic sub-sample slack at the boundary).
        assert!(tau.abs() <= 6.0, "expected within ±6, got {tau}");
    }
}
