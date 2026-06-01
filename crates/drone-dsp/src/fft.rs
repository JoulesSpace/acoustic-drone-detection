//! Real FFT and magnitude-spectrum computation.
//!
//! Backed by [`microfft`], which is pure Rust and `no_std`, so this lowers onto
//! edge targets unchanged. The FFT size is fixed to [`crate::FRAME_SIZE`]; if
//! you change that constant you must swap the `rfft_*` call below to match.

use crate::{Frame, Spectrum, NUM_BINS};

/// Compute the linear magnitude spectrum of a windowed frame.
///
/// The frame is consumed in place by the FFT (microfft works on the input
/// buffer), so callers should pass a scratch copy of their audio.
///
/// Note on bin 0: microfft packs the real-valued DC term into `re` and the
/// real-valued Nyquist term into `im` of the first bin. We expose `|DC|` as
/// `spectrum[0]` and drop the Nyquist magnitude - it almost never matters for
/// drone signatures and keeping the layout flat keeps the downstream feature
/// code simple. See `agent-memory/dsp-notes.md`.
pub fn magnitude_spectrum(frame: &mut Frame) -> Spectrum {
    let bins = microfft::real::rfft_1024(frame);
    debug_assert_eq!(bins.len(), NUM_BINS);

    let mut mag = [0.0_f32; NUM_BINS];
    mag[0] = libm::fabsf(bins[0].re);
    for i in 1..NUM_BINS {
        let re = bins[i].re;
        let im = bins[i].im;
        mag[i] = libm::sqrtf(re * re + im * im);
    }
    mag
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FRAME_SIZE;
    use core::f32::consts::PI;

    #[test]
    fn pure_tone_peaks_in_the_expected_bin() {
        // A sine at exactly bin 32 should dominate the spectrum at bin 32.
        let target_bin = 32usize;
        let mut frame = [0.0_f32; FRAME_SIZE];
        for (i, s) in frame.iter_mut().enumerate() {
            *s = libm::sinf(2.0 * PI * target_bin as f32 * i as f32 / FRAME_SIZE as f32);
        }
        let mag = magnitude_spectrum(&mut frame);

        let mut peak = 0usize;
        for i in 1..NUM_BINS {
            if mag[i] > mag[peak] {
                peak = i;
            }
        }
        assert_eq!(peak, target_bin);
    }
}
