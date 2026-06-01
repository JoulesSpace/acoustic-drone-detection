//! Window functions applied before the FFT to reduce spectral leakage.

use crate::{Frame, FRAME_SIZE};
use core::f32::consts::PI;

/// Apply a periodic Hann window to `frame` in place.
///
/// Hann is a sensible default for tonal/harmonic signals like drone motors:
/// good main-lobe/side-lobe trade-off and cheap to compute. We compute the
/// coefficients on the fly via `libm::cosf` so there is no static table to
/// carry around on the embedded target.
pub fn hann_in_place(frame: &mut Frame) {
    let denom = (FRAME_SIZE - 1) as f32;
    for (i, sample) in frame.iter_mut().enumerate() {
        let w = 0.5 - 0.5 * libm::cosf(2.0 * PI * i as f32 / denom);
        *sample *= w;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hann_zeroes_the_edges() {
        let mut frame = [1.0_f32; FRAME_SIZE];
        hann_in_place(&mut frame);
        assert!(frame[0].abs() < 1e-6);
        assert!(frame[FRAME_SIZE - 1].abs() < 1e-6);
        // Middle of the window is ~1.0 for a unit input.
        assert!((frame[FRAME_SIZE / 2] - 1.0).abs() < 1e-2);
    }
}
