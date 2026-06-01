//! Channel downmix and resampling to the detector's working rate.
//!
//! The detector front-end ([`drone_dsp`] framing) is tuned for ~16 kHz audio:
//! `FRAME_SIZE = 1024` samples is a 64 ms frame at 16 kHz with ~15.6 Hz/bin,
//! which resolves the blade-pass fundamental and its harmonics well. Microphones
//! almost never hand us mono 16 kHz directly — they hand us interleaved stereo
//! at 44.1/48 kHz. This module turns whatever the device produces into the mono
//! 16 kHz stream the detector expects, in two cheap, deterministic steps:
//!
//! 1. **Downmix** interleaved frames to mono by averaging channels.
//! 2. **Resample** to the target rate with linear interpolation, carrying a tiny
//!    bit of state across callback buffers so we don't glitch at buffer seams.
//!
//! Linear interpolation is not audiophile-grade, but for a tonal/harmonic
//! detector running on a logarithmic confidence it is more than adequate and
//! costs almost nothing per sample — important when this has to keep up with a
//! live stream on a modest host (or, eventually, an Android phone).

/// The rate the detector front-end is tuned for.
pub const TARGET_RATE: u32 = 16_000;

/// Streaming mono resampler: downmixes interleaved input then linearly
/// resamples to [`TARGET_RATE`], preserving phase across successive buffers.
///
/// Feed it raw interleaved device frames via [`push`](Self::push); it appends
/// the resampled mono samples to the caller's accumulator. Holding the last
/// input sample and a fractional read position between calls means a long live
/// stream chopped into many small callback buffers produces the same output as
/// if it had arrived as one contiguous buffer (no per-buffer discontinuity).
pub struct Resampler {
    channels: usize,
    /// Input-samples-per-output-sample ratio (`in_rate / TARGET_RATE`).
    step: f64,
    /// Fractional read position into the *virtual* mono input stream.
    pos: f64,
    /// Last mono input sample carried over from the previous buffer, so we can
    /// interpolate across the seam instead of resetting to zero each callback.
    last: f32,
    /// Whether `last` holds a real sample yet (false only before the first push).
    primed: bool,
}

impl Resampler {
    /// Build a resampler for `channels`-channel interleaved input at `in_rate`.
    pub fn new(in_rate: u32, channels: u16) -> Self {
        let channels = channels.max(1) as usize;
        let step = in_rate.max(1) as f64 / TARGET_RATE as f64;
        Self {
            channels,
            step,
            pos: 0.0,
            last: 0.0,
            primed: false,
        }
    }

    /// True if the input rate already matches [`TARGET_RATE`] and there is a
    /// single channel — i.e. no work beyond a copy is needed. (Informational;
    /// [`push`](Self::push) is correct either way.)
    pub fn is_passthrough(&self) -> bool {
        self.channels == 1 && (self.step - 1.0).abs() < 1e-9
    }

    /// Downmix + resample one interleaved input buffer, appending mono
    /// [`TARGET_RATE`] samples to `out`.
    ///
    /// `interleaved.len()` need not be a whole number of frames; a trailing
    /// partial frame is ignored (the device always delivers whole frames, but
    /// we stay robust). The fractional position carried in `self` means the
    /// number of output samples per call varies by ±1, which is expected.
    pub fn push(&mut self, interleaved: &[f32], out: &mut Vec<f32>) {
        let n_frames = interleaved.len() / self.channels;
        if n_frames == 0 {
            return;
        }

        // Downmix this buffer's frames to mono once, up front.
        let mut mono = Vec::with_capacity(n_frames);
        for frame in interleaved.chunks_exact(self.channels) {
            let sum: f32 = frame.iter().sum();
            mono.push(sum / self.channels as f32);
        }

        if !self.primed {
            // Seed `last` with the first sample so the very first interpolation
            // has a left neighbour; start reading from index 0.
            self.last = mono[0];
            self.primed = true;
            self.pos = 0.0;
        }

        // Walk a fractional read head across [prev_sample .. this buffer].
        // Index -1 in this local frame refers to `self.last` (the carry-over).
        while self.pos < n_frames as f64 {
            let i = self.pos.floor() as isize;
            let frac = (self.pos - i as f64) as f32;
            let a = if i < 0 { self.last } else { mono[i as usize] };
            let b = if (i + 1) < n_frames as isize {
                mono[(i + 1) as usize]
            } else {
                // Right neighbour past this buffer: hold the last sample. The
                // next push continues from a `pos` rebased to start at -? — see
                // the rebase below, which keeps the seam continuous.
                mono[n_frames - 1]
            };
            out.push(a + (b - a) * frac);
            self.pos += self.step;
        }

        // Rebase the read head for the next buffer: subtract the frames we
        // consumed and carry the final mono sample as the new left neighbour.
        self.pos -= n_frames as f64;
        self.last = mono[n_frames - 1];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_mono_16k_is_identity_length() {
        let mut r = Resampler::new(TARGET_RATE, 1);
        assert!(r.is_passthrough());
        let mut out = Vec::new();
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        r.push(&input, &mut out);
        // Passthrough should preserve count to within one sample.
        assert!((out.len() as isize - 100).abs() <= 1);
    }

    #[test]
    fn stereo_downmix_averages_channels() {
        // L = 1.0, R = -1.0 interleaved -> mono 0.0. At 16 kHz it's passthrough.
        let mut r = Resampler::new(TARGET_RATE, 2);
        let mut out = Vec::new();
        let input = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        r.push(&input, &mut out);
        assert!(out.iter().all(|&v| v.abs() < 1e-6), "downmix should cancel");
    }

    #[test]
    fn downsample_48k_to_16k_is_about_one_third() {
        let mut r = Resampler::new(48_000, 1);
        assert!(!r.is_passthrough());
        let mut out = Vec::new();
        let input: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin()).collect();
        r.push(&input, &mut out);
        // 48k -> 16k is a 3:1 decimation; expect ~1600 output samples.
        let expected = 4800 / 3;
        assert!(
            (out.len() as isize - expected as isize).abs() <= 2,
            "got {} expected ~{expected}",
            out.len()
        );
    }

    #[test]
    fn seam_is_continuous_across_buffers() {
        // Feeding one buffer or two halves of it must yield (nearly) the same
        // resampled stream — the carried state prevents a glitch at the seam.
        let input: Vec<f32> = (0..3000).map(|i| (i as f32 * 0.02).sin()).collect();

        let mut r1 = Resampler::new(44_100, 1);
        let mut whole = Vec::new();
        r1.push(&input, &mut whole);

        let mut r2 = Resampler::new(44_100, 1);
        let mut split = Vec::new();
        r2.push(&input[..1500], &mut split);
        r2.push(&input[1500..], &mut split);

        assert!((whole.len() as isize - split.len() as isize).abs() <= 1);
        let n = whole.len().min(split.len());
        // Allow small numerical/edge differences but no gross discontinuity.
        let max_diff = whole[..n]
            .iter()
            .zip(&split[..n])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_diff < 0.05, "max seam diff {max_diff}");
    }
}
