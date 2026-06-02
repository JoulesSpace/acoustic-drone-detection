//! Estimate a drone's blade-pass fundamental frequency from audio.
//!
//! Multirotor drones radiate a strongly harmonic acoustic signature: each rotor
//! emits a blade-pass tone at `f0 = (rotor_rate_hz) * B`, where `B` is the
//! number of blades, plus a stack of harmonics at `2·f0, 3·f0, …`. The
//! blade-pass fundamental `f0` is therefore a directly measurable drone
//! *property*, and the rotor RPM follows from it:
//!
//! ```text
//! rotor_rate_hz ≈ f0 / B
//! rotor_rpm     ≈ 60 · f0 / B
//! ```
//!
//! This crate estimates `f0` (Hz) robustly over the ~50-400 Hz band that covers
//! typical small-multirotor blade-pass rates. Robustness comes from fusing
//! three independent cues per frame - Harmonic Product Spectrum, cepstrum, and
//! time-domain autocorrelation - and cross-checking them to kill the octave
//! errors any single cue is prone to, then taking a confidence-weighted robust
//! median across the clip's frames.
//!
//! Entry point: [`estimate_f0`] (and [`estimate_f0_conf`] for a confidence too).

#![forbid(unsafe_code)]

pub mod autocorr;
pub mod cepstrum;
pub mod fuse;
pub mod hps;

use drone_bench::util::spectra;
use drone_dsp::{Frame, FRAME_SIZE};

use fuse::{fuse, Candidate, FrameEstimate};

/// Hop between successive analysis frames (matches `drone_bench::util::HOP`,
/// i.e. 50% overlap) so the time-domain framing lines up with the spectra.
const HOP: usize = FRAME_SIZE / 2;

/// Default lower bound of the f0 search band, in Hz.
pub const F0_MIN_HZ: f32 = 50.0;
/// Default upper bound of the f0 search band, in Hz.
pub const F0_MAX_HZ: f32 = 400.0;

/// A clip-level f0 estimate with a confidence.
#[derive(Clone, Copy, Debug)]
pub struct Estimate {
    /// Blade-pass fundamental in Hz, or `NaN` if no frame yielded a usable
    /// estimate.
    pub f0_hz: f32,
    /// Confidence in `[0, 1]`: how strongly and consistently the frames agreed.
    pub confidence: f32,
    /// Number of frames that contributed an estimate.
    pub n_frames: usize,
}

impl Estimate {
    /// Rotor rotation rate (Hz) implied by this f0 for a `blades`-blade rotor.
    pub fn rotor_rate_hz(&self, blades: u32) -> f32 {
        self.f0_hz / blades.max(1) as f32
    }

    /// Rotor RPM implied by this f0 for a `blades`-blade rotor.
    pub fn rotor_rpm(&self, blades: u32) -> f32 {
        60.0 * self.rotor_rate_hz(blades)
    }
}

/// Estimate the blade-pass fundamental `f0` (Hz) of a clip.
///
/// Returns `NaN` for empty/silent clips. This is the headline API; see
/// [`estimate_f0_conf`] for the confidence and frame count.
pub fn estimate_f0(samples: &[f32], sr: u32) -> f32 {
    estimate_f0_conf(samples, sr).f0_hz
}

/// Estimate `f0` over the default `[F0_MIN_HZ, F0_MAX_HZ]` band, returning the
/// fused frequency, a confidence, and how many frames contributed.
pub fn estimate_f0_conf(samples: &[f32], sr: u32) -> Estimate {
    estimate_f0_band(samples, sr, F0_MIN_HZ, F0_MAX_HZ)
}

/// Estimate `f0` over an explicit `[f_lo, f_hi]` Hz band.
pub fn estimate_f0_band(samples: &[f32], sr: u32, f_lo: f32, f_hi: f32) -> Estimate {
    if samples.is_empty() || sr == 0 {
        return Estimate {
            f0_hz: f32::NAN,
            confidence: 0.0,
            n_frames: 0,
        };
    }

    // Per-frame magnitude spectra (windowed FFT) from the shared front-end.
    let specs = spectra(samples);
    // Matching time-domain frames for the autocorrelation cue. We frame the same
    // way `spectra` does (50% hop, single zero-padded frame for short clips) but
    // keep the raw samples so autocorrelation sees the waveform, not the window.
    let frames = time_frames(samples);

    let n = specs.len().min(frames.len());
    let mut per_frame: Vec<FrameEstimate> = Vec::with_capacity(n);

    for i in 0..n {
        let spec = &specs[i];
        let hps_c = hps::hps_f0(spec, sr, f_lo, f_hi).map(|(f0, conf)| Candidate { f0, conf });
        let cep_c =
            cepstrum::cepstral_f0(spec, sr, f_lo, f_hi).map(|(f0, conf)| Candidate { f0, conf });
        let ac_c = autocorr::autocorr_f0(&frames[i], sr, f_lo, f_hi)
            .map(|(f0, conf)| Candidate { f0, conf });

        if let Some(est) = fuse(&[hps_c, cep_c, ac_c], spec, sr) {
            per_frame.push(est);
        }
    }

    if per_frame.is_empty() {
        return Estimate {
            f0_hz: f32::NAN,
            confidence: 0.0,
            n_frames: 0,
        };
    }

    aggregate(&per_frame)
}

/// Aggregate per-frame estimates into a clip estimate.
///
/// We take the (confidence-weighted) robust median of the per-frame f0s. The
/// median resists the occasional outlier frame (a gust, a transient, a frame
/// that octave-flipped despite fusion). Confidence is the mean per-frame
/// confidence scaled by how tightly the frames cluster around the median.
fn aggregate(per_frame: &[FrameEstimate]) -> Estimate {
    // Weighted median by confidence (fall back to unweighted if all-zero).
    let mut idx: Vec<usize> = (0..per_frame.len()).collect();
    idx.sort_by(|&a, &b| per_frame[a].f0.partial_cmp(&per_frame[b].f0).unwrap());

    let total_w: f32 = per_frame.iter().map(|e| e.conf.max(1e-3)).sum();
    let mut cum = 0.0_f32;
    let mut median = per_frame[idx[per_frame.len() / 2]].f0;
    for &i in &idx {
        cum += per_frame[i].conf.max(1e-3);
        if cum >= 0.5 * total_w {
            median = per_frame[i].f0;
            break;
        }
    }

    let mean_conf = per_frame.iter().map(|e| e.conf).sum::<f32>() / per_frame.len() as f32;

    // Cluster tightness: fraction of frames within 6% of the median.
    let near = per_frame
        .iter()
        .filter(|e| (e.f0 - median).abs() <= 0.06 * median)
        .count() as f32
        / per_frame.len() as f32;

    Estimate {
        f0_hz: median,
        confidence: (mean_conf * near).clamp(0.0, 1.0),
        n_frames: per_frame.len(),
    }
}

/// Frame a clip into raw (un-windowed) time-domain frames matching the framing
/// used by `drone_bench::util::spectra` (50% hop, single zero-padded frame for
/// clips shorter than one frame).
fn time_frames(samples: &[f32]) -> Vec<Frame> {
    let mut out = Vec::new();
    if samples.is_empty() {
        return out;
    }
    if samples.len() < FRAME_SIZE {
        let mut frame: Frame = [0.0; FRAME_SIZE];
        frame[..samples.len()].copy_from_slice(samples);
        out.push(frame);
        return out;
    }
    let mut start = 0;
    while start + FRAME_SIZE <= samples.len() {
        let mut frame: Frame = [0.0; FRAME_SIZE];
        frame.copy_from_slice(&samples[start..start + FRAME_SIZE]);
        out.push(frame);
        start += HOP;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    /// Build a harmonic drone-like clip with a known f0.
    fn synth_clip(f0: f32, sr: u32, secs: f32, n_harm: usize) -> Vec<f32> {
        let n = (secs * sr as f32) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let mut v = 0.0;
                for h in 1..=n_harm {
                    v += (1.0 / h as f32) * (2.0 * PI * f0 * h as f32 * t).sin();
                }
                0.5 * v
            })
            .collect()
    }

    #[test]
    fn estimates_known_f0() {
        let sr = 16_000;
        for &truth in &[90.0_f32, 120.0, 160.0, 210.0] {
            let clip = synth_clip(truth, sr, 0.5, 8);
            let f0 = estimate_f0(&clip, sr);
            let err = (f0 - truth).abs();
            assert!(err < 4.0, "f0 {f0} truth {truth} err {err}");
        }
    }

    #[test]
    fn empty_clip_is_nan() {
        assert!(estimate_f0(&[], 16_000).is_nan());
        assert!(estimate_f0(&[0.0; 100], 0).is_nan());
    }

    #[test]
    fn missing_fundamental_no_octave_error() {
        // Harmonics start at 2·f0: a naive peak picker would say 2·f0. Fusion
        // should still report f0.
        let sr = 16_000;
        let truth = 130.0;
        let n = (0.5 * sr as f32) as usize;
        let clip: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / sr as f32;
                let mut v = 0.0;
                for h in 2..=8 {
                    v += (1.0 / h as f32) * (2.0 * PI * truth * h as f32 * t).sin();
                }
                0.5 * v
            })
            .collect();
        let f0 = estimate_f0(&clip, sr);
        assert!((f0 - truth).abs() < 8.0, "f0 {f0} truth {truth}");
    }

    #[test]
    fn rpm_relation() {
        let est = Estimate {
            f0_hz: 120.0,
            confidence: 1.0,
            n_frames: 10,
        };
        // 2-blade rotor: rate = 60 Hz, rpm = 3600.
        assert!((est.rotor_rate_hz(2) - 60.0).abs() < 1e-3);
        assert!((est.rotor_rpm(2) - 3600.0).abs() < 1e-3);
    }
}
