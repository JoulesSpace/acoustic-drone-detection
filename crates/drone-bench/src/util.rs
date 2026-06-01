//! Shared helpers for approaches: framing audio into magnitude spectra.

use drone_dsp::{hann_in_place, magnitude_spectrum, Frame, Spectrum, FRAME_SIZE};

/// Hop between successive frames (50% overlap).
pub const HOP: usize = FRAME_SIZE / 2;

/// Window a clip into overlapping frames and return one magnitude spectrum per
/// frame. Short clips (< one frame) are zero-padded into a single frame.
///
/// This is the common front-end almost every approach wants, so they don't each
/// re-implement framing + windowing + FFT.
pub fn spectra(samples: &[f32]) -> Vec<Spectrum> {
    let mut out = Vec::new();
    if samples.is_empty() {
        return out;
    }
    if samples.len() < FRAME_SIZE {
        let mut frame: Frame = [0.0; FRAME_SIZE];
        frame[..samples.len()].copy_from_slice(samples);
        hann_in_place(&mut frame);
        out.push(magnitude_spectrum(&mut frame));
        return out;
    }
    let mut start = 0;
    while start + FRAME_SIZE <= samples.len() {
        let mut frame: Frame = [0.0; FRAME_SIZE];
        frame.copy_from_slice(&samples[start..start + FRAME_SIZE]);
        hann_in_place(&mut frame);
        out.push(magnitude_spectrum(&mut frame));
        start += HOP;
    }
    out
}

/// Mean spectrum across all frames of a clip (a cheap clip-level summary).
pub fn mean_spectrum(samples: &[f32]) -> Spectrum {
    let frames = spectra(samples);
    let mut acc: Spectrum = [0.0; drone_dsp::NUM_BINS];
    if frames.is_empty() {
        return acc;
    }
    for sp in &frames {
        for (a, v) in acc.iter_mut().zip(sp.iter()) {
            *a += *v;
        }
    }
    let n = frames.len() as f32;
    for a in acc.iter_mut() {
        *a /= n;
    }
    acc
}

/// L2-normalize a vector in place (no-op if the norm is ~0).
pub fn l2_normalize(v: &mut [f32]) {
    let norm = libm_sqrt(v.iter().map(|x| x * x).sum::<f32>());
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity of two equal-length vectors, in `[-1, 1]`.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = libm_sqrt(a.iter().map(|x| x * x).sum::<f32>());
    let nb = libm_sqrt(b.iter().map(|x| x * x).sum::<f32>());
    if na > 1e-12 && nb > 1e-12 {
        dot / (na * nb)
    } else {
        0.0
    }
}

#[inline]
fn libm_sqrt(x: f32) -> f32 {
    x.sqrt()
}
