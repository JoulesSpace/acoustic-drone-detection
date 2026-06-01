//! 2D spectro-temporal template matching (supervised).
//!
//! Each clip is reduced to a small fixed-size log-mel spectrogram "patch":
//! `N_MELS` mel bins × `N_TIME` time slots, flattened to an `N_MELS * N_TIME`
//! vector. A mel filterbank (mel = 2595*log10(1+f/700)) maps the 512-bin Hann
//! magnitude spectrum of every frame to `N_MELS` log(1+energy) mel bins. To get
//! a fixed number of time slots from a variable-length clip, the frames are
//! split into `N_TIME` contiguous groups and each group's mel vector is
//! averaged, so short and long clips both map to the same `N_MELS × N_TIME`
//! grid (capturing coarse time structure, not just a static spectrum).
//!
//! `fit`: build the mean flattened patch over drone (`label == 1`) train clips
//! and over non-drone (`label == 0`) clips, then L2-normalize each template.
//!
//! `score`: build and L2-normalize the clip's flattened patch, compute cosine
//! similarity to each template, and map `0.5 * (1 + drone_sim - nondrone_sim)`
//! into `[0, 1]`. This contrastive mapping separates better than the raw
//! drone similarity alone. Returns `0.5` when unfit.
//!
//! Everything is deterministic and pure Rust.

use crate::dataset::Sample;
use crate::util::{cosine, l2_normalize, spectra};
use crate::Approach;
use drone_dsp::{bin_to_hz, NUM_BINS};

/// Number of mel filterbank channels (rows of the patch).
const N_MELS: usize = 24;
/// Number of time slots (columns of the patch).
const N_TIME: usize = 16;
/// Flattened patch length.
const PATCH_LEN: usize = N_MELS * N_TIME;

pub struct SpectrogramTemplate {
    /// L2-normalized mean patch of drone clips.
    drone_template: [f32; PATCH_LEN],
    /// L2-normalized mean patch of non-drone clips.
    nondrone_template: [f32; PATCH_LEN],
    /// Whether `fit` produced usable templates.
    fitted: bool,
}

impl Default for SpectrogramTemplate {
    fn default() -> Self {
        Self {
            drone_template: [0.0; PATCH_LEN],
            nondrone_template: [0.0; PATCH_LEN],
            fitted: false,
        }
    }
}

impl SpectrogramTemplate {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Approach for SpectrogramTemplate {
    fn name(&self) -> &str {
        "spectrogram_template"
    }

    fn description(&self) -> &str {
        "2D log-mel spectro-temporal patch templates, contrastive cosine match"
    }

    fn fit(&mut self, train: &[Sample]) {
        let mut drone_acc = [0.0f64; PATCH_LEN];
        let mut nondrone_acc = [0.0f64; PATCH_LEN];
        let mut n_drone = 0usize;
        let mut n_nondrone = 0usize;

        for s in train {
            let patch = clip_patch(&s.samples, s.sample_rate);
            if s.label == 1 {
                for (a, &v) in drone_acc.iter_mut().zip(patch.iter()) {
                    *a += v as f64;
                }
                n_drone += 1;
            } else {
                for (a, &v) in nondrone_acc.iter_mut().zip(patch.iter()) {
                    *a += v as f64;
                }
                n_nondrone += 1;
            }
        }

        if n_drone == 0 || n_nondrone == 0 {
            return;
        }

        let mut drone_t = [0.0f32; PATCH_LEN];
        let mut nondrone_t = [0.0f32; PATCH_LEN];
        for (t, &a) in drone_t.iter_mut().zip(drone_acc.iter()) {
            *t = (a / n_drone as f64) as f32;
        }
        for (t, &a) in nondrone_t.iter_mut().zip(nondrone_acc.iter()) {
            *t = (a / n_nondrone as f64) as f32;
        }
        l2_normalize(&mut drone_t);
        l2_normalize(&mut nondrone_t);

        self.drone_template = drone_t;
        self.nondrone_template = nondrone_t;
        self.fitted = true;
    }

    fn score(&self, samples: &[f32], sample_rate: u32) -> f32 {
        if !self.fitted {
            return 0.5;
        }
        let mut patch = clip_patch(samples, sample_rate);
        l2_normalize(&mut patch);
        let drone_sim = cosine(&patch, &self.drone_template);
        let nondrone_sim = cosine(&patch, &self.nondrone_template);
        // Contrastive mapping into [0, 1]: prefer the template the clip is
        // closer to. With unit-norm vectors each cosine is in [-1, 1], so the
        // difference is in [-2, 2] and the result is in [-0.5, 1.5]; clamp.
        let raw = 0.5 * (1.0 + drone_sim - nondrone_sim);
        raw.clamp(0.0, 1.0)
    }
}

/// Build the flattened `N_MELS * N_TIME` log-mel patch for a clip.
///
/// Layout is row-major over time: index `t * N_MELS + m` is mel channel `m` in
/// time slot `t`. Empty/silent clips yield an all-zero patch.
fn clip_patch(samples: &[f32], sample_rate: u32) -> [f32; PATCH_LEN] {
    let mut patch = [0.0f32; PATCH_LEN];
    let frames = spectra(samples);
    if frames.is_empty() {
        return patch;
    }

    let fb = mel_filterbank(sample_rate);

    // Per-frame log-mel vectors.
    let n_frames = frames.len();
    let mut mels: Vec<[f32; N_MELS]> = Vec::with_capacity(n_frames);
    for sp in &frames {
        let mut log_mel = [0.0f32; N_MELS];
        for (m, filt) in fb.iter().enumerate() {
            let mut e = 0.0f32;
            for &(bin, w) in filt {
                // Power (magnitude squared) accumulation.
                e += w * sp[bin] * sp[bin];
            }
            // log(1 + energy), naturally bottoming out at 0 for silence.
            log_mel[m] = (1.0 + e).ln();
        }
        mels.push(log_mel);
    }

    // Split the `n_frames` frames into `N_TIME` contiguous groups and average
    // each group's mel vector. Using float boundaries spreads frames evenly
    // even when `n_frames` is not a multiple of `N_TIME`; if there are fewer
    // frames than slots, later slots reuse the nearest frame (group is empty
    // -> falls back to a single frame index).
    for t in 0..N_TIME {
        let start = (t * n_frames) / N_TIME;
        let mut end = ((t + 1) * n_frames) / N_TIME;
        if end <= start {
            // Empty group (n_frames < N_TIME): use the single nearest frame.
            end = (start + 1).min(n_frames);
        }
        let count = (end - start) as f32;
        let mut avg = [0.0f32; N_MELS];
        for f in &mels[start..end] {
            for (a, &v) in avg.iter_mut().zip(f.iter()) {
                *a += v;
            }
        }
        for (m, a) in avg.iter().enumerate() {
            patch[t * N_MELS + m] = a / count;
        }
    }

    patch
}

/// Build a mel filterbank as a list of `(bin, weight)` pairs per channel.
///
/// `N_MELS` triangular filters are spaced equally on the mel scale between 0 Hz
/// and the Nyquist frequency, then mapped back to the linear FFT bins.
fn mel_filterbank(sample_rate: u32) -> Vec<Vec<(usize, f32)>> {
    let f_max = sample_rate as f32 / 2.0;
    let mel_max = hz_to_mel(f_max);

    let n_points = N_MELS + 2;
    let mut centers_hz = [0.0f32; N_MELS + 2];
    for (i, c) in centers_hz.iter_mut().enumerate() {
        let mel = mel_max * i as f32 / (n_points - 1) as f32;
        *c = mel_to_hz(mel);
    }

    let mut fb: Vec<Vec<(usize, f32)>> = Vec::with_capacity(N_MELS);
    for m in 0..N_MELS {
        let lo = centers_hz[m];
        let ctr = centers_hz[m + 1];
        let hi = centers_hz[m + 2];
        let mut filt = Vec::new();
        for bin in 0..NUM_BINS {
            let f = bin_to_hz(bin, sample_rate);
            let w = if f >= lo && f <= ctr && ctr > lo {
                (f - lo) / (ctr - lo)
            } else if f > ctr && f <= hi && hi > ctr {
                (hi - f) / (hi - ctr)
            } else {
                0.0
            };
            if w > 0.0 {
                filt.push((bin, w));
            }
        }
        fb.push(filt);
    }
    fb
}

/// Hz → mel (`2595*log10(1+f/700)`).
#[inline]
fn hz_to_mel(f: f32) -> f32 {
    2595.0 * (1.0 + f / 700.0).log10()
}

/// Mel → Hz (inverse of [`hz_to_mel`]).
#[inline]
fn mel_to_hz(m: f32) -> f32 {
    700.0 * (10.0f32.powf(m / 2595.0) - 1.0)
}
