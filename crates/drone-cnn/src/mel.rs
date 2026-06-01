//! Log-mel spectrogram front-end for the CNN.
//!
//! The upstream literature (Al-Emadi et al., and the MDPI drone-audio CNNs)
//! feeds a 2-D time-frequency image - almost always a (log-)mel spectrogram -
//! into a small convolutional net. We reproduce that input faithfully, but on
//! top of the repo's existing DSP front-end so the comparison is apples to
//! apples: the magnitude spectra come from `drone_bench::util::spectra` (Hann
//! window, 1024-pt real FFT, 50% hop), exactly what every heuristic approach in
//! the harness sees.
//!
//! Pipeline per clip:
//!   1. magnitude spectra -> one column per STFT frame (`NUM_BINS` linear bins);
//!   2. project each column through a triangular mel filterbank -> `N_MELS`;
//!   3. `log(1 + mel)` compression (stable, monotone, no negative-infinity);
//!   4. resample the time axis to a fixed `N_FRAMES` columns by nearest-frame
//!      pick (clips vary in length; the CNN wants a fixed `N_MELS x N_FRAMES`
//!      image), and per-feature standardize with dataset statistics.
//!
//! Everything here is deterministic: no RNG, no parallelism with nondeterministic
//! reduction order.

use drone_bench::util::spectra;
use drone_dsp::{bin_to_hz, NUM_BINS};

/// Number of mel bands (image height). 64 is in the band the upstream CNNs use
/// (Al-Emadi uses ~ this; MDPI papers use 40-128).
pub const N_MELS: usize = 64;

/// Number of time frames after resampling (image width). ~32 frames covers a
/// ~1 s clip at the harness hop and matches the task's "~32 time frames".
pub const N_FRAMES: usize = 32;

/// Upper edge of the mel filterbank (Hz). Drone blade-pass + harmonics and the
/// confusable broadband cues all live well under this; it also stays below the
/// 8 kHz Nyquist of the 16 kHz pipeline.
const MEL_HZ_MAX: f32 = 8_000.0;
/// Lower edge of the mel filterbank (Hz).
const MEL_HZ_MIN: f32 = 20.0;

/// A fixed-size log-mel image, row-major `[mel][frame]`, flattened to
/// `N_MELS * N_FRAMES`. Stored flat so it drops straight into a candle tensor.
pub type MelImage = Vec<f32>;

/// HTK mel scale.
fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}
fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

/// Precomputed triangular mel filterbank over the `NUM_BINS` linear FFT bins.
/// Built once for a sample rate and reused for every clip.
pub struct MelBank {
    /// `filters[m]` is the list of `(bin, weight)` with nonzero overlap for mel
    /// band `m`. Sparse so the projection is cheap and order-deterministic.
    filters: Vec<Vec<(usize, f32)>>,
}

impl MelBank {
    /// Build the filterbank for `sample_rate`. Triangular filters spaced evenly
    /// on the mel scale between [`MEL_HZ_MIN`, `MEL_HZ_MAX`].
    pub fn new(sample_rate: u32) -> Self {
        let mel_min = hz_to_mel(MEL_HZ_MIN);
        let mel_max = hz_to_mel(MEL_HZ_MAX.min(sample_rate as f32 / 2.0));
        // N_MELS+2 edge points -> N_MELS triangles.
        let mut edges = [0.0f32; N_MELS + 2];
        for (i, e) in edges.iter_mut().enumerate() {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (N_MELS + 1) as f32;
            *e = mel_to_hz(mel);
        }
        // Precompute each FFT bin's center frequency once.
        let bin_hz: Vec<f32> = (0..NUM_BINS).map(|b| bin_to_hz(b, sample_rate)).collect();

        let mut filters = Vec::with_capacity(N_MELS);
        for m in 0..N_MELS {
            let (lo, ctr, hi) = (edges[m], edges[m + 1], edges[m + 2]);
            let mut taps = Vec::new();
            for (b, &f) in bin_hz.iter().enumerate() {
                let w = if f >= lo && f <= ctr && ctr > lo {
                    (f - lo) / (ctr - lo)
                } else if f > ctr && f <= hi && hi > ctr {
                    (hi - f) / (hi - ctr)
                } else {
                    0.0
                };
                if w > 0.0 {
                    taps.push((b, w));
                }
            }
            filters.push(taps);
        }
        Self { filters }
    }

    /// Project one linear magnitude spectrum (`NUM_BINS`) into `N_MELS` bands.
    fn project(&self, spectrum: &[f32]) -> [f32; N_MELS] {
        let mut out = [0.0f32; N_MELS];
        for (m, taps) in self.filters.iter().enumerate() {
            let mut acc = 0.0f32;
            for &(b, w) in taps {
                acc += spectrum[b] * w;
            }
            out[m] = acc;
        }
        out
    }

    /// Compute the fixed-size log-mel image for one clip.
    ///
    /// Returns a flat `N_MELS * N_FRAMES` row-major vector (`[mel][frame]`).
    /// Time frames are resampled to `N_FRAMES` by nearest-frame indexing, which
    /// is deterministic and length-agnostic. Empty/silent clips yield zeros.
    pub fn log_mel_image(&self, samples: &[f32]) -> MelImage {
        let frames = spectra(samples); // Vec<[f32; NUM_BINS]>
        let mut img = vec![0.0f32; N_MELS * N_FRAMES];
        if frames.is_empty() {
            return img;
        }
        let n_in = frames.len();
        for t in 0..N_FRAMES {
            // Nearest source frame for output column t.
            let src = if N_FRAMES == 1 {
                0
            } else {
                ((t as f32) * (n_in.saturating_sub(1)) as f32 / (N_FRAMES - 1) as f32).round()
                    as usize
            };
            let src = src.min(n_in - 1);
            let mel = self.project(&frames[src]);
            for (m, &v) in mel.iter().enumerate() {
                // log(1+x): stable, monotone, zero at zero. Standard log-mel
                // compression sans the -inf of plain log.
                img[m * N_FRAMES + t] = (1.0 + v).ln();
            }
        }
        img
    }
}

/// Per-feature mean/std over a set of images, for input standardization.
/// Computed on the TRAIN set only and reused for every split (no test peeking).
pub struct Standardizer {
    mean: Vec<f32>,
    inv_std: Vec<f32>,
}

impl Standardizer {
    /// Fit channel-wise (per pixel position) mean/std over `images`. With a
    /// single fixed image geometry this is `N_MELS * N_FRAMES` statistics.
    pub fn fit(images: &[MelImage]) -> Self {
        let dim = N_MELS * N_FRAMES;
        let mut mean = vec![0.0f32; dim];
        let mut var = vec![0.0f32; dim];
        let n = images.len().max(1) as f32;
        for img in images {
            for (i, &v) in img.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n;
        }
        for img in images {
            for (i, &v) in img.iter().enumerate() {
                let d = v - mean[i];
                var[i] += d * d;
            }
        }
        let inv_std: Vec<f32> = var.iter().map(|v| 1.0 / ((v / n).sqrt() + 1e-6)).collect();
        Self { mean, inv_std }
    }

    /// Standardize an image in place: `(x - mean) / std`.
    pub fn apply(&self, img: &mut MelImage) {
        for (i, v) in img.iter_mut().enumerate() {
            *v = (*v - self.mean[i]) * self.inv_std[i];
        }
    }
}
