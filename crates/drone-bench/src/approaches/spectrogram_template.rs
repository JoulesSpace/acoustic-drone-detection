//! 2D spectro-temporal template matching.
//!
//! TODO(approach-wave-2): implement. Build an averaged 2D log-mel-spectrogram
//! "patch" template from drone training clips (fixed mel-bins × time-frames,
//! e.g. resampled/pooled to a small grid), then score a clip by the best
//! normalized cross-correlation of its spectrogram patch against the template.
//! The owner's template idea, extended to 2D (captures time structure). Confidence
//! in [0,1] from the correlation.

use crate::Approach;

#[derive(Default)]
pub struct SpectrogramTemplate;

impl SpectrogramTemplate {
    pub fn new() -> Self {
        Self
    }
}

impl Approach for SpectrogramTemplate {
    fn name(&self) -> &str {
        "spectrogram_template"
    }
    fn description(&self) -> &str {
        "2D log-mel spectro-temporal template correlation (stub)"
    }
    fn score(&self, _samples: &[f32], _sample_rate: u32) -> f32 {
        0.5
    }
}
