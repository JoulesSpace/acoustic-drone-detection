//! The pluggable detection-approach contract.

use crate::dataset::Sample;

/// A drone-detection approach: clip in, confidence out.
///
/// Implementations live one-per-file under [`crate::approaches`]. Keep them
/// self-contained so they can be developed and benchmarked independently.
pub trait Approach {
    /// Stable short name, also used as the results filename (`<name>.json`).
    fn name(&self) -> &str;

    /// One-line description of the method (shown in summaries).
    fn description(&self) -> &str {
        ""
    }

    /// Train on labelled samples. Default is a no-op for unsupervised /
    /// threshold-based methods (e.g. spectral gates). Supervised methods
    /// (template averaging, MFCC + logistic regression) override this.
    ///
    /// The harness calls `fit` on the **train** split only, then `score` on the
    /// held-out **test** split.
    fn fit(&mut self, _train: &[Sample]) {}

    /// Clip-level confidence in `[0, 1]` that a drone is present.
    ///
    /// `samples` is mono audio in `[-1, 1]`; `sample_rate` is in Hz. The return
    /// value MUST be finite and within `[0, 1]` (the harness debug-asserts this)
    /// so scores are comparable across approaches via ROC/PR.
    fn score(&self, samples: &[f32], sample_rate: u32) -> f32;
}
