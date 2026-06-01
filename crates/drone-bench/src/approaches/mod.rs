//! Registry of detection approaches.
//!
//! Each approach is one self-contained file implementing [`crate::Approach`].
//! To add one: create `src/approaches/<name>.rs`, then add a `mod` line and one
//! entry to [`all`]. Nothing else in the harness needs to change.

mod band_ratio;
mod cepstrum;
mod hps;
mod mfcc_lr;
mod spectral_gate;
mod template;

use crate::Approach;

/// Every approach the benchmark runs, in display order.
pub fn all() -> Vec<Box<dyn Approach>> {
    vec![
        Box::new(band_ratio::BandRatio::new()),
        Box::new(template::Template::new()),
        Box::new(hps::Hps::new()),
        Box::new(spectral_gate::SpectralGate::new()),
        Box::new(cepstrum::Cepstrum::new()),
        Box::new(mfcc_lr::MfccLr::new()),
    ]
}
