//! Registry of detection approaches.
//!
//! Each approach is one self-contained file implementing [`crate::Approach`].
//! To add one: create `src/approaches/<name>.rs`, then add a `mod` line and one
//! entry to [`all`]. Nothing else in the harness needs to change.

mod band_ratio;
mod cepstrum;
mod envelope_periodicity;
mod feature_fusion;
mod fusion;
mod gtcc_lr;
mod hps;
mod mfcc_lr;
mod mfcc_mlp;
mod physics_fused;
mod sentry;
mod spectral_gate;
mod spectrogram_template;
mod template;

use crate::Approach;

/// The "classic" base approaches that the meta/ensemble approach (`fusion`)
/// stacks over. Kept separate from [`all`] so `fusion` can instantiate them
/// without recursing on itself.
pub fn stackable() -> Vec<Box<dyn Approach>> {
    vec![
        Box::new(band_ratio::BandRatio::new()),
        Box::new(template::Template::new()),
        Box::new(hps::Hps::new()),
        Box::new(spectral_gate::SpectralGate::new()),
        Box::new(cepstrum::Cepstrum::new()),
        Box::new(mfcc_lr::MfccLr::new()),
    ]
}

/// Every approach the benchmark runs, in display order.
pub fn all() -> Vec<Box<dyn Approach>> {
    let mut v = stackable();
    v.push(Box::new(mfcc_mlp::MfccMlp::new()));
    v.push(Box::new(gtcc_lr::GtccLr::new()));
    v.push(Box::new(feature_fusion::FeatureFusion::new()));
    v.push(Box::new(spectrogram_template::SpectrogramTemplate::new()));
    v.push(Box::new(envelope_periodicity::EnvelopePeriodicity::new()));
    v.push(Box::new(physics_fused::PhysicsFused::new()));
    v.push(Box::new(fusion::Fusion::new()));
    v.push(Box::new(sentry::Sentry::new()));
    v
}
