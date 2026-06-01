//! Multi-vendor drone brand/model recognition.
//!
//! This is the "recognition" head scaled toward many vendors/models - broader
//! than the 3-class `drone-id`. A clip of mono audio is turned into MFCC (plus
//! optional spectral/harmonic) features ([`features`]) and classified by a
//! deterministic multinomial logistic-regression (softmax) [`model`], with the
//! evaluation reported as accuracy, macro-F1, per-class precision/recall/F1 and
//! a confusion matrix ([`metrics`]).
//!
//! The data source ([`data`]) is CLI-selectable: a deterministic synthetic
//! 12-brand generator, or a real on-disk dataset (class folders, or a flat set
//! of brand-named WAVs windowed into segments).
//!
//! It reuses `drone_bench::util::spectra` (the shared framing/windowing/FFT
//! front-end) and `drone_bench::dataset::Sample`, so its notion of a clip and of
//! "MFCC" matches the rest of the repository. Everything is pure Rust, uses no
//! ML crate, and is fully deterministic.

#![forbid(unsafe_code)]

pub mod data;
pub mod features;
pub mod metrics;
pub mod model;

pub use data::VendorDataset;
pub use metrics::{evaluate, MulticlassReport};
pub use model::SoftmaxClassifier;
