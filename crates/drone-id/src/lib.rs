//! Multiclass drone-TYPE identification — the "recognition" head.
//!
//! Where `drone-detect` / `drone-bench` answer the binary question *"is a drone
//! present?"*, this crate answers *"which TYPE of drone is it?"* — the
//! recognition head sitting alongside the detection head, insightface-style. It
//! deliberately reuses the shared infrastructure: clips are
//! [`drone_bench::dataset::Sample`]s, the MFCC front-end mirrors `drone-bench`'s
//! `mfcc_lr` and runs on [`drone_bench::util::spectra`] over
//! [`drone_dsp`] primitives.
//!
//! Pipeline:
//! 1. [`data`] — build a `K`-class labelled set from real class folders or a
//!    deterministic synthetic generator, and split it K-class-stratified.
//! 2. [`features`] — per-clip MFCC features (mel filterbank -> log -> DCT,
//!    mean+std pooled).
//! 3. [`model`] — multinomial logistic regression (softmax) trained by
//!    deterministic gradient descent (no ML crate).
//! 4. [`metrics`] — confusion matrix, per-class precision/recall/F1, accuracy
//!    and macro-F1.

#![forbid(unsafe_code)]

pub mod data;
pub mod features;
pub mod metrics;
pub mod model;

pub use data::MultiDataset;
pub use metrics::{evaluate, ClassMetrics, MulticlassReport};
pub use model::SoftmaxClassifier;
