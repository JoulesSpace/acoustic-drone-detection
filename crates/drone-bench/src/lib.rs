//! Benchmark harness for acoustic drone-detection approaches.
//!
//! The contract is small on purpose: an [`Approach`] turns a clip of mono audio
//! into a confidence in `[0, 1]` that a drone is present. The harness handles
//! everything else - loading/splitting data, timing, metrics, JSON output - so
//! a new approach is just one file implementing one trait.

pub mod approach;
pub mod approaches;
pub mod dataset;
pub mod metrics;
pub mod util;

pub use approach::Approach;
pub use dataset::{Dataset, Sample};
pub use metrics::Metrics;
