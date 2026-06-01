//! Choosing and (optionally) fitting a detection approach for live use.
//!
//! Live detection reuses the exact [`drone_bench::Approach`] implementations the
//! benchmark harness evaluates — there is no separate "live" scorer to drift out
//! of sync. The wrinkle is that **most** approaches are supervised: their
//! [`fit`](drone_bench::Approach::fit) trains a classifier (logistic regression,
//! an MLP, template averaging, ...) and an *unfit* instance scores garbage. At
//! runtime there is no labelled audio, so by default we restrict to approaches
//! that are meaningful with **no training**:
//!
//! * **`hps`** — Harmonic Product Spectrum + harmonic-comb contrast. Fully
//!   unsupervised; `fit` only nudges a logistic centre, and the default prior is
//!   sensible. This is the live default.
//! * **`band_ratio`** — the `drone-detect` band-energy heuristic. A pure rule;
//!   `fit` is a no-op.
//! * **`spectral_gate`** — self-calibrating logistic gate that *falls back to a
//!   hand-designed monotonic rule* when never fit, so it is usable unfit too.
//!
//! Any other approach (`mfcc_lr`, `mfcc_mlp`, `template`, `fusion`, ...) requires
//! `--train <dir>` of labelled audio; without it we refuse rather than emit
//! noise. With `--train`, we fit the chosen approach first, so any approach is
//! fair game.

use std::error::Error;
use std::path::Path;

use drone_bench::approaches;
use drone_bench::dataset::Dataset;
use drone_bench::Approach;

/// Approaches that produce meaningful confidences with **no** training.
///
/// Keep this in sync with the doc comment above; it is the allowlist used when
/// `--train` is *not* supplied.
pub const UNFIT_OK: &[&str] = &["hps", "band_ratio", "spectral_gate"];

/// The default live approach: unsupervised, needs no labelled data at runtime.
pub const DEFAULT_APPROACH: &str = "hps";

/// Build a fresh instance of the named approach from the benchmark registry.
///
/// Returns `None` if the name is unknown, listing the valid names is left to the
/// caller (see [`available_names`]).
fn instantiate(name: &str) -> Option<Box<dyn Approach>> {
    approaches::all().into_iter().find(|a| a.name() == name)
}

/// All approach names known to the benchmark registry, for help/error text.
pub fn available_names() -> Vec<String> {
    approaches::all()
        .iter()
        .map(|a| a.name().to_string())
        .collect()
}

/// Resolve the `--approach` / `--train` flags into a ready-to-score detector.
///
/// * If `train_dir` is `Some`, the dataset's `labels.csv` (header `path,label`)
///   is loaded and the chosen approach is fit on **all** of it before returning.
/// * If `train_dir` is `None`, the approach must be in [`UNFIT_OK`]; otherwise we
///   return an error explaining that it needs `--train`.
///
/// On success returns the fitted approach and a human-readable note describing
/// what was done (printed once at startup).
pub fn build(
    approach_name: &str,
    train_dir: Option<&Path>,
) -> Result<(Box<dyn Approach>, String), Box<dyn Error>> {
    let mut approach = instantiate(approach_name).ok_or_else(|| {
        format!(
            "unknown approach '{approach_name}'. available: {}",
            available_names().join(", ")
        )
    })?;

    match train_dir {
        Some(dir) => {
            let manifest = dir.join("labels.csv");
            let dataset = Dataset::load_csv(dir, &manifest).map_err(|e| {
                format!(
                    "failed to load training data from {}: {e}",
                    manifest.display()
                )
            })?;
            if dataset.is_empty() {
                return Err(format!("training dataset {} is empty", dir.display()).into());
            }
            let n = dataset.len();
            let n_pos = dataset.n_pos();
            approach.fit(&dataset.samples);
            let note = format!(
                "approach '{}' fit on {n} clips ({n_pos} drone / {} non-drone) from {}",
                approach.name(),
                n - n_pos,
                dir.display()
            );
            Ok((approach, note))
        }
        None => {
            if !UNFIT_OK.contains(&approach_name) {
                return Err(format!(
                    "approach '{approach_name}' is supervised and needs training; \
                     either pass --train <dir> (a folder with labels.csv, header \
                     path,label) or pick an unsupervised approach: {}",
                    UNFIT_OK.join(", ")
                )
                .into());
            }
            let note = format!(
                "approach '{}' running UNFIT (unsupervised / rule fallback — no labelled data at runtime)",
                approach.name()
            );
            Ok((approach, note))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_approach_is_in_unfit_allowlist() {
        assert!(UNFIT_OK.contains(&DEFAULT_APPROACH));
    }

    #[test]
    fn unfit_allowlist_names_all_exist() {
        let known = available_names();
        for name in UNFIT_OK {
            assert!(
                known.iter().any(|k| k == name),
                "{name} missing from registry"
            );
        }
    }

    #[test]
    fn unsupervised_builds_without_training() {
        let (a, _note) = build(DEFAULT_APPROACH, None).expect("hps should build unfit");
        assert_eq!(a.name(), DEFAULT_APPROACH);
    }

    #[test]
    fn supervised_without_training_is_rejected() {
        // `mfcc_lr` is supervised and not in the allowlist, so unfit use errors.
        let err = build("mfcc_lr", None);
        assert!(err.is_err(), "supervised approach should require --train");
    }

    #[test]
    fn unknown_approach_is_rejected() {
        assert!(build("does_not_exist", None).is_err());
    }
}
