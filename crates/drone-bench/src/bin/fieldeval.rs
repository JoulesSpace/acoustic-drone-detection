//! `fieldeval` - the genuinely held-out field evaluation.
//!
//! Every other number in this repo is at risk of recording-level leakage: DADS
//! clips cut from one recording can land in both train and test, so a detector
//! can "recognize the recording" rather than "recognize a drone". `xeval`
//! mitigates this by testing cross-dataset (Al-Emadi + ESC-50), but those are
//! still public corpora.
//!
//! This binary closes the last gap. It FITs every `drone_bench::approaches::all()`
//! detector on the full DADS dataset, then TESTs on a FIELD set the owner records
//! themselves with [`drone-live record`] - the owner's real drone, the owner's
//! own mics (phone + laptop), at known distances/azimuths, against real hard
//! negatives (cars, power tools, wind, aircraft, music). Because the field set
//! shares NO recordings with DADS, the resulting ROC-AUC / calibrated-F1 is the
//! honest, leakage-free "does this actually detect MY drone in MY environment"
//! number - the only basis on which "beats upstream" can be claimed truthfully.
//!
//! ```text
//!   1. drone-live record --label drone    --seconds 120   # writes data/field/drone/*.wav
//!   2. drone-live record --label nondrone --seconds 120   # writes data/field/nondrone/*.wav
//!   3. cargo run --bin fieldeval -- --field data/field    # this binary
//! ```
//!
//! The field set is loaded from a `labels.csv` dir exactly as the recorder writes
//! it (header `path,label`; resolved relative to the dir), reusing
//! `drone_bench::dataset::Dataset::load_csv`. DADS and field are both 16 kHz, so
//! no resampling is needed (the recorder already targets 16 kHz). Reuses the
//! harness contract verbatim: `drone_bench::{approaches, dataset, metrics}`.

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use serde::Serialize;

use drone_bench::approaches;
use drone_bench::dataset::Dataset;
use drone_bench::metrics::evaluate;
use drone_bench::Approach;

#[derive(Parser)]
#[command(
    name = "fieldeval",
    version,
    about = "Held-out FIELD evaluation: fit on DADS, test on owner-recorded field clips"
)]
struct Cli {
    /// DADS dataset root containing `labels.csv` (the TRAIN source).
    #[arg(long, default_value = "data/dads")]
    dads: PathBuf,
    /// DADS manifest filename inside `--dads`.
    #[arg(long, default_value = "labels.csv")]
    dads_manifest: String,
    /// Field set root containing `labels.csv` (header `path,label`), as written
    /// by `drone-live record`. This is the held-out TEST set.
    #[arg(long, default_value = "data/field")]
    field: PathBuf,
    /// Field manifest filename inside `--field`.
    #[arg(long, default_value = "labels.csv")]
    field_manifest: String,
    /// Fixed decision threshold for the headline confusion + fixed-threshold F1.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,
    /// Output JSON path.
    #[arg(long, default_value = "benchmarks/results/fieldeval.json")]
    out: PathBuf,
}

/// Per-approach held-out field metrics.
#[derive(Serialize)]
struct ApproachFieldEval {
    approach: String,
    description: String,
    /// Threshold-free ranking quality on the field set.
    field_roc_auc: f32,
    field_pr_auc: f32,
    /// F1 at the fixed `--threshold`.
    field_f1: f32,
    /// Calibrated (best-threshold) F1 and the threshold achieving it.
    field_f1_best: f32,
    field_threshold_best: f32,
    field_precision: f32,
    field_recall: f32,
    field_accuracy: f32,
    field_brier: f32,
    mean_infer_ms: f64,
}

/// Top-level report serialized to JSON.
#[derive(Serialize)]
struct FieldEvalReport {
    description: &'static str,
    threshold: f32,
    n_dads_train: usize,
    n_field_test: usize,
    n_field_pos: usize,
    n_field_neg: usize,
    approaches: Vec<ApproachFieldEval>,
}

/// Score a slice of samples, returning `(score, label)` pairs and total infer secs.
fn score_samples(
    approach: &dyn Approach,
    samples: &[drone_bench::Sample],
) -> (Vec<(f32, u8)>, f64) {
    let mut scored = Vec::with_capacity(samples.len());
    let mut infer_secs = 0.0_f64;
    for s in samples {
        let t0 = Instant::now();
        let conf = approach.score(&s.samples, s.sample_rate).clamp(0.0, 1.0);
        infer_secs += t0.elapsed().as_secs_f64();
        scored.push((conf, s.label));
    }
    (scored, infer_secs)
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // --- TEST set: the held-out field recordings. Bail early with guidance if
    //     the owner hasn't recorded anything yet. ---
    let field_manifest = cli.field.join(&cli.field_manifest);
    if !field_manifest.exists() {
        println!(
            "no field data found at {} - record some with `drone-live record`:\n\
             \n    drone-live record --label drone    --seconds 120\n    \
             drone-live record --label nondrone --seconds 120\n\
             \nsee FIELD_PROTOCOL.md for the full capture protocol.",
            field_manifest.display()
        );
        return Ok(());
    }
    let field = Dataset::load_csv(&cli.field, &field_manifest)?;
    if field.is_empty() {
        println!(
            "field set at {} is empty - record some with `drone-live record` \
             (see FIELD_PROTOCOL.md).",
            cli.field.display()
        );
        return Ok(());
    }
    let n_field_pos = field.n_pos();
    let n_field_neg = field.len() - n_field_pos;
    if n_field_pos == 0 || n_field_neg == 0 {
        println!(
            "field set needs BOTH classes for a meaningful eval (have {} drone / {} nondrone). \
             Record the missing class with `drone-live record`.",
            n_field_pos, n_field_neg
        );
        return Ok(());
    }

    // --- TRAIN source: all of DADS (no split - the field set is the test). ---
    let dads_manifest = cli.dads.join(&cli.dads_manifest);
    println!(
        "loading DADS (train source) from {}",
        dads_manifest.display()
    );
    let dads = Dataset::load_csv(&cli.dads, &dads_manifest)?;
    if dads.is_empty() {
        return Err("DADS dataset is empty (check --dads)".into());
    }
    let train = dads.samples;
    println!(
        "DADS: {} train clips. FIELD: {} test clips ({} drone / {} nondrone).",
        train.len(),
        field.len(),
        n_field_pos,
        n_field_neg,
    );

    let registry: Vec<(String, String)> = approaches::all()
        .iter()
        .map(|a| (a.name().to_string(), a.description().to_string()))
        .collect();

    println!("\n=== Held-out FIELD evaluation (DADS-train -> field test) ===");
    println!(
        "{:<20} {:>9} {:>9} {:>7} {:>7} {:>7} {:>7}",
        "approach", "ROC-AUC", "PR-AUC", "F1", "F1*", "prec", "recall"
    );
    println!("{}", "-".repeat(74));

    let mut results: Vec<ApproachFieldEval> = Vec::new();
    for (name, description) in &registry {
        let mut approach = instantiate(name);
        approach.fit(&train);

        let (scored, infer_secs) = score_samples(approach.as_ref(), &field.samples);
        let m = evaluate(&scored, cli.threshold);
        let mean_infer_ms = infer_secs * 1000.0 / scored.len().max(1) as f64;

        println!(
            "{:<20} {:>9.3} {:>9.3} {:>7.3} {:>7.3} {:>7.3} {:>7.3}",
            name, m.roc_auc, m.pr_auc, m.f1, m.f1_best, m.precision, m.recall,
        );

        results.push(ApproachFieldEval {
            approach: name.clone(),
            description: description.clone(),
            field_roc_auc: m.roc_auc,
            field_pr_auc: m.pr_auc,
            field_f1: m.f1,
            field_f1_best: m.f1_best,
            field_threshold_best: m.threshold_best,
            field_precision: m.precision,
            field_recall: m.recall,
            field_accuracy: m.accuracy,
            field_brier: m.brier,
            mean_infer_ms,
        });
    }

    let report = FieldEvalReport {
        description: "Held-out FIELD eval. FIT on DADS; TEST on owner-recorded field clips \
                      (drone-live record). The field set shares NO recordings with DADS, so \
                      these ROC-AUC / calibrated-F1 numbers are leakage-free.",
        threshold: cli.threshold,
        n_dads_train: train.len(),
        n_field_test: field.len(),
        n_field_pos,
        n_field_neg,
        approaches: results,
    };
    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out, serde_json::to_string_pretty(&report)?)?;
    println!("\nresults written to {}", cli.out.display());

    Ok(())
}

/// Instantiate an approach by name from the registry.
fn instantiate(name: &str) -> Box<dyn Approach> {
    approaches::all()
        .into_iter()
        .find(|a| a.name() == name)
        .unwrap_or_else(|| panic!("unknown approach {name}"))
}
