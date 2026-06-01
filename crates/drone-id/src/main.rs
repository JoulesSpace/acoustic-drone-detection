//! `drone-id` binary: train and benchmark the multiclass drone-type recognizer.
//!
//! Usage:
//! ```text
//! drone-id --synth                       # deterministic synthetic 4-class set
//! drone-id --data <dir>                  # Al-Emadi Multiclass_Drone_Audio/ layout
//! ```
//! Writes `benchmarks/results/drone_id.json` (confusion matrix, per-class
//! metrics, id<->name map) and prints a readable summary + confusion matrix.

#![forbid(unsafe_code)]

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use serde::Serialize;

use drone_id::data::MultiDataset;
use drone_id::metrics::{evaluate, MulticlassReport};
use drone_id::model::SoftmaxClassifier;

#[derive(Parser, Debug)]
#[command(
    name = "drone-id",
    about = "Multiclass drone-type identification (the recognition head)."
)]
struct Cli {
    /// Path to a class-folder dataset root (e.g. Multiclass_Drone_Audio/).
    #[arg(long, value_name = "DIR", conflicts_with = "synth")]
    data: Option<PathBuf>,

    /// Use the deterministic synthetic 4-class dataset instead of real data.
    #[arg(long)]
    synth: bool,

    /// Clips per class for the synthetic generator.
    #[arg(long, default_value_t = 150)]
    synth_per_class: usize,

    /// Cap on clips drawn per real class (0 = no cap). Balances the skewed
    /// `unknown` folder and keeps runtime sane.
    #[arg(long, default_value_t = 600)]
    max_per_class: usize,

    /// Fraction of each class used for training.
    #[arg(long, default_value_t = 0.7)]
    train_frac: f32,

    /// RNG seed for the stratified split (and synthetic generation).
    #[arg(long, default_value_t = 42)]
    seed: u32,

    /// Where to write the JSON results.
    #[arg(long, default_value = "benchmarks/results/drone_id.json")]
    out: PathBuf,
}

/// The on-disk JSON document: the report plus run provenance.
#[derive(Serialize)]
struct ResultsJson {
    source: String,
    seed: u32,
    train_frac: f32,
    n_train: usize,
    sample_rate_note: String,
    train_secs: f64,
    #[serde(flatten)]
    report: MulticlassReport,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let (dataset, source) = if let Some(dir) = &cli.data {
        let ds = MultiDataset::load_dir(dir, cli.max_per_class)?;
        (ds, format!("real:{}", dir.display()))
    } else if cli.synth {
        let ds = MultiDataset::synth(cli.synth_per_class, 16_000, cli.seed);
        (ds, "synthetic".to_string())
    } else {
        return Err("choose a data source: --synth or --data <dir>".into());
    };

    println!("== drone-id : multiclass drone-type recognition ==");
    println!("source        : {source}");
    println!("classes ({}) : {}", dataset.n_classes(), {
        let mut s = String::new();
        for (i, name) in dataset.class_names.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("{i}={name}"));
        }
        s
    });

    // Note any clips that aren't at the 16 kHz the front-end was tuned for.
    let off_rate = dataset
        .samples
        .iter()
        .filter(|s| s.sample_rate != 16_000)
        .count();
    let sr_note = if off_rate == 0 {
        "all clips at 16 kHz".to_string()
    } else {
        format!(
            "{off_rate}/{} clips not at 16 kHz; feature pipeline still runs but bins shift",
            dataset.samples.len()
        )
    };
    println!("sample rate   : {sr_note}");

    let (train, test) = dataset.split(cli.train_frac, cli.seed);
    println!(
        "split         : {} train / {} test (stratified, seed {})",
        train.len(),
        test.len(),
        cli.seed
    );

    let mut clf = SoftmaxClassifier::new(dataset.n_classes());
    let t0 = Instant::now();
    clf.fit(&train);
    let train_secs = t0.elapsed().as_secs_f64();

    let pairs: Vec<(usize, usize)> = test
        .iter()
        .map(|s| (s.label as usize, clf.predict(&s.samples, s.sample_rate)))
        .collect();

    let report = evaluate(&pairs, dataset.n_classes(), &dataset.class_names);

    print_summary(&report);

    let results = ResultsJson {
        source,
        seed: cli.seed,
        train_frac: cli.train_frac,
        n_train: train.len(),
        sample_rate_note: sr_note,
        train_secs,
        report,
    };
    write_json(&cli.out, &results)?;
    println!("\nwrote {}", cli.out.display());
    Ok(())
}

/// Print a human-readable per-class table and the confusion matrix.
fn print_summary(r: &MulticlassReport) {
    println!("\n-- per-class metrics --");
    println!(
        "{:<16} {:>9} {:>9} {:>9} {:>8}",
        "class", "precision", "recall", "f1", "support"
    );
    for m in &r.per_class {
        println!(
            "{:<16} {:>9.3} {:>9.3} {:>9.3} {:>8}",
            truncate(&m.class_name, 16),
            m.precision,
            m.recall,
            m.f1,
            m.support
        );
    }
    println!(
        "\naccuracy = {:.4}   macro-F1 = {:.4}   weighted-F1 = {:.4}   (n_test = {})",
        r.accuracy, r.macro_f1, r.weighted_f1, r.n_test
    );

    println!("\n-- confusion matrix (rows = true, cols = predicted) --");
    print!("{:<16}", "true\\pred");
    for c in 0..r.n_classes {
        print!("{:>8}", c);
    }
    println!();
    for (i, row) in r.confusion.iter().enumerate() {
        print!("{:<16}", format!("{i} {}", truncate(&r.class_names[i], 13)));
        for &v in row {
            print!("{v:>8}");
        }
        println!();
    }
}

/// Truncate a string to `max` chars for table alignment.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
    }
}

/// Serialize `value` as pretty JSON, creating parent dirs as needed.
fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(value)?;
    std::fs::write(path, json)?;
    Ok(())
}
