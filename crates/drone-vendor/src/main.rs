//! `drone-vendor` binary: train and benchmark the multi-vendor brand/model
//! recognizer.
//!
//! Usage:
//! ```text
//! drone-vendor --synth-brands             # deterministic synthetic 12-brand set
//! drone-vendor --data <dir>               # real audio: class folders OR a flat
//!                                          # directory of brand-named WAVs
//! ```
//! Writes `benchmarks/results/vendor.json` (confusion matrix, per-class metrics,
//! id<->brand map, run provenance) and prints a readable per-class table and the
//! confusion matrix. Clearly reports which data source was used.

#![forbid(unsafe_code)]

use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use serde::Serialize;

use drone_vendor::data::{VendorDataset, DEFAULT_SEGMENT_SECS};
use drone_vendor::metrics::{evaluate, MulticlassReport};
use drone_vendor::model::SoftmaxClassifier;

#[derive(Parser, Debug)]
#[command(
    name = "drone-vendor",
    about = "Multi-vendor drone brand/model recognition (the recognition head, scaled to many vendors)."
)]
struct Cli {
    /// Path to a real dataset root: either class subfolders (one per brand) or a
    /// flat directory of brand-named WAVs (e.g. DJI_Mavic2pro_81.wav).
    #[arg(long, value_name = "DIR", conflicts_with = "synth_brands")]
    data: Option<PathBuf>,

    /// Use the deterministic synthetic 12-brand dataset instead of real data.
    #[arg(long)]
    synth_brands: bool,

    /// Clips per brand for the synthetic generator.
    #[arg(long, default_value_t = 120)]
    synth_per_class: usize,

    /// Cap on clips drawn per real class for the class-folder layout (0 = none).
    #[arg(long, default_value_t = 0)]
    max_per_class: usize,

    /// Segment length (seconds) used to window single-clip-per-brand flat
    /// datasets into multiple labelled examples.
    #[arg(long, default_value_t = DEFAULT_SEGMENT_SECS)]
    segment_secs: f32,

    /// Drop the spectral/harmonic descriptors and use the MFCC block only.
    #[arg(long)]
    mfcc_only: bool,

    /// Fraction of each class used for training.
    #[arg(long, default_value_t = 0.7)]
    train_frac: f32,

    /// RNG seed for the stratified split (and synthetic generation).
    #[arg(long, default_value_t = 42)]
    seed: u32,

    /// Where to write the JSON results.
    #[arg(long, default_value = "benchmarks/results/vendor.json")]
    out: PathBuf,
}

/// The on-disk JSON document: the report plus run provenance.
#[derive(Serialize)]
struct ResultsJson {
    source: String,
    feature_set: String,
    n_features: usize,
    seed: u32,
    train_frac: f32,
    n_train: usize,
    sample_rate_note: String,
    caveat: String,
    train_secs: f64,
    #[serde(flatten)]
    report: MulticlassReport,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let use_spectral = !cli.mfcc_only;

    let (dataset, source, caveat) = if let Some(dir) = &cli.data {
        let ds = VendorDataset::load_dir(dir, cli.max_per_class, cli.segment_secs)?;
        let caveat = format!(
            "real audio; flat single-recording-per-brand sets are segment-windowed at {}s, \
             so train/test segments may share a recording (within-recording separability, \
             not cross-recording generalization)",
            cli.segment_secs
        );
        (ds, format!("real:{}", dir.display()), caveat)
    } else if cli.synth_brands {
        let ds = VendorDataset::synth(cli.synth_per_class, 16_000, cli.seed);
        (
            ds,
            "synthetic-12-brand".to_string(),
            "synthetic generator: a deterministic capability/architecture proof, NOT real-world \
             accuracy"
                .to_string(),
        )
    } else {
        return Err("choose a data source: --synth-brands or --data <dir>".into());
    };

    let feature_set = if use_spectral {
        "mfcc+spectral_harmonic"
    } else {
        "mfcc_only"
    };

    println!("== drone-vendor : multi-vendor brand/model recognition ==");
    println!("source        : {source}");
    println!("feature set   : {feature_set}");
    println!("brands ({})   :", dataset.n_classes());
    for (i, name) in dataset.class_names.iter().enumerate() {
        println!("                {i:>3} = {name}");
    }

    // Note clips that aren't at 16 kHz the front-end was tuned for. The mel
    // filterbank adapts to the sample rate, but bin spacing still shifts.
    let off_rate = dataset
        .samples
        .iter()
        .filter(|s| s.sample_rate != 16_000)
        .count();
    let sr_note = if off_rate == 0 {
        "all clips at 16 kHz".to_string()
    } else {
        format!(
            "{off_rate}/{} clips not at 16 kHz; feature pipeline adapts to sr but bins shift",
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

    let mut clf = SoftmaxClassifier::new(dataset.n_classes(), use_spectral);
    let t0 = Instant::now();
    clf.fit(&train);
    let train_secs = t0.elapsed().as_secs_f64();

    let pairs: Vec<(usize, usize)> = test
        .iter()
        .map(|s| (s.label as usize, clf.predict(&s.samples, s.sample_rate)))
        .collect();

    let report = evaluate(&pairs, dataset.n_classes(), &dataset.class_names);

    print_summary(&report);
    println!("\ncaveat        : {caveat}");

    let results = ResultsJson {
        source,
        feature_set: feature_set.to_string(),
        n_features: clf.n_feat(),
        seed: cli.seed,
        train_frac: cli.train_frac,
        n_train: train.len(),
        sample_rate_note: sr_note,
        caveat,
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
        "{:<22} {:>9} {:>9} {:>9} {:>8}",
        "brand", "precision", "recall", "f1", "support"
    );
    for m in &r.per_class {
        println!(
            "{:<22} {:>9.3} {:>9.3} {:>9.3} {:>8}",
            truncate(&m.class_name, 22),
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
    print!("{:<22}", "true\\pred");
    for c in 0..r.n_classes {
        print!("{c:>5}");
    }
    println!();
    for (i, row) in r.confusion.iter().enumerate() {
        print!("{:<22}", format!("{i} {}", truncate(&r.class_names[i], 18)));
        for &v in row {
            print!("{v:>5}");
        }
        println!();
    }
}

/// Truncate a string to `max` chars for table alignment.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        s.chars().take(max.saturating_sub(1)).collect::<String>() + "."
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
