//! `drone-bench` — run every registered detection approach over a dataset and
//! write per-approach metric JSON for plotting.
//!
//! Two data sources:
//!   * `--synth`        — generate a deterministic synthetic dataset (no files).
//!   * `--data <dir>`   — load `<dir>/labels.csv` (header `path,label`, label 0/1).

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use drone_bench::approaches;
use drone_bench::dataset::Dataset;
use drone_bench::metrics::{evaluate, pr_curve, roc_curve, ApproachResult, ScoreLabel};

#[derive(Parser)]
#[command(
    name = "drone-bench",
    version,
    about = "Benchmark drone-detection approaches"
)]
struct Cli {
    /// Use a synthetic dataset instead of files.
    #[arg(long)]
    synth: bool,
    /// Samples per class for `--synth`.
    #[arg(long, default_value_t = 200)]
    n: usize,
    /// Dataset root containing `labels.csv` (use instead of `--synth`).
    #[arg(long)]
    data: Option<PathBuf>,
    /// Manifest filename inside `--data`.
    #[arg(long, default_value = "labels.csv")]
    manifest: String,
    /// Fraction of each class used for training.
    #[arg(long, default_value_t = 0.5)]
    train_frac: f32,
    /// Decision threshold for the headline confusion metrics.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,
    /// RNG seed (synth generation and split).
    #[arg(long, default_value_t = 1)]
    seed: u32,
    /// Sample rate for synthetic data.
    #[arg(long, default_value_t = 16_000)]
    sample_rate: u32,
    /// Output directory for `<approach>.json`.
    #[arg(long, default_value = "benchmarks/results")]
    out_dir: PathBuf,
    /// Only run approaches whose name contains this substring.
    #[arg(long)]
    only: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let dataset = if let Some(dir) = &cli.data {
        let manifest = dir.join(&cli.manifest);
        println!("loading dataset from {}", manifest.display());
        Dataset::load_csv(dir, &manifest)?
    } else if cli.synth {
        println!("generating synthetic dataset: {} per class", cli.n);
        Dataset::synth(cli.n, cli.sample_rate, cli.seed)
    } else {
        return Err("specify --synth or --data <dir>".into());
    };

    if dataset.is_empty() {
        return Err("dataset is empty".into());
    }
    let (train, test) = dataset.split(cli.train_frac, cli.seed);
    let test_pos = test.iter().filter(|s| s.label == 1).count();
    println!(
        "dataset: {} samples ({} pos) → train {}, test {} ({} pos)",
        dataset.len(),
        dataset.n_pos(),
        train.len(),
        test.len(),
        test_pos,
    );
    std::fs::create_dir_all(&cli.out_dir)?;

    println!(
        "\n{:<16} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>9}",
        "approach", "acc", "prec", "rec", "F1", "ROC-AUC", "PR-AUC", "ms/clip"
    );
    println!("{}", "-".repeat(72));

    for mut approach in approaches::all() {
        if let Some(filter) = &cli.only {
            if !approach.name().contains(filter.as_str()) {
                continue;
            }
        }
        approach.fit(&train);

        // Time scoring across the test split.
        let start = Instant::now();
        let mut scored: Vec<(f32, u8)> = Vec::with_capacity(test.len());
        for s in &test {
            let conf = approach.score(&s.samples, s.sample_rate);
            debug_assert!(
                conf.is_finite() && (0.0..=1.0).contains(&conf),
                "{} returned out-of-range score {conf}",
                approach.name()
            );
            scored.push((conf.clamp(0.0, 1.0), s.label));
        }
        let mean_infer_ms = start.elapsed().as_secs_f64() * 1000.0 / test.len().max(1) as f64;

        let metrics = evaluate(&scored, cli.threshold);
        let n_pos = scored.iter().filter(|&&(_, y)| y == 1).count();
        let result = ApproachResult {
            approach: approach.name().to_string(),
            description: approach.description().to_string(),
            n_test: scored.len(),
            n_pos,
            n_neg: scored.len() - n_pos,
            mean_infer_ms,
            metrics: metrics.clone(),
            scores: scored.iter().map(|&(s, y)| ScoreLabel { s, y }).collect(),
            roc: roc_curve(&scored),
            pr: pr_curve(&scored),
        };

        let path = cli.out_dir.join(format!("{}.json", approach.name()));
        std::fs::write(&path, serde_json::to_string_pretty(&result)?)?;

        println!(
            "{:<16} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>8.3} {:>8.3} {:>9.3}",
            approach.name(),
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.roc_auc,
            metrics.pr_auc,
            mean_infer_ms,
        );
    }

    println!("\nresults written to {}", cli.out_dir.display());
    Ok(())
}
