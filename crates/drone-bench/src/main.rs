//! `drone-bench` — run every registered detection approach over a dataset and
//! write per-approach metric JSON for plotting.
//!
//! Data sources: `--synth` (a deterministic synthetic dataset, no files) or
//! `--data <dir>` (load `<dir>/labels.csv`, header `path,label`, label 0/1).
//!
//! Evaluation: a single stratified train/test split by default; `--kfold K` runs
//! K-fold CV with metrics on pooled out-of-fold predictions (each clip scored by
//! a model that didn't see it); `--snr <dB>` adds white noise to the test clips
//! before scoring (a robustness sweep).

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use drone_bench::approaches;
use drone_bench::dataset::{Dataset, Sample};
use drone_bench::metrics::{evaluate, pr_curve, roc_curve, ApproachResult, ScoreLabel};
use drone_bench::Approach;

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
    /// Fraction of each class used for training (single-split mode).
    #[arg(long, default_value_t = 0.5)]
    train_frac: f32,
    /// Decision threshold for the headline confusion metrics.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,
    /// K-fold cross-validation (pooled out-of-fold predictions). 1 = single split.
    #[arg(long, default_value_t = 1)]
    kfold: usize,
    /// If set, add white noise to TEST clips at this SNR (dB) — robustness eval.
    #[arg(long)]
    snr: Option<f32>,
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
    let kfold = cli.kfold.max(1);
    println!(
        "dataset: {} samples ({} pos){}{}",
        dataset.len(),
        dataset.n_pos(),
        if kfold > 1 {
            format!(" — {kfold}-fold CV")
        } else {
            format!(" — single split (train_frac {})", cli.train_frac)
        },
        match cli.snr {
            Some(db) => format!(", test noise @ {db} dB SNR"),
            None => String::new(),
        },
    );
    std::fs::create_dir_all(&cli.out_dir)?;

    println!(
        "\n{:<20} {:>6} {:>6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>9} {:>8}",
        "approach", "acc", "prec", "rec", "F1", "F1*", "ROC-AUC", "PR-AUC", "ms/clip", "xRT"
    );
    println!("{}", "-".repeat(92));

    // We iterate by (name, description) so we can re-instantiate a fresh model
    // per fold (fitting mutates state and trait objects aren't cloneable).
    let registry: Vec<(String, String)> = approaches::all()
        .iter()
        .map(|a| (a.name().to_string(), a.description().to_string()))
        .collect();

    for (name, description) in registry {
        if let Some(filter) = &cli.only {
            if !name.contains(filter.as_str()) {
                continue;
            }
        }

        let (scored, infer_secs, audio_secs) = if kfold > 1 {
            kfold_scored(&dataset, kfold, &name, cli.snr, cli.seed)
        } else {
            let (train, test) = dataset.split(cli.train_frac, cli.seed);
            let mut approach = instantiate(&name);
            approach.fit(&train);
            score_set(approach.as_ref(), &test, cli.snr)
        };
        let n = scored.len().max(1) as f64;
        let mean_infer_ms = infer_secs * 1000.0 / n;
        let ms_per_audio_sec = if audio_secs > 0.0 {
            infer_secs * 1000.0 / audio_secs
        } else {
            0.0
        };
        let realtime_factor = if audio_secs > 0.0 {
            infer_secs / audio_secs
        } else {
            0.0
        };

        let metrics = evaluate(&scored, cli.threshold);
        let n_pos = scored.iter().filter(|&&(_, y)| y == 1).count();
        let result = ApproachResult {
            approach: name.clone(),
            description: description.clone(),
            n_test: scored.len(),
            n_pos,
            n_neg: scored.len() - n_pos,
            mean_infer_ms,
            ms_per_audio_sec,
            realtime_factor,
            metrics: metrics.clone(),
            scores: scored.iter().map(|&(s, y)| ScoreLabel { s, y }).collect(),
            roc: roc_curve(&scored),
            pr: pr_curve(&scored),
        };
        std::fs::write(
            cli.out_dir.join(format!("{name}.json")),
            serde_json::to_string_pretty(&result)?,
        )?;

        // xRT = how many times faster than real time (1 / realtime_factor).
        let xrt = if realtime_factor > 0.0 {
            1.0 / realtime_factor
        } else {
            f64::INFINITY
        };
        println!(
            "{:<20} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>6.3} {:>8.3} {:>8.3} {:>9.3} {:>8.0}",
            name,
            metrics.accuracy,
            metrics.precision,
            metrics.recall,
            metrics.f1,
            metrics.f1_best,
            metrics.roc_auc,
            metrics.pr_auc,
            mean_infer_ms,
            xrt,
        );
    }

    println!("\nresults written to {}", cli.out_dir.display());
    Ok(())
}

/// Build a fresh instance of the named approach from the registry.
fn instantiate(name: &str) -> Box<dyn Approach> {
    approaches::all()
        .into_iter()
        .find(|a| a.name() == name)
        .unwrap_or_else(|| panic!("unknown approach {name}"))
}

/// Score a set of clips with a fitted approach. Returns the `(score, label)`
/// pairs, the total wall-clock time spent inside `score()` (seconds), and the
/// total audio duration scored (seconds) — the latter two give the real-time
/// factor. Noise injection (for `--snr`) is done outside the timed region so the
/// measurement reflects detector cost, not the eval harness.
fn score_set(
    approach: &dyn Approach,
    samples: &[Sample],
    snr: Option<f32>,
) -> (Vec<(f32, u8)>, f64, f64) {
    let mut scored = Vec::with_capacity(samples.len());
    let mut infer_secs = 0.0_f64;
    let mut audio_secs = 0.0_f64;
    for (i, s) in samples.iter().enumerate() {
        let noisy = snr.map(|db| add_noise(&s.samples, db, i as u32 + 1));
        let buf: &[f32] = noisy.as_deref().unwrap_or(&s.samples);
        let t0 = Instant::now();
        let conf = approach.score(buf, s.sample_rate).clamp(0.0, 1.0);
        infer_secs += t0.elapsed().as_secs_f64();
        audio_secs += s.samples.len() as f64 / s.sample_rate.max(1) as f64;
        scored.push((conf, s.label));
    }
    (scored, infer_secs, audio_secs)
}

/// K-fold CV producing pooled out-of-fold `(score, label)` predictions: each
/// sample is scored by a model fit on the other folds. Folds are stratified by
/// class via a seeded shuffle.
fn kfold_scored(
    ds: &Dataset,
    k: usize,
    name: &str,
    snr: Option<f32>,
    seed: u32,
) -> (Vec<(f32, u8)>, f64, f64) {
    // Assign each sample a fold id, balanced within each class.
    let mut fold_of = vec![0usize; ds.samples.len()];
    let mut rng = seed.max(1);
    for class in [0u8, 1u8] {
        let mut idx: Vec<usize> = ds
            .samples
            .iter()
            .enumerate()
            .filter(|(_, s)| s.label == class)
            .map(|(i, _)| i)
            .collect();
        for i in (1..idx.len()).rev() {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            let j = (rng as usize) % (i + 1);
            idx.swap(i, j);
        }
        for (rank, &i) in idx.iter().enumerate() {
            fold_of[i] = rank % k;
        }
    }

    let mut out = Vec::with_capacity(ds.samples.len());
    let mut infer_secs = 0.0_f64;
    let mut audio_secs = 0.0_f64;
    for f in 0..k {
        let train: Vec<Sample> = ds
            .samples
            .iter()
            .enumerate()
            .filter(|(i, _)| fold_of[*i] != f)
            .map(|(_, s)| s.clone())
            .collect();
        let test: Vec<Sample> = ds
            .samples
            .iter()
            .enumerate()
            .filter(|(i, _)| fold_of[*i] == f)
            .map(|(_, s)| s.clone())
            .collect();
        let mut approach = instantiate(name);
        approach.fit(&train);
        let (mut scored, inf, aud) = score_set(approach.as_ref(), &test, snr);
        out.append(&mut scored);
        infer_secs += inf;
        audio_secs += aud;
    }
    (out, infer_secs, audio_secs)
}

/// Add uniform white noise to a clip to hit a target SNR (dB) relative to the
/// clip's signal power. Deterministic given `seed`.
fn add_noise(x: &[f32], snr_db: f32, seed: u32) -> Vec<f32> {
    let n = x.len().max(1) as f32;
    let ps: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    if ps <= 0.0 {
        return x.to_vec();
    }
    let pn = ps / 10f32.powf(snr_db / 10.0);
    // Uniform in [-a, a] has variance a^2/3, so a = sqrt(3 * pn).
    let a = (3.0 * pn).sqrt();
    let mut st = seed.max(1);
    x.iter()
        .map(|&v| {
            st ^= st << 13;
            st ^= st >> 17;
            st ^= st << 5;
            let u = (st as f32 / u32::MAX as f32) * 2.0 - 1.0;
            v + a * u
        })
        .collect()
}
