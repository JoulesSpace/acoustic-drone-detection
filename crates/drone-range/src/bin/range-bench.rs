//! `range-bench` - validate acoustic distance estimation on the physics range
//! simulator.
//!
//! For each `(range, seed)` cell the simulator renders a drone clip as heard at
//! that distance (spherical spreading + frequency-dependent air absorption +
//! fixed ambient floor, so SNR falls with range). We frame each clip with
//! `drone_bench::util::spectra` (the project's shared front-end), extract
//! clip-level features, then:
//!
//!   * train **ridge regression** to predict range in metres, and
//!   * train a **multinomial-logistic** classifier over fixed-width range bins.
//!
//! Seeds are split into train / test, so test clips are genuinely held out.
//! We report distance **MAE (m)**, **median error (m)**, and **per-bin
//! accuracy**, plus a **level-only vs tilt-included ablation** that shows the
//! air-absorption tilt feature carries range information beyond raw loudness.
//!
//! Everything is deterministic: clip `(range, seed)` uses a seed derived from
//! the CLI `--seed`, the range index and the seed index.
//!
//! Honest caveats are printed and embedded in the JSON: the simulator is
//! idealized (no ground reflection, wind, turbulence, or drone-type loudness
//! variation), and real distance is confounded by source loudness and
//! environment. The literature band is ~61-98% depending on setup; distance is
//! the hardest of our property heads.

use std::error::Error;
use std::path::PathBuf;

use clap::Parser;
use serde::Serialize;

use drone_bench::util::spectra;
use drone_range::classify::{BinModel, RangeBins};
use drone_range::features::{clip_features, ClipFeatures, FeatureSet};
use drone_range::regress::RidgeModel;
use drone_range::sim::{simulate_clip, snr_db, RangeSimConfig, SourceConfig};

#[derive(Parser)]
#[command(
    name = "range-bench",
    version,
    about = "Benchmark acoustic distance estimation on a physics range simulator"
)]
struct Cli {
    /// Sample rate in Hz.
    #[arg(long, default_value_t = 16_000)]
    sample_rate: u32,
    /// Clip length in samples (16000 = 1 s @ 16 kHz).
    #[arg(long, default_value_t = 16_000)]
    num_samples: usize,
    /// Nearest range in the sweep (metres).
    #[arg(long, default_value_t = 10.0)]
    range_min: f32,
    /// Farthest range in the sweep (metres).
    #[arg(long, default_value_t = 200.0)]
    range_max: f32,
    /// Range sweep step (metres).
    #[arg(long, default_value_t = 10.0)]
    range_step: f32,
    /// Clips (distinct noise seeds) per range.
    #[arg(long, default_value_t = 24)]
    clips_per_range: usize,
    /// Fraction of clips (per range) used for training; the rest are held out.
    #[arg(long, default_value_t = 0.5)]
    train_frac: f32,
    /// Bin width for the classification mode (metres).
    #[arg(long, default_value_t = 20.0)]
    bin_width: f32,
    /// Ambient noise std (absolute; SNR falls with range as the signal shrinks).
    #[arg(long, default_value_t = 0.01)]
    noise_std: f32,
    /// Ridge regularization strength.
    #[arg(long, default_value_t = 1.0)]
    lambda: f32,
    /// Jitter (+/- fraction) applied to per-clip source loudness, modelling the
    /// real-world confounder that a louder drone looks closer. 0 disables it.
    #[arg(long, default_value_t = 0.3)]
    gain_jitter: f32,
    /// Base RNG seed.
    #[arg(long, default_value_t = 1)]
    seed: u32,
    /// Output JSON path.
    #[arg(long, default_value = "benchmarks/results/range.json")]
    out: PathBuf,
}

/// One simulated, feature-extracted clip with its ground-truth range.
struct Clip {
    range_m: f32,
    feats: ClipFeatures,
    is_train: bool,
}

#[derive(Serialize)]
struct AblationResult {
    feature_set: String,
    mae_m: f32,
    median_m: f32,
    bin_accuracy: f32,
}

#[derive(Serialize)]
struct PerBinAccuracy {
    bin: usize,
    lo_m: f32,
    hi_m: f32,
    accuracy: f32,
    n: usize,
}

#[derive(Serialize)]
struct PerRangeError {
    range_m: f32,
    mae_m: f32,
    n: usize,
    snr_db: f32,
}

#[derive(Serialize)]
struct TestClip {
    range_true_m: f32,
    range_pred_m: f32,
    abs_error_m: f32,
    bin_true: usize,
    bin_pred: usize,
}

#[derive(Serialize)]
struct RangeResult {
    sample_rate: u32,
    num_samples: usize,
    range_min_m: f32,
    range_max_m: f32,
    range_step_m: f32,
    clips_per_range: usize,
    train_frac: f32,
    bin_width_m: f32,
    n_bins: usize,
    noise_std: f32,
    lambda: f32,
    gain_jitter: f32,
    n_train: usize,
    n_test: usize,
    // Regression (full feature set).
    regression_mae_m: f32,
    regression_median_m: f32,
    // Classification (full feature set).
    bin_accuracy: f32,
    // Ablation across feature sets (level-only vs tilt vs full).
    ablation: Vec<AblationResult>,
    per_bin: Vec<PerBinAccuracy>,
    per_range: Vec<PerRangeError>,
    literature_band: String,
    caveats: Vec<String>,
    test_clips: Vec<TestClip>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let ranges = sweep(cli.range_min, cli.range_max, cli.range_step);
    let bins = RangeBins::new(cli.bin_width, cli.range_max + cli.range_step);
    let n_train_per_range = ((cli.clips_per_range as f32) * cli.train_frac).round() as usize;

    println!(
        "range sim: {:.0}..{:.0} m step {:.0} m  x  {} clips/range  ({} train / {} test per range)",
        cli.range_min,
        cli.range_max,
        cli.range_step,
        cli.clips_per_range,
        n_train_per_range,
        cli.clips_per_range - n_train_per_range,
    );
    println!(
        "bins: {} x {:.0} m  |  noise_std {}  |  gain_jitter +/-{:.0}%  |  {} samples @ {} Hz\n",
        bins.n_bins,
        cli.bin_width,
        cli.noise_std,
        cli.gain_jitter * 100.0,
        cli.num_samples,
        cli.sample_rate,
    );

    // ---- Generate every clip deterministically and extract features. --------
    let base_src = SourceConfig::default();
    let mut clips: Vec<Clip> = Vec::new();
    // Per-range true SNR (uses unit gain; diagnostic only).
    let mut per_range_snr: Vec<(f32, f32)> = Vec::new();

    for (ri, &range_m) in ranges.iter().enumerate() {
        let diag_cfg = RangeSimConfig {
            sample_rate: cli.sample_rate,
            num_samples: cli.num_samples,
            range_m,
            noise_std: cli.noise_std,
            seed: 1,
            ..Default::default()
        };
        per_range_snr.push((range_m, snr_db(&base_src, &diag_cfg)));

        for c in 0..cli.clips_per_range {
            let seed = cli
                .seed
                .wrapping_mul(1_000_003)
                .wrapping_add((ri as u32).wrapping_mul(9_973))
                .wrapping_add(c as u32 + 1);

            // Deterministic per-clip loudness jitter (the confounder).
            let gain = if cli.gain_jitter > 0.0 {
                let u = unit_from_seed(seed.wrapping_mul(2_246_822_519));
                1.0 + cli.gain_jitter * (2.0 * u - 1.0)
            } else {
                1.0
            };
            let mut src = base_src.clone();
            src.source_gain = gain;

            let cfg = RangeSimConfig {
                sample_rate: cli.sample_rate,
                num_samples: cli.num_samples,
                range_m,
                noise_std: cli.noise_std,
                seed,
                ..Default::default()
            };
            let samples = simulate_clip(&src, &cfg);
            let frames = spectra(&samples);
            let feats = clip_features(&frames, cli.sample_rate);
            clips.push(Clip {
                range_m,
                feats,
                is_train: c < n_train_per_range,
            });
        }
    }

    let n_train = clips.iter().filter(|c| c.is_train).count();
    let n_test = clips.len() - n_train;

    // ---- Ablation: level-only vs level+tilt vs full. ------------------------
    let sets = [
        FeatureSet::LevelOnly,
        FeatureSet::LevelPlusTilt,
        FeatureSet::Full,
    ];
    let mut ablation = Vec::new();
    // Hold the full-set test predictions for the detailed per-bin/per-range tables.
    let mut full_test_preds: Vec<(f32, f32)> = Vec::new(); // (true, pred)

    for &set in &sets {
        let (mae, median, bin_acc, test_preds) = eval_set(&clips, set, cli.lambda, &bins);
        ablation.push(AblationResult {
            feature_set: set.name().to_string(),
            mae_m: mae,
            median_m: median,
            bin_accuracy: bin_acc,
        });
        if set == FeatureSet::Full {
            full_test_preds = test_preds;
        }
    }

    // ---- Detailed tables from the FULL-set test predictions. ----------------
    let mut per_bin_correct = vec![0usize; bins.n_bins];
    let mut per_bin_total = vec![0usize; bins.n_bins];
    let mut per_range_abs: Vec<(f32, Vec<f32>)> = ranges.iter().map(|&r| (r, Vec::new())).collect();
    let mut test_clips = Vec::new();

    for &(truth, pred) in &full_test_preds {
        let bt = bins.index_of(truth);
        let bp = bins.index_of(pred);
        per_bin_total[bt] += 1;
        if bt == bp {
            per_bin_correct[bt] += 1;
        }
        if let Some(slot) = per_range_abs
            .iter_mut()
            .find(|(r, _)| (*r - truth).abs() < 1e-3)
        {
            slot.1.push((pred - truth).abs());
        }
        test_clips.push(TestClip {
            range_true_m: truth,
            range_pred_m: pred,
            abs_error_m: (pred - truth).abs(),
            bin_true: bt,
            bin_pred: bp,
        });
    }

    let per_bin: Vec<PerBinAccuracy> = (0..bins.n_bins)
        .map(|b| {
            let (lo, hi) = bins.bounds(b);
            let n = per_bin_total[b];
            PerBinAccuracy {
                bin: b,
                lo_m: lo,
                hi_m: hi,
                accuracy: if n > 0 {
                    per_bin_correct[b] as f32 / n as f32
                } else {
                    0.0
                },
                n,
            }
        })
        .collect();

    let snr_lookup = |r: f32| -> f32 {
        per_range_snr
            .iter()
            .find(|(rr, _)| (*rr - r).abs() < 1e-3)
            .map(|(_, s)| *s)
            .unwrap_or(0.0)
    };
    let per_range: Vec<PerRangeError> = per_range_abs
        .iter()
        .map(|(r, errs)| PerRangeError {
            range_m: *r,
            mae_m: mae(errs),
            n: errs.len(),
            snr_db: snr_lookup(*r),
        })
        .collect();

    let reg_full = ablation
        .iter()
        .find(|a| a.feature_set == "full")
        .expect("full set evaluated");

    // ---- Print tables. ------------------------------------------------------
    println!("== ablation: distance MAE / median / bin-accuracy by feature set ==");
    println!(
        "{:>12}  {:>9}  {:>9}  {:>10}",
        "feature set", "MAE(m)", "median(m)", "bin-acc"
    );
    println!("{}", "-".repeat(46));
    for a in &ablation {
        println!(
            "{:>12}  {:>9.2}  {:>9.2}  {:>9.1}%",
            a.feature_set,
            a.mae_m,
            a.median_m,
            a.bin_accuracy * 100.0
        );
    }
    println!("{}", "-".repeat(46));
    println!(
        "tilt adds {:.2} m MAE improvement over level-only ({:.2} -> {:.2})\n",
        ablation[0].mae_m - reg_full.mae_m,
        ablation[0].mae_m,
        reg_full.mae_m,
    );

    println!("== per-bin accuracy (full feature set) ==");
    println!("{:>4}  {:>10}  {:>9}  {:>6}", "bin", "range(m)", "acc", "n");
    println!("{}", "-".repeat(36));
    for b in &per_bin {
        let hi = if b.hi_m.is_finite() {
            format!("{:.0}", b.hi_m)
        } else {
            "inf".to_string()
        };
        println!(
            "{:>4}  {:>4.0}-{:<5}  {:>8.1}%  {:>6}",
            b.bin,
            b.lo_m,
            hi,
            b.accuracy * 100.0,
            b.n
        );
    }
    println!();

    println!("== per-range distance error (full feature set) ==");
    println!(
        "{:>9}  {:>9}  {:>9}  {:>6}",
        "range(m)", "MAE(m)", "SNR(dB)", "n"
    );
    println!("{}", "-".repeat(40));
    for p in &per_range {
        println!(
            "{:>9.0}  {:>9.2}  {:>9.1}  {:>6}",
            p.range_m, p.mae_m, p.snr_db, p.n
        );
    }
    println!("{}", "-".repeat(40));
    println!(
        "overall: MAE {:.2} m | median {:.2} m | bin-accuracy {:.1}%",
        reg_full.mae_m,
        reg_full.median_m,
        reg_full.bin_accuracy * 100.0
    );

    let caveats = vec![
        "Simulator is idealized: no ground reflection, wind, turbulence, or \
         drone-type loudness variation."
            .to_string(),
        "Real distance is confounded by source loudness and environment; a \
         louder/closer-looking drone confounds level."
            .to_string(),
        "Air-absorption uses a classical f^2 law (documented constants), not the \
         full ISO 9613-1 humidity/relaxation model."
            .to_string(),
        "Distance is the hardest of our property heads.".to_string(),
        "No distance-labeled real dataset was used; validation is simulator-only \
         (analogous to drone-doa)."
            .to_string(),
    ];
    let literature_band =
        "audio-only distance ~61-98% depending on setup (Kim 2023 drone-to-drone \
         61-78%; Kang ground-array 94-98% on 5-50 m bins)"
            .to_string();

    println!("\nliterature band: {literature_band}");
    println!("caveats:");
    for c in &caveats {
        println!("  - {c}");
    }

    let result = RangeResult {
        sample_rate: cli.sample_rate,
        num_samples: cli.num_samples,
        range_min_m: cli.range_min,
        range_max_m: cli.range_max,
        range_step_m: cli.range_step,
        clips_per_range: cli.clips_per_range,
        train_frac: cli.train_frac,
        bin_width_m: cli.bin_width,
        n_bins: bins.n_bins,
        noise_std: cli.noise_std,
        lambda: cli.lambda,
        gain_jitter: cli.gain_jitter,
        n_train,
        n_test,
        regression_mae_m: reg_full.mae_m,
        regression_median_m: reg_full.median_m,
        bin_accuracy: reg_full.bin_accuracy,
        ablation,
        per_bin,
        per_range,
        literature_band,
        caveats,
        test_clips,
    };

    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out, serde_json::to_string_pretty(&result)?)?;
    println!("\nresults written to {}", cli.out.display());
    Ok(())
}

/// Train both heads on the train split for `set`, evaluate on the held-out
/// test split, return `(mae, median, bin_accuracy, test_(true,pred))`.
fn eval_set(
    clips: &[Clip],
    set: FeatureSet,
    lambda: f32,
    bins: &RangeBins,
) -> (f32, f32, f32, Vec<(f32, f32)>) {
    // Build train matrices.
    let mut x_train: Vec<Vec<f32>> = Vec::new();
    let mut y_train: Vec<f32> = Vec::new();
    let mut lbl_train: Vec<usize> = Vec::new();
    for c in clips.iter().filter(|c| c.is_train) {
        x_train.push(c.feats.select(set));
        y_train.push(c.range_m);
        lbl_train.push(bins.index_of(c.range_m));
    }

    let reg = RidgeModel::fit(&x_train, &y_train, lambda);
    let clf = BinModel::fit(&x_train, &lbl_train, bins.n_bins);

    let mut abs_errs: Vec<f32> = Vec::new();
    let mut correct = 0usize;
    let mut total = 0usize;
    let mut test_preds: Vec<(f32, f32)> = Vec::new();

    for c in clips.iter().filter(|c| !c.is_train) {
        let feat = c.feats.select(set);
        let pred = reg.predict(&feat);
        abs_errs.push((pred - c.range_m).abs());
        let bp = clf.predict(&feat);
        let bt = bins.index_of(c.range_m);
        if bp == bt {
            correct += 1;
        }
        total += 1;
        test_preds.push((c.range_m, pred));
    }

    let bin_acc = if total > 0 {
        correct as f32 / total as f32
    } else {
        0.0
    };
    (mae(&abs_errs), median(&abs_errs), bin_acc, test_preds)
}

/// Mean absolute value of a slice (already-absolute errors), 0 if empty.
fn mae(errs: &[f32]) -> f32 {
    if errs.is_empty() {
        return 0.0;
    }
    errs.iter().map(|e| e.abs()).sum::<f32>() / errs.len() as f32
}

/// Median of absolute errors (0 if empty). Sorts a copy; deterministic.
fn median(errs: &[f32]) -> f32 {
    if errs.is_empty() {
        return 0.0;
    }
    let mut v: Vec<f32> = errs.iter().map(|e| e.abs()).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Deterministic uniform `(0,1)` from a seed (one xorshift step), for the
/// loudness jitter so the benchmark stays reproducible.
fn unit_from_seed(seed: u32) -> f32 {
    let mut x = seed.max(1);
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    (x as f32 + 1.0) / (u32::MAX as f32 + 2.0)
}

/// Inclusive sweep `start..=stop` by `step` (handles float accumulation).
fn sweep(start: f32, stop: f32, step: f32) -> Vec<f32> {
    let mut out = Vec::new();
    if step <= 0.0 {
        out.push(start);
        return out;
    }
    let n = ((stop - start) / step).round() as i32;
    for i in 0..=n.max(0) {
        out.push(start + i as f32 * step);
    }
    out
}
