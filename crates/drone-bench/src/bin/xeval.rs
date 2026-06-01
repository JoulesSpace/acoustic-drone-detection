//! `xeval` — leakage-honest CROSS-DATASET + HARD-NEGATIVE evaluation.
//!
//! Our in-distribution DADS numbers (~F1 1.0) are almost certainly inflated by
//! recording-level leakage: clips cut from the same recording land in both the
//! train and test split, so a detector can "recognize the recording" instead of
//! "recognizing a drone". This binary measures the two things that leakage can't
//! fake:
//!
//! * **Experiment A — cross-dataset detection.** FIT every approach on the DADS
//!   *train* split, then TEST on a disjoint corpus: Al-Emadi drone clips as the
//!   positives and ESC-50 confusable classes as the negatives. Different
//!   microphones, rooms, drones, and noise — so a high score here is real
//!   generalization, not memorized recordings. We report ROC-AUC, fixed-threshold
//!   F1, and calibrated (best-threshold) F1, and compare against the
//!   in-distribution DADS split so the drop is visible and honest.
//!
//! * **Experiment B — hard-negative confusion.** At one fixed operating
//!   threshold, for the top detectors, report the false-positive rate (fraction
//!   of clips called "drone") per ESC-50 class. This exposes *which* sounds fool
//!   *which* methods: harmonic/rotor confounders (helicopter, airplane, engine,
//!   chainsaw) vs. broadband ones (wind, crackling_fire) vs. unrelated controls.
//!
//! Reuses the harness contract verbatim: `drone_bench::{approaches, dataset,
//! metrics, util, Approach}`. The only new audio plumbing is a resampler, since
//! ESC-50 is 44.1 kHz and the pipeline is tuned for 16 kHz.
//!
//! ## Resampling choice (documented, per the task)
//! ESC-50 clips are 44.1 kHz; DADS and Al-Emadi are 16 kHz. To make ESC-50
//! comparable we resample to 16 kHz with an **anti-aliased rational resampler**:
//! a single-pass linear interpolation at the target rate, preceded by a simple
//! moving-average (box) low-pass whose width tracks the decimation ratio
//! (`floor(src/dst) = 2` taps for 44.1→16 k). The box filter knocks down energy
//! above the 8 kHz Nyquist enough to avoid gross aliasing into the harmonic band
//! the detectors care about (drone blade-pass + harmonics live well below 4 kHz),
//! while staying dependency-free and fully deterministic. We deliberately avoid a
//! sharp windowed-sinc FIR: it is more correct but heavier, and the drone cues
//! are low-frequency, so the cheaper filter does not change the conclusions. This
//! is a benchmark front-end, not a production decimator — the trade is noted on
//! purpose.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use serde::Serialize;

use drone_bench::approaches;
use drone_bench::dataset::{read_mono_wav, Dataset, Sample};
use drone_bench::metrics::evaluate;
use drone_bench::Approach;

/// Target sample rate the 16 kHz-tuned pipeline expects.
const TARGET_SR: u32 = 16_000;

/// ESC-50 classes we treat as hard NEGATIVES. The first group is acoustically
/// confusable with rotor/engine drone audio (harmonic / mechanical), the second
/// is broadband, and a few unrelated controls are added at random by the loader.
const HARD_NEG_CLASSES: &[&str] = &[
    "airplane",
    "helicopter",
    "engine",
    "chainsaw",
    "wind",
    "crackling_fire",
];

/// A few extra ESC-50 classes pulled in as "random others" negatives so the
/// negative set isn't only the maximally-confusable classes (keeps the FPR
/// table honest about easy classes too). Deterministic, not actually random.
const EXTRA_NEG_CLASSES: &[&str] = &["rain", "sea_waves", "clapping", "dog"];

#[derive(Parser)]
#[command(
    name = "xeval",
    version,
    about = "Cross-dataset + hard-negative (leakage-honest) drone-detection eval"
)]
struct Cli {
    /// DADS dataset root containing `labels.csv` (the TRAIN source).
    #[arg(
        long,
        default_value = "C:/Users/julia/Development/acoustic-drone-detection/data/dads"
    )]
    dads: PathBuf,
    /// DADS manifest filename inside `--dads`.
    #[arg(long, default_value = "labels.csv")]
    dads_manifest: String,
    /// Al-Emadi root containing `Binary_Drone_Audio/yes_drone` (TEST positives).
    #[arg(
        long,
        default_value = "C:/Users/julia/Development/acoustic-drone-detection/workspace/al-emadi"
    )]
    alemadi: PathBuf,
    /// ESC-50 root containing `audio/` and `meta/esc50.csv` (TEST negatives).
    #[arg(
        long,
        default_value = "C:/Users/julia/Development/acoustic-drone-detection/workspace/esc50"
    )]
    esc50: PathBuf,
    /// Fraction of each DADS class used for training (the rest is the
    /// in-distribution comparison split).
    #[arg(long, default_value_t = 0.5)]
    train_frac: f32,
    /// Fixed decision threshold for headline confusion + the per-class FPR table.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,
    /// Cap on Al-Emadi positive clips loaded (deterministic stride subsample).
    /// 0 = use all.
    #[arg(long, default_value_t = 600)]
    max_pos: usize,
    /// Cap on ESC-50 clips PER negative class loaded (ESC-50 has 40/class).
    #[arg(long, default_value_t = 40)]
    per_class: usize,
    /// RNG seed (DADS split only; everything else is deterministic by id sort).
    #[arg(long, default_value_t = 1)]
    seed: u32,
    /// Output JSON path.
    #[arg(long, default_value = "benchmarks/results/xeval.json")]
    out: PathBuf,
    /// How many top approaches (by cross-dataset ROC-AUC) to include in the
    /// per-confounder FPR table.
    #[arg(long, default_value_t = 4)]
    top: usize,
}

// ----------------------------------------------------------------------------
// Resampling (see module docs for the rationale).
// ----------------------------------------------------------------------------

/// Resample `x` from `src_sr` to `dst_sr` with anti-aliased linear interpolation.
/// When downsampling, a moving-average box low-pass of width `floor(src/dst)` is
/// applied first to suppress energy above the new Nyquist. Deterministic.
fn resample(x: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if src_sr == dst_sr || x.is_empty() {
        return x.to_vec();
    }
    // Anti-alias only when downsampling.
    let filtered: Vec<f32> = if src_sr > dst_sr {
        let width = (src_sr / dst_sr).max(1) as usize;
        if width <= 1 {
            x.to_vec()
        } else {
            box_lowpass(x, width)
        }
    } else {
        x.to_vec()
    };
    // Linear interpolation onto the target grid.
    let ratio = src_sr as f64 / dst_sr as f64;
    let out_len = ((x.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let i0 = pos.floor() as usize;
        let frac = (pos - i0 as f64) as f32;
        let a = filtered[i0.min(filtered.len() - 1)];
        let b = filtered[(i0 + 1).min(filtered.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Centered moving-average low-pass of the given odd-ish width (clamped at ends).
fn box_lowpass(x: &[f32], width: usize) -> Vec<f32> {
    let w = width.max(1);
    let half = w / 2;
    let mut out = vec![0.0_f32; x.len()];
    for (i, o) in out.iter_mut().enumerate() {
        let lo = i.saturating_sub(half);
        let hi = (i + half).min(x.len() - 1);
        let mut acc = 0.0_f32;
        for &v in &x[lo..=hi] {
            acc += v;
        }
        *o = acc / (hi - lo + 1) as f32;
    }
    out
}

// ----------------------------------------------------------------------------
// Dataset loading for the cross-dataset test set.
// ----------------------------------------------------------------------------

/// Load Al-Emadi drone clips (`Binary_Drone_Audio/yes_drone/*.wav`) as positives.
/// Deterministic stride subsample to `max` if `max > 0`.
fn load_alemadi_positives(root: &Path, max: usize) -> Result<Vec<Sample>, Box<dyn Error>> {
    let dir = root.join("Binary_Drone_Audio").join("yes_drone");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("reading {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "wav").unwrap_or(false))
        .collect();
    files.sort(); // deterministic order
    let stride = if max > 0 && files.len() > max {
        files.len().div_ceil(max)
    } else {
        1
    };
    let mut out = Vec::new();
    for p in files
        .iter()
        .step_by(stride)
        .take(if max > 0 { max } else { usize::MAX })
    {
        let (audio, sr) = read_mono_wav(p)?;
        let audio = resample(&audio, sr, TARGET_SR);
        out.push(Sample {
            id: format!("alemadi/{}", p.file_name().unwrap().to_string_lossy()),
            samples: audio,
            sample_rate: TARGET_SR,
            label: 1,
        });
    }
    Ok(out)
}

/// One ESC-50 negative clip, carrying its class so we can build the per-class
/// FPR table later.
struct EscClip {
    sample: Sample,
    category: String,
}

/// Load ESC-50 clips for the chosen negative classes (hard + extra), resampled
/// to 16 kHz. `per_class` caps clips per class (ESC-50 has 40). Deterministic by
/// filename sort.
fn load_esc50_negatives(root: &Path, per_class: usize) -> Result<Vec<EscClip>, Box<dyn Error>> {
    let meta = root.join("meta").join("esc50.csv");
    let text =
        std::fs::read_to_string(&meta).map_err(|e| format!("reading {}: {e}", meta.display()))?;
    let wanted: Vec<&str> = HARD_NEG_CLASSES
        .iter()
        .chain(EXTRA_NEG_CLASSES)
        .copied()
        .collect();

    // Collect (category -> sorted filenames) for the wanted classes.
    let mut by_class: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 {
            continue; // header: filename,fold,target,category,esc10,src_file,take
        }
        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 4 {
            continue;
        }
        let filename = cols[0].trim();
        let category = cols[3].trim();
        if wanted.contains(&category) {
            by_class
                .entry(category.to_string())
                .or_default()
                .push(filename.to_string());
        }
    }

    let audio_dir = root.join("audio");
    let mut out = Vec::new();
    for (category, mut files) in by_class {
        files.sort();
        for f in files
            .into_iter()
            .take(if per_class > 0 { per_class } else { usize::MAX })
        {
            let p = audio_dir.join(&f);
            let (audio, sr) = read_mono_wav(&p)?;
            let audio = resample(&audio, sr, TARGET_SR);
            out.push(EscClip {
                sample: Sample {
                    id: format!("esc50/{category}/{f}"),
                    samples: audio,
                    sample_rate: TARGET_SR,
                    label: 0,
                },
                category: category.clone(),
            });
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------------
// Serializable results.
// ----------------------------------------------------------------------------

#[derive(Serialize)]
struct ApproachXEval {
    approach: String,
    description: String,
    /// Cross-dataset (DADS-train → Al-Emadi/ESC-50) metrics.
    cross_roc_auc: f32,
    cross_f1: f32,
    cross_f1_best: f32,
    cross_threshold_best: f32,
    cross_precision: f32,
    cross_recall: f32,
    cross_accuracy: f32,
    cross_brier: f32,
    /// In-distribution DADS held-out split, same fitted model — the inflated
    /// baseline we are being honest about.
    indist_roc_auc: f32,
    indist_f1: f32,
    indist_f1_best: f32,
    /// Drops (in-distribution minus cross-dataset). Positive = generalization gap.
    roc_auc_drop: f32,
    f1_best_drop: f32,
    mean_infer_ms: f64,
}

#[derive(Serialize)]
struct ClassFpr {
    category: String,
    n: usize,
    false_positives: usize,
    fpr: f32,
}

#[derive(Serialize)]
struct DetectorFpr {
    approach: String,
    /// Per-class FPR at the fixed threshold, ordered by FPR descending.
    per_class: Vec<ClassFpr>,
    /// Overall FPR across all ESC-50 negatives at the fixed threshold.
    overall_fpr: f32,
}

#[derive(Serialize)]
struct XEvalReport {
    description: &'static str,
    target_sample_rate: u32,
    threshold: f32,
    train_frac: f32,
    n_dads_train: usize,
    n_dads_indist_test: usize,
    n_cross_pos: usize,
    n_cross_neg: usize,
    hard_neg_classes: Vec<String>,
    extra_neg_classes: Vec<String>,
    resampling: &'static str,
    experiment_a_cross_dataset: Vec<ApproachXEval>,
    experiment_b_confusion: Vec<DetectorFpr>,
}

// ----------------------------------------------------------------------------

fn instantiate(name: &str) -> Box<dyn Approach> {
    approaches::all()
        .into_iter()
        .find(|a| a.name() == name)
        .unwrap_or_else(|| panic!("unknown approach {name}"))
}

/// Score a slice of samples, returning `(score, label)` pairs and total infer secs.
fn score_samples(approach: &dyn Approach, samples: &[Sample]) -> (Vec<(f32, u8)>, f64) {
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

    // --- TRAIN source: DADS, split into train + an in-distribution test half. ---
    let dads_manifest = cli.dads.join(&cli.dads_manifest);
    println!(
        "loading DADS (train source) from {}",
        dads_manifest.display()
    );
    let dads = Dataset::load_csv(&cli.dads, &dads_manifest)?;
    if dads.is_empty() {
        return Err("DADS dataset is empty".into());
    }
    let (train, indist_test) = dads.split(cli.train_frac, cli.seed);
    println!(
        "DADS: {} clips ({} pos) -> {} train / {} in-distribution test",
        dads.len(),
        dads.n_pos(),
        train.len(),
        indist_test.len()
    );

    // --- TEST corpus: Al-Emadi positives + ESC-50 hard negatives. ---
    println!("loading Al-Emadi positives from {}", cli.alemadi.display());
    let pos = load_alemadi_positives(&cli.alemadi, cli.max_pos)?;
    println!("loading ESC-50 negatives from {}", cli.esc50.display());
    let esc = load_esc50_negatives(&cli.esc50, cli.per_class)?;
    if pos.is_empty() || esc.is_empty() {
        return Err("cross-dataset test set is empty (check --alemadi / --esc50)".into());
    }
    // Flat negative samples for scoring, plus a category lookup for Exp. B.
    let neg: Vec<Sample> = esc.iter().map(|c| c.sample.clone()).collect();
    let mut cross_test: Vec<Sample> = Vec::with_capacity(pos.len() + neg.len());
    cross_test.extend(pos.iter().cloned());
    cross_test.extend(neg.iter().cloned());
    println!(
        "cross-dataset test: {} positives (Al-Emadi) + {} negatives (ESC-50, {} classes)",
        pos.len(),
        neg.len(),
        HARD_NEG_CLASSES.len() + EXTRA_NEG_CLASSES.len()
    );

    let registry: Vec<(String, String)> = approaches::all()
        .iter()
        .map(|a| (a.name().to_string(), a.description().to_string()))
        .collect();

    // ---- Experiment A: cross-dataset, with in-distribution comparison. ----
    println!("\n=== Experiment A: cross-dataset detection (DADS-train -> Al-Emadi/ESC-50) ===");
    println!(
        "{:<20} {:>9} {:>7} {:>7} | {:>9} {:>7} | {:>8} {:>8}",
        "approach", "xROC-AUC", "xF1", "xF1*", "idROC", "idF1*", "dAUC", "dF1*"
    );
    println!("{}", "-".repeat(86));

    let mut exp_a: Vec<ApproachXEval> = Vec::new();
    // Keep cross-dataset scores per approach for Experiment B.
    let mut cross_scores_by_approach: BTreeMap<String, Vec<(f32, u8)>> = BTreeMap::new();

    for (name, description) in &registry {
        let mut approach = instantiate(name);
        approach.fit(&train);

        // Cross-dataset test.
        let (cross_scored, infer_secs) = score_samples(approach.as_ref(), &cross_test);
        let cross_m = evaluate(&cross_scored, cli.threshold);
        // In-distribution test (same fitted model).
        let (indist_scored, _) = score_samples(approach.as_ref(), &indist_test);
        let indist_m = evaluate(&indist_scored, cli.threshold);

        let mean_infer_ms = infer_secs * 1000.0 / cross_scored.len().max(1) as f64;

        cross_scores_by_approach.insert(name.clone(), cross_scored.clone());

        let roc_auc_drop = indist_m.roc_auc - cross_m.roc_auc;
        let f1_best_drop = indist_m.f1_best - cross_m.f1_best;

        println!(
            "{:<20} {:>9.3} {:>7.3} {:>7.3} | {:>9.3} {:>7.3} | {:>8.3} {:>8.3}",
            name,
            cross_m.roc_auc,
            cross_m.f1,
            cross_m.f1_best,
            indist_m.roc_auc,
            indist_m.f1_best,
            roc_auc_drop,
            f1_best_drop,
        );

        exp_a.push(ApproachXEval {
            approach: name.clone(),
            description: description.clone(),
            cross_roc_auc: cross_m.roc_auc,
            cross_f1: cross_m.f1,
            cross_f1_best: cross_m.f1_best,
            cross_threshold_best: cross_m.threshold_best,
            cross_precision: cross_m.precision,
            cross_recall: cross_m.recall,
            cross_accuracy: cross_m.accuracy,
            cross_brier: cross_m.brier,
            indist_roc_auc: indist_m.roc_auc,
            indist_f1: indist_m.f1,
            indist_f1_best: indist_m.f1_best,
            roc_auc_drop,
            f1_best_drop,
            mean_infer_ms,
        });
    }

    // ---- Experiment B: per-confounder FPR for the top detectors. ----
    // Rank by cross-dataset ROC-AUC (NaN sorts last).
    let mut ranked: Vec<&ApproachXEval> = exp_a.iter().collect();
    ranked.sort_by(|a, b| {
        let av = if a.cross_roc_auc.is_nan() {
            f32::NEG_INFINITY
        } else {
            a.cross_roc_auc
        };
        let bv = if b.cross_roc_auc.is_nan() {
            f32::NEG_INFINITY
        } else {
            b.cross_roc_auc
        };
        bv.partial_cmp(&av).unwrap()
    });
    let top_names: Vec<String> = ranked
        .iter()
        .take(cli.top.max(1))
        .map(|a| a.approach.clone())
        .collect();

    // Build a per-detector, per-clip score map indexed the same way `neg` is
    // ordered, by re-scoring just the negatives (cheap, deterministic). We reuse
    // the cross-dataset run instead: negatives are the LAST `neg.len()` entries
    // of `cross_test`, so the tail of each approach's cross scores aligns 1:1.
    println!(
        "\n=== Experiment B: hard-negative confusion (FPR per ESC-50 class @ threshold {}) ===",
        cli.threshold
    );

    let n_pos = pos.len();
    let mut exp_b: Vec<DetectorFpr> = Vec::new();

    for name in &top_names {
        let scored = &cross_scores_by_approach[name];
        // Negative scores are the tail (after the positives).
        let neg_scores = &scored[n_pos..];
        debug_assert_eq!(neg_scores.len(), esc.len());

        // Aggregate FP / N per category.
        let mut counts: BTreeMap<String, (usize, usize)> = BTreeMap::new(); // (fp, n)
        for (clip, &(s, _y)) in esc.iter().zip(neg_scores.iter()) {
            let e = counts.entry(clip.category.clone()).or_insert((0, 0));
            e.1 += 1;
            if s >= cli.threshold {
                e.0 += 1;
            }
        }
        let mut per_class: Vec<ClassFpr> = counts
            .into_iter()
            .map(|(category, (fp, n))| ClassFpr {
                category,
                n,
                false_positives: fp,
                fpr: if n > 0 { fp as f32 / n as f32 } else { 0.0 },
            })
            .collect();
        per_class.sort_by(|a, b| b.fpr.partial_cmp(&a.fpr).unwrap());

        let total_fp: usize = per_class.iter().map(|c| c.false_positives).sum();
        let total_n: usize = per_class.iter().map(|c| c.n).sum();
        let overall_fpr = if total_n > 0 {
            total_fp as f32 / total_n as f32
        } else {
            0.0
        };

        // Print a compact row: detector then "class=fpr" for the worst few.
        print!("{name:<20} overall FPR {overall_fpr:>5.3} | worst: ");
        for c in per_class.iter().take(6) {
            print!("{}={:.2} ", c.category, c.fpr);
        }
        println!();

        exp_b.push(DetectorFpr {
            approach: name.clone(),
            per_class,
            overall_fpr,
        });
    }

    // ---- Write JSON. ----
    let report = XEvalReport {
        description: "Leakage-honest cross-dataset (A) + hard-negative confusion (B) eval. \
                      FIT on DADS train; TEST on Al-Emadi drones (pos) + ESC-50 confusable \
                      classes (neg). In-distribution DADS metrics included for the honest drop.",
        target_sample_rate: TARGET_SR,
        threshold: cli.threshold,
        train_frac: cli.train_frac,
        n_dads_train: train.len(),
        n_dads_indist_test: indist_test.len(),
        n_cross_pos: pos.len(),
        n_cross_neg: neg.len(),
        hard_neg_classes: HARD_NEG_CLASSES.iter().map(|s| s.to_string()).collect(),
        extra_neg_classes: EXTRA_NEG_CLASSES.iter().map(|s| s.to_string()).collect(),
        resampling: "44.1k->16k: box low-pass (width floor(src/dst)) + linear interpolation",
        experiment_a_cross_dataset: exp_a,
        experiment_b_confusion: exp_b,
    };
    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out, serde_json::to_string_pretty(&report)?)?;
    println!("\nresults written to {}", cli.out.display());

    Ok(())
}
