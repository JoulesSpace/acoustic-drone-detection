//! `heldout32` - TRULY HELD-OUT generalization eval on drones NOT in DADS.
//!
//! Our cross-dataset `xeval` used Al-Emadi drones + ESC-50, but BOTH of those
//! sources are inside the DADS merge, so even the "cross-dataset" numbers are
//! optimistic: a DADS-fitted detector may have seen those exact recordings (or
//! sibling clips from them) during training. This binary closes that gap on the
//! positive side.
//!
//! The **32-brand College-of-Charleston drone set** (the audio shipped with the
//! `mackenzie-jane/drone-visualization` repo, referenced by arXiv 2509.04715) is
//! NOT part of DADS. It is therefore a genuinely held-out source of drone
//! POSITIVES across 32 unseen brands/models. Testing DADS-trained detectors on
//! it measures real generalization to drones never seen in training - the
//! decisive number, obtainable entirely from public data with no field
//! recording.
//!
//! ## Honest framing (printed in the output too)
//! * POSITIVES are TRULY held-out: 32 unseen drone makes/models, none in DADS.
//!   This is the strongest public generalization test we have. The headline
//!   metric is therefore **recall on unseen drones** (TPR), which needs no
//!   negatives at all and so cannot be gamed by negative-set leakage.
//! * NEGATIVES are NOT held-out: ESC-50 confusable classes ARE inside DADS. So
//!   the **ROC-AUC and calibrated-F1 are indicative, not clean** - they pair
//!   clean positives with leaky negatives. We report them for a full ROC but
//!   lead with recall-on-unseen-drones.
//!
//! ## The protocol
//! 1. FIT every approach from `approaches::all()` on the FULL DADS train split.
//! 2. Pick that approach's DADS best-F1 threshold on a held-out DADS slice
//!    (calibration only - no test peeking).
//! 3. On the held-out TEST set (32-brand drones = positives, ESC-50 = negatives):
//!    - ROC-AUC and calibrated-F1 (best-threshold on the test scores).
//!    - Recall on unseen drones at the fixed 0.5 threshold AND at the
//!      DADS-calibrated threshold. Both reported; this is the purest
//!      positive-generalization metric.
//!
//! Reuses the harness contract verbatim: `drone_bench::{approaches, dataset,
//! metrics, Approach}`. The only new audio plumbing is the same anti-aliased
//! resampler `xeval` uses (the 32-brand WAVs are 44.1 kHz, the pipeline is tuned
//! for 16 kHz), plus windowing the ~5 s brand clips into ~1 s clips.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use serde::Serialize;

use drone_bench::approaches;
use drone_bench::dataset::{read_mono_wav, Dataset, Sample};
use drone_bench::metrics::{best_f1, evaluate};
use drone_bench::Approach;

/// Target sample rate the 16 kHz-tuned pipeline expects.
const TARGET_SR: u32 = 16_000;

/// Window length (seconds) each 32-brand clip is cut into. ~1.0 s matches the
/// DADS / Al-Emadi clip granularity the detectors were tuned around.
const WINDOW_SECS: f32 = 1.0;

/// Approaches we call out by name in the printed summary (per the task).
const HEADLINE_APPROACHES: &[&str] = &[
    "physics_fused",
    "feature_fusion",
    "hps",
    "envelope_periodicity",
];

/// ESC-50 classes used as NEGATIVES (NOT held out - they live inside DADS, so
/// this side of the ROC is indicative only). The first group is acoustically
/// confusable with rotor/engine drone audio; the second group is a few controls.
const NEG_CONFUSABLE_CLASSES: &[&str] = &["airplane", "helicopter", "engine", "chainsaw", "wind"];
const NEG_CONTROL_CLASSES: &[&str] = &["rain", "sea_waves", "clapping"];

#[derive(Parser)]
#[command(
    name = "heldout32",
    version,
    about = "Truly held-out generalization eval: DADS-trained detectors vs 32 UNSEEN drone brands"
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
    /// Root of the 32-brand drone-visualization clone; WAVs live under
    /// `public/droneAudio/` (TEST positives, truly held out).
    #[arg(
        long,
        default_value = "C:/Users/julia/Development/acoustic-drone-detection/workspace/dv"
    )]
    drones: PathBuf,
    /// ESC-50 root containing `audio/` and `meta/esc50.csv` (TEST negatives;
    /// NOT held out - in DADS). Defaults to the main repo workspace clone.
    #[arg(
        long,
        default_value = "C:/Users/julia/Development/acoustic-drone-detection/workspace/esc50"
    )]
    esc50: PathBuf,
    /// Fraction of each DADS class used for fitting; the remainder is the DADS
    /// calibration slice used to pick the best-F1 operating threshold.
    #[arg(long, default_value_t = 0.7)]
    train_frac: f32,
    /// Fixed decision threshold for the un-calibrated recall column.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,
    /// Cap on ESC-50 clips PER negative class loaded (ESC-50 has 40/class).
    #[arg(long, default_value_t = 40)]
    per_class: usize,
    /// RNG seed for the DADS fit/calibration split (everything else is
    /// deterministic by id sort).
    #[arg(long, default_value_t = 1)]
    seed: u32,
    /// Output JSON path.
    #[arg(long, default_value = "benchmarks/results/heldout32.json")]
    out: PathBuf,
}

// ----------------------------------------------------------------------------
// Resampling (identical scheme to xeval; see its module docs for the rationale).
// 44.1k -> 16k: moving-average box low-pass of width floor(src/dst) to suppress
// energy above the new Nyquist, then linear interpolation onto the target grid.
// Dependency-free and fully deterministic. The drone cues (blade-pass +
// harmonics) live well below 4 kHz, so the cheap filter does not move the
// conclusions; this is a benchmark front-end, not a production decimator.
// ----------------------------------------------------------------------------

/// Resample `x` from `src_sr` to `dst_sr` with anti-aliased linear interpolation.
fn resample(x: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if src_sr == dst_sr || x.is_empty() {
        return x.to_vec();
    }
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

/// Centered moving-average low-pass of the given width (clamped at the ends).
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
// Dataset loading for the held-out test set.
// ----------------------------------------------------------------------------

/// Parse the brand/model name from a 32-set filename, e.g.
/// `DJI_Mavic_Air2_63.wav` -> `DJI_Mavic_Air2`. The trailing numeric token is a
/// take index, not part of the model name, so we drop it for per-brand grouping.
fn brand_of(file_stem: &str) -> String {
    match file_stem.rsplit_once('_') {
        Some((head, tail)) if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) => {
            head.to_string()
        }
        _ => file_stem.to_string(),
    }
}

/// One held-out positive clip plus the brand it came from (for the per-brand
/// recall breakdown).
struct DroneClip {
    sample: Sample,
    brand: String,
}

/// Load the 32-brand drone clips: decode to mono, resample to 16 kHz, then cut
/// into non-overlapping ~`WINDOW_SECS` windows. All label 1. A trailing partial
/// window shorter than half a window is dropped. Deterministic by filename sort.
fn load_heldout_drones(root: &Path) -> Result<Vec<DroneClip>, Box<dyn Error>> {
    let dir = root.join("public").join("droneAudio");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map_err(|e| format!("reading {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "wav").unwrap_or(false))
        .collect();
    files.sort(); // deterministic order
    if files.is_empty() {
        return Err(format!("no WAVs under {}", dir.display()).into());
    }

    let win = (WINDOW_SECS * TARGET_SR as f32) as usize;
    let mut out = Vec::new();
    for p in &files {
        let stem = p.file_stem().unwrap().to_string_lossy().to_string();
        let brand = brand_of(&stem);
        let (audio, sr) = read_mono_wav(p)?;
        let audio = resample(&audio, sr, TARGET_SR);
        let n_full = audio.len() / win;
        let mut made = 0usize;
        for w in 0..n_full {
            let start = w * win;
            out.push(DroneClip {
                sample: Sample {
                    id: format!("dv/{stem}#{w}"),
                    samples: audio[start..start + win].to_vec(),
                    sample_rate: TARGET_SR,
                    label: 1,
                },
                brand: brand.clone(),
            });
            made += 1;
        }
        // Keep a long-enough trailing remainder so short files still contribute.
        let rem = audio.len() - n_full * win;
        if made == 0 && rem >= win / 2 && !audio.is_empty() {
            out.push(DroneClip {
                sample: Sample {
                    id: format!("dv/{stem}#0"),
                    samples: audio.clone(),
                    sample_rate: TARGET_SR,
                    label: 1,
                },
                brand: brand.clone(),
            });
        }
    }
    Ok(out)
}

/// Load ESC-50 negatives for the chosen confusable + control classes, resampled
/// to 16 kHz. `per_class` caps clips per class. Deterministic by filename sort.
fn load_esc50_negatives(root: &Path, per_class: usize) -> Result<Vec<Sample>, Box<dyn Error>> {
    let meta = root.join("meta").join("esc50.csv");
    let text =
        std::fs::read_to_string(&meta).map_err(|e| format!("reading {}: {e}", meta.display()))?;
    let wanted: Vec<&str> = NEG_CONFUSABLE_CLASSES
        .iter()
        .chain(NEG_CONTROL_CLASSES)
        .copied()
        .collect();

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
            out.push(Sample {
                id: format!("esc50/{category}/{f}"),
                samples: audio,
                sample_rate: TARGET_SR,
                label: 0,
            });
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------------
// Serializable results.
// ----------------------------------------------------------------------------

#[derive(Serialize)]
struct BrandRecall {
    brand: String,
    n: usize,
    /// TPR at the fixed 0.5 threshold.
    recall_fixed: f32,
    /// TPR at the DADS-calibrated (best-F1) threshold.
    recall_calibrated: f32,
}

#[derive(Serialize)]
struct ApproachHeldout {
    approach: String,
    description: String,
    /// HEADLINE: recall (TPR) on the 32 unseen drone brands - needs no
    /// negatives, so it cannot be gamed by negative-set leakage.
    recall_unseen_fixed: f32,
    recall_unseen_calibrated: f32,
    /// The DADS best-F1 threshold used for the calibrated column (calibration
    /// slice only - no test peeking).
    dads_calibrated_threshold: f32,
    dads_calib_f1: f32,
    /// INDICATIVE (negatives are NOT held out): full-ROC metrics pairing the
    /// clean unseen-drone positives with leaky ESC-50 negatives.
    roc_auc: f32,
    f1_fixed: f32,
    f1_calibrated_best: f32,
    threshold_calibrated_best: f32,
    precision_fixed: f32,
    accuracy_fixed: f32,
    brier: f32,
    /// Per-brand recall breakdown (sorted by calibrated recall ascending - the
    /// hardest unseen models first).
    per_brand: Vec<BrandRecall>,
    mean_infer_ms: f64,
}

#[derive(Serialize)]
struct HeldoutReport {
    description: &'static str,
    framing: &'static str,
    target_sample_rate: u32,
    window_secs: f32,
    fixed_threshold: f32,
    train_frac: f32,
    n_dads_fit: usize,
    n_dads_calib: usize,
    n_test_pos_clips: usize,
    n_test_pos_brands: usize,
    n_test_neg_clips: usize,
    neg_confusable_classes: Vec<String>,
    neg_control_classes: Vec<String>,
    resampling: &'static str,
    positives_held_out: bool,
    negatives_held_out: bool,
    /// Sorted by ROC-AUC descending.
    approaches: Vec<ApproachHeldout>,
}

// ----------------------------------------------------------------------------

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

/// Recall (TPR) over positives only, at a given threshold.
fn recall_at(scored: &[(f32, u8)], threshold: f32) -> f32 {
    let mut tp = 0usize;
    let mut pos = 0usize;
    for &(s, y) in scored {
        if y == 1 {
            pos += 1;
            if s >= threshold {
                tp += 1;
            }
        }
    }
    if pos == 0 {
        0.0
    } else {
        tp as f32 / pos as f32
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    // --- TRAIN/CALIBRATION source: DADS, split into a fit half + calib slice. ---
    let dads_manifest = cli.dads.join(&cli.dads_manifest);
    println!(
        "loading DADS (train source) from {}",
        dads_manifest.display()
    );
    let dads = Dataset::load_csv(&cli.dads, &dads_manifest)?;
    if dads.is_empty() {
        return Err("DADS dataset is empty".into());
    }
    let (fit, calib) = dads.split(cli.train_frac, cli.seed);
    println!(
        "DADS: {} clips ({} pos) -> {} fit / {} calibration",
        dads.len(),
        dads.n_pos(),
        fit.len(),
        calib.len()
    );

    // --- TEST corpus: 32-brand drones (held out) + ESC-50 (NOT held out). ---
    println!(
        "loading 32-brand held-out drone positives from {}",
        cli.drones.display()
    );
    let drones = load_heldout_drones(&cli.drones)?;
    println!("loading ESC-50 negatives from {}", cli.esc50.display());
    let neg = load_esc50_negatives(&cli.esc50, cli.per_class)?;
    if drones.is_empty() || neg.is_empty() {
        return Err("held-out test set is empty (check --drones / --esc50)".into());
    }

    let brand_set: std::collections::BTreeSet<String> =
        drones.iter().map(|c| c.brand.clone()).collect();
    let pos: Vec<Sample> = drones.iter().map(|c| c.sample.clone()).collect();
    let mut test: Vec<Sample> = Vec::with_capacity(pos.len() + neg.len());
    test.extend(pos.iter().cloned());
    test.extend(neg.iter().cloned());
    println!(
        "held-out test: {} positive clips from {} unseen brands + {} ESC-50 negatives",
        pos.len(),
        brand_set.len(),
        neg.len()
    );

    let registry: Vec<(String, String)> = approaches::all()
        .iter()
        .map(|a| (a.name().to_string(), a.description().to_string()))
        .collect();

    println!("\n=== Held-out generalization: DADS-trained detectors vs UNSEEN drone brands ===");
    println!(
        "POSITIVES truly held out (32 unseen models); NEGATIVES (ESC-50) are in DADS -> ROC indicative."
    );
    println!(
        "{:<22} {:>9} {:>9} | {:>8} {:>8} {:>8} | {:>7}",
        "approach", "Rec@0.5", "Rec@cal", "ROC-AUC", "F1@0.5", "F1*", "calThr"
    );
    println!("{}", "-".repeat(86));

    let n_pos = pos.len();
    let mut results: Vec<ApproachHeldout> = Vec::new();

    for (name, description) in &registry {
        let mut approach = approaches::all()
            .into_iter()
            .find(|a| a.name() == name)
            .unwrap_or_else(|| panic!("unknown approach {name}"));
        approach.fit(&fit);

        // DADS calibration: best-F1 threshold on the held-out DADS slice only.
        let (calib_scored, _) = score_samples(approach.as_ref(), &calib);
        let (dads_thr, dads_calib_f1) = best_f1(&calib_scored);

        // Held-out test scoring.
        let (test_scored, infer_secs) = score_samples(approach.as_ref(), &test);
        let test_m = evaluate(&test_scored, cli.threshold);
        let mean_infer_ms = infer_secs * 1000.0 / test_scored.len().max(1) as f64;

        // Positive-only scores (the first n_pos entries of `test`).
        let pos_scored = &test_scored[..n_pos];
        let recall_unseen_fixed = recall_at(pos_scored, cli.threshold);
        let recall_unseen_calibrated = recall_at(pos_scored, dads_thr);

        // Per-brand recall, aligned to the positive ordering in `drones`.
        let mut by_brand: BTreeMap<String, Vec<(f32, u8)>> = BTreeMap::new();
        for (clip, &sl) in drones.iter().zip(pos_scored.iter()) {
            by_brand.entry(clip.brand.clone()).or_default().push(sl);
        }
        let mut per_brand: Vec<BrandRecall> = by_brand
            .into_iter()
            .map(|(brand, sls)| BrandRecall {
                brand,
                n: sls.len(),
                recall_fixed: recall_at(&sls, cli.threshold),
                recall_calibrated: recall_at(&sls, dads_thr),
            })
            .collect();
        per_brand.sort_by(|a, b| {
            a.recall_calibrated
                .partial_cmp(&b.recall_calibrated)
                .unwrap()
                .then(a.brand.cmp(&b.brand))
        });

        println!(
            "{:<22} {:>9.3} {:>9.3} | {:>8.3} {:>8.3} {:>8.3} | {:>7.3}",
            name,
            recall_unseen_fixed,
            recall_unseen_calibrated,
            test_m.roc_auc,
            test_m.f1,
            test_m.f1_best,
            dads_thr,
        );

        results.push(ApproachHeldout {
            approach: name.clone(),
            description: description.clone(),
            recall_unseen_fixed,
            recall_unseen_calibrated,
            dads_calibrated_threshold: dads_thr,
            dads_calib_f1,
            roc_auc: test_m.roc_auc,
            f1_fixed: test_m.f1,
            f1_calibrated_best: test_m.f1_best,
            threshold_calibrated_best: test_m.threshold_best,
            precision_fixed: test_m.precision,
            accuracy_fixed: test_m.accuracy,
            brier: test_m.brier,
            per_brand,
            mean_infer_ms,
        });
    }

    // Sort by ROC-AUC descending (NaN last) for the report ordering.
    results.sort_by(|a, b| {
        let av = if a.roc_auc.is_nan() {
            f32::NEG_INFINITY
        } else {
            a.roc_auc
        };
        let bv = if b.roc_auc.is_nan() {
            f32::NEG_INFINITY
        } else {
            b.roc_auc
        };
        bv.partial_cmp(&av).unwrap()
    });

    // ---- Headline recall ranking (the metric that is NOT leakage-tainted). ----
    let mut by_recall: Vec<&ApproachHeldout> = results.iter().collect();
    by_recall.sort_by(|a, b| {
        b.recall_unseen_calibrated
            .partial_cmp(&a.recall_unseen_calibrated)
            .unwrap()
            .then(a.approach.cmp(&b.approach))
    });
    println!("\n=== HEADLINE: recall on UNSEEN drone brands (calibrated thr), best first ===");
    for r in &by_recall {
        println!(
            "  {:<22} recall@cal {:>5.3}  recall@0.5 {:>5.3}  (ROC-AUC {:>5.3}, indicative)",
            r.approach, r.recall_unseen_calibrated, r.recall_unseen_fixed, r.roc_auc
        );
    }

    println!("\n=== Called-out approaches (per task) ===");
    for want in HEADLINE_APPROACHES {
        if let Some(r) = results.iter().find(|r| r.approach == *want) {
            println!(
                "  {:<22} recall@0.5 {:>5.3}  recall@cal {:>5.3}  ROC-AUC {:>5.3} (indicative)  F1* {:>5.3}",
                r.approach,
                r.recall_unseen_fixed,
                r.recall_unseen_calibrated,
                r.roc_auc,
                r.f1_calibrated_best
            );
        }
    }

    println!(
        "\nCAVEAT: positives are the strongest public generalization test we have (32 UNSEEN\n\
         drone models, none in DADS) -> recall-on-unseen-drones is the headline. Negatives\n\
         (ESC-50) ARE inside DADS, so ROC-AUC / F1 are indicative, not a clean cross-source\n\
         number."
    );

    // ---- Write JSON. ----
    let report = HeldoutReport {
        description: "Truly held-out generalization eval. FIT every approach on the full DADS \
                      train split, calibrate its threshold on a held-out DADS slice, then TEST \
                      on the 32-brand College-of-Charleston drone set (positives, NOT in DADS) \
                      vs ESC-50 confusable classes (negatives, IN DADS).",
        framing: "POSITIVES are truly held out (32 unseen drone models -> strongest public \
                  generalization test); recall-on-unseen-drones is the headline and needs no \
                  negatives. NEGATIVES (ESC-50) are NOT held out (they are in DADS), so ROC-AUC \
                  and calibrated-F1 are indicative only.",
        target_sample_rate: TARGET_SR,
        window_secs: WINDOW_SECS,
        fixed_threshold: cli.threshold,
        train_frac: cli.train_frac,
        n_dads_fit: fit.len(),
        n_dads_calib: calib.len(),
        n_test_pos_clips: pos.len(),
        n_test_pos_brands: brand_set.len(),
        n_test_neg_clips: neg.len(),
        neg_confusable_classes: NEG_CONFUSABLE_CLASSES
            .iter()
            .map(|s| s.to_string())
            .collect(),
        neg_control_classes: NEG_CONTROL_CLASSES.iter().map(|s| s.to_string()).collect(),
        resampling: "44.1k->16k: box low-pass (width floor(src/dst)) + linear interpolation",
        positives_held_out: true,
        negatives_held_out: false,
        approaches: results,
    };
    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out, serde_json::to_string_pretty(&report)?)?;
    println!("\nresults written to {}", cli.out.display());

    Ok(())
}
