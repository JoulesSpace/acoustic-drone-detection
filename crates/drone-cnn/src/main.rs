//! `drone-cnn` - the published-SOTA upstream baseline (a mel-spectrogram CNN),
//! benchmarked HEAD-TO-HEAD against our heuristic detectors on the
//! leakage-proof unseen-drone test.
//!
//! ## Why this exists
//! Upstream drone-audio papers (Al-Emadi et al.'s CNN ~93%, the MDPI CNNs
//! ~94-98%) report only IN-DISTRIBUTION accuracy - train and test drawn from the
//! same drones. Our harness's `heldout32` showed our best detectors get recall
//! 0.72 (hps) to 0.87@0.5 (sentry ensemble) and ROC-AUC ~0.855 on 32 UNSEEN
//! drone models. Nobody has run a drone CNN on genuinely-unseen drones. This
//! binary does exactly that, so we can state - with evidence - whether our suite
//! matches/beats the true upstream SOTA on trustworthy evaluation.
//!
//! ## What it does
//! 1. Trains a small mel-spectrogram CNN on DADS (the upstream representative).
//! 2. IN-DISTRIBUTION sanity: evaluates on a held-out DADS split - it should
//!    reach the published ~0.9+ ROC-AUC/F1 ballpark, confirming it is faithful.
//! 3. LEAKAGE-PROOF unseen-drone test: evaluates on the SAME corpus `heldout32`
//!    uses - 32-brand drone-visualization positives (truly held out, none in
//!    DADS) + ESC-50 negatives - and reports recall@0.5, recall@calibrated
//!    (threshold from a DADS slice), and ROC-AUC.
//! 4. Writes `benchmarks/results/cnn.json` and prints a table putting the CNN's
//!    unseen-drone numbers next to ours (hps 0.72/0.855, sentry 0.87@0.5).
//!
//! The honest question: does the upstream-style CNN BEAT our best on the
//! leakage-proof unseen-drone test, or does it also collapse out-of-domain like
//! the literature warns? Reported truthfully either way.

mod data;
mod mel;
mod model;
mod train;

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use serde::Serialize;

use candle_core::Device;
use drone_bench::dataset::{Dataset, Sample};
use drone_bench::metrics::{best_f1, evaluate};

use data::{load_esc50_negatives, load_heldout_drones, neg_classes, TARGET_SR};
use mel::{MelBank, MelImage, Standardizer};
use model::DroneCnn;
use train::{predict, train, TrainCfg};

/// Our published unseen-drone numbers, hard-coded for the side-by-side table.
/// Source: the harness `heldout32` run summarized in the task.
const OURS: &[(&str, f32, f32, f32)] = &[
    // (name, recall@0.5, recall@cal-or-best, ROC-AUC)
    ("hps (ours)", f32::NAN, 0.72, 0.855),
    ("sentry ensemble (ours)", 0.87, f32::NAN, f32::NAN),
];

#[derive(Parser)]
#[command(
    name = "drone-cnn",
    version,
    about = "Mel-spectrogram CNN (upstream-SOTA baseline) vs our detectors on UNSEEN drones"
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
    /// Root of the 32-brand drone-visualization clone (unseen-drone positives).
    #[arg(
        long,
        default_value = "C:/Users/julia/Development/acoustic-drone-detection/workspace/dv"
    )]
    drones: PathBuf,
    /// ESC-50 root (unseen-drone-test negatives; NOT held out - in DADS).
    #[arg(
        long,
        default_value = "C:/Users/julia/Development/acoustic-drone-detection/workspace/esc50"
    )]
    esc50: PathBuf,
    /// Fraction of DADS used for the train+val pool; the rest is in-dist test.
    #[arg(long, default_value_t = 0.7)]
    train_frac: f32,
    /// Fraction of the train pool held out as the validation/calibration slice.
    #[arg(long, default_value_t = 0.2)]
    val_frac: f32,
    /// Fixed decision threshold for the un-calibrated recall column.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,
    /// Cap on ESC-50 clips PER negative class loaded.
    #[arg(long, default_value_t = 40)]
    per_class: usize,
    /// Master RNG seed (model init + batch shuffles + DADS split).
    #[arg(long, default_value_t = 1)]
    seed: u32,
    /// Training epochs (early stopping usually halts sooner).
    #[arg(long, default_value_t = 60)]
    epochs: usize,
    /// Print per-epoch training progress.
    #[arg(long, default_value_t = false)]
    verbose: bool,
    /// Output JSON path.
    #[arg(long, default_value = "benchmarks/results/cnn.json")]
    out: PathBuf,
}

// ---------------------------------------------------------------------------
// Serializable report.
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct InDistMetrics {
    n_test: usize,
    n_pos: usize,
    n_neg: usize,
    roc_auc: f32,
    pr_auc: f32,
    f1_fixed: f32,
    f1_best: f32,
    threshold_best: f32,
    accuracy_fixed: f32,
    precision_fixed: f32,
    recall_fixed: f32,
    brier: f32,
}

#[derive(Serialize)]
struct BrandRecall {
    brand: String,
    n: usize,
    recall_fixed: f32,
    recall_calibrated: f32,
}

#[derive(Serialize)]
struct UnseenMetrics {
    n_pos_clips: usize,
    n_pos_brands: usize,
    n_neg_clips: usize,
    /// HEADLINE: recall (TPR) on the 32 unseen drone brands.
    recall_unseen_fixed: f32,
    recall_unseen_calibrated: f32,
    /// DADS best-F1 threshold used for the calibrated column.
    dads_calibrated_threshold: f32,
    dads_calib_f1: f32,
    /// INDICATIVE (negatives in DADS): full-ROC metrics on the mixed test set.
    roc_auc: f32,
    f1_fixed: f32,
    f1_best: f32,
    threshold_best: f32,
    precision_fixed: f32,
    accuracy_fixed: f32,
    brier: f32,
    per_brand: Vec<BrandRecall>,
}

#[derive(Serialize)]
struct ComparisonRow {
    method: String,
    family: String,
    recall_unseen_fixed: Option<f32>,
    recall_unseen_calibrated: Option<f32>,
    roc_auc_unseen: Option<f32>,
}

#[derive(Serialize)]
struct TrainSummary {
    epochs_run: usize,
    best_epoch: usize,
    best_val_loss: f32,
    final_train_loss: f32,
    n_train: usize,
    n_val: usize,
}

#[derive(Serialize)]
struct CnnReport {
    description: &'static str,
    framing: &'static str,
    model: &'static str,
    backend: &'static str,
    n_mels: usize,
    n_frames: usize,
    target_sample_rate: u32,
    seed: u32,
    resampling: &'static str,
    positives_held_out: bool,
    negatives_held_out: bool,
    neg_confusable_classes: Vec<String>,
    neg_control_classes: Vec<String>,
    training: TrainSummary,
    in_distribution: InDistMetrics,
    unseen_drones: UnseenMetrics,
    verdict: String,
    comparison: Vec<ComparisonRow>,
}

// ---------------------------------------------------------------------------

/// Build standardized mel images for a set of samples (resampled to 16 kHz if
/// needed). Returns `(images, labels f32)`.
fn featurize(bank: &MelBank, samples: &[Sample]) -> (Vec<MelImage>, Vec<f32>) {
    let mut imgs = Vec::with_capacity(samples.len());
    let mut labels = Vec::with_capacity(samples.len());
    for s in samples {
        let audio = if s.sample_rate == TARGET_SR {
            s.samples.clone()
        } else {
            data::resample(&s.samples, s.sample_rate, TARGET_SR)
        };
        imgs.push(bank.log_mel_image(&audio));
        labels.push(s.label as f32);
    }
    (imgs, labels)
}

/// Recall (TPR) over positives only, at a given threshold.
fn recall_at(scores: &[f32], labels: &[f32], threshold: f32) -> f32 {
    let mut tp = 0usize;
    let mut pos = 0usize;
    for (&s, &y) in scores.iter().zip(labels) {
        if y > 0.5 {
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

fn fmt_opt(v: Option<f32>) -> String {
    match v {
        Some(x) if !x.is_nan() => format!("{x:.3}"),
        _ => "  -  ".to_string(),
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    let device = Device::Cpu;
    let t_start = Instant::now();

    // --- DADS: train+val pool (train_frac) and in-distribution test (rest). ---
    let dads_manifest = cli.dads.join(&cli.dads_manifest);
    println!(
        "loading DADS (train source) from {}",
        dads_manifest.display()
    );
    let dads = Dataset::load_csv(&cli.dads, &dads_manifest)?;
    if dads.is_empty() {
        return Err("DADS dataset is empty".into());
    }
    let (pool, dads_test) = dads.split(cli.train_frac, cli.seed);
    // Split the pool again into train / val (the val slice doubles as the
    // threshold-calibration slice - calibration only, never the in-dist test).
    let pool_ds = Dataset { samples: pool };
    let (train_samples, val_samples) = pool_ds.split(1.0 - cli.val_frac, cli.seed.wrapping_add(7));
    println!(
        "DADS: {} clips ({} pos) -> {} train / {} val(calib) / {} in-dist test",
        dads.len(),
        dads.n_pos(),
        train_samples.len(),
        val_samples.len(),
        dads_test.len()
    );

    // --- Mel front-end: fit the filterbank, featurize, standardize on TRAIN. ---
    let bank = MelBank::new(TARGET_SR);
    let (mut train_imgs, train_labels) = featurize(&bank, &train_samples);
    let (mut val_imgs, val_labels) = featurize(&bank, &val_samples);
    let (mut test_imgs, test_labels) = featurize(&bank, &dads_test);

    let std = Standardizer::fit(&train_imgs);
    for im in train_imgs.iter_mut() {
        std.apply(im);
    }
    for im in val_imgs.iter_mut() {
        std.apply(im);
    }
    for im in test_imgs.iter_mut() {
        std.apply(im);
    }

    // --- Build + train the CNN (deterministic init from the seed). ---
    println!(
        "\nbuilding mel-spectrogram CNN (candle-core/candle-nn, CPU, seed {})",
        cli.seed
    );
    let cnn = DroneCnn::new(cli.seed as u64, &device)?;
    let cfg = TrainCfg {
        epochs: cli.epochs,
        seed: cli.seed as u64,
        ..Default::default()
    };
    println!("training (BCE + AdamW, early stop on val) ...");
    let tr = train(
        &cnn,
        &train_imgs,
        &train_labels,
        &val_imgs,
        &val_labels,
        &cfg,
        cli.verbose,
    )?;
    println!(
        "  done: {} epochs run, best @ {} (val_loss {:.4}), final train_loss {:.4}",
        tr.epochs_run, tr.best_epoch, tr.best_val_loss, tr.final_train_loss
    );

    // --- DADS threshold calibration: best-F1 on the val slice (no test peek). ---
    let val_scores = predict(&cnn, &val_imgs)?;
    let val_scored: Vec<(f32, u8)> = val_scores
        .iter()
        .zip(val_labels.iter())
        .map(|(&s, &y)| (s, (y > 0.5) as u8))
        .collect();
    let (dads_thr, dads_calib_f1) = best_f1(&val_scored);

    // ============================= IN-DISTRIBUTION ==========================
    let test_scores = predict(&cnn, &test_imgs)?;
    let test_scored: Vec<(f32, u8)> = test_scores
        .iter()
        .zip(test_labels.iter())
        .map(|(&s, &y)| (s, (y > 0.5) as u8))
        .collect();
    let m = evaluate(&test_scored, cli.threshold);
    let n_pos = test_labels.iter().filter(|&&y| y > 0.5).count();
    let in_dist = InDistMetrics {
        n_test: test_scored.len(),
        n_pos,
        n_neg: test_scored.len() - n_pos,
        roc_auc: m.roc_auc,
        pr_auc: m.pr_auc,
        f1_fixed: m.f1,
        f1_best: m.f1_best,
        threshold_best: m.threshold_best,
        accuracy_fixed: m.accuracy,
        precision_fixed: m.precision,
        recall_fixed: m.recall,
        brier: m.brier,
    };

    // ====================== LEAKAGE-PROOF UNSEEN DRONES =====================
    println!(
        "\nloading UNSEEN-drone positives from {}",
        cli.drones.display()
    );
    let drones = load_heldout_drones(&cli.drones)?;
    println!("loading ESC-50 negatives from {}", cli.esc50.display());
    let neg = load_esc50_negatives(&cli.esc50, cli.per_class)?;
    if drones.is_empty() || neg.is_empty() {
        return Err("unseen-drone test set is empty (check --drones / --esc50)".into());
    }
    let brand_set: std::collections::BTreeSet<String> =
        drones.iter().map(|c| c.brand.clone()).collect();

    let pos_samples: Vec<Sample> = drones.iter().map(|c| c.sample.clone()).collect();
    let (pos_imgs_raw, _) = featurize(&bank, &pos_samples);
    let (neg_imgs_raw, _) = featurize(&bank, &neg);
    let mut pos_imgs = pos_imgs_raw;
    let mut neg_imgs = neg_imgs_raw;
    for im in pos_imgs.iter_mut() {
        std.apply(im);
    }
    for im in neg_imgs.iter_mut() {
        std.apply(im);
    }

    let pos_scores = predict(&cnn, &pos_imgs)?;
    let neg_scores = predict(&cnn, &neg_imgs)?;

    // Mixed test set for the (indicative) ROC/F1.
    let mut unseen_scored: Vec<(f32, u8)> = Vec::with_capacity(pos_scores.len() + neg_scores.len());
    unseen_scored.extend(pos_scores.iter().map(|&s| (s, 1u8)));
    unseen_scored.extend(neg_scores.iter().map(|&s| (s, 0u8)));
    let um = evaluate(&unseen_scored, cli.threshold);

    let pos_labels = vec![1.0f32; pos_scores.len()];
    let recall_unseen_fixed = recall_at(&pos_scores, &pos_labels, cli.threshold);
    let recall_unseen_calibrated = recall_at(&pos_scores, &pos_labels, dads_thr);

    // Per-brand recall.
    let mut by_brand: std::collections::BTreeMap<String, Vec<f32>> =
        std::collections::BTreeMap::new();
    for (clip, &s) in drones.iter().zip(pos_scores.iter()) {
        by_brand.entry(clip.brand.clone()).or_default().push(s);
    }
    let mut per_brand: Vec<BrandRecall> = by_brand
        .into_iter()
        .map(|(brand, scores)| {
            let labels = vec![1.0f32; scores.len()];
            BrandRecall {
                n: scores.len(),
                recall_fixed: recall_at(&scores, &labels, cli.threshold),
                recall_calibrated: recall_at(&scores, &labels, dads_thr),
                brand,
            }
        })
        .collect();
    per_brand.sort_by(|a, b| {
        a.recall_calibrated
            .partial_cmp(&b.recall_calibrated)
            .unwrap()
            .then(a.brand.cmp(&b.brand))
    });

    let unseen = UnseenMetrics {
        n_pos_clips: pos_scores.len(),
        n_pos_brands: brand_set.len(),
        n_neg_clips: neg_scores.len(),
        recall_unseen_fixed,
        recall_unseen_calibrated,
        dads_calibrated_threshold: dads_thr,
        dads_calib_f1,
        roc_auc: um.roc_auc,
        f1_fixed: um.f1,
        f1_best: um.f1_best,
        threshold_best: um.threshold_best,
        precision_fixed: um.precision,
        accuracy_fixed: um.accuracy,
        brier: um.brier,
        per_brand,
    };

    // ============================== VERDICT =================================
    // Our best on the honest test: sentry recall@0.5 = 0.87, hps ROC-AUC = 0.855.
    let our_best_recall_fixed = 0.87f32;
    let our_best_roc = 0.855f32;
    let cnn_beats_recall = unseen.recall_unseen_fixed > our_best_recall_fixed + 0.01;
    let cnn_beats_roc = unseen.roc_auc > our_best_roc + 0.01;
    let collapses = unseen.recall_unseen_fixed < in_dist.recall_fixed - 0.10
        || unseen.roc_auc < in_dist.roc_auc - 0.10;
    let verdict = if cnn_beats_recall && cnn_beats_roc {
        format!(
            "UPSTREAM CNN WINS: on the leakage-proof unseen-drone test it beats our best \
             (recall@0.5 {:.3} > 0.87, ROC-AUC {:.3} > 0.855). The published SOTA generalizes \
             to unseen drones better than our heuristic suite.",
            unseen.recall_unseen_fixed, unseen.roc_auc
        )
    } else if collapses {
        format!(
            "UPSTREAM CNN COLLAPSES OUT-OF-DOMAIN: in-distribution ROC-AUC {:.3} / recall {:.3} \
             but on UNSEEN drones recall@0.5 {:.3}, recall@cal {:.3}, ROC-AUC {:.3} (indicative). \
             Exactly the literature's warning - high in-dist, mediocre on unseen drones. Our \
             honest 0.72-0.87 is therefore competitive with the true SOTA on trustworthy eval.",
            in_dist.roc_auc,
            in_dist.recall_fixed,
            unseen.recall_unseen_fixed,
            unseen.recall_unseen_calibrated,
            unseen.roc_auc
        )
    } else {
        format!(
            "MIXED: the CNN does not clearly beat our best on the unseen-drone test \
             (CNN recall@0.5 {:.3} vs ours 0.87; CNN ROC-AUC {:.3} vs ours 0.855). It neither \
             dominates nor fully collapses - our honest numbers sit in the same band as the \
             upstream SOTA on trustworthy eval.",
            unseen.recall_unseen_fixed, unseen.roc_auc
        )
    };

    // ============================== PRINT ===================================
    println!("\n=== IN-DISTRIBUTION (DADS held-out split) - faithfulness sanity ===");
    println!(
        "  ROC-AUC {:.3}  PR-AUC {:.3}  F1@0.5 {:.3}  F1* {:.3}  acc {:.3}  (n={}, {}+/{}-)",
        in_dist.roc_auc,
        in_dist.pr_auc,
        in_dist.f1_fixed,
        in_dist.f1_best,
        in_dist.accuracy_fixed,
        in_dist.n_test,
        in_dist.n_pos,
        in_dist.n_neg
    );
    let in_dist_ok = in_dist.roc_auc >= 0.9 || in_dist.f1_best >= 0.9;
    println!(
        "  {} published ballpark (~0.9+ ROC-AUC/F1): faithful baseline {}",
        if in_dist_ok { "REACHES" } else { "BELOW" },
        if in_dist_ok {
            "confirmed"
        } else {
            "(see caveats)"
        }
    );

    println!("\n=== LEAKAGE-PROOF UNSEEN-DRONE TEST (32 brands not in DADS) ===");
    println!(
        "  recall@0.5 {:.3}  recall@cal {:.3} (DADS thr {:.3})  ROC-AUC {:.3} (indicative)  F1* {:.3}",
        unseen.recall_unseen_fixed,
        unseen.recall_unseen_calibrated,
        unseen.dads_calibrated_threshold,
        unseen.roc_auc,
        unseen.f1_best
    );
    println!(
        "  ({} positive clips / {} unseen brands + {} ESC-50 negatives)",
        unseen.n_pos_clips, unseen.n_pos_brands, unseen.n_neg_clips
    );

    println!("\n=== HEAD-TO-HEAD on the honest test (unseen drones) ===");
    println!(
        "  {:<26} {:>10} {:>10} {:>10}",
        "method", "Rec@0.5", "Rec@cal", "ROC-AUC"
    );
    println!("  {}", "-".repeat(58));
    println!(
        "  {:<26} {:>10.3} {:>10.3} {:>10.3}   <- upstream-SOTA baseline",
        "mel-CNN (this crate)",
        unseen.recall_unseen_fixed,
        unseen.recall_unseen_calibrated,
        unseen.roc_auc
    );
    let mut comparison = vec![ComparisonRow {
        method: "mel-CNN (this crate)".to_string(),
        family: "upstream-SOTA baseline".to_string(),
        recall_unseen_fixed: Some(unseen.recall_unseen_fixed),
        recall_unseen_calibrated: Some(unseen.recall_unseen_calibrated),
        roc_auc_unseen: Some(unseen.roc_auc),
    }];
    for &(name, r05, rcal, roc) in OURS {
        println!(
            "  {:<26} {:>10} {:>10} {:>10}",
            name,
            fmt_opt(opt(r05)),
            fmt_opt(opt(rcal)),
            fmt_opt(opt(roc))
        );
        comparison.push(ComparisonRow {
            method: name.to_string(),
            family: "ours (heuristic)".to_string(),
            recall_unseen_fixed: opt(r05),
            recall_unseen_calibrated: opt(rcal),
            roc_auc_unseen: opt(roc),
        });
    }

    println!("\n=== VERDICT ===\n  {verdict}");
    println!(
        "\nCAVEAT: unseen-drone POSITIVES are truly held out (32 models, none in DADS) -> recall\n\
         on unseen drones is the headline. ESC-50 NEGATIVES are inside DADS, so ROC-AUC / F1 are\n\
         indicative, not a clean cross-source number. The CNN is a faithful small upstream model,\n\
         not a max-capacity production net; DADS is ~600 clips, so absolute in-dist numbers are\n\
         dataset-bound. Same resampling/windowing as `heldout32` for an apples-to-apples compare."
    );

    // ============================== JSON ====================================
    let (neg_conf, neg_ctrl) = neg_classes();
    let report = CnnReport {
        description: "Mel-spectrogram CNN (Al-Emadi / MDPI style) trained on DADS, evaluated \
                      in-distribution (DADS held-out split) and on the leakage-proof unseen-drone \
                      test (32-brand drone-visualization positives + ESC-50 negatives), the same \
                      corpus drone-bench's heldout32 uses for our heuristic detectors.",
        framing: "Upstream papers report only IN-DISTRIBUTION accuracy. This runs the upstream \
                  representative on genuinely UNSEEN drones. POSITIVES are truly held out (headline \
                  = recall on unseen drones); NEGATIVES (ESC-50) are in DADS, so ROC-AUC/F1 are \
                  indicative.",
        model: "1->conv8(3x3,relu,pool2)->conv16(3x3,relu,pool2)->flatten->dense32(relu)->dense1->sigmoid",
        backend: "candle-core + candle-nn (pure Rust, CPU); deterministic seed-driven weight init \
                  (candle's CPU RNG is not seedable, so all init comes from our own xorshift)",
        n_mels: mel::N_MELS,
        n_frames: mel::N_FRAMES,
        target_sample_rate: TARGET_SR,
        seed: cli.seed,
        resampling: "44.1k->16k: box low-pass (width floor(src/dst)) + linear interpolation \
                     (identical to heldout32/xeval)",
        positives_held_out: true,
        negatives_held_out: false,
        neg_confusable_classes: neg_conf,
        neg_control_classes: neg_ctrl,
        training: TrainSummary {
            epochs_run: tr.epochs_run,
            best_epoch: tr.best_epoch,
            best_val_loss: tr.best_val_loss,
            final_train_loss: tr.final_train_loss,
            n_train: train_imgs.len(),
            n_val: val_imgs.len(),
        },
        in_distribution: in_dist,
        unseen_drones: unseen,
        verdict,
        comparison,
    };
    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out, serde_json::to_string_pretty(&report)?)?;
    println!("\nresults written to {}", cli.out.display());
    println!("total wall time: {:.1}s", t_start.elapsed().as_secs_f64());

    Ok(())
}

/// `f32::NAN` -> `None`, else `Some`.
fn opt(v: f32) -> Option<f32> {
    if v.is_nan() {
        None
    } else {
        Some(v)
    }
}
