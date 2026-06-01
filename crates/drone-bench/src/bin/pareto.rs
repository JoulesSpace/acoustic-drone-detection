//! `pareto` - speed/accuracy trade-off benchmark and hardware-tier mapping.
//!
//! For every registered detection approach this binary measures:
//!
//!   * **inference latency** - `score()` timed over many clips, reported both
//!     per clip and normalised to microseconds per 1024-sample frame
//!     ([`drone_dsp::FRAME_SIZE`]) and per second of audio;
//!   * **real-time factor** - inference seconds per audio second (`< 1.0` runs
//!     faster than real time on this machine, one stream);
//!   * **accuracy** - ROC-AUC and the calibrated best-F1, from a single
//!     stratified split (default) or k-fold pooled out-of-fold predictions;
//!   * a rough **feature dimension** / relative compute cost (a static design
//!     property of each approach, not a measurement).
//!
//! Each approach is mapped to a **hardware tier** - `tiny-edge`, `balanced`, or
//! `max-accuracy` - reflecting where it can plausibly run. The result set is
//! written to `benchmarks/results/pareto.json` and printed as a table sorted by
//! accuracy, flagging the **Pareto frontier**: an approach is on the frontier
//! when no other approach is simultaneously faster *and* more accurate.
//!
//! Usage mirrors `drone-bench`: `--data <dir>` (default the DADS dataset) or
//! `--synth`, optional `--kfold K`, `--repeats R` (time each clip R times for a
//! stabler latency estimate). Deterministic given the same seed.

use std::error::Error;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use drone_bench::approaches;
use drone_bench::dataset::{Dataset, Sample};
use drone_bench::metrics::{best_f1, roc_auc};
use drone_bench::Approach;
use drone_dsp::FRAME_SIZE;
use serde::Serialize;

/// Default DADS dataset location (absolute, per the project layout).
const DEFAULT_DATA: &str = "C:/Users/julia/Development/acoustic-drone-detection/data/dads";

#[derive(Parser)]
#[command(
    name = "pareto",
    version,
    about = "Speed/accuracy Pareto benchmark + hardware-tier mapping"
)]
struct Cli {
    /// Dataset root containing `labels.csv` (header `path,label`, label 0/1).
    #[arg(long, default_value = DEFAULT_DATA)]
    data: PathBuf,
    /// Manifest filename inside `--data`.
    #[arg(long, default_value = "labels.csv")]
    manifest: String,
    /// Use a deterministic synthetic dataset instead of `--data`.
    #[arg(long)]
    synth: bool,
    /// Samples per class for `--synth`.
    #[arg(long, default_value_t = 200)]
    n: usize,
    /// Fraction of each class used for training (single-split mode).
    #[arg(long, default_value_t = 0.5)]
    train_frac: f32,
    /// K-fold cross-validation (pooled out-of-fold predictions). 1 = single split.
    #[arg(long, default_value_t = 1)]
    kfold: usize,
    /// Time each clip this many times when measuring latency (stabler estimate).
    #[arg(long, default_value_t = 3)]
    repeats: usize,
    /// RNG seed (synth generation and split).
    #[arg(long, default_value_t = 1)]
    seed: u32,
    /// Sample rate for synthetic data.
    #[arg(long, default_value_t = 16_000)]
    sample_rate: u32,
    /// Output JSON path.
    #[arg(long, default_value = "benchmarks/results/pareto.json")]
    out: PathBuf,
}

/// Hardware tier an approach is targeted at.
#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Tier {
    /// esp32-class MCU: ~16 kHz, KB of RAM, no FPU-heavy work.
    TinyEdge,
    /// Phone / Raspberry Pi / Android: a real CPU but a power/thermal budget.
    Balanced,
    /// Server / workstation: ensembles and large feature stacks are fine.
    MaxAccuracy,
}

impl Tier {
    fn as_str(self) -> &'static str {
        match self {
            Tier::TinyEdge => "tiny-edge",
            Tier::Balanced => "balanced",
            Tier::MaxAccuracy => "max-accuracy",
        }
    }
}

/// Static design properties of each approach: its hardware tier, a rough feature
/// dimension, and a deterministic relative-cost rank.
///
/// * `feature_dim` - dimensionality of the clip-level descriptor the model
///   scores over (a structural complexity proxy, reported as-is).
/// * `cost_rank` - a small ordinal capturing *relative inference cost* (lower =
///   cheaper), assigned from each approach's algorithm, not from a stopwatch.
///   It is used as the **deterministic speed axis** for the Pareto frontier so
///   the frontier does not flap with measurement noise (see [`flag_pareto`]).
///   The rank mirrors the measured ordering: O(spectrum) scalar statistics are
///   cheapest; a single learned head / template / patch is next; methods that
///   iterate heavily over the signal (autocorrelation, long-envelope analysis)
///   and the stacked ensemble are most expensive.
///
/// Tiers reflect structural cost, justified in `MODEL_CARDS.md`:
///   * **tiny-edge** - single scalar statistics over a magnitude spectrum, no
///     learned matrix, no per-frame ML. Runnable on an MCU with KB of RAM.
///   * **balanced** - a learned linear/MLP head over a modest (tens of dims)
///     cepstral/spectral feature, or a single template/patch correlation.
///   * **max-accuracy** - multi-feature fusion or an ensemble stacked over the
///     base detectors; the heaviest, server-class.
fn profile(name: &str) -> (Tier, u32, u32) {
    match name {
        // (tier, feature_dim, cost_rank)
        // tiny-edge: cheap spectral statistics, no learned weights to store.
        "band_ratio" => (Tier::TinyEdge, 1, 10), // in-band / total energy ratio
        "hps" => (Tier::TinyEdge, 1, 12),        // harmonic-product-spectrum score
        "spectral_gate" => (Tier::TinyEdge, 5, 14), // 5 spectral summary features
        // balanced: a learned head over a tens-of-dims feature, or one template.
        "mfcc_lr" => (Tier::Balanced, 27, 20), // 13 MFCC mean+std (+1 bias) -> LR
        "template" => (Tier::Balanced, 512, 22), // cosine vs a 512-bin log-spectrum
        "spectrogram_template" => (Tier::Balanced, 384, 24), // 24 mel x 16 time patch
        "mfcc_mlp" => (Tier::Balanced, 27, 26), // 27 features -> 24-unit MLP
        "gtcc_lr" => (Tier::Balanced, 27, 28), // 13 GTCC mean+std (+1 bias) -> LR
        // max-accuracy: multi-feature fusion (modest) ...
        "feature_fusion" => (Tier::MaxAccuracy, 34, 40), // MFCC + 8 extra -> LR
        // signal-iterating methods and the ensemble are the most expensive.
        "cepstrum" => (Tier::Balanced, 2, 60), // quefrency peak + comb energy blend
        "fusion" => (Tier::MaxAccuracy, 6, 70), // stacked over 6 base approaches
        "envelope_periodicity" => (Tier::Balanced, 2, 80), // long AM-envelope analysis
        // Unknown approaches: balanced, unknown dim, mid cost rank.
        _ => (Tier::Balanced, 0, 50),
    }
}

/// One row of the emitted `pareto.json`.
#[derive(Serialize)]
struct ParetoRow {
    approach: String,
    tier: &'static str,
    /// Inference latency normalised to microseconds per 1024-sample frame.
    latency_us_per_frame: f64,
    /// Inference time per second of audio (ms). Length-normalised cost.
    ms_per_audio_sec: f64,
    /// Real-time factor = inference seconds / audio seconds (`< 1` is faster
    /// than real time, one stream, this machine).
    realtime_factor: f64,
    /// ROC-AUC (threshold-free ranking quality).
    roc_auc: f64,
    /// Best F1 over all thresholds (calibrated operating point).
    f1: f64,
    /// Rough feature dimension (structural complexity proxy).
    feature_dim: u32,
    /// Deterministic relative-cost rank (lower = cheaper); the speed axis used
    /// for the Pareto frontier. See [`profile`].
    cost_rank: u32,
    /// Mean wall-clock inference time per clip (ms), for reference.
    mean_infer_ms: f64,
    /// Whether this approach is on the speed/accuracy Pareto frontier.
    pareto_frontier: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let dataset = if cli.synth {
        println!("generating synthetic dataset: {} per class", cli.n);
        Dataset::synth(cli.n, cli.sample_rate, cli.seed)
    } else {
        let manifest = cli.data.join(&cli.manifest);
        println!("loading dataset from {}", manifest.display());
        Dataset::load_csv(&cli.data, &manifest)?
    };
    if dataset.is_empty() {
        return Err("dataset is empty".into());
    }
    let kfold = cli.kfold.max(1);
    let repeats = cli.repeats.max(1);
    println!(
        "dataset: {} samples ({} pos){} - latency timed {repeats}x per clip",
        dataset.len(),
        dataset.n_pos(),
        if kfold > 1 {
            format!(" - {kfold}-fold CV")
        } else {
            format!(" - single split (train_frac {})", cli.train_frac)
        },
    );

    // We iterate by name so we can re-instantiate a fresh model per fold
    // (fitting mutates state and trait objects aren't cloneable).
    let names: Vec<String> = approaches::all()
        .iter()
        .map(|a| a.name().to_string())
        .collect();

    let mut rows: Vec<ParetoRow> = Vec::with_capacity(names.len());
    for name in &names {
        let (scored, infer_secs, audio_secs, frames) = if kfold > 1 {
            measure_kfold(&dataset, kfold, name, repeats, cli.seed)
        } else {
            let (train, test) = dataset.split(cli.train_frac, cli.seed);
            let mut approach = instantiate(name);
            approach.fit(&train);
            measure(approach.as_ref(), &test, repeats)
        };

        let n_clips = scored.len().max(1) as f64;
        let mean_infer_ms = infer_secs * 1000.0 / n_clips;
        let latency_us_per_frame = if frames > 0.0 {
            infer_secs * 1e6 / frames
        } else {
            0.0
        };
        let (ms_per_audio_sec, realtime_factor) = if audio_secs > 0.0 {
            (infer_secs * 1000.0 / audio_secs, infer_secs / audio_secs)
        } else {
            (0.0, 0.0)
        };

        let auc = roc_auc(&scored);
        let (_t, f1) = best_f1(&scored);
        let (tier, feature_dim, cost_rank) = profile(name);
        rows.push(ParetoRow {
            approach: name.clone(),
            tier: tier.as_str(),
            latency_us_per_frame,
            ms_per_audio_sec,
            realtime_factor,
            roc_auc: nan_to_zero(auc as f64),
            f1: f1 as f64,
            feature_dim,
            cost_rank,
            mean_infer_ms,
            pareto_frontier: false,
        });
    }

    flag_pareto(&mut rows);

    // Sort by accuracy (ROC-AUC) descending for both the table and the JSON.
    rows.sort_by(|a, b| b.roc_auc.partial_cmp(&a.roc_auc).unwrap());

    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out, serde_json::to_string_pretty(&rows)?)?;

    print_table(&rows);
    println!("\nresults written to {}", cli.out.display());
    Ok(())
}

/// Build a fresh instance of the named approach from the registry.
fn instantiate(name: &str) -> Box<dyn Approach> {
    approaches::all()
        .into_iter()
        .find(|a| a.name() == name)
        .unwrap_or_else(|| panic!("unknown approach {name}"))
}

/// Score a fitted approach over `samples`, timing only `score()`. Each clip is
/// scored `repeats` times and the *total* time is accumulated; the audio-second
/// and frame counts are likewise multiplied by `repeats` so the per-frame and
/// real-time figures stay correctly normalised. Returns the first-pass
/// `(score, label)` pairs (scores are deterministic across repeats), total
/// inference seconds, total audio seconds, and total 1024-sample frames.
fn measure(
    approach: &dyn Approach,
    samples: &[Sample],
    repeats: usize,
) -> (Vec<(f32, u8)>, f64, f64, f64) {
    let mut scored = Vec::with_capacity(samples.len());
    let mut infer_secs = 0.0_f64;
    let mut audio_secs = 0.0_f64;
    let mut frames = 0.0_f64;
    for s in samples {
        let sr = s.sample_rate.max(1);
        let mut last = 0.0_f32;
        let t0 = Instant::now();
        for _ in 0..repeats {
            last = approach.score(&s.samples, sr).clamp(0.0, 1.0);
        }
        infer_secs += t0.elapsed().as_secs_f64();
        let clip_audio = s.samples.len() as f64 / sr as f64;
        audio_secs += clip_audio * repeats as f64;
        // Number of full 1024-sample analysis frames in the clip.
        frames += (s.samples.len() / FRAME_SIZE) as f64 * repeats as f64;
        scored.push((last, s.label));
    }
    (scored, infer_secs, audio_secs, frames)
}

/// K-fold CV producing pooled out-of-fold predictions plus accumulated timing,
/// mirroring the `drone-bench` k-fold path. Folds are stratified by class.
fn measure_kfold(
    ds: &Dataset,
    k: usize,
    name: &str,
    repeats: usize,
    seed: u32,
) -> (Vec<(f32, u8)>, f64, f64, f64) {
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
    let (mut infer_secs, mut audio_secs, mut frames) = (0.0_f64, 0.0_f64, 0.0_f64);
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
        let (mut scored, inf, aud, fr) = measure(approach.as_ref(), &test, repeats);
        out.append(&mut scored);
        infer_secs += inf;
        audio_secs += aud;
        frames += fr;
    }
    (out, infer_secs, audio_secs, frames)
}

/// Mark each row on the speed/accuracy Pareto frontier: no other approach is
/// *both* cheaper *and* more accurate. "Cheaper" is decided by the deterministic
/// `cost_rank` (the static relative-cost proxy from [`profile`]), not by the
/// measured latency - latency is a stopwatch reading that jitters run-to-run and
/// would make the reported frontier non-deterministic for speed-tied approaches.
/// `cost_rank` mirrors the measured speed ordering but is fixed, so the frontier
/// is the same on every run for fixed accuracy. Equal cost ranks are broken by
/// approach name, also deterministic.
///
/// Concretely: an approach is dominated iff some other approach has a strictly
/// lower `(cost_rank, name)` key *and* a strictly higher ROC-AUC. Equivalently,
/// walking approaches cheapest-first, a row is on the frontier iff no cheaper row
/// is more accurate. The plotted/reported `latency_us_per_frame` is still the
/// measured value; only the frontier *membership* uses the deterministic proxy.
fn flag_pareto(rows: &mut [ParetoRow]) {
    // Deterministic cheapest-first order: by cost rank, then approach name.
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by(|&a, &b| {
        (rows[a].cost_rank, &rows[a].approach).cmp(&(rows[b].cost_rank, &rows[b].approach))
    });
    // Sweep cheap -> expensive; a row survives iff it beats the best accuracy
    // seen among all strictly-cheaper rows.
    let mut best_auc_so_far = f64::NEG_INFINITY;
    for &i in &order {
        rows[i].pareto_frontier = rows[i].roc_auc > best_auc_so_far;
        best_auc_so_far = best_auc_so_far.max(rows[i].roc_auc);
    }
}

/// Print the result table, sorted by accuracy, flagging frontier rows with `*`.
fn print_table(rows: &[ParetoRow]) {
    println!(
        "\n{:<22} {:<13} {:>10} {:>11} {:>9} {:>8} {:>7} {:>6} {:>4}",
        "approach", "tier", "us/frame", "ms/audio-s", "xRT", "ROC-AUC", "F1*", "dim", "PF"
    );
    println!("{}", "-".repeat(98));
    for r in rows {
        let xrt = if r.realtime_factor > 0.0 {
            1.0 / r.realtime_factor
        } else {
            f64::INFINITY
        };
        println!(
            "{:<22} {:<13} {:>10.2} {:>11.3} {:>9.0} {:>8.3} {:>7.3} {:>6} {:>4}",
            r.approach,
            r.tier,
            r.latency_us_per_frame,
            r.ms_per_audio_sec,
            xrt,
            r.roc_auc,
            r.f1,
            r.feature_dim,
            if r.pareto_frontier { "*" } else { "" },
        );
    }
    let frontier: Vec<&str> = rows
        .iter()
        .filter(|r| r.pareto_frontier)
        .map(|r| r.approach.as_str())
        .collect();
    println!("\nPareto frontier (* - none is both faster and more accurate):");
    println!("  {}", frontier.join(", "));
}

/// Map a NaN (e.g. a degenerate single-class AUC) to 0.0 so the JSON is valid.
fn nan_to_zero(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else {
        x
    }
}
