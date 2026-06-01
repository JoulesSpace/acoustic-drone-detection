//! `ratesweep` - sample-rate and bit-depth robustness sweep for the detector
//! pipeline, run over the real DADS dataset (16 kHz mono native).
//!
//! Answers two practical mic/ADC questions:
//!
//! 1. **Sample rate.** Can we run higher rates, and what do cheap low-rate mics
//!    cost us? For each target rate in {8000, 16000, 22050, 44100} we resample
//!    every clip from the 16 kHz native rate to the target, then run the
//!    detector pipeline *at that rate* (the pipeline is rate-aware via
//!    `sample_rate`). Honest caveat: DADS is 16 kHz native, so upsampling adds
//!    no new spectral information above 8 kHz - this measures pipeline
//!    behaviour and the fixed-1024-frame resolution effect (bin width = rate /
//!    1024 grows with rate), not extra signal. Downsampling to 8 kHz *does*
//!    discard real content above 4 kHz, so its result is physically meaningful.
//!
//! 2. **Bit depth.** What do cheap ADCs / edge mics cost? For each bit depth in
//!    {16, 12, 10, 8, 6, 4} we uniformly quantize the 16 kHz float clips, then
//!    run detection. Models low-resolution capture (quantization noise floor).
//!
//! Resampler: a windowed-sinc (Lanczos, a = 8) fractional resampler - correct
//! and band-limited. For the 16 kHz → 8 kHz downsample the sinc kernel's cutoff
//! is set to the *output* Nyquist (4 kHz), so it doubles as the anti-alias
//! low-pass; see `resample`.
//!
//! Quantizer: uniform mid-tread quantization to `2^(b-1)` levels per sign
//! (i.e. step = 1 / 2^(b-1) over the [-1, 1] range), with optional TPDF dither
//! at ±1 LSB before rounding (`--dither`); see `quantize`. Dither is off by
//! default so the headline run is deterministic and reproducible.
//!
//! Output: `benchmarks/results/ratesweep.json` plus printed tables. Everything
//! is deterministic for a fixed seed (and with dither off).

use std::error::Error;
use std::f32::consts::PI;
use std::path::PathBuf;

use clap::Parser;
use drone_bench::dataset::{Dataset, Sample};
use drone_bench::metrics::{evaluate, Metrics};
use drone_bench::{approaches, Approach};
use serde::Serialize;

/// Default representative detector subset (keeps runtime sane). Covers a
/// heuristic (band_ratio), a gate (spectral_gate), a harmonic method (hps), a
/// supervised baseline (mfcc_lr), and a fusion model (feature_fusion).
const DEFAULT_DETECTORS: &[&str] = &[
    "band_ratio",
    "spectral_gate",
    "hps",
    "mfcc_lr",
    "feature_fusion",
];

/// Target sample rates for the rate sweep (Hz). 16 kHz is the native rate.
const RATES: &[u32] = &[8_000, 16_000, 22_050, 44_100];

/// Target bit depths for the bit-depth sweep.
const BITS: &[u32] = &[16, 12, 10, 8, 6, 4];

#[derive(Parser)]
#[command(
    name = "ratesweep",
    version,
    about = "Sample-rate and bit-depth robustness sweep on DADS"
)]
struct Cli {
    /// DADS dataset root containing `labels.csv` (header `path,label`).
    #[arg(
        long,
        default_value = "C:/Users/julia/Development/acoustic-drone-detection/data/dads"
    )]
    data: PathBuf,
    /// Manifest filename inside `--data`.
    #[arg(long, default_value = "labels.csv")]
    manifest: String,
    /// Comma-separated detector subset (defaults to a representative set).
    #[arg(long, value_delimiter = ',')]
    detectors: Option<Vec<String>>,
    /// Fraction of each class used for training (single stratified split).
    #[arg(long, default_value_t = 0.5)]
    train_frac: f32,
    /// Decision threshold for the headline confusion metrics.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,
    /// RNG seed (split + dither).
    #[arg(long, default_value_t = 1)]
    seed: u32,
    /// Add TPDF dither (±1 LSB) before quantizing. Off by default (deterministic).
    #[arg(long)]
    dither: bool,
    /// Output JSON path.
    #[arg(long, default_value = "benchmarks/results/ratesweep.json")]
    out: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let detectors: Vec<String> = cli
        .detectors
        .clone()
        .unwrap_or_else(|| DEFAULT_DETECTORS.iter().map(|s| s.to_string()).collect());
    // Validate names up front so a typo fails fast rather than silently dropping.
    let known: Vec<String> = approaches::all()
        .iter()
        .map(|a| a.name().to_string())
        .collect();
    for d in &detectors {
        if !known.contains(d) {
            return Err(format!("unknown detector '{d}'; known: {}", known.join(", ")).into());
        }
    }

    let manifest_path = cli.data.join(&cli.manifest);
    println!("=== ratesweep: sample-rate + bit-depth robustness on DADS ===");
    println!("loading dataset from {}", manifest_path.display());
    let ds = Dataset::load_csv(&cli.data, &manifest_path)?;
    if ds.is_empty() {
        return Err("dataset is empty".into());
    }
    let native_sr = ds.samples[0].sample_rate;
    println!(
        "dataset: {} clips ({} drone) - native {} Hz mono\n",
        ds.len(),
        ds.n_pos(),
        native_sr
    );
    println!("detectors: {}", detectors.join(", "));
    println!("rates    : {RATES:?} Hz  (native = {native_sr})");
    println!(
        "bits     : {BITS:?}  (dither: {})\n",
        if cli.dither { "TPDF ±1 LSB" } else { "off" }
    );

    // One stratified split, reused across every condition so the test set is
    // identical and the curves are directly comparable.
    let (train, test) = ds.split(cli.train_frac, cli.seed);

    let mut rate_rows: Vec<RateRow> = Vec::new();
    let mut bit_rows: Vec<BitRow> = Vec::new();

    // ---- sample-rate sweep ----
    println!("--- sample-rate sweep (resample {native_sr} Hz -> target) ---");
    print_header("rate(Hz)");
    for &rate in RATES {
        let train_r = map_clips(&train, |s| Sample {
            id: s.id.clone(),
            samples: resample(&s.samples, s.sample_rate, rate),
            sample_rate: rate,
            label: s.label,
        });
        let test_r = map_clips(&test, |s| Sample {
            id: s.id.clone(),
            samples: resample(&s.samples, s.sample_rate, rate),
            sample_rate: rate,
            label: s.label,
        });
        let bin_hz = rate as f32 / 1024.0;
        for det in &detectors {
            let m = run_one(det, &train_r, &test_r, cli.threshold);
            print_row(&format!("{rate}"), det, &m);
            rate_rows.push(RateRow {
                rate_hz: rate,
                bin_hz,
                detector: det.clone(),
                metrics: m,
            });
        }
    }

    // ---- bit-depth sweep (always at native rate) ----
    println!("\n--- bit-depth sweep (quantize at {native_sr} Hz) ---");
    print_header("bits");
    for &bits in BITS {
        // Per-bit-depth dither stream is seeded so a dithered run is still
        // reproducible for a fixed seed.
        let train_q = map_clips(&train, |s| Sample {
            id: s.id.clone(),
            samples: quantize(
                &s.samples,
                bits,
                cli.dither,
                seed_for(cli.seed, bits, &s.id),
            ),
            sample_rate: s.sample_rate,
            label: s.label,
        });
        let test_q = map_clips(&test, |s| Sample {
            id: s.id.clone(),
            samples: quantize(
                &s.samples,
                bits,
                cli.dither,
                seed_for(cli.seed, bits, &s.id),
            ),
            sample_rate: s.sample_rate,
            label: s.label,
        });
        for det in &detectors {
            let m = run_one(det, &train_q, &test_q, cli.threshold);
            print_row(&format!("{bits}"), det, &m);
            bit_rows.push(BitRow {
                bits,
                levels: 1u32 << (bits - 1),
                detector: det.clone(),
                metrics: m,
            });
        }
    }

    let report = Report {
        dataset: manifest_path.display().to_string(),
        native_sample_rate: native_sr,
        n_test: test.len(),
        n_test_pos: test.iter().filter(|s| s.label == 1).count(),
        frame_size: 1024,
        train_frac: cli.train_frac,
        threshold: cli.threshold,
        seed: cli.seed,
        dither: cli.dither,
        detectors: detectors.clone(),
        resampler: "windowed-sinc (Lanczos, a=8); downsample cutoff = output Nyquist (anti-alias)"
            .to_string(),
        quantizer: "uniform mid-tread, 2^(b-1) levels/sign, step = 2^-(b-1); optional TPDF dither"
            .to_string(),
        rates: RATES.to_vec(),
        bits: BITS.to_vec(),
        rate_sweep: rate_rows,
        bit_sweep: bit_rows,
    };

    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out, serde_json::to_string_pretty(&report)?)?;
    println!("\nresults written to {}", cli.out.display());
    Ok(())
}

/// Build a fresh instance of the named approach.
fn instantiate(name: &str) -> Box<dyn Approach> {
    approaches::all()
        .into_iter()
        .find(|a| a.name() == name)
        .unwrap_or_else(|| panic!("unknown approach {name}"))
}

/// Fit on `train`, score `test`, return the metric bundle at `threshold`.
fn run_one(name: &str, train: &[Sample], test: &[Sample], threshold: f32) -> Metrics {
    let mut approach = instantiate(name);
    approach.fit(train);
    let scored: Vec<(f32, u8)> = test
        .iter()
        .map(|s| {
            (
                approach.score(&s.samples, s.sample_rate).clamp(0.0, 1.0),
                s.label,
            )
        })
        .collect();
    evaluate(&scored, threshold)
}

/// Map a transform over a clip slice.
fn map_clips(clips: &[Sample], f: impl Fn(&Sample) -> Sample) -> Vec<Sample> {
    clips.iter().map(f).collect()
}

// --------------------------------------------------------------------------
// Resampling: windowed-sinc (Lanczos) fractional resampler.
// --------------------------------------------------------------------------

/// Lanczos kernel half-width in input samples (at the sinc's own scale).
const LANCZOS_A: i64 = 8;

/// Resample `x` from `from_hz` to `to_hz` with a windowed-sinc (Lanczos)
/// kernel. When downsampling (`to_hz < from_hz`) the kernel is scaled by the
/// ratio so its cutoff sits at the *output* Nyquist, which band-limits the
/// input first - i.e. the anti-alias low-pass is built into the same kernel.
/// When upsampling the cutoff stays at the input Nyquist (no new content is
/// invented). Identity when the rates match.
fn resample(x: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || x.is_empty() {
        return x.to_vec();
    }
    let ratio = to_hz as f64 / from_hz as f64; // output samples per input sample
    let n_out = ((x.len() as f64) * ratio).round() as usize;
    // Kernel scale: <1 when downsampling (widens kernel, lowers cutoff).
    let scale = ratio.min(1.0);
    let half = (LANCZOS_A as f64 / scale).ceil() as i64;
    let mut out = Vec::with_capacity(n_out);
    for i in 0..n_out {
        // Position in input-sample units for output sample i.
        let center = i as f64 / ratio;
        let c0 = center.floor() as i64;
        let mut acc = 0.0_f64;
        let mut wsum = 0.0_f64;
        for k in (c0 - half)..=(c0 + half) {
            let dist = (center - k as f64) * scale;
            let w = lanczos(dist, LANCZOS_A) * scale;
            if w == 0.0 {
                continue;
            }
            let xi = k.clamp(0, x.len() as i64 - 1) as usize;
            acc += w * x[xi] as f64;
            wsum += w;
        }
        // Normalize by the realized window sum to keep DC gain at unity even at
        // the clip edges (where the kernel is clipped).
        let v = if wsum.abs() > 1e-12 { acc / wsum } else { 0.0 };
        out.push(v as f32);
    }
    out
}

/// Lanczos windowed sinc: sinc(x) * sinc(x / a) for |x| < a, else 0.
fn lanczos(x: f64, a: i64) -> f64 {
    if x == 0.0 {
        return 1.0;
    }
    let af = a as f64;
    if x.abs() >= af {
        return 0.0;
    }
    let px = PI as f64 * x;
    (px.sin() / px) * ((px / af).sin() / (px / af))
}

// --------------------------------------------------------------------------
// Bit-depth quantization.
// --------------------------------------------------------------------------

/// Uniformly quantize `x` (in [-1, 1]) to `bits`-bit resolution: a mid-tread
/// quantizer with `2^(bits-1)` steps per sign, i.e. step = 1 / 2^(bits-1).
/// With `dither`, add TPDF (triangular, sum of two independent uniforms) noise
/// of ±1 LSB peak before rounding - the standard de-correlating dither - using
/// a deterministic per-clip RNG seed.
fn quantize(x: &[f32], bits: u32, dither: bool, seed: u32) -> Vec<f32> {
    let levels = (1u32 << (bits - 1)) as f32; // steps per sign
    let step = 1.0 / levels;
    let mut rng = seed.max(1);
    x.iter()
        .map(|&v| {
            let mut s = v;
            if dither {
                // TPDF in [-1, 1] LSB: u1 + u2, each uniform in [-0.5, 0.5] LSB.
                let u1 = next_unit(&mut rng) - 0.5;
                let u2 = next_unit(&mut rng) - 0.5;
                s += (u1 + u2) * step;
            }
            // Mid-tread rounding to the nearest level, clamped to full scale.
            let q = (s / step).round() * step;
            q.clamp(-1.0, 1.0)
        })
        .collect()
}

/// Advance an xorshift32 RNG and return a uniform in [0, 1).
fn next_unit(state: &mut u32) -> f32 {
    let mut s = *state;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *state = s;
    s as f32 / u32::MAX as f32
}

/// Deterministic per-(seed, bits, clip-id) dither seed so dithered runs are
/// reproducible. A small FNV-1a-style hash over the clip id mixes in identity.
fn seed_for(seed: u32, bits: u32, id: &str) -> u32 {
    let mut h: u32 = 2166136261;
    for b in id.bytes() {
        h ^= b as u32;
        h = h.wrapping_mul(16777619);
    }
    h ^ seed
        .wrapping_mul(2654435761)
        .wrapping_add(bits.wrapping_mul(40503))
}

// --------------------------------------------------------------------------
// Pretty-printing.
// --------------------------------------------------------------------------

fn print_header(first: &str) {
    println!(
        "{first:>9} {:<16} {:>7} {:>7} {:>7} {:>9} {:>9}",
        "detector", "acc", "F1", "F1*", "ROC-AUC", "PR-AUC"
    );
    println!("{}", "-".repeat(9 + 1 + 16 + 8 * 5));
}

fn print_row(first: &str, det: &str, m: &Metrics) {
    println!(
        "{first:>9} {det:<16} {:>7.3} {:>7.3} {:>7.3} {:>9.3} {:>9.3}",
        m.accuracy, m.f1, m.f1_best, m.roc_auc, m.pr_auc
    );
}

// --------------------------------------------------------------------------
// JSON report types.
// --------------------------------------------------------------------------

#[derive(Serialize)]
struct RateRow {
    rate_hz: u32,
    /// FFT bin width at this rate for the fixed 1024-point frame (Hz/bin).
    bin_hz: f32,
    detector: String,
    #[serde(flatten)]
    metrics: Metrics,
}

#[derive(Serialize)]
struct BitRow {
    bits: u32,
    /// Quantization levels per sign (2^(bits-1)).
    levels: u32,
    detector: String,
    #[serde(flatten)]
    metrics: Metrics,
}

#[derive(Serialize)]
struct Report {
    dataset: String,
    native_sample_rate: u32,
    n_test: usize,
    n_test_pos: usize,
    frame_size: u32,
    train_frac: f32,
    threshold: f32,
    seed: u32,
    dither: bool,
    detectors: Vec<String>,
    resampler: String,
    quantizer: String,
    rates: Vec<u32>,
    bits: Vec<u32>,
    rate_sweep: Vec<RateRow>,
    bit_sweep: Vec<BitRow>,
}
