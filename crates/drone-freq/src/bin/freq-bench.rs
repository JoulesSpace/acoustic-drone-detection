//! `freq-bench` - benchmark the blade-pass f0 estimator.
//!
//! Two parts:
//!
//! 1. **Synthetic accuracy**: deterministically generate harmonic-stack clips
//!    with a *known* f0 swept over 80-250 Hz at several SNR levels, estimate f0,
//!    and report MAE, median absolute error (Hz and %), and the octave-error
//!    rate (estimates near 0.5× or 2× the truth) per SNR.
//!
//! 2. **Real report**: load DADS drone clips (label 1) and characterize the
//!    distribution of estimated f0 - median, IQR, histogram - with no ground
//!    truth.
//!
//! Results are written to `benchmarks/results/freq.json` and summarized to
//! stdout. Everything is deterministic.

use std::error::Error;
use std::f32::consts::PI;
use std::path::PathBuf;

use clap::Parser;
use drone_bench::dataset::Dataset;
use drone_freq::{estimate_f0_conf, F0_MAX_HZ, F0_MIN_HZ};
use serde::Serialize;

#[derive(Parser)]
#[command(
    name = "freq-bench",
    version,
    about = "Benchmark the drone blade-pass f0 estimator"
)]
struct Cli {
    /// DADS dataset root containing `labels.csv` (header `path,label`). If
    /// omitted or unreadable, only the synthetic benchmark runs.
    #[arg(long)]
    data: Option<PathBuf>,
    /// Manifest filename inside `--data`.
    #[arg(long, default_value = "labels.csv")]
    manifest: String,
    /// Sample rate for synthetic clips (Hz).
    #[arg(long, default_value_t = 16_000)]
    sample_rate: u32,
    /// Duration of each synthetic clip (seconds).
    #[arg(long, default_value_t = 0.75)]
    synth_secs: f32,
    /// Output JSON path.
    #[arg(long, default_value = "benchmarks/results/freq.json")]
    out: PathBuf,
    /// Assumed blade count for the reported RPM context.
    #[arg(long, default_value_t = 2)]
    blades: u32,
}

/// Per-SNR synthetic error summary.
#[derive(Serialize)]
struct SnrResult {
    snr_db: f32,
    n: usize,
    mae_hz: f32,
    median_abs_err_hz: f32,
    mae_pct: f32,
    median_abs_err_pct: f32,
    octave_error_rate: f32,
}

/// Real-data f0 distribution summary.
#[derive(Serialize)]
struct RealResult {
    n_clips: usize,
    blades_assumed: u32,
    median_f0_hz: f32,
    iqr_lo_hz: f32,
    iqr_hi_hz: f32,
    min_hz: f32,
    max_hz: f32,
    median_rotor_rpm: f32,
    /// Histogram bin edges (Hz), length = counts.len() + 1.
    hist_edges_hz: Vec<f32>,
    hist_counts: Vec<u32>,
    /// All per-clip f0 estimates (Hz), for downstream plotting.
    f0_hz: Vec<f32>,
}

#[derive(Serialize)]
struct Report {
    band_hz: [f32; 2],
    synth_sample_rate: u32,
    synth_secs: f32,
    synth_truth_f0_hz: Vec<f32>,
    synth: Vec<SnrResult>,
    real: Option<RealResult>,
}

/// Deterministic xorshift32 PRNG (no rng-crate dependency).
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn bipolar(&mut self) -> f32 {
        (self.next() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Generate a deterministic drone-like harmonic-stack clip with a known f0,
/// then add white noise to hit `snr_db`. Each (f0, snr) pair is seeded so the
/// whole benchmark is reproducible.
fn synth_clip(f0: f32, sr: u32, secs: f32, snr_db: f32, seed: u32) -> Vec<f32> {
    let n = (secs * sr as f32) as usize;
    let mut rng = Rng(seed.max(1));
    // 1/h harmonic roll-off plus mild amplitude modulation, like a real rotor.
    let am_hz = 8.0;
    let mut clip: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            let am = 1.0 + 0.2 * (2.0 * PI * am_hz * t).sin();
            let mut v = 0.0;
            for h in 1..=8 {
                v += (1.0 / h as f32) * (2.0 * PI * f0 * h as f32 * t).sin();
            }
            0.5 * am * v
        })
        .collect();

    // Add white noise at the requested SNR (skip for very high SNR sentinel).
    if snr_db.is_finite() {
        let ps: f32 = clip.iter().map(|v| v * v).sum::<f32>() / n.max(1) as f32;
        if ps > 0.0 {
            let pn = ps / 10f32.powf(snr_db / 10.0);
            let a = (3.0 * pn).sqrt(); // uniform[-a,a] has variance a^2/3
            for s in clip.iter_mut() {
                *s += a * rng.bipolar();
            }
        }
    }
    clip
}

/// Median of a slice (sorts a copy). Returns NaN for empty input.
fn median(xs: &[f32]) -> f32 {
    if xs.is_empty() {
        return f32::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        0.5 * (v[m - 1] + v[m])
    }
}

/// Percentile (linear, 0..=100) of a slice. Returns NaN for empty input.
fn percentile(xs: &[f32], p: f32) -> f32 {
    if xs.is_empty() {
        return f32::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rank = (p / 100.0) * (v.len() - 1) as f32;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let frac = rank - lo as f32;
        v[lo] * (1.0 - frac) + v[hi] * frac
    }
}

fn run_synth(sr: u32, secs: f32) -> (Vec<f32>, Vec<SnrResult>) {
    // Sweep f0 over 80-250 Hz (covers small-multirotor blade-pass).
    let truths: Vec<f32> = (0..18).map(|i| 80.0 + i as f32 * 10.0).collect();
    // SNR levels (dB). `INFINITY` == clean (no added noise).
    let snrs = [f32::INFINITY, 20.0, 10.0, 5.0, 0.0];

    let mut results = Vec::new();
    for (si, &snr) in snrs.iter().enumerate() {
        let mut abs_hz = Vec::new();
        let mut abs_pct = Vec::new();
        let mut octave = 0usize;
        for (ti, &truth) in truths.iter().enumerate() {
            let seed = 1 + (si as u32) * 1000 + ti as u32;
            let clip = synth_clip(truth, sr, secs, snr, seed);
            let est = estimate_f0_conf(&clip, sr).f0_hz;
            if !est.is_finite() {
                // Count a non-estimate as a full-scale miss so it can't be hidden.
                abs_hz.push(truth);
                abs_pct.push(100.0);
                continue;
            }
            let e = (est - truth).abs();
            abs_hz.push(e);
            abs_pct.push(100.0 * e / truth);
            // Octave error: estimate near half or double the truth.
            let ratio = est / truth;
            if (ratio - 0.5).abs() < 0.1 || (ratio - 2.0).abs() < 0.2 {
                octave += 1;
            }
        }
        let n = truths.len();
        let mae_hz = abs_hz.iter().sum::<f32>() / n as f32;
        let mae_pct = abs_pct.iter().sum::<f32>() / n as f32;
        results.push(SnrResult {
            snr_db: if snr.is_finite() { snr } else { f32::INFINITY },
            n,
            mae_hz,
            median_abs_err_hz: median(&abs_hz),
            mae_pct,
            median_abs_err_pct: median(&abs_pct),
            octave_error_rate: octave as f32 / n as f32,
        });
    }
    (truths, results)
}

fn run_real(
    dir: &std::path::Path,
    manifest: &str,
    blades: u32,
) -> Result<RealResult, Box<dyn Error>> {
    let manifest_path = dir.join(manifest);
    let ds = Dataset::load_csv(dir, &manifest_path)?;
    let mut f0s = Vec::new();
    for s in ds.samples.iter().filter(|s| s.label == 1) {
        let est = estimate_f0_conf(&s.samples, s.sample_rate).f0_hz;
        if est.is_finite() {
            f0s.push(est);
        }
    }
    if f0s.is_empty() {
        return Err("no drone clips produced a finite f0 estimate".into());
    }

    let med = median(&f0s);
    let iqr_lo = percentile(&f0s, 25.0);
    let iqr_hi = percentile(&f0s, 75.0);
    let min_hz = f0s.iter().copied().fold(f32::INFINITY, f32::min);
    let max_hz = f0s.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Histogram over the search band in 10 Hz bins.
    let bin_w = 10.0_f32;
    let n_bins = ((F0_MAX_HZ - F0_MIN_HZ) / bin_w).ceil() as usize;
    let mut counts = vec![0u32; n_bins];
    let edges: Vec<f32> = (0..=n_bins).map(|i| F0_MIN_HZ + i as f32 * bin_w).collect();
    for &f in &f0s {
        let idx = (((f - F0_MIN_HZ) / bin_w) as usize).min(n_bins - 1);
        counts[idx] += 1;
    }

    Ok(RealResult {
        n_clips: f0s.len(),
        blades_assumed: blades,
        median_f0_hz: med,
        iqr_lo_hz: iqr_lo,
        iqr_hi_hz: iqr_hi,
        min_hz,
        max_hz,
        median_rotor_rpm: 60.0 * med / blades.max(1) as f32,
        hist_edges_hz: edges,
        hist_counts: counts,
        f0_hz: f0s,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    println!("=== drone-freq: blade-pass f0 estimator benchmark ===");
    println!("search band: {F0_MIN_HZ}-{F0_MAX_HZ} Hz\n");

    // --- synthetic accuracy ---
    println!("synthetic accuracy (f0 swept 80-250 Hz, deterministic):");
    let (truths, synth) = run_synth(cli.sample_rate, cli.synth_secs);
    println!(
        "{:>8}  {:>5}  {:>8}  {:>10}  {:>8}  {:>9}  {:>10}",
        "SNR(dB)", "n", "MAE(Hz)", "med|e|(Hz)", "MAE(%)", "med|e|(%)", "octave-rate"
    );
    for r in &synth {
        let snr_str = if r.snr_db.is_finite() {
            format!("{:.0}", r.snr_db)
        } else {
            "clean".to_string()
        };
        println!(
            "{:>8}  {:>5}  {:>8.2}  {:>10.2}  {:>8.2}  {:>9.2}  {:>10.2}",
            snr_str,
            r.n,
            r.mae_hz,
            r.median_abs_err_hz,
            r.mae_pct,
            r.median_abs_err_pct,
            r.octave_error_rate
        );
    }

    // --- real report ---
    let real = match &cli.data {
        Some(dir) => match run_real(dir, &cli.manifest, cli.blades) {
            Ok(r) => {
                println!("\nreal DADS drone clips (label 1, no ground truth):");
                println!("  clips estimated : {}", r.n_clips);
                println!("  median f0       : {:.1} Hz", r.median_f0_hz);
                println!(
                    "  IQR             : {:.1} - {:.1} Hz",
                    r.iqr_lo_hz, r.iqr_hi_hz
                );
                println!("  range           : {:.1} - {:.1} Hz", r.min_hz, r.max_hz);
                println!(
                    "  rotor RPM (B={}) : {:.0} rpm  (rate ≈ f0/B)",
                    r.blades_assumed, r.median_rotor_rpm
                );
                println!("  histogram (10 Hz bins over band):");
                for (i, &c) in r.hist_counts.iter().enumerate() {
                    if c == 0 {
                        continue;
                    }
                    let lo = r.hist_edges_hz[i];
                    let hi = r.hist_edges_hz[i + 1];
                    let bar = "#".repeat((c as usize).min(60));
                    println!("    {lo:>5.0}-{hi:<5.0} {c:>4}  {bar}");
                }
                Some(r)
            }
            Err(e) => {
                eprintln!("\nreal report skipped: {e}");
                None
            }
        },
        None => {
            println!("\nreal report skipped: no --data provided");
            None
        }
    };

    let report = Report {
        band_hz: [F0_MIN_HZ, F0_MAX_HZ],
        synth_sample_rate: cli.sample_rate,
        synth_secs: cli.synth_secs,
        synth_truth_f0_hz: truths,
        synth,
        real,
    };

    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out, serde_json::to_string_pretty(&report)?)?;
    println!("\nresults written to {}", cli.out.display());
    Ok(())
}
