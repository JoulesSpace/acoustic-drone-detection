//! `doa-bench` - sweep true azimuth × SNR × trials through the DoA pipeline and
//! report angular error.
//!
//! For each `(angle, snr, trial)` the simulator produces `M` noisy ULA channels;
//! the estimator recovers the azimuth; we accumulate the signed error. Output is
//! a summary table (overall + per-SNR MAE/RMSE in degrees) plus a JSON dump at
//! `benchmarks/results/doa.json` with every estimated-vs-true pair for plotting.
//!
//! Everything is deterministic: trial `t` at `(angle, snr)` uses a seed derived
//! from the CLI `--seed`, the angle, the SNR and the trial index.

use std::error::Error;
use std::path::PathBuf;

use clap::Parser;
use serde::Serialize;

use drone_doa::azimuth::estimate_azimuth;
use drone_doa::gcc_phat::GccConfig;
use drone_doa::geometry::UlaGeometry;
use drone_doa::sim::{simulate_array, DroneSource, SimConfig};

#[derive(Parser)]
#[command(
    name = "doa-bench",
    version,
    about = "Benchmark acoustic DoA (GCC-PHAT + ULA) in simulation",
    // Azimuths are negative; let `--angle-min -80` parse without `=`.
    allow_negative_numbers = true
)]
struct Cli {
    /// Number of microphones in the ULA.
    #[arg(long, default_value_t = 4)]
    mics: usize,
    /// Inter-mic spacing in metres. Default 0.043 m ≈ λ/2 at 4 kHz.
    #[arg(long, default_value_t = 0.043)]
    spacing: f32,
    /// Sample rate in Hz.
    #[arg(long, default_value_t = 16_000)]
    sample_rate: u32,
    /// Samples per channel per trial (also the GCC-PHAT block length).
    #[arg(long, default_value_t = 2048)]
    num_samples: usize,
    /// First true azimuth in the sweep (degrees).
    #[arg(long, default_value_t = -80.0)]
    angle_min: f32,
    /// Last true azimuth in the sweep (degrees).
    #[arg(long, default_value_t = 80.0)]
    angle_max: f32,
    /// Azimuth sweep step (degrees).
    #[arg(long, default_value_t = 10.0)]
    angle_step: f32,
    /// SNR levels to sweep, in dB (repeat the flag or comma-separate).
    #[arg(long, value_delimiter = ',', default_values_t = [20.0, 10.0, 0.0])]
    snr: Vec<f32>,
    /// Trials per (angle, snr) cell.
    #[arg(long, default_value_t = 20)]
    trials: usize,
    /// Fundamental frequency of the synthetic drone source (Hz).
    #[arg(long, default_value_t = 120.0)]
    fundamental: f32,
    /// Base RNG seed.
    #[arg(long, default_value_t = 1)]
    seed: u32,
    /// Output JSON path.
    #[arg(long, default_value = "benchmarks/results/doa.json")]
    out: PathBuf,
}

#[derive(Serialize)]
struct Trial {
    angle_true: f32,
    angle_est: f32,
    snr_db: f32,
    error_deg: f32,
}

#[derive(Serialize)]
struct CellError {
    angle_true: f32,
    snr_db: f32,
    mae: f32,
    rmse: f32,
}

#[derive(Serialize)]
struct SnrSummary {
    snr_db: f32,
    mae: f32,
    rmse: f32,
    n: usize,
}

#[derive(Serialize)]
struct DoaResult {
    mics: usize,
    spacing_m: f32,
    sample_rate: u32,
    num_samples: usize,
    fundamental_hz: f32,
    aliasing_free_max_hz: f32,
    angle_min: f32,
    angle_max: f32,
    angle_step: f32,
    trials_per_cell: usize,
    overall_mae: f32,
    overall_rmse: f32,
    per_snr: Vec<SnrSummary>,
    per_cell: Vec<CellError>,
    trials: Vec<Trial>,
}

fn rmse(errs: &[f32]) -> f32 {
    if errs.is_empty() {
        return 0.0;
    }
    (errs.iter().map(|e| e * e).sum::<f32>() / errs.len() as f32).sqrt()
}

fn mae(errs: &[f32]) -> f32 {
    if errs.is_empty() {
        return 0.0;
    }
    errs.iter().map(|e| e.abs()).sum::<f32>() / errs.len() as f32
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    let geom = UlaGeometry::new(cli.mics, cli.spacing);
    let src = DroneSource {
        fundamental_hz: cli.fundamental,
        ..DroneSource::default()
    };
    let cfg_template = GccConfig::default();

    let angles = sweep(cli.angle_min, cli.angle_max, cli.angle_step);

    println!(
        "ULA: {} mics @ {:.3} m  (alias-free < {:.0} Hz),  source {:.0} Hz,  {} samples @ {} Hz",
        cli.mics,
        cli.spacing,
        geom.aliasing_free_max_hz(),
        cli.fundamental,
        cli.num_samples,
        cli.sample_rate,
    );
    println!(
        "sweep: {:.0}°..{:.0}° step {:.0}°  ×  SNR {:?} dB  ×  {} trials\n",
        cli.angle_min, cli.angle_max, cli.angle_step, cli.snr, cli.trials
    );

    let mut trials = Vec::new();
    let mut per_cell = Vec::new();
    let mut all_errs = Vec::new();
    // SNR -> accumulated errors.
    let mut snr_errs: Vec<(f32, Vec<f32>)> = cli.snr.iter().map(|&s| (s, Vec::new())).collect();

    for &snr in &cli.snr {
        for (ai, &angle) in angles.iter().enumerate() {
            let mut cell_errs = Vec::with_capacity(cli.trials);
            for t in 0..cli.trials {
                let seed = cli
                    .seed
                    .wrapping_mul(1_000_003)
                    .wrapping_add((ai as u32).wrapping_mul(9_973))
                    .wrapping_add((snr as i32 as u32).wrapping_mul(101))
                    .wrapping_add(t as u32 + 1);
                let sim = SimConfig {
                    sample_rate: cli.sample_rate,
                    num_samples: cli.num_samples,
                    true_azimuth_deg: angle,
                    snr_db: snr,
                    seed,
                };
                let ch = simulate_array(&src, &geom, &sim);
                let refs: Vec<&[f32]> = ch.iter().map(|c| c.as_slice()).collect();
                let est = estimate_azimuth(&refs, &geom, cli.sample_rate, &cfg_template);
                let err = est.azimuth_deg - angle;
                cell_errs.push(err);
                all_errs.push(err);
                trials.push(Trial {
                    angle_true: angle,
                    angle_est: est.azimuth_deg,
                    snr_db: snr,
                    error_deg: err,
                });
            }
            if let Some(slot) = snr_errs.iter_mut().find(|(s, _)| *s == snr) {
                slot.1.extend_from_slice(&cell_errs);
            }
            per_cell.push(CellError {
                angle_true: angle,
                snr_db: snr,
                mae: mae(&cell_errs),
                rmse: rmse(&cell_errs),
            });
        }
    }

    // Per-SNR summary table.
    println!(
        "{:>8}  {:>8}  {:>8}  {:>6}",
        "SNR(dB)", "MAE(°)", "RMSE(°)", "n"
    );
    println!("{}", "-".repeat(36));
    let mut per_snr = Vec::new();
    for (snr, errs) in &snr_errs {
        println!(
            "{:>8.0}  {:>8.3}  {:>8.3}  {:>6}",
            snr,
            mae(errs),
            rmse(errs),
            errs.len()
        );
        per_snr.push(SnrSummary {
            snr_db: *snr,
            mae: mae(errs),
            rmse: rmse(errs),
            n: errs.len(),
        });
    }
    println!("{}", "-".repeat(36));
    println!(
        "{:>8}  {:>8.3}  {:>8.3}  {:>6}",
        "all",
        mae(&all_errs),
        rmse(&all_errs),
        all_errs.len()
    );

    let result = DoaResult {
        mics: cli.mics,
        spacing_m: cli.spacing,
        sample_rate: cli.sample_rate,
        num_samples: cli.num_samples,
        fundamental_hz: cli.fundamental,
        aliasing_free_max_hz: geom.aliasing_free_max_hz(),
        angle_min: cli.angle_min,
        angle_max: cli.angle_max,
        angle_step: cli.angle_step,
        trials_per_cell: cli.trials,
        overall_mae: mae(&all_errs),
        overall_rmse: rmse(&all_errs),
        per_snr,
        per_cell,
        trials,
    };

    if let Some(parent) = cli.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out, serde_json::to_string_pretty(&result)?)?;
    println!("\nresults written to {}", cli.out.display());
    Ok(())
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
