//! Benchmark binary for `drone-bss`.
//!
//! Two questions, both answered over a sweep of seeds and conditions:
//!
//! 1. **Separation quality.** For each scene (2 drones; drone + noise;
//!    drone + tonal interferer) we mix the known sources, run FastICA, and report
//!    the mean **SIR improvement (dB)** of the recovered drone component over the
//!    best raw channel, plus how often the drone is recovered (best-match
//!    correlation above threshold).
//!
//! 2. **Detection improvement (the payoff).** We take a real detector from the
//!    sibling `drone-bench` crate (`hps` by default, `spectral_gate` as a check),
//!    calibrate it once on `drone_bench::dataset::synth`, and compare its
//!    confidence on the drone when it scores (a) the raw mixed channel vs (b) the
//!    FastICA-separated drone component. The lift is BSS earning its keep as a
//!    detection front-end in multi-source scenes.
//!
//! Results are written to `benchmarks/results/bss.json` and printed as a table.
//! Everything is seeded, so reruns are identical.

use std::path::PathBuf;

use serde::Serialize;

use drone_bench::approaches;
use drone_bench::dataset::Dataset;
use drone_bench::Approach;

use drone_bss::fastica::{fastica, FastIcaConfig};
use drone_bss::metrics::drone_separation_quality;
use drone_bss::mix::{scene, ExtraSource, MixConfig, Mixture};

/// Seeds swept per condition. More seeds tighten the means; 12 is plenty for a
/// quick-running benchmark.
const SEEDS: &[u64] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];

/// Recovery threshold on best-match absolute correlation.
const RECOVERY_THRESHOLD: f64 = 0.9;

/// One mixing scene definition.
struct Condition {
    name: &'static str,
    drone_f0s: &'static [f64],
    extra: Vec<ExtraSource>,
    /// Drone level relative to the interferers (dB). Negative = quiet drone
    /// buried in a louder scene, the regime where single-mic detection fails.
    drone_gain_db: f64,
}

#[derive(Serialize)]
struct ConditionResult {
    condition: String,
    mics: usize,
    sources: usize,
    seeds: usize,
    /// Mean SIR improvement (dB) of the recovered drone vs the best raw channel.
    mean_sir_improvement_db: f64,
    /// Mean SIR (dB) of the recovered drone component (separated).
    mean_sir_separated_db: f64,
    /// Mean best raw-channel SIR (dB) for the drone.
    mean_sir_mixture_db: f64,
    /// Mean absolute correlation of the recovered drone to the clean source.
    mean_drone_correlation: f64,
    /// Fraction of seeds where the drone was recovered (corr >= threshold).
    drone_recovery_rate: f64,
    /// Fraction of runs where FastICA reported convergence.
    convergence_rate: f64,
}

#[derive(Serialize)]
struct DetectionResult {
    detector: String,
    condition: String,
    seeds: usize,
    /// Mean detector confidence on the raw mixed channel (drone present).
    mean_confidence_mixture: f64,
    /// Mean detector confidence on the FastICA-separated drone component.
    mean_confidence_separated: f64,
    /// Mean lift (separated minus mixture confidence).
    mean_confidence_lift: f64,
    /// Recall on the raw mixed channel at threshold 0.5.
    recall_mixture: f64,
    /// Recall on the separated drone component at threshold 0.5.
    recall_separated: f64,
}

#[derive(Serialize)]
struct BssReport {
    description: String,
    caveats: Vec<String>,
    recovery_threshold: f64,
    seeds: Vec<u64>,
    separation: Vec<ConditionResult>,
    detection: Vec<DetectionResult>,
    /// Overall mean SIR improvement (dB) across all conditions and seeds.
    overall_mean_sir_improvement_db: f64,
    /// Overall drone recovery rate across all conditions and seeds.
    overall_drone_recovery_rate: f64,
}

fn conditions() -> Vec<Condition> {
    vec![
        // Two equal-loudness drones: a multi-UAV scene. No level offset; the
        // challenge is two overlapping harmonic combs, not masking.
        Condition {
            name: "two_drones",
            drone_f0s: &[110.0, 190.0],
            extra: vec![],
            drone_gain_db: 0.0,
        },
        // A quiet drone buried ~18 dB under broadband wind/rotor noise.
        Condition {
            name: "drone_plus_noise",
            drone_f0s: &[120.0],
            extra: vec![ExtraSource::Noise],
            drone_gain_db: -18.0,
        },
        // A quiet drone buried ~18 dB under a loud tonal interferer.
        Condition {
            name: "drone_plus_tone",
            drone_f0s: &[120.0],
            extra: vec![ExtraSource::Tone(1850.0)],
            drone_gain_db: -18.0,
        },
    ]
}

/// Build a mixture for a condition at a given seed.
fn build_mixture(cond: &Condition, seed: u64) -> Mixture {
    let cfg = MixConfig {
        n: 16_000,
        sample_rate: 16_000,
        seed,
        drone_gain_db: cond.drone_gain_db,
    };
    scene(cond.drone_f0s, &cond.extra, &cfg)
}

/// Scale a signal to roughly match the playback level the synthetic detectors
/// expect (peak ~0.8), so absolute-level features behave. Separation returns
/// unit-variance components, raw channels are sums of unit-RMS sources, so we
/// normalize both the same way for an apples-to-apples detector comparison.
fn level_match(x: &[f64]) -> Vec<f32> {
    let peak = x.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
    let g = if peak > 1e-9 { 0.8 / peak } else { 0.0 };
    x.iter().map(|&v| (v * g) as f32).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("benchmarks/results"));

    let sample_rate = 16_000u32;
    let conds = conditions();
    let ica_cfg = FastIcaConfig::default();

    // ---- 1. Separation quality ----
    println!("=== FastICA separation quality (instantaneous M x M mixing) ===");
    println!(
        "{:<18} {:>4} {:>5} {:>10} {:>10} {:>10} {:>9} {:>8} {:>8}",
        "condition",
        "mics",
        "seeds",
        "SIRimp_dB",
        "SIRsep_dB",
        "SIRmix_dB",
        "corr",
        "recov%",
        "conv%"
    );

    let mut separation = Vec::new();
    let mut all_sir_imps: Vec<f64> = Vec::new();
    let mut all_recovered: Vec<bool> = Vec::new();

    for cond in &conds {
        let mut sir_imp = Vec::new();
        let mut sir_sep = Vec::new();
        let mut sir_mix = Vec::new();
        let mut corrs = Vec::new();
        let mut recovered = 0usize;
        let mut converged = 0usize;
        let mut mics = 0usize;

        for &seed in SEEDS {
            let mix = build_mixture(cond, seed);
            mics = mix.channels.len();
            let res = fastica(&mix.channels, &ica_cfg);
            if res.converged {
                converged += 1;
            }
            let truth: Vec<Vec<f64>> = mix.sources.iter().map(|s| s.signal.clone()).collect();
            let q = drone_separation_quality(
                &truth,
                mix.drone_index().unwrap(),
                &res.sources,
                &mix.channels,
                RECOVERY_THRESHOLD,
            );
            sir_imp.push(q.drone_sir_improvement_db);
            sir_sep.push(q.drone_sir_separated_db);
            sir_mix.push(q.drone_sir_mixture_db);
            corrs.push(q.drone_correlation);
            if q.drone_recovered {
                recovered += 1;
            }
            all_sir_imps.push(q.drone_sir_improvement_db);
            all_recovered.push(q.drone_recovered);
        }

        let n = SEEDS.len() as f64;
        let cr = ConditionResult {
            condition: cond.name.to_string(),
            mics,
            sources: mics,
            seeds: SEEDS.len(),
            mean_sir_improvement_db: mean(&sir_imp),
            mean_sir_separated_db: mean(&sir_sep),
            mean_sir_mixture_db: mean(&sir_mix),
            mean_drone_correlation: mean(&corrs),
            drone_recovery_rate: recovered as f64 / n,
            convergence_rate: converged as f64 / n,
        };
        println!(
            "{:<18} {:>4} {:>5} {:>10.2} {:>10.2} {:>10.2} {:>9.3} {:>7.0}% {:>7.0}%",
            cr.condition,
            cr.mics,
            cr.seeds,
            cr.mean_sir_improvement_db,
            cr.mean_sir_separated_db,
            cr.mean_sir_mixture_db,
            cr.mean_drone_correlation,
            cr.drone_recovery_rate * 100.0,
            cr.convergence_rate * 100.0,
        );
        separation.push(cr);
    }

    // ---- 2. Detection improvement ----
    // Calibrate the detectors once on the shared synthetic dataset so their
    // logistic mapping is sensible, then reuse the fitted detector across scenes.
    let train = Dataset::synth(64, sample_rate, 20240601);

    println!("\n=== Detection improvement: confidence on drone, raw mix vs FastICA-separated ===");
    println!(
        "{:<14} {:<18} {:>5} {:>10} {:>10} {:>8} {:>9} {:>9}",
        "detector", "condition", "seeds", "conf_mix", "conf_sep", "lift", "rec_mix", "rec_sep"
    );

    let mut detection = Vec::new();
    for det_name in ["hps", "spectral_gate"] {
        let mut detector = instantiate(det_name);
        detector.fit(&train.samples);

        for cond in &conds {
            let mut conf_mix = Vec::new();
            let mut conf_sep = Vec::new();
            let mut hit_mix = 0usize;
            let mut hit_sep = 0usize;

            for &seed in SEEDS {
                let mix = build_mixture(cond, seed);
                let res = fastica(&mix.channels, &ica_cfg);
                let drone_idx = mix.drone_index().unwrap();
                let truth: Vec<Vec<f64>> = mix.sources.iter().map(|s| s.signal.clone()).collect();
                let q = drone_separation_quality(
                    &truth,
                    drone_idx,
                    &res.sources,
                    &mix.channels,
                    RECOVERY_THRESHOLD,
                );

                // (a) Raw mixed channel: pick the channel the drone is loudest in
                // (the most favorable single microphone) for a conservative
                // baseline.
                let best_ch = (0..mix.channels.len())
                    .max_by(|&a, &b| {
                        mix.mixing[a][drone_idx]
                            .abs()
                            .partial_cmp(&mix.mixing[b][drone_idx].abs())
                            .unwrap()
                    })
                    .unwrap();
                let mix_sig = level_match(&mix.channels[best_ch]);
                // (b) FastICA-separated drone component.
                let sep_sig = level_match(&res.sources[q.drone_component]);

                let cm = detector.score(&mix_sig, sample_rate) as f64;
                let cs = detector.score(&sep_sig, sample_rate) as f64;
                conf_mix.push(cm);
                conf_sep.push(cs);
                if cm >= 0.5 {
                    hit_mix += 1;
                }
                if cs >= 0.5 {
                    hit_sep += 1;
                }
            }

            let n = SEEDS.len() as f64;
            let cmix = mean(&conf_mix);
            let csep = mean(&conf_sep);
            let dr = DetectionResult {
                detector: det_name.to_string(),
                condition: cond.name.to_string(),
                seeds: SEEDS.len(),
                mean_confidence_mixture: cmix,
                mean_confidence_separated: csep,
                mean_confidence_lift: csep - cmix,
                recall_mixture: hit_mix as f64 / n,
                recall_separated: hit_sep as f64 / n,
            };
            println!(
                "{:<14} {:<18} {:>5} {:>10.3} {:>10.3} {:>+8.3} {:>8.0}% {:>8.0}%",
                dr.detector,
                dr.condition,
                dr.seeds,
                dr.mean_confidence_mixture,
                dr.mean_confidence_separated,
                dr.mean_confidence_lift,
                dr.recall_mixture * 100.0,
                dr.recall_separated * 100.0,
            );
            detection.push(dr);
        }
    }

    // ---- Report ----
    let overall_sir = mean(&all_sir_imps);
    let overall_recov =
        all_recovered.iter().filter(|&&b| b).count() as f64 / all_recovered.len().max(1) as f64;

    let report = BssReport {
        description: "FastICA blind source separation as a detection front-end for multi-UAV / noisy acoustic scenes. Separation quality (SIR improvement dB, drone recovery rate) and detection-confidence lift (raw mixed channel vs FastICA-separated drone component)."
            .to_string(),
        caveats: vec![
            "Instantaneous mixing (x = A s): real acoustic mixing is convolutive (per-path impulse responses); frequency-domain ICA / IVA is the documented next step.".to_string(),
            "Requires at least as many microphones as sources (M >= K); the determined M = K case is benchmarked here.".to_string(),
            "ICA recovers sources only up to permutation and sign/scale; the benchmark resolves these by best-match absolute correlation before scoring.".to_string(),
            "Synthetic sources and a conditioned random mixing matrix: this validates the method and shows the detection lift, it is not a field measurement.".to_string(),
        ],
        recovery_threshold: RECOVERY_THRESHOLD,
        seeds: SEEDS.to_vec(),
        separation,
        detection,
        overall_mean_sir_improvement_db: overall_sir,
        overall_drone_recovery_rate: overall_recov,
    };

    std::fs::create_dir_all(&out_dir)?;
    let path = out_dir.join("bss.json");
    std::fs::write(&path, serde_json::to_string_pretty(&report)?)?;

    println!(
        "\noverall mean SIR improvement: {:.2} dB   overall drone recovery: {:.0}%",
        overall_sir,
        overall_recov * 100.0
    );
    println!("results written to {}", path.display());
    Ok(())
}

/// Build a detector from the `drone-bench` registry by name.
fn instantiate(name: &str) -> Box<dyn Approach> {
    approaches::all()
        .into_iter()
        .find(|a| a.name() == name)
        .unwrap_or_else(|| panic!("unknown approach {name}"))
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}
