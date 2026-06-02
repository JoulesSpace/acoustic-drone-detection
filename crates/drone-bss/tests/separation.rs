//! Integration tests: FastICA actually separates known mixtures.
//!
//! These exercise the full public pipeline (`mix` -> `fastica` -> `metrics`) and
//! pin the central claim of the crate: that on a known instantaneous mixture the
//! sources come back out, up to permutation/sign.

use drone_bench::approaches;
use drone_bench::dataset::Dataset;

use drone_bss::fastica::{fastica, FastIcaConfig};
use drone_bss::metrics::{best_match, drone_separation_quality};
use drone_bss::mix::{scene, ExtraSource, MixConfig};

/// The headline test the task asks for: a known 2x2 mix is separated.
#[test]
fn fastica_separates_known_2x2_mix() {
    let cfg = MixConfig {
        n: 16_000,
        sample_rate: 16_000,
        seed: 1,
        ..Default::default()
    };
    // Two independent sources: a drone harmonic stack and a tonal interferer.
    let mix = scene(&[120.0], &[ExtraSource::Tone(1850.0)], &cfg);
    assert_eq!(mix.channels.len(), 2, "expected a 2x2 mixture");

    let res = fastica(&mix.channels, &FastIcaConfig::default());
    assert!(res.converged, "FastICA did not converge");

    // Each true source should match some recovered component near-perfectly.
    let truth: Vec<Vec<f64>> = mix.sources.iter().map(|s| s.signal.clone()).collect();
    let matches = best_match(&truth, &res.sources);
    for (i, (_, corr)) in matches.iter().enumerate() {
        assert!(
            *corr > 0.95,
            "source {i} ({}) recovered with corr {corr} <= 0.95",
            mix.sources[i].label
        );
    }
    // The two sources must map to distinct components (a real unmix, not a
    // collapse onto one channel).
    assert_ne!(matches[0].0, matches[1].0, "components collapsed");
}

/// Separation must measurably improve the drone SIR over the raw mixture.
#[test]
fn separation_improves_drone_sir() {
    let cfg = MixConfig {
        n: 16_000,
        sample_rate: 16_000,
        seed: 7,
        ..Default::default()
    };
    let mix = scene(&[110.0], &[ExtraSource::Noise], &cfg);
    let res = fastica(&mix.channels, &FastIcaConfig::default());

    let truth: Vec<Vec<f64>> = mix.sources.iter().map(|s| s.signal.clone()).collect();
    let q = drone_separation_quality(
        &truth,
        mix.drone_index().unwrap(),
        &res.sources,
        &mix.channels,
        0.9,
    );
    assert!(
        q.drone_recovered,
        "drone not recovered: corr {}",
        q.drone_correlation
    );
    assert!(
        q.drone_sir_improvement_db > 3.0,
        "SIR improvement {} dB too small",
        q.drone_sir_improvement_db
    );
}

/// A 3x3 mix (two drones + noise) should still separate the drones.
#[test]
fn fastica_separates_3x3_two_drones_plus_noise() {
    let cfg = MixConfig {
        n: 16_000,
        sample_rate: 16_000,
        seed: 3,
        ..Default::default()
    };
    let mix = scene(&[110.0, 190.0], &[ExtraSource::Noise], &cfg);
    assert_eq!(mix.channels.len(), 3);

    let res = fastica(&mix.channels, &FastIcaConfig::default());
    let truth: Vec<Vec<f64>> = mix.sources.iter().map(|s| s.signal.clone()).collect();
    let matches = best_match(&truth, &res.sources);
    // Both drone sources (indices 0 and 1) should be well recovered.
    assert!(matches[0].1 > 0.9, "drone0 corr {}", matches[0].1);
    assert!(matches[1].1 > 0.9, "drone1 corr {}", matches[1].1);
}

/// The payoff: a `drone-bench` detector that fails on the masked raw mixture
/// succeeds on the FastICA-separated drone component. Locks the "BSS rescues
/// detection in multi-source scenes" claim with a reused real detector.
#[test]
fn separation_rescues_detection() {
    // Quiet drone buried ~18 dB under broadband noise: single-mic detection
    // should miss it, separation should recover it.
    let cfg = MixConfig {
        n: 16_000,
        sample_rate: 16_000,
        seed: 4,
        drone_gain_db: -18.0,
    };
    let mix = scene(&[120.0], &[ExtraSource::Noise], &cfg);
    let res = fastica(&mix.channels, &FastIcaConfig::default());
    let drone_idx = mix.drone_index().unwrap();
    let truth: Vec<Vec<f64>> = mix.sources.iter().map(|s| s.signal.clone()).collect();
    let q = drone_separation_quality(&truth, drone_idx, &res.sources, &mix.channels, 0.9);
    assert!(q.drone_recovered);

    // Calibrate the harmonic-comb detector on the shared synthetic dataset.
    let train = Dataset::synth(64, 16_000, 20_240_601);
    let mut det = approaches::all()
        .into_iter()
        .find(|a| a.name() == "spectral_gate")
        .expect("spectral_gate detector present");
    det.fit(&train.samples);

    let level_match = |x: &[f64]| -> Vec<f32> {
        let peak = x.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let g = if peak > 1e-9 { 0.8 / peak } else { 0.0 };
        x.iter().map(|&v| (v * g) as f32).collect()
    };

    // Best raw channel for the drone vs the separated component.
    let best_ch = (0..mix.channels.len())
        .max_by(|&a, &b| {
            mix.mixing[a][drone_idx]
                .abs()
                .partial_cmp(&mix.mixing[b][drone_idx].abs())
                .unwrap()
        })
        .unwrap();
    let conf_mix = det.score(&level_match(&mix.channels[best_ch]), 16_000);
    let conf_sep = det.score(&level_match(&res.sources[q.drone_component]), 16_000);

    assert!(
        conf_sep > conf_mix + 0.3,
        "expected a detection lift: mixture {conf_mix:.3} -> separated {conf_sep:.3}"
    );
    assert!(
        conf_sep >= 0.5,
        "separated drone should be detected: {conf_sep:.3}"
    );
}
