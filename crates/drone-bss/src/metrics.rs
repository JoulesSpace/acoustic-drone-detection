//! Separation-quality metrics.
//!
//! ICA recovers sources only up to an arbitrary **permutation** (component order
//! is undefined) and **sign/scale** (each component can be flipped and rescaled).
//! Before we can score a recovered source against its ground truth we must
//! resolve those ambiguities. [`best_match`] does this by greedily assigning each
//! true source to the recovered component it correlates with most strongly (in
//! absolute value, so a sign flip does not hurt).
//!
//! The headline number is **SIR improvement (dB)** - how much cleaner the target
//! source is in the best-matched separated component than in a representative raw
//! mixture channel. Signal-to-Interference Ratio treats the projection of the
//! estimate onto the target source as "signal" and everything else as
//! "interference"; the improvement is `SIR_separated - SIR_mixture`.

/// Pearson correlation between two equal-length signals, in `[-1, 1]`.
pub fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 || b.len() != n {
        return 0.0;
    }
    let inv = 1.0 / n as f64;
    let ma = a.iter().sum::<f64>() * inv;
    let mb = b.iter().sum::<f64>() * inv;
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 1e-300 || vb <= 1e-300 {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

/// Assign each true source (row) to the recovered component (col) it best
/// matches by absolute correlation, greedily and without reuse.
///
/// Returns `assignment[i] = (component_index, abs_correlation)` for true source
/// `i`. This resolves ICA's permutation ambiguity; the absolute value resolves
/// the sign ambiguity.
pub fn best_match(truth: &[Vec<f64>], recovered: &[Vec<f64>]) -> Vec<(usize, f64)> {
    let k = truth.len();
    let m = recovered.len();
    // Correlation magnitude table.
    let mut corr = vec![vec![0.0f64; m]; k];
    for (i, t) in truth.iter().enumerate() {
        for (j, r) in recovered.iter().enumerate() {
            corr[i][j] = correlation(t, r).abs();
        }
    }
    let mut used = vec![false; m];
    let mut out = vec![(0usize, 0.0f64); k];
    // Greedy: repeatedly take the globally-best unused (i, j) pair.
    let mut assigned = vec![false; k];
    for _ in 0..k {
        let mut best = (-1.0f64, 0usize, 0usize);
        for i in 0..k {
            if assigned[i] {
                continue;
            }
            for j in 0..m {
                if used[j] {
                    continue;
                }
                if corr[i][j] > best.0 {
                    best = (corr[i][j], i, j);
                }
            }
        }
        if best.0 < 0.0 {
            break;
        }
        out[best.1] = (best.2, best.0);
        assigned[best.1] = true;
        used[best.2] = true;
    }
    out
}

/// Signal-to-Interference Ratio (dB) of an `estimate` with respect to a target
/// `source`, ignoring sign/scale.
///
/// We decompose the (zero-mean, unit-variance) estimate into the part explained
/// by the target source and a residual: `SIR = 10 log10( ||proj||^2 / ||resid||^2 )`.
/// Concretely, with `rho` the correlation between estimate and source,
/// `SIR = 10 log10( rho^2 / (1 - rho^2) )`. A perfect recovery (`|rho| -> 1`)
/// gives `+inf`; an unrelated estimate (`rho -> 0`) gives a large negative dB.
pub fn sir_db(estimate: &[f64], source: &[f64]) -> f64 {
    let rho = correlation(estimate, source);
    let r2 = (rho * rho).clamp(0.0, 1.0 - 1e-12);
    10.0 * (r2 / (1.0 - r2)).max(1e-12).log10()
}

/// SIR improvement (dB) for one target source: the SIR of the best-matched
/// separated component minus the SIR of the *best* raw mixture channel for that
/// same source.
///
/// Using the best raw channel as the baseline is the honest, conservative
/// choice: it asks "did separation beat the luckiest single microphone?" rather
/// than a cherry-picked bad one.
pub fn sir_improvement_db(
    source: &[f64],
    separated_component: &[f64],
    mixture_channels: &[Vec<f64>],
) -> f64 {
    let sep = sir_db(separated_component, source);
    let raw = mixture_channels
        .iter()
        .map(|ch| sir_db(ch, source))
        .fold(f64::NEG_INFINITY, f64::max);
    sep - raw
}

/// A bundle of separation-quality numbers for a single mixture, focused on the
/// drone source.
#[derive(Clone, Debug)]
pub struct SeparationQuality {
    /// Index of the recovered component best matching the drone source.
    pub drone_component: usize,
    /// Absolute correlation of that component to the clean drone source.
    pub drone_correlation: f64,
    /// SIR (dB) of the recovered drone component vs the clean drone source.
    pub drone_sir_separated_db: f64,
    /// Best SIR (dB) of any raw mixture channel vs the clean drone source.
    pub drone_sir_mixture_db: f64,
    /// SIR improvement (dB): separated minus mixture.
    pub drone_sir_improvement_db: f64,
    /// Whether the drone was "recovered": its best-match correlation exceeds the
    /// recovery threshold.
    pub drone_recovered: bool,
}

/// Score how well a separation recovered the drone source.
///
/// `truth` are the clean sources, `drone_idx` is which is the drone, `recovered`
/// are the FastICA components, and `mixture_channels` are the raw observations.
/// `recovery_threshold` is the absolute-correlation bar for counting the drone
/// as recovered (0.9 is a sensible default for instantaneous mixing).
pub fn drone_separation_quality(
    truth: &[Vec<f64>],
    drone_idx: usize,
    recovered: &[Vec<f64>],
    mixture_channels: &[Vec<f64>],
    recovery_threshold: f64,
) -> SeparationQuality {
    let matches = best_match(truth, recovered);
    let (component, corr) = matches[drone_idx];
    let drone_src = &truth[drone_idx];
    let sep_sir = sir_db(&recovered[component], drone_src);
    let mix_sir = mixture_channels
        .iter()
        .map(|ch| sir_db(ch, drone_src))
        .fold(f64::NEG_INFINITY, f64::max);
    SeparationQuality {
        drone_component: component,
        drone_correlation: corr,
        drone_sir_separated_db: sep_sir,
        drone_sir_mixture_db: mix_sir,
        drone_sir_improvement_db: sep_sir - mix_sir,
        drone_recovered: corr >= recovery_threshold,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlation_of_identical_is_one() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        assert!((correlation(&a, &a) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn correlation_of_negation_is_minus_one() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b: Vec<f64> = a.iter().map(|v| -v).collect();
        assert!((correlation(&a, &b) + 1.0).abs() < 1e-12);
    }

    #[test]
    fn best_match_recovers_permutation() {
        let s0 = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let s1 = vec![1.0, 1.0, -1.0, -1.0, 1.0];
        let truth = vec![s0.clone(), s1.clone()];
        // Recovered in swapped order, with s0 sign-flipped.
        let recovered = vec![s1.clone(), s0.iter().map(|v| -v).collect()];
        let m = best_match(&truth, &recovered);
        assert_eq!(m[0].0, 1); // true s0 -> recovered component 1
        assert_eq!(m[1].0, 0);
        assert!(m[0].1 > 0.99 && m[1].1 > 0.99);
    }

    #[test]
    fn sir_higher_for_better_estimate() {
        let src = vec![0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0];
        let good: Vec<f64> = src.iter().map(|v| v + 0.05).collect();
        let bad = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        assert!(sir_db(&good, &src) > sir_db(&bad, &src));
    }
}
