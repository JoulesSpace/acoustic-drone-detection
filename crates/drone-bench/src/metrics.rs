//! Classification metrics and curve computation for benchmark results.

use serde::Serialize;

/// Headline metrics at a fixed decision threshold, plus threshold-free AUCs.
#[derive(Debug, Clone, Serialize)]
pub struct Metrics {
    pub threshold: f32,
    pub accuracy: f32,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub tp: usize,
    pub fp: usize,
    #[serde(rename = "fn")]
    pub fn_: usize,
    pub tn: usize,
    pub roc_auc: f32,
    pub pr_auc: f32,
    /// Brier score (mean squared error of the confidence vs the label).
    pub brier: f32,
}

/// A point on a curve (ROC or PR). `x`/`y` meaning depends on the curve.
#[derive(Debug, Clone, Serialize)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

/// Confusion counts at a threshold (`score >= threshold` → predict drone).
pub fn confusion(scored: &[(f32, u8)], threshold: f32) -> (usize, usize, usize, usize) {
    let (mut tp, mut fp, mut fn_, mut tn) = (0, 0, 0, 0);
    for &(s, y) in scored {
        let pred = s >= threshold;
        match (pred, y == 1) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    (tp, fp, fn_, tn)
}

/// Full metric bundle for a set of `(score, label)` pairs.
pub fn evaluate(scored: &[(f32, u8)], threshold: f32) -> Metrics {
    let (tp, fp, fn_, tn) = confusion(scored, threshold);
    let total = scored.len().max(1) as f32;
    let accuracy = (tp + tn) as f32 / total;
    let precision = safe_div(tp as f32, (tp + fp) as f32);
    let recall = safe_div(tp as f32, (tp + fn_) as f32);
    let f1 = safe_div(2.0 * precision * recall, precision + recall);
    let brier = scored
        .iter()
        .map(|&(s, y)| {
            let d = s - y as f32;
            d * d
        })
        .sum::<f32>()
        / total;
    Metrics {
        threshold,
        accuracy,
        precision,
        recall,
        f1,
        tp,
        fp,
        fn_,
        tn,
        roc_auc: roc_auc(scored),
        pr_auc: pr_auc(scored),
        brier,
    }
}

/// ROC-AUC via the rank-sum (Mann–Whitney U) statistic, tie-aware.
pub fn roc_auc(scored: &[(f32, u8)]) -> f32 {
    let n_pos = scored.iter().filter(|&&(_, y)| y == 1).count();
    let n_neg = scored.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f32::NAN;
    }
    // Average ranks (1-based) over scores, ties shared.
    let mut idx: Vec<usize> = (0..scored.len()).collect();
    idx.sort_by(|&a, &b| scored[a].0.partial_cmp(&scored[b].0).unwrap());
    let mut ranks = vec![0.0_f32; scored.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i + 1;
        while j < idx.len() && scored[idx[j]].0 == scored[idx[i]].0 {
            j += 1;
        }
        // ranks i..j share the average rank (1-based)
        let avg = ((i + 1 + j) as f32) / 2.0; // mean of (i+1 .. j)
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    let sum_pos_ranks: f32 = scored
        .iter()
        .zip(ranks.iter())
        .filter(|((_, y), _)| *y == 1)
        .map(|(_, r)| *r)
        .sum();
    let u = sum_pos_ranks - (n_pos * (n_pos + 1)) as f32 / 2.0;
    u / (n_pos * n_neg) as f32
}

/// Sweep thresholds at distinct scores; return ROC points (x=FPR, y=TPR).
pub fn roc_curve(scored: &[(f32, u8)]) -> Vec<Point> {
    thresholds(scored)
        .into_iter()
        .map(|t| {
            let (tp, fp, fn_, tn) = confusion(scored, t);
            Point {
                x: safe_div(fp as f32, (fp + tn) as f32),  // FPR
                y: safe_div(tp as f32, (tp + fn_) as f32), // TPR
            }
        })
        .collect()
}

/// Precision-Recall points (x=recall, y=precision).
pub fn pr_curve(scored: &[(f32, u8)]) -> Vec<Point> {
    thresholds(scored)
        .into_iter()
        .map(|t| {
            let (tp, fp, fn_, _tn) = confusion(scored, t);
            Point {
                x: safe_div(tp as f32, (tp + fn_) as f32), // recall
                y: safe_div(tp as f32, (tp + fp) as f32),  // precision
            }
        })
        .collect()
}

/// PR-AUC via trapezoidal integration over recall (curve sorted by recall).
pub fn pr_auc(scored: &[(f32, u8)]) -> f32 {
    let mut pts = pr_curve(scored);
    if pts.len() < 2 {
        return f32::NAN;
    }
    pts.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
    let mut area = 0.0;
    for w in pts.windows(2) {
        let dx = w[1].x - w[0].x;
        area += dx * (w[0].y + w[1].y) / 2.0;
    }
    area
}

/// Candidate thresholds: just below each distinct score, plus 0 and 1 ends.
fn thresholds(scored: &[(f32, u8)]) -> Vec<f32> {
    let mut ts: Vec<f32> = scored.iter().map(|&(s, _)| s).collect();
    ts.push(0.0);
    ts.push(1.0001);
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ts.dedup();
    ts
}

#[inline]
fn safe_div(a: f32, b: f32) -> f32 {
    if b.abs() > f32::EPSILON {
        a / b
    } else {
        0.0
    }
}

/// The full JSON result for one approach, written to `benchmarks/results/`.
#[derive(Debug, Clone, Serialize)]
pub struct ApproachResult {
    pub approach: String,
    pub description: String,
    pub n_test: usize,
    pub n_pos: usize,
    pub n_neg: usize,
    pub mean_infer_ms: f64,
    #[serde(flatten)]
    pub metrics: Metrics,
    /// Per-clip `(score, label)` for downstream plotting.
    pub scores: Vec<ScoreLabel>,
    pub roc: Vec<Point>,
    pub pr: Vec<Point>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoreLabel {
    pub s: f32,
    pub y: u8,
}
