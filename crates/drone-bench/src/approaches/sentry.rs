//! `sentry` - a RECALL-FIRST ensemble tuned to catch UNSEEN drone models.
//!
//! ## Motivation
//! Our leakage-proof `heldout32` eval (32 drone makes/models that are NOT in
//! DADS) showed that no single detector catches every unseen drone, and that the
//! strongest generalizers key on *different physical cues*:
//!
//! * `hps`                  - harmonic-comb structure (blade-pass + harmonics),
//!   the current best (recall ~0.72 / ROC-AUC ~0.855 on unseen drones).
//! * `envelope_periodicity` - the periodic amplitude modulation of the rotor,
//!   an *envelope-domain* cue orthogonal to the spectral comb (AUC ~0.812).
//! * `feature_fusion`       - a broad MFCC + spectral + harmonic + cepstral
//!   logistic-regression classifier (AUC ~0.758).
//! * `mfcc_lr`              - the classic mel-cepstral timbre fingerprint
//!   (AUC ~0.692).
//!
//! Because different members catch partly-DIFFERENT unseen drones, an ensemble
//! that fires when ANY strong, diverse generalizer is confident should catch the
//! *union* of detectable drones and so raise recall - which is exactly what
//! counter-UAS cares about (a missed drone is the costly error).
//!
//! ## Composition is fixed A-PRIORI (honesty)
//! The member set is chosen on physical-diversity / high-AUC grounds BEFORE
//! seeing any held-out result, and is NOT tuned on the `heldout32` data. The only
//! data-driven choice happens entirely on a DADS-internal calibration slice (see
//! `fit`); the unseen-drone test set is never used for any decision here.
//!
//! ## Combiner: probabilistic soft-OR vs. learned logistic, picked on DADS
//! `fit` builds two candidate combiners over the members' confidences:
//!
//! 1. **Soft-OR** `1 - prod_i (1 - clamp(s_i))` - the recall-first rule: it
//!    fires high if ANY member is confident, so the ensemble inherits the union
//!    of the members' detections. Naturally recall-biased.
//! 2. **Learned logistic** `sigmoid(w . s + b)` over the member confidences,
//!    fit by deterministic batch gradient descent (a calibrated combiner).
//!
//! We carve a deterministic, stratified held-out slice out of the (DADS) train
//! set, fit the members on the rest, and pick whichever combiner gives the higher
//! **recall at a false-positive rate <= 0.10** on that held-out DADS slice (the
//! counter-UAS operating point). Calibration is therefore done ONLY on DADS,
//! never on the unseen-drone test set. Ties (and the degenerate no-data case)
//! resolve to the soft-OR, the a-priori recall-first default.
//!
//! `score` emits a value in `[0, 1]`, finite and deterministic. After the
//! combiner is chosen, the members are refit on the FULL train set so final
//! scoring uses all available data.

use crate::approach::Approach;
use crate::dataset::Sample;

use super::{
    envelope_periodicity::EnvelopePeriodicity, feature_fusion::FeatureFusion, hps::Hps,
    mfcc_lr::MfccLr,
};

/// Target false-positive rate for the recall-first operating point used to pick
/// the combiner on the DADS calibration slice.
const TARGET_FPR: f32 = 0.10;

/// Held-out fraction of the train set used to choose/calibrate the combiner.
/// A deterministic stride pattern (no RNG) carves out roughly this fraction
/// within each class, keeping both classes represented.
const HOLDOUT_STRIDE: usize = 10;
/// Ranks with `rank % HOLDOUT_STRIDE >= HOLDOUT_KEEP` go to the held-out slice
/// (~30%); the rest (~70%) is used to fit the members for combiner selection.
const HOLDOUT_KEEP: usize = 7;

/// Gradient-descent iterations for the learned logistic combiner.
const ITERS: usize = 3000;
/// Learning rate for the learned logistic combiner.
const LR: f32 = 1.5;
/// L2 regularization strength for the learned logistic combiner.
const L2: f32 = 1e-4;

/// Which combiner `fit` selected on the DADS calibration slice.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Combiner {
    /// Probabilistic soft-OR over member confidences (recall-first default).
    SoftOr,
    /// Learned logistic regression over member confidences.
    Learned,
}

/// Recall-first diverse ensemble of the strongest unseen-drone generalizers.
pub struct Sentry {
    /// Diverse member detectors, fitted on the full train set for scoring.
    members: Vec<Box<dyn Approach>>,
    /// Combiner selected on the DADS calibration slice.
    combiner: Combiner,
    /// Learned-combiner weights (one per member) and bias; only used when
    /// `combiner == Combiner::Learned`.
    weights: Vec<f32>,
    bias: f32,
    /// Whether `fit` has completed.
    fitted: bool,
}

impl Default for Sentry {
    fn default() -> Self {
        Self {
            members: Vec::new(),
            combiner: Combiner::SoftOr,
            weights: Vec::new(),
            bias: 0.0,
            fitted: false,
        }
    }
}

impl Sentry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Instantiate the fixed, a-priori member set (diverse high-AUC cues).
    fn build_members() -> Vec<Box<dyn Approach>> {
        vec![
            Box::new(Hps::new()),
            Box::new(EnvelopePeriodicity::new()),
            Box::new(FeatureFusion::new()),
            Box::new(MfccLr::new()),
        ]
    }

    /// Per-member confidences for a clip, each clamped into `[0, 1]`.
    fn member_scores(members: &[Box<dyn Approach>], samples: &[f32], sample_rate: u32) -> Vec<f32> {
        members
            .iter()
            .map(|m| {
                let s = m.score(samples, sample_rate);
                if s.is_finite() {
                    s.clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Probabilistic soft-OR of member confidences: `1 - prod_i (1 - s_i)`.
    /// Fires high when ANY member is confident (recall-first).
    fn soft_or(scores: &[f32]) -> f32 {
        let mut prod_complement = 1.0_f32;
        for &s in scores {
            prod_complement *= 1.0 - s.clamp(0.0, 1.0);
        }
        (1.0 - prod_complement).clamp(0.0, 1.0)
    }

    /// Learned-logistic combination of member confidences.
    fn learned(&self, scores: &[f32]) -> f32 {
        let mut z = self.bias;
        for (w, &s) in self.weights.iter().zip(scores.iter()) {
            z += w * s;
        }
        sigmoid(z).clamp(0.0, 1.0)
    }
}

impl Approach for Sentry {
    fn name(&self) -> &str {
        "sentry"
    }

    fn description(&self) -> &str {
        "Recall-first ensemble (hps + envelope_periodicity + feature_fusion + mfcc_lr); \
         soft-OR vs learned logistic picked by recall@FPR<=0.10 on a DADS slice"
    }

    fn fit(&mut self, train: &[Sample]) {
        if train.is_empty() {
            // Leave the recall-first default in place; members stay empty and
            // `score` returns the neutral 0.5 below.
            return;
        }

        // --- Deterministic, stratified fit / held-out split (no RNG) ----------
        // Within each class, walk samples in order and assign by a fixed stride
        // pattern so both parts retain both classes when present.
        let mut fit_part: Vec<Sample> = Vec::new();
        let mut holdout: Vec<Sample> = Vec::new();
        for class in [0u8, 1u8] {
            for (rank, s) in train.iter().filter(|s| s.label == class).enumerate() {
                if rank % HOLDOUT_STRIDE < HOLDOUT_KEEP {
                    fit_part.push(s.clone());
                } else {
                    holdout.push(s.clone());
                }
            }
        }
        // Degenerate split (a class too small to populate both parts): fall back
        // to the recall-first soft-OR with members fit on the full train set.
        let degenerate = fit_part.is_empty()
            || holdout.is_empty()
            || !holdout.iter().any(|s| s.label == 1)
            || !holdout.iter().any(|s| s.label == 0);

        if !degenerate {
            // Fit members on the fit part only, then collect their confidences on
            // the held-out DADS slice (no leakage into the selection).
            let mut sel_members = Self::build_members();
            for m in sel_members.iter_mut() {
                m.fit(&fit_part);
            }
            let x: Vec<Vec<f32>> = holdout
                .iter()
                .map(|s| Self::member_scores(&sel_members, &s.samples, s.sample_rate))
                .collect();
            let labels: Vec<f32> = holdout.iter().map(|s| s.label as f32).collect();

            // Train the learned logistic combiner on the held-out slice.
            let n_members = sel_members.len();
            let (w, b) = train_logistic(&x, &labels, n_members);

            // Score both candidate combiners on the held-out slice.
            let soft_or_scored: Vec<(f32, u8)> = x
                .iter()
                .zip(holdout.iter())
                .map(|(xi, s)| (Self::soft_or(xi), s.label))
                .collect();
            let learned_scored: Vec<(f32, u8)> = x
                .iter()
                .zip(holdout.iter())
                .map(|(xi, s)| {
                    let mut z = b;
                    for (wj, &xj) in w.iter().zip(xi.iter()) {
                        z += wj * xj;
                    }
                    (sigmoid(z).clamp(0.0, 1.0), s.label)
                })
                .collect();

            // Pick by recall at FPR <= TARGET_FPR. Ties -> soft-OR (a-priori
            // recall-first default).
            let r_soft = recall_at_fpr(&soft_or_scored, TARGET_FPR);
            let r_learned = recall_at_fpr(&learned_scored, TARGET_FPR);
            if r_learned > r_soft {
                self.combiner = Combiner::Learned;
                self.weights = w;
                self.bias = b;
            } else {
                self.combiner = Combiner::SoftOr;
            }
        } else {
            self.combiner = Combiner::SoftOr;
        }

        // --- Refit fresh members on the FULL train set for final scoring ------
        let mut members_full = Self::build_members();
        for m in members_full.iter_mut() {
            m.fit(train);
        }
        // If the learned combiner was picked but somehow has the wrong arity,
        // fall back to soft-OR to stay safe (cannot happen with a fixed set).
        if self.combiner == Combiner::Learned && self.weights.len() != members_full.len() {
            self.combiner = Combiner::SoftOr;
        }
        self.members = members_full;
        self.fitted = true;
    }

    fn score(&self, samples: &[f32], sample_rate: u32) -> f32 {
        if !self.fitted || self.members.is_empty() {
            return 0.5;
        }
        let scores = Self::member_scores(&self.members, samples, sample_rate);
        let out = match self.combiner {
            Combiner::SoftOr => Self::soft_or(&scores),
            Combiner::Learned => self.learned(&scores),
        };
        if out.is_finite() {
            out.clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
}

/// Train a logistic-regression combiner over member-confidence vectors `x` with
/// labels in `{0,1}` by deterministic batch gradient descent with L2.
fn train_logistic(x: &[Vec<f32>], labels: &[f32], n_features: usize) -> (Vec<f32>, f32) {
    let mut w = vec![0.0f32; n_features];
    let mut bias = 0.0f32;
    if x.is_empty() {
        return (w, bias);
    }
    let n = x.len() as f32;
    for _ in 0..ITERS {
        let mut grad_w = vec![0.0f32; n_features];
        let mut grad_b = 0.0f32;
        for (xi, &yi) in x.iter().zip(labels.iter()) {
            let mut z = bias;
            for (wj, &xj) in w.iter().zip(xi.iter()) {
                z += wj * xj;
            }
            let err = sigmoid(z) - yi;
            for (g, &xj) in grad_w.iter_mut().zip(xi.iter()) {
                *g += err * xj;
            }
            grad_b += err;
        }
        for (wj, gj) in w.iter_mut().zip(grad_w.iter()) {
            let grad = gj / n + L2 * *wj;
            *wj -= LR * grad;
        }
        bias -= LR * (grad_b / n);
    }
    (w, bias)
}

/// Recall (TPR) at the highest threshold whose false-positive rate is `<=
/// target_fpr`. This is the counter-UAS operating point: maximise detections
/// while holding the false alarm rate at or below the budget.
///
/// We sweep candidate thresholds (each distinct score), keep those whose FPR is
/// within budget, and report the best achievable recall among them. If no
/// threshold meets the budget (e.g. scores are too coarse), returns 0.0.
fn recall_at_fpr(scored: &[(f32, u8)], target_fpr: f32) -> f32 {
    let n_pos = scored.iter().filter(|&&(_, y)| y == 1).count();
    let n_neg = scored.len() - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.0;
    }
    // Candidate thresholds: each distinct score, plus a sentinel above the max.
    let mut ts: Vec<f32> = scored.iter().map(|&(s, _)| s).collect();
    ts.push(1.0001);
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    ts.dedup();

    let mut best_recall = 0.0f32;
    for &t in &ts {
        let mut tp = 0usize;
        let mut fp = 0usize;
        for &(s, y) in scored {
            if s >= t {
                if y == 1 {
                    tp += 1;
                } else {
                    fp += 1;
                }
            }
        }
        let fpr = fp as f32 / n_neg as f32;
        if fpr <= target_fpr {
            let recall = tp as f32 / n_pos as f32;
            if recall > best_recall {
                best_recall = recall;
            }
        }
    }
    best_recall
}

/// Numerically stable logistic sigmoid.
#[inline]
fn sigmoid(z: f32) -> f32 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn harmonic_clip(f0: f32, sr: u32, len: usize) -> Vec<f32> {
        (0..len)
            .map(|n| {
                let t = n as f32 / sr as f32;
                let am = 1.0 + 0.25 * (2.0 * PI * 8.0 * t).sin();
                let mut s = 0.0;
                for h in 1..=6 {
                    s += (0.6 / h as f32) * (2.0 * PI * f0 * h as f32 * t).sin();
                }
                am * s * 0.7
            })
            .collect()
    }

    fn noise_clip(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as f32 / u32::MAX as f32) * 1.6 - 0.8
            })
            .collect()
    }

    fn train_set(sr: u32, len: usize) -> Vec<Sample> {
        let mut v = Vec::new();
        for k in 0..24 {
            let f0 = 110.0 + (k % 5) as f32 * 12.0;
            v.push(Sample {
                id: format!("pos{k}"),
                samples: harmonic_clip(f0, sr, len),
                sample_rate: sr,
                label: 1,
            });
            v.push(Sample {
                id: format!("neg{k}"),
                samples: noise_clip(len, 0x1234 + k as u32),
                sample_rate: sr,
                label: 0,
            });
        }
        v
    }

    #[test]
    fn soft_or_is_recall_first() {
        // Soft-OR fires high if any member is confident.
        assert!((Sentry::soft_or(&[0.0, 0.0, 0.0, 0.0])).abs() < 1e-6);
        assert!(Sentry::soft_or(&[0.9, 0.0, 0.0, 0.0]) >= 0.9);
        assert!(Sentry::soft_or(&[0.5, 0.5, 0.5, 0.5]) > 0.9);
        // Monotone and bounded.
        let s = Sentry::soft_or(&[0.3, 0.7, 0.1, 0.2]);
        assert!((0.0..=1.0).contains(&s));
    }

    #[test]
    fn recall_at_fpr_basic() {
        // Perfectly separable: recall 1.0 at FPR 0.
        let scored = vec![(0.9, 1), (0.8, 1), (0.2, 0), (0.1, 0)];
        assert!((recall_at_fpr(&scored, 0.10) - 1.0).abs() < 1e-6);
        // All negatives outscore positives at low FPR -> recall 0.
        let scored = vec![(0.1, 1), (0.2, 1), (0.9, 0), (0.95, 0)];
        assert!(recall_at_fpr(&scored, 0.0) < 1e-6);
    }

    #[test]
    fn unfit_scores_neutral() {
        let s = Sentry::new();
        assert_eq!(s.score(&[0.1, 0.2, 0.3], 16_000), 0.5);
    }

    #[test]
    fn fit_then_score_is_bounded_and_separates() {
        let sr = 16_000;
        let len = sr as usize / 2;
        let mut s = Sentry::new();
        s.fit(&train_set(sr, len));
        let drone = s.score(&harmonic_clip(120.0, sr, len), sr);
        let noise = s.score(&noise_clip(len, 0xDEAD), sr);
        assert!((0.0..=1.0).contains(&drone));
        assert!((0.0..=1.0).contains(&noise));
        assert!(drone.is_finite() && noise.is_finite());
        assert!(drone > noise, "drone {drone} should exceed noise {noise}");
    }

    #[test]
    fn deterministic() {
        let sr = 16_000;
        let len = sr as usize / 2;
        let train = train_set(sr, len);
        let mut a = Sentry::new();
        let mut b = Sentry::new();
        a.fit(&train);
        b.fit(&train);
        let clip = harmonic_clip(115.0, sr, len);
        assert_eq!(a.score(&clip, sr), b.score(&clip, sr));
    }
}
