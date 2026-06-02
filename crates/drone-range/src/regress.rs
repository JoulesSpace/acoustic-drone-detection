//! Ridge (L2-regularized linear) regression for distance in metres (mode (a)).
//!
//! Features are standardized with the train-set mean/std, then the ridge
//! solution `w = (XᵀX + λI)⁻¹ Xᵀy` is computed in closed form via Gaussian
//! elimination with partial pivoting. There is no RNG and no iteration order
//! dependence, so the fit is fully deterministic and needs no ML crate.
//!
//! Predicting range *directly in metres* is the most honest target; we do not
//! log-transform so the reported MAE is in real metres.

use alloc::vec;
use alloc::vec::Vec;

use libm::sqrtf;

/// A fitted ridge-regression distance model.
#[derive(Debug, Clone)]
pub struct RidgeModel {
    /// Weights, one per (standardized) feature.
    weights: Vec<f32>,
    /// Bias term.
    bias: f32,
    /// Per-feature train mean (for standardization).
    feat_mean: Vec<f32>,
    /// Per-feature train std (for standardization).
    feat_std: Vec<f32>,
    /// Whether `fit` has run.
    fitted: bool,
}

impl RidgeModel {
    /// Fit on standardized features `x` (each row a sample) against targets `y`
    /// (range in metres) with ridge strength `lambda`.
    ///
    /// `x` rows must all share the same length; an empty train set yields an
    /// unfitted model that predicts the target mean (0 if none).
    pub fn fit(x: &[Vec<f32>], y: &[f32], lambda: f32) -> Self {
        if x.is_empty() || x[0].is_empty() || x.len() != y.len() {
            return Self {
                weights: Vec::new(),
                bias: mean(y),
                feat_mean: Vec::new(),
                feat_std: Vec::new(),
                fitted: false,
            };
        }
        let d = x[0].len();
        let n = x.len();

        // Standardization stats.
        let mut feat_mean = vec![0.0_f32; d];
        for row in x {
            for (m, &v) in feat_mean.iter_mut().zip(row.iter()) {
                *m += v;
            }
        }
        for m in feat_mean.iter_mut() {
            *m /= n as f32;
        }
        let mut feat_std = vec![0.0_f32; d];
        for row in x {
            for (s, (&v, &m)) in feat_std.iter_mut().zip(row.iter().zip(feat_mean.iter())) {
                let dv = v - m;
                *s += dv * dv;
            }
        }
        for s in feat_std.iter_mut() {
            let val = sqrtf(*s / n as f32);
            *s = if val > 1e-6 { val } else { 1.0 };
        }

        // Standardized design matrix.
        let xs: Vec<Vec<f32>> = x
            .iter()
            .map(|row| {
                row.iter()
                    .zip(feat_mean.iter().zip(feat_std.iter()))
                    .map(|(&v, (&m, &s))| (v - m) / s)
                    .collect::<Vec<f32>>()
            })
            .collect();

        // Centre the targets so the bias is just the target mean (the ridge
        // penalty then never shrinks the intercept).
        let y_mean = mean(y);

        // Normal equations: (XᵀX + λI) w = Xᵀ(y - y_mean).
        let mut ata = vec![vec![0.0_f32; d]; d];
        let mut atb = vec![0.0_f32; d];
        for (row, &yi) in xs.iter().zip(y.iter()) {
            let yc = yi - y_mean;
            for i in 0..d {
                atb[i] += row[i] * yc;
                for j in 0..d {
                    ata[i][j] += row[i] * row[j];
                }
            }
        }
        for (i, atai) in ata.iter_mut().enumerate() {
            atai[i] += lambda;
        }

        let weights = solve(ata, atb).unwrap_or_else(|| vec![0.0_f32; d]);

        Self {
            weights,
            bias: y_mean,
            feat_mean,
            feat_std,
            fitted: true,
        }
    }

    /// Predict the range in metres for a raw (un-standardized) feature vector.
    ///
    /// The prediction is clamped to be non-negative (a distance cannot be < 0).
    pub fn predict(&self, raw: &[f32]) -> f32 {
        if !self.fitted {
            return self.bias.max(0.0);
        }
        let mut z = self.bias;
        for (i, &w) in self.weights.iter().enumerate() {
            let xs = (raw[i] - self.feat_mean[i]) / self.feat_std[i];
            z += w * xs;
        }
        z.max(0.0)
    }
}

/// Mean of a slice (0 if empty).
fn mean(v: &[f32]) -> f32 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f32>() / v.len() as f32
    }
}

/// Solve `A x = b` for square `A` via Gaussian elimination with partial
/// pivoting. Returns `None` if the matrix is singular.
///
/// The elimination updates row `r` from pivot row `col` simultaneously, so the
/// index-based loops are clearer (and avoid disjoint-borrow gymnastics) than an
/// iterator rewrite would be.
#[allow(clippy::needless_range_loop)]
fn solve(mut a: Vec<Vec<f32>>, mut b: Vec<f32>) -> Option<Vec<f32>> {
    let n = b.len();
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            let v = a[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-9 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);

        // Eliminate below.
        let pivot = a[col][col];
        for r in (col + 1)..n {
            let factor = a[r][col] / pivot;
            if factor != 0.0 {
                for c in col..n {
                    a[r][c] -= factor * a[col][c];
                }
                b[r] -= factor * b[col];
            }
        }
    }
    // Back-substitution.
    let mut x = vec![0.0_f32; n];
    for row in (0..n).rev() {
        let mut acc = b[row];
        for c in (row + 1)..n {
            acc -= a[row][c] * x[c];
        }
        x[row] = acc / a[row][row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_exact_linear_relationship() {
        // y = 2*x0 - 3*x1 + 5, no noise. Ridge with tiny lambda should be close.
        let x = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![2.0, 1.0],
            vec![1.0, 2.0],
        ];
        let y: Vec<f32> = x.iter().map(|r| 2.0 * r[0] - 3.0 * r[1] + 5.0).collect();
        let model = RidgeModel::fit(&x, &y, 1e-6);
        for (r, &yi) in x.iter().zip(y.iter()) {
            assert!((model.predict(r) - yi).abs() < 0.05, "row {r:?}");
        }
    }

    #[test]
    fn prediction_is_non_negative() {
        let x = vec![vec![0.0], vec![1.0], vec![2.0]];
        let y = vec![10.0, -5.0, -20.0]; // forces a steep negative slope
        let model = RidgeModel::fit(&x, &y, 1e-6);
        assert!(model.predict(&[100.0]) >= 0.0);
    }

    #[test]
    fn unfitted_predicts_target_mean() {
        let model = RidgeModel::fit(&[], &[], 1.0);
        assert_eq!(model.predict(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn solve_identity() {
        let a = vec![vec![2.0, 0.0], vec![0.0, 4.0]];
        let b = vec![6.0, 8.0];
        let x = solve(a, b).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-6);
        assert!((x[1] - 2.0).abs() < 1e-6);
    }
}
