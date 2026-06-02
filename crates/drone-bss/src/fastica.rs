//! FastICA: fixed-point Independent Component Analysis.
//!
//! Given `M` observed mixture channels (each a length-`T` time series), FastICA
//! recovers `M` maximally-independent source estimates under the instantaneous
//! linear model `x = A s`. The pipeline is the textbook one (Hyvarinen &amp; Oja):
//!
//! 1. **Center** each channel to zero mean.
//! 2. **Whiten** via PCA: eigendecompose the covariance (using our own
//!    [`crate::eig`] Jacobi solver), then map `z = D^{-1/2} E^T x_centered` so
//!    `cov(z) = I`. Whitening removes second-order correlation, leaving only the
//!    rotation that ICA must find.
//! 3. **Fixed-point iteration** with the `g = tanh` nonlinearity, finding weight
//!    vectors `w` that maximize non-Gaussianity (negentropy). After each update
//!    the full weight matrix `W` is re-orthogonalized by **symmetric
//!    decorrelation** `W <- (W W^T)^{-1/2} W`, so all components are extracted in
//!    parallel without one dominating.
//!
//! The unmixing is then `S_hat = W Z`, and the unmixing matrix back in the
//! original observation space is `W_full = W D^{-1/2} E^T`. ICA's inherent
//! permutation and sign/scale ambiguities are left as-is here and resolved
//! downstream by [`crate::metrics::best_match`].
//!
//! The 2x2 case (`M = 2`) is just the general path with `M = 2`; nothing is
//! special-cased, which keeps the one code path exercised by every test.
//!
//! Determinism: weight vectors are initialized from a seeded [`crate::rng::Rng`],
//! so a given `(input, seed)` always yields the same separation.

// The fixed-point and decorrelation kernels index matrices/data in two
// dimensions in lockstep; the explicit index loops mirror the linear algebra and
// are clearer than iterator chains here.
#![allow(clippy::needless_range_loop)]

use crate::eig::jacobi_eigen;
use crate::rng::Rng;

/// Tuning for [`fastica`].
#[derive(Clone, Debug)]
pub struct FastIcaConfig {
    /// Maximum fixed-point iterations before giving up on convergence.
    pub max_iter: usize,
    /// Convergence tolerance on `1 - |<w_new, w_old>|` (closeness of the new
    /// weight matrix to the old, up to sign).
    pub tol: f64,
    /// Seed for the (deterministic) weight initialization.
    pub seed: u64,
}

impl Default for FastIcaConfig {
    fn default() -> Self {
        Self {
            max_iter: 400,
            tol: 1e-8,
            seed: 0x0B55_1CA0,
        }
    }
}

/// Result of running [`fastica`].
#[derive(Clone, Debug)]
pub struct FastIcaResult {
    /// Separated source estimates: `sources[i]` is the i-th recovered component
    /// (length `T`), unit variance, arbitrary sign and order.
    pub sources: Vec<Vec<f64>>,
    /// Unmixing matrix mapping centered observations to sources:
    /// `S_hat = unmixing * (X - mean)`. Shape `M x M`, row-major.
    pub unmixing: Vec<Vec<f64>>,
    /// Whether the fixed-point iteration converged within `max_iter`.
    pub converged: bool,
    /// Number of iterations actually run.
    pub iterations: usize,
}

/// Run FastICA on `observations`, where `observations[c]` is mixture channel `c`
/// (all the same length `T`). Returns the separated sources and unmixing matrix.
///
/// # Panics
///
/// Panics if `observations` is empty, channels differ in length, or `T == 0`.
pub fn fastica(observations: &[Vec<f64>], cfg: &FastIcaConfig) -> FastIcaResult {
    let m = observations.len();
    assert!(m > 0, "fastica: no channels");
    let t = observations[0].len();
    assert!(t > 0, "fastica: empty channels");
    assert!(
        observations.iter().all(|c| c.len() == t),
        "fastica: ragged channels"
    );

    // --- 1. Center each channel. ---
    let mut x = observations.to_vec();
    let mut means = vec![0.0; m];
    for (row, mean) in x.iter_mut().zip(means.iter_mut()) {
        let mu = row.iter().sum::<f64>() / t as f64;
        *mean = mu;
        for v in row.iter_mut() {
            *v -= mu;
        }
    }

    // --- 2. Whiten. cov = (1/T) X X^T (X already centered). ---
    let cov = covariance(&x, t);
    let eig = jacobi_eigen(&cov);

    // Whitening matrix K = D^{-1/2} E^T. Floor tiny/negative eigenvalues (a
    // near-rank-deficient mixture) so D^{-1/2} stays finite.
    let mut k = vec![vec![0.0; m]; m];
    for i in 0..m {
        let lam = eig.values[i].max(1e-12);
        let inv_sqrt = 1.0 / lam.sqrt();
        for j in 0..m {
            // E^T row i is column i of E -> eig.vectors[j][i].
            k[i][j] = inv_sqrt * eig.vectors[j][i];
        }
    }
    // Whitened data Z = K X, shape M x T.
    let z = matmul_mt(&k, &x, t);

    // --- 3. Symmetric FastICA on the whitened data. ---
    let mut rng = Rng::new(cfg.seed);
    let mut w = random_orthonormal(m, &mut rng);

    let mut converged = false;
    let mut iterations = 0;
    for it in 0..cfg.max_iter {
        iterations = it + 1;
        let w_old = w.clone();

        // For each component, the fixed-point update with g = tanh:
        //   w+ = E{ z g(w^T z) } - E{ g'(w^T z) } w
        let mut w_new = vec![vec![0.0; m]; m];
        for i in 0..m {
            // Projection y = w_i^T z over all samples.
            let wi = &w[i];
            let mut e_zg = vec![0.0; m]; // E{ z * g(y) }
            let mut e_gp = 0.0; // E{ g'(y) } = E{ 1 - tanh^2(y) }
            for n in 0..t {
                let mut y = 0.0;
                for (d, wid) in wi.iter().enumerate() {
                    y += wid * z[d][n];
                }
                let g = y.tanh();
                let gp = 1.0 - g * g;
                for (d, ezg) in e_zg.iter_mut().enumerate() {
                    *ezg += z[d][n] * g;
                }
                e_gp += gp;
            }
            let inv_t = 1.0 / t as f64;
            for ezg in e_zg.iter_mut() {
                *ezg *= inv_t;
            }
            e_gp *= inv_t;
            for d in 0..m {
                w_new[i][d] = e_zg[d] - e_gp * wi[d];
            }
        }

        // Symmetric decorrelation: W <- (W W^T)^{-1/2} W.
        symmetric_decorrelate(&mut w_new);

        // Convergence: max over components of |1 - |<w_i_new, w_i_old>||.
        let mut max_dev: f64 = 0.0;
        for i in 0..m {
            let dot: f64 = (0..m).map(|d| w_new[i][d] * w_old[i][d]).sum();
            max_dev = max_dev.max((1.0 - dot.abs()).abs());
        }
        w = w_new;
        if max_dev < cfg.tol {
            converged = true;
            break;
        }
    }

    // Sources in whitened space: S_hat = W Z.
    let sources = matmul_mt(&w, &z, t);

    // Full unmixing back to centered observation space: W_full = W K.
    let unmixing = matmul_square(&w, &k);

    FastIcaResult {
        sources,
        unmixing,
        converged,
        iterations,
    }
}

/// Covariance of centered data `x` (`M x T`): `(1/T) X X^T`, an `M x M` matrix.
fn covariance(x: &[Vec<f64>], t: usize) -> Vec<Vec<f64>> {
    let m = x.len();
    let mut cov = vec![vec![0.0; m]; m];
    let inv_t = 1.0 / t as f64;
    for i in 0..m {
        for j in i..m {
            let mut acc = 0.0;
            for n in 0..t {
                acc += x[i][n] * x[j][n];
            }
            let c = acc * inv_t;
            cov[i][j] = c;
            cov[j][i] = c;
        }
    }
    cov
}

/// Multiply an `M x M` matrix `a` by an `M x T` data matrix `b`, giving `M x T`.
fn matmul_mt(a: &[Vec<f64>], b: &[Vec<f64>], t: usize) -> Vec<Vec<f64>> {
    let m = a.len();
    let k = b.len();
    let mut out = vec![vec![0.0; t]; m];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i][p];
            if aip == 0.0 {
                continue;
            }
            let brow = &b[p];
            let orow = &mut out[i];
            for n in 0..t {
                orow[n] += aip * brow[n];
            }
        }
    }
    out
}

/// Multiply two square `M x M` matrices.
fn matmul_square(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let mut out = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0;
            for p in 0..m {
                acc += a[i][p] * b[p][j];
            }
            out[i][j] = acc;
        }
    }
    out
}

/// A deterministic random orthonormal `M x M` matrix: fill with Gaussians, then
/// symmetric-decorrelate. Used to seed the FastICA weights.
fn random_orthonormal(m: usize, rng: &mut Rng) -> Vec<Vec<f64>> {
    let mut w = vec![vec![0.0; m]; m];
    for row in w.iter_mut() {
        for v in row.iter_mut() {
            *v = rng.gaussian();
        }
    }
    symmetric_decorrelate(&mut w);
    w
}

/// Symmetric decorrelation `W <- (W W^T)^{-1/2} W`, making the rows orthonormal
/// while treating all of them symmetrically (no Gram-Schmidt ordering bias).
///
/// `(W W^T)^{-1/2}` is computed from the eigendecomposition of the symmetric
/// `W W^T`.
fn symmetric_decorrelate(w: &mut [Vec<f64>]) {
    let m = w.len();
    // G = W W^T (symmetric, M x M).
    let mut g = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in i..m {
            let mut acc = 0.0;
            for p in 0..m {
                acc += w[i][p] * w[j][p];
            }
            g[i][j] = acc;
            g[j][i] = acc;
        }
    }
    let eig = jacobi_eigen(&g);
    // G^{-1/2} = E diag(lam^{-1/2}) E^T.
    let mut g_inv_sqrt = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            let mut acc = 0.0;
            for p in 0..m {
                let lam = eig.values[p].max(1e-12);
                acc += eig.vectors[i][p] * (1.0 / lam.sqrt()) * eig.vectors[j][p];
            }
            g_inv_sqrt[i][j] = acc;
        }
    }
    // W <- G^{-1/2} W.
    let new_w = matmul_square(&g_inv_sqrt, w);
    for (dst, src) in w.iter_mut().zip(new_w.into_iter()) {
        dst.copy_from_slice(&src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitened_random_init_is_orthonormal() {
        let mut rng = Rng::new(1);
        let w = random_orthonormal(3, &mut rng);
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|p| w[i][p] * w[j][p]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((dot - want).abs() < 1e-9, "{i}{j} {dot}");
            }
        }
    }
}
