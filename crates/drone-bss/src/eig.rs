//! A from-scratch symmetric eigensolver (cyclic Jacobi rotations).
//!
//! FastICA whitens the data by `D^{-1/2} E^T`, where `E` holds the eigenvectors
//! and `D` the eigenvalues of the (symmetric, positive-semidefinite) covariance
//! matrix. Rather than pull in a linear-algebra crate we implement the classic
//! **Jacobi eigenvalue algorithm**: repeatedly apply Givens rotations that zero
//! the largest off-diagonal entry, driving the matrix toward diagonal form. For
//! the small `M x M` covariance matrices here (`M` = number of mics, typically
//! 2-4) this is accurate, simple, and obviously correct - exactly what we want
//! for a `forbid(unsafe_code)` core.
//!
//! The implementation is pure and deterministic: no randomness, no allocation
//! beyond the result vectors.

// Matrix kernels here index two dimensions of `Vec<Vec<f64>>` in lockstep
// (`m[k][p]`, `v[row][p]`, ...). Rewriting these as iterator chains obscures the
// row/column algebra they implement, so the index-loop form is intentional.
#![allow(clippy::needless_range_loop)]

/// Eigen-decomposition of a real symmetric matrix.
///
/// `values[i]` is the eigenvalue whose eigenvector is column `i` of `vectors`
/// (i.e. `vectors[r][i]`). Eigenvalues are returned in **descending** order, so
/// `values[0]` is the largest - the convention PCA whitening expects.
#[derive(Clone, Debug)]
pub struct Eigen {
    /// Eigenvalues, descending.
    pub values: Vec<f64>,
    /// Eigenvectors as columns: `vectors[row][col]`.
    pub vectors: Vec<Vec<f64>>,
}

/// Symmetric eigendecomposition via cyclic Jacobi rotations.
///
/// `a` must be a square, symmetric `n x n` matrix (only the values matter; we do
/// not check symmetry, but the covariance matrices we feed it are symmetric by
/// construction). Returns eigenvalues in descending order with matching
/// eigenvector columns.
///
/// # Panics
///
/// Panics if `a` is empty or not square.
pub fn jacobi_eigen(a: &[Vec<f64>]) -> Eigen {
    let n = a.len();
    assert!(n > 0, "eigen: empty matrix");
    assert!(a.iter().all(|row| row.len() == n), "eigen: not square");

    // Working copy that we rotate toward diagonal form.
    let mut m: Vec<Vec<f64>> = a.to_vec();
    // Accumulated eigenvectors, start at identity.
    let mut v: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();

    // Cyclic Jacobi: sweep over all (p, q) pairs until off-diagonal mass is
    // negligible or we hit the iteration cap. 100 sweeps is far more than the
    // ~10 a tiny matrix ever needs; it is a safety bound, not a tuning knob.
    const MAX_SWEEPS: usize = 100;
    for _ in 0..MAX_SWEEPS {
        let off = off_diagonal_norm(&m);
        if off < 1e-18 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p][q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                // Rotation angle that zeros m[p][q]:
                //   cot(2θ) = (a_qq - a_pp) / (2 a_pq)
                let app = m[p][p];
                let aqq = m[q][q];
                let phi = 0.5 * (aqq - app) / apq;
                // t = sign(phi) / (|phi| + sqrt(phi^2 + 1)), the smaller root,
                // for numerical stability.
                let t = phi.signum() / (phi.abs() + (phi * phi + 1.0).sqrt());
                let t = if phi == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // Apply the Givens rotation J^T M J in place over rows/cols p,q.
                for k in 0..n {
                    let mkp = m[k][p];
                    let mkq = m[k][q];
                    m[k][p] = c * mkp - s * mkq;
                    m[k][q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let mpk = m[p][k];
                    let mqk = m[q][k];
                    m[p][k] = c * mpk - s * mqk;
                    m[q][k] = s * mpk + c * mqk;
                }
                // Accumulate the rotation into the eigenvector matrix.
                for row in v.iter_mut() {
                    let vp = row[p];
                    let vq = row[q];
                    row[p] = c * vp - s * vq;
                    row[q] = s * vp + c * vq;
                }
            }
        }
    }

    // Diagonal now holds the eigenvalues; sort descending and reorder vectors.
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| {
        m[j][j]
            .partial_cmp(&m[i][i])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let values: Vec<f64> = idx.iter().map(|&i| m[i][i]).collect();
    let vectors: Vec<Vec<f64>> = (0..n)
        .map(|r| idx.iter().map(|&c| v[r][c]).collect())
        .collect();

    Eigen { values, vectors }
}

/// Frobenius norm of the strictly-upper off-diagonal part (the quantity Jacobi
/// drives to zero).
fn off_diagonal_norm(m: &[Vec<f64>]) -> f64 {
    let n = m.len();
    let mut acc = 0.0;
    for p in 0..n {
        for q in (p + 1)..n {
            acc += m[p][q] * m[p][q];
        }
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = a.len();
        let m = b[0].len();
        let k = b.len();
        let mut out = vec![vec![0.0; m]; n];
        for (i, orow) in out.iter_mut().enumerate() {
            for (j, ocell) in orow.iter_mut().enumerate() {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += a[i][p] * b[p][j];
                }
                *ocell = acc;
            }
        }
        out
    }

    #[test]
    fn diagonal_matrix_trivial() {
        let a = vec![vec![3.0, 0.0], vec![0.0, 1.0]];
        let e = jacobi_eigen(&a);
        assert!((e.values[0] - 3.0).abs() < 1e-12);
        assert!((e.values[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn reconstructs_known_symmetric() {
        // A = V diag(l) V^T should be recovered.
        let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let e = jacobi_eigen(&a);
        // Eigenvalues of [[2,1],[1,2]] are 3 and 1.
        assert!((e.values[0] - 3.0).abs() < 1e-10, "{:?}", e.values);
        assert!((e.values[1] - 1.0).abs() < 1e-10, "{:?}", e.values);

        // V^T A V should be diagonal with the eigenvalues.
        let n = 2;
        let vt: Vec<Vec<f64>> = (0..n)
            .map(|i| (0..n).map(|j| e.vectors[j][i]).collect())
            .collect();
        let recon = matmul(&matmul(&vt, &a), &e.vectors);
        assert!((recon[0][0] - e.values[0]).abs() < 1e-9);
        assert!((recon[1][1] - e.values[1]).abs() < 1e-9);
        assert!(recon[0][1].abs() < 1e-9);
    }

    #[test]
    fn orthonormal_eigenvectors() {
        let a = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.2],
            vec![0.5, 0.2, 2.0],
        ];
        let e = jacobi_eigen(&a);
        let n = 3;
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..n).map(|r| e.vectors[r][i] * e.vectors[r][j]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-9, "i{i} j{j} dot{dot}");
            }
        }
        // Eigenvalues descending.
        assert!(e.values[0] >= e.values[1] && e.values[1] >= e.values[2]);
    }
}
