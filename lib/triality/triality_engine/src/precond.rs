//! Preconditioners for iterative solvers.
//!
//! A preconditioner M approximates A⁻¹ so that M·A has a smaller condition number,
//! making iterative solvers converge faster.

use crate::sparse::CsrMatrix;

/// Preconditioner trait: apply M⁻¹ to a vector.
pub trait Preconditioner: Send + Sync {
    fn apply(&self, r: &[f64], z: &mut [f64]);
}

/// Identity preconditioner (no preconditioning).
pub struct IdentityPrecond;

impl Preconditioner for IdentityPrecond {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        z.copy_from_slice(r);
    }
}

/// Jacobi (diagonal) preconditioner: M = diag(A).
/// Cheap to build, effective for diagonally dominant systems.
pub struct JacobiPrecond {
    inv_diag: Vec<f64>,
}

impl JacobiPrecond {
    pub fn new(a: &CsrMatrix) -> Self {
        let diag = a.diagonal();
        let inv_diag: Vec<f64> = diag
            .iter()
            .map(|&d| if d.abs() > 1e-15 { 1.0 / d } else { 1.0 })
            .collect();
        Self { inv_diag }
    }
}

impl Preconditioner for JacobiPrecond {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        for i in 0..r.len() {
            z[i] = self.inv_diag[i] * r[i];
        }
    }
}

/// SSOR preconditioner: Symmetric Successive Over-Relaxation.
/// Better than Jacobi for many FDM/FEM systems.
pub struct SsorPrecond {
    /// Reference to matrix data (owned copy for safety).
    nrows: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<f64>,
    omega: f64,
    diag: Vec<f64>,
}

impl SsorPrecond {
    pub fn new(a: &CsrMatrix, omega: f64) -> Self {
        Self {
            nrows: a.nrows,
            row_ptr: a.row_ptr.clone(),
            col_idx: a.col_idx.clone(),
            values: a.values.clone(),
            omega,
            diag: a.diagonal(),
        }
    }
}

impl Preconditioner for SsorPrecond {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        let n = self.nrows;
        let w = self.omega;

        // Forward sweep: (D + wL) z_half = w * r
        let mut z_half = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                let j = self.col_idx[idx];
                if j < i {
                    sum += self.values[idx] * z_half[j];
                }
            }
            let d = self.diag[i];
            if d.abs() > 1e-15 {
                z_half[i] = w * (r[i] - w * sum) / d;
            } else {
                z_half[i] = r[i];
            }
        }

        // Backward sweep: (D + wU) z = D * z_half
        for i in 0..n {
            z[i] = self.diag[i] * z_half[i];
        }
        for i in (0..n).rev() {
            let mut sum = 0.0;
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                let j = self.col_idx[idx];
                if j > i {
                    sum += self.values[idx] * z[j];
                }
            }
            let d = self.diag[i];
            if d.abs() > 1e-15 {
                z[i] = (z[i] - w * sum) / d;
            }
        }
    }
}

/// Incomplete LU(0) preconditioner.
/// Factorizes A ≈ L·U keeping the same sparsity pattern as A.
/// Most effective preconditioner we offer, but costlier to build.
pub struct Ilu0Precond {
    nrows: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    /// Combined L and U factors stored in the same sparsity pattern.
    lu_values: Vec<f64>,
}

impl Ilu0Precond {
    pub fn new(a: &CsrMatrix) -> Self {
        let n = a.nrows;
        let mut lu = a.values.clone();
        let row_ptr = &a.row_ptr;
        let col_idx = &a.col_idx;

        // IKJ variant of ILU(0)
        for i in 1..n {
            let row_start = row_ptr[i];
            let row_end = row_ptr[i + 1];

            // For each k < i in row i
            for idx_k in row_start..row_end {
                let k = col_idx[idx_k];
                if k >= i {
                    break;
                }

                // Find diagonal of row k
                let mut diag_k = 0.0;
                for jdx in row_ptr[k]..row_ptr[k + 1] {
                    if col_idx[jdx] == k {
                        diag_k = lu[jdx];
                        break;
                    }
                }
                if diag_k.abs() < 1e-15 {
                    continue;
                }

                lu[idx_k] /= diag_k;
                let l_ik = lu[idx_k];

                // Update: row_i[j] -= l_ik * row_k[j] for j > k
                for jdx_k in row_ptr[k]..row_ptr[k + 1] {
                    let j = col_idx[jdx_k];
                    if j <= k {
                        continue;
                    }
                    let u_kj = lu[jdx_k];
                    // Find j in row i
                    for jdx_i in row_start..row_end {
                        if col_idx[jdx_i] == j {
                            lu[jdx_i] -= l_ik * u_kj;
                            break;
                        }
                    }
                }
            }
        }

        Self {
            nrows: n,
            row_ptr: row_ptr.clone(),
            col_idx: col_idx.clone(),
            lu_values: lu,
        }
    }
}

impl Preconditioner for Ilu0Precond {
    fn apply(&self, r: &[f64], z: &mut [f64]) {
        let n = self.nrows;

        // Forward solve: L * y = r
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for idx in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[idx];
                if j < i {
                    sum += self.lu_values[idx] * y[j];
                }
            }
            y[i] = r[i] - sum;
        }

        // Backward solve: U * z = y
        for i in (0..n).rev() {
            let mut sum = 0.0;
            let mut diag = 1.0;
            for idx in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_idx[idx];
                if j > i {
                    sum += self.lu_values[idx] * z[j];
                } else if j == i {
                    diag = self.lu_values[idx];
                }
            }
            if diag.abs() > 1e-15 {
                z[i] = (y[i] - sum) / diag;
            } else {
                z[i] = y[i] - sum;
            }
        }
    }
}

/// Build preconditioner by name.
pub fn build_preconditioner(
    name: &str,
    a: &CsrMatrix,
) -> Box<dyn Preconditioner> {
    match name {
        "jacobi" => Box::new(JacobiPrecond::new(a)),
        "ssor" => Box::new(SsorPrecond::new(a, 1.2)),
        "ilu0" | "ilu" => Box::new(Ilu0Precond::new(a)),
        _ => Box::new(IdentityPrecond),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::TripletBuilder;

    fn make_tridiag(n: usize) -> CsrMatrix {
        let mut tb = TripletBuilder::new(n, n);
        for i in 0..n {
            tb.add(i, i, 2.0);
            if i > 0 {
                tb.add(i, i - 1, -1.0);
            }
            if i < n - 1 {
                tb.add(i, i + 1, -1.0);
            }
        }
        tb.build()
    }

    #[test]
    fn test_jacobi() {
        let a = make_tridiag(5);
        let pc = JacobiPrecond::new(&a);
        let r = vec![1.0; 5];
        let mut z = vec![0.0; 5];
        pc.apply(&r, &mut z);
        assert!((z[0] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_ilu0_solve() {
        let a = make_tridiag(4);
        let pc = Ilu0Precond::new(&a);
        // Apply to known rhs
        let r = vec![1.0, 0.0, 0.0, 1.0];
        let mut z = vec![0.0; 4];
        pc.apply(&r, &mut z);
        // Verify A * z ≈ r
        let residual = a.spmv(&z);
        let err: f64 = r.iter().zip(residual.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        assert!(err < 1e-10, "ILU0 forward/back solve error: {}", err);
    }
}
