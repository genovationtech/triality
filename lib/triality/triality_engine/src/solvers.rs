//! Iterative and direct linear solvers.
//!
//! All solvers operate on CSR matrices and return a SolveResult.
//! The Python layer selects which solver to call; Rust just executes fast.

use crate::precond::build_preconditioner;
use crate::sparse::CsrMatrix;

/// Result of a linear solve.
#[derive(Debug, Clone)]
pub struct SolveResult {
    pub x: Vec<f64>,
    pub converged: bool,
    pub iterations: usize,
    pub residual: f64,
}

/// Compute ||b - A*x||₂.
fn residual_norm(a: &CsrMatrix, x: &[f64], b: &[f64]) -> f64 {
    let ax = a.spmv(x);
    ax.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (bi - ai).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Dot product.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Vector norm.
#[inline]
fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// axpy: y = a*x + y.
#[inline]
fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

// ─── Conjugate Gradient (for SPD systems) ───────────────────────────────

/// Preconditioned Conjugate Gradient solver.
/// Requires A to be symmetric positive definite.
pub fn solve_cg(
    a: &CsrMatrix,
    b: &[f64],
    tol: f64,
    max_iter: usize,
    precond_name: &str,
) -> SolveResult {
    let n = b.len();
    let pc = build_preconditioner(precond_name, a);

    let mut x = vec![0.0; n];
    let mut r: Vec<f64> = b.to_vec();
    let mut z = vec![0.0; n];
    pc.apply(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);

    let b_norm = norm(b);
    if b_norm < 1e-15 {
        return SolveResult {
            x,
            converged: true,
            iterations: 0,
            residual: 0.0,
        };
    }

    for iter in 0..max_iter {
        let ap = a.spmv(&p);
        let pap = dot(&p, &ap);
        if pap.abs() < 1e-30 {
            break;
        }
        let alpha = rz / pap;

        axpy(alpha, &p, &mut x);

        // r = r - alpha * ap
        for (ri, api) in r.iter_mut().zip(ap.iter()) {
            *ri -= alpha * api;
        }

        let r_norm = norm(&r);
        if r_norm / b_norm < tol {
            return SolveResult {
                x,
                converged: true,
                iterations: iter + 1,
                residual: r_norm,
            };
        }

        pc.apply(&r, &mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;

        // p = z + beta * p
        for (pi, zi) in p.iter_mut().zip(z.iter()) {
            *pi = *zi + beta * *pi;
        }

        rz = rz_new;
    }

    let residual = residual_norm(a, &x, b);
    SolveResult {
        x,
        converged: false,
        iterations: max_iter,
        residual,
    }
}

// ─── GMRES(m) ───────────────────────────────────────────────────────────

/// Restarted GMRES solver. Works for any nonsingular system.
pub fn solve_gmres(
    a: &CsrMatrix,
    b: &[f64],
    tol: f64,
    max_iter: usize,
    restart: usize,
    precond_name: &str,
) -> SolveResult {
    let n = b.len();
    let pc = build_preconditioner(precond_name, a);
    let mut x = vec![0.0; n];
    let b_norm = norm(b);
    if b_norm < 1e-15 {
        return SolveResult {
            x,
            converged: true,
            iterations: 0,
            residual: 0.0,
        };
    }

    let m = restart.min(n).min(max_iter);
    let mut total_iter = 0;

    for _outer in 0..(max_iter / m + 1) {
        // r = b - A*x
        let ax = a.spmv(&x);
        let mut r: Vec<f64> = b.iter().zip(ax.iter()).map(|(bi, ai)| bi - ai).collect();

        let mut z_tmp = vec![0.0; n];
        pc.apply(&r, &mut z_tmp);
        r = z_tmp;

        let beta = norm(&r);
        if beta / b_norm < tol {
            return SolveResult {
                x,
                converged: true,
                iterations: total_iter,
                residual: beta,
            };
        }

        // Arnoldi basis
        let mut v: Vec<Vec<f64>> = vec![vec![0.0; n]; m + 1];
        for i in 0..n {
            v[0][i] = r[i] / beta;
        }

        // Upper Hessenberg matrix H (stored as (m+1) x m)
        let mut h = vec![vec![0.0f64; m]; m + 1];
        // Givens rotation parameters
        let mut cs = vec![0.0f64; m];
        let mut sn = vec![0.0f64; m];
        let mut g = vec![0.0f64; m + 1];
        g[0] = beta;

        let mut j = 0;
        while j < m && total_iter < max_iter {
            total_iter += 1;

            // w = M⁻¹ A v[j]
            let av = a.spmv(&v[j]);
            let mut w = vec![0.0; n];
            pc.apply(&av, &mut w);

            // Modified Gram-Schmidt
            for i in 0..=j {
                h[i][j] = dot(&w, &v[i]);
                axpy(-h[i][j], &v[i], &mut w);
            }
            h[j + 1][j] = norm(&w);

            if h[j + 1][j].abs() < 1e-15 {
                // Lucky breakdown
                j += 1;
                break;
            }
            for i in 0..n {
                v[j + 1][i] = w[i] / h[j + 1][j];
            }

            // Apply previous Givens rotations to column j of H
            for i in 0..j {
                let tmp = cs[i] * h[i][j] + sn[i] * h[i + 1][j];
                h[i + 1][j] = -sn[i] * h[i][j] + cs[i] * h[i + 1][j];
                h[i][j] = tmp;
            }

            // Compute new Givens rotation
            let rr = (h[j][j].powi(2) + h[j + 1][j].powi(2)).sqrt();
            if rr.abs() < 1e-30 {
                j += 1;
                break;
            }
            cs[j] = h[j][j] / rr;
            sn[j] = h[j + 1][j] / rr;
            h[j][j] = rr;
            h[j + 1][j] = 0.0;

            g[j + 1] = -sn[j] * g[j];
            g[j] = cs[j] * g[j];

            if g[j + 1].abs() / b_norm < tol {
                j += 1;
                break;
            }

            j += 1;
        }

        // Back-substitution: solve H[0..j, 0..j] * y = g[0..j]
        let mut y = vec![0.0f64; j];
        for i in (0..j).rev() {
            y[i] = g[i];
            for k in (i + 1)..j {
                y[i] -= h[i][k] * y[k];
            }
            if h[i][i].abs() > 1e-15 {
                y[i] /= h[i][i];
            }
        }

        // x = x + V * y
        for i in 0..j {
            axpy(y[i], &v[i], &mut x);
        }

        let residual = residual_norm(a, &x, b);
        if residual / b_norm < tol {
            return SolveResult {
                x,
                converged: true,
                iterations: total_iter,
                residual,
            };
        }
    }

    let residual = residual_norm(a, &x, b);
    SolveResult {
        x,
        converged: false,
        iterations: total_iter,
        residual,
    }
}

// ─── BiCGSTAB ───────────────────────────────────────────────────────────

/// BiCGSTAB solver. Good for nonsymmetric systems.
pub fn solve_bicgstab(
    a: &CsrMatrix,
    b: &[f64],
    tol: f64,
    max_iter: usize,
    precond_name: &str,
) -> SolveResult {
    let n = b.len();
    let pc = build_preconditioner(precond_name, a);
    let mut x = vec![0.0; n];
    let b_norm = norm(b);
    if b_norm < 1e-15 {
        return SolveResult {
            x,
            converged: true,
            iterations: 0,
            residual: 0.0,
        };
    }

    let mut r: Vec<f64> = b.to_vec();
    let r0_hat = r.clone();

    let mut rho = 1.0f64;
    let mut alpha = 1.0f64;
    let mut omega = 1.0f64;

    let mut v = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut t_vec = vec![0.0; n];
    let mut z = vec![0.0; n];

    for iter in 0..max_iter {
        let rho_new = dot(&r0_hat, &r);
        if rho_new.abs() < 1e-30 {
            break;
        }

        let beta = (rho_new / rho) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        }

        // v = A * M⁻¹ * p
        pc.apply(&p, &mut z);
        v = a.spmv(&z);

        let denom = dot(&r0_hat, &v);
        if denom.abs() < 1e-30 {
            break;
        }
        alpha = rho_new / denom;

        // s = r - alpha * v
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
        }

        let s_norm = norm(&s);
        if s_norm / b_norm < tol {
            axpy(alpha, &z, &mut x);
            return SolveResult {
                x,
                converged: true,
                iterations: iter + 1,
                residual: s_norm,
            };
        }

        // t = A * M⁻¹ * s
        let mut z2 = vec![0.0; n];
        pc.apply(&s, &mut z2);
        t_vec = a.spmv(&z2);

        let tt = dot(&t_vec, &t_vec);
        omega = if tt.abs() > 1e-30 {
            dot(&t_vec, &s) / tt
        } else {
            1.0
        };

        // x = x + alpha * M⁻¹p + omega * M⁻¹s
        axpy(alpha, &z, &mut x);
        axpy(omega, &z2, &mut x);

        // r = s - omega * t
        for i in 0..n {
            r[i] = s[i] - omega * t_vec[i];
        }

        let r_norm = norm(&r);
        if r_norm / b_norm < tol {
            return SolveResult {
                x,
                converged: true,
                iterations: iter + 1,
                residual: r_norm,
            };
        }

        rho = rho_new;
    }

    let residual = residual_norm(a, &x, b);
    SolveResult {
        x,
        converged: false,
        iterations: max_iter,
        residual,
    }
}

// ─── Direct solver (LU factorization) ───────────────────────────────────

/// Simple dense LU solver for small systems (< ~5000 DOF).
/// For larger systems, use iterative solvers above.
pub fn solve_direct(a: &CsrMatrix, b: &[f64]) -> SolveResult {
    let n = a.nrows;
    assert_eq!(n, a.ncols, "Direct solver requires square matrix");
    assert_eq!(n, b.len(), "RHS length mismatch");

    // Convert to dense (only practical for small systems)
    let mut lu = vec![0.0f64; n * n];
    for i in 0..n {
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            lu[i * n + a.col_idx[idx]] = a.values[idx];
        }
    }

    // LU factorization with partial pivoting
    let mut perm: Vec<usize> = (0..n).collect();
    for k in 0..n {
        // Find pivot
        let mut max_val = lu[perm[k] * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let val = lu[perm[i] * n + k].abs();
            if val > max_val {
                max_val = val;
                max_row = i;
            }
        }
        perm.swap(k, max_row);

        let pivot = lu[perm[k] * n + k];
        if pivot.abs() < 1e-15 {
            return SolveResult {
                x: vec![0.0; n],
                converged: false,
                iterations: 0,
                residual: f64::INFINITY,
            };
        }

        for i in (k + 1)..n {
            let factor = lu[perm[i] * n + k] / pivot;
            lu[perm[i] * n + k] = factor;
            for j in (k + 1)..n {
                let pk = perm[k];
                let pi = perm[i];
                lu[pi * n + j] -= factor * lu[pk * n + j];
            }
        }
    }

    // Forward substitution: Ly = Pb
    let mut y = vec![0.0; n];
    for i in 0..n {
        y[i] = b[perm[i]];
        for j in 0..i {
            y[i] -= lu[perm[i] * n + j] * y[j];
        }
    }

    // Backward substitution: Ux = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = y[i];
        for j in (i + 1)..n {
            x[i] -= lu[perm[i] * n + j] * x[j];
        }
        x[i] /= lu[perm[i] * n + i];
    }

    let residual = residual_norm(a, &x, b);
    SolveResult {
        x,
        converged: true,
        iterations: 1,
        residual,
    }
}

/// Top-level dispatch: pick solver by name.
pub fn solve_linear(
    a: &CsrMatrix,
    b: &[f64],
    method: &str,
    precond: &str,
    tol: f64,
    max_iter: usize,
) -> SolveResult {
    match method {
        "cg" => solve_cg(a, b, tol, max_iter, precond),
        "gmres" => solve_gmres(a, b, tol, max_iter, 30, precond),
        "bicgstab" => solve_bicgstab(a, b, tol, max_iter, precond),
        "direct" => solve_direct(a, b),
        "auto" => {
            // Default to GMRES for general case
            solve_gmres(a, b, tol, max_iter, 30, precond)
        }
        _ => solve_gmres(a, b, tol, max_iter, 30, precond),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::TripletBuilder;

    fn make_poisson_1d(n: usize) -> (CsrMatrix, Vec<f64>) {
        let h = 1.0 / (n as f64 + 1.0);
        let mut tb = TripletBuilder::new(n, n);
        let mut b = vec![0.0; n];

        for i in 0..n {
            tb.add(i, i, 2.0 / (h * h));
            if i > 0 {
                tb.add(i, i - 1, -1.0 / (h * h));
            }
            if i < n - 1 {
                tb.add(i, i + 1, -1.0 / (h * h));
            }
            // f(x) = 1 everywhere
            b[i] = 1.0;
        }

        (tb.build(), b)
    }

    #[test]
    fn test_cg_poisson() {
        let (a, b) = make_poisson_1d(50);
        let result = solve_cg(&a, &b, 1e-10, 200, "none");
        assert!(result.converged, "CG did not converge");
        assert!(result.residual < 1e-6);
    }

    #[test]
    fn test_gmres_poisson() {
        let (a, b) = make_poisson_1d(50);
        let result = solve_gmres(&a, &b, 1e-10, 200, 30, "jacobi");
        assert!(result.converged, "GMRES did not converge");
        assert!(result.residual < 1e-6);
    }

    #[test]
    fn test_bicgstab_poisson() {
        let (a, b) = make_poisson_1d(50);
        let result = solve_bicgstab(&a, &b, 1e-10, 200, "none");
        assert!(result.converged, "BiCGSTAB did not converge");
        assert!(result.residual < 1e-6);
    }

    #[test]
    fn test_direct_small() {
        let (a, b) = make_poisson_1d(10);
        let result = solve_direct(&a, &b);
        assert!(result.converged);
        assert!(result.residual < 1e-10);
    }
}
