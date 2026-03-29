//! Time integration methods for parabolic and hyperbolic PDEs.
//!
//! These turn du/dt = F(u) into a sequence of linear solves or explicit updates.
//! The spatial operator (e.g. Laplacian) comes from the FDM module.

use crate::sparse::CsrMatrix;
use crate::solvers::solve_linear;

/// Result of a time-stepping simulation.
#[derive(Debug, Clone)]
pub struct TimeStepResult {
    /// Solution at each saved time step. Each entry is the solution vector.
    pub snapshots: Vec<Vec<f64>>,
    /// Times corresponding to each snapshot.
    pub times: Vec<f64>,
    /// Total number of time steps taken.
    pub steps_taken: usize,
}

// ─── Forward Euler (explicit) ───────────────────────────────────────────

/// Forward Euler: u^{n+1} = u^n + dt * A * u^n + dt * f.
///
/// Simple but has CFL stability restriction: dt < h²/(2*d*max_coeff) for diffusion.
/// Good for: quick prototyping, hyperbolic PDEs.
pub fn forward_euler(
    a: &CsrMatrix,
    u0: &[f64],
    f_rhs: &[f64],
    dt: f64,
    n_steps: usize,
    save_every: usize,
) -> TimeStepResult {
    let n = u0.len();
    let mut u = u0.to_vec();
    let mut snapshots = vec![u.clone()];
    let mut times = vec![0.0];

    for step in 1..=n_steps {
        let au = a.spmv(&u);
        for i in 0..n {
            u[i] += dt * (au[i] + f_rhs[i]);
        }

        if step % save_every == 0 || step == n_steps {
            snapshots.push(u.clone());
            times.push(step as f64 * dt);
        }
    }

    TimeStepResult {
        snapshots,
        times,
        steps_taken: n_steps,
    }
}

// ─── Backward Euler (implicit, unconditionally stable) ──────────────────

/// Backward Euler: (I - dt*A) u^{n+1} = u^n + dt * f.
///
/// Unconditionally stable (can use large dt). Requires one linear solve per step.
/// Good for: diffusion problems, stiff systems.
pub fn backward_euler(
    a: &CsrMatrix,
    u0: &[f64],
    f_rhs: &[f64],
    dt: f64,
    n_steps: usize,
    save_every: usize,
    solver_method: &str,
    precond: &str,
) -> TimeStepResult {
    let n = u0.len();

    // Build (I - dt*A) matrix
    let lhs = build_implicit_lhs(a, dt);

    let mut u = u0.to_vec();
    let mut snapshots = vec![u.clone()];
    let mut times = vec![0.0];

    let tol = 1e-10;
    let max_iter = n * 5;

    for step in 1..=n_steps {
        // RHS = u^n + dt * f
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            rhs[i] = u[i] + dt * f_rhs[i];
        }

        let result = solve_linear(&lhs, &rhs, solver_method, precond, tol, max_iter);
        u = result.x;

        if step % save_every == 0 || step == n_steps {
            snapshots.push(u.clone());
            times.push(step as f64 * dt);
        }
    }

    TimeStepResult {
        snapshots,
        times,
        steps_taken: n_steps,
    }
}

// ─── Crank-Nicolson (implicit, 2nd-order in time) ───────────────────────

/// Crank-Nicolson: (I - dt/2*A) u^{n+1} = (I + dt/2*A) u^n + dt * f.
///
/// Second-order accurate in time AND unconditionally stable.
/// The sweet spot for diffusion equations.
pub fn crank_nicolson(
    a: &CsrMatrix,
    u0: &[f64],
    f_rhs: &[f64],
    dt: f64,
    n_steps: usize,
    save_every: usize,
    solver_method: &str,
    precond: &str,
) -> TimeStepResult {
    let n = u0.len();

    // LHS: (I - dt/2 * A)
    let lhs = build_implicit_lhs(a, dt / 2.0);

    let mut u = u0.to_vec();
    let mut snapshots = vec![u.clone()];
    let mut times = vec![0.0];

    let tol = 1e-10;
    let max_iter = n * 5;

    for step in 1..=n_steps {
        // RHS: (I + dt/2 * A) u^n + dt * f
        let au = a.spmv(&u);
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            rhs[i] = u[i] + (dt / 2.0) * au[i] + dt * f_rhs[i];
        }

        let result = solve_linear(&lhs, &rhs, solver_method, precond, tol, max_iter);
        u = result.x;

        if step % save_every == 0 || step == n_steps {
            snapshots.push(u.clone());
            times.push(step as f64 * dt);
        }
    }

    TimeStepResult {
        snapshots,
        times,
        steps_taken: n_steps,
    }
}

// ─── Classical RK4 (explicit, 4th-order) ────────────────────────────────

/// Classical Runge-Kutta 4th order: du/dt = A*u + f.
///
/// High accuracy per step, but explicit (CFL restricted).
/// Good for: wave equations, non-stiff systems where accuracy matters.
pub fn rk4(
    a: &CsrMatrix,
    u0: &[f64],
    f_rhs: &[f64],
    dt: f64,
    n_steps: usize,
    save_every: usize,
) -> TimeStepResult {
    let n = u0.len();
    let mut u = u0.to_vec();
    let mut snapshots = vec![u.clone()];
    let mut times = vec![0.0];

    let eval_rhs = |u_in: &[f64]| -> Vec<f64> {
        let au = a.spmv(u_in);
        au.iter().zip(f_rhs.iter()).map(|(a, f)| a + f).collect()
    };

    for step in 1..=n_steps {
        let k1 = eval_rhs(&u);

        let mut u_tmp = vec![0.0; n];
        for i in 0..n {
            u_tmp[i] = u[i] + 0.5 * dt * k1[i];
        }
        let k2 = eval_rhs(&u_tmp);

        for i in 0..n {
            u_tmp[i] = u[i] + 0.5 * dt * k2[i];
        }
        let k3 = eval_rhs(&u_tmp);

        for i in 0..n {
            u_tmp[i] = u[i] + dt * k3[i];
        }
        let k4 = eval_rhs(&u_tmp);

        for i in 0..n {
            u[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }

        if step % save_every == 0 || step == n_steps {
            snapshots.push(u.clone());
            times.push(step as f64 * dt);
        }
    }

    TimeStepResult {
        snapshots,
        times,
        steps_taken: n_steps,
    }
}

// ─── BDF2 (implicit, 2nd-order, A-stable) ──────────────────────────────

/// BDF2: (3/2) u^{n+1} - 2 u^n + (1/2) u^{n-1} = dt * (A u^{n+1} + f).
///
/// A-stable, 2nd-order, excellent for stiff problems.
/// Uses Backward Euler for the first step.
pub fn bdf2(
    a: &CsrMatrix,
    u0: &[f64],
    f_rhs: &[f64],
    dt: f64,
    n_steps: usize,
    save_every: usize,
    solver_method: &str,
    precond: &str,
) -> TimeStepResult {
    let n = u0.len();
    let tol = 1e-10;
    let max_iter = n * 5;

    // (3/(2*dt)) I - A
    let coeff = 3.0 / (2.0 * dt);
    let lhs = build_scaled_identity_minus_a(a, coeff);

    let mut u_prev = u0.to_vec();
    let mut snapshots = vec![u_prev.clone()];
    let mut times = vec![0.0];

    // First step: Backward Euler (need u^1 before BDF2 kicks in)
    let lhs_be = build_implicit_lhs(a, dt);
    let mut rhs_be = vec![0.0; n];
    for i in 0..n {
        rhs_be[i] = u_prev[i] + dt * f_rhs[i];
    }
    let result = solve_linear(&lhs_be, &rhs_be, solver_method, precond, tol, max_iter);
    let mut u_curr = result.x;

    if save_every == 1 {
        snapshots.push(u_curr.clone());
        times.push(dt);
    }

    // BDF2 steps
    for step in 2..=n_steps {
        // RHS = (2/dt) u^n - (1/(2*dt)) u^{n-1} + f
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            rhs[i] = (2.0 / dt) * u_curr[i] - (1.0 / (2.0 * dt)) * u_prev[i] + f_rhs[i];
        }

        let result = solve_linear(&lhs, &rhs, solver_method, precond, tol, max_iter);

        u_prev = u_curr;
        u_curr = result.x;

        if step % save_every == 0 || step == n_steps {
            snapshots.push(u_curr.clone());
            times.push(step as f64 * dt);
        }
    }

    TimeStepResult {
        snapshots,
        times,
        steps_taken: n_steps,
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────

/// Build (I - dt * A) as a new CSR matrix.
fn build_implicit_lhs(a: &CsrMatrix, dt: f64) -> CsrMatrix {
    let n = a.nrows;
    let row_ptr = a.row_ptr.clone();
    let col_idx = a.col_idx.clone();
    let mut values: Vec<f64> = a.values.iter().map(|&v| -dt * v).collect();

    // Add identity: need to find or insert diagonal entries
    // Since our FDM matrices always have diagonal entries, just add 1.0 to them
    for i in 0..n {
        let mut found = false;
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            if a.col_idx[idx] == i {
                values[idx] += 1.0;
                found = true;
                break;
            }
        }
        // If diagonal wasn't in sparsity pattern, we'd need to insert it.
        // For FDM matrices this never happens, but guard anyway.
        if !found {
            // This is a simplified fallback - in practice FDM always has diag
            // We'd need a full rebuild to insert, but skip for now
        }
    }

    CsrMatrix {
        nrows: n,
        ncols: a.ncols,
        row_ptr,
        col_idx,
        values,
    }
}

/// Build (c*I - A) as a new CSR matrix.
fn build_scaled_identity_minus_a(a: &CsrMatrix, c: f64) -> CsrMatrix {
    let n = a.nrows;
    let mut values: Vec<f64> = a.values.iter().map(|&v| -v).collect();

    for i in 0..n {
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            if a.col_idx[idx] == i {
                values[idx] += c;
                break;
            }
        }
    }

    CsrMatrix {
        nrows: n,
        ncols: a.ncols,
        row_ptr: a.row_ptr.clone(),
        col_idx: a.col_idx.clone(),
        values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fdm::laplacian_1d;

    #[test]
    fn test_heat_equation_euler() {
        // Heat equation: du/dt = d²u/dx², u(0)=u(1)=0, u(x,0) = sin(pi*x)
        // Exact: u(x,t) = exp(-pi²*t) * sin(pi*x)
        let n = 51;
        let (a, grid) = laplacian_1d(0.0, 1.0, n);

        // Initial condition
        let u0: Vec<f64> = grid
            .iter()
            .map(|&x| (std::f64::consts::PI * x).sin())
            .collect();
        let f_rhs = vec![0.0; n];

        let h = 1.0 / (n as f64 - 1.0);
        let dt = 0.4 * h * h; // CFL safe
        let n_steps = 100;

        let result = forward_euler(&a, &u0, &f_rhs, dt, n_steps, n_steps);
        assert_eq!(result.snapshots.len(), 2); // initial + final

        // Check solution hasn't blown up (stability check)
        let u_final = &result.snapshots[1];
        let max_val = u_final.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert!(max_val < 2.0, "Forward Euler blew up: max = {}", max_val);
    }

    #[test]
    fn test_heat_equation_backward_euler() {
        let n = 21;
        let (a, grid) = laplacian_1d(0.0, 1.0, n);
        let u0: Vec<f64> = grid
            .iter()
            .map(|&x| (std::f64::consts::PI * x).sin())
            .collect();
        let f_rhs = vec![0.0; n];

        // Large dt (unconditionally stable)
        let dt = 0.01;
        let n_steps = 10;

        let result = backward_euler(&a, &u0, &f_rhs, dt, n_steps, n_steps, "cg", "none");

        let u_final = &result.snapshots[1];
        let max_val = u_final.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        assert!(max_val < 1.0, "Solution should decay");
        assert!(max_val > 0.0, "Solution should not be zero");
    }

    #[test]
    fn test_crank_nicolson_accuracy() {
        let n = 51;
        let (a, grid) = laplacian_1d(0.0, 1.0, n);
        let u0: Vec<f64> = grid
            .iter()
            .map(|&x| (std::f64::consts::PI * x).sin())
            .collect();
        let f_rhs = vec![0.0; n];

        let dt = 0.001;
        let t_final = 0.01;
        let n_steps = (t_final / dt) as usize;

        let result = crank_nicolson(&a, &u0, &f_rhs, dt, n_steps, n_steps, "cg", "none");

        // Exact at t=0.01: exp(-pi²*0.01) * sin(pi*x)
        let decay = (-std::f64::consts::PI.powi(2) * t_final).exp();
        let u_final = &result.snapshots[1];
        let mid = n / 2;
        let exact_mid = decay * (std::f64::consts::PI * grid[mid]).sin();
        let error = (u_final[mid] - exact_mid).abs();
        assert!(
            error < 0.01,
            "CN error at midpoint: {} (got {}, expected {})",
            error,
            u_final[mid],
            exact_mid
        );
    }
}
