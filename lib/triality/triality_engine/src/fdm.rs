//! Finite Difference Method stencil assembly.
//!
//! Builds CSR matrices for standard PDE operators on uniform grids.
//! This is the most common bottleneck in Triality's existing Python code
//! (the nested Python loops in fdm.py).

use crate::sparse::{CsrMatrix, TripletBuilder};

/// Assemble 1D Laplacian: d²u/dx² on [a, b] with n points.
///
/// Returns (A, grid) where A is the n×n stencil matrix [1, -2, 1]/h².
/// Boundary rows are identity (for Dirichlet BC injection).
pub fn laplacian_1d(a: f64, b: f64, n: usize) -> (CsrMatrix, Vec<f64>) {
    let h = (b - a) / (n as f64 - 1.0);
    let h2 = h * h;
    let grid: Vec<f64> = (0..n).map(|i| a + i as f64 * h).collect();

    // Estimate: 3 entries per interior row + 1 per boundary row
    let mut tb = TripletBuilder::with_capacity(n, n, 3 * n);

    // Boundary: row 0
    tb.add(0, 0, 1.0);

    // Interior: [1, -2, 1] / h²
    for i in 1..(n - 1) {
        tb.add(i, i - 1, 1.0 / h2);
        tb.add(i, i, -2.0 / h2);
        tb.add(i, i + 1, 1.0 / h2);
    }

    // Boundary: row n-1
    tb.add(n - 1, n - 1, 1.0);

    (tb.build(), grid)
}

/// Assemble 1D Laplacian with variable coefficient:
/// d/dx[k(x) du/dx] on [a, b] with n points.
///
/// k_values: conductivity/diffusivity at each grid point.
/// Uses harmonic averaging at cell interfaces (physically correct for discontinuous k).
pub fn laplacian_1d_variable(
    a: f64,
    b: f64,
    n: usize,
    k_values: &[f64],
) -> (CsrMatrix, Vec<f64>) {
    assert_eq!(k_values.len(), n, "k_values length must match grid size");
    let h = (b - a) / (n as f64 - 1.0);
    let h2 = h * h;
    let grid: Vec<f64> = (0..n).map(|i| a + i as f64 * h).collect();

    let mut tb = TripletBuilder::with_capacity(n, n, 3 * n);

    tb.add(0, 0, 1.0);

    for i in 1..(n - 1) {
        // Harmonic mean at interfaces
        let k_left = 2.0 * k_values[i - 1] * k_values[i] / (k_values[i - 1] + k_values[i]);
        let k_right = 2.0 * k_values[i] * k_values[i + 1] / (k_values[i] + k_values[i + 1]);

        tb.add(i, i - 1, k_left / h2);
        tb.add(i, i, -(k_left + k_right) / h2);
        tb.add(i, i + 1, k_right / h2);
    }

    tb.add(n - 1, n - 1, 1.0);

    (tb.build(), grid)
}

/// Assemble 2D Laplacian: ∇²u = d²u/dx² + d²u/dy² on [x0,x1]×[y0,y1].
///
/// nx × ny grid. Returns (A, grid_x, grid_y).
/// Uses 5-point stencil. Boundary rows are identity.
pub fn laplacian_2d(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    nx: usize,
    ny: usize,
) -> (CsrMatrix, Vec<f64>, Vec<f64>) {
    let dx = (x1 - x0) / (nx as f64 - 1.0);
    let dy = (y1 - y0) / (ny as f64 - 1.0);
    let dx2 = dx * dx;
    let dy2 = dy * dy;

    let grid_x: Vec<f64> = (0..nx).map(|i| x0 + i as f64 * dx).collect();
    let grid_y: Vec<f64> = (0..ny).map(|j| y0 + j as f64 * dy).collect();

    let ndof = nx * ny;
    let mut tb = TripletBuilder::with_capacity(ndof, ndof, 5 * ndof);

    let idx = |i: usize, j: usize| -> usize { i * ny + j };

    for i in 0..nx {
        for j in 0..ny {
            let k = idx(i, j);

            // Boundary point
            if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 {
                tb.add(k, k, 1.0);
                continue;
            }

            // Interior: 5-point stencil
            tb.add(k, k, -2.0 / dx2 - 2.0 / dy2);
            tb.add(k, idx(i - 1, j), 1.0 / dx2);
            tb.add(k, idx(i + 1, j), 1.0 / dx2);
            tb.add(k, idx(i, j - 1), 1.0 / dy2);
            tb.add(k, idx(i, j + 1), 1.0 / dy2);
        }
    }

    (tb.build(), grid_x, grid_y)
}

/// Assemble 2D Laplacian with variable coefficient: ∇·(k(x,y) ∇u).
///
/// k_values: nx×ny array of conductivity values (row-major).
pub fn laplacian_2d_variable(
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    nx: usize,
    ny: usize,
    k_values: &[f64],
) -> (CsrMatrix, Vec<f64>, Vec<f64>) {
    assert_eq!(k_values.len(), nx * ny);
    let dx = (x1 - x0) / (nx as f64 - 1.0);
    let dy = (y1 - y0) / (ny as f64 - 1.0);
    let dx2 = dx * dx;
    let dy2 = dy * dy;

    let grid_x: Vec<f64> = (0..nx).map(|i| x0 + i as f64 * dx).collect();
    let grid_y: Vec<f64> = (0..ny).map(|j| y0 + j as f64 * dy).collect();

    let ndof = nx * ny;
    let mut tb = TripletBuilder::with_capacity(ndof, ndof, 5 * ndof);

    let idx = |i: usize, j: usize| -> usize { i * ny + j };
    let k_at = |i: usize, j: usize| -> f64 { k_values[i * ny + j] };

    for i in 0..nx {
        for j in 0..ny {
            let k = idx(i, j);

            if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 {
                tb.add(k, k, 1.0);
                continue;
            }

            // Harmonic averages at interfaces
            let k_c = k_at(i, j);
            let k_xm = 2.0 * k_at(i - 1, j) * k_c / (k_at(i - 1, j) + k_c);
            let k_xp = 2.0 * k_c * k_at(i + 1, j) / (k_c + k_at(i + 1, j));
            let k_ym = 2.0 * k_at(i, j - 1) * k_c / (k_at(i, j - 1) + k_c);
            let k_yp = 2.0 * k_c * k_at(i, j + 1) / (k_c + k_at(i, j + 1));

            tb.add(k, idx(i - 1, j), k_xm / dx2);
            tb.add(k, idx(i + 1, j), k_xp / dx2);
            tb.add(k, idx(i, j - 1), k_ym / dy2);
            tb.add(k, idx(i, j + 1), k_yp / dy2);
            tb.add(k, k, -(k_xm + k_xp) / dx2 - (k_ym + k_yp) / dy2);
        }
    }

    (tb.build(), grid_x, grid_y)
}

/// Apply Dirichlet boundary conditions to 1D system.
/// bc_left, bc_right: boundary values. Modifies rhs in-place.
pub fn apply_bc_1d(rhs: &mut [f64], bc_left: f64, bc_right: f64) {
    let n = rhs.len();
    rhs[0] = bc_left;
    if n > 1 {
        rhs[n - 1] = bc_right;
    }
}

/// Apply Dirichlet boundary conditions to 2D system.
/// bc: (left, right, bottom, top) boundary values.
pub fn apply_bc_2d(rhs: &mut [f64], nx: usize, ny: usize, bc: (f64, f64, f64, f64)) {
    let (bc_left, bc_right, bc_bottom, bc_top) = bc;
    let idx = |i: usize, j: usize| -> usize { i * ny + j };

    for j in 0..ny {
        rhs[idx(0, j)] = bc_left;
        rhs[idx(nx - 1, j)] = bc_right;
    }
    for i in 0..nx {
        rhs[idx(i, 0)] = bc_bottom;
        rhs[idx(i, ny - 1)] = bc_top;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solvers::solve_cg;

    #[test]
    fn test_laplacian_1d_poisson() {
        // Solve -u'' = 1, u(0) = 0, u(1) = 0
        // Exact: u(x) = x(1-x)/2
        let n = 101;
        let (a, grid) = laplacian_1d(0.0, 1.0, n);
        let mut rhs = vec![1.0; n];
        // Negate because stencil gives d²u/dx², but equation is -d²u/dx² = f
        for i in 1..n - 1 {
            rhs[i] = -1.0; // because A encodes d²/dx², so Au = f means d²u/dx² = f; we want -d²u/dx² = 1
        }
        // Actually: A encodes [1,-2,1]/h², so Au = b means u'' ≈ b. For -u''=1, use b = -1.
        apply_bc_1d(&mut rhs, 0.0, 0.0);

        let result = solve_cg(&a, &rhs, 1e-12, 500, "none");
        assert!(result.converged);

        // Check against exact solution at midpoint
        let mid = n / 2;
        let x = grid[mid];
        let exact = x * (1.0 - x) / 2.0;
        assert!(
            (result.x[mid] - exact).abs() < 0.01,
            "1D Poisson: got {}, expected {}",
            result.x[mid],
            exact
        );
    }

    #[test]
    fn test_laplacian_2d_size() {
        let (a, gx, gy) = laplacian_2d(0.0, 1.0, 0.0, 1.0, 10, 10);
        assert_eq!(a.nrows, 100);
        assert_eq!(a.ncols, 100);
        assert_eq!(gx.len(), 10);
        assert_eq!(gy.len(), 10);
    }
}
