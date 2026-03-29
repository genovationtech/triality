"""Rust backend integration for Triality solvers.

This module provides a transparent bridge between Triality's Python solver pipeline
and the Rust numerical engine (triality_engine). When the Rust engine is available,
it accelerates:
  - Sparse matrix assembly (FDM stencils)
  - Linear solves (CG, GMRES, BiCGSTAB, Direct LU)
  - Preconditioning (Jacobi, ILU0, SSOR)
  - Time integration (Forward Euler, Backward Euler, Crank-Nicolson, RK4, BDF2)

When the Rust engine is NOT available, everything falls back to the existing
pure-Python implementation (NumPy + SciPy). No user code needs to change.

Usage:
    from triality.solvers.rust_backend import get_backend
    backend = get_backend()
    if backend.is_rust:
        print(f"Using Rust engine v{backend.version}")
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any

# ─── Try importing Rust engine ───────────────────────────────────────────

_rust_engine = None
_rust_available = False

try:
    import triality_engine as _rust_engine
    _rust_available = _rust_engine.is_available()
except ImportError:
    _rust_available = False


def rust_available() -> bool:
    """Check if the Rust backend is compiled and available."""
    return _rust_available


def rust_version() -> Optional[str]:
    """Get Rust engine version, or None if not available."""
    if _rust_available:
        return _rust_engine.version()
    return None


# ─── Unified Backend Interface ───────────────────────────────────────────

@dataclass
class LinearResult:
    """Linear solver result (matches triality.solvers.linear.LinearResult)."""
    x: np.ndarray
    converged: bool
    iterations: int
    residual: float


class RustBackend:
    """Backend that delegates to the Rust engine."""

    is_rust = True

    @property
    def version(self):
        return _rust_engine.version()

    def solve_linear(self, A, b, method='auto', precond='none', tol=1e-8, maxiter=None):
        """Solve Ax = b using Rust solvers."""
        if maxiter is None:
            maxiter = len(b) * 10

        # Convert SciPy sparse → Rust CSR
        from scipy import sparse as sp
        A_csr = sp.csr_matrix(A)
        rust_mat = _rust_engine.CsrMatrix.from_scipy(
            A_csr.shape[0], A_csr.shape[1],
            np.asarray(A_csr.indptr, dtype=np.int64),
            np.asarray(A_csr.indices, dtype=np.int64),
            np.asarray(A_csr.data, dtype=np.float64),
        )

        # Map preconditioner names
        precond_map = {'none': 'none', 'jacobi': 'jacobi', 'ilu': 'ilu0', 'ilu0': 'ilu0', 'ssor': 'ssor'}
        rust_precond = precond_map.get(precond, 'none')

        result = _rust_engine.solve_linear(
            rust_mat, np.asarray(b, dtype=np.float64),
            method=method, precond=rust_precond, tol=tol, max_iter=maxiter,
        )

        return LinearResult(
            x=np.asarray(result['x']),
            converged=result['converged'],
            iterations=result['iterations'],
            residual=result['residual'],
        )

    def assemble_laplacian_1d(self, a, b, n):
        """Assemble 1D Laplacian using Rust. Returns (scipy CSR, grid)."""
        rust_mat, grid = _rust_engine.assemble_laplacian_1d(a, b, n)
        # Convert back to SciPy for compatibility with existing code
        return self._rust_csr_to_scipy(rust_mat), np.asarray(grid)

    def assemble_laplacian_2d(self, x0, x1, y0, y1, nx, ny):
        """Assemble 2D Laplacian using Rust. Returns (scipy CSR, grid_x, grid_y)."""
        rust_mat, gx, gy = _rust_engine.assemble_laplacian_2d(x0, x1, y0, y1, nx, ny)
        return self._rust_csr_to_scipy(rust_mat), np.asarray(gx), np.asarray(gy)

    def apply_bc_1d(self, rhs, bc_left, bc_right):
        """Apply 1D Dirichlet BCs using Rust."""
        return np.asarray(_rust_engine.apply_bc_1d(
            np.asarray(rhs, dtype=np.float64), bc_left, bc_right
        ))

    def apply_bc_2d(self, rhs, nx, ny, bc_left, bc_right, bc_bottom, bc_top):
        """Apply 2D Dirichlet BCs using Rust."""
        return np.asarray(_rust_engine.apply_bc_2d(
            np.asarray(rhs, dtype=np.float64), nx, ny,
            bc_left, bc_right, bc_bottom, bc_top
        ))

    def timestep(self, method, A, u0, f_rhs, dt, n_steps, save_every=1,
                 solver='cg', precond='none'):
        """Run time integration using Rust."""
        from scipy import sparse as sp
        A_csr = sp.csr_matrix(A)
        rust_mat = _rust_engine.CsrMatrix.from_scipy(
            A_csr.shape[0], A_csr.shape[1],
            np.asarray(A_csr.indptr, dtype=np.int64),
            np.asarray(A_csr.indices, dtype=np.int64),
            np.asarray(A_csr.data, dtype=np.float64),
        )

        u0_arr = np.asarray(u0, dtype=np.float64)
        f_arr = np.asarray(f_rhs, dtype=np.float64)

        dispatch = {
            'forward_euler': lambda: _rust_engine.timestep_forward_euler(
                rust_mat, u0_arr, f_arr, dt, n_steps, save_every),
            'backward_euler': lambda: _rust_engine.timestep_backward_euler(
                rust_mat, u0_arr, f_arr, dt, n_steps, save_every, solver, precond),
            'crank_nicolson': lambda: _rust_engine.timestep_crank_nicolson(
                rust_mat, u0_arr, f_arr, dt, n_steps, save_every, solver, precond),
            'rk4': lambda: _rust_engine.timestep_rk4(
                rust_mat, u0_arr, f_arr, dt, n_steps, save_every),
            'bdf2': lambda: _rust_engine.timestep_bdf2(
                rust_mat, u0_arr, f_arr, dt, n_steps, save_every, solver, precond),
        }

        result = dispatch[method]()
        return {
            'snapshots': [np.asarray(s) for s in result['snapshots']],
            'times': np.asarray(result['times']),
            'steps': result['steps'],
        }


    def assemble_fem3d_poisson(self, mesh, source=0.0):
        """Assemble 3D FEM Poisson operator via Rust."""
        return _rust_engine.fem3d_assemble_poisson(mesh, source)

    def apply_fem3d_dirichlet(self, matrix, rhs, dofs, values):
        """Apply Dirichlet elimination for 3D FEM."""
        return _rust_engine.fem3d_apply_dirichlet(
            matrix,
            np.asarray(rhs, dtype=np.float64),
            [int(i) for i in dofs],
            [float(v) for v in values],
        )

    def export_fem3d_vtu(self, mesh, values, field_name='u'):
        """Export nodal field to VTU (string payload)."""
        return _rust_engine.fem3d_export_vtu(mesh, field_name, np.asarray(values, dtype=np.float64))

    def _rust_csr_to_scipy(self, rust_mat):
        """Convert Rust CsrMatrix back to SciPy (for downstream compatibility)."""
        from scipy import sparse as sp
        # We can use spmv with identity-like vectors to extract, but simpler
        # to just re-assemble from the Rust matrix's Python-accessible properties.
        # For now, return the Rust matrix wrapped in a thin scipy-compatible adapter.
        # This is a placeholder — the real pipeline will use Rust end-to-end.
        return _RustCsrAdapter(rust_mat)


class _RustCsrAdapter:
    """Thin adapter making a Rust CsrMatrix behave enough like scipy.sparse.csr_matrix
    for use in Triality's existing pipeline (supports @ operator and .shape)."""

    def __init__(self, rust_mat):
        self._mat = rust_mat
        self.shape = (rust_mat.nrows, rust_mat.ncols)

    def __matmul__(self, x):
        return np.asarray(self._mat.spmv(np.asarray(x, dtype=np.float64)))

    @property
    def nnz(self):
        return self._mat.nnz

    def diagonal(self):
        return np.asarray(self._mat.diagonal())

    def dot(self, x):
        return self.__matmul__(x)

    def tocsr(self):
        return self

    def toarray(self):
        n = self.shape[0]
        dense = np.zeros(self.shape)
        for i in range(n):
            e = np.zeros(self.shape[1])
            e[i] = 1.0
            dense[:, i] = self @ e
        return dense


class PythonBackend:
    """Fallback backend using existing NumPy/SciPy implementation."""

    is_rust = False
    version = None

    def solve_linear(self, A, b, method='auto', precond='none', tol=1e-8, maxiter=None):
        """Solve using existing SciPy implementation."""
        from triality.solvers.linear import solve_linear as _scipy_solve
        result = _scipy_solve(A, b, method=method, precond=precond, tol=tol, maxiter=maxiter)
        return LinearResult(
            x=result.x,
            converged=result.converged,
            iterations=result.iterations,
            residual=result.residual,
        )

    def assemble_laplacian_1d(self, a, b, n):
        """Use Python FDM assembly (existing code path)."""
        return None  # Signal caller to use existing Python path

    def assemble_laplacian_2d(self, x0, x1, y0, y1, nx, ny):
        """Use Python FDM assembly (existing code path)."""
        return None  # Signal caller to use existing Python path

    def apply_bc_1d(self, rhs, bc_left, bc_right):
        rhs = rhs.copy()
        rhs[0] = bc_left
        rhs[-1] = bc_right
        return rhs

    def apply_bc_2d(self, rhs, nx, ny, bc_left, bc_right, bc_bottom, bc_top):
        rhs = rhs.copy()
        idx = lambda i, j: i * ny + j
        for j in range(ny):
            rhs[idx(0, j)] = bc_left
            rhs[idx(nx - 1, j)] = bc_right
        for i in range(nx):
            rhs[idx(i, 0)] = bc_bottom
            rhs[idx(i, ny - 1)] = bc_top
        return rhs


    def assemble_fem3d_poisson(self, mesh, source=0.0):
        return None

    def apply_fem3d_dirichlet(self, matrix, rhs, dofs, values):
        return None

    def export_fem3d_vtu(self, mesh, values, field_name='u'):
        return None
    def timestep(self, method, A, u0, f_rhs, dt, n_steps, save_every=1,
                 solver='cg', precond='none'):
        """Pure Python time stepping (basic Forward Euler fallback)."""
        from triality.solvers.linear import solve_linear as _scipy_solve
        from scipy import sparse as sp

        u = np.array(u0, dtype=np.float64)
        snapshots = [u.copy()]
        times = [0.0]

        if method == 'forward_euler':
            for step in range(1, n_steps + 1):
                au = A @ u
                u = u + dt * (au + f_rhs)
                if step % save_every == 0 or step == n_steps:
                    snapshots.append(u.copy())
                    times.append(step * dt)
        elif method in ('backward_euler', 'crank_nicolson', 'bdf2'):
            n = len(u0)
            I = sp.eye(n, format='csr')
            if method == 'backward_euler':
                lhs = I - dt * A
            elif method == 'crank_nicolson':
                lhs = I - (dt / 2.0) * A
            else:
                lhs = (3.0 / (2.0 * dt)) * I - A

            for step in range(1, n_steps + 1):
                if method == 'backward_euler':
                    rhs_vec = u + dt * f_rhs
                elif method == 'crank_nicolson':
                    rhs_vec = u + (dt / 2.0) * (A @ u) + dt * f_rhs
                else:  # bdf2
                    if step == 1:
                        lhs_be = I - dt * A
                        rhs_vec = u + dt * f_rhs
                        result = _scipy_solve(lhs_be, rhs_vec, method=solver)
                        u_prev = u.copy()
                        u = result.x
                        if save_every == 1:
                            snapshots.append(u.copy())
                            times.append(dt)
                        continue
                    rhs_vec = (2.0 / dt) * u - (1.0 / (2.0 * dt)) * u_prev + f_rhs
                    u_prev = u.copy()

                result = _scipy_solve(lhs, rhs_vec, method=solver)
                u = result.x
                if step % save_every == 0 or step == n_steps:
                    snapshots.append(u.copy())
                    times.append(step * dt)
        elif method == 'rk4':
            for step in range(1, n_steps + 1):
                k1 = A @ u + f_rhs
                k2 = A @ (u + 0.5 * dt * k1) + f_rhs
                k3 = A @ (u + 0.5 * dt * k2) + f_rhs
                k4 = A @ (u + dt * k3) + f_rhs
                u = u + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
                if step % save_every == 0 or step == n_steps:
                    snapshots.append(u.copy())
                    times.append(step * dt)

        return {
            'snapshots': snapshots,
            'times': np.array(times),
            'steps': n_steps,
        }


# ─── Singleton accessor ─────────────────────────────────────────────────

_backend = None


def get_backend():
    """Get the best available backend (Rust if compiled, else Python)."""
    global _backend
    if _backend is None:
        if _rust_available:
            _backend = RustBackend()
        else:
            _backend = PythonBackend()
    return _backend
