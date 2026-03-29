"""Tests for the Rust backend integration.

These tests run against whichever backend is available:
- If triality_engine is compiled: tests validate Rust solvers
- If not: tests validate the Python fallback path

Either way, the solver pipeline must produce correct results.
"""

import numpy as np
import pytest
from scipy import sparse

from triality.solvers.rust_backend import get_backend, rust_available, rust_version
from triality.solvers.linear import solve_linear, LinearResult


class TestBackendDetection:
    """Test backend availability detection."""

    def test_get_backend_returns_object(self):
        backend = get_backend()
        assert hasattr(backend, 'is_rust')
        assert hasattr(backend, 'solve_linear')

    def test_rust_available_returns_bool(self):
        assert isinstance(rust_available(), bool)

    def test_version_consistent(self):
        if rust_available():
            v = rust_version()
            assert v is not None
            assert isinstance(v, str)
        else:
            assert rust_version() is None


class TestLinearSolvers:
    """Test linear solvers through the backend abstraction."""

    def _make_poisson_1d(self, n=50):
        """Build 1D Poisson system: -u'' = 1, u(0)=u(n-1)=0."""
        h = 1.0 / (n - 1)
        diag = np.full(n, 2.0 / h**2)
        off = np.full(n - 1, -1.0 / h**2)
        A = sparse.diags([off, diag, off], [-1, 0, 1], format='csr')
        b = np.ones(n)
        # Apply BC
        A_dense = A.toarray()
        A_dense[0, :] = 0; A_dense[0, 0] = 1; b[0] = 0
        A_dense[-1, :] = 0; A_dense[-1, -1] = 1; b[-1] = 0
        A = sparse.csr_matrix(A_dense)
        return A, b

    def _make_poisson_2d(self, n=20):
        """Build 2D Poisson system on unit square."""
        N = n * n
        h = 1.0 / (n - 1)
        h2 = h * h

        rows, cols, vals = [], [], []
        b = np.zeros(N)

        def idx(i, j):
            return i * n + j

        for i in range(n):
            for j in range(n):
                k = idx(i, j)
                if i == 0 or i == n-1 or j == 0 or j == n-1:
                    rows.append(k); cols.append(k); vals.append(1.0)
                    b[k] = 0.0
                else:
                    rows.append(k); cols.append(k); vals.append(-4.0/h2)
                    rows.append(k); cols.append(idx(i-1,j)); vals.append(1.0/h2)
                    rows.append(k); cols.append(idx(i+1,j)); vals.append(1.0/h2)
                    rows.append(k); cols.append(idx(i,j-1)); vals.append(1.0/h2)
                    rows.append(k); cols.append(idx(i,j+1)); vals.append(1.0/h2)
                    b[k] = 1.0

        A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
        return A, b

    def test_direct_solve_1d(self):
        A, b = self._make_poisson_1d()
        result = solve_linear(A, b, method='direct')
        assert result.converged
        assert result.residual < 1e-8

    def test_cg_solve_1d(self):
        A, b = self._make_poisson_1d()
        result = solve_linear(A, b, method='cg')
        assert result.converged
        assert result.residual < 1e-6

    def test_gmres_solve_1d(self):
        A, b = self._make_poisson_1d()
        result = solve_linear(A, b, method='gmres')
        assert result.converged
        assert result.residual < 1e-6

    def test_bicgstab_solve_1d(self):
        A, b = self._make_poisson_1d()
        result = solve_linear(A, b, method='bicgstab')
        assert result.converged
        assert result.residual < 1e-6

    def test_auto_solve_1d(self):
        A, b = self._make_poisson_1d()
        result = solve_linear(A, b, method='auto')
        assert result.converged

    def test_solve_2d(self):
        A, b = self._make_poisson_2d(n=15)
        result = solve_linear(A, b, method='gmres', precond='jacobi')
        assert result.converged
        assert result.residual < 1e-4

    def test_solution_accuracy_1d(self):
        """Check solution matches exact solution of -u'' = 1."""
        n = 101
        A, b = self._make_poisson_1d(n)
        result = solve_linear(A, b, method='direct')

        # Exact: u(x) = x(1-x)/2
        x = np.linspace(0, 1, n)
        exact = x * (1 - x) / 2.0

        error = np.max(np.abs(result.x - exact))
        assert error < 0.01, f"Solution error too large: {error}"


class TestTimeSteppers:
    """Test time integration through the backend."""

    def _make_heat_1d(self, n=31):
        """Build 1D heat equation operator: du/dt = d²u/dx²."""
        h = 1.0 / (n - 1)
        diag = np.full(n, -2.0 / h**2)
        off = np.full(n - 1, 1.0 / h**2)
        A = sparse.diags([off, diag, off], [-1, 0, 1], format='csr')
        # Zero out boundary rows (Dirichlet u=0)
        A_dense = A.toarray()
        A_dense[0, :] = 0
        A_dense[-1, :] = 0
        return sparse.csr_matrix(A_dense)

    def test_forward_euler(self):
        n = 31
        A = self._make_heat_1d(n)
        x = np.linspace(0, 1, n)
        u0 = np.sin(np.pi * x)
        f_rhs = np.zeros(n)

        backend = get_backend()
        h = 1.0 / (n - 1)
        dt = 0.4 * h**2  # CFL safe

        result = backend.timestep('forward_euler', A, u0, f_rhs, dt, 50, save_every=50)
        assert len(result['snapshots']) == 2
        # Solution should decay, not blow up
        assert np.max(np.abs(result['snapshots'][-1])) < 1.5

    def test_backward_euler(self):
        n = 21
        A = self._make_heat_1d(n)
        x = np.linspace(0, 1, n)
        u0 = np.sin(np.pi * x)
        f_rhs = np.zeros(n)

        backend = get_backend()
        result = backend.timestep('backward_euler', A, u0, f_rhs, 0.01, 10,
                                  save_every=10, solver='gmres')
        u_final = result['snapshots'][-1]
        # Should decay
        assert np.max(np.abs(u_final)) < 1.0
        assert np.max(np.abs(u_final)) > 0.0

    def test_rk4(self):
        n = 31
        A = self._make_heat_1d(n)
        x = np.linspace(0, 1, n)
        u0 = np.sin(np.pi * x)
        f_rhs = np.zeros(n)

        backend = get_backend()
        h = 1.0 / (n - 1)
        dt = 0.4 * h**2

        result = backend.timestep('rk4', A, u0, f_rhs, dt, 20, save_every=20)
        assert np.max(np.abs(result['snapshots'][-1])) < 1.5


class TestExistingPipeline:
    """Verify that the existing Triality solve() pipeline still works."""

    def test_solve_1d_poisson(self):
        """Existing API should work regardless of backend."""
        from triality import Field, laplacian, Eq, Interval, solve

        u = Field("u")
        sol = solve(
            Eq(laplacian(u), 1),
            Interval(0, 1),
            bc={'left': 0, 'right': 0},
            resolution=50,
            verbose=False,
        )
        assert sol.converged
        # Check midpoint: laplacian(u) == 1 means u'' = 1, so u = x(1-x)/2 * (-1) = -x(1-x)/2
        mid_val = sol(0.5)
        expected = -0.125  # -x(1-x)/2 at x=0.5
        assert abs(mid_val - expected) < 0.01

    def test_solve_2d_laplace(self):
        """2D Laplace equation through existing API."""
        from triality import Field, laplacian, Eq, Square, solve

        u = Field("u")
        sol = solve(
            Eq(laplacian(u), 0),
            Square(1.0),
            bc={'left': 0, 'right': 1, 'top': 0, 'bottom': 0},
            resolution=20,
            verbose=False,
        )
        assert sol.converged


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
