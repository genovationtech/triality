"""Linear system solvers"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from dataclasses import dataclass


@dataclass
class LinearResult:
    """Linear solver result"""
    x: np.ndarray
    converged: bool
    iterations: int
    residual: float

    def __repr__(self):
        status = "converged" if self.converged else "NOT CONVERGED"
        return f"LinearResult({status}, iter={self.iterations}, res={self.residual:.2e})"


def solve_linear(A, b, method='direct', precond='none', tol=1e-8, maxiter=None, verbose=False):
    """
    Solve linear system Ax = b with numerical validation.

    Automatically uses the Rust backend (triality_engine) when available for
    iterative solvers (CG, GMRES, BiCGSTAB). Falls back to SciPy otherwise.
    """
    # Try Rust backend for iterative solvers
    if method in ('cg', 'gmres', 'bicgstab', 'auto') and not verbose:
        try:
            from triality.solvers.rust_backend import get_backend
            backend = get_backend()
            if backend.is_rust:
                result = backend.solve_linear(A, b, method=method, precond=precond,
                                              tol=tol, maxiter=maxiter)
                return LinearResult(
                    x=result.x, converged=result.converged,
                    iterations=result.iterations, residual=result.residual,
                )
        except Exception:
            pass  # Fall through to SciPy

    return _solve_linear_scipy(A, b, method=method, precond=precond,
                                tol=tol, maxiter=maxiter, verbose=verbose)


def _solve_linear_scipy(A, b, method='direct', precond='none', tol=1e-8, maxiter=None, verbose=False):
    """
    Solve linear system Ax = b with numerical validation.

    Args:
        A: Sparse matrix
        b: RHS vector
        method: 'direct', 'cg', 'gmres', 'bicgstab', 'auto'
        precond: 'none', 'jacobi', 'ilu'
        tol: Convergence tolerance
        maxiter: Max iterations (default: 10 * len(b))
        verbose: Print solver diagnostics

    Returns:
        LinearResult with solution and convergence info
    """

    if maxiter is None:
        maxiter = len(b) * 10

    # Numerical validation of inputs
    if not np.all(np.isfinite(b)):
        raise ValueError("RHS vector contains NaN or Inf")

    if not np.all(np.isfinite(A.data)):
        raise ValueError("Matrix contains NaN or Inf")

    # Direct solver
    if method == 'direct':
        try:
            x = spla.spsolve(A, b)
            residual = np.linalg.norm(A @ x - b)
            return LinearResult(x=x, converged=True, iterations=1, residual=residual)
        except:
            # Fallback to dense for small problems
            if A.shape[0] < 1000:
                x = np.linalg.solve(A.toarray(), b)
                residual = np.linalg.norm(A @ x - b)
                return LinearResult(x=x, converged=True, iterations=1, residual=residual)
            raise

    # Build preconditioner
    M = _build_precond(A, precond)

    # Iteration counter
    iter_count = [0]
    def callback(*args):
        iter_count[0] += 1

    # Auto method selection: use GMRES for variable-coefficient systems
    # CG requires SPD, which isn't guaranteed with cost fields and BCs
    if method == 'auto':
        method = 'gmres'  # GMRES is more robust for general systems

    # Iterative solvers (handle both old and new SciPy versions)
    # SciPy 1.11+ uses rtol/atol instead of tol
    import scipy
    scipy_version = tuple(map(int, scipy.__version__.split('.')[:2]))

    # Suppress RuntimeWarnings during iterative solve and catch them
    import warnings
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always", RuntimeWarning)

        if method == 'cg':
            # CG requires SPD matrix - warn if used
            if verbose:
                print("  WARNING: Using CG solver - matrix must be SPD")
            if scipy_version >= (1, 11):
                x, info = spla.cg(A, b, M=M, rtol=tol, atol=0, maxiter=maxiter, callback=callback)
            else:
                x, info = spla.cg(A, b, M=M, tol=tol, maxiter=maxiter, callback=callback)
        elif method == 'gmres':
            if scipy_version >= (1, 11):
                x, info = spla.gmres(A, b, M=M, rtol=tol, atol=0, maxiter=maxiter, callback=callback, callback_type='legacy')
            else:
                x, info = spla.gmres(A, b, M=M, tol=tol, maxiter=maxiter, callback=callback, callback_type='legacy')
        elif method == 'bicgstab':
            if scipy_version >= (1, 11):
                x, info = spla.bicgstab(A, b, M=M, rtol=tol, atol=0, maxiter=maxiter, callback=callback)
            else:
                x, info = spla.bicgstab(A, b, M=M, tol=tol, maxiter=maxiter, callback=callback)
        else:
            raise ValueError(f"Unknown method: {method}")

    # Check for numerical warnings during solve
    numerical_warnings = [w for w in warning_list if issubclass(w.category, RuntimeWarning)]
    if numerical_warnings and verbose:
        print(f"  ⚠️ Numerical warnings during {method} solve:")
        for w in numerical_warnings:
            print(f"    {w.category.__name__}: {w.message}")

    # Validate solution
    if not np.all(np.isfinite(x)):
        if verbose:
            print(f"  ⚠️ Solver returned NaN/Inf - trying fallback")
        # Fallback to direct solver
        try:
            x_direct = spla.spsolve(A, b)
            if np.all(np.isfinite(x_direct)):
                x = x_direct
                info = 0
                iter_count[0] = 1
                if verbose:
                    print(f"  ✓ Direct solver recovered valid solution")
            else:
                raise ValueError("Both iterative and direct solvers failed")
        except Exception as e:
            raise ValueError(f"Numerical failure in linear solve: {e}")

    converged = (info == 0)
    residual = np.linalg.norm(A @ x - b)

    # Numerical health check
    if numerical_warnings:
        # Had warnings but still got valid result - downgrade convergence status
        converged = False

    if verbose:
        status = "✓" if converged else "⚠"
        print(f"  {status} Solver: {method}, iter={iter_count[0]}, res={residual:.2e}")

    return LinearResult(
        x=x,
        converged=converged,
        iterations=iter_count[0],
        residual=residual
    )


def _build_precond(A, precond_type):
    """Build preconditioner"""
    if precond_type == 'none':
        return None

    if precond_type == 'jacobi':
        diag = A.diagonal()
        diag[diag == 0] = 1.0
        M = sparse.diags(1.0 / diag)
        return spla.LinearOperator(A.shape, matvec=M.dot)

    if precond_type == 'ilu':
        try:
            ilu = spla.spilu(A.tocsc())
            return spla.LinearOperator(A.shape, matvec=ilu.solve)
        except:
            return _build_precond(A, 'jacobi')  # Fallback

    return None
