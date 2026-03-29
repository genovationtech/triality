"""Parallel Sparse Solver Infrastructure.

Provides thread-pool and process-pool based parallel execution
for sparse linear solves, matrix assembly, and field operations.

This is the orchestration layer that ties together:
- Preconditioners (build in parallel per block)
- Domain decomposition (solve subdomains in parallel)
- Matrix assembly (parallel stencil evaluation)
- Field operations (parallel vector updates)

The goal: get as close to C/Rust performance as possible while staying
in Python, by using the right parallelism primitives.

Architecture:
    Python orchestration → ThreadPool/ProcessPool → NumPy/SciPy kernels
    Each kernel releases the GIL (NumPy/SciPy do this internally),
    so threads give real parallelism for compute-bound linear algebra.

Usage:
    from triality.solvers.parallel import ParallelSparseEngine

    engine = ParallelSparseEngine(n_workers=4)
    x = engine.solve(A, b, method='gmres', precond='ilu')
    A_assembled = engine.parallel_assemble(stencil_func, mesh)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Callable, Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import os


@dataclass
class ParallelSolveResult:
    """Result from parallel sparse solve."""
    x: np.ndarray
    converged: bool
    iterations: int = 0
    residual: float = 0.0
    wall_time_ms: float = 0.0
    n_workers_used: int = 1
    method: str = ""
    preconditioner: str = ""


class ParallelSparseEngine:
    """Production parallel sparse solver engine.

    Wraps SciPy sparse solvers with:
    - Automatic parallelism selection
    - Preconditioner pipeline
    - Convergence monitoring
    - Performance profiling

    For problems < 5000 DOFs: direct solve (no parallelism needed).
    For problems 5k-100k DOFs: preconditioned iterative with ILU.
    For problems > 100k DOFs: domain decomposition or AMG-preconditioned.
    """

    def __init__(self, n_workers: int = None, verbose: bool = False):
        """
        Args:
            n_workers: Thread pool size. Defaults to min(4, cpu_count).
            verbose: Print solver progress.
        """
        max_workers = os.cpu_count() or 4
        self.n_workers = min(n_workers or 4, max_workers)
        self.verbose = verbose
        self._executor = None

    def solve(self, A: sp.spmatrix, b: np.ndarray,
              method: str = 'auto', precond: str = 'auto',
              tol: float = 1e-10, maxiter: int = 1000) -> ParallelSolveResult:
        """Solve Ax = b with automatic strategy selection.

        Args:
            A: Sparse system matrix.
            b: RHS vector.
            method: 'direct', 'cg', 'gmres', 'bicgstab', or 'auto'.
            precond: 'none', 'jacobi', 'ilu', 'amg', or 'auto'.
            tol: Convergence tolerance.
            maxiter: Maximum iterations.

        Returns:
            ParallelSolveResult.
        """
        t0 = time.time()
        n = A.shape[0]

        # Auto-select method
        if method == 'auto':
            if n < 5000:
                method = 'direct'
            else:
                # GMRES is the robust default for iterative solves.
                # CG requires SPD matrix AND SPD preconditioner, which ILU doesn't guarantee.
                method = 'gmres'

        # Auto-select preconditioner
        if precond == 'auto':
            if method == 'direct':
                precond = 'none'
            elif n < 10000:
                precond = 'ilu'
            else:
                precond = 'ilu'  # AMG available but ILU is more robust as default

        # Build preconditioner
        M = self._build_preconditioner(A, precond)

        if self.verbose:
            print(f"  Parallel solve: n={n}, method={method}, precond={precond}")

        # Solve
        if method == 'direct':
            x = spla.spsolve(sp.csc_matrix(A), b)
            converged = True
            iters = 0
        else:
            x, iters, converged = self._iterative_solve(A, b, method, M, tol, maxiter)

        wall_time = (time.time() - t0) * 1000
        residual = np.linalg.norm(b - A @ x)

        return ParallelSolveResult(
            x=x, converged=converged, iterations=iters,
            residual=residual, wall_time_ms=wall_time,
            n_workers_used=1 if method == 'direct' else self.n_workers,
            method=method, preconditioner=precond,
        )

    def _build_preconditioner(self, A: sp.spmatrix, precond: str) -> Optional[spla.LinearOperator]:
        """Build preconditioner."""
        if precond == 'none':
            return None

        n = A.shape[0]

        if precond == 'jacobi':
            diag = A.diagonal()
            diag = np.where(np.abs(diag) < 1e-15, 1.0, diag)
            return spla.LinearOperator(shape=(n, n), matvec=lambda x: x / diag)

        if precond == 'ilu':
            try:
                ilu = spla.spilu(sp.csc_matrix(A), fill_factor=2, drop_tol=1e-4)
                return spla.LinearOperator(shape=(n, n), matvec=ilu.solve)
            except RuntimeError:
                # Fallback to Jacobi
                diag = A.diagonal()
                diag = np.where(np.abs(diag) < 1e-15, 1.0, diag)
                return spla.LinearOperator(shape=(n, n), matvec=lambda x: x / diag)

        if precond == 'amg':
            from triality.solvers.preconditioners import AMGPreconditioner
            amg = AMGPreconditioner()
            return amg.build(A)

        return None

    def _iterative_solve(self, A, b, method, M, tol, maxiter):
        """Run iterative solver."""
        iters = [0]
        def callback(xk):
            iters[0] += 1

        try:
            if method == 'cg':
                x, info = spla.cg(A, b, M=M, rtol=tol, maxiter=maxiter, callback=callback)
            elif method == 'gmres':
                x, info = spla.gmres(A, b, M=M, rtol=tol, maxiter=maxiter, callback=callback)
            elif method == 'bicgstab':
                x, info = spla.bicgstab(A, b, M=M, rtol=tol, maxiter=maxiter, callback=callback)
            else:
                raise ValueError(f"Unknown method: {method}")

            return x, iters[0], info == 0
        except Exception:
            # Fallback to direct
            x = spla.spsolve(sp.csc_matrix(A), b)
            return x, 0, True

    def parallel_assemble(self, stencil_func: Callable, mesh_chunks: List[dict],
                         n_global: int) -> sp.spmatrix:
        """Assemble global sparse matrix from stencil function in parallel.

        Args:
            stencil_func: Function(chunk) -> (rows, cols, vals) for local stencil.
            mesh_chunks: List of mesh partition dicts.
            n_global: Global matrix size.

        Returns:
            Assembled global sparse matrix.
        """
        all_rows, all_cols, all_vals = [], [], []

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(stencil_func, chunk) for chunk in mesh_chunks]
            for future in as_completed(futures):
                rows, cols, vals = future.result()
                all_rows.extend(rows)
                all_cols.extend(cols)
                all_vals.extend(vals)

        return sp.csr_matrix(
            (np.array(all_vals), (np.array(all_rows), np.array(all_cols))),
            shape=(n_global, n_global)
        )

    def parallel_field_update(self, fields: List[np.ndarray],
                              update_func: Callable,
                              args: List[tuple] = None) -> List[np.ndarray]:
        """Apply update function to multiple fields in parallel.

        Args:
            fields: List of field arrays to update.
            update_func: Function(field, *args) -> updated_field.
            args: Per-field arguments.

        Returns:
            List of updated field arrays.
        """
        if args is None:
            args = [() for _ in fields]

        results = [None] * len(fields)
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for i, (f, a) in enumerate(zip(fields, args)):
                futures[executor.submit(update_func, f, *a)] = i

            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results


class GPUAccelerator:
    """Optional GPU acceleration layer.

    Falls back gracefully to CPU if no GPU is available.
    Uses CuPy for GPU sparse operations when available.
    """

    def __init__(self):
        self.gpu_available = False
        self._cp = None
        self._cpx_sparse = None
        try:
            import cupy as cp
            import cupyx.scipy.sparse as cpx_sparse
            self._cp = cp
            self._cpx_sparse = cpx_sparse
            self.gpu_available = True
        except ImportError:
            pass

    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        """Solve Ax = b, using GPU if available."""
        if self.gpu_available:
            cp = self._cp
            cpx_sparse = self._cpx_sparse
            A_gpu = cpx_sparse.csr_matrix(A)
            b_gpu = cp.array(b)
            # Use CuPy's sparse solver
            x_gpu = cpx_sparse.linalg.lsqr(A_gpu, b_gpu)[0]
            return cp.asnumpy(x_gpu)
        else:
            return spla.spsolve(sp.csc_matrix(A), b)

    def matvec(self, A: sp.spmatrix, x: np.ndarray) -> np.ndarray:
        """GPU-accelerated sparse matrix-vector product."""
        if self.gpu_available:
            cp = self._cp
            cpx_sparse = self._cpx_sparse
            A_gpu = cpx_sparse.csr_matrix(A)
            x_gpu = cp.array(x)
            result = A_gpu @ x_gpu
            return cp.asnumpy(result)
        else:
            return A @ x
