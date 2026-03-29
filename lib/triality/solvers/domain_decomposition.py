"""Domain Decomposition Methods for Parallel PDE Solves.

Implements Schwarz-type domain decomposition for splitting large PDE problems
into smaller, independently solvable subdomains that can run in parallel.

This is the bridge between "single-core prototype" and "production HPC."

Methods:
- Additive Schwarz (parallel): All subdomains solved simultaneously.
  Fast but may need more iterations.
- Multiplicative Schwarz (sequential): Subdomains solved in order,
  each using latest data from neighbors. Fewer iterations but serial.
- Restricted Additive Schwarz (RAS): Like additive, but restricts
  updates to owned cells only. Best parallel efficiency.

Usage:
    from triality.solvers.domain_decomposition import SchwartzDD

    dd = SchwartzDD(n_subdomains=4, overlap=2)
    dd.decompose(global_mesh)
    x = dd.solve(A_global, b_global, method='additive')
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class Subdomain:
    """A single subdomain in the decomposition."""
    id: int
    global_indices: np.ndarray  # Indices into global system
    owned_indices: np.ndarray   # Non-overlapping owned indices
    overlap_left: int = 0
    overlap_right: int = 0
    A_local: sp.spmatrix = None
    rhs_local: np.ndarray = None


@dataclass
class DDResult:
    """Result from domain decomposition solve."""
    x: np.ndarray
    converged: bool
    iterations: int
    residual_history: List[float] = field(default_factory=list)
    subdomain_solve_times: List[float] = field(default_factory=list)
    method: str = ""


class SchwartzDD:
    """Schwarz Domain Decomposition solver.

    Decomposes a 1D (or block-structured) system into overlapping subdomains,
    solves each locally, and iterates to global convergence.

    The overlap region is where subdomains share DOFs. More overlap = faster
    convergence but more redundant work.
    """

    def __init__(self, n_subdomains: int = 4, overlap: int = 2,
                 max_iter: int = 100, tol: float = 1e-8,
                 n_workers: int = None):
        """
        Args:
            n_subdomains: Number of subdomains.
            overlap: Number of overlapping cells between adjacent subdomains.
            max_iter: Maximum DD iterations.
            tol: Convergence tolerance (relative residual).
            n_workers: Thread pool size for parallel solves (None = n_subdomains).
        """
        self.n_subdomains = n_subdomains
        self.overlap = overlap
        self.max_iter = max_iter
        self.tol = tol
        self.n_workers = n_workers or n_subdomains
        self.subdomains: List[Subdomain] = []

    def decompose(self, n_global: int) -> List[Subdomain]:
        """Create overlapping subdomain decomposition for n_global DOFs.

        Splits [0, n_global) into n_subdomains with 'overlap' cells of overlap.
        """
        base_size = n_global // self.n_subdomains
        remainder = n_global % self.n_subdomains

        self.subdomains = []
        start = 0

        for i in range(self.n_subdomains):
            # Base partition (non-overlapping)
            owned_size = base_size + (1 if i < remainder else 0)
            owned_start = start
            owned_end = start + owned_size

            # Extended partition (with overlap)
            ext_start = max(0, owned_start - self.overlap)
            ext_end = min(n_global, owned_end + self.overlap)

            global_idx = np.arange(ext_start, ext_end)
            owned_idx = np.arange(owned_start, owned_end)

            sd = Subdomain(
                id=i,
                global_indices=global_idx,
                owned_indices=owned_idx,
                overlap_left=owned_start - ext_start,
                overlap_right=ext_end - owned_end,
            )
            self.subdomains.append(sd)
            start = owned_end

        return self.subdomains

    def _extract_local(self, A: sp.spmatrix, b: np.ndarray, sd: Subdomain):
        """Extract local system from global matrix."""
        idx = sd.global_indices
        sd.A_local = A[np.ix_(idx, idx)]
        sd.rhs_local = b[idx].copy()

    def _solve_local(self, sd: Subdomain, x_global: np.ndarray) -> np.ndarray:
        """Solve local subdomain problem with Dirichlet BCs from neighbors."""
        import time
        t0 = time.time()

        A_loc = sd.A_local.copy()
        b_loc = sd.rhs_local.copy()

        # Set boundary values from global solution at overlap boundaries
        if sd.overlap_left > 0:
            b_loc[0] = x_global[sd.global_indices[0]]
            A_loc = sp.lil_matrix(A_loc)
            A_loc[0, :] = 0
            A_loc[0, 0] = 1.0
            A_loc = sp.csr_matrix(A_loc)

        if sd.overlap_right > 0:
            b_loc[-1] = x_global[sd.global_indices[-1]]
            A_loc = sp.lil_matrix(A_loc)
            A_loc[-1, :] = 0
            A_loc[-1, -1] = 1.0
            A_loc = sp.csr_matrix(A_loc)

        x_local = spla.spsolve(sp.csc_matrix(A_loc), b_loc)

        solve_time = time.time() - t0
        return x_local, solve_time

    def solve(self, A: sp.spmatrix, b: np.ndarray,
              method: str = 'additive',
              x0: np.ndarray = None) -> DDResult:
        """Solve Ax = b using Schwarz domain decomposition.

        Args:
            A: Global sparse system matrix.
            b: Global RHS vector.
            method: 'additive' (parallel), 'multiplicative' (sequential),
                    or 'ras' (restricted additive Schwarz).
            x0: Initial guess (zeros if None).

        Returns:
            DDResult with solution and diagnostics.
        """
        n = A.shape[0]
        if x0 is None:
            x = np.zeros(n)
        else:
            x = x0.copy()

        # Decompose if not done
        if not self.subdomains:
            self.decompose(n)

        # Extract local systems
        for sd in self.subdomains:
            self._extract_local(A, b, sd)

        b_norm = np.linalg.norm(b)
        if b_norm < 1e-15:
            b_norm = 1.0

        history = []
        all_solve_times = []

        for iteration in range(self.max_iter):
            # Check convergence
            r = b - A @ x
            res_norm = np.linalg.norm(r) / b_norm
            history.append(res_norm)

            if res_norm < self.tol:
                return DDResult(
                    x=x, converged=True, iterations=iteration,
                    residual_history=history,
                    subdomain_solve_times=all_solve_times,
                    method=method
                )

            if method == 'additive':
                x = self._additive_step(A, b, x, all_solve_times)
            elif method == 'multiplicative':
                x = self._multiplicative_step(A, b, x, all_solve_times)
            elif method == 'ras':
                x = self._ras_step(A, b, x, all_solve_times)
            else:
                raise ValueError(f"Unknown DD method: {method}")

        # Final residual
        r = b - A @ x
        history.append(np.linalg.norm(r) / b_norm)

        return DDResult(
            x=x, converged=history[-1] < self.tol,
            iterations=self.max_iter,
            residual_history=history,
            subdomain_solve_times=all_solve_times,
            method=method
        )

    def _additive_step(self, A, b, x, solve_times):
        """Additive Schwarz: solve all subdomains in parallel, then combine."""
        corrections = {}

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for sd in self.subdomains:
                futures[executor.submit(self._solve_local, sd, x)] = sd

            for future in as_completed(futures):
                sd = futures[future]
                x_local, solve_time = future.result()
                corrections[sd.id] = (sd, x_local)
                solve_times.append(solve_time)

        # Weighted average of subdomain solutions
        x_new = np.zeros(len(x))
        weight = np.zeros(len(x))

        for sd_id, (sd, x_local) in corrections.items():
            for i, gi in enumerate(sd.global_indices):
                x_new[gi] += x_local[i]
                weight[gi] += 1.0

        # Average in overlap regions
        weight = np.maximum(weight, 1.0)
        x_new /= weight
        return x_new

    def _multiplicative_step(self, A, b, x, solve_times):
        """Multiplicative Schwarz: solve subdomains sequentially."""
        for sd in self.subdomains:
            # Update local RHS with latest global solution
            sd.rhs_local = b[sd.global_indices].copy()
            x_local, solve_time = self._solve_local(sd, x)
            solve_times.append(solve_time)

            # Update global solution immediately (sequential)
            for i, gi in enumerate(sd.global_indices):
                x[gi] = x_local[i]

        return x

    def _ras_step(self, A, b, x, solve_times):
        """Restricted Additive Schwarz: parallel solve, update only owned DOFs."""
        corrections = {}

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}
            for sd in self.subdomains:
                futures[executor.submit(self._solve_local, sd, x)] = sd

            for future in as_completed(futures):
                sd = futures[future]
                x_local, solve_time = future.result()
                corrections[sd.id] = (sd, x_local)
                solve_times.append(solve_time)

        # Apply corrections ONLY to owned cells (the "restricted" part)
        x_new = x.copy()
        for sd_id, (sd, x_local) in corrections.items():
            for i, gi in enumerate(sd.global_indices):
                if gi in sd.owned_indices:
                    # Find local index within the extended region
                    x_new[gi] = x_local[i]

        return x_new


class ParallelBlockSolver:
    """Block-structured parallel solver using domain decomposition internally.

    Wraps SchwartzDD to provide a simple solve(A, b) interface
    that automatically decomposes and solves in parallel.

    For systems where the matrix has block-tridiagonal structure
    (common in 1D PDE discretizations), this gives near-linear speedup.
    """

    def __init__(self, n_blocks: int = 4, overlap: int = 2,
                 method: str = 'ras', tol: float = 1e-10):
        self.dd = SchwartzDD(n_subdomains=n_blocks, overlap=overlap, tol=tol)
        self.method = method

    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        """Solve Ax = b using parallel domain decomposition.

        Returns solution vector x.
        """
        result = self.dd.solve(A, b, method=self.method)
        return result.x
