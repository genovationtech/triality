"""Production Preconditioners for Sparse Linear Systems.

Provides ILU(k), AMG-lite (Algebraic Multigrid), and SPAI (Sparse Approximate Inverse)
preconditioners for use with iterative Krylov solvers.

These are the workhorses that turn "converges in 5000 iterations" into "converges in 20".
Without them, any real-world problem with condition number > 1e6 is dead on arrival.

Usage:
    from triality.solvers.preconditioners import ILUPreconditioner, AMGPreconditioner

    A = build_your_system_matrix()
    precond = ILUPreconditioner(fill_factor=2)  # ILU(2)
    M = precond.build(A)
    # Use M as preconditioner in CG/GMRES/BiCGSTAB
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class PreconditionerStats:
    """Diagnostics from preconditioner construction."""
    name: str
    setup_time_ms: float = 0.0
    nnz_original: int = 0
    nnz_preconditioner: int = 0
    fill_ratio: float = 1.0
    estimated_condition: float = 0.0
    levels: int = 0  # For AMG


class ILUPreconditioner:
    """Incomplete LU Factorization with configurable fill level.

    ILU(0): Same sparsity as A. Fast setup, weak preconditioning.
    ILU(k): Allows k levels of fill. Better preconditioning, more memory.

    For problems up to ~100k DOFs, ILU is the workhorse.
    Beyond that, AMG is needed.
    """

    def __init__(self, fill_factor: int = 0, drop_tol: float = 1e-4):
        """
        Args:
            fill_factor: Fill level for ILU. 0=ILU(0), 1=ILU(1), etc.
                         Higher = better preconditioning, more memory.
            drop_tol: Drop tolerance for ILU(T). Entries below this are dropped.
        """
        self.fill_factor = fill_factor
        self.drop_tol = drop_tol
        self._M = None
        self.stats = PreconditionerStats(name=f"ILU({fill_factor})")

    def build(self, A: sp.spmatrix) -> spla.LinearOperator:
        """Construct ILU preconditioner from matrix A.

        Returns a LinearOperator M such that M @ x ≈ A^(-1) @ x.
        """
        import time
        t0 = time.time()

        A_csc = sp.csc_matrix(A)
        n = A_csc.shape[0]
        self.stats.nnz_original = A_csc.nnz

        try:
            # scipy's spilu supports fill_factor and drop_tol
            ilu = spla.spilu(
                A_csc,
                fill_factor=max(1, self.fill_factor + 1),
                drop_tol=self.drop_tol,
            )
            self._M = spla.LinearOperator(
                shape=(n, n),
                matvec=ilu.solve,
                dtype=A.dtype,
            )
            self.stats.nnz_preconditioner = ilu.nnz
            self.stats.fill_ratio = ilu.nnz / max(1, A_csc.nnz)
        except RuntimeError:
            # Fallback: Jacobi (diagonal) preconditioning
            diag = A_csc.diagonal()
            diag = np.where(np.abs(diag) < 1e-15, 1.0, diag)
            inv_diag = 1.0 / diag
            self._M = spla.LinearOperator(
                shape=(n, n),
                matvec=lambda x: inv_diag * x,
                dtype=A.dtype,
            )
            self.stats.name = "ILU(fallback→Jacobi)"
            self.stats.nnz_preconditioner = n

        self.stats.setup_time_ms = (time.time() - t0) * 1000
        return self._M


class AMGPreconditioner:
    """Algebraic Multigrid (AMG-lite) Preconditioner.

    Pure Python AMG using classical Ruge-Stuben coarsening.
    This is the preconditioner you need when your problem is too large
    or too ill-conditioned for ILU.

    Works by:
    1. Building a hierarchy of progressively coarser grids
    2. Smoothing on each level (Gauss-Seidel)
    3. Coarse-grid correction to eliminate low-frequency error

    For Laplacian-like operators, AMG gives mesh-independent convergence:
    the iteration count stays roughly constant as you refine the mesh.
    """

    def __init__(self, max_levels: int = 10, coarse_size: int = 50,
                 strength_threshold: float = 0.25, n_smooth: int = 2):
        """
        Args:
            max_levels: Maximum number of multigrid levels.
            coarse_size: Stop coarsening when matrix is this small.
            strength_threshold: Connection strength threshold for coarsening.
            n_smooth: Pre/post smoothing sweeps per level.
        """
        self.max_levels = max_levels
        self.coarse_size = coarse_size
        self.strength_threshold = strength_threshold
        self.n_smooth = n_smooth
        self._levels = []
        self.stats = PreconditionerStats(name="AMG-lite")

    def _strength_of_connection(self, A: sp.csr_matrix) -> sp.csr_matrix:
        """Identify strong connections using classical AMG criterion.

        Point i is strongly connected to j if:
        |a_ij| >= theta * max_{k!=i} |a_ik|
        """
        n = A.shape[0]
        rows, cols, vals = [], [], []
        A_csr = sp.csr_matrix(A)

        for i in range(n):
            start, end = A_csr.indptr[i], A_csr.indptr[i+1]
            col_idx = A_csr.indices[start:end]
            data = A_csr.data[start:end]

            # Exclude diagonal
            off_diag_mask = col_idx != i
            if not np.any(off_diag_mask):
                continue

            off_diag_abs = np.abs(data[off_diag_mask])
            if len(off_diag_abs) == 0:
                continue

            max_off_diag = np.max(off_diag_abs)
            threshold = self.strength_threshold * max_off_diag

            for k in range(start, end):
                j = A_csr.indices[k]
                if j != i and np.abs(A_csr.data[k]) >= threshold:
                    rows.append(i)
                    cols.append(j)
                    vals.append(1.0)

        if len(rows) == 0:
            return sp.csr_matrix((n, n))
        return sp.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))), shape=(n, n))

    def _coarsen(self, S: sp.csr_matrix, n: int) -> np.ndarray:
        """PMIS-inspired coarsening: select C/F splitting.

        Returns array of 0 (fine) and 1 (coarse) for each DOF.
        """
        # Compute influence measure (number of strong connections)
        influence = np.array(S.sum(axis=0)).flatten() + np.random.rand(n) * 0.01

        cf_splitting = np.zeros(n, dtype=int)  # 0=undecided, 1=coarse, -1=fine
        remaining = set(range(n))

        while remaining:
            # Pick the most influential undecided point
            candidates = list(remaining)
            if not candidates:
                break
            best = max(candidates, key=lambda i: influence[i])

            # Make it a coarse point
            cf_splitting[best] = 1
            remaining.discard(best)

            # Make all its strong connections fine points
            start, end = S.indptr[best], S.indptr[best+1]
            neighbors = S.indices[start:end]
            for j in neighbors:
                if j in remaining:
                    cf_splitting[j] = -1
                    remaining.discard(j)
                    # Boost influence of j's remaining strong neighbors
                    jstart, jend = S.indptr[j], S.indptr[j+1]
                    for k in S.indices[jstart:jend]:
                        if k in remaining:
                            influence[k] += 1.0

        # Any remaining undecided -> coarse
        cf_splitting[cf_splitting == 0] = 1
        return cf_splitting

    def _build_interpolation(self, A: sp.csr_matrix, cf_splitting: np.ndarray) -> sp.csr_matrix:
        """Build prolongation operator P (interpolation from coarse to fine).

        Uses direct interpolation: fine point values are weighted averages
        of their coarse neighbors.
        """
        n = A.shape[0]
        coarse_idx = np.where(cf_splitting == 1)[0]
        nc = len(coarse_idx)

        if nc == 0 or nc == n:
            return sp.eye(n, format='csr')

        # Map global coarse index to local coarse index
        coarse_map = -np.ones(n, dtype=int)
        coarse_map[coarse_idx] = np.arange(nc)

        rows, cols, vals = [], [], []
        A_csr = sp.csr_matrix(A)

        for i in range(n):
            if cf_splitting[i] == 1:
                # Coarse point: inject directly
                rows.append(i)
                cols.append(coarse_map[i])
                vals.append(1.0)
            else:
                # Fine point: interpolate from coarse neighbors
                start, end = A_csr.indptr[i], A_csr.indptr[i+1]
                col_idx = A_csr.indices[start:end]
                data = A_csr.data[start:end]

                # Find coarse neighbors
                coarse_neighbor_mask = np.array([cf_splitting[j] == 1 for j in col_idx])
                c_cols = col_idx[coarse_neighbor_mask]
                c_data = data[coarse_neighbor_mask]

                if len(c_cols) == 0:
                    # No coarse neighbors: pick closest coarse point
                    if nc > 0:
                        rows.append(i)
                        cols.append(0)
                        vals.append(1.0)
                    continue

                # Weight by |a_ij| / sum(|a_ij| for coarse j)
                weights = np.abs(c_data)
                weight_sum = np.sum(weights)
                if weight_sum < 1e-15:
                    weights = np.ones(len(c_data)) / len(c_data)
                else:
                    weights = weights / weight_sum

                for j, w in zip(c_cols, weights):
                    rows.append(i)
                    cols.append(coarse_map[j])
                    vals.append(w)

        P = sp.csr_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(n, nc)
        )
        return P

    def _gauss_seidel_sweep(self, A: sp.csr_matrix, x: np.ndarray, b: np.ndarray,
                            n_sweeps: int = 1) -> np.ndarray:
        """Forward Gauss-Seidel smoothing (the engine of multigrid)."""
        x = x.copy()
        diag = A.diagonal()
        for _ in range(n_sweeps):
            for i in range(A.shape[0]):
                if abs(diag[i]) < 1e-15:
                    continue
                start, end = A.indptr[i], A.indptr[i+1]
                col_idx = A.indices[start:end]
                data = A.data[start:end]
                sigma = np.dot(data, x[col_idx])
                x[i] = (b[i] - sigma + diag[i] * x[i]) / diag[i]
        return x

    def build(self, A: sp.spmatrix) -> spla.LinearOperator:
        """Construct AMG hierarchy from matrix A."""
        import time
        t0 = time.time()

        A_csr = sp.csr_matrix(A)
        n = A_csr.shape[0]
        self.stats.nnz_original = A_csr.nnz

        self._levels = [{'A': A_csr}]

        for level in range(self.max_levels - 1):
            A_level = self._levels[-1]['A']
            n_level = A_level.shape[0]

            if n_level <= self.coarse_size:
                break

            # 1. Strength of connection
            S = self._strength_of_connection(A_level)

            # 2. Coarsen
            cf = self._coarsen(S, n_level)

            # 3. Build interpolation
            P = self._build_interpolation(A_level, cf)
            R = P.T  # Restriction = transpose of prolongation

            # 4. Galerkin coarse operator: A_c = R @ A @ P
            A_coarse = R @ A_level @ P
            A_coarse = sp.csr_matrix(A_coarse)

            self._levels[-1]['P'] = P
            self._levels[-1]['R'] = R
            self._levels.append({'A': A_coarse})

            if A_coarse.shape[0] <= self.coarse_size or A_coarse.shape[0] >= n_level * 0.9:
                break

        self.stats.levels = len(self._levels)

        # Build LU for coarsest level
        coarsest = self._levels[-1]['A']
        if coarsest.shape[0] < 2000:
            try:
                self._levels[-1]['LU'] = spla.splu(sp.csc_matrix(coarsest))
            except Exception:
                self._levels[-1]['LU'] = None
        else:
            self._levels[-1]['LU'] = None

        def v_cycle(b: np.ndarray) -> np.ndarray:
            return self._v_cycle(b, level=0)

        M = spla.LinearOperator(shape=(n, n), matvec=v_cycle, dtype=A.dtype)

        self.stats.setup_time_ms = (time.time() - t0) * 1000
        return M

    def _v_cycle(self, b: np.ndarray, level: int) -> np.ndarray:
        """Single V-cycle: the fundamental multigrid operation."""
        lvl = self._levels[level]
        A = lvl['A']
        n = A.shape[0]

        # Coarsest level: direct solve
        if level == len(self._levels) - 1:
            if 'LU' in lvl and lvl['LU'] is not None:
                return lvl['LU'].solve(b)
            return spla.spsolve(sp.csc_matrix(A), b)

        # Pre-smoothing
        x = np.zeros(n)
        x = self._gauss_seidel_sweep(A, x, b, self.n_smooth)

        # Compute residual
        r = b - A @ x

        # Restrict residual to coarse grid
        R = lvl['R']
        r_coarse = R @ r

        # Recurse on coarse grid
        e_coarse = self._v_cycle(r_coarse, level + 1)

        # Prolongate correction
        P = lvl['P']
        x = x + P @ e_coarse

        # Post-smoothing
        x = self._gauss_seidel_sweep(A, x, b, self.n_smooth)

        return x


class SPAIPreconditioner:
    """Sparse Approximate Inverse Preconditioner.

    Computes M ≈ A^(-1) directly as a sparse matrix.
    The key advantage: M @ v is a sparse matrix-vector product,
    which is trivially parallelizable (unlike triangular solves in ILU).

    Best for: GPU-bound problems, highly parallel architectures.
    """

    def __init__(self, sparsity_pattern: str = 'diagonal', n_augment: int = 0):
        """
        Args:
            sparsity_pattern: 'diagonal' (Jacobi), 'A_pattern' (match A's pattern),
                             or 'augmented' (A + neighbors).
            n_augment: Extra fill levels for augmented pattern.
        """
        self.sparsity_pattern = sparsity_pattern
        self.n_augment = n_augment
        self.stats = PreconditionerStats(name=f"SPAI({sparsity_pattern})")

    def build(self, A: sp.spmatrix) -> spla.LinearOperator:
        """Construct SPAI by minimizing ||AM - I||_F column-by-column."""
        import time
        t0 = time.time()

        A_csr = sp.csr_matrix(A)
        n = A_csr.shape[0]
        self.stats.nnz_original = A_csr.nnz

        if self.sparsity_pattern == 'diagonal':
            # Jacobi: M = diag(1/a_ii)
            diag = A_csr.diagonal()
            diag = np.where(np.abs(diag) < 1e-15, 1.0, diag)
            inv_diag = 1.0 / diag
            M_sparse = sp.diags(inv_diag, format='csr')
        elif self.sparsity_pattern == 'A_pattern':
            # SPAI with A's sparsity pattern
            M_sparse = self._build_spai_pattern(A_csr)
        else:
            # Default to A_pattern
            M_sparse = self._build_spai_pattern(A_csr)

        self.stats.nnz_preconditioner = M_sparse.nnz
        self.stats.fill_ratio = M_sparse.nnz / max(1, A_csr.nnz)
        self.stats.setup_time_ms = (time.time() - t0) * 1000

        return spla.LinearOperator(
            shape=(n, n),
            matvec=lambda x: M_sparse @ x,
            dtype=A.dtype,
        )

    def _build_spai_pattern(self, A: sp.csr_matrix) -> sp.csr_matrix:
        """Build SPAI matching A's sparsity pattern.

        For each column k, solve: min ||A @ m_k - e_k||_2
        subject to m_k having same sparsity as A's column k.
        """
        n = A.shape[0]
        A_csc = sp.csc_matrix(A)
        rows, cols, vals = [], [], []

        for k in range(n):
            # Get column k's sparsity pattern
            start, end = A_csc.indptr[k], A_csc.indptr[k+1]
            J = A_csc.indices[start:end]  # Row indices of nonzeros in column k

            if len(J) == 0:
                continue

            # Get rows of A corresponding to pattern
            # We need I = union of nonzero columns in A[J, :]
            I_set = set()
            for j in J:
                row_start, row_end = A.indptr[j], A.indptr[j+1]
                I_set.update(A.indices[row_start:row_end])
            I = np.array(sorted(I_set))

            if len(I) == 0:
                continue

            # Extract submatrix A_hat = A[I, J]
            A_hat = A[np.ix_(I, J)].toarray()

            # RHS: e_k restricted to I
            rhs = np.zeros(len(I))
            k_in_I = np.searchsorted(I, k)
            if k_in_I < len(I) and I[k_in_I] == k:
                rhs[k_in_I] = 1.0

            # Solve least-squares: min ||A_hat @ m - rhs||
            m_k, _, _, _ = np.linalg.lstsq(A_hat, rhs, rcond=None)

            for j_idx, j in enumerate(J):
                if abs(m_k[j_idx]) > 1e-15:
                    rows.append(j)
                    cols.append(k)
                    vals.append(m_k[j_idx])

        return sp.csr_matrix((np.array(vals), (np.array(rows), np.array(cols))), shape=(n, n))


def auto_preconditioner(A: sp.spmatrix, problem_size_hint: int = None) -> Tuple[spla.LinearOperator, PreconditionerStats]:
    """Automatically select the best preconditioner based on problem characteristics.

    Rules:
    - N < 500: No preconditioning (direct solve is fine)
    - 500 <= N < 10000: ILU(0) or ILU(2)
    - 10000 <= N < 100000: AMG
    - N >= 100000: AMG with more levels

    Returns:
        (preconditioner_operator, stats)
    """
    n = A.shape[0] if problem_size_hint is None else problem_size_hint

    if n < 500:
        # Direct solve territory - Jacobi is enough if iterative
        precond = ILUPreconditioner(fill_factor=0)
    elif n < 10000:
        precond = ILUPreconditioner(fill_factor=2, drop_tol=1e-3)
    elif n < 100000:
        precond = AMGPreconditioner(max_levels=6, coarse_size=100)
    else:
        precond = AMGPreconditioner(max_levels=10, coarse_size=200, n_smooth=3)

    M = precond.build(A)
    return M, precond.stats
