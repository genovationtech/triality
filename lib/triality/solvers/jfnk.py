"""Jacobian-Free Newton-Krylov (JFNK) Solver.

JFNK is the heavy artillery of nonlinear solvers. It combines:
1. Newton's method for the outer nonlinear iteration
2. Krylov subspace methods (GMRES/BiCGSTAB) for the inner linear solve
3. Matrix-free Jacobian-vector products via finite differences

The key insight: you never need to form or store the Jacobian matrix.
Instead, you approximate J @ v using:

    J(x) @ v ≈ [F(x + eps*v) - F(x)] / eps

This makes JFNK ideal for:
- Very large systems where J is too expensive to form
- Complex multiphysics where analytic J is impossible
- Problems where J changes every iteration

With a good preconditioner, JFNK converges in ~5-15 outer Newton iterations,
each requiring ~10-30 Krylov iterations for the inner solve.

Usage:
    from triality.solvers.jfnk import JFNKSolver

    def residual(x):
        return compute_residual(x)  # Returns F(x), want F(x) = 0

    solver = JFNKSolver(n_dof=1000, krylov='gmres')
    x, info = solver.solve(x0, residual)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Callable, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class JFNKResult:
    """Result from JFNK solver."""
    x: np.ndarray
    converged: bool
    outer_iterations: int = 0
    total_krylov_iterations: int = 0
    residual_history: list = field(default_factory=list)
    reason: str = ""


class JFNKSolver:
    """Jacobian-Free Newton-Krylov nonlinear solver.

    Solves F(x) = 0 using inexact Newton iteration where the linear
    system J(x_k) @ dx = -F(x_k) is solved approximately via Krylov methods.

    Features:
    - Matrix-free Jacobian-vector products (no J assembly needed)
    - Choice of Krylov solver: GMRES, BiCGSTAB, or LGMRES
    - Eisenstat-Walker forcing for adaptive Krylov tolerance
    - Backtracking line search for globalization
    - Optional physics-based right preconditioning
    - Automatic step-size selection for finite differences
    """

    def __init__(self, n_dof: int, krylov: str = 'gmres',
                 max_newton: int = 30, newton_tol: float = 1e-8,
                 max_krylov: int = 100, krylov_tol: float = 1e-3,
                 fd_eps: float = None,
                 eisenstat_walker: bool = True,
                 line_search: bool = True):
        """
        Args:
            n_dof: Number of degrees of freedom.
            krylov: Inner Krylov solver ('gmres', 'bicgstab', 'lgmres').
            max_newton: Maximum outer Newton iterations.
            newton_tol: Outer Newton convergence tolerance.
            max_krylov: Maximum inner Krylov iterations per Newton step.
            krylov_tol: Initial Krylov tolerance (adapted by Eisenstat-Walker).
            fd_eps: Finite difference perturbation. Auto-computed if None.
            eisenstat_walker: Use adaptive Krylov tolerance.
            line_search: Use backtracking line search.
        """
        self.n_dof = n_dof
        self.krylov = krylov
        self.max_newton = max_newton
        self.newton_tol = newton_tol
        self.max_krylov = max_krylov
        self.krylov_tol = krylov_tol
        self.fd_eps = fd_eps
        self.eisenstat_walker = eisenstat_walker
        self.line_search = line_search

    def _jvp(self, F: Callable, x: np.ndarray, F_x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Jacobian-vector product via finite differences.

        J(x) @ v ≈ [F(x + eps*v) - F(x)] / eps

        Uses the optimal perturbation size from Dennis & Schnabel:
            eps = sqrt(machine_eps) * max(||x||, 1) / ||v||
        """
        if self.fd_eps is not None:
            eps = self.fd_eps
        else:
            # Optimal perturbation for double precision
            eps_machine = np.finfo(float).eps
            x_norm = max(np.linalg.norm(x), 1.0)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-15:
                return np.zeros_like(F_x)
            eps = np.sqrt(eps_machine) * x_norm / v_norm

        return (F(x + eps * v) - F_x) / eps

    def solve(self, x0: np.ndarray, F: Callable,
              preconditioner: spla.LinearOperator = None,
              callback: Callable = None) -> JFNKResult:
        """Solve F(x) = 0 using JFNK.

        Args:
            x0: Initial guess.
            F: Residual function F(x) -> np.ndarray.
            preconditioner: Optional right preconditioner M such that
                           J @ M^(-1) is better conditioned.
            callback: Called as callback(k, x, ||F||) at each Newton step.

        Returns:
            JFNKResult with solution and diagnostics.
        """
        x = x0.copy()
        F_x = F(x)
        res_norm = np.linalg.norm(F_x)
        res_norm_0 = res_norm
        history = [res_norm]
        total_krylov = 0

        # Eisenstat-Walker parameters
        eta = self.krylov_tol  # Forcing term
        eta_max = 0.9
        gamma = 0.9

        for k in range(self.max_newton):
            if res_norm < self.newton_tol:
                return JFNKResult(
                    x=x, converged=True,
                    outer_iterations=k,
                    total_krylov_iterations=total_krylov,
                    residual_history=history,
                    reason="Converged"
                )

            if callback is not None:
                callback(k, x, res_norm)

            # Build matrix-free Jacobian operator
            jvp_op = spla.LinearOperator(
                shape=(self.n_dof, self.n_dof),
                matvec=lambda v, _x=x, _Fx=F_x: self._jvp(F, _x, _Fx, v),
                dtype=float,
            )

            # Eisenstat-Walker: adapt Krylov tolerance
            if self.eisenstat_walker and k > 0:
                res_prev = history[-2]
                eta_new = gamma * (res_norm / res_prev) ** 2
                # Safeguard: don't over-tighten
                eta_safeguard = gamma * eta ** 2
                eta = min(eta_max, max(eta_new, eta_safeguard))
                eta = min(eta, eta_max)
            else:
                eta = self.krylov_tol

            # Inner Krylov solve: J @ dx = -F(x)
            krylov_iters = [0]

            def krylov_callback(xk):
                krylov_iters[0] += 1

            try:
                if self.krylov == 'gmres':
                    dx, info = spla.gmres(
                        jvp_op, -F_x,
                        M=preconditioner,
                        rtol=eta,
                        maxiter=self.max_krylov,
                        callback=krylov_callback,
                    )
                elif self.krylov == 'bicgstab':
                    dx, info = spla.bicgstab(
                        jvp_op, -F_x,
                        M=preconditioner,
                        rtol=eta,
                        maxiter=self.max_krylov,
                        callback=krylov_callback,
                    )
                elif self.krylov == 'lgmres':
                    dx, info = spla.lgmres(
                        jvp_op, -F_x,
                        M=preconditioner,
                        rtol=eta,
                        maxiter=self.max_krylov,
                    )
                    krylov_iters[0] = self.max_krylov  # lgmres doesn't support callback
                else:
                    raise ValueError(f"Unknown Krylov method: {self.krylov}")
            except Exception as e:
                return JFNKResult(
                    x=x, converged=False,
                    outer_iterations=k,
                    total_krylov_iterations=total_krylov,
                    residual_history=history,
                    reason=f"Krylov solve failed: {e}"
                )

            total_krylov += krylov_iters[0]

            # Line search (backtracking)
            if self.line_search:
                alpha = 1.0
                while alpha > 1e-4:
                    x_trial = x + alpha * dx
                    F_trial = F(x_trial)
                    res_trial = np.linalg.norm(F_trial)
                    if res_trial < (1.0 - 1e-4 * alpha) * res_norm:
                        x = x_trial
                        F_x = F_trial
                        res_norm = res_trial
                        break
                    alpha *= 0.5

                if alpha <= 1e-4:
                    # Accept anyway to avoid stalling
                    x = x + 1e-4 * dx
                    F_x = F(x)
                    res_norm = np.linalg.norm(F_x)
            else:
                x = x + dx
                F_x = F(x)
                res_norm = np.linalg.norm(F_x)

            history.append(res_norm)

        return JFNKResult(
            x=x, converged=False,
            outer_iterations=self.max_newton,
            total_krylov_iterations=total_krylov,
            residual_history=history,
            reason="Max Newton iterations reached"
        )


class PhysicsBasedPreconditioner:
    """Build a preconditioner from a simplified physics operator.

    For JFNK, the preconditioner M doesn't need to be accurate —
    it just needs to capture the dominant physics.

    Common strategies:
    - Laplacian preconditioner for diffusion-dominated problems
    - Block diagonal for loosely coupled multiphysics
    - Operator splitting: precondition with one physics at a time
    """

    @staticmethod
    def from_laplacian(n: int, dx: float, coeff: float = 1.0) -> spla.LinearOperator:
        """Build a 1D Laplacian-based preconditioner.

        Good for: heat equation, diffusion, Poisson-like problems.
        """
        diag = np.ones(n) * (-2.0 * coeff / dx**2)
        off = np.ones(n - 1) * (coeff / dx**2)
        A_approx = sp.diags([diag, off, off], [0, -1, 1], format='csc')
        # Fix boundaries
        A_approx = sp.lil_matrix(A_approx)
        A_approx[0, :] = 0; A_approx[0, 0] = 1.0
        A_approx[-1, :] = 0; A_approx[-1, -1] = 1.0

        lu = spla.splu(sp.csc_matrix(A_approx))
        return spla.LinearOperator(shape=(n, n), matvec=lu.solve, dtype=float)

    @staticmethod
    def block_diagonal(blocks: list) -> spla.LinearOperator:
        """Build block-diagonal preconditioner from list of small matrices.

        For multiphysics: one block per physics field.
        """
        n_total = sum(b.shape[0] for b in blocks)
        block_sizes = [b.shape[0] for b in blocks]

        # Pre-factorize each block
        factored = []
        for B in blocks:
            if sp.issparse(B):
                factored.append(spla.splu(sp.csc_matrix(B)))
            else:
                from scipy.linalg import lu_factor, lu_solve
                factored.append(lu_factor(B))

        def matvec(x):
            result = np.zeros_like(x)
            offset = 0
            for i, (fac, size) in enumerate(zip(factored, block_sizes)):
                x_block = x[offset:offset + size]
                if hasattr(fac, 'solve'):
                    result[offset:offset + size] = fac.solve(x_block)
                else:
                    from scipy.linalg import lu_solve
                    result[offset:offset + size] = lu_solve(fac, x_block)
                offset += size
            return result

        return spla.LinearOperator(shape=(n_total, n_total), matvec=matvec, dtype=float)
