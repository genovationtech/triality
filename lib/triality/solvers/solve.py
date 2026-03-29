"""Main solve function - the user-facing API"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Optional

from triality.solvers.classify import classify
from triality.solvers.select import select_solver
from triality.solvers.linear import solve_linear
from triality.solvers.assumptions import Assumptions, make_assumptions
from triality.solvers.wellposedness import check_wellposedness
from triality.core.validation import (
    validate_boundary_conditions,
    validate_forcing,
    validate_resolution,
    validate_domain
)
from triality.geometry.fdm import discretize_1d, discretize_2d


@dataclass
class Solution:
    """PDE solution with metadata"""
    u: np.ndarray  # Solution values
    grid: object   # Grid points
    domain: object
    classification: object
    plan: object
    converged: bool
    iterations: int
    residual: float
    time: float
    assumptions: Optional[Assumptions] = None  # What assumptions were made

    def __call__(self, *args):
        """Evaluate solution at point(s)"""
        if self.domain.dim == 1:
            return np.interp(args[0], self.grid, self.u)
        elif self.domain.dim == 2:
            from scipy.interpolate import RegularGridInterpolator
            x_grid, y_grid = self.grid
            u_2d = self.u.reshape(len(x_grid), len(y_grid))
            interp = RegularGridInterpolator((x_grid, y_grid), u_2d)
            result = interp([[args[0], args[1]]])
            return float(result[0])

    def plot(self):
        """Plot solution"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed - cannot plot")
            return

        if self.domain.dim == 1:
            plt.figure(figsize=(10, 6))
            plt.plot(self.grid, self.u, 'b-', linewidth=2)
            plt.xlabel('x')
            plt.ylabel('u(x)')
            plt.title('Solution')
            plt.grid(True)
            plt.show()

        elif self.domain.dim == 2:
            from mpl_toolkits.mplot3d import Axes3D
            x_grid, y_grid = self.grid
            u_2d = self.u.reshape(len(x_grid), len(y_grid))
            X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

            fig = plt.figure(figsize=(15, 5))

            ax1 = fig.add_subplot(131, projection='3d')
            ax1.plot_surface(X, Y, u_2d, cmap='viridis')
            ax1.set_title('Solution (3D)')

            ax2 = fig.add_subplot(132)
            ax2.contourf(X, Y, u_2d, levels=20, cmap='viridis')
            ax2.set_title('Solution (Contour)')
            ax2.set_aspect('equal')

            ax3 = fig.add_subplot(133)
            ax3.imshow(u_2d.T, origin='lower', cmap='viridis', aspect='equal')
            ax3.set_title('Solution (Heatmap)')

            plt.tight_layout()
            plt.show()

    def __repr__(self):
        return (f"Solution(converged={self.converged}, "
                f"residual={self.residual:.2e}, time={self.time:.3f}s)")


def solve(equation, domain, bc=None, resolution=50, verbose=True, forcing=None):
    """
    Solve a PDE automatically.

    Args:
        equation: PDE equation (e.g., laplacian(u) == 1)
        domain: Geometric domain (e.g., Interval(0, 1))
        bc: Boundary conditions (e.g., {'left': 0, 'right': 0})
        resolution: Grid resolution
        verbose: Print progress
        forcing: Optional forcing term override (callable, array, or scalar)

    Returns:
        Solution object

    Example:
        >>> from triality import *
        >>> u = Field("u")
        >>> sol = solve(laplacian(u) == 1, Interval(0, 1), bc={'left': 0, 'right': 0})
        >>> sol.plot()
    """

    start = time.time()

    if bc is None:
        bc = {}

    # Step 0: Validate inputs (surgical fix for NaN/Inf)
    resolution = validate_resolution(resolution)
    validate_domain(domain)
    bc = validate_boundary_conditions(bc, domain)
    if forcing is not None:
        forcing = validate_forcing(forcing)

    # Step 1: Classify
    if verbose:
        print("=" * 60)
        print("Triality: Automatic PDE Solver")
        print("=" * 60)
        print("\n[1/6] Classifying problem...")

    classification = classify(equation, domain)

    if verbose:
        print(f"  Type: {classification.pde_type}")
        print(f"  Linear: {classification.is_linear}")
        print(f"  Dimension: {classification.dimension}D")

    # Step 2: Check well-posedness (surgical fix for pure Neumann, etc.)
    if verbose:
        print("\n[2/6] Checking well-posedness...")

    wellposed = check_wellposedness(classification, domain, bc)
    if not wellposed.is_wellposed:
        raise ValueError(
            f"❌ Ill-posed problem detected:\n"
            f"   {wellposed.issue}\n"
            f"   Suggestion: {wellposed.suggestion}"
        )

    if verbose:
        print(f"  ✓ Problem is well-posed")

    # Step 3: Make assumptions and verify problem
    assumptions = make_assumptions(equation, domain, classification, resolution)

    # Check forcing if provided
    if forcing is not None:
        if callable(forcing):
            # Can't check function easily
            pass
        else:
            assumptions.check_forcing(forcing)

    # Step 4: Select solver
    if verbose:
        print("\n[3/6] Selecting solver strategy...")

    size_est = resolution ** classification.dimension
    plan = select_solver(classification, size_est)

    if verbose:
        print(f"  Discretization: {plan.discretization}")
        print(f"  Linear solver: {plan.linear_solver}")

    # Step 5: Discretize
    if verbose:
        print(f"\n[4/6] Discretizing on {resolution}^{classification.dimension} grid...")

    if classification.dimension == 1:
        A, b, grid = discretize_1d(equation, domain, bc, resolution, forcing=forcing)
    elif classification.dimension == 2:
        A, b, grid = discretize_2d(equation, domain, bc, resolution, forcing=forcing)
    else:
        raise NotImplementedError(f"{classification.dimension}D not supported yet")

    if verbose:
        print(f"  System size: {A.shape[0]} DOFs")

    # Step 6: Solve
    if verbose:
        print("\n[5/6] Solving linear system...")

    result = solve_linear(A, b, method=plan.linear_solver, precond=plan.preconditioner)

    if verbose:
        status = "✓" if result.converged else "✗"
        print(f"  {status} Converged: {result.converged}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Residual: {result.residual:.2e}")

    # Step 7: Verify and package
    solve_time = time.time() - start

    # Estimate condition number if needed
    if not result.converged and assumptions.is_wellposed:
        assumptions.add_warning("Solver did not converge - problem may be ill-conditioned")

    if verbose:
        print(f"\n[6/6] Verifying solution...")
        if assumptions.warnings:
            for warning in assumptions.warnings:
                print(f"  ⚠️  {warning}")
        if assumptions.caveats:
            for caveat in assumptions.caveats:
                print(f"  ℹ️  {caveat}")

        print(f"\n{'='*60}")
        print(f"✓ Solved in {solve_time:.3f} seconds")
        print(f"{'='*60}\n")

    return Solution(
        u=result.x,
        grid=grid,
        domain=domain,
        classification=classification,
        plan=plan,
        converged=result.converged,
        iterations=result.iterations,
        residual=result.residual,
        time=solve_time,
        assumptions=assumptions
    )
