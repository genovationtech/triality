"""
Conservation Checks

Verifies that discretization preserves conservation laws. For elliptic PDEs,
we check integral identities and boundary flux conservation.

For Poisson equation ∇²u = f with homogeneous Dirichlet BC:
- ∫∫ ∇²u dA = ∫∫ f dA (integral identity)
- ∮ ∂u/∂n ds = ∫∫ f dA (Gauss's theorem)

Reference:
- LeVeque (2002). "Finite Volume Methods for Hyperbolic Problems"
- Toro (2009). "Riemann Solvers and Numerical Methods for Fluid Dynamics"
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ConservationResult:
    """Results from conservation check"""
    integral_lhs: float
    integral_rhs: float
    relative_error: float
    tolerance: float
    passed: bool
    test_name: str


class ConservationTest:
    """Conservation verification for PDE discretizations"""

    def __init__(self, tolerance: float = 1e-10):
        """
        Args:
            tolerance: Relative error tolerance for conservation
        """
        self.tolerance = tolerance

    def verify_integral_identity_1d(self,
                                     u: np.ndarray,
                                     f: np.ndarray,
                                     grid: np.ndarray) -> ConservationResult:
        """
        Verify integral identity for 1D Poisson: ∫ u'' dx = ∫ f dx

        For homogeneous BC, ∫ u'' dx = u'(b) - u'(a) ≈ 0

        Args:
            u: Solution values
            f: Forcing term values
            grid: Spatial grid

        Returns:
            ConservationResult
        """
        # Compute integral of f using trapezoidal rule
        try:
            integral_f = np.trapezoid(f, grid)  # NumPy >= 1.25
        except AttributeError:
            integral_f = np.trapezoid(f, grid)  # NumPy < 1.25

        # Compute u'(b) - u'(a) using finite differences
        h = grid[1] - grid[0]
        du_dx_left = (u[1] - u[0]) / h
        du_dx_right = (u[-1] - u[-2]) / h
        integral_u_pp = du_dx_right - du_dx_left

        # They should be equal (Gauss's theorem)
        rel_error = abs(integral_u_pp - integral_f) / (abs(integral_f) + 1e-16)
        passed = rel_error < self.tolerance

        return ConservationResult(
            integral_lhs=integral_u_pp,
            integral_rhs=integral_f,
            relative_error=rel_error,
            tolerance=self.tolerance,
            passed=passed,
            test_name="1D Integral Identity"
        )

    def verify_integral_identity_2d(self,
                                     u: np.ndarray,
                                     f: np.ndarray,
                                     grid_x: np.ndarray,
                                     grid_y: np.ndarray) -> ConservationResult:
        """
        Verify integral identity for 2D Poisson: ∫∫ ∇²u dA = ∫∫ f dA

        Args:
            u: Solution values (flattened or 2D)
            f: Forcing term values
            grid_x: X grid
            grid_y: Y grid

        Returns:
            ConservationResult
        """
        # Reshape if needed
        Nx, Ny = len(grid_x), len(grid_y)
        if u.ndim == 1:
            u_2d = u.reshape(Nx, Ny)
        else:
            u_2d = u

        # Compute integral of f using 2D trapezoidal rule
        hx = grid_x[1] - grid_x[0]
        hy = grid_y[1] - grid_y[0]

        if isinstance(f, (int, float)):
            integral_f = f * (grid_x[-1] - grid_x[0]) * (grid_y[-1] - grid_y[0])
        else:
            if f.ndim == 1:
                f_2d = f.reshape(Nx, Ny)
            else:
                f_2d = f
            try:
                integral_f = np.trapezoid(np.trapezoid(f_2d, grid_x, axis=0), grid_y)  # NumPy >= 1.25
            except AttributeError:
                integral_f = np.trapezoid(np.trapezoid(f_2d, grid_x, axis=0), grid_y)  # NumPy < 1.25

        # Compute boundary flux ∮ ∂u/∂n ds
        # Left boundary (x=0): flux = -∂u/∂x
        flux_left = -np.sum((u_2d[1, :] - u_2d[0, :]) / hx) * hy

        # Right boundary (x=1): flux = ∂u/∂x
        flux_right = np.sum((u_2d[-1, :] - u_2d[-2, :]) / hx) * hy

        # Bottom boundary (y=0): flux = -∂u/∂y
        flux_bottom = -np.sum((u_2d[:, 1] - u_2d[:, 0]) / hy) * hx

        # Top boundary (y=1): flux = ∂u/∂y
        flux_top = np.sum((u_2d[:, -1] - u_2d[:, -2]) / hy) * hx

        # Total flux (should equal integral of f)
        total_flux = flux_left + flux_right + flux_bottom + flux_top

        rel_error = abs(total_flux - integral_f) / (abs(integral_f) + 1e-16)
        passed = rel_error < self.tolerance

        return ConservationResult(
            integral_lhs=total_flux,
            integral_rhs=integral_f,
            relative_error=rel_error,
            tolerance=self.tolerance,
            passed=passed,
            test_name="2D Integral Identity (Gauss's Theorem)"
        )

    def verify_symmetry(self,
                       u: np.ndarray,
                       grid: np.ndarray,
                       expected_symmetry: str = 'even') -> ConservationResult:
        """
        Verify solution symmetry for symmetric problems

        Args:
            u: Solution values
            grid: Spatial grid
            expected_symmetry: 'even' or 'odd'

        Returns:
            ConservationResult
        """
        # Find midpoint
        mid_idx = len(grid) // 2

        # Check symmetry
        left_half = u[:mid_idx]
        right_half = u[-mid_idx:][::-1]  # Reversed

        if expected_symmetry == 'even':
            # u(x) = u(-x)
            error = np.max(np.abs(left_half - right_half))
        elif expected_symmetry == 'odd':
            # u(x) = -u(-x)
            error = np.max(np.abs(left_half + right_half))
        else:
            raise ValueError(f"Unknown symmetry: {expected_symmetry}")

        max_u = np.max(np.abs(u))
        rel_error = error / (max_u + 1e-16)
        passed = rel_error < self.tolerance

        return ConservationResult(
            integral_lhs=0.0,
            integral_rhs=0.0,
            relative_error=rel_error,
            tolerance=self.tolerance,
            passed=passed,
            test_name=f"{expected_symmetry.capitalize()} Symmetry"
        )

    def verify_maximum_principle(self,
                                 u: np.ndarray,
                                 f: np.ndarray,
                                 bc_values: Tuple[float, float]) -> ConservationResult:
        """
        Verify maximum principle for Poisson equation

        For -∇²u = f with f > 0, the maximum of u occurs on the boundary.

        Args:
            u: Solution values
            f: Forcing term
            bc_values: Boundary condition values (left, right)

        Returns:
            ConservationResult
        """
        # Check if all f > 0
        f_positive = np.all(f >= 0) if isinstance(f, np.ndarray) else f >= 0

        if not f_positive:
            # Maximum principle only applies when f has constant sign
            return ConservationResult(
                integral_lhs=0.0,
                integral_rhs=0.0,
                relative_error=0.0,
                tolerance=self.tolerance,
                passed=True,
                test_name="Maximum Principle (Not Applicable)"
            )

        # Maximum should be at boundary
        max_interior = np.max(u[1:-1])
        max_boundary = max(bc_values[0], bc_values[1])

        # Interior max should not exceed boundary max (with small tolerance)
        violation = max_interior - max_boundary
        rel_error = max(0, violation) / (max_boundary + 1e-16)
        passed = rel_error < self.tolerance

        return ConservationResult(
            integral_lhs=max_interior,
            integral_rhs=max_boundary,
            relative_error=rel_error,
            tolerance=self.tolerance,
            passed=passed,
            test_name="Maximum Principle"
        )

    @staticmethod
    def print_results(result: ConservationResult):
        """Pretty print conservation results"""
        print(f"\n{'='*70}")
        print(f"  {result.test_name}")
        print(f"{'='*70}")

        if "Integral" in result.test_name or "Gauss" in result.test_name:
            print(f"  LHS: {result.integral_lhs:15.6e}")
            print(f"  RHS: {result.integral_rhs:15.6e}")
            print(f"  Relative Error: {result.relative_error:.6e}")
        elif "Symmetry" in result.test_name:
            print(f"  Relative Error: {result.relative_error:.6e}")
        elif "Maximum" in result.test_name:
            print(f"  Interior Max: {result.integral_lhs:.6e}")
            print(f"  Boundary Max: {result.integral_rhs:.6e}")
            print(f"  Violation:    {result.relative_error:.6e}")

        print(f"  Tolerance:      {result.tolerance:.6e}")

        if result.passed:
            print(f"\n✅ PASSED - Conservation property satisfied")
        else:
            print(f"\n❌ FAILED - Conservation property violated")
        print(f"{'='*70}\n")
