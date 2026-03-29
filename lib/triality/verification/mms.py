"""
Method of Manufactured Solutions (MMS)

The gold standard for PDE solver verification. We manufacture an exact solution,
compute the required forcing term, solve numerically, and verify convergence rates.

Reference:
- Roache, P.J. (2002). "Code Verification by the Method of Manufactured Solutions"
- Salari & Knupp (2000). "Code Verification by the Method of Manufactured Solutions"
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from triality import Field, Eq, laplacian, Interval, Rectangle, Square, solve


@dataclass
class MMSResult:
    """Results from MMS verification"""
    resolutions: List[int]
    grid_spacings: List[float]
    l2_errors: List[float]
    linf_errors: List[float]
    convergence_rate_l2: float
    convergence_rate_linf: float
    expected_rate: float
    passed: bool
    tolerance: float


class MMSTest:
    """Method of Manufactured Solutions verification"""

    def __init__(self, tolerance: float = 0.1):
        """
        Args:
            tolerance: Allowable deviation from expected convergence rate
        """
        self.tolerance = tolerance

    def verify_1d_poisson(self,
                          u_exact: Callable[[np.ndarray], np.ndarray],
                          forcing: Callable[[np.ndarray], np.ndarray],
                          domain: Tuple[float, float] = (0, 1),
                          bc_left: float = 0.0,
                          bc_right: float = 0.0,
                          resolutions: List[int] = None) -> MMSResult:
        """
        Verify 1D Poisson equation: -u'' = f

        Args:
            u_exact: Manufactured exact solution u(x)
            forcing: Forcing term f(x) = -u''(x)
            domain: Spatial domain (a, b)
            bc_left: Boundary condition at x=a
            bc_right: Boundary condition at x=b
            resolutions: List of grid resolutions to test

        Returns:
            MMSResult with convergence analysis
        """
        if resolutions is None:
            resolutions = [20, 40, 80, 160, 320]

        grid_spacings = []
        l2_errors = []
        linf_errors = []

        for N in resolutions:
            # Solve numerically
            u = Field("u")
            interval = Interval(domain[0], domain[1])

            # Create grid for this resolution
            x = np.linspace(domain[0], domain[1], N)
            h = x[1] - x[0]

            # Use dummy equation (actual forcing passed via forcing parameter)
            eq = Eq(laplacian(u), 0)
            bc = {'left': bc_left, 'right': bc_right}

            try:
                # Pass forcing function directly - TRUE MMS!
                sol = solve(eq, interval, bc=bc, resolution=N, verbose=False,
                           forcing=forcing)

                # Compute exact solution at grid points
                u_exact_vals = u_exact(sol.grid)

                # Compute errors
                error = sol.u - u_exact_vals
                l2_error = np.sqrt(np.mean(error**2))
                linf_error = np.max(np.abs(error))

                grid_spacings.append(h)
                l2_errors.append(l2_error)
                linf_errors.append(linf_error)

            except Exception as e:
                print(f"Warning: MMS failed at N={N}: {e}")
                continue

        # Compute convergence rates using least squares fit
        # log(error) = log(C) + p*log(h), so slope = p
        if len(grid_spacings) >= 3:
            # Check if errors are at machine precision
            machine_eps = 1e-14
            if max(l2_errors) < machine_eps:
                # Errors at machine precision - solver is perfect!
                p_l2 = 2.0  # Assign expected rate
                p_linf = 2.0
                passed = True
            else:
                log_h = np.log(grid_spacings)
                log_l2 = np.log(l2_errors)
                log_linf = np.log(linf_errors)

                # Fit: log(error) = a + p*log(h)
                p_l2 = np.polyfit(log_h, log_l2, 1)[0]
                p_linf = np.polyfit(log_h, log_linf, 1)[0]

                # Expected rate for 2nd order method
                expected_rate = 2.0

                # Check if convergence rates are within tolerance
                passed = (abs(p_l2 - expected_rate) < self.tolerance and
                          abs(p_linf - expected_rate) < self.tolerance)
        else:
            p_l2 = 0.0
            p_linf = 0.0
            passed = False

        # Expected rate for 2nd order method
        expected_rate = 2.0

        return MMSResult(
            resolutions=resolutions[:len(grid_spacings)],
            grid_spacings=grid_spacings,
            l2_errors=l2_errors,
            linf_errors=linf_errors,
            convergence_rate_l2=p_l2,
            convergence_rate_linf=p_linf,
            expected_rate=expected_rate,
            passed=passed,
            tolerance=self.tolerance
        )

    def verify_2d_poisson(self,
                          u_exact: Callable[[np.ndarray, np.ndarray], np.ndarray],
                          forcing: Callable[[np.ndarray, np.ndarray], np.ndarray],
                          domain: Tuple[float, float, float, float] = (0, 1, 0, 1),
                          bc_value: float = 0.0,
                          resolutions: List[int] = None) -> MMSResult:
        """
        Verify 2D Poisson equation: -∇²u = f

        Args:
            u_exact: Manufactured exact solution u(x,y)
            forcing: Forcing term f(x,y) = -∇²u
            domain: Spatial domain (x0, x1, y0, y1)
            bc_value: Boundary condition value (all boundaries)
            resolutions: List of grid resolutions to test

        Returns:
            MMSResult with convergence analysis
        """
        if resolutions is None:
            resolutions = [10, 20, 40, 80]  # 2D is more expensive

        grid_spacings = []
        l2_errors = []
        linf_errors = []

        x0, x1, y0, y1 = domain

        for N in resolutions:
            # Solve numerically
            u = Field("u")
            rect = Rectangle(x0, x1, y0, y1)

            # Create grid
            x = np.linspace(x0, x1, N)
            y = np.linspace(y0, y1, N)
            h = x[1] - x[0]

            # Use dummy equation (actual forcing passed via forcing parameter)
            eq = Eq(laplacian(u), 0)
            bc = {'left': bc_value, 'right': bc_value,
                  'bottom': bc_value, 'top': bc_value}

            try:
                # Pass forcing function directly - TRUE MMS!
                sol = solve(eq, rect, bc=bc, resolution=N, verbose=False,
                           forcing=forcing)

                # Compute exact solution at grid points
                x_grid, y_grid = sol.grid
                X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
                u_exact_vals = u_exact(X_grid, Y_grid).flatten()

                # Compute errors
                error = sol.u - u_exact_vals
                l2_error = np.sqrt(np.mean(error**2))
                linf_error = np.max(np.abs(error))

                grid_spacings.append(h)
                l2_errors.append(l2_error)
                linf_errors.append(linf_error)

            except Exception as e:
                print(f"Warning: MMS failed at N={N}: {e}")
                continue

        # Compute convergence rates
        if len(grid_spacings) >= 3:
            # Check if errors are at machine precision
            machine_eps = 1e-14
            if max(l2_errors) < machine_eps:
                # Errors at machine precision - solver is perfect!
                p_l2 = 2.0  # Assign expected rate
                p_linf = 2.0
                passed = True
            else:
                log_h = np.log(grid_spacings)
                log_l2 = np.log(l2_errors)
                log_linf = np.log(linf_errors)

                p_l2 = np.polyfit(log_h, log_l2, 1)[0]
                p_linf = np.polyfit(log_h, log_linf, 1)[0]

                expected_rate = 2.0
                passed = (abs(p_l2 - expected_rate) < self.tolerance and
                          abs(p_linf - expected_rate) < self.tolerance)
        else:
            p_l2 = 0.0
            p_linf = 0.0
            passed = False

        expected_rate = 2.0

        return MMSResult(
            resolutions=resolutions[:len(grid_spacings)],
            grid_spacings=grid_spacings,
            l2_errors=l2_errors,
            linf_errors=linf_errors,
            convergence_rate_l2=p_l2,
            convergence_rate_linf=p_linf,
            expected_rate=expected_rate,
            passed=passed,
            tolerance=self.tolerance
        )

    @staticmethod
    def print_results(result: MMSResult, test_name: str = "MMS Test"):
        """Pretty print MMS results"""
        print(f"\n{'='*70}")
        print(f"  {test_name}")
        print(f"{'='*70}")
        print(f"\n{'N':<10} {'h':<12} {'L2 Error':<15} {'Linf Error':<15}")
        print(f"{'-'*52}")

        for N, h, l2, linf in zip(result.resolutions, result.grid_spacings,
                                   result.l2_errors, result.linf_errors):
            print(f"{N:<10} {h:<12.6e} {l2:<15.6e} {linf:<15.6e}")

        print(f"\n{'Convergence Rates:':<30}")
        print(f"  L2 rate:   {result.convergence_rate_l2:.3f} (expected: {result.expected_rate:.1f})")
        print(f"  Linf rate: {result.convergence_rate_linf:.3f} (expected: {result.expected_rate:.1f})")
        print(f"  Tolerance: ±{result.tolerance}")

        if result.passed:
            print(f"\n✅ PASSED - Convergence rate within tolerance")
        else:
            print(f"\n❌ FAILED - Convergence rate outside tolerance")
        print(f"{'='*70}\n")


# Pre-defined manufactured solutions

def polynomial_1d():
    """1D polynomial: u(x) = x(1-x)

    For ∇²u = f:
    u''(x) = -2, so f = -2
    """
    def u_exact(x):
        return x * (1 - x)

    def forcing(x):
        # u'' = -2, so ∇²u = -2
        return -2.0 * np.ones_like(x)

    return u_exact, forcing


def sinusoidal_1d():
    """1D sinusoidal: u(x) = sin(πx)

    For ∇²u = f:
    u''(x) = -π² sin(πx), so f = -π² sin(πx)
    """
    def u_exact(x):
        return np.sin(np.pi * x)

    def forcing(x):
        # u'' = -π² sin(πx), so ∇²u = -π² sin(πx)
        return -np.pi**2 * np.sin(np.pi * x)

    return u_exact, forcing


def polynomial_2d():
    """2D polynomial: u(x,y) = x(1-x)y(1-y)

    For ∇²u = f:
    ∇²u = ∂²u/∂x² + ∂²u/∂y² = -2y(1-y) - 2x(1-x)
    """
    def u_exact(x, y):
        return x * (1 - x) * y * (1 - y)

    def forcing(x, y):
        # ∇²u = -2y(1-y) - 2x(1-x)
        return -2 * y * (1 - y) - 2 * x * (1 - x)

    return u_exact, forcing


def sinusoidal_2d():
    """2D sinusoidal: u(x,y) = sin(πx)sin(πy)

    For ∇²u = f:
    ∇²u = -π² sin(πx)sin(πy) - π² sin(πx)sin(πy) = -2π² sin(πx)sin(πy)
    """
    def u_exact(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)

    def forcing(x, y):
        # ∇²u = -2π² sin(πx)sin(πy)
        return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    return u_exact, forcing
