"""
Grid Convergence Verification

Verifies that numerical solutions converge to exact solutions as grid spacing
decreases. Computes observed order of accuracy and compares with theory.

Reference:
- Oberkampf & Roy (2010). "Verification and Validation in Scientific Computing"
- Roache (1998). "Verification of Codes and Calculations"
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


@dataclass
class ConvergenceResult:
    """Results from grid convergence study"""
    resolutions: List[int]
    grid_spacings: List[float]
    errors: List[float]
    observed_order: float
    theoretical_order: float
    asymptotic_error_constant: float
    passed: bool
    tolerance: float


class GridConvergenceTest:
    """Grid convergence verification"""

    def __init__(self, tolerance: float = 0.2):
        """
        Args:
            tolerance: Allowable deviation from theoretical order
        """
        self.tolerance = tolerance

    def compute_convergence_rate(self,
                                  resolutions: List[int],
                                  errors: List[float]) -> Tuple[float, float]:
        """
        Compute convergence rate from error data

        Uses least-squares fit: log(error) = log(C) + p*log(h)
        where p is the order of accuracy

        Args:
            resolutions: Grid resolutions
            errors: Errors at each resolution

        Returns:
            (observed_order, error_constant)
        """
        # Check if errors are at machine precision
        machine_eps = 1e-14
        if max(errors) < machine_eps:
            # Errors at machine precision - solver is perfect!
            return 2.0, min(errors)  # Return expected order

        # Compute grid spacings (assuming uniform grid on [0,1])
        h = [1.0 / (N - 1) for N in resolutions]

        # Log-log fit
        log_h = np.log(h)
        log_err = np.log(errors)

        # Fit: log(error) = log(C) + p*log(h)
        coeffs = np.polyfit(log_h, log_err, 1)
        observed_order = coeffs[0]
        log_C = coeffs[1]
        error_constant = np.exp(log_C)

        return observed_order, error_constant

    def verify_convergence(self,
                           resolutions: List[int],
                           errors: List[float],
                           theoretical_order: float = 2.0) -> ConvergenceResult:
        """
        Verify that errors converge at expected rate

        Args:
            resolutions: Grid resolutions tested
            errors: Errors at each resolution
            theoretical_order: Expected order of accuracy

        Returns:
            ConvergenceResult with analysis
        """
        observed_order, error_constant = self.compute_convergence_rate(
            resolutions, errors)

        # Check if within tolerance
        passed = abs(observed_order - theoretical_order) < self.tolerance

        h = [1.0 / (N - 1) for N in resolutions]

        return ConvergenceResult(
            resolutions=resolutions,
            grid_spacings=h,
            errors=errors,
            observed_order=observed_order,
            theoretical_order=theoretical_order,
            asymptotic_error_constant=error_constant,
            passed=passed,
            tolerance=self.tolerance
        )

    def plot_convergence(self,
                        result: ConvergenceResult,
                        filename: str = 'convergence.png',
                        title: str = 'Grid Convergence Study'):
        """
        Create convergence plot

        Args:
            result: ConvergenceResult to plot
            filename: Output filename
            title: Plot title
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Log-log convergence
        h = result.grid_spacings
        errors = result.errors

        ax1.loglog(h, errors, 'bo-', markersize=8, linewidth=2, label='Observed')

        # Plot theoretical line
        C = result.asymptotic_error_constant
        p_obs = result.observed_order
        p_theory = result.theoretical_order

        h_fit = np.array([min(h), max(h)])
        error_theory = C * h_fit**p_theory

        ax1.loglog(h_fit, error_theory, 'r--', linewidth=2,
                   label=f'Theory (p={p_theory:.1f})')

        ax1.set_xlabel('Grid Spacing (h)', fontsize=12)
        ax1.set_ylabel('Error', fontsize=12)
        ax1.set_title(f'{title}\nObserved Order: {p_obs:.3f}', fontsize=13)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Right plot: Error vs resolution
        N = result.resolutions
        ax2.semilogy(N, errors, 'go-', markersize=8, linewidth=2)
        ax2.set_xlabel('Grid Resolution (N)', fontsize=12)
        ax2.set_ylabel('Error', fontsize=12)
        ax2.set_title('Error vs Resolution', fontsize=13)
        ax2.grid(True, alpha=0.3)

        # Add pass/fail indicator
        status = "✅ PASSED" if result.passed else "❌ FAILED"
        fig.text(0.5, 0.02, status, ha='center', fontsize=14,
                 weight='bold',
                 color='green' if result.passed else 'red')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Convergence plot saved to: {filename}")

    @staticmethod
    def print_results(result: ConvergenceResult, test_name: str = "Convergence Test"):
        """Pretty print convergence results"""
        print(f"\n{'='*70}")
        print(f"  {test_name}")
        print(f"{'='*70}")
        print(f"\n{'N':<10} {'h':<15} {'Error':<15} {'Ratio':<12}")
        print(f"{'-'*52}")

        for i, (N, h, err) in enumerate(zip(result.resolutions,
                                             result.grid_spacings,
                                             result.errors)):
            if i > 0:
                ratio = result.errors[i-1] / err
                print(f"{N:<10} {h:<15.6e} {err:<15.6e} {ratio:<12.3f}")
            else:
                print(f"{N:<10} {h:<15.6e} {err:<15.6e} {'—':<12}")

        print(f"\n{'Analysis:':<30}")
        print(f"  Observed order:    {result.observed_order:.3f}")
        print(f"  Theoretical order: {result.theoretical_order:.1f}")
        print(f"  Difference:        {abs(result.observed_order - result.theoretical_order):.3f}")
        print(f"  Tolerance:         ±{result.tolerance}")
        print(f"  Error constant:    {result.asymptotic_error_constant:.3e}")

        if result.passed:
            print(f"\n✅ PASSED - Convergence order within tolerance")
        else:
            print(f"\n❌ FAILED - Convergence order outside tolerance")
        print(f"{'='*70}\n")


class RichardsonExtrapolation:
    """Richardson extrapolation for error estimation"""

    @staticmethod
    def extrapolate(f_coarse: float,
                    f_fine: float,
                    f_finer: float,
                    refinement_ratio: float = 2.0) -> Tuple[float, float]:
        """
        Compute Richardson extrapolation

        Args:
            f_coarse: Solution on coarse grid
            f_fine: Solution on fine grid
            f_finer: Solution on finer grid
            refinement_ratio: Grid refinement ratio (usually 2)

        Returns:
            (extrapolated_value, estimated_order)
        """
        r = refinement_ratio

        # Estimate order of accuracy
        if f_fine != f_coarse:
            p = np.log((f_finer - f_fine) / (f_fine - f_coarse)) / np.log(r)
        else:
            p = 0.0

        # Extrapolated value
        f_extrap = f_fine + (f_fine - f_coarse) / (r**p - 1)

        return f_extrap, p

    @staticmethod
    def compute_gci(f_coarse: float,
                    f_fine: float,
                    refinement_ratio: float = 2.0,
                    order: float = 2.0,
                    safety_factor: float = 1.25) -> float:
        """
        Compute Grid Convergence Index (GCI)

        The GCI is a standardized measure of discretization error

        Args:
            f_coarse: Solution on coarse grid
            f_fine: Solution on fine grid
            refinement_ratio: Grid refinement ratio
            order: Order of accuracy
            safety_factor: Safety factor (1.25 for 3+ grids, 3.0 for 2 grids)

        Returns:
            GCI value
        """
        r = refinement_ratio
        epsilon = abs((f_fine - f_coarse) / f_fine) if f_fine != 0 else 0
        gci = safety_factor * epsilon / (r**order - 1)
        return gci
