"""
Production-Grade Verification Test Suite

This is the test suite that proves correctness. When someone asks
"how do I know this is correct?", point them here.

Run with:
    python test_verification.py
    python test_verification.py --save-benchmarks
    python test_verification.py --plots
"""

import sys
import os
import argparse
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from triality import Field, Eq, laplacian, Interval, Rectangle, Square, solve
from triality.verification.mms import MMSTest, polynomial_1d, sinusoidal_1d, polynomial_2d, sinusoidal_2d
from triality.verification.convergence import GridConvergenceTest
from triality.verification.conservation import ConservationTest
from triality.verification.regression import RegressionBenchmark


class ProductionTestSuite:
    """Production-grade verification test suite"""

    def __init__(self, save_plots: bool = False, save_benchmarks: bool = False):
        self.save_plots = save_plots
        self.save_benchmarks = save_benchmarks
        self.results = {
            'mms': [],
            'convergence': [],
            'conservation': [],
            'regression': None
        }

    def run_all_tests(self):
        """Run complete verification suite"""
        print("\n" + "="*70)
        print("  TRIALITY PRODUCTION VERIFICATION SUITE")
        print("="*70)

        # 1. Method of Manufactured Solutions
        print("\n" + "="*70)
        print("  PART 1: METHOD OF MANUFACTURED SOLUTIONS")
        print("="*70)
        self.run_mms_tests()

        # 2. Grid Convergence
        print("\n" + "="*70)
        print("  PART 2: GRID CONVERGENCE ANALYSIS")
        print("="*70)
        self.run_convergence_tests()

        # 3. Conservation
        print("\n" + "="*70)
        print("  PART 3: CONSERVATION CHECKS")
        print("="*70)
        self.run_conservation_tests()

        # 4. Regression Benchmarks
        print("\n" + "="*70)
        print("  PART 4: REGRESSION BENCHMARKS")
        print("="*70)
        self.run_regression_tests()

        # Summary
        self.print_summary()

    def run_mms_tests(self):
        """Run MMS verification tests"""
        mms = MMSTest(tolerance=0.15)  # Allow 15% deviation in convergence rate

        # Test 1: 1D Polynomial
        print("\n[1/4] MMS Test: 1D Polynomial")
        u_exact, forcing = polynomial_1d()
        result = mms.verify_1d_poisson(
            u_exact, forcing,
            domain=(0, 1),
            bc_left=0.0,
            bc_right=0.0,
            resolutions=[20, 40, 80, 160]
        )
        mms.print_results(result, "MMS: 1D Polynomial u(x)=x(1-x)")
        self.results['mms'].append(('1D Polynomial', result.passed))

        # Test 2: 1D Sinusoidal
        print("\n[2/4] MMS Test: 1D Sinusoidal")
        u_exact, forcing = sinusoidal_1d()
        result = mms.verify_1d_poisson(
            u_exact, forcing,
            domain=(0, 1),
            bc_left=0.0,
            bc_right=0.0,
            resolutions=[20, 40, 80, 160]
        )
        mms.print_results(result, "MMS: 1D Sinusoidal u(x)=sin(πx)")
        self.results['mms'].append(('1D Sinusoidal', result.passed))

        # Test 3: 2D Polynomial
        print("\n[3/4] MMS Test: 2D Polynomial")
        u_exact, forcing = polynomial_2d()
        result = mms.verify_2d_poisson(
            u_exact, forcing,
            domain=(0, 1, 0, 1),
            bc_value=0.0,
            resolutions=[10, 20, 40]  # 2D is expensive
        )
        mms.print_results(result, "MMS: 2D Polynomial u(x,y)=x(1-x)y(1-y)")
        self.results['mms'].append(('2D Polynomial', result.passed))

        # Test 4: 2D Sinusoidal
        print("\n[4/4] MMS Test: 2D Sinusoidal")
        u_exact, forcing = sinusoidal_2d()
        result = mms.verify_2d_poisson(
            u_exact, forcing,
            domain=(0, 1, 0, 1),
            bc_value=0.0,
            resolutions=[10, 20, 40]
        )
        mms.print_results(result, "MMS: 2D Sinusoidal u(x,y)=sin(πx)sin(πy)")
        self.results['mms'].append(('2D Sinusoidal', result.passed))

    def run_convergence_tests(self):
        """Run grid convergence tests"""
        conv = GridConvergenceTest(tolerance=0.2)

        # Test 1: 1D Convergence
        print("\n[1/2] Grid Convergence: 1D Poisson")
        resolutions = [20, 40, 80, 160, 320]
        errors = []

        for N in resolutions:
            u = Field("u")
            eq = Eq(laplacian(u), -1)
            interval = Interval(0, 1)
            bc = {'left': 0, 'right': 0}

            sol = solve(eq, interval, bc=bc, resolution=N, verbose=False)

            # Exact solution
            x = sol.grid
            u_exact = x * (1 - x) / 2

            # L2 error
            error = np.sqrt(np.mean((sol.u - u_exact)**2))
            errors.append(error)

        result = conv.verify_convergence(resolutions, errors, theoretical_order=2.0)
        conv.print_results(result, "1D Poisson Convergence")

        if self.save_plots:
            conv.plot_convergence(result, 'convergence_1d.png', '1D Poisson Convergence')

        self.results['convergence'].append(('1D Convergence', result.passed))

        # Test 2: 2D Convergence
        print("\n[2/2] Grid Convergence: 2D Poisson")
        resolutions = [10, 20, 40, 80]
        errors = []

        # Use manufactured sinusoidal solution
        def forcing_2d(x, y):
            return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

        for N in resolutions:
            u = Field("u")
            eq = Eq(laplacian(u), 0)  # Dummy equation
            square = Square(1.0)
            bc = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}

            sol = solve(eq, square, bc=bc, resolution=N, verbose=False,
                       forcing=forcing_2d)

            # Use sinusoidal solution for comparison
            x_grid, y_grid = sol.grid
            X, Y = np.meshgrid(x_grid, y_grid)
            u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)

            # Compare against exact manufactured solution
            error = np.sqrt(np.mean((sol.u - u_exact.flatten())**2))
            errors.append(error)

        result = conv.verify_convergence(resolutions, errors, theoretical_order=2.0)
        conv.print_results(result, "2D Poisson Convergence")

        if self.save_plots:
            conv.plot_convergence(result, 'convergence_2d.png', '2D Poisson Convergence')

        self.results['convergence'].append(('2D Convergence', result.passed))

    def run_conservation_tests(self):
        """Run conservation verification"""
        cons = ConservationTest(tolerance=0.1)

        # Test 1: 1D Integral Identity
        print("\n[1/3] Conservation: 1D Integral Identity")
        u = Field("u")
        eq = Eq(laplacian(u), -1)
        interval = Interval(0, 1)
        bc = {'left': 0, 'right': 0}
        sol = solve(eq, interval, bc=bc, resolution=100, verbose=False)

        # Use the actual forcing from the equation (constant -1)
        f = -np.ones_like(sol.u)
        result = cons.verify_integral_identity_1d(sol.u, f, sol.grid)
        cons.print_results(result)
        self.results['conservation'].append(('1D Integral Identity', result.passed))

        # Test 2: 2D Integral Identity (Gauss's Theorem)
        print("\n[2/3] Conservation: 2D Gauss's Theorem")
        u = Field("u")
        eq = Eq(laplacian(u), -2)
        square = Square(1.0)
        bc = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}
        sol = solve(eq, square, bc=bc, resolution=40, verbose=False)

        x_grid, y_grid = sol.grid
        # Use actual forcing from equation (constant -2)
        result = cons.verify_integral_identity_2d(sol.u, -2.0, x_grid, y_grid)
        cons.print_results(result)
        self.results['conservation'].append(('2D Gauss Theorem', result.passed))

        # Test 3: Maximum Principle
        print("\n[3/3] Conservation: Maximum Principle")
        # For ∇²u = -2 (which means -∇²u = 2, a positive forcing)
        # Maximum should be in interior, not on boundary
        # Actually, for ∇²u = -f with f > 0, minimum is in interior
        # Skip maximum principle test for now (need to clarify sign conventions)
        result = cons.verify_maximum_principle(sol.u, -2.0, (0.0, 0.0))
        cons.print_results(result)
        self.results['conservation'].append(('Maximum Principle', result.passed))

    def run_regression_tests(self):
        """Run regression benchmarks"""
        benchmark_file = os.path.join(os.path.dirname(__file__), "benchmarks.json")
        reg = RegressionBenchmark(benchmark_file)

        # Collect test results
        test_results = {}

        # 1D Poisson with constant forcing
        u = Field("u")
        eq = Eq(laplacian(u), -1)
        interval = Interval(0, 1)
        bc = {'left': 0, 'right': 0}
        sol = solve(eq, interval, bc=bc, resolution=100, verbose=False)
        test_results['1d_poisson_constant_N100'] = (
            sol.u, sol.residual, sol.iterations, sol.converged
        )

        # 2D Poisson with constant forcing
        u = Field("u")
        eq = Eq(laplacian(u), -2)
        square = Square(1.0)
        bc = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}
        sol = solve(eq, square, bc=bc, resolution=30, verbose=False)
        test_results['2d_poisson_constant_N30'] = (
            sol.u, sol.residual, sol.iterations, sol.converged
        )

        # If saving benchmarks, add them
        if self.save_benchmarks:
            print("\n📝 Saving new benchmarks...")
            for name, (u, res, iters, conv) in test_results.items():
                dim = 1 if '1d' in name else 2
                N = int(name.split('_N')[-1])
                reg.add_benchmark(name, dim, N, u, res, iters, conv, overwrite=True)
            print("✓ Benchmarks saved")

        # Run regression tests
        report = reg.run_regression_tests(test_results, tolerance=1e-10)
        reg.print_report(report)
        self.results['regression'] = report

    def print_summary(self):
        """Print final summary"""
        print("\n" + "="*70)
        print("  VERIFICATION SUMMARY")
        print("="*70)

        # MMS
        mms_passed = sum(1 for _, passed in self.results['mms'] if passed)
        mms_total = len(self.results['mms'])
        print(f"\nMethod of Manufactured Solutions: {mms_passed}/{mms_total} passed")
        for name, passed in self.results['mms']:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")

        # Convergence
        conv_passed = sum(1 for _, passed in self.results['convergence'] if passed)
        conv_total = len(self.results['convergence'])
        print(f"\nGrid Convergence: {conv_passed}/{conv_total} passed")
        for name, passed in self.results['convergence']:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")

        # Conservation
        cons_passed = sum(1 for _, passed in self.results['conservation'] if passed)
        cons_total = len(self.results['conservation'])
        print(f"\nConservation: {cons_passed}/{cons_total} passed")
        for name, passed in self.results['conservation']:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")

        # Regression
        if self.results['regression']:
            report = self.results['regression']
            print(f"\nRegression: {report.passed}/{report.total_tests} passed")
            if report.new > 0:
                print(f"  ℹ {report.new} new benchmarks")

        # Overall
        total_passed = mms_passed + conv_passed + cons_passed
        total_tests = mms_total + conv_total + cons_total
        if self.results['regression']:
            total_passed += report.passed
            total_tests += report.total_tests

        print(f"\n{'='*70}")
        print(f"  OVERALL: {total_passed}/{total_tests} tests passed")

        if total_passed == total_tests:
            print(f"\n  🎉🎉🎉 ALL VERIFICATION TESTS PASSED 🎉🎉🎉")
        else:
            print(f"\n  ⚠️  Some tests failed - investigation required")

        print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Triality Production Verification Suite")
    parser.add_argument('--plots', action='store_true',
                       help='Generate convergence plots')
    parser.add_argument('--save-benchmarks', action='store_true',
                       help='Save new regression benchmarks')
    args = parser.parse_args()

    suite = ProductionTestSuite(
        save_plots=args.plots,
        save_benchmarks=args.save_benchmarks
    )
    suite.run_all_tests()
