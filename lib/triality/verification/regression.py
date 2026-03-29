"""
Regression Benchmarks

Maintains a suite of known solutions to prevent regressions. When someone
asks "how do I know this is correct?", point them to this directory.

Reference:
- NIST PDE Benchmark Problems: https://math.nist.gov/
- Roache (1998). "Verification and Validation in Computational Science"
"""

import numpy as np
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    name: str
    dimension: int
    resolution: int
    l2_norm: float
    linf_norm: float
    residual: float
    iterations: int
    converged: bool
    checksum: str


@dataclass
class RegressionReport:
    """Results from regression testing"""
    total_tests: int
    passed: int
    failed: int
    new: int
    changed: List[str]
    results: List[BenchmarkResult]


class RegressionBenchmark:
    """Regression testing against known solutions"""

    def __init__(self, benchmark_file: str = "benchmarks.json"):
        """
        Args:
            benchmark_file: Path to store benchmark results
        """
        self.benchmark_file = Path(benchmark_file)
        self.benchmarks: Dict[str, BenchmarkResult] = {}

        # Load existing benchmarks
        if self.benchmark_file.exists():
            self.load_benchmarks()

    def load_benchmarks(self):
        """Load benchmarks from file"""
        try:
            with open(self.benchmark_file, 'r') as f:
                data = json.load(f)
                for name, result_dict in data.items():
                    self.benchmarks[name] = BenchmarkResult(**result_dict)
        except Exception as e:
            print(f"Warning: Could not load benchmarks: {e}")

    def save_benchmarks(self):
        """Save benchmarks to file"""
        data = {name: asdict(result) for name, result in self.benchmarks.items()}
        with open(self.benchmark_file, 'w') as f:
            json.dump(data, f, indent=2)

    def compute_checksum(self, u: np.ndarray) -> str:
        """
        Compute checksum of solution array

        Args:
            u: Solution array

        Returns:
            SHA256 checksum
        """
        # Convert to bytes and hash
        u_bytes = u.tobytes()
        checksum = hashlib.sha256(u_bytes).hexdigest()[:16]
        return checksum

    def add_benchmark(self,
                     name: str,
                     dimension: int,
                     resolution: int,
                     u: np.ndarray,
                     residual: float,
                     iterations: int,
                     converged: bool,
                     overwrite: bool = False):
        """
        Add a new benchmark result

        Args:
            name: Benchmark name
            dimension: Problem dimension
            resolution: Grid resolution
            u: Solution array
            residual: Final residual
            iterations: Number of iterations
            converged: Whether solver converged
            overwrite: Whether to overwrite existing benchmark
        """
        if name in self.benchmarks and not overwrite:
            print(f"Warning: Benchmark '{name}' already exists. Use overwrite=True.")
            return

        # Compute norms
        l2_norm = np.linalg.norm(u)
        linf_norm = np.max(np.abs(u))
        checksum = self.compute_checksum(u)

        result = BenchmarkResult(
            name=name,
            dimension=dimension,
            resolution=resolution,
            l2_norm=l2_norm,
            linf_norm=linf_norm,
            residual=residual,
            iterations=iterations,
            converged=converged,
            checksum=checksum
        )

        self.benchmarks[name] = result
        self.save_benchmarks()

        print(f"✓ Added benchmark: {name}")

    def verify_benchmark(self,
                        name: str,
                        u: np.ndarray,
                        residual: float,
                        iterations: int,
                        converged: bool,
                        tolerance: float = 1e-10) -> Tuple[bool, str]:
        """
        Verify against existing benchmark

        Args:
            name: Benchmark name
            u: Current solution
            residual: Current residual
            iterations: Current iterations
            converged: Whether current solver converged
            tolerance: Relative tolerance for comparison

        Returns:
            (passed, message)
        """
        if name not in self.benchmarks:
            return False, f"Benchmark '{name}' not found"

        ref = self.benchmarks[name]

        # Compute current norms
        l2_norm = np.linalg.norm(u)
        linf_norm = np.max(np.abs(u))
        checksum = self.compute_checksum(u)

        # Check norms
        l2_diff = abs(l2_norm - ref.l2_norm) / (ref.l2_norm + 1e-16)
        linf_diff = abs(linf_norm - ref.linf_norm) / (ref.linf_norm + 1e-16)

        passed = True
        messages = []

        if l2_diff > tolerance:
            passed = False
            messages.append(f"L2 norm differs: {l2_diff:.2e} (tol: {tolerance:.2e})")

        if linf_diff > tolerance:
            passed = False
            messages.append(f"Linf norm differs: {linf_diff:.2e} (tol: {tolerance:.2e})")

        if checksum != ref.checksum:
            passed = False
            messages.append(f"Checksum differs: {checksum} vs {ref.checksum}")

        if converged != ref.converged:
            passed = False
            messages.append(f"Convergence differs: {converged} vs {ref.converged}")

        if passed:
            return True, "✓ Matches benchmark"
        else:
            return False, "✗ " + "; ".join(messages)

    def run_regression_tests(self,
                            test_results: Dict[str, Tuple[np.ndarray, float, int, bool]],
                            tolerance: float = 1e-10) -> RegressionReport:
        """
        Run regression tests on multiple results

        Args:
            test_results: Dict mapping test names to (u, residual, iterations, converged)
            tolerance: Relative tolerance

        Returns:
            RegressionReport
        """
        total = len(test_results)
        passed = 0
        failed = 0
        new = 0
        changed = []
        results = []

        for name, (u, residual, iterations, converged) in test_results.items():
            if name in self.benchmarks:
                # Verify existing benchmark
                test_passed, message = self.verify_benchmark(
                    name, u, residual, iterations, converged, tolerance)

                if test_passed:
                    passed += 1
                    print(f"✓ {name}: {message}")
                else:
                    failed += 1
                    changed.append(name)
                    print(f"✗ {name}: {message}")

                # Store result
                l2_norm = np.linalg.norm(u)
                linf_norm = np.max(np.abs(u))
                checksum = self.compute_checksum(u)

                # Get dimension and resolution from stored benchmark
                ref = self.benchmarks[name]
                result = BenchmarkResult(
                    name=name,
                    dimension=ref.dimension,
                    resolution=ref.resolution,
                    l2_norm=l2_norm,
                    linf_norm=linf_norm,
                    residual=residual,
                    iterations=iterations,
                    converged=converged,
                    checksum=checksum
                )
                results.append(result)

            else:
                # New benchmark
                new += 1
                print(f"ℹ {name}: New benchmark (not verified)")

        return RegressionReport(
            total_tests=total,
            passed=passed,
            failed=failed,
            new=new,
            changed=changed,
            results=results
        )

    @staticmethod
    def print_report(report: RegressionReport):
        """Pretty print regression report"""
        print(f"\n{'='*70}")
        print(f"  Regression Test Report")
        print(f"{'='*70}")
        print(f"\n  Total Tests: {report.total_tests}")
        print(f"  ✓ Passed:    {report.passed}")
        print(f"  ✗ Failed:    {report.failed}")
        print(f"  ℹ New:       {report.new}")

        if report.changed:
            print(f"\n  Changed benchmarks:")
            for name in report.changed:
                print(f"    - {name}")

        overall_passed = report.failed == 0
        if overall_passed:
            print(f"\n✅ ALL REGRESSION TESTS PASSED")
        else:
            print(f"\n❌ SOME REGRESSION TESTS FAILED")
        print(f"{'='*70}\n")


# Pre-defined benchmark problems

BENCHMARK_PROBLEMS = {
    "1d_poisson_constant": {
        "description": "1D Poisson with constant forcing",
        "equation": "-u'' = 1",
        "domain": "[0, 1]",
        "bc": "u(0) = u(1) = 0",
        "exact": "u(x) = x(1-x)/2"
    },
    "1d_poisson_sine": {
        "description": "1D Poisson with sinusoidal forcing",
        "equation": "-u'' = π²sin(πx)",
        "domain": "[0, 1]",
        "bc": "u(0) = u(1) = 0",
        "exact": "u(x) = sin(πx)"
    },
    "2d_poisson_constant": {
        "description": "2D Poisson with constant forcing",
        "equation": "-∇²u = 2",
        "domain": "[0, 1]²",
        "bc": "u = 0 on ∂Ω",
        "exact": "u(x,y) = (analytical not simple)"
    },
    "2d_poisson_sine": {
        "description": "2D Poisson with sinusoidal forcing",
        "equation": "-∇²u = 2π²sin(πx)sin(πy)",
        "domain": "[0, 1]²",
        "bc": "u = 0 on ∂Ω",
        "exact": "u(x,y) = sin(πx)sin(πy)"
    },
}
