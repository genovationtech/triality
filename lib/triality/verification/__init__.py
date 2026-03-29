"""
Triality Verification Suite

Production-grade verification tools for ensuring solver correctness.
"""

from .mms import MMSTest
from .convergence import GridConvergenceTest
from .conservation import ConservationTest
from .regression import RegressionBenchmark

__all__ = ['MMSTest', 'GridConvergenceTest', 'ConservationTest', 'RegressionBenchmark']
