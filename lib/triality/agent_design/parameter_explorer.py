"""
Parameter space exploration and Design of Experiments (DOE).

Advanced sampling strategies for efficient design space coverage.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import itertools


class SamplingMethod(Enum):
    """Sampling methods for parameter exploration"""
    LATIN_HYPERCUBE = "latin_hypercube"  # LHS
    SOBOL = "sobol"  # Quasi-random low-discrepancy
    RANDOM = "random"  # Pure random (Monte Carlo)
    GRID = "grid"  # Regular grid
    ADAPTIVE = "adaptive"  # Adaptive refinement


@dataclass
class ParameterSpace:
    """
    Multi-dimensional parameter space definition.

    Supports continuous, discrete, and categorical parameters.
    """
    continuous_params: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # Format: {name: (min, max)}

    discrete_params: Dict[str, List[float]] = field(default_factory=dict)
    # Format: {name: [allowed_values]}

    categorical_params: Dict[str, List[str]] = field(default_factory=dict)
    # Format: {name: [categories]}

    def add_continuous(self, name: str, min_val: float, max_val: float):
        """Add continuous parameter"""
        self.continuous_params[name] = (min_val, max_val)

    def add_discrete(self, name: str, values: List[float]):
        """Add discrete parameter"""
        self.discrete_params[name] = values

    def add_categorical(self, name: str, categories: List[str]):
        """Add categorical parameter"""
        self.categorical_params[name] = categories

    @property
    def n_continuous(self) -> int:
        return len(self.continuous_params)

    @property
    def n_discrete(self) -> int:
        return len(self.discrete_params)

    @property
    def n_categorical(self) -> int:
        return len(self.categorical_params)

    @property
    def dimensionality(self) -> int:
        """Total number of parameters"""
        return self.n_continuous + self.n_discrete + self.n_categorical


class DOE:
    """
    Design of Experiments (DOE) generator.

    Classical DOE methods:
    - Full factorial
    - Fractional factorial
    - Taguchi orthogonal arrays
    - Central composite design (CCD)
    """

    @staticmethod
    def full_factorial(factors: Dict[str, List[float]]) -> List[Dict[str, float]]:
        """
        Generate full factorial design.

        For k factors with n_i levels each, generates ∏n_i experiments.

        Args:
            factors: Dictionary of factor names and their levels

        Returns:
            List of experiment parameter sets
        """
        factor_names = list(factors.keys())
        factor_levels = [factors[name] for name in factor_names]

        designs = []
        for combination in itertools.product(*factor_levels):
            design = dict(zip(factor_names, combination))
            designs.append(design)

        return designs

    @staticmethod
    def fractional_factorial_2level(n_factors: int, resolution: int = 4) -> np.ndarray:
        """
        Generate fractional factorial design for 2-level factors.

        Resolution III: Main effects not confounded with each other
        Resolution IV: Main effects not confounded with 2-factor interactions
        Resolution V: Main effects and 2-factor interactions not confounded

        Args:
            n_factors: Number of factors
            resolution: Design resolution (III, IV, or V)

        Returns:
            Design matrix (n_runs × n_factors) with -1/+1 levels
        """
        # Number of runs for given resolution
        if resolution >= 5:
            n_runs = 2 ** (n_factors - 1)
        elif resolution >= 4:
            n_runs = 2 ** (n_factors - 2) if n_factors > 3 else 2 ** n_factors
        else:  # Resolution III
            n_runs = 2 ** (n_factors - 3) if n_factors > 4 else 2 ** n_factors

        n_runs = max(n_runs, 4)  # Minimum 4 runs

        # Generate base design (full factorial for fewer factors)
        n_base = int(np.log2(n_runs))
        design_base = DOE._full_factorial_2level_matrix(n_base)

        # Add confounded factors using generator columns
        design = np.zeros((n_runs, n_factors))
        design[:, :n_base] = design_base

        for i in range(n_base, n_factors):
            # Simple confounding: multiply columns (can be more sophisticated)
            # E.g., column i = column (i % n_base) * column ((i+1) % n_base)
            col1 = i % n_base
            col2 = (i + 1) % n_base
            design[:, i] = design[:, col1] * design[:, col2]

        return design

    @staticmethod
    def _full_factorial_2level_matrix(n_factors: int) -> np.ndarray:
        """Generate full factorial matrix for 2-level factors"""
        n_runs = 2 ** n_factors
        design = np.zeros((n_runs, n_factors))

        for i in range(n_factors):
            # Alternating pattern: +1, -1, +1, -1, ...
            period = 2 ** (n_factors - i - 1)
            pattern = np.array([1, -1])
            design[:, i] = np.tile(np.repeat(pattern, period), 2 ** i)

        return design

    @staticmethod
    def central_composite_design(n_factors: int, alpha: Optional[float] = None) -> np.ndarray:
        """
        Generate Central Composite Design (CCD) for response surface modeling.

        CCD consists of:
        1. Factorial points (2^k or fractional factorial)
        2. Axial points (2k points at ±α on each axis)
        3. Center point(s)

        Args:
            n_factors: Number of factors
            alpha: Axial distance (None = face-centered, sqrt(k) = rotatable)

        Returns:
            Design matrix (n_runs × n_factors)
        """
        if alpha is None:
            alpha = 1.0  # Face-centered CCD

        # 1. Factorial points (2^k)
        n_factorial = 2 ** n_factors
        factorial = DOE._full_factorial_2level_matrix(n_factors)

        # 2. Axial points (2k)
        axial = np.zeros((2 * n_factors, n_factors))
        for i in range(n_factors):
            axial[2*i, i] = alpha
            axial[2*i + 1, i] = -alpha

        # 3. Center point
        center = np.zeros((1, n_factors))

        # Combine
        design = np.vstack([factorial, axial, center])

        return design

    @staticmethod
    def taguchi_L8(n_factors: int = 7) -> np.ndarray:
        """
        Taguchi L8 orthogonal array (2^7 design in 8 runs).

        Can accommodate up to 7 two-level factors.
        """
        if n_factors > 7:
            raise ValueError("L8 array supports max 7 factors")

        # Standard L8 array
        L8 = np.array([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 2, 2, 2, 2],
            [1, 2, 2, 1, 1, 2, 2],
            [1, 2, 2, 2, 2, 1, 1],
            [2, 1, 2, 1, 2, 1, 2],
            [2, 1, 2, 2, 1, 2, 1],
            [2, 2, 1, 1, 2, 2, 1],
            [2, 2, 1, 2, 1, 1, 2]
        ])

        # Return only needed columns
        return L8[:, :n_factors]


class ParameterExplorer:
    """
    Explore parameter space using various sampling strategies.

    Provides adaptive sampling and importance-based refinement.
    """

    def __init__(self, param_space: ParameterSpace):
        self.param_space = param_space
        self.samples: List[Dict[str, float]] = []

    def latin_hypercube_sample(self, n_samples: int, seed: Optional[int] = None) -> List[Dict[str, float]]:
        """
        Generate Latin Hypercube samples for continuous parameters.

        Args:
            n_samples: Number of samples
            seed: Random seed

        Returns:
            List of parameter dictionaries
        """
        if seed is not None:
            np.random.seed(seed)

        param_names = list(self.param_space.continuous_params.keys())
        n_params = len(param_names)

        if n_params == 0:
            return []

        # LHS matrix [0, 1]^n
        lhs = np.zeros((n_samples, n_params))
        for i in range(n_params):
            intervals = np.arange(n_samples)
            shuffled = np.random.permutation(intervals)
            uniform = np.random.uniform(0, 1, n_samples)
            lhs[:, i] = (shuffled + uniform) / n_samples

        # Map to parameter ranges
        samples = []
        for i in range(n_samples):
            sample = {}
            for j, param_name in enumerate(param_names):
                min_val, max_val = self.param_space.continuous_params[param_name]
                sample[param_name] = min_val + lhs[i, j] * (max_val - min_val)
            samples.append(sample)

        self.samples.extend(samples)
        return samples

    def sobol_sequence(self, n_samples: int) -> List[Dict[str, float]]:
        """
        Generate Sobol quasi-random sequence (low-discrepancy).

        Better space-filling than random sampling.

        Args:
            n_samples: Number of samples

        Returns:
            List of parameter dictionaries
        """
        param_names = list(self.param_space.continuous_params.keys())
        n_params = len(param_names)

        if n_params == 0:
            return []

        # Simplified Sobol sequence (for production, use scipy.stats.qmc.Sobol)
        # Here we use van der Corput sequences for each dimension
        sobol_matrix = np.zeros((n_samples, n_params))
        for dim in range(n_params):
            sobol_matrix[:, dim] = self._van_der_corput_sequence(n_samples, base=2 + dim)

        # Map to parameter ranges
        samples = []
        for i in range(n_samples):
            sample = {}
            for j, param_name in enumerate(param_names):
                min_val, max_val = self.param_space.continuous_params[param_name]
                sample[param_name] = min_val + sobol_matrix[i, j] * (max_val - min_val)
            samples.append(sample)

        self.samples.extend(samples)
        return samples

    @staticmethod
    def _van_der_corput_sequence(n: int, base: int = 2) -> np.ndarray:
        """Generate van der Corput sequence (1D low-discrepancy)"""
        sequence = np.zeros(n)
        for i in range(n):
            n_i = i
            denom = 1.0
            while n_i > 0:
                denom *= base
                remainder = n_i % base
                n_i //= base
                sequence[i] += remainder / denom
        return sequence

    def grid_sample(self, n_points_per_dim: int) -> List[Dict[str, float]]:
        """
        Generate regular grid samples.

        Args:
            n_points_per_dim: Number of points along each dimension

        Returns:
            List of parameter dictionaries
        """
        param_names = list(self.param_space.continuous_params.keys())
        n_params = len(param_names)

        if n_params == 0:
            return []

        # Create 1D arrays for each parameter
        param_arrays = []
        for param_name in param_names:
            min_val, max_val = self.param_space.continuous_params[param_name]
            param_arrays.append(np.linspace(min_val, max_val, n_points_per_dim))

        # Create meshgrid
        grids = np.meshgrid(*param_arrays, indexing='ij')

        # Flatten and create samples
        n_samples = n_points_per_dim ** n_params
        samples = []
        for i in range(n_samples):
            sample = {}
            for j, param_name in enumerate(param_names):
                idx = np.unravel_index(i, [n_points_per_dim] * n_params)
                sample[param_name] = grids[j][idx]
            samples.append(sample)

        self.samples.extend(samples)
        return samples

    def adaptive_sample(self, initial_samples: int, refinement_samples: int,
                       metric_function, n_refinement_regions: int = 5) -> List[Dict[str, float]]:
        """
        Adaptive sampling: focus on regions of interest.

        Args:
            initial_samples: Number of initial samples (LHS)
            refinement_samples: Additional samples in interesting regions
            metric_function: Function to evaluate "interest" (e.g., gradient magnitude)
            n_refinement_regions: Number of regions to refine

        Returns:
            Combined list of initial + refined samples
        """
        # Initial sampling
        initial = self.latin_hypercube_sample(initial_samples)

        # Evaluate metric for each sample
        metrics = [metric_function(sample) for sample in initial]

        # Find top regions to refine
        sorted_indices = np.argsort(metrics)[-n_refinement_regions:]
        refinement_centers = [initial[i] for i in sorted_indices]

        # Add samples around these centers
        param_names = list(self.param_space.continuous_params.keys())
        refined_samples = []

        for center in refinement_centers:
            # Sample locally around center (10% of range)
            for _ in range(refinement_samples // n_refinement_regions):
                sample = {}
                for param_name in param_names:
                    min_val, max_val = self.param_space.continuous_params[param_name]
                    range_val = max_val - min_val
                    local_perturbation = np.random.uniform(-0.05, 0.05) * range_val
                    sample[param_name] = np.clip(
                        center[param_name] + local_perturbation,
                        min_val, max_val
                    )
                refined_samples.append(sample)

        self.samples.extend(refined_samples)
        return initial + refined_samples

    def coverage_metric(self) -> float:
        """
        Calculate space-filling quality metric (average minimum distance).

        Higher is better.
        """
        if len(self.samples) < 2:
            return 0.0

        # Normalize samples to [0, 1]^n
        param_names = list(self.param_space.continuous_params.keys())
        normalized = np.zeros((len(self.samples), len(param_names)))

        for i, sample in enumerate(self.samples):
            for j, param_name in enumerate(param_names):
                min_val, max_val = self.param_space.continuous_params[param_name]
                normalized[i, j] = (sample[param_name] - min_val) / (max_val - min_val)

        # Compute minimum distances
        min_distances = []
        for i in range(len(normalized)):
            distances = np.linalg.norm(normalized - normalized[i], axis=1)
            distances[i] = np.inf  # Exclude self
            min_distances.append(np.min(distances))

        return np.mean(min_distances)
