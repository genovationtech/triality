"""
Sensitivity analysis for understanding parameter importance.

Implements Morris screening and Sobol variance-based methods.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Tuple, Optional


@dataclass
class SensitivityResult:
    """Results from sensitivity analysis"""
    parameter_names: List[str]
    sensitivities: Dict[str, float]
    rankings: List[Tuple[str, float]]  # Sorted by sensitivity


class MorrisScreening:
    """
    Morris One-At-a-Time (OAT) screening method.

    Efficient method for identifying important parameters in high-dimensional spaces.

    Elementary Effects:
        EE_i = [f(x + Δe_i) - f(x)] / Δ

    where e_i is unit vector in dimension i.

    Metrics:
        μ* = mean of |EE_i| (overall importance)
        σ = std of EE_i (interaction effects / non-linearity)
    """

    def __init__(self, parameter_space: Dict[str, Tuple[float, float]],
                 model_function: Callable[[Dict[str, float]], float]):
        """
        Initialize Morris screening.

        Args:
            parameter_space: Dict of {param_name: (min, max)}
            model_function: Function to evaluate (takes parameter dict, returns scalar)
        """
        self.parameter_space = parameter_space
        self.model_function = model_function
        self.param_names = list(parameter_space.keys())
        self.n_params = len(self.param_names)

    def generate_trajectory(self, n_levels: int = 10, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate Morris trajectory (one-at-a-time path through parameter space).

        Returns:
            Trajectory matrix (n_params+1 × n_params)
        """
        if seed is not None:
            np.random.seed(seed)

        # Initialize at random starting point
        start = np.random.randint(0, n_levels, size=self.n_params)

        # Build trajectory
        trajectory = np.zeros((self.n_params + 1, self.n_params))
        trajectory[0, :] = start

        # Randomize parameter order
        param_order = np.random.permutation(self.n_params)

        # Step size (typically n_levels / 2)
        delta = n_levels // 2

        current = start.copy()
        for i, param_idx in enumerate(param_order):
            # Perturb one parameter
            direction = np.random.choice([-1, 1])
            current[param_idx] = np.clip(current[param_idx] + direction * delta, 0, n_levels - 1)
            trajectory[i + 1, :] = current

        return trajectory

    def trajectory_to_parameters(self, trajectory: np.ndarray, n_levels: int) -> List[Dict[str, float]]:
        """
        Convert normalized trajectory to actual parameter values.

        Args:
            trajectory: Trajectory in [0, n_levels-1]
            n_levels: Number of levels

        Returns:
            List of parameter dictionaries
        """
        n_points = trajectory.shape[0]
        param_dicts = []

        for i in range(n_points):
            params = {}
            for j, param_name in enumerate(self.param_names):
                min_val, max_val = self.parameter_space[param_name]
                # Map [0, n_levels-1] to [min, max]
                normalized = trajectory[i, j] / (n_levels - 1)
                params[param_name] = min_val + normalized * (max_val - min_val)
            param_dicts.append(params)

        return param_dicts

    def compute_elementary_effects(self, trajectory: np.ndarray, n_levels: int) -> np.ndarray:
        """
        Compute elementary effects along a trajectory.

        Returns:
            Elementary effects (n_params,)
        """
        # Convert to parameters
        param_list = self.trajectory_to_parameters(trajectory, n_levels)

        # Evaluate model at each point
        outputs = np.array([self.model_function(params) for params in param_list])

        # Compute elementary effects
        elementary_effects = np.zeros(self.n_params)

        for i in range(self.n_params):
            # Find which parameter changed at step i
            diff = trajectory[i + 1] - trajectory[i]
            changed_idx = np.argmax(np.abs(diff))

            # Elementary effect
            delta = diff[changed_idx] / (n_levels - 1)  # Normalized delta
            min_val, max_val = self.parameter_space[self.param_names[changed_idx]]
            actual_delta = delta * (max_val - min_val)

            if actual_delta != 0:
                elementary_effects[changed_idx] = (outputs[i + 1] - outputs[i]) / actual_delta
            else:
                elementary_effects[changed_idx] = 0.0

        return elementary_effects

    def analyze(self, n_trajectories: int = 10, n_levels: int = 10,
                seed: Optional[int] = None) -> SensitivityResult:
        """
        Perform Morris screening analysis.

        Args:
            n_trajectories: Number of trajectories (typically 10-50)
            n_levels: Grid levels (typically 4-10)
            seed: Random seed

        Returns:
            SensitivityResult with μ* rankings
        """
        if seed is not None:
            np.random.seed(seed)

        # Store elementary effects for each parameter
        all_effects = {name: [] for name in self.param_names}

        # Generate and analyze trajectories
        for traj_idx in range(n_trajectories):
            trajectory = self.generate_trajectory(n_levels, seed=seed+traj_idx if seed else None)
            effects = self.compute_elementary_effects(trajectory, n_levels)

            for i, param_name in enumerate(self.param_names):
                all_effects[param_name].append(effects[i])

        # Compute statistics
        sensitivities = {}
        for param_name in self.param_names:
            effects = np.array(all_effects[param_name])
            # μ* (mean of absolute values)
            mu_star = np.mean(np.abs(effects))
            sensitivities[param_name] = mu_star

        # Rank parameters
        rankings = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)

        return SensitivityResult(
            parameter_names=self.param_names,
            sensitivities=sensitivities,
            rankings=rankings
        )


class SobolIndices:
    """
    Sobol variance-based global sensitivity analysis.

    Sobol indices decompose output variance:
        Var(Y) = Σ V_i + Σ V_ij + ... + V_12...k

    First-order index: S_i = V_i / Var(Y)
        (fraction of variance due to X_i alone)

    Total-order index: S_Ti = 1 - V_~i / Var(Y)
        (total effect including interactions)

    More expensive than Morris (requires many more samples).
    """

    def __init__(self, parameter_space: Dict[str, Tuple[float, float]],
                 model_function: Callable[[Dict[str, float]], float]):
        """
        Initialize Sobol analysis.

        Args:
            parameter_space: Dict of {param_name: (min, max)}
            model_function: Function to evaluate
        """
        self.parameter_space = parameter_space
        self.model_function = model_function
        self.param_names = list(parameter_space.keys())
        self.n_params = len(self.param_names)

    def sample_sobol_matrices(self, n_samples: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Sobol sample matrices A and B.

        Args:
            n_samples: Number of base samples
            seed: Random seed

        Returns:
            (A, B) matrices (n_samples × n_params) in [0, 1]
        """
        if seed is not None:
            np.random.seed(seed)

        A = np.random.random((n_samples, self.n_params))
        B = np.random.random((n_samples, self.n_params))

        return A, B

    def matrices_to_parameters(self, matrix: np.ndarray) -> List[Dict[str, float]]:
        """
        Convert [0, 1]^n matrix to parameter dictionaries.

        Args:
            matrix: (n_samples × n_params) in [0, 1]

        Returns:
            List of parameter dictionaries
        """
        n_samples = matrix.shape[0]
        param_list = []

        for i in range(n_samples):
            params = {}
            for j, param_name in enumerate(self.param_names):
                min_val, max_val = self.parameter_space[param_name]
                params[param_name] = min_val + matrix[i, j] * (max_val - min_val)
            param_list.append(params)

        return param_list

    def compute_first_order_indices(self, n_samples: int = 1000,
                                    seed: Optional[int] = None) -> Dict[str, float]:
        """
        Compute first-order Sobol indices.

        S_i = V_i / Var(Y)

        where V_i = Var[E(Y|X_i)]

        Args:
            n_samples: Number of samples (larger = more accurate)
            seed: Random seed

        Returns:
            Dict of {param_name: S_i}
        """
        # Generate sample matrices
        A, B = self.sample_sobol_matrices(n_samples, seed)

        # Evaluate f(A) and f(B)
        f_A = np.array([self.model_function(p) for p in self.matrices_to_parameters(A)])
        f_B = np.array([self.model_function(p) for p in self.matrices_to_parameters(B)])

        # Total variance
        f0 = np.mean(np.concatenate([f_A, f_B]))
        total_variance = np.var(np.concatenate([f_A, f_B]))

        if total_variance == 0:
            return {name: 0.0 for name in self.param_names}

        # First-order indices
        first_order = {}

        for i, param_name in enumerate(self.param_names):
            # Create A_B^(i) matrix: A with column i from B
            A_Bi = A.copy()
            A_Bi[:, i] = B[:, i]

            # Evaluate f(A_B^(i))
            f_ABi = np.array([self.model_function(p) for p in self.matrices_to_parameters(A_Bi)])

            # First-order variance
            # V_i ≈ (1/N) Σ f(B)·[f(A_B^(i)) - f(A)]
            V_i = np.mean(f_B * (f_ABi - f_A))

            # First-order index
            S_i = V_i / total_variance

            first_order[param_name] = max(0.0, S_i)  # Clamp to [0, 1]

        return first_order

    def compute_total_order_indices(self, n_samples: int = 1000,
                                    seed: Optional[int] = None) -> Dict[str, float]:
        """
        Compute total-order Sobol indices.

        S_Ti = 1 - V_~i / Var(Y)

        where V_~i = Var[E(Y|X_~i)] (all except i)

        Args:
            n_samples: Number of samples
            seed: Random seed

        Returns:
            Dict of {param_name: S_Ti}
        """
        # Generate sample matrices
        A, B = self.sample_sobol_matrices(n_samples, seed)

        # Evaluate f(A) and f(B)
        f_A = np.array([self.model_function(p) for p in self.matrices_to_parameters(A)])
        f_B = np.array([self.model_function(p) for p in self.matrices_to_parameters(B)])

        # Total variance
        total_variance = np.var(np.concatenate([f_A, f_B]))

        if total_variance == 0:
            return {name: 0.0 for name in self.param_names}

        # Total-order indices
        total_order = {}

        for i, param_name in enumerate(self.param_names):
            # Create A with column i from B (complement)
            A_Bi = A.copy()
            A_Bi[:, i] = B[:, i]

            # Evaluate f(A_B^(i))
            f_ABi = np.array([self.model_function(p) for p in self.matrices_to_parameters(A_Bi)])

            # Total effect variance
            # E_~i ≈ (1/2N) Σ [f(A) - f(A_B^(i))]²
            E_not_i = 0.5 * np.mean((f_A - f_ABi) ** 2)

            # Total-order index
            S_Ti = E_not_i / total_variance

            total_order[param_name] = min(1.0, max(0.0, S_Ti))  # Clamp to [0, 1]

        return total_order


class SensitivityAnalyzer:
    """
    Unified sensitivity analysis interface.

    Provides both Morris screening (fast) and Sobol indices (comprehensive).
    """

    def __init__(self, parameter_space: Dict[str, Tuple[float, float]],
                 model_function: Callable[[Dict[str, float]], float]):
        """
        Initialize sensitivity analyzer.

        Args:
            parameter_space: Dict of {param_name: (min, max)}
            model_function: Model to analyze
        """
        self.parameter_space = parameter_space
        self.model_function = model_function

        self.morris = MorrisScreening(parameter_space, model_function)
        self.sobol = SobolIndices(parameter_space, model_function)

    def quick_screening(self, n_trajectories: int = 20) -> SensitivityResult:
        """
        Fast screening using Morris method.

        Good for initial exploration with many parameters.

        Args:
            n_trajectories: Number of trajectories (20-50 typical)

        Returns:
            SensitivityResult ranked by importance
        """
        return self.morris.analyze(n_trajectories=n_trajectories)

    def comprehensive_analysis(self, n_samples: int = 1000) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive Sobol analysis.

        More expensive but provides variance decomposition.

        Args:
            n_samples: Number of samples (1000+ recommended)

        Returns:
            Dict with 'first_order' and 'total_order' indices
        """
        first_order = self.sobol.compute_first_order_indices(n_samples)
        total_order = self.sobol.compute_total_order_indices(n_samples)

        return {
            'first_order': first_order,
            'total_order': total_order,
            'interactions': {name: total_order[name] - first_order[name]
                           for name in first_order.keys()}
        }

    def identify_critical_parameters(self, threshold: float = 0.1,
                                    method: str = 'morris') -> List[str]:
        """
        Identify critical parameters above sensitivity threshold.

        Args:
            threshold: Sensitivity threshold
            method: 'morris' or 'sobol'

        Returns:
            List of critical parameter names
        """
        if method == 'morris':
            result = self.quick_screening()
            critical = [name for name, sens in result.rankings if sens > threshold]
        else:  # sobol
            analysis = self.comprehensive_analysis()
            critical = [name for name, sens in analysis['total_order'].items()
                       if sens > threshold]

        return critical
