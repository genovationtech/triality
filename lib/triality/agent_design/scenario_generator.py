"""
Autonomous scenario generation for multi-physics testing.

Generates comprehensive test scenarios covering operating envelope,
corner cases, and failure modes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional
from enum import Enum
import itertools


class ScenarioType(Enum):
    """Types of test scenarios"""
    NOMINAL = "nominal"  # Expected operating point
    CORNER_CASE = "corner_case"  # Boundary of operating envelope
    STRESS_TEST = "stress_test"  # Beyond normal operation
    FAILURE_MODE = "failure_mode"  # Intentional failure scenarios
    RANDOM = "random"  # Random sampling within envelope


@dataclass
class OperatingEnvelope:
    """
    Define operating envelope for a system.

    Each parameter has min, max, and nominal values.
    """
    parameters: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    # Format: {param_name: (min, nominal, max)}

    def add_parameter(self, name: str, min_val: float, nominal: float, max_val: float):
        """Add a parameter to the envelope"""
        if min_val > max_val:
            raise ValueError(f"min ({min_val}) > max ({max_val}) for {name}")
        if not (min_val <= nominal <= max_val):
            raise ValueError(f"Nominal {nominal} outside [{min_val}, {max_val}] for {name}")

        self.parameters[name] = (min_val, nominal, max_val)

    def get_nominal_point(self) -> Dict[str, float]:
        """Return nominal operating point"""
        return {name: vals[1] for name, vals in self.parameters.items()}

    def get_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return parameter ranges (min, max)"""
        return {name: (vals[0], vals[2]) for name, vals in self.parameters.items()}

    def is_within_envelope(self, point: Dict[str, float]) -> bool:
        """Check if a point is within the operating envelope"""
        for name, value in point.items():
            if name not in self.parameters:
                return False
            min_val, _, max_val = self.parameters[name]
            if not (min_val <= value <= max_val):
                return False
        return True

    def distance_to_boundary(self, point: Dict[str, float]) -> float:
        """
        Calculate normalized distance to nearest boundary.

        Returns 0 at boundary, 1 at center (nominal).
        """
        if not self.is_within_envelope(point):
            return -1.0  # Outside envelope

        min_normalized_distance = 1.0

        for name, value in point.items():
            min_val, nominal, max_val = self.parameters[name]
            range_val = max_val - min_val

            # Distance to lower and upper boundary (normalized)
            dist_lower = (value - min_val) / range_val
            dist_upper = (max_val - value) / range_val

            # Minimum distance to any boundary
            min_dist = min(dist_lower, dist_upper)
            min_normalized_distance = min(min_normalized_distance, min_dist)

        return min_normalized_distance


@dataclass
class Scenario:
    """A single test scenario"""
    name: str
    scenario_type: ScenarioType
    parameters: Dict[str, float]
    expected_behavior: Optional[str] = None
    criticality: int = 1  # 1-5, 5 = most critical

    def __repr__(self):
        params_str = ", ".join([f"{k}={v:.3g}" for k, v in self.parameters.items()])
        return f"Scenario({self.name}, {self.scenario_type.value}, {params_str})"


class ScenarioGenerator:
    """
    Autonomous scenario generation for comprehensive testing.

    Generates scenarios using various strategies:
    1. Nominal operation
    2. Corner cases (vertices of hypercube)
    3. Edge cases (edges of hypercube)
    4. Stress tests (beyond limits)
    5. Random sampling
    """

    def __init__(self, envelope: OperatingEnvelope):
        self.envelope = envelope
        self.scenarios: List[Scenario] = []

    def generate_nominal_scenario(self) -> Scenario:
        """Generate nominal operating point scenario"""
        params = self.envelope.get_nominal_point()
        scenario = Scenario(
            name="Nominal Operation",
            scenario_type=ScenarioType.NOMINAL,
            parameters=params,
            expected_behavior="Normal operation",
            criticality=3
        )
        return scenario

    def generate_corner_cases(self) -> List[Scenario]:
        """
        Generate all corner cases (vertices of hypercube).

        For n parameters, generates 2^n scenarios.
        """
        scenarios = []
        param_names = list(self.envelope.parameters.keys())
        n_params = len(param_names)

        # Generate all combinations of min/max
        for i, combo in enumerate(itertools.product([0, 2], repeat=n_params)):
            # combo is tuple of 0s and 2s (indices for min/max)
            params = {}
            for j, param_name in enumerate(param_names):
                min_val, nominal, max_val = self.envelope.parameters[param_name]
                if combo[j] == 0:
                    params[param_name] = min_val
                else:
                    params[param_name] = max_val

            scenario = Scenario(
                name=f"Corner Case {i+1}",
                scenario_type=ScenarioType.CORNER_CASE,
                parameters=params,
                expected_behavior="Boundary operation",
                criticality=4
            )
            scenarios.append(scenario)

        return scenarios

    def generate_edge_cases(self, n_points_per_edge: int = 3) -> List[Scenario]:
        """
        Generate edge cases (along edges of hypercube).

        Sample along 1D edges connecting corner points.
        """
        scenarios = []
        param_names = list(self.envelope.parameters.keys())
        n_params = len(param_names)

        # For each parameter, vary it while keeping others at min or max
        count = 0
        for vary_idx in range(n_params):
            vary_name = param_names[vary_idx]
            min_val, nominal, max_val = self.envelope.parameters[vary_name]

            # Generate points along this dimension
            vary_values = np.linspace(min_val, max_val, n_points_per_edge)

            # For other parameters, try both min and max
            other_combos = itertools.product([0, 2], repeat=n_params-1)

            for combo in other_combos:
                for vary_val in vary_values:
                    params = {}
                    combo_list = list(combo)

                    for i, param_name in enumerate(param_names):
                        if i == vary_idx:
                            params[param_name] = vary_val
                        else:
                            idx = combo_list.pop(0)
                            min_v, nom_v, max_v = self.envelope.parameters[param_name]
                            params[param_name] = min_v if idx == 0 else max_v

                    scenario = Scenario(
                        name=f"Edge Case {count+1}",
                        scenario_type=ScenarioType.CORNER_CASE,
                        parameters=params,
                        expected_behavior="Edge operation",
                        criticality=3
                    )
                    scenarios.append(scenario)
                    count += 1

        return scenarios

    def generate_stress_tests(self, overstress_factor: float = 1.2) -> List[Scenario]:
        """
        Generate stress test scenarios beyond normal limits.

        Args:
            overstress_factor: Factor to exceed limits (e.g., 1.2 = 20% beyond)
        """
        scenarios = []
        param_names = list(self.envelope.parameters.keys())

        # Stress each parameter individually
        for i, param_name in enumerate(param_names):
            min_val, nominal, max_val = self.envelope.parameters[param_name]
            range_val = max_val - min_val

            # Overstress high
            params_high = self.envelope.get_nominal_point()
            params_high[param_name] = max_val + (overstress_factor - 1.0) * range_val

            scenario_high = Scenario(
                name=f"Stress Test: {param_name} High",
                scenario_type=ScenarioType.STRESS_TEST,
                parameters=params_high,
                expected_behavior="Potential failure or degradation",
                criticality=5
            )
            scenarios.append(scenario_high)

            # Overstress low
            params_low = self.envelope.get_nominal_point()
            params_low[param_name] = min_val - (overstress_factor - 1.0) * range_val

            scenario_low = Scenario(
                name=f"Stress Test: {param_name} Low",
                scenario_type=ScenarioType.STRESS_TEST,
                parameters=params_low,
                expected_behavior="Potential failure or degradation",
                criticality=5
            )
            scenarios.append(scenario_low)

        return scenarios

    def generate_random_scenarios(self, n_scenarios: int = 100,
                                  seed: Optional[int] = None) -> List[Scenario]:
        """
        Generate random scenarios within envelope using Latin Hypercube Sampling.

        Args:
            n_scenarios: Number of scenarios to generate
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        scenarios = []
        param_names = list(self.envelope.parameters.keys())
        n_params = len(param_names)

        # Latin Hypercube Sampling for better coverage
        # Divide each dimension into n_scenarios intervals
        intervals = np.arange(n_scenarios)

        lhs_samples = np.zeros((n_scenarios, n_params))
        for i in range(n_params):
            # Shuffle intervals for this parameter
            shuffled = np.random.permutation(intervals)
            # Sample uniformly within each interval
            uniform_samples = np.random.uniform(0, 1, n_scenarios)
            lhs_samples[:, i] = (shuffled + uniform_samples) / n_scenarios

        # Map to actual parameter ranges
        for i in range(n_scenarios):
            params = {}
            for j, param_name in enumerate(param_names):
                min_val, nominal, max_val = self.envelope.parameters[param_name]
                # Map [0,1] to [min, max]
                params[param_name] = min_val + lhs_samples[i, j] * (max_val - min_val)

            scenario = Scenario(
                name=f"Random Scenario {i+1}",
                scenario_type=ScenarioType.RANDOM,
                parameters=params,
                expected_behavior="Normal operation (random)",
                criticality=2
            )
            scenarios.append(scenario)

        return scenarios

    def generate_comprehensive_suite(self, n_random: int = 50) -> List[Scenario]:
        """
        Generate comprehensive test suite combining all methods.

        Returns:
            List of all generated scenarios
        """
        all_scenarios = []

        # 1. Nominal
        all_scenarios.append(self.generate_nominal_scenario())

        # 2. Corner cases
        all_scenarios.extend(self.generate_corner_cases())

        # 3. Stress tests
        all_scenarios.extend(self.generate_stress_tests())

        # 4. Random scenarios
        all_scenarios.extend(self.generate_random_scenarios(n_random))

        self.scenarios = all_scenarios
        return all_scenarios

    def filter_by_criticality(self, min_criticality: int = 3) -> List[Scenario]:
        """Return scenarios with criticality >= threshold"""
        return [s for s in self.scenarios if s.criticality >= min_criticality]

    def filter_by_type(self, scenario_type: ScenarioType) -> List[Scenario]:
        """Return scenarios of specific type"""
        return [s for s in self.scenarios if s.scenario_type == scenario_type]

    def rank_by_risk(self, risk_function: Callable[[Dict[str, float]], float]) -> List[Scenario]:
        """
        Rank scenarios by risk score from provided function.

        Args:
            risk_function: Function that takes parameters dict and returns risk score

        Returns:
            Scenarios sorted by risk (highest first)
        """
        scored_scenarios = []
        for scenario in self.scenarios:
            risk = risk_function(scenario.parameters)
            scored_scenarios.append((risk, scenario))

        # Sort by risk descending
        scored_scenarios.sort(key=lambda x: x[0], reverse=True)

        return [s for _, s in scored_scenarios]
