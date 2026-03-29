"""Cost Field Builders from Physics

Converts electromagnetic analysis results into cost fields for routing:
- Electric field → clearance cost
- Current density → thermal risk cost
- Field gradients → EMI risk cost
- Multi-conductor proximity → crosstalk cost
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict
from dataclasses import dataclass

from triality.electrostatics.field_solver import ElectrostaticResult
from triality.electrostatics.conduction import ConductionResult
from triality.electrostatics.derived_quantities import ElectricField, GradientAnalysis


@dataclass
class PhysicsCostField:
    """Physics-derived cost field for routing

    Compatible with Spatial Flow Engine cost field interface.
    """
    name: str
    cost_function: Callable[[float, float], float]
    description: str

    def __call__(self, x: float, y: float) -> float:
        """Evaluate cost at point (x, y)"""
        return self.cost_function(x, y)


class ElectricFieldCostBuilder:
    """
    Build cost field from electric field magnitude.

    High electric field → High cost (clearance requirement)

    Use cases:
    - Avoid routing near high-voltage conductors
    - Maintain clearance requirements
    - Prevent dielectric breakdown
    """

    @staticmethod
    def from_result(result,
                   threshold: float = None,
                   scaling: str = 'exponential',
                   base_cost: float = 1.0) -> PhysicsCostField:
        """
        Create cost field from electrostatic result.

        Args:
            result: ElectrostaticResult from Layer 1
            threshold: Field threshold above which cost increases [V/m]
                      If None, uses 50th percentile
            scaling: 'linear', 'quadratic', or 'exponential'
            base_cost: Base cost multiplier

        Returns:
            PhysicsCostField compatible with routing engine
        """
        # Compute field magnitude grid
        field_data = ElectricField.from_result(result)
        E_mag = field_data.E_magnitude

        if threshold is None:
            threshold = np.percentile(E_mag, 50)  # Median field

        # Create interpolation function
        def cost_func(x: float, y: float) -> float:
            # Bilinear interpolation of field magnitude
            grid_x = result.grid_x
            grid_y = result.grid_y

            if x < grid_x[0] or x > grid_x[-1] or y < grid_y[0] or y > grid_y[-1]:
                return base_cost  # Outside domain

            i = np.searchsorted(grid_x, x) - 1
            j = np.searchsorted(grid_y, y) - 1
            i = np.clip(i, 0, len(grid_x) - 2)
            j = np.clip(j, 0, len(grid_y) - 2)

            # Simple nearest-neighbor for efficiency
            E = E_mag[i, j]

            if E < threshold:
                return base_cost

            # Scale cost based on field strength
            ratio = E / threshold

            if scaling == 'linear':
                cost = base_cost * ratio
            elif scaling == 'quadratic':
                cost = base_cost * (ratio ** 2)
            elif scaling == 'exponential':
                cost = base_cost * np.exp(ratio - 1)
            else:
                cost = base_cost * ratio

            return cost

        return PhysicsCostField(
            name='ElectricFieldCost',
            cost_function=cost_func,
            description=f'Clearance cost from |E| (threshold={threshold:.1e} V/m, {scaling} scaling)'
        )


class CurrentDensityCostBuilder:
    """
    Build cost field from current density.

    High current density → High cost (thermal risk)

    Use cases:
    - Avoid routing through high-current zones
    - Prevent hotspot formation
    - Thermal management
    """

    @staticmethod
    def from_result(conduction_result: ConductionResult,
                   threshold: float = None,
                   scaling: str = 'quadratic',
                   base_cost: float = 1.0) -> PhysicsCostField:
        """
        Create cost field from conduction result.

        Args:
            conduction_result: ConductionResult from Layer 1
            threshold: Current density threshold [A/m²]
            scaling: 'linear', 'quadratic', or 'exponential'
            base_cost: Base cost multiplier

        Returns:
            PhysicsCostField for thermal risk
        """
        # Compute current density magnitude on grid
        n = len(conduction_result.grid_x)
        J_mag = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                x = conduction_result.grid_x[i]
                y = conduction_result.grid_y[j]
                J_mag[i, j] = conduction_result.current_density_magnitude(x, y)

        if threshold is None:
            threshold = np.percentile(J_mag[J_mag > 0], 75)  # 75th percentile

        def cost_func(x: float, y: float) -> float:
            grid_x = conduction_result.grid_x
            grid_y = conduction_result.grid_y

            if x < grid_x[0] or x > grid_x[-1] or y < grid_y[0] or y > grid_y[-1]:
                return base_cost

            i = np.searchsorted(grid_x, x) - 1
            j = np.searchsorted(grid_y, y) - 1
            i = np.clip(i, 0, len(grid_x) - 2)
            j = np.clip(j, 0, len(grid_y) - 2)

            J = J_mag[i, j]

            if J < threshold:
                return base_cost

            ratio = J / threshold

            if scaling == 'linear':
                cost = base_cost * ratio
            elif scaling == 'quadratic':
                cost = base_cost * (ratio ** 2)
            elif scaling == 'exponential':
                cost = base_cost * np.exp(ratio - 1)
            else:
                cost = base_cost * ratio

            return cost

        return PhysicsCostField(
            name='CurrentDensityCost',
            cost_function=cost_func,
            description=f'Thermal risk from |J| (threshold={threshold:.1e} A/m², {scaling} scaling)'
        )


class EMICostBuilder:
    """
    Build cost field from field gradients.

    High ∇|E| → High cost (EMI risk)

    Use cases:
    - Avoid routing near sharp edges/corners
    - Prevent EMI hotspots
    - Reduce radiated emissions
    """

    @staticmethod
    def from_result(result,
                   threshold: float = None,
                   scaling: str = 'exponential',
                   base_cost: float = 1.0) -> PhysicsCostField:
        """
        Create EMI risk cost field from field gradients.

        Args:
            result: ElectrostaticResult from Layer 1
            threshold: Gradient threshold [V/m²]
            scaling: Cost scaling function
            base_cost: Base cost multiplier

        Returns:
            PhysicsCostField for EMI risk
        """
        field_data = ElectricField.from_result(result)
        grad_x, grad_y = GradientAnalysis.field_gradient(field_data)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        if threshold is None:
            threshold = np.percentile(grad_mag, 80)  # 80th percentile

        def cost_func(x: float, y: float) -> float:
            grid_x = result.grid_x
            grid_y = result.grid_y

            if x < grid_x[0] or x > grid_x[-1] or y < grid_y[0] or y > grid_y[-1]:
                return base_cost

            i = np.searchsorted(grid_x, x) - 1
            j = np.searchsorted(grid_y, y) - 1
            i = np.clip(i, 0, len(grid_x) - 2)
            j = np.clip(j, 0, len(grid_y) - 2)

            grad = grad_mag[i, j]

            if grad < threshold:
                return base_cost

            ratio = grad / threshold

            if scaling == 'linear':
                cost = base_cost * ratio
            elif scaling == 'quadratic':
                cost = base_cost * (ratio ** 2)
            elif scaling == 'exponential':
                cost = base_cost * np.exp(ratio - 1)
            else:
                cost = base_cost * ratio

            return min(cost, base_cost * 100)  # Cap at 100x

        return PhysicsCostField(
            name='EMICost',
            cost_function=cost_func,
            description=f'EMI risk from |∇E| (threshold={threshold:.1e}, {scaling} scaling)'
        )


class ThermalRiskCostBuilder:
    """
    Build cost field from power density.

    High power density → High cost (hotspot risk)

    Use cases:
    - Avoid routing through hotspot regions
    - Thermal-aware layout
    - Reliability optimization
    """

    @staticmethod
    def from_result(conduction_result: ConductionResult,
                   threshold: float = 1e6,  # 1 MW/m³
                   scaling: str = 'exponential',
                   base_cost: float = 1.0) -> PhysicsCostField:
        """
        Create thermal risk cost field from power density.

        Args:
            conduction_result: ConductionResult from Layer 1
            threshold: Power density threshold [W/m³]
            scaling: Cost scaling function
            base_cost: Base cost multiplier

        Returns:
            PhysicsCostField for thermal risk
        """
        P_grid = conduction_result.power_density_grid()

        def cost_func(x: float, y: float) -> float:
            grid_x = conduction_result.grid_x
            grid_y = conduction_result.grid_y

            if x < grid_x[0] or x > grid_x[-1] or y < grid_y[0] or y > grid_y[-1]:
                return base_cost

            i = np.searchsorted(grid_x, x) - 1
            j = np.searchsorted(grid_y, y) - 1
            i = np.clip(i, 0, len(grid_x) - 2)
            j = np.clip(j, 0, len(grid_y) - 2)

            P = P_grid[i, j]

            if P < threshold:
                return base_cost

            ratio = P / threshold

            if scaling == 'linear':
                cost = base_cost * ratio
            elif scaling == 'quadratic':
                cost = base_cost * (ratio ** 2)
            elif scaling == 'exponential':
                cost = base_cost * np.exp(ratio - 1)
            else:
                cost = base_cost * ratio

            return min(cost, base_cost * 100)  # Cap at 100x

        return PhysicsCostField(
            name='ThermalRiskCost',
            cost_function=cost_func,
            description=f'Hotspot risk from power density (threshold={threshold:.1e} W/m³, {scaling} scaling)'
        )


class ClearanceCostBuilder:
    """
    Build cost field for maintaining clearance from conductors.

    Proximity to conductors → High cost (clearance violation)

    Use cases:
    - Maintain minimum spacing
    - Prevent arcing/breakdown
    - Meet safety standards
    """

    @staticmethod
    def from_conductors(conductor_regions: list,
                       min_clearance: float,
                       violation_cost: float = 100.0,
                       base_cost: float = 1.0) -> PhysicsCostField:
        """
        Create clearance cost field from conductor locations.

        Args:
            conductor_regions: List of (x, y, radius) tuples for conductors
            min_clearance: Minimum required clearance [m]
            violation_cost: Cost for clearance violation
            base_cost: Base cost away from conductors

        Returns:
            PhysicsCostField for clearance requirements
        """
        def cost_func(x: float, y: float) -> float:
            min_dist = float('inf')

            for cx, cy, radius in conductor_regions:
                dist = np.sqrt((x - cx)**2 + (y - cy)**2) - radius
                min_dist = min(min_dist, dist)

            if min_dist < 0:
                # Inside conductor - extremely high cost
                return violation_cost * 10
            elif min_dist < min_clearance:
                # Too close - scale cost inversely with distance
                ratio = (min_clearance - min_dist) / min_clearance
                return base_cost + (violation_cost - base_cost) * ratio
            else:
                # Adequate clearance
                return base_cost

        return PhysicsCostField(
            name='ClearanceCost',
            cost_function=cost_func,
            description=f'Clearance requirement (min={min_clearance:.3f}m)'
        )


class CombinedCostBuilder:
    """
    Combine multiple physics cost fields with weights.

    Allows multi-objective optimization:
    - Weight EMI risk vs thermal risk vs clearance
    - Trade off different physical constraints
    """

    @staticmethod
    def combine(cost_fields: Dict[str, PhysicsCostField],
               weights: Dict[str, float]) -> PhysicsCostField:
        """
        Combine multiple cost fields with weights.

        Args:
            cost_fields: Dictionary of {name: PhysicsCostField}
            weights: Dictionary of {name: weight}

        Returns:
            Combined PhysicsCostField
        """
        def cost_func(x: float, y: float) -> float:
            total_cost = 0.0

            for name, field in cost_fields.items():
                weight = weights.get(name, 1.0)
                total_cost += weight * field(x, y)

            return total_cost

        descriptions = [f"{name}(w={weights.get(name, 1.0)})"
                       for name in cost_fields.keys()]

        return PhysicsCostField(
            name='CombinedPhysicsCost',
            cost_function=cost_func,
            description=f'Combined: {", ".join(descriptions)}'
        )
