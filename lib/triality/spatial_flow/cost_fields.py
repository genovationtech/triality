"""Cost field definitions for spatial flow problems"""

import numpy as np
from typing import Callable, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CostField:
    """Spatial cost field for routing optimization

    A cost field assigns a cost value to each point in space, representing
    the expense or penalty of routing through that location.

    Args:
        function: Callable (x, y) -> cost or numpy array
        weight: Multiplier for this cost component
        name: Descriptive name (e.g., "distance", "heat", "EMI")
        description: Human-readable explanation
    """
    function: Callable[[float, float], float]
    weight: float = 1.0
    name: str = "cost"
    description: str = ""

    def __call__(self, x: float, y: float) -> float:
        """Evaluate cost at a point"""
        return self.weight * self.function(x, y)

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"Cost weight must be non-negative, got {self.weight}")


class CostFieldBuilder:
    """Builder for common cost field patterns"""

    @staticmethod
    def uniform(weight: float = 1.0) -> CostField:
        """Uniform cost field (pure distance minimization)

        Args:
            weight: Cost multiplier

        Returns:
            CostField with constant value
        """
        return CostField(
            function=lambda x, y: 1.0,
            weight=weight,
            name="distance",
            description="Uniform cost - minimizes total path length"
        )

    @staticmethod
    def gaussian_hotspot(center: Tuple[float, float], sigma: float,
                        amplitude: float = 1.0, weight: float = 1.0) -> CostField:
        """Gaussian cost hotspot (e.g., heat source, EMI emitter)

        Args:
            center: (x, y) center of hotspot
            sigma: Standard deviation (spread)
            amplitude: Peak cost value
            weight: Cost multiplier

        Returns:
            CostField with Gaussian profile
        """
        cx, cy = center

        def gaussian(x, y):
            r2 = (x - cx)**2 + (y - cy)**2
            return amplitude * np.exp(-r2 / (2 * sigma**2))

        return CostField(
            function=gaussian,
            weight=weight,
            name=f"hotspot_at_{center}",
            description=f"Gaussian hotspot centered at {center} with σ={sigma}"
        )

    @staticmethod
    def linear_gradient(direction: Tuple[float, float],
                       weight: float = 1.0) -> CostField:
        """Linear cost gradient (e.g., elevation change)

        Args:
            direction: (dx, dy) gradient direction (normalized internally)
            weight: Cost multiplier

        Returns:
            CostField with linear gradient
        """
        dx, dy = direction
        norm = np.sqrt(dx**2 + dy**2)
        if norm < 1e-10:
            raise ValueError("Gradient direction must be non-zero")

        dx_n, dy_n = dx / norm, dy / norm

        def gradient(x, y):
            return dx_n * x + dy_n * y

        return CostField(
            function=gradient,
            weight=weight,
            name="gradient",
            description=f"Linear gradient in direction {direction}"
        )

    @staticmethod
    def radial(center: Tuple[float, float], power: float = 1.0,
              weight: float = 1.0) -> CostField:
        """Radial cost field (distance from center)

        Args:
            center: (x, y) center point
            power: Exponent (1=linear, 2=quadratic)
            weight: Cost multiplier

        Returns:
            CostField with radial profile
        """
        cx, cy = center

        def radial(x, y):
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            return r ** power

        return CostField(
            function=radial,
            weight=weight,
            name=f"radial_from_{center}",
            description=f"Radial cost from {center} with power={power}"
        )

    @staticmethod
    def combine(fields: Dict[str, CostField]) -> CostField:
        """Combine multiple cost fields with weighted sum

        Args:
            fields: Dictionary of {name: CostField}

        Returns:
            Combined CostField
        """
        if not fields:
            raise ValueError("Must provide at least one cost field")

        def combined(x, y):
            total = 0.0
            for field in fields.values():
                total += field(x, y)
            return total

        names = ", ".join(fields.keys())
        return CostField(
            function=combined,
            weight=1.0,
            name="combined",
            description=f"Combination of: {names}"
        )


def validate_cost_field(field: CostField, domain_bounds: Tuple[Tuple[float, float], ...],
                       num_samples: int = 100) -> bool:
    """Validate cost field produces valid outputs

    Args:
        field: CostField to validate
        domain_bounds: ((xmin, xmax), (ymin, ymax))
        num_samples: Number of random points to test

    Returns:
        True if valid, raises ValueError if not
    """
    (xmin, xmax), (ymin, ymax) = domain_bounds

    # Sample random points
    x_samples = np.random.uniform(xmin, xmax, num_samples)
    y_samples = np.random.uniform(ymin, ymax, num_samples)

    for x, y in zip(x_samples, y_samples):
        try:
            cost = field(x, y)
            if np.isnan(cost):
                raise ValueError(
                    f"Cost field '{field.name}' produced NaN at ({x:.3f}, {y:.3f})"
                )
            if np.isinf(cost):
                raise ValueError(
                    f"Cost field '{field.name}' produced Inf at ({x:.3f}, {y:.3f})"
                )
            if cost < 0:
                raise ValueError(
                    f"Cost field '{field.name}' produced negative value {cost:.3e} at ({x:.3f}, {y:.3f})\n"
                    f"  Suggestion: Costs must be non-negative"
                )
        except Exception as e:
            raise ValueError(
                f"Cost field '{field.name}' evaluation failed at ({x:.3f}, {y:.3f}):\n"
                f"  {str(e)}"
            )

    return True
