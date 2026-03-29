"""Routing Integration with Physics

Integrates physics-aware cost fields with Spatial Flow Engine for
intelligent, constraint-aware routing.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from triality.spatial_flow import SpatialFlowEngine
from .cost_field_builders import PhysicsCostField, CombinedCostBuilder


class OptimizationObjective(Enum):
    """Routing optimization objectives"""
    MIN_LENGTH = 'min_length'              # Shortest path
    MIN_EMI = 'min_emi'                    # Minimize EMI risk
    MIN_THERMAL = 'min_thermal'            # Minimize thermal risk
    MIN_CROSSTALK = 'min_crosstalk'        # Minimize coupling
    BALANCED = 'balanced'                   # Balance all objectives
    CUSTOM = 'custom'                       # User-defined weights


@dataclass
class RouteResult:
    """Results from physics-aware routing"""
    path: List[Tuple[float, float]]       # Route waypoints
    length: float                          # Total path length [m]
    cost: float                            # Total routing cost
    emi_score: Optional[float] = None      # EMI risk score (0-1)
    thermal_score: Optional[float] = None  # Thermal risk score (0-1)
    clearance_score: Optional[float] = None  # Clearance adequacy (0-1)


class PhysicsAwareRouter:
    """
    Physics-aware routing engine.

    Combines Spatial Flow Engine with electromagnetic analysis to create
    routes that respect physical constraints:
    - EMI minimization
    - Thermal management
    - Clearance requirements
    - Return path optimization
    """

    def __init__(self):
        self.physics_cost_fields: Dict[str, PhysicsCostField] = {}
        self.domain_x = None
        self.domain_y = None
        self.resolution = 50  # Default resolution

    def set_domain(self, x_range: Tuple[float, float], y_range: Tuple[float, float]):
        """Set routing domain"""
        self.domain_x = x_range
        self.domain_y = y_range

    def set_resolution(self, n: int):
        """Set grid resolution"""
        self.resolution = n

    def add_physics_cost(self, name: str, cost_field: PhysicsCostField, weight: float = 1.0):
        """
        Add physics-aware cost field to routing.

        Args:
            name: Cost field identifier
            cost_field: PhysicsCostField from Layer 2 builders
            weight: Weight for multi-objective optimization
        """
        self.physics_cost_fields[name] = (cost_field, weight)

    def route(self,
             start: Tuple[float, float],
             end: Tuple[float, float],
             objective: OptimizationObjective = OptimizationObjective.BALANCED,
             custom_weights: Optional[Dict[str, float]] = None,
             verbose: bool = False) -> RouteResult:
        """
        Compute physics-aware route from start to end.

        Args:
            start: Starting point (x, y)
            end: Ending point (x, y)
            objective: Optimization objective
            custom_weights: Custom weights for CUSTOM objective
            verbose: Print diagnostics

        Returns:
            RouteResult with path and metrics
        """
        # Create fresh engine for this route
        engine = SpatialFlowEngine()
        engine.set_domain(self.domain_x, self.domain_y)
        engine.set_resolution(self.resolution)

        # Set up source and sink
        engine.add_source(start, label='Start')
        engine.add_sink(end, label='End')

        # Build combined cost field based on objective
        weights = self._get_objective_weights(objective, custom_weights)

        if self.physics_cost_fields:
            cost_fields = {name: field for name, (field, _) in self.physics_cost_fields.items()}
            combined_cost = CombinedCostBuilder.combine(cost_fields, weights)

            # Set cost field in engine
            engine.set_cost_field(combined_cost)

        # Solve routing problem
        network = engine.solve(verbose=verbose)

        if not network.paths:
            raise RuntimeError("No route found")

        # Extract path
        path_obj = network.paths[0]
        path_points = path_obj.points
        path_length = path_obj.length()

        # Compute total cost
        total_cost = 0.0
        for i in range(len(path_points) - 1):
            x1, y1 = path_points[i]
            x2, y2 = path_points[i + 1]
            xm, ym = (x1 + x2) / 2, (y1 + y2) / 2  # Midpoint

            for name, (cost_field, weight) in self.physics_cost_fields.items():
                cost_at_point = cost_field(xm, ym) * weight
                segment_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_cost += cost_at_point * segment_length

        # Compute risk scores
        emi_score = self._compute_emi_score(path_points)
        thermal_score = self._compute_thermal_score(path_points)
        clearance_score = self._compute_clearance_score(path_points)

        return RouteResult(
            path=path_points,
            length=path_length,
            cost=total_cost,
            emi_score=emi_score,
            thermal_score=thermal_score,
            clearance_score=clearance_score,
        )

    def _get_objective_weights(self,
                              objective: OptimizationObjective,
                              custom_weights: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Get cost field weights based on optimization objective"""
        if objective == OptimizationObjective.CUSTOM and custom_weights:
            return custom_weights

        # Default weights for each objective
        weights = {}

        if objective == OptimizationObjective.MIN_LENGTH:
            # Minimize all costs (shortest path through low-cost regions)
            for name in self.physics_cost_fields:
                weights[name] = 0.1  # Small but non-zero

        elif objective == OptimizationObjective.MIN_EMI:
            # Prioritize EMI reduction
            for name in self.physics_cost_fields:
                if 'EMI' in name or 'Electric' in name:
                    weights[name] = 10.0
                else:
                    weights[name] = 1.0

        elif objective == OptimizationObjective.MIN_THERMAL:
            # Prioritize thermal management
            for name in self.physics_cost_fields:
                if 'Thermal' in name or 'Current' in name:
                    weights[name] = 10.0
                else:
                    weights[name] = 1.0

        elif objective == OptimizationObjective.MIN_CROSSTALK:
            # Prioritize separation/clearance
            for name in self.physics_cost_fields:
                if 'Clearance' in name or 'Electric' in name:
                    weights[name] = 10.0
                else:
                    weights[name] = 1.0

        elif objective == OptimizationObjective.BALANCED:
            # Equal weights
            for name in self.physics_cost_fields:
                weights[name] = 1.0

        return weights

    def _compute_emi_score(self, path: List[Tuple[float, float]]) -> Optional[float]:
        """Compute EMI risk score for path (0=low, 1=high)"""
        if 'EMICost' not in self.physics_cost_fields:
            return None

        cost_field, _ = self.physics_cost_fields['EMICost']
        costs = [cost_field(x, y) for x, y in path]

        # Normalize to 0-1 range
        avg_cost = np.mean(costs)
        return min(1.0, avg_cost / 10.0)  # Assuming cost ~10 is "high"

    def _compute_thermal_score(self, path: List[Tuple[float, float]]) -> Optional[float]:
        """Compute thermal risk score for path (0=low, 1=high)"""
        if 'ThermalRiskCost' not in self.physics_cost_fields and 'CurrentDensityCost' not in self.physics_cost_fields:
            return None

        # Try thermal first, then current
        field_name = 'ThermalRiskCost' if 'ThermalRiskCost' in self.physics_cost_fields else 'CurrentDensityCost'
        cost_field, _ = self.physics_cost_fields[field_name]

        costs = [cost_field(x, y) for x, y in path]
        avg_cost = np.mean(costs)
        return min(1.0, avg_cost / 10.0)

    def _compute_clearance_score(self, path: List[Tuple[float, float]]) -> Optional[float]:
        """Compute clearance adequacy score (0=inadequate, 1=good)"""
        if 'ClearanceCost' not in self.physics_cost_fields:
            return None

        cost_field, _ = self.physics_cost_fields['ClearanceCost']
        costs = [cost_field(x, y) for x, y in path]

        # Invert: low cost = good clearance
        avg_cost = np.mean(costs)
        return max(0.0, 1.0 - avg_cost / 100.0)


def RouteWithPhysics(engine: SpatialFlowEngine,
                    physics_costs: Dict[str, PhysicsCostField],
                    weights: Optional[Dict[str, float]] = None) -> SpatialFlowEngine:
    """
    Helper function: Add physics costs to existing Spatial Flow Engine.

    Args:
        engine: Existing SpatialFlowEngine
        physics_costs: Dictionary of {name: PhysicsCostField}
        weights: Optional weights for each cost field

    Returns:
        Modified engine with physics costs
    """
    if weights is None:
        weights = {name: 1.0 for name in physics_costs}

    combined_cost = CombinedCostBuilder.combine(physics_costs, weights)
    engine.set_cost_field(combined_cost)

    return engine


class MultiRouteOptimizer:
    """
    Optimize multiple routes simultaneously considering interactions.

    Use cases:
    - Bus routing (multiple parallel signals)
    - Differential pairs
    - Power distribution networks
    """

    def __init__(self):
        self.router = PhysicsAwareRouter()

    def route_multiple(self,
                      route_pairs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                      min_separation: float = 0.001,
                      objective: OptimizationObjective = OptimizationObjective.BALANCED,
                      verbose: bool = False) -> List[RouteResult]:
        """
        Route multiple paths with crosstalk awareness.

        Args:
            route_pairs: List of (start, end) tuples
            min_separation: Minimum separation between routes [m]
            objective: Optimization objective
            verbose: Print diagnostics

        Returns:
            List of RouteResult for each path
        """
        routes = []

        # Route sequentially, adding separation costs for completed routes
        for i, (start, end) in enumerate(route_pairs):
            if verbose:
                print(f"\nRouting path {i+1}/{len(route_pairs)}: {start} → {end}")

            # Add clearance costs from previously routed paths
            if routes:
                # Create cost field that avoids previous routes
                self._add_route_avoidance_cost(routes, min_separation)

            # Route this path
            result = self.router.route(start, end, objective=objective, verbose=verbose)
            routes.append(result)

        return routes

    def _add_route_avoidance_cost(self, existing_routes: List[RouteResult], min_separation: float):
        """Add cost field to avoid existing routes"""
        from .cost_field_builders import ClearanceCostBuilder

        # Convert existing routes to conductor regions
        conductor_regions = []
        for route in existing_routes:
            for x, y in route.path:
                conductor_regions.append((x, y, min_separation / 2))

        # Create clearance cost
        clearance_cost = ClearanceCostBuilder.from_conductors(
            conductor_regions,
            min_clearance=min_separation,
            violation_cost=100.0
        )

        self.router.add_physics_cost('RouteAvoidance', clearance_cost, weight=10.0)
