"""Cable routing template - high-level API for cable layout problems"""

import numpy as np
from typing import List, Tuple, Dict, Optional

from ..engine import SpatialFlowEngine, Network
from ..sources_sinks import Source, Sink
from ..cost_fields import CostFieldBuilder, CostField
from ..constraints import ObstacleBuilder, Obstacle, ObstacleType


def route_cable(
    sources: List[Tuple[float, float]],
    sinks: List[Tuple[float, float]],
    domain_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 1), (0, 1)),
    obstacles: Optional[List[Dict]] = None,
    cost_factors: Optional[Dict[str, float]] = None,
    resolution: int = 100,
    verbose: bool = True
) -> Network:
    """
    Route cables from sources to sinks using physics-based optimization.

    This is a high-level template that configures the SpatialFlowEngine
    for typical cable routing problems.

    Args:
        sources: List of (x, y) source positions
        sinks: List of (x, y) sink positions
        domain_bounds: ((xmin, xmax), (ymin, ymax))
        obstacles: List of obstacle specifications (see below)
        cost_factors: Dict of cost weights (see below)
        resolution: Grid resolution
        verbose: Print progress

    Obstacle specification:
        Each obstacle is a dict with:
        - 'type': 'rectangle', 'circle', or 'polygon'
        - 'params': geometry parameters
        - 'obstacle_type': 'hard' (forbidden) or 'soft' (high cost)

        Examples:
            {'type': 'rectangle', 'params': [0.2, 0.4, 0.3, 0.5]}  # xmin, xmax, ymin, ymax
            {'type': 'circle', 'params': [(0.5, 0.5), 0.1]}  # center, radius
            {'type': 'polygon', 'params': [[(0.1, 0.1), (0.2, 0.1), (0.2, 0.2)]]}

    Cost factors:
        - 'distance': Weight for path length (default: 1.0)
        - 'heat': Weight for heat avoidance (requires heat_sources)
        - 'emi': Weight for EMI avoidance (requires emi_sources)

    Returns:
        Network object with extracted cable paths

    Example:
        >>> from triality.spatial_flow.templates import cable_routing
        >>> sources = [(0.1, 0.5)]
        >>> sinks = [(0.9, 0.5)]
        >>> obstacles = [{'type': 'rectangle', 'params': [0.4, 0.6, 0.3, 0.7]}]
        >>> network = cable_routing.route_cable(sources, sinks, obstacles=obstacles)
    """

    # Create engine
    engine = SpatialFlowEngine()

    # Set domain
    engine.set_domain(*domain_bounds)
    engine.set_resolution(resolution)

    # Add sources
    for i, pos in enumerate(sources):
        engine.add_source(pos, weight=1.0, label=f"S{i}")

    # Add sinks
    for i, pos in enumerate(sinks):
        engine.add_sink(pos, weight=1.0, label=f"D{i}")

    # Add obstacles
    if obstacles:
        for obs_spec in obstacles:
            obs_type = obs_spec.get('type', 'rectangle')
            params = obs_spec['params']
            hard_or_soft = obs_spec.get('obstacle_type', 'hard')

            obstacle_type = ObstacleType.HARD if hard_or_soft == 'hard' else ObstacleType.SOFT

            if obs_type == 'rectangle':
                obs = ObstacleBuilder.rectangle(*params, obstacle_type=obstacle_type)
            elif obs_type == 'circle':
                obs = ObstacleBuilder.circle(*params, obstacle_type=obstacle_type)
            elif obs_type == 'polygon':
                obs = ObstacleBuilder.polygon(*params, obstacle_type=obstacle_type)
            else:
                raise ValueError(f"Unknown obstacle type: {obs_type}")

            engine.add_obstacle(obs)

    # Build cost field
    if cost_factors is None:
        cost_factors = {'distance': 1.0}

    cost_fields = {}

    # Distance cost (uniform)
    if 'distance' in cost_factors:
        cost_fields['distance'] = CostFieldBuilder.uniform(weight=cost_factors['distance'])

    # Heat avoidance
    if 'heat' in cost_factors:
        heat_sources = cost_factors.get('heat_sources', [])
        for i, (center, intensity) in enumerate(heat_sources):
            field = CostFieldBuilder.gaussian_hotspot(
                center=center,
                sigma=0.1,
                amplitude=intensity,
                weight=cost_factors['heat']
            )
            cost_fields[f'heat_{i}'] = field

    # EMI avoidance
    if 'emi' in cost_factors:
        emi_sources = cost_factors.get('emi_sources', [])
        for i, (center, intensity) in enumerate(emi_sources):
            field = CostFieldBuilder.gaussian_hotspot(
                center=center,
                sigma=0.15,
                amplitude=intensity,
                weight=cost_factors['emi']
            )
            cost_fields[f'emi_{i}'] = field

    # Combine cost fields
    if len(cost_fields) == 1:
        engine.set_cost_field(list(cost_fields.values())[0])
    else:
        combined = CostFieldBuilder.combine(cost_fields)
        engine.set_cost_field(combined)

    # Solve
    return engine.solve(verbose=verbose)


def route_power_cable(
    power_source: Tuple[float, float],
    devices: List[Tuple[float, float]],
    domain_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 1), (0, 1)),
    heat_sources: Optional[List[Tuple[Tuple[float, float], float]]] = None,
    obstacles: Optional[List[Dict]] = None,
    heat_weight: float = 0.5,
    resolution: int = 100,
    verbose: bool = True
) -> Network:
    """
    Route power cables from a central source to multiple devices.

    This is a specialized cable routing template for power distribution.

    Args:
        power_source: (x, y) location of power source
        devices: List of (x, y) device locations
        domain_bounds: ((xmin, xmax), (ymin, ymax))
        heat_sources: List of ((x, y), intensity) heat-emitting components
        obstacles: List of obstacle specifications
        heat_weight: Weight for heat avoidance
        resolution: Grid resolution
        verbose: Print progress

    Returns:
        Network object with power cable routes
    """
    cost_factors = {'distance': 1.0}

    if heat_sources:
        cost_factors['heat'] = heat_weight
        cost_factors['heat_sources'] = heat_sources

    return route_cable(
        sources=[power_source],
        sinks=devices,
        domain_bounds=domain_bounds,
        obstacles=obstacles,
        cost_factors=cost_factors,
        resolution=resolution,
        verbose=verbose
    )


def route_signal_cable(
    source: Tuple[float, float],
    destination: Tuple[float, float],
    domain_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 1), (0, 1)),
    emi_sources: Optional[List[Tuple[Tuple[float, float], float]]] = None,
    obstacles: Optional[List[Dict]] = None,
    emi_weight: float = 1.0,
    resolution: int = 100,
    verbose: bool = True
) -> Network:
    """
    Route signal cables avoiding electromagnetic interference.

    This is a specialized cable routing template for sensitive signal cables.

    Args:
        source: (x, y) signal source location
        destination: (x, y) signal destination location
        domain_bounds: ((xmin, xmax), (ymin, ymax))
        emi_sources: List of ((x, y), intensity) EMI-emitting components
        obstacles: List of obstacle specifications
        emi_weight: Weight for EMI avoidance
        resolution: Grid resolution
        verbose: Print progress

    Returns:
        Network object with signal cable route
    """
    cost_factors = {'distance': 1.0}

    if emi_sources:
        cost_factors['emi'] = emi_weight
        cost_factors['emi_sources'] = emi_sources

    return route_cable(
        sources=[source],
        sinks=[destination],
        domain_bounds=domain_bounds,
        obstacles=obstacles,
        cost_factors=cost_factors,
        resolution=resolution,
        verbose=verbose
    )
