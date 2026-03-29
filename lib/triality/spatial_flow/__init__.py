"""
Triality Spatial Flow Engine

Continuous routing and distribution problems using physics-based field optimization.

Main API:
    - SpatialFlowEngine: Low-level engine for custom problems
    - Templates: High-level APIs for common routing patterns

Example (Low-level):
    >>> from triality.spatial_flow import SpatialFlowEngine
    >>> engine = SpatialFlowEngine()
    >>> engine.add_source((0.1, 0.5), label="A")
    >>> engine.add_sink((0.9, 0.5), label="B")
    >>> engine.set_domain((0, 1), (0, 1))
    >>> network = engine.solve()

Example (High-level):
    >>> from triality.spatial_flow.templates import cable_routing
    >>> network = cable_routing.route_cable(
    ...     sources=[(0.1, 0.5)],
    ...     sinks=[(0.9, 0.5)],
    ...     obstacles=[{'type': 'rectangle', 'params': [0.4, 0.6, 0.3, 0.7]}]
    ... )
"""

from .engine import SpatialFlowEngine, FlowProblem
from .sources_sinks import Source, Sink
from .cost_fields import CostField, CostFieldBuilder
from .constraints import Obstacle, ObstacleBuilder, ObstacleType
from .extraction import Path, Network

__all__ = [
    'SpatialFlowEngine',
    'FlowProblem',
    'Source',
    'Sink',
    'CostField',
    'CostFieldBuilder',
    'Obstacle',
    'ObstacleBuilder',
    'ObstacleType',
    'Path',
    'Network',
]
