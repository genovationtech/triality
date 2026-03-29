"""
Triality Geospatial Module

Physics-based geospatial analysis for logistics, infrastructure, and location feasibility.
Provides travel time calculations, isochrone analysis, population overlays, and kill-switch
logic for location-dependent systems.

Key Capabilities:
- Travel time calculations (distance + road network physics)
- Isochrone generation (reachability zones)
- Population coverage analysis
- Geospatial feasibility checking (kill-switches)

Philosophy: Physics-grounded by default, works with real data (OSM, WorldPop)

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

from .travel_time import (
    calculate_travel_time,
    haversine_distance,
    RoadType
)

from .isochrones import (
    calculate_isochrone,
    calculate_coverage,
    calculate_multi_warehouse_coverage
)

from .population import (
    estimate_population_in_radius,
    calculate_population_coverage,
    get_india_population_centers
)

from .feasibility import (
    GeospatialFeasibilityChecker,
    check_24h_coverage,
    check_delivery_feasibility
)

# Production data integration
from .config import (
    GeospatialConfig,
    get_config
)

from .osrm_client import (
    OSRMClient,
    OSRMRouteResult
)

from .data_loaders import (
    GeospatialDataLoader,
    DatasetInfo,
    download_osm_data,
    list_osm_regions
)

from .solver import (
    GeospatialSolver,
    GeospatialSolverResult,
    FacilityLocationConfig,
)

__version__ = "1.0.0"

__all__ = [
    # Travel time
    'calculate_travel_time',
    'haversine_distance',
    'RoadType',

    # Isochrones
    'calculate_isochrone',
    'calculate_coverage',
    'calculate_multi_warehouse_coverage',

    # Population
    'estimate_population_in_radius',
    'calculate_population_coverage',
    'get_india_population_centers',

    # Feasibility
    'GeospatialFeasibilityChecker',
    'check_24h_coverage',
    'check_delivery_feasibility',

    # Production data integration
    'GeospatialConfig',
    'get_config',
    'OSRMClient',
    'OSRMRouteResult',
    'GeospatialDataLoader',
    'DatasetInfo',
    'download_osm_data',
    'list_osm_regions',

    # Solver
    'GeospatialSolver',
    'GeospatialSolverResult',
    'FacilityLocationConfig',
]
