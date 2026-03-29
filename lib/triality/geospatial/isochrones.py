"""
Isochrone Analysis

Calculates reachability zones (isochrones) - all locations reachable within a given time limit.
Uses simplified grid-based approach for conceptual analysis.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from .travel_time import (
    haversine_distance,
    calculate_travel_time,
    RoadType
)


@dataclass
class IsochroneResult:
    """Result of isochrone calculation"""
    center_location: Tuple[float, float]
    time_limit_hours: float
    max_radius_km: float
    coverage_area_km2: float
    road_type: RoadType
    reachable_points: List[Tuple[float, float]]


def calculate_isochrone(
    center: Tuple[float, float],
    time_limit_hours: float,
    road_type: RoadType = RoadType.STATE_HIGHWAY,
    speed_override_kmh: Optional[float] = None,
    circuity_factor: float = 1.3,
    grid_resolution_km: float = 50.0
) -> IsochroneResult:
    """
    Calculate isochrone (reachability zone) from a center point

    An isochrone is the set of all locations reachable within a given time limit.

    Simplified approach:
    - Creates grid of candidate points
    - Tests travel time to each point
    - Identifies reachable points

    Args:
        center: (latitude, longitude) of center point
        time_limit_hours: Maximum travel time
        road_type: Type of road network
        speed_override_kmh: Custom speed (optional)
        circuity_factor: Road distance multiplier
        grid_resolution_km: Spacing between test points (km)

    Returns:
        IsochroneResult with reachable zone

    Examples:
        >>> # 24-hour reachability from Mumbai
        >>> result = calculate_isochrone((19.0760, 72.8777), 24)
        >>> print(f"Max radius: {result.max_radius_km:.0f} km")
        Max radius: 1108 km  # 24h * 60 km/h / 1.3 circuity

        >>> # Coverage area
        >>> print(f"Coverage: {result.coverage_area_km2/1e6:.2f} million km²")
        Coverage: 3.86 million km²
    """
    lat_center, lon_center = center

    # Determine average speed
    if speed_override_kmh is not None:
        speed = speed_override_kmh
    else:
        speed = road_type.value

    # Maximum straight-line distance reachable
    max_road_distance_km = speed * time_limit_hours
    max_straight_line_distance_km = max_road_distance_km / circuity_factor

    # Create grid of test points
    # Convert km to approximate degrees (rough: 1 degree ≈ 111 km at equator)
    km_per_degree = 111.0
    grid_resolution_deg = grid_resolution_km / km_per_degree

    # Grid bounds
    lat_range = max_straight_line_distance_km / km_per_degree
    lon_range = max_straight_line_distance_km / (km_per_degree * np.cos(np.radians(lat_center)))

    # Generate grid points
    lat_points = np.arange(lat_center - lat_range, lat_center + lat_range, grid_resolution_deg)
    lon_points = np.arange(lon_center - lon_range, lon_center + lon_range, grid_resolution_deg)

    reachable_points = []

    for lat in lat_points:
        for lon in lon_points:
            result = calculate_travel_time(
                center,
                (lat, lon),
                road_type=road_type,
                speed_override_kmh=speed_override_kmh,
                circuity_factor=circuity_factor
            )

            if result.travel_time_hours <= time_limit_hours:
                reachable_points.append((lat, lon))

    # Estimate coverage area (simplified: π * r²)
    coverage_area_km2 = np.pi * (max_straight_line_distance_km ** 2)

    return IsochroneResult(
        center_location=center,
        time_limit_hours=time_limit_hours,
        max_radius_km=max_straight_line_distance_km,
        coverage_area_km2=coverage_area_km2,
        road_type=road_type,
        reachable_points=reachable_points
    )


def calculate_coverage(
    warehouses: List[Tuple[float, float]],
    target_locations: List[Tuple[float, float]],
    time_limit_hours: float,
    road_type: RoadType = RoadType.STATE_HIGHWAY,
    circuity_factor: float = 1.3
) -> float:
    """
    Calculate fraction of target locations reachable from warehouses

    Args:
        warehouses: List of warehouse (lat, lon) locations
        target_locations: List of target (lat, lon) to check
        time_limit_hours: Maximum delivery time
        road_type: Road network type
        circuity_factor: Road distance multiplier

    Returns:
        Fraction of targets reachable (0.0 to 1.0)

    Examples:
        >>> warehouses = [(19.0760, 72.8777)]  # Mumbai only
        >>> cities = [(28.7041, 77.1025), (12.9716, 77.5946), ...]  # 10 cities
        >>> coverage = calculate_coverage(warehouses, cities, 24)
        >>> print(f"Coverage: {coverage*100:.1f}%")
        Coverage: 65.0%
    """
    reachable_count = 0

    for target in target_locations:
        # Check if target is reachable from ANY warehouse
        is_reachable = False

        for warehouse in warehouses:
            result = calculate_travel_time(
                warehouse,
                target,
                road_type=road_type,
                circuity_factor=circuity_factor
            )

            if result.travel_time_hours <= time_limit_hours:
                is_reachable = True
                break

        if is_reachable:
            reachable_count += 1

    return reachable_count / len(target_locations) if target_locations else 0.0


def calculate_multi_warehouse_coverage(
    warehouse_locations: List[Tuple[float, float]],
    population_centers: List[Tuple[float, float, float]],  # (lat, lon, population_millions)
    time_limit_hours: float,
    road_type: RoadType = RoadType.STATE_HIGHWAY
) -> Tuple[float, float]:
    """
    Calculate population coverage from multiple warehouses

    Args:
        warehouse_locations: List of warehouse (lat, lon)
        population_centers: List of (lat, lon, population_millions)
        time_limit_hours: Maximum delivery time
        road_type: Road network type

    Returns:
        (coverage_fraction, total_population_covered_millions)

    Examples:
        >>> warehouses = [(19.0760, 72.8777), (28.7041, 77.1025)]  # Mumbai + Delhi
        >>> cities = [
        ...     (19.0760, 72.8777, 20),   # Mumbai 20M
        ...     (28.7041, 77.1025, 30),   # Delhi 30M
        ...     (12.9716, 77.5946, 12)    # Bangalore 12M
        ... ]
        >>> coverage, pop_covered = calculate_multi_warehouse_coverage(warehouses, cities, 24)
        >>> print(f"{coverage*100:.0f}% coverage, {pop_covered:.0f}M people")
        82% coverage, 50M people
    """
    total_population = sum(pop for _, _, pop in population_centers)
    covered_population = 0.0

    for lat, lon, pop in population_centers:
        # Check if reachable from any warehouse
        is_reachable = False

        for warehouse in warehouse_locations:
            result = calculate_travel_time(
                warehouse,
                (lat, lon),
                road_type=road_type
            )

            if result.travel_time_hours <= time_limit_hours:
                is_reachable = True
                break

        if is_reachable:
            covered_population += pop

    coverage_fraction = covered_population / total_population if total_population > 0 else 0.0

    return coverage_fraction, covered_population
