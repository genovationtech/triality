"""
Travel Time Physics

Calculates realistic travel times based on distance, road network topology,
and environmental constraints. Uses haversine distance for geodesic calculations
and road-type-specific speeds.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from enum import Enum
from typing import Tuple, Optional
from dataclasses import dataclass


# Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0


class RoadType(Enum):
    """Road classification with representative average speeds"""
    NATIONAL_HIGHWAY = 60  # km/h (India NH, US Interstate equivalent)
    STATE_HIGHWAY = 45     # km/h (State highways, major roads)
    URBAN = 25             # km/h (City streets with traffic)
    RURAL = 30             # km/h (Rural roads, village connections)
    MOUNTAINOUS = 20       # km/h (Hill roads, difficult terrain)


@dataclass
class TravelTimeResult:
    """Result of travel time calculation"""
    distance_km: float
    travel_time_hours: float
    average_speed_kmh: float
    road_type: RoadType
    feasible: bool
    notes: str = ""
    source: str = "haversine"  # 'haversine' or 'osrm'


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate great-circle distance between two points on Earth

    Uses the haversine formula for geodesic distance calculation.

    Args:
        lat1: Latitude of point 1 (degrees)
        lon1: Longitude of point 1 (degrees)
        lat2: Latitude of point 2 (degrees)
        lon2: Longitude of point 2 (degrees)

    Returns:
        Distance in kilometers

    Examples:
        >>> haversine_distance(28.7041, 77.1025, 19.0760, 72.8777)  # Delhi to Mumbai
        1156.3  # km
    """
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    # Haversine formula
    a = np.sin(delta_lat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance_km = EARTH_RADIUS_KM * c

    return distance_km


def calculate_travel_time(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    road_type: RoadType = RoadType.STATE_HIGHWAY,
    speed_override_kmh: Optional[float] = None,
    circuity_factor: float = 1.3,
    use_osrm: bool = False
) -> TravelTimeResult:
    """
    Calculate realistic travel time between two locations

    Physics:
    - Uses haversine distance (great-circle, geodesic) by default
    - Optionally uses OSRM for real road network routing
    - Applies circuity factor (road distance > straight-line distance)
    - Road-type-specific average speeds
    - Accounts for urban congestion, terrain, road quality

    Args:
        origin: (latitude, longitude) of starting point
        destination: (latitude, longitude) of end point
        road_type: Type of road network (affects speed)
        speed_override_kmh: Custom average speed (overrides road_type default)
        circuity_factor: Road distance / straight-line distance
                        1.0 = perfect straight road
                        1.3 = typical (30% longer than straight line)
                        1.5 = winding/congested
                        2.0 = very circuitous
        use_osrm: If True, use OSRM API for real routing (requires internet)
                 If False, use haversine distance + circuity factor

    Returns:
        TravelTimeResult with distance, time, feasibility

    Examples:
        >>> # Delhi to Mumbai on national highway (haversine)
        >>> result = calculate_travel_time((28.7041, 77.1025), (19.0760, 72.8777))
        >>> print(f"{result.travel_time_hours:.1f} hours")
        25.0 hours  # ~1500 km at 60 km/h

        >>> # Use real OSRM routing
        >>> result = calculate_travel_time(
        ...     (28.7041, 77.1025), (19.0760, 72.8777),
        ...     use_osrm=True
        ... )
        >>> print(f"{result.travel_time_hours:.1f} hours, source: {result.source}")
        23.2 hours, source: osrm  # Real road network distance

        >>> # Override for faster highway
        >>> result = calculate_travel_time(
        ...     (28.7041, 77.1025), (19.0760, 72.8777),
        ...     speed_override_kmh=80  # Modern expressway
        ... )
        >>> print(f"{result.travel_time_hours:.1f} hours")
        18.8 hours
    """
    lat1, lon1 = origin
    lat2, lon2 = destination

    # Try OSRM if requested
    if use_osrm:
        try:
            from .osrm_client import OSRMClient
            from .config import get_config

            config = get_config()
            client = OSRMClient(endpoint=config.osrm_endpoint)
            osrm_result = client.route(origin, destination)

            if osrm_result:
                # OSRM provides real distance and duration
                distance_km = osrm_result.distance_km
                duration_hours = osrm_result.duration_hours
                avg_speed = distance_km / duration_hours if duration_hours > 0 else 0

                feasible = duration_hours <= 72.0
                notes = "Real routing via OSRM"
                if not feasible:
                    notes += f" (exceeds 72h limit)"

                return TravelTimeResult(
                    distance_km=distance_km,
                    travel_time_hours=duration_hours,
                    average_speed_kmh=avg_speed,
                    road_type=road_type,
                    feasible=feasible,
                    notes=notes,
                    source='osrm'
                )
        except Exception as e:
            # Fall back to haversine if OSRM fails
            # Don't raise error - graceful degradation
            pass

    # Haversine calculation (fallback or default)
    # Calculate geodesic distance
    straight_line_distance_km = haversine_distance(lat1, lon1, lat2, lon2)

    # Apply circuity factor (roads are not straight lines)
    actual_road_distance_km = straight_line_distance_km * circuity_factor

    # Determine average speed
    if speed_override_kmh is not None:
        average_speed_kmh = speed_override_kmh
    else:
        average_speed_kmh = road_type.value

    # Calculate travel time
    travel_time_hours = actual_road_distance_km / average_speed_kmh

    # Feasibility check (arbitrary limit: 72 hours = 3 days)
    feasible = travel_time_hours <= 72.0

    notes = ""
    if not feasible:
        notes = f"Travel time exceeds 72 hours (3 days)"

    return TravelTimeResult(
        distance_km=actual_road_distance_km,
        travel_time_hours=travel_time_hours,
        average_speed_kmh=average_speed_kmh,
        road_type=road_type,
        feasible=feasible,
        notes=notes,
        source='haversine'
    )


def calculate_max_reachable_distance(
    time_limit_hours: float,
    road_type: RoadType = RoadType.STATE_HIGHWAY,
    speed_override_kmh: Optional[float] = None
) -> float:
    """
    Calculate maximum distance reachable within time limit

    Args:
        time_limit_hours: Maximum travel time
        road_type: Type of road network
        speed_override_kmh: Custom speed (optional)

    Returns:
        Maximum reachable distance in kilometers

    Examples:
        >>> # How far can we reach in 24 hours on highway?
        >>> max_dist = calculate_max_reachable_distance(24, RoadType.NATIONAL_HIGHWAY)
        >>> print(f"{max_dist:.0f} km")
        1440 km  # 24 hours * 60 km/h
    """
    if speed_override_kmh is not None:
        speed = speed_override_kmh
    else:
        speed = road_type.value

    return time_limit_hours * speed


def estimate_travel_time_matrix(
    locations: list,
    road_type: RoadType = RoadType.STATE_HIGHWAY,
    circuity_factor: float = 1.3
) -> np.ndarray:
    """
    Calculate travel time matrix between all location pairs

    Args:
        locations: List of (lat, lon) tuples
        road_type: Road network type
        circuity_factor: Road distance multiplier

    Returns:
        NxN matrix of travel times (hours) where N = len(locations)

    Examples:
        >>> cities = [(28.7041, 77.1025), (19.0760, 72.8777), (12.9716, 77.5946)]
        >>> matrix = estimate_travel_time_matrix(cities)
        >>> matrix.shape
        (3, 3)
    """
    n = len(locations)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.0
            else:
                result = calculate_travel_time(
                    locations[i],
                    locations[j],
                    road_type=road_type,
                    circuity_factor=circuity_factor
                )
                matrix[i, j] = result.travel_time_hours

    return matrix
