"""
OSRM Client for Real Routing

Integrates with OSRM (Open Source Routing Machine) for production-grade
travel time calculations using actual road networks.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import json
import hashlib
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .config import get_config


@dataclass
class OSRMRouteResult:
    """Result from OSRM routing query"""
    distance_km: float
    duration_hours: float
    geometry: Optional[str] = None
    source: str = "osrm"


class OSRMClient:
    """
    Client for OSRM routing API

    Provides real-world travel times and distances using actual road networks.
    Supports caching to reduce API calls.

    Examples:
        >>> client = OSRMClient()
        >>> route = client.route((28.7041, 77.1025), (19.0760, 72.8777))
        >>> print(f"{route.distance_km:.0f} km in {route.duration_hours:.1f} hours")
        1400 km in 18.5 hours  # Real routing via highways
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        profile: str = 'car',
        use_cache: bool = True
    ):
        """
        Initialize OSRM client

        Args:
            endpoint: OSRM server endpoint (default: public instance)
            profile: Routing profile ('car', 'bike', 'foot')
            use_cache: Cache routing results locally
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library required for OSRM integration. "
                "Install with: pip install requests"
            )

        config = get_config()
        self.endpoint = endpoint or config.osrm_endpoint
        self.profile = profile
        self.use_cache = use_cache and config.use_cache
        self.cache_dir = config.data_dir / 'cache' if self.use_cache else None

    def route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        alternatives: bool = False
    ) -> Optional[OSRMRouteResult]:
        """
        Calculate route between two points using OSRM

        Args:
            origin: (latitude, longitude) of start point
            destination: (latitude, longitude) of end point
            alternatives: Return alternative routes

        Returns:
            OSRMRouteResult or None if routing fails

        Examples:
            >>> client = OSRMClient()
            >>> route = client.route((28.7041, 77.1025), (19.0760, 72.8777))
            >>> route.distance_km
            1398.5
            >>> route.duration_hours
            18.2
        """
        # Check cache first
        if self.use_cache:
            cached = self._get_cached_route(origin, destination)
            if cached:
                return cached

        # Build OSRM query URL
        # Format: /route/v1/{profile}/{lon},{lat};{lon},{lat}
        url = (
            f"{self.endpoint}/route/v1/{self.profile}/"
            f"{origin[1]},{origin[0]};{destination[1]},{destination[0]}"
        )

        params = {
            'overview': 'false',  # Don't need detailed geometry
            'alternatives': 'true' if alternatives else 'false',
            'steps': 'false'
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get('code') != 'Ok':
                print(f"OSRM routing failed: {data.get('code')}")
                return None

            routes = data.get('routes', [])
            if not routes:
                return None

            # Use first (best) route
            route = routes[0]
            distance_m = route.get('distance', 0)
            duration_s = route.get('duration', 0)

            result = OSRMRouteResult(
                distance_km=distance_m / 1000.0,
                duration_hours=duration_s / 3600.0,
                geometry=route.get('geometry'),
                source='osrm'
            )

            # Cache result
            if self.use_cache:
                self._cache_route(origin, destination, result)

            return result

        except requests.exceptions.RequestException as e:
            print(f"OSRM API error: {e}")
            return None

    def table(
        self,
        sources: list,
        destinations: list
    ) -> Optional[Dict]:
        """
        Calculate distance/duration matrix

        Args:
            sources: List of (lat, lon) source points
            destinations: List of (lat, lon) destination points

        Returns:
            Dictionary with 'durations' and 'distances' matrices

        Examples:
            >>> client = OSRMClient()
            >>> sources = [(28.7041, 77.1025), (19.0760, 72.8777)]
            >>> dests = [(12.9716, 77.5946), (22.5726, 88.3639)]
            >>> matrix = client.table(sources, dests)
            >>> matrix['durations']  # Hours
            [[25.2, 32.1], [15.8, 28.9]]
        """
        # Build coordinates string
        coords = []
        for lat, lon in sources + destinations:
            coords.append(f"{lon},{lat}")

        coord_string = ";".join(coords)

        # Build source/destination indices
        source_indices = list(range(len(sources)))
        dest_indices = list(range(len(sources), len(sources) + len(destinations)))

        url = f"{self.endpoint}/table/v1/{self.profile}/{coord_string}"
        params = {
            'sources': ';'.join(map(str, source_indices)),
            'destinations': ';'.join(map(str, dest_indices))
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()

            data = response.json()

            if data.get('code') != 'Ok':
                print(f"OSRM table query failed: {data.get('code')}")
                return None

            # Convert to hours and kilometers
            durations_s = data.get('durations', [])
            distances_m = data.get('distances', [])

            durations_h = [[d / 3600.0 if d is not None else None for d in row] for row in durations_s]
            distances_km = [[d / 1000.0 if d is not None else None for d in row] for row in distances_m]

            return {
                'durations': durations_h,
                'distances': distances_km
            }

        except requests.exceptions.RequestException as e:
            print(f"OSRM table API error: {e}")
            return None

    def _get_cache_key(self, origin: Tuple[float, float], destination: Tuple[float, float]) -> str:
        """Generate cache key for route"""
        key_str = f"{origin[0]:.4f},{origin[1]:.4f}-{destination[0]:.4f},{destination[1]:.4f}-{self.profile}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_cached_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float]
    ) -> Optional[OSRMRouteResult]:
        """Retrieve cached route result"""
        if not self.cache_dir:
            return None

        cache_key = self._get_cache_key(origin, destination)
        cache_file = self.cache_dir / f'route_{cache_key}.json'

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    return OSRMRouteResult(**data)
            except (json.JSONDecodeError, TypeError):
                return None

        return None

    def _cache_route(
        self,
        origin: Tuple[float, float],
        destination: Tuple[float, float],
        result: OSRMRouteResult
    ):
        """Cache route result"""
        if not self.cache_dir:
            return

        cache_key = self._get_cache_key(origin, destination)
        cache_file = self.cache_dir / f'route_{cache_key}.json'

        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'distance_km': result.distance_km,
                    'duration_hours': result.duration_hours,
                    'source': result.source
                }, f)
        except IOError:
            pass  # Caching is best-effort


def calculate_osrm_travel_time(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    profile: str = 'car'
) -> Optional[Tuple[float, float]]:
    """
    Quick function to get travel time via OSRM

    Args:
        origin: (lat, lon) start point
        destination: (lat, lon) end point
        profile: Routing profile ('car', 'bike', 'foot')

    Returns:
        (distance_km, duration_hours) or None if failed

    Examples:
        >>> result = calculate_osrm_travel_time((28.7041, 77.1025), (19.0760, 72.8777))
        >>> if result:
        ...     dist, time = result
        ...     print(f"{dist:.0f} km, {time:.1f} hours")
        1400 km, 18.5 hours
    """
    if not REQUESTS_AVAILABLE:
        return None

    client = OSRMClient(profile=profile)
    route = client.route(origin, destination)

    if route:
        return (route.distance_km, route.duration_hours)

    return None
