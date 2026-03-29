"""
Configuration for Triality Geospatial Module

Manages data paths, API endpoints, and caching for production use.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import os
from pathlib import Path
from typing import Optional


# Default data directory (relative to user's home)
DEFAULT_DATA_DIR = Path.home() / '.triality' / 'geospatial_data'

# API endpoints
OSRM_API_BASE = "http://router.project-osrm.org"  # Public OSRM instance
OVERPASS_API_BASE = "https://overpass-api.de/api"
WORLDPOP_API_BASE = "https://www.worldpop.org/rest/data"


class GeospatialConfig:
    """Configuration manager for geospatial data and APIs"""

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        osrm_endpoint: Optional[str] = None,
        use_cache: bool = True
    ):
        """
        Initialize geospatial configuration

        Args:
            data_dir: Directory for cached data (default: ~/.triality/geospatial_data)
            osrm_endpoint: Custom OSRM endpoint (default: public instance)
            use_cache: Whether to cache downloaded data
        """
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        self.osrm_endpoint = osrm_endpoint or OSRM_API_BASE
        self.use_cache = use_cache

        # Create data directories if they don't exist
        if use_cache:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            (self.data_dir / 'osm').mkdir(exist_ok=True)
            (self.data_dir / 'population').mkdir(exist_ok=True)
            (self.data_dir / 'cache').mkdir(exist_ok=True)

    def get_osm_path(self, region: str) -> Path:
        """Get path for OSM data file"""
        return self.data_dir / 'osm' / f'{region}.osm.pbf'

    def get_population_path(self, region: str) -> Path:
        """Get path for population data file"""
        return self.data_dir / 'population' / f'{region}_pop.tif'

    def get_cache_path(self, cache_key: str) -> Path:
        """Get path for cached result"""
        return self.data_dir / 'cache' / f'{cache_key}.json'


# Global configuration instance
_config = GeospatialConfig()


def get_config() -> GeospatialConfig:
    """Get global geospatial configuration"""
    return _config


def set_data_dir(data_dir: Path):
    """Set global data directory"""
    global _config
    _config.data_dir = Path(data_dir)
    _config.data_dir.mkdir(parents=True, exist_ok=True)


def set_osrm_endpoint(endpoint: str):
    """Set custom OSRM endpoint (for self-hosted OSRM)"""
    global _config
    _config.osrm_endpoint = endpoint
