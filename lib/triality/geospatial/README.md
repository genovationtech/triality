# Triality Geospatial Module - Production Ready

Physics-based geospatial analysis for logistics, infrastructure, and location feasibility with **real data integration**.

## Overview

The Triality geospatial module provides kill-switch logic for location-dependent systems. It works in **two modes**:

1. **Demo Mode** (default): Uses haversine distance + representative data for rapid prototyping
2. **Production Mode**: Integrates with real data sources (OSM, OSRM, WorldPop) for deployment

---

## Quick Start

### Demo Mode (No Setup Required)

```python
from triality.geospatial import calculate_travel_time, RoadType

# Delhi to Mumbai using haversine + circuity factor
result = calculate_travel_time(
    (28.7041, 77.1025),  # Delhi
    (19.0760, 72.8777),  # Mumbai
    road_type=RoadType.NATIONAL_HIGHWAY
)

print(f"Distance: {result.distance_km:.0f} km")
print(f"Travel time: {result.travel_time_hours:.1f} hours")
# Output: Distance: 1499 km, Travel time: 25.0 hours
```

### Production Mode (Real Routing)

```python
from triality.geospatial import calculate_travel_time

# Use OSRM for real road network routing
result = calculate_travel_time(
    (28.7041, 77.1025),  # Delhi
    (19.0760, 72.8777),  # Mumbai
    use_osrm=True  # Enable real routing
)

print(f"Distance: {result.distance_km:.0f} km (source: {result.source})")
# Output: Distance: 1423 km (source: osrm)
# Falls back to haversine if OSRM unavailable
```

---

## Production Data Integration

### 1. OSRM - Real Road Network Routing

**What it does**: Queries OSRM (Open Source Routing Machine) for real road network routes instead of haversine estimates.

**Setup**:

```python
from triality.geospatial import GeospatialConfig, get_config

# Option 1: Use public OSRM endpoint (rate limited)
config = get_config()  # Uses http://router.project-osrm.org by default

# Option 2: Self-hosted OSRM instance (recommended for production)
from triality.geospatial import GeospatialConfig
config = GeospatialConfig(osrm_endpoint="http://localhost:5000")
```

**Usage**:

```python
from triality.geospatial import calculate_travel_time

# Enable OSRM routing
result = calculate_travel_time(
    origin=(28.7041, 77.1025),
    destination=(19.0760, 72.8777),
    use_osrm=True  # Query OSRM API
)

# Check which routing method was used
if result.source == 'osrm':
    print("Real routing data used")
else:
    print("Fell back to haversine (OSRM unavailable)")
```

**Advanced - OSRM Client**:

```python
from triality.geospatial import OSRMClient

client = OSRMClient(endpoint="http://localhost:5000")

# Get detailed route
route = client.route(
    (28.7041, 77.1025),
    (19.0760, 72.8777),
    alternatives=True,  # Get alternative routes
    steps=True  # Get turn-by-turn directions
)

print(f"Distance: {route.distance_km:.1f} km")
print(f"Duration: {route.duration_hours:.2f} hours")

# Get distance matrix for multiple locations
cities = [
    (28.7041, 77.1025),  # Delhi
    (19.0760, 72.8777),  # Mumbai
    (12.9716, 77.5946),  # Bangalore
]

matrix = client.table(cities, cities)
print(matrix['durations_h'])  # NxN matrix of travel times
```

---

### 2. OpenStreetMap - Download Regional Data

**What it does**: Downloads OSM data files from Geofabrik for offline use.

**Usage**:

```python
from triality.geospatial import GeospatialDataLoader, list_osm_regions

# List available regions
regions = list_osm_regions()
print(regions)
# ['india', 'usa', 'uk', 'germany', 'france', 'china', 'japan', 'brazil']

# Download OSM data for India
loader = GeospatialDataLoader()
osm_file = loader.download_osm('india')
# Downloading OSM data for india...
# URL: https://download.geofabrik.de/asia/india-latest.osm.pbf
# Downloaded: /root/.triality/geospatial_data/osm/india-latest.osm.pbf
# Size: 891.2 MB

# Check if region data exists
path = loader.get_osm_path('india')
if path:
    print(f"OSM data available: {path}")
```

**Quick download function**:

```python
from triality.geospatial import download_osm_data

# One-liner to download OSM data
osm_file = download_osm_data('india')
```

---

### 3. WorldPop - Population Grids (Manual Download)

**What it does**: Provides instructions for downloading high-resolution population data.

**Note**: WorldPop requires manual download or API key.

```python
from triality.geospatial import GeospatialDataLoader

loader = GeospatialDataLoader()
pop_file = loader.download_worldpop('IND', year=2020, resolution='1km')

# Prints instructions:
# WorldPop download requires manual process or API key.
# Visit: https://www.worldpop.org/geodata/listing?id=29
# Download file and place in: /root/.triality/geospatial_data/population/
# Expected filename: IND_2020_1km.tif
```

---

## Configuration

### Data Directory

All downloaded data is cached in `~/.triality/geospatial_data/`:

```
~/.triality/geospatial_data/
├── osm/               # OSM PBF files
│   ├── india-latest.osm.pbf
│   └── usa-latest.osm.pbf
├── population/        # WorldPop rasters
│   └── IND_2020_1km.tif
└── cache/            # OSRM query cache
```

**Custom data directory**:

```python
from pathlib import Path
from triality.geospatial import GeospatialConfig

config = GeospatialConfig(
    data_dir=Path('/data/geospatial'),
    osrm_endpoint='http://localhost:5000',
    use_cache=True  # Cache OSRM queries
)
```

---

## Use Cases

### Use Case 1: Warehouse Network Feasibility (Demo)

```python
from triality.geospatial import GeospatialFeasibilityChecker

# Check if 3 warehouses can cover 95% of India in 24 hours
warehouses = [
    (19.0760, 72.8777),  # Mumbai
    (28.7041, 77.1025),  # Delhi
    (12.9716, 77.5946),  # Bangalore
]

result = GeospatialFeasibilityChecker.check_24h_coverage(
    warehouses,
    target_coverage=0.95,
    time_limit_hours=24.0
)

print(f"Status: {result.status}")
print(f"Coverage: {result.coverage_achieved*100:.1f}%")
# Status: MARGINAL
# Coverage: 86.2%
```

### Use Case 2: Production Deployment with OSRM

```python
from triality.geospatial import (
    GeospatialConfig,
    calculate_travel_time,
    GeospatialFeasibilityChecker
)

# Configure for production
config = GeospatialConfig(
    osrm_endpoint='http://osrm.company.internal:5000',
    use_cache=True
)

# Check warehouse coverage using real routing
result = GeospatialFeasibilityChecker.check_24h_coverage(
    warehouses,
    target_coverage=0.95,
    use_osrm=True  # Use real road network
)
```

### Use Case 3: Download OSM Data for Offline Use

```python
from triality.geospatial import download_osm_data

# Download multiple regions
regions = ['india', 'usa', 'uk']

for region in regions:
    print(f"Downloading {region}...")
    osm_file = download_osm_data(region)
    print(f"Saved: {osm_file}")
```

---

## API Reference

### Travel Time Functions

```python
calculate_travel_time(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    road_type: RoadType = RoadType.STATE_HIGHWAY,
    speed_override_kmh: Optional[float] = None,
    circuity_factor: float = 1.3,
    use_osrm: bool = False  # NEW: Enable real routing
) -> TravelTimeResult
```

### OSRM Client

```python
OSRMClient(endpoint: str = "http://router.project-osrm.org")
    .route(origin, destination, alternatives=False, steps=False) -> OSRMRouteResult
    .table(sources, destinations) -> Dict[str, np.ndarray]
```

### Data Loader

```python
GeospatialDataLoader(data_dir: Optional[Path] = None)
    .download_osm(region: str, force: bool = False) -> Path
    .download_worldpop(country_code: str, year: int, resolution: str) -> Path
    .list_available_regions() -> List[str]
    .get_osm_path(region: str) -> Optional[Path]
    .list_downloaded_datasets() -> List[DatasetInfo]
```

### Configuration

```python
GeospatialConfig(
    data_dir: Optional[Path] = None,  # Default: ~/.triality/geospatial_data
    osrm_endpoint: str = "http://router.project-osrm.org",
    use_cache: bool = True
)

get_config() -> GeospatialConfig  # Get default config
```

---

## Graceful Degradation

The module is designed to work **with or without** external data sources:

1. **OSRM unavailable** → Falls back to haversine + circuity factor
2. **OSM data not downloaded** → Uses representative population centers
3. **Internet unavailable** → All calculations work offline with demo data

This ensures Triality demos run immediately without setup, while production deployments can integrate real data sources.

---

## Dependencies

### Required (already installed):
- NumPy

### Optional (for production features):
- `requests` - For OSRM API and OSM downloads
- `rasterio` - For WorldPop raster data (future)
- `geopandas` - For advanced GIS operations (future)

Install optional dependencies:
```bash
pip install requests rasterio geopandas
```

---

## Performance Notes

### Haversine Mode (Demo)
- **Speed**: Instant (<1ms per route)
- **Accuracy**: ±10-30% (uses 1.3x circuity factor)
- **Use case**: Prototyping, demos, order-of-magnitude estimates

### OSRM Mode (Production)
- **Speed**: 50-200ms per route (network latency)
- **Accuracy**: Real road network distance (±1%)
- **Use case**: Production deployments, detailed planning

### Caching
- OSRM queries are cached locally to reduce API calls
- Cache location: `~/.triality/geospatial_data/cache/`

---

## Examples

### Example 1: Compare Haversine vs OSRM

```python
from triality.geospatial import calculate_travel_time

delhi = (28.7041, 77.1025)
mumbai = (19.0760, 72.8777)

# Haversine estimate
result_demo = calculate_travel_time(delhi, mumbai)
print(f"Haversine: {result_demo.distance_km:.0f} km, {result_demo.travel_time_hours:.1f}h")

# OSRM real routing
result_prod = calculate_travel_time(delhi, mumbai, use_osrm=True)
print(f"OSRM: {result_prod.distance_km:.0f} km, {result_prod.travel_time_hours:.1f}h")

# Haversine: 1499 km, 25.0h
# OSRM: 1423 km, 23.2h (if API available)
```

### Example 2: Self-Hosted OSRM Setup

```bash
# Download OSM data
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-extract \
  -p /opt/car.lua /data/india-latest.osm.pbf

# Prepare routing graph
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-partition /data/india-latest.osm.pbf
docker run -t -v "${PWD}:/data" osrm/osrm-backend osrm-customize /data/india-latest.osm.pbf

# Run OSRM server
docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend \
  osrm-routed --algorithm mld /data/india-latest.osm.pbf
```

Then use in Triality:
```python
from triality.geospatial import GeospatialConfig

config = GeospatialConfig(osrm_endpoint="http://localhost:5000")
```

---

## Philosophy

**Physics-grounded by default, works with real data**:

1. Demo mode uses physics-based estimates (haversine, circuity, representative speeds)
2. Production mode integrates real data sources (OSRM, OSM, WorldPop)
3. Graceful degradation ensures robustness
4. Kill-switch logic remains the same (geography + physics → constraints)

---

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
