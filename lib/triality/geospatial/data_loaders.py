"""
Data Loaders for Geospatial Datasets

Downloads and manages OSM, WorldPop, and other geospatial data sources
for production use.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .config import get_config


@dataclass
class DatasetInfo:
    """Information about a downloaded dataset"""
    name: str
    region: str
    source: str
    file_path: Path
    metadata: Dict


class GeospatialDataLoader:
    """
    Manager for downloading and caching geospatial datasets

    Supports:
    - OpenStreetMap extracts (Geofabrik)
    - WorldPop population grids
    - Census data
    """

    # Geofabrik OSM download URLs (regional extracts)
    GEOFABRIK_BASE = "https://download.geofabrik.de"
    GEOFABRIK_REGIONS = {
        'india': 'asia/india-latest.osm.pbf',
        'usa': 'north-america/us-latest.osm.pbf',
        'uk': 'europe/great-britain-latest.osm.pbf',
        'germany': 'europe/germany-latest.osm.pbf',
        'france': 'europe/france-latest.osm.pbf',
        'china': 'asia/china-latest.osm.pbf',
        'japan': 'asia/japan-latest.osm.pbf',
        'brazil': 'south-america/brazil-latest.osm.pbf',
    }

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader

        Args:
            data_dir: Directory for storing data (default: ~/.triality/geospatial_data)
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library required for data downloads. "
                "Install with: pip install requests"
            )

        config = get_config()
        self.data_dir = data_dir or config.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_osm(
        self,
        region: str,
        force: bool = False
    ) -> Optional[Path]:
        """
        Download OpenStreetMap data for a region

        Args:
            region: Region name ('india', 'usa', 'uk', etc.)
            force: Force re-download even if file exists

        Returns:
            Path to downloaded OSM file or None if failed

        Examples:
            >>> loader = GeospatialDataLoader()
            >>> osm_file = loader.download_osm('india')
            Downloading OSM data for India...
            Downloaded: /home/user/.triality/geospatial_data/osm/india-latest.osm.pbf
            >>> osm_file
            PosixPath('/home/user/.triality/geospatial_data/osm/india-latest.osm.pbf')
        """
        if region not in self.GEOFABRIK_REGIONS:
            print(f"Region '{region}' not available. "
                  f"Available: {', '.join(self.GEOFABRIK_REGIONS.keys())}")
            return None

        osm_dir = self.data_dir / 'osm'
        osm_dir.mkdir(exist_ok=True)

        osm_file = osm_dir / f'{region}-latest.osm.pbf'

        # Check if already downloaded
        if osm_file.exists() and not force:
            print(f"OSM data already exists: {osm_file}")
            return osm_file

        # Download from Geofabrik
        url = f"{self.GEOFABRIK_BASE}/{self.GEOFABRIK_REGIONS[region]}"
        print(f"Downloading OSM data for {region}...")
        print(f"URL: {url}")
        print(f"This may take several minutes for large regions...")

        try:
            response = requests.get(url, stream=True, timeout=600)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1 MB

            with open(osm_file, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(block_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded / 1e6:.1f} MB)", end='')

            print(f"\nDownloaded: {osm_file}")
            print(f"Size: {osm_file.stat().st_size / 1e6:.1f} MB")

            # Save metadata
            self._save_metadata(osm_file, {
                'region': region,
                'source': 'geofabrik',
                'url': url,
                'type': 'osm'
            })

            return osm_file

        except requests.exceptions.RequestException as e:
            print(f"Download failed: {e}")
            if osm_file.exists():
                osm_file.unlink()  # Remove partial download
            return None

    def list_available_regions(self) -> List[str]:
        """
        List available regions for OSM download

        Returns:
            List of region names

        Examples:
            >>> loader = GeospatialDataLoader()
            >>> regions = loader.list_available_regions()
            >>> print(regions)
            ['india', 'usa', 'uk', 'germany', 'france', 'china', 'japan', 'brazil']
        """
        return list(self.GEOFABRIK_REGIONS.keys())

    def get_osm_path(self, region: str) -> Optional[Path]:
        """
        Get path to OSM data file if it exists

        Args:
            region: Region name

        Returns:
            Path to OSM file or None if not downloaded

        Examples:
            >>> loader = GeospatialDataLoader()
            >>> path = loader.get_osm_path('india')
            >>> if path and path.exists():
            ...     print(f"OSM data available: {path}")
        """
        osm_file = self.data_dir / 'osm' / f'{region}-latest.osm.pbf'
        return osm_file if osm_file.exists() else None

    def download_worldpop(
        self,
        country_code: str,
        year: int = 2020,
        resolution: str = '1km'
    ) -> Optional[Path]:
        """
        Download WorldPop population data

        NOTE: This is a placeholder. Real implementation would use WorldPop API.
        For production, use WorldPop REST API or download directly from:
        https://www.worldpop.org/geodata/listing?id=29

        Args:
            country_code: ISO3 country code (e.g., 'IND', 'USA')
            year: Data year
            resolution: Grid resolution ('1km', '100m')

        Returns:
            Path to population raster file

        Examples:
            >>> loader = GeospatialDataLoader()
            >>> pop_file = loader.download_worldpop('IND', year=2020)
            WorldPop download requires manual process.
            Visit: https://www.worldpop.org/geodata/listing?id=29
            Download file and place in: /home/user/.triality/geospatial_data/population/
        """
        pop_dir = self.data_dir / 'population'
        pop_dir.mkdir(exist_ok=True)

        print("WorldPop download requires manual process or API key.")
        print("Visit: https://www.worldpop.org/geodata/listing?id=29")
        print(f"Download file and place in: {pop_dir}")
        print(f"Expected filename: {country_code}_{year}_{resolution}.tif")

        expected_file = pop_dir / f"{country_code}_{year}_{resolution}.tif"
        return expected_file if expected_file.exists() else None

    def _save_metadata(self, data_file: Path, metadata: Dict):
        """Save metadata for downloaded dataset"""
        metadata_file = data_file.with_suffix('.json')
        try:
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except IOError:
            pass  # Metadata is optional

    def list_downloaded_datasets(self) -> List[DatasetInfo]:
        """
        List all downloaded datasets

        Returns:
            List of DatasetInfo objects

        Examples:
            >>> loader = GeospatialDataLoader()
            >>> datasets = loader.list_downloaded_datasets()
            >>> for ds in datasets:
            ...     print(f"{ds.name}: {ds.file_path}")
            india-osm: /home/user/.triality/geospatial_data/osm/india-latest.osm.pbf
        """
        datasets = []

        # Find OSM files
        osm_dir = self.data_dir / 'osm'
        if osm_dir.exists():
            for osm_file in osm_dir.glob('*.osm.pbf'):
                metadata_file = osm_file.with_suffix('.json')
                metadata = {}
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                    except json.JSONDecodeError:
                        pass

                datasets.append(DatasetInfo(
                    name=osm_file.stem,
                    region=metadata.get('region', 'unknown'),
                    source=metadata.get('source', 'osm'),
                    file_path=osm_file,
                    metadata=metadata
                ))

        # Find population files
        pop_dir = self.data_dir / 'population'
        if pop_dir.exists():
            for pop_file in pop_dir.glob('*.tif'):
                datasets.append(DatasetInfo(
                    name=pop_file.stem,
                    region='unknown',
                    source='worldpop',
                    file_path=pop_file,
                    metadata={}
                ))

        return datasets


# Convenience functions
def download_osm_data(region: str, output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Quick function to download OSM data

    Args:
        region: Region name ('india', 'usa', etc.)
        output_dir: Output directory (optional)

    Returns:
        Path to downloaded file

    Examples:
        >>> from triality.geospatial import download_osm_data
        >>> osm_file = download_osm_data('india')
        Downloading OSM data for india...
        Downloaded: /home/user/.triality/geospatial_data/osm/india-latest.osm.pbf
    """
    loader = GeospatialDataLoader(data_dir=output_dir)
    return loader.download_osm(region)


def list_osm_regions() -> List[str]:
    """
    List available OSM regions

    Returns:
        List of region names

    Examples:
        >>> from triality.geospatial import list_osm_regions
        >>> regions = list_osm_regions()
        >>> print(regions)
        ['india', 'usa', 'uk', 'germany', ...]
    """
    loader = GeospatialDataLoader()
    return loader.list_available_regions()
