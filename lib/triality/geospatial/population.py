"""
Population Analysis

Provides population estimation and coverage calculations for geospatial feasibility.
Uses simplified representative data for India and other regions.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass

from .travel_time import haversine_distance


# Representative population centers for India (simplified)
# Real implementation would use WorldPop or Census data
INDIA_POPULATION_CENTERS = [
    # (latitude, longitude, population_millions, name)
    (28.7041, 77.1025, 30.0, 'Delhi NCR'),
    (19.0760, 72.8777, 20.0, 'Mumbai'),
    (12.9716, 77.5946, 12.0, 'Bangalore'),
    (22.5726, 88.3639, 15.0, 'Kolkata'),
    (13.0827, 80.2707, 10.0, 'Chennai'),
    (17.3850, 78.4867, 10.0, 'Hyderabad'),
    (23.0225, 72.5714, 8.0, 'Ahmedabad'),
    (18.5204, 73.8567, 6.5, 'Pune'),
    (26.9124, 75.7873, 3.5, 'Jaipur'),
    (30.7333, 76.7794, 1.5, 'Chandigarh'),
    (25.5941, 85.1376, 2.5, 'Patna'),
    (26.8467, 80.9462, 3.5, 'Lucknow'),
    (21.1458, 79.0882, 3.0, 'Nagpur'),
    (15.3173, 75.7139, 0.5, 'Hubli'),
    (11.0168, 76.9558, 1.0, 'Coimbatore'),
    (10.8505, 76.2711, 0.8, 'Thrissur'),
    (23.8103, 91.4000, 0.5, 'Agartala'),
    (24.8829, 91.8790, 0.4, 'Silchar'),
]

# Total India population (approximate)
INDIA_TOTAL_POPULATION_MILLIONS = sum(pop for _, _, pop, _ in INDIA_POPULATION_CENTERS)


@dataclass
class PopulationCoverageResult:
    """Result of population coverage analysis"""
    total_population_millions: float
    covered_population_millions: float
    coverage_fraction: float
    uncovered_population_millions: float
    uncovered_regions: List[str]


def estimate_population_in_radius(
    center: Tuple[float, float],
    radius_km: float,
    country: str = 'india'
) -> float:
    """
    Estimate population within radius of a location

    Simplified approach: Sums populations of cities within radius

    Args:
        center: (latitude, longitude) of center point
        radius_km: Radius in kilometers
        country: Country code ('india' supported)

    Returns:
        Estimated population in millions

    Examples:
        >>> # Population within 500 km of Mumbai
        >>> pop = estimate_population_in_radius((19.0760, 72.8777), 500)
        >>> print(f"{pop:.1f} million people")
        35.0 million people
    """
    if country.lower() == 'india':
        population_centers = INDIA_POPULATION_CENTERS
    else:
        raise ValueError(f"Country '{country}' not supported. Use 'india'.")

    total_pop = 0.0

    for lat, lon, pop, name in population_centers:
        distance = haversine_distance(center[0], center[1], lat, lon)
        if distance <= radius_km:
            total_pop += pop

    return total_pop


def calculate_population_coverage(
    service_centers: List[Tuple[float, float]],
    radius_km: float,
    country: str = 'india'
) -> PopulationCoverageResult:
    """
    Calculate population coverage from service centers

    Args:
        service_centers: List of (lat, lon) for warehouses/facilities
        radius_km: Service radius from each center
        country: Country code

    Returns:
        PopulationCoverageResult with coverage statistics

    Examples:
        >>> # Coverage from Mumbai alone (500 km radius)
        >>> result = calculate_population_coverage([(19.0760, 72.8777)], 500)
        >>> print(f"{result.coverage_fraction*100:.1f}% of India")
        28.0% of India

        >>> # Coverage from Mumbai + Delhi
        >>> result = calculate_population_coverage([
        ...     (19.0760, 72.8777),  # Mumbai
        ...     (28.7041, 77.1025)   # Delhi
        ... ], 500)
        >>> print(f"{result.coverage_fraction*100:.1f}% of India")
        64.0% of India
    """
    if country.lower() == 'india':
        population_centers = INDIA_POPULATION_CENTERS
        total_population = INDIA_TOTAL_POPULATION_MILLIONS
    else:
        raise ValueError(f"Country '{country}' not supported. Use 'india'.")

    covered_pop = 0.0
    uncovered_regions = []

    for lat, lon, pop, name in population_centers:
        is_covered = False

        # Check if covered by any service center
        for service_center in service_centers:
            distance = haversine_distance(
                service_center[0], service_center[1],
                lat, lon
            )
            if distance <= radius_km:
                is_covered = True
                break

        if is_covered:
            covered_pop += pop
        else:
            uncovered_regions.append(name)

    coverage_fraction = covered_pop / total_population
    uncovered_pop = total_population - covered_pop

    return PopulationCoverageResult(
        total_population_millions=total_population,
        covered_population_millions=covered_pop,
        coverage_fraction=coverage_fraction,
        uncovered_population_millions=uncovered_pop,
        uncovered_regions=uncovered_regions
    )


def get_india_population_centers() -> List[Tuple[float, float, float, str]]:
    """
    Get representative population centers for India

    Returns:
        List of (lat, lon, population_millions, name)

    Examples:
        >>> centers = get_india_population_centers()
        >>> len(centers)
        18
        >>> centers[0]  # Delhi
        (28.7041, 77.1025, 30.0, 'Delhi NCR')
    """
    return INDIA_POPULATION_CENTERS.copy()


def get_population_density_estimate(
    location: Tuple[float, float],
    country: str = 'india'
) -> float:
    """
    Estimate population density at a location

    Simplified: Returns density based on distance to nearest major city

    Args:
        location: (latitude, longitude)
        country: Country code

    Returns:
        Estimated population density (people per km²)

    Examples:
        >>> # Density in Mumbai
        >>> density = get_population_density_estimate((19.0760, 72.8777))
        >>> print(f"{density:.0f} people/km²")
        20000 people/km²  # Urban core

        >>> # Density in rural area
        >>> density = get_population_density_estimate((20.0, 75.0))
        >>> print(f"{density:.0f} people/km²")
        500 people/km²  # Rural
    """
    if country.lower() == 'india':
        population_centers = INDIA_POPULATION_CENTERS
    else:
        raise ValueError(f"Country '{country}' not supported. Use 'india'.")

    # Find nearest city
    min_distance = float('inf')
    nearest_pop = 0.0

    for lat, lon, pop, name in population_centers:
        distance = haversine_distance(location[0], location[1], lat, lon)
        if distance < min_distance:
            min_distance = distance
            nearest_pop = pop

    # Simple model: density decreases with distance from city
    if min_distance < 10:
        # Urban core
        density = 20000 * (nearest_pop / 10)  # Scale with city size
    elif min_distance < 50:
        # Suburban
        density = 5000 * (nearest_pop / 10)
    elif min_distance < 100:
        # Peri-urban
        density = 1000 * (nearest_pop / 10)
    elif min_distance < 200:
        # Rural with nearby city influence
        density = 500
    else:
        # Remote rural
        density = 200

    return density
