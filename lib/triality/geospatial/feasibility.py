"""
Geospatial Feasibility Checker

Kill-switch logic for location-dependent systems. Checks whether logistics,
infrastructure, or service requirements are physically achievable given
geographic, population, and network constraints.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum

from .travel_time import calculate_travel_time, RoadType
from .isochrones import calculate_multi_warehouse_coverage
from .population import (
    calculate_population_coverage,
    get_india_population_centers
)


class FeasibilityStatus(Enum):
    """Feasibility check result"""
    FEASIBLE = "feasible"
    MARGINAL = "marginal"
    NOT_FEASIBLE = "not_feasible"


@dataclass
class FeasibilityIssue:
    """Description of a feasibility problem"""
    gate: str
    severity: str  # 'showstopper', 'critical', 'warning'
    issue: str
    physics: str
    recommendation: str
    key_numbers: Dict[str, float]


@dataclass
class GeospatialFeasibilityResult:
    """Result of geospatial feasibility analysis"""
    status: FeasibilityStatus
    coverage_achieved: float
    coverage_required: float
    issues: List[FeasibilityIssue]
    recommendations: List[str]
    warehouse_count: int
    total_population_millions: float
    covered_population_millions: float


class GeospatialFeasibilityChecker:
    """
    Geospatial kill-switch checker for logistics and location-based systems

    Checks whether geographic constraints allow system requirements to be met.
    """

    @staticmethod
    def check_24h_coverage(
        warehouse_locations: List[Tuple[float, float]],
        target_coverage: float = 0.95,
        time_limit_hours: float = 24.0,
        country: str = 'india',
        road_type: RoadType = RoadType.STATE_HIGHWAY
    ) -> GeospatialFeasibilityResult:
        """
        Check if warehouse network can deliver to target coverage in time limit

        Kill-Switch Logic:
        - Geography + road physics → minimum warehouse count
        - Population distribution → coverage gaps
        - Physics cannot be negotiated

        Args:
            warehouse_locations: List of (lat, lon) for warehouses
            target_coverage: Required population coverage (0-1)
            time_limit_hours: Maximum delivery time
            country: Country code
            road_type: Road network type

        Returns:
            GeospatialFeasibilityResult with kill-switch analysis

        Examples:
            >>> # Check single warehouse in Mumbai
            >>> result = GeospatialFeasibilityChecker.check_24h_coverage(
            ...     [(19.0760, 72.8777)],
            ...     target_coverage=0.95
            ... )
            >>> print(result.status)
            FeasibilityStatus.NOT_FEASIBLE

            >>> # Check 3 warehouses
            >>> result = GeospatialFeasibilityChecker.check_24h_coverage([
            ...     (19.0760, 72.8777),  # Mumbai
            ...     (28.7041, 77.1025),  # Delhi
            ...     (12.9716, 77.5946)   # Bangalore
            ... ])
            >>> print(result.status)
            FeasibilityStatus.FEASIBLE
        """
        issues = []
        recommendations = []

        # Get population centers
        if country.lower() == 'india':
            population_centers = get_india_population_centers()
            pop_with_coords = [(lat, lon, pop) for lat, lon, pop, _ in population_centers]
        else:
            raise ValueError(f"Country '{country}' not supported")

        # Calculate actual coverage
        coverage_fraction, covered_pop = calculate_multi_warehouse_coverage(
            warehouse_locations,
            pop_with_coords,
            time_limit_hours,
            road_type
        )

        total_pop = sum(pop for _, _, pop in pop_with_coords)

        # GATE 1: Coverage Requirement
        coverage_gap = target_coverage - coverage_fraction

        if coverage_gap > 0.05:  # >5% gap = showstopper
            issues.append(FeasibilityIssue(
                gate="Coverage Requirement",
                severity="showstopper",
                issue=f"Only {coverage_fraction*100:.1f}% coverage achieved, need {target_coverage*100:.0f}%",
                physics=f"Geography + road network physics → cannot reach {coverage_gap*100:.1f}% of population in {time_limit_hours}h",
                recommendation=f"Add {estimate_additional_warehouses_needed(coverage_gap)} more warehouses OR relax coverage requirement",
                key_numbers={
                    'coverage_achieved': coverage_fraction,
                    'coverage_required': target_coverage,
                    'gap_percent': coverage_gap * 100,
                    'uncovered_population_millions': total_pop * coverage_gap
                }
            ))

        elif coverage_gap > 0:  # 0-5% gap = marginal
            issues.append(FeasibilityIssue(
                gate="Coverage Requirement",
                severity="critical",
                issue=f"{coverage_fraction*100:.1f}% coverage (marginal, need {target_coverage*100:.0f}%)",
                physics="Close but not meeting SLA",
                recommendation="Add 1 more warehouse for margin OR accept risk",
                key_numbers={
                    'coverage_achieved': coverage_fraction,
                    'coverage_required': target_coverage,
                    'gap_percent': coverage_gap * 100
                }
            ))

        # GATE 2: Single-Point-of-Failure Risk
        if len(warehouse_locations) == 1:
            issues.append(FeasibilityIssue(
                gate="Reliability",
                severity="critical",
                issue="Single warehouse = single point of failure",
                physics="Any disruption → 100% service loss",
                recommendation="Minimum 2 warehouses for redundancy",
                key_numbers={
                    'warehouse_count': len(warehouse_locations),
                    'minimum_recommended': 2
                }
            ))

        # GATE 3: Regional Balance
        if len(warehouse_locations) >= 2:
            # Check if warehouses are too clustered
            min_separation_km = float('inf')
            from .travel_time import haversine_distance

            for i in range(len(warehouse_locations)):
                for j in range(i + 1, len(warehouse_locations)):
                    dist = haversine_distance(
                        warehouse_locations[i][0], warehouse_locations[i][1],
                        warehouse_locations[j][0], warehouse_locations[j][1]
                    )
                    min_separation_km = min(min_separation_km, dist)

            if min_separation_km < 200:  # Too close
                issues.append(FeasibilityIssue(
                    gate="Geographic Distribution",
                    severity="warning",
                    issue=f"Warehouses too clustered (min separation: {min_separation_km:.0f} km)",
                    physics="Overlapping coverage zones → inefficient",
                    recommendation="Spread warehouses >500 km apart for better coverage",
                    key_numbers={
                        'min_separation_km': min_separation_km,
                        'recommended_min_km': 500
                    }
                ))

        # Determine overall status
        has_showstopper = any(issue.severity == 'showstopper' for issue in issues)
        has_critical = any(issue.severity == 'critical' for issue in issues)

        if has_showstopper:
            status = FeasibilityStatus.NOT_FEASIBLE
        elif has_critical:
            status = FeasibilityStatus.MARGINAL
        else:
            status = FeasibilityStatus.FEASIBLE

        # Generate recommendations
        if status == FeasibilityStatus.NOT_FEASIBLE:
            recommendations.append(f"❌ KILL PROJECT or ADD {estimate_additional_warehouses_needed(coverage_gap)} warehouses")
            recommendations.append(f"Current {len(warehouse_locations)} warehouse(s) → {coverage_fraction*100:.1f}% coverage")
            recommendations.append(f"Need ≥{len(warehouse_locations) + estimate_additional_warehouses_needed(coverage_gap)} warehouses for {target_coverage*100:.0f}% coverage")
        elif status == FeasibilityStatus.MARGINAL:
            recommendations.append(f"⚠ MARGINAL: Add 1 more warehouse for safety margin")
        else:
            recommendations.append(f"✓ FEASIBLE: {coverage_fraction*100:.1f}% coverage with {len(warehouse_locations)} warehouses")

        return GeospatialFeasibilityResult(
            status=status,
            coverage_achieved=coverage_fraction,
            coverage_required=target_coverage,
            issues=issues,
            recommendations=recommendations,
            warehouse_count=len(warehouse_locations),
            total_population_millions=total_pop,
            covered_population_millions=covered_pop
        )


def check_24h_coverage(
    warehouse_locations: List[Tuple[float, float]],
    target_coverage: float = 0.95,
    country: str = 'india'
) -> bool:
    """
    Quick check: Can warehouses deliver to target coverage in 24 hours?

    Convenience function that returns simple True/False

    Args:
        warehouse_locations: List of (lat, lon)
        target_coverage: Required coverage fraction (0-1)
        country: Country code

    Returns:
        True if feasible, False otherwise

    Examples:
        >>> # Single warehouse
        >>> check_24h_coverage([(19.0760, 72.8777)])
        False

        >>> # Three warehouses
        >>> check_24h_coverage([
        ...     (19.0760, 72.8777),
        ...     (28.7041, 77.1025),
        ...     (12.9716, 77.5946)
        ... ])
        True
    """
    result = GeospatialFeasibilityChecker.check_24h_coverage(
        warehouse_locations,
        target_coverage,
        country=country
    )
    return result.status == FeasibilityStatus.FEASIBLE


def check_delivery_feasibility(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    max_time_hours: float,
    road_type: RoadType = RoadType.STATE_HIGHWAY
) -> Tuple[bool, str]:
    """
    Check if delivery from origin to destination is feasible within time limit

    Args:
        origin: (lat, lon) of warehouse
        destination: (lat, lon) of delivery location
        max_time_hours: Maximum delivery time
        road_type: Road network type

    Returns:
        (is_feasible, explanation)

    Examples:
        >>> # Mumbai to Delhi in 24h?
        >>> feasible, msg = check_delivery_feasibility(
        ...     (19.0760, 72.8777),
        ...     (28.7041, 77.1025),
        ...     24
        ... )
        >>> print(feasible, msg)
        True "Delivery time: 25.0 hours (within 24.0h limit)"
    """
    result = calculate_travel_time(origin, destination, road_type=road_type)

    if result.travel_time_hours <= max_time_hours:
        return True, f"Delivery time: {result.travel_time_hours:.1f}h (within {max_time_hours:.1f}h limit)"
    else:
        return False, f"Delivery time: {result.travel_time_hours:.1f}h (exceeds {max_time_hours:.1f}h limit by {result.travel_time_hours - max_time_hours:.1f}h)"


def estimate_additional_warehouses_needed(coverage_gap: float) -> int:
    """
    Estimate number of additional warehouses needed to close coverage gap

    Simplified heuristic: Each warehouse adds ~30-35% coverage

    Args:
        coverage_gap: Coverage gap (0-1)

    Returns:
        Estimated number of additional warehouses

    Examples:
        >>> estimate_additional_warehouses_needed(0.30)  # 30% gap
        1
        >>> estimate_additional_warehouses_needed(0.60)  # 60% gap
        2
    """
    if coverage_gap <= 0:
        return 0
    elif coverage_gap < 0.35:
        return 1
    elif coverage_gap < 0.65:
        return 2
    else:
        return 3
