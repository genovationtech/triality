"""Constraint definitions for spatial flow problems"""

import numpy as np
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ObstacleType(Enum):
    """Types of obstacles"""
    HARD = "hard"      # Completely forbidden
    SOFT = "soft"      # High cost but traversable
    CLEARANCE = "clearance"  # Requires minimum distance


@dataclass
class Obstacle:
    """Spatial obstacle for routing constraints

    Args:
        geometry: Callable (x, y) -> bool (True if inside obstacle)
        obstacle_type: Hard (forbidden) or soft (high cost)
        cost_multiplier: Cost penalty for soft obstacles
        clearance: Required clearance distance (for CLEARANCE type)
        label: Optional identifier
    """
    geometry: Callable[[float, float], bool]
    obstacle_type: ObstacleType = ObstacleType.HARD
    cost_multiplier: float = 1000.0
    clearance: float = 0.0
    label: Optional[str] = None

    def __post_init__(self):
        if self.obstacle_type == ObstacleType.SOFT and self.cost_multiplier <= 1.0:
            raise ValueError(
                f"Soft obstacle cost_multiplier must be > 1.0, got {self.cost_multiplier}"
            )
        if self.obstacle_type == ObstacleType.CLEARANCE and self.clearance <= 0:
            raise ValueError(
                f"Clearance obstacle must have clearance > 0, got {self.clearance}"
            )

    def is_inside(self, x: float, y: float) -> bool:
        """Check if point is inside obstacle"""
        return self.geometry(x, y)

    def get_cost(self, x: float, y: float, base_cost: float = 1.0) -> float:
        """Get cost at point considering obstacle"""
        if not self.is_inside(x, y):
            return base_cost

        if self.obstacle_type == ObstacleType.HARD:
            return np.inf
        elif self.obstacle_type == ObstacleType.SOFT:
            return base_cost * self.cost_multiplier
        else:
            return base_cost


class ObstacleBuilder:
    """Builder for common obstacle geometries"""

    @staticmethod
    def rectangle(xmin: float, xmax: float, ymin: float, ymax: float,
                 obstacle_type: ObstacleType = ObstacleType.HARD,
                 label: Optional[str] = None) -> Obstacle:
        """Rectangular obstacle

        Args:
            xmin, xmax, ymin, ymax: Rectangle bounds
            obstacle_type: Hard or soft obstacle
            label: Optional identifier

        Returns:
            Obstacle with rectangular geometry
        """
        if xmin >= xmax or ymin >= ymax:
            raise ValueError(
                f"Invalid rectangle bounds: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]"
            )

        def inside(x, y):
            return (xmin <= x <= xmax) and (ymin <= y <= ymax)

        return Obstacle(
            geometry=inside,
            obstacle_type=obstacle_type,
            label=label or f"rect_{xmin}_{xmax}_{ymin}_{ymax}"
        )

    @staticmethod
    def circle(center: Tuple[float, float], radius: float,
              obstacle_type: ObstacleType = ObstacleType.HARD,
              label: Optional[str] = None) -> Obstacle:
        """Circular obstacle

        Args:
            center: (x, y) center point
            radius: Circle radius
            obstacle_type: Hard or soft obstacle
            label: Optional identifier

        Returns:
            Obstacle with circular geometry
        """
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")

        cx, cy = center

        def inside(x, y):
            return (x - cx)**2 + (y - cy)**2 <= radius**2

        return Obstacle(
            geometry=inside,
            obstacle_type=obstacle_type,
            label=label or f"circle_{center}_r{radius}"
        )

    @staticmethod
    def polygon(vertices: List[Tuple[float, float]],
               obstacle_type: ObstacleType = ObstacleType.HARD,
               label: Optional[str] = None) -> Obstacle:
        """Polygonal obstacle (ray casting algorithm)

        Args:
            vertices: List of (x, y) vertices in order
            obstacle_type: Hard or soft obstacle
            label: Optional identifier

        Returns:
            Obstacle with polygonal geometry
        """
        if len(vertices) < 3:
            raise ValueError(f"Polygon must have at least 3 vertices, got {len(vertices)}")

        vertices = np.array(vertices)

        def inside(x, y):
            """Ray casting algorithm for point-in-polygon test"""
            n = len(vertices)
            inside = False
            p1x, p1y = vertices[0]

            for i in range(1, n + 1):
                p2x, p2y = vertices[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y

            return inside

        return Obstacle(
            geometry=inside,
            obstacle_type=obstacle_type,
            label=label or f"polygon_{len(vertices)}vertices"
        )

    @staticmethod
    def union(obstacles: List[Obstacle],
             label: Optional[str] = None) -> Obstacle:
        """Union of multiple obstacles

        Args:
            obstacles: List of obstacles to combine
            label: Optional identifier

        Returns:
            Obstacle representing union
        """
        if not obstacles:
            raise ValueError("Must provide at least one obstacle")

        # Union is hard if any component is hard
        is_hard = any(obs.obstacle_type == ObstacleType.HARD for obs in obstacles)
        obstacle_type = ObstacleType.HARD if is_hard else ObstacleType.SOFT

        def inside(x, y):
            return any(obs.is_inside(x, y) for obs in obstacles)

        return Obstacle(
            geometry=inside,
            obstacle_type=obstacle_type,
            label=label or f"union_of_{len(obstacles)}"
        )


def validate_obstacles_in_domain(obstacles: List[Obstacle],
                                domain_bounds: Tuple[Tuple[float, float], ...],
                                num_samples: int = 100) -> bool:
    """Validate obstacles don't completely block the domain

    Args:
        obstacles: List of obstacles
        domain_bounds: ((xmin, xmax), (ymin, ymax))
        num_samples: Number of random points to test

    Returns:
        True if valid, raises ValueError if domain is completely blocked
    """
    (xmin, xmax), (ymin, ymax) = domain_bounds

    # Sample random points
    x_samples = np.random.uniform(xmin, xmax, num_samples)
    y_samples = np.random.uniform(ymin, ymax, num_samples)

    blocked_count = 0
    for x, y in zip(x_samples, y_samples):
        if any(obs.is_inside(x, y) and obs.obstacle_type == ObstacleType.HARD
               for obs in obstacles):
            blocked_count += 1

    blocked_fraction = blocked_count / num_samples

    if blocked_fraction > 0.95:
        raise ValueError(
            f"Domain is {blocked_fraction*100:.1f}% blocked by hard obstacles\n"
            f"  Suggestion: Reduce obstacle sizes or use soft obstacles"
        )

    return True


def check_sources_sinks_not_blocked(sources, sinks, obstacles: List[Obstacle]) -> bool:
    """Verify sources and sinks are not inside hard obstacles

    Args:
        sources: List of Source objects
        sinks: List of Sink objects
        obstacles: List of obstacles

    Returns:
        True if valid, raises ValueError if any point is blocked
    """
    # Check sources
    for src in sources:
        x, y = src.position[:2]  # Take first 2D coordinates
        for obs in obstacles:
            if obs.obstacle_type == ObstacleType.HARD and obs.is_inside(x, y):
                raise ValueError(
                    f"Source at {src.position} is inside hard obstacle '{obs.label}'\n"
                    f"  Suggestion: Move source outside obstacle or change obstacle to soft"
                )

    # Check sinks
    for snk in sinks:
        x, y = snk.position[:2]  # Take first 2D coordinates
        for obs in obstacles:
            if obs.obstacle_type == ObstacleType.HARD and obs.is_inside(x, y):
                raise ValueError(
                    f"Sink at {snk.position} is inside hard obstacle '{obs.label}'\n"
                    f"  Suggestion: Move sink outside obstacle or change obstacle to soft"
                )

    return True
