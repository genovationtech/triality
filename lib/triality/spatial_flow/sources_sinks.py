"""Source and sink definitions for spatial flow problems"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class Source:
    """Flow source (origin point)

    Args:
        position: (x, y) or (x, y, z) coordinates
        weight: Flow magnitude (positive)
        label: Optional identifier
    """
    position: Tuple[float, ...]
    weight: float = 1.0
    label: Optional[str] = None

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError(f"Source weight must be positive, got {self.weight}")
        if len(self.position) not in [2, 3]:
            raise ValueError(f"Source position must be 2D or 3D, got {len(self.position)}D")


@dataclass
class Sink:
    """Flow sink (destination point)

    Args:
        position: (x, y) or (x, y, z) coordinates
        weight: Flow magnitude (positive, represents demand)
        label: Optional identifier
    """
    position: Tuple[float, ...]
    weight: float = 1.0
    label: Optional[str] = None

    def __post_init__(self):
        if self.weight <= 0:
            raise ValueError(f"Sink weight must be positive, got {self.weight}")
        if len(self.position) not in [2, 3]:
            raise ValueError(f"Sink position must be 2D or 3D, got {len(self.position)}D")


def validate_flow_balance(sources: List[Source], sinks: List[Sink],
                         tolerance: float = 1e-6) -> bool:
    """Check if total source flow equals total sink demand

    Args:
        sources: List of flow sources
        sinks: List of flow sinks
        tolerance: Acceptable imbalance (relative)

    Returns:
        True if balanced, raises ValueError if not
    """
    total_source = sum(s.weight for s in sources)
    total_sink = sum(s.weight for s in sinks)

    if abs(total_source - total_sink) > tolerance * max(total_source, total_sink):
        raise ValueError(
            f"Flow imbalance detected:\n"
            f"  Total source flow: {total_source:.6f}\n"
            f"  Total sink demand: {total_sink:.6f}\n"
            f"  Imbalance: {abs(total_source - total_sink):.6e}\n"
            f"  Suggestion: Adjust weights to balance flow"
        )

    return True


def check_minimum_distance(sources: List[Source], sinks: List[Sink],
                          min_distance: float) -> bool:
    """Verify sources and sinks are not too close together

    Args:
        sources: List of flow sources
        sinks: List of flow sinks
        min_distance: Minimum allowed distance

    Returns:
        True if all distances are sufficient, raises ValueError if not
    """
    # Check source-sink distances
    for src in sources:
        for snk in sinks:
            dist = np.linalg.norm(np.array(src.position) - np.array(snk.position))
            if dist < min_distance:
                raise ValueError(
                    f"Source and sink too close:\n"
                    f"  Source: {src.label or src.position} at {src.position}\n"
                    f"  Sink: {snk.label or snk.position} at {snk.position}\n"
                    f"  Distance: {dist:.6f} < minimum {min_distance:.6f}\n"
                    f"  Suggestion: Move points apart or reduce resolution"
                )

    # Check source-source distances
    for i, src1 in enumerate(sources):
        for src2 in sources[i+1:]:
            dist = np.linalg.norm(np.array(src1.position) - np.array(src2.position))
            if dist < min_distance:
                raise ValueError(
                    f"Sources too close:\n"
                    f"  Source 1: {src1.label or src1.position} at {src1.position}\n"
                    f"  Source 2: {src2.label or src2.position} at {src2.position}\n"
                    f"  Distance: {dist:.6f} < minimum {min_distance:.6f}"
                )

    # Check sink-sink distances
    for i, snk1 in enumerate(sinks):
        for snk2 in sinks[i+1:]:
            dist = np.linalg.norm(np.array(snk1.position) - np.array(snk2.position))
            if dist < min_distance:
                raise ValueError(
                    f"Sinks too close:\n"
                    f"  Sink 1: {snk1.label or snk1.position} at {snk1.position}\n"
                    f"  Sink 2: {snk2.label or snk2.position} at {snk2.position}\n"
                    f"  Distance: {dist:.6f} < minimum {min_distance:.6f}"
                )

    return True
