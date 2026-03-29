"""Path and network extraction from potential fields"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class Path:
    """Extracted path from potential field

    Args:
        points: List of (x, y) points along path
        cost: Total path cost
        source: Source label or position
        sink: Sink label or position
    """
    points: List[Tuple[float, float]]
    cost: float
    source: str
    sink: str

    def length(self) -> float:
        """Compute total Euclidean path length"""
        total = 0.0
        for i in range(len(self.points) - 1):
            p1 = np.array(self.points[i])
            p2 = np.array(self.points[i + 1])
            total += np.linalg.norm(p2 - p1)
        return total

    def __repr__(self):
        return f"Path({self.source} → {self.sink}, {len(self.points)} points, cost={self.cost:.3f})"


@dataclass
class Network:
    """Extracted network of paths

    Args:
        paths: List of individual paths
        total_cost: Total network cost
    """
    paths: List[Path]
    total_cost: float

    def __repr__(self):
        return f"Network({len(self.paths)} paths, total_cost={self.total_cost:.3f})"


class GradientTracer:
    """Trace paths through potential field using gradient descent

    The potential field solver produces a scalar field where:
    - Low values indicate desirable routing locations
    - Gradient points in direction of steepest cost increase
    - Negative gradient points toward optimal path

    This class traces paths by following the negative gradient from sinks back to sources.
    """

    def __init__(self, potential_field: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray):
        """Initialize gradient tracer

        Args:
            potential_field: 2D array of potential values (from PDE solver)
            grid_x: 1D array of x-coordinates
            grid_y: 1D array of y-coordinates
        """
        self.potential = potential_field
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.dx = grid_x[1] - grid_x[0]
        self.dy = grid_y[1] - grid_y[0]

    def interpolate(self, x: float, y: float) -> float:
        """Bilinear interpolation of potential at arbitrary point

        Args:
            x, y: Query point

        Returns:
            Interpolated potential value
        """
        # Find grid cell
        i = np.searchsorted(self.grid_x, x) - 1
        j = np.searchsorted(self.grid_y, y) - 1

        # Clamp to valid range
        i = np.clip(i, 0, len(self.grid_x) - 2)
        j = np.clip(j, 0, len(self.grid_y) - 2)

        # Bilinear interpolation weights
        x0, x1 = self.grid_x[i], self.grid_x[i + 1]
        y0, y1 = self.grid_y[j], self.grid_y[j + 1]

        wx = (x - x0) / (x1 - x0)
        wy = (y - y0) / (y1 - y0)

        # Interpolate
        v00 = self.potential[i, j]
        v10 = self.potential[i + 1, j]
        v01 = self.potential[i, j + 1]
        v11 = self.potential[i + 1, j + 1]

        v0 = v00 * (1 - wx) + v10 * wx
        v1 = v01 * (1 - wx) + v11 * wx

        return v0 * (1 - wy) + v1 * wy

    def gradient(self, x: float, y: float) -> Tuple[float, float]:
        """Compute gradient at point using finite differences

        Args:
            x, y: Query point

        Returns:
            (grad_x, grad_y) gradient vector
        """
        # Sample points for finite difference
        h = min(self.dx, self.dy) * 0.5

        v_xp = self.interpolate(x + h, y)
        v_xm = self.interpolate(x - h, y)
        v_yp = self.interpolate(x, y + h)
        v_ym = self.interpolate(x, y - h)

        grad_x = (v_xp - v_xm) / (2 * h)
        grad_y = (v_yp - v_ym) / (2 * h)

        return grad_x, grad_y

    def trace_path(self, start: Tuple[float, float], end: Tuple[float, float],
                   max_steps: int = 50000, step_size: float = None,
                   tolerance: float = None, debug: bool = False) -> List[Tuple[float, float]]:
        """Trace path from start to end following negative gradient

        Args:
            start: Starting point (typically a sink)
            end: Target point (typically a source)
            max_steps: Maximum iterations
            step_size: Integration step (default: 0.5 * min(dx, dy))
            tolerance: Convergence tolerance (default: min(dx, dy))
            debug: Print debugging information

        Returns:
            List of (x, y) points along path
        """
        if step_size is None:
            step_size = 0.5 * min(self.dx, self.dy)

        if tolerance is None:
            tolerance = min(self.dx, self.dy)

        # Domain bounds for validation
        x_min, x_max = self.grid_x[0], self.grid_x[-1]
        y_min, y_max = self.grid_y[0], self.grid_y[-1]

        path = [start]
        x, y = start

        for step in range(max_steps):
            # Check if close enough to target
            dist_to_end = np.sqrt((x - end[0])**2 + (y - end[1])**2)
            if dist_to_end < tolerance:
                if path[-1] != end:
                    path.append(end)
                break

            # Compute gradient
            grad_x, grad_y = self.gradient(x, y)
            grad_norm = np.sqrt(grad_x**2 + grad_y**2)

            if debug and step < 50:
                print(f"  Step {step}: pos=({x:.3f},{y:.3f}), grad=({grad_x:.3e},{grad_y:.3e}), |grad|={grad_norm:.3e}, dist_to_end={dist_to_end:.3f}")

            if grad_norm < 1e-10:
                if debug:
                    print(f"  Stopped at step {step}: gradient too small")
                if path[-1] != end:
                    path.append(end)
                break

            # Adaptive step size based on gradient strength
            if grad_norm > 0.1:
                adaptive_step = step_size * 2.0
            elif grad_norm < 0.01:
                adaptive_step = step_size * 0.5
            else:
                adaptive_step = step_size

            # Move in negative gradient direction
            dx = -grad_x / grad_norm * adaptive_step
            dy = -grad_y / grad_norm * adaptive_step

            x_new = x + dx
            y_new = y + dy

            # CRITICAL: Bounds checking - prevent path from leaving domain
            margin = 0.1 * min(self.dx, self.dy)
            x_new = np.clip(x_new, x_min + margin, x_max - margin)
            y_new = np.clip(y_new, y_min + margin, y_max - margin)

            # Numerical validation
            if not (np.isfinite(x_new) and np.isfinite(y_new)):
                if debug:
                    print(f"  ERROR: Non-finite coordinates at step {step}")
                path.append(end)
                break

            path.append((x_new, y_new))
            x, y = x_new, y_new

            # Check if stuck
            if len(path) > 20:
                recent = np.array(path[-10:])
                displacement = np.linalg.norm(recent[-1] - recent[0])
                if displacement < adaptive_step * 2:
                    if debug:
                        print(f"  Stopped: stuck at step {step}")
                    path.append(end)
                    break

        # Ensure target is added
        if len(path) < 2 or np.linalg.norm(np.array(path[-1]) - np.array(end)) > tolerance:
            path.append(end)

        return path

    def extract_paths(self, sources, sinks, debug=False) -> List[Path]:
        """Extract paths from sources to sinks

        Args:
            sources: List of Source objects
            sinks: List of Sink objects
            debug: Enable debug output

        Returns:
            List of Path objects
        """
        paths = []

        # Create paths from each source to each sink (all-to-all routing)
        # This is standard for physics-based flow distribution
        for src in sources:
            for snk in sinks:
                # Get original positions
                src_pos_orig = src.position[:2]
                snk_pos_orig = snk.position[:2]

                # Snap to grid to avoid oscillation artifacts
                # (BCs were applied at grid points, so trace should start there)
                src_i = np.argmin(np.abs(self.grid_x - src_pos_orig[0]))
                src_j = np.argmin(np.abs(self.grid_y - src_pos_orig[1]))
                src_pos = (self.grid_x[src_i], self.grid_y[src_j])

                snk_i = np.argmin(np.abs(self.grid_x - snk_pos_orig[0]))
                snk_j = np.argmin(np.abs(self.grid_y - snk_pos_orig[1]))
                snk_pos = (self.grid_x[snk_i], self.grid_y[snk_j])

                if debug:
                    print(f"\nTracing path: {snk.label} → {src.label}")
                    print(f"  From {snk_pos_orig} (grid-snapped to {snk_pos})")
                    print(f"  To   {src_pos_orig} (grid-snapped to {src_pos})")

                points = self.trace_path(snk_pos, src_pos, debug=debug)

                # Compute path cost
                cost = 0.0
                for i in range(len(points) - 1):
                    p1 = np.array(points[i])
                    p2 = np.array(points[i + 1])
                    cost += np.linalg.norm(p2 - p1)

                path = Path(
                    points=points,
                    cost=cost,
                    source=src.label or str(src.position),
                    sink=snk.label or str(snk.position)
                )
                paths.append(path)

        return paths


def extract_network(potential_field: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray,
                   sources, sinks, debug=False) -> Network:
    """Extract complete network from potential field

    Args:
        potential_field: 2D array of potential values
        grid_x, grid_y: Grid coordinates
        sources: List of Source objects
        sinks: List of Sink objects
        debug: Enable debug output

    Returns:
        Network object with extracted paths
    """
    tracer = GradientTracer(potential_field, grid_x, grid_y)
    paths = tracer.extract_paths(sources, sinks, debug=debug)

    total_cost = sum(path.cost for path in paths)

    return Network(paths=paths, total_cost=total_cost)


def simplify_path(path: Path, tolerance: float = 0.01) -> Path:
    """Simplify path using Douglas-Peucker algorithm

    Args:
        path: Path to simplify
        tolerance: Maximum deviation tolerance

    Returns:
        Simplified path
    """
    if len(path.points) <= 2:
        return path

    def perpendicular_distance(point, line_start, line_end):
        """Distance from point to line segment"""
        p = np.array(point)
        start = np.array(line_start)
        end = np.array(line_end)

        if np.allclose(start, end):
            return np.linalg.norm(p - start)

        line_vec = end - start
        point_vec = p - start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len

        proj_length = np.dot(point_vec, line_unitvec)
        proj_length = np.clip(proj_length, 0, line_len)

        closest = start + proj_length * line_unitvec
        return np.linalg.norm(p - closest)

    def douglas_peucker(points, tolerance, depth=0, max_depth=100):
        """Recursive Douglas-Peucker simplification with depth limiting"""
        if len(points) <= 2 or depth >= max_depth:
            return points

        # Find point with maximum distance
        dmax = 0.0
        index = 0
        for i in range(1, len(points) - 1):
            d = perpendicular_distance(points[i], points[0], points[-1])
            if d > dmax:
                index = i
                dmax = d

        # If max distance is greater than tolerance, recursively simplify
        if dmax > tolerance:
            left = douglas_peucker(points[:index + 1], tolerance, depth + 1, max_depth)
            right = douglas_peucker(points[index:], tolerance, depth + 1, max_depth)
            return left[:-1] + right
        else:
            # For nearly straight segments, keep a few intermediate points
            # to maintain path detail and ensure minimum point count
            if len(points) > 50:
                # Keep points at 1/4, 1/2, 3/4 for long straight segments
                quarter = len(points) // 4
                return [points[0], points[quarter], points[len(points)//2],
                       points[3*quarter], points[-1]]
            elif len(points) > 10:
                # For medium segments, keep start, middle, end
                return [points[0], points[len(points)//2], points[-1]]
            elif len(points) > 2:
                # For short segments, keep at least 3 points
                return [points[0], points[len(points)//2], points[-1]]
            else:
                # Already minimal
                return points

    simplified_points = douglas_peucker(path.points, tolerance)

    return Path(
        points=simplified_points,
        cost=path.cost,
        source=path.source,
        sink=path.sink
    )
