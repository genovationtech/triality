"""Spatial Flow Engine - Main API for continuous routing problems"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from triality.solvers.linear import solve_linear

from .sources_sinks import Source, Sink, validate_flow_balance, check_minimum_distance
from .cost_fields import CostField, CostFieldBuilder, validate_cost_field
from .constraints import Obstacle, validate_obstacles_in_domain, check_sources_sinks_not_blocked
from .extraction import extract_network, Network, simplify_path
from .cost_aware_solver import build_cost_aware_system


@dataclass
class FlowProblem:
    """Complete spatial flow problem specification

    Args:
        sources: Flow origin points
        sinks: Flow destination points
        domain_bounds: ((xmin, xmax), (ymin, ymax))
        cost_field: Spatial cost function
        obstacles: List of obstacles (optional)
        resolution: Grid resolution for PDE solver
    """
    sources: List[Source]
    sinks: List[Sink]
    domain_bounds: Tuple[Tuple[float, float], Tuple[float, float]]
    cost_field: CostField
    obstacles: List[Obstacle]
    resolution: int = 100

    def validate(self):
        """Validate problem specification"""
        # Note: Flow balance not strictly required for routing problems
        # (unlike fluid flow where conservation is mandatory)

        # Check minimum distances
        (xmin, xmax), (ymin, ymax) = self.domain_bounds
        domain_size = min(xmax - xmin, ymax - ymin)
        min_dist = domain_size / (self.resolution * 2)
        check_minimum_distance(self.sources, self.sinks, min_dist)

        # Validate cost field
        validate_cost_field(self.cost_field, self.domain_bounds)

        # Validate obstacles
        if self.obstacles:
            validate_obstacles_in_domain(self.obstacles, self.domain_bounds)
            check_sources_sinks_not_blocked(self.sources, self.sinks, self.obstacles)


class SpatialFlowEngine:
    """
    Spatial Flow Engine for continuous routing and distribution problems.

    Uses physics-based field optimization to solve routing problems:
    - Computes potential field via Laplace/Poisson equation
    - Extracts optimal paths following gradient descent
    - Handles obstacles, cost fields, and multiple sources/sinks

    Example:
        >>> engine = SpatialFlowEngine()
        >>> engine.add_source((0.1, 0.5), weight=1.0, label="A")
        >>> engine.add_sink((0.9, 0.5), weight=1.0, label="B")
        >>> engine.set_domain((0, 1), (0, 1))
        >>> network = engine.solve()
    """

    def __init__(self):
        """Initialize empty spatial flow engine"""
        self.sources: List[Source] = []
        self.sinks: List[Sink] = []
        self.obstacles: List[Obstacle] = []
        self.cost_field: Optional[CostField] = None
        self.domain_bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self.resolution: int = 100

    def add_source(self, position: Tuple[float, float], weight: float = 1.0,
                   label: Optional[str] = None):
        """Add flow source

        Args:
            position: (x, y) coordinates
            weight: Flow magnitude
            label: Optional identifier
        """
        self.sources.append(Source(position, weight, label))

    def add_sink(self, position: Tuple[float, float], weight: float = 1.0,
                label: Optional[str] = None):
        """Add flow sink

        Args:
            position: (x, y) coordinates
            weight: Flow demand
            label: Optional identifier
        """
        self.sinks.append(Sink(position, weight, label))

    def add_obstacle(self, obstacle: Obstacle):
        """Add obstacle constraint

        Args:
            obstacle: Obstacle object
        """
        self.obstacles.append(obstacle)

    def set_cost_field(self, cost_field: CostField):
        """Set spatial cost field

        Args:
            cost_field: CostField object
        """
        self.cost_field = cost_field

    def set_domain(self, x_range: Tuple[float, float], y_range: Tuple[float, float]):
        """Set spatial domain

        Args:
            x_range: (xmin, xmax)
            y_range: (ymin, ymax)
        """
        self.domain_bounds = (x_range, y_range)

    def set_resolution(self, resolution: int):
        """Set grid resolution

        Args:
            resolution: Number of grid points per dimension
        """
        if resolution < 10:
            raise ValueError(f"Resolution must be >= 10, got {resolution}")
        self.resolution = resolution

    def solve(self, verbose: bool = True, simplify: bool = True) -> Network:
        """
        Solve the spatial flow problem using physics-based optimization.

        This is the main API - it:
        1. Validates the problem specification
        2. Constructs a Poisson equation with source/sink terms
        3. Applies cost field and obstacle constraints
        4. Solves the PDE using the core Triality solver
        5. Extracts optimal paths via gradient descent

        Args:
            verbose: Print progress
            simplify: Simplify paths using Douglas-Peucker

        Returns:
            Network object with extracted paths

        Raises:
            ValueError: If problem specification is invalid
        """
        # Validate problem
        if not self.sources:
            raise ValueError("Must specify at least one source")
        if not self.sinks:
            raise ValueError("Must specify at least one sink")
        if self.domain_bounds is None:
            raise ValueError("Must specify domain bounds")

        # Use uniform cost if none specified
        if self.cost_field is None:
            self.cost_field = CostFieldBuilder.uniform()

        # Create problem and validate
        problem = FlowProblem(
            sources=self.sources,
            sinks=self.sinks,
            domain_bounds=self.domain_bounds,
            cost_field=self.cost_field,
            obstacles=self.obstacles,
            resolution=self.resolution
        )
        problem.validate()

        if verbose:
            print("=" * 60)
            print("Triality Spatial Flow Engine")
            print("=" * 60)
            print(f"\nSources: {len(self.sources)}")
            print(f"Sinks: {len(self.sinks)}")
            print(f"Obstacles: {len(self.obstacles)}")
            print(f"Resolution: {self.resolution}×{self.resolution}")
            print(f"Cost field: {self.cost_field.name}")

        # Step 1: Build cost-aware PDE system
        # We solve: ∇·(1/c(x,y) ∇φ) = 0
        # Where c(x,y) is the cost field (high cost → avoid)
        # This produces a potential field φ where paths follow -∇φ

        if verbose:
            print("\nBuilding cost-aware potential field...")

        (xmin, xmax), (ymin, ymax) = self.domain_bounds

        # Build finite difference system with cost field
        A, b, grid_x, grid_y, cost_grid = build_cost_aware_system(
            domain=self.domain_bounds,
            resolution=self.resolution,
            cost_field=self.cost_field,
            obstacles=self.obstacles,
            sources=self.sources,
            sinks=self.sinks
        )

        if verbose:
            print(f"  System size: {A.shape[0]} DOFs")
            print(f"  Cost range: [{np.min(cost_grid):.2e}, {np.max(cost_grid):.2e}]")

        # Step 2: Solve linear system
        if verbose:
            print("\nSolving potential field...")

        # Use GMRES instead of CG - more robust for variable-coefficient systems
        # CG requires SPD which isn't guaranteed with cost fields and BCs
        result = solve_linear(A, b, method='gmres', precond='jacobi', verbose=verbose)

        if not result.converged:
            # Try BiCGSTAB as first fallback
            if verbose:
                print("  GMRES failed, trying BiCGSTAB...")
            result = solve_linear(A, b, method='bicgstab', precond='jacobi', verbose=verbose)

        if not result.converged:
            # Try direct solver as final fallback
            if verbose:
                print("  Iterative solvers failed, trying direct solver...")
            result = solve_linear(A, b, method='direct', verbose=verbose)

        if not result.converged:
            raise RuntimeError(
                f"PDE solver did not converge (residual={result.residual:.2e})\n"
                f"  Suggestion: Increase resolution or check problem specification"
            )

        if verbose:
            print(f"  ✓ Converged ({result.iterations} iterations, residual={result.residual:.2e})")

        # Step 3: Extract paths from potential field
        if verbose:
            print("\nExtracting paths from potential field...")

        # Reshape solution to 2D grid
        n = len(grid_x)
        potential = result.x.reshape(n, n)

        # Numerical sanity check on potential field
        if not np.all(np.isfinite(potential)):
            raise RuntimeError(
                "Potential field contains NaN or Inf values - numerical instability detected"
            )

        network = extract_network(potential, grid_x, grid_y, self.sources, self.sinks, debug=verbose)

        # Numerical sanity check on path lengths
        domain_diagonal = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
        for path in network.paths:
            if not np.all(np.isfinite([p[0] for p in path.points] + [p[1] for p in path.points])):
                raise RuntimeError(f"Path {path.source} → {path.sink} contains NaN or Inf coordinates")

            # Warn if path is suspiciously long (>10x domain diagonal)
            if path.length() > 10 * domain_diagonal:
                if verbose:
                    print(f"  ⚠️ WARNING: Path {path.source} → {path.sink} is unusually long ({path.length():.1f} vs {domain_diagonal:.1f} diagonal)")

        # Step 4: Simplify paths if requested
        if simplify:
            if verbose:
                print("Simplifying paths...")

            domain_size = min(xmax - xmin, ymax - ymin)
            tolerance = domain_size / 100  # 1% of domain size

            network.paths = [simplify_path(path, tolerance) for path in network.paths]

        if verbose:
            print(f"\n{'='*60}")
            print(f"✓ Extracted {len(network.paths)} paths")
            print(f"  Total network cost: {network.total_cost:.3f}")
            for path in network.paths:
                print(f"    {path.source} → {path.sink}: {path.cost:.3f} (length={path.length():.3f})")
            print(f"{'='*60}\n")

        return network

    def visualize(self, network: Network, show_potential: bool = True):
        """Visualize the routing solution

        Args:
            network: Network to visualize
            show_potential: Show underlying potential field
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed - cannot visualize")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot obstacles
        if self.obstacles:
            (xmin, xmax), (ymin, ymax) = self.domain_bounds
            x = np.linspace(xmin, xmax, 200)
            y = np.linspace(ymin, ymax, 200)
            X, Y = np.meshgrid(x, y)

            for obs in self.obstacles:
                Z = np.zeros_like(X)
                for i in range(len(x)):
                    for j in range(len(y)):
                        Z[j, i] = obs.is_inside(x[i], y[j])

                ax.contourf(X, Y, Z, levels=[0.5, 1.5], colors=['gray'], alpha=0.3)

        # Plot paths
        for path in network.paths:
            points = np.array(path.points)
            ax.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, alpha=0.7)

        # Plot sources and sinks
        for src in self.sources:
            ax.plot(src.position[0], src.position[1], 'go', markersize=12,
                   label='Source' if src == self.sources[0] else '')
        for snk in self.sinks:
            ax.plot(snk.position[0], snk.position[1], 'ro', markersize=12,
                   label='Sink' if snk == self.sinks[0] else '')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Spatial Flow Network')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()
