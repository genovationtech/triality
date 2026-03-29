"""
Cost-field-aware PDE solver for spatial flow routing.

Solves the variable-coefficient Poisson equation:
    ∇·(1/c(x,y) ∇φ) = f

Where:
- c(x,y) is the cost field (higher cost → less preferable routing)
- φ is the potential field
- Paths follow -∇φ (negative gradient)
- High-cost regions create natural avoidance
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix


def build_cost_aware_system(domain, resolution, cost_field, obstacles, sources=None, sinks=None):
    """
    Build finite difference system for cost-aware routing.

    Discretizes: ∇·(1/c ∇φ) = 0

    Using finite volumes:
    For each interior cell (i,j), the flux balance is:
        (φ[i+1,j] - φ[i,j])/c[i+1/2,j] - (φ[i,j] - φ[i-1,j])/c[i-1/2,j]
      + (φ[i,j+1] - φ[i,j])/c[i,j+1/2] - (φ[i,j] - φ[i,j-1])/c[i,j-1/2] = 0

    Args:
        domain: ((xmin, xmax), (ymin, ymax))
        resolution: Grid resolution
        cost_field: Callable (x, y) -> cost
        obstacles: List of obstacles
        sources: List of Source objects (high potential)
        sinks: List of Sink objects (low potential)

    Returns:
        A: Coefficient matrix (sparse)
        b: RHS vector
        grid_x, grid_y: Grid coordinates
        cost_grid: 2D array of cost values
    """

    if sources is None:
        sources = []
    if sinks is None:
        sinks = []

    (xmin, xmax), (ymin, ymax) = domain

    # Create grid
    x = np.linspace(xmin, xmax, resolution)
    y = np.linspace(ymin, ymax, resolution)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    n = resolution
    N = n * n

    # Build cost field on grid
    X, Y = np.meshgrid(x, y, indexing='ij')
    cost_grid = np.ones_like(X)

    # Evaluate cost field
    for i in range(n):
        for j in range(n):
            base_cost = 1.0

            if cost_field is not None:
                # Cost field provides the cost value
                field_cost = cost_field(X[i, j], Y[i, j])
                # For soft obstacles (cost fields), amplify the effect exponentially
                # This makes high-cost regions much more expensive to traverse
                if field_cost > 2.0:
                    # Very strong amplification for significant hotspots: cost⁴
                    cost_grid[i, j] = field_cost ** 4.0
                elif field_cost > 1.5:
                    # Strong amplification: cost³
                    cost_grid[i, j] = field_cost ** 3.0
                elif field_cost > 1.0:
                    # Moderate amplification: cost²
                    cost_grid[i, j] = field_cost ** 2.0
                else:
                    cost_grid[i, j] = field_cost

            # Apply obstacle costs
            for obs in obstacles:
                if obs.is_inside(X[i, j], Y[i, j]):
                    if obs.obstacle_type.value == 'hard':
                        cost_grid[i, j] = 1e10  # Effectively infinite
                    else:  # soft
                        cost_grid[i, j] *= obs.cost_multiplier

    # Clamp costs to reasonable range
    cost_grid = np.clip(cost_grid, 0.1, 1e10)

    # Build system matrix
    A = lil_matrix((N, N))
    b = np.zeros(N)

    def idx(i, j):
        """Convert 2D indices to 1D index"""
        return i * n + j

    # Find grid points closest to sources and sinks
    source_points = set()
    sink_points = set()
    source_positions = []
    sink_positions = []

    # Map sources to grid points (low potential - attractors)
    for src in sources:
        src_x, src_y = src.position[:2]
        i_src = np.argmin(np.abs(x - src_x))
        j_src = np.argmin(np.abs(y - src_y))
        source_points.add((i_src, j_src))
        source_positions.append((src_x, src_y))

    # Map sinks to grid points (high potential - repellers)
    for snk in sinks:
        snk_x, snk_y = snk.position[:2]
        i_snk = np.argmin(np.abs(x - snk_x))
        j_snk = np.argmin(np.abs(y - snk_y))
        sink_points.add((i_snk, j_snk))
        sink_positions.append((snk_x, snk_y))

    # Identify hard obstacle points (these will be excluded from solution)
    hard_obstacle_points = set()
    for i in range(n):
        for j in range(n):
            pt_x, pt_y = x[i], y[j]
            for obs in obstacles:
                if obs.obstacle_type.value == 'hard' and obs.is_inside(pt_x, pt_y):
                    hard_obstacle_points.add((i, j))
                    break

    # Build finite volume discretization
    for i in range(n):
        for j in range(n):
            ij = idx(i, j)

            # Check if this point is a boundary point
            is_boundary = (i == 0 or i == n-1 or j == 0 or j == n-1)

            # Apply Dirichlet BC at sources and sinks (highest priority)
            if (i, j) in source_points:
                # Source: LOW potential (attractor, φ = 0.0)
                # Paths flow toward sources following -∇φ
                A[ij, ij] = 1.0
                b[ij] = 0.0
            elif (i, j) in sink_points:
                # Sink: HIGH potential (repeller, φ = 1.0)
                # Paths flow away from sinks following -∇φ
                A[ij, ij] = 1.0
                b[ij] = 1.0
            elif (i, j) in hard_obstacle_points:
                # Hard obstacle: Set high potential to create impenetrable barrier
                # High potential creates a "hill" that forces paths to detour around
                # Use φ = 2.0 (higher than max sink potential of 1.0)
                A[ij, ij] = 1.0
                b[ij] = 2.0
            elif is_boundary:
                # Domain boundaries: set potential based on proximity to sources/sinks
                # This avoids creating competing gradients near sources/sinks

                # Find distance to nearest source and sink
                pt_x, pt_y = x[i], y[j]
                min_dist_to_source = float('inf')
                min_dist_to_sink = float('inf')

                for src_pos in source_positions:
                    dist = np.sqrt((pt_x - src_pos[0])**2 + (pt_y - src_pos[1])**2)
                    min_dist_to_source = min(min_dist_to_source, dist)

                for snk_pos in sink_positions:
                    dist = np.sqrt((pt_x - snk_pos[0])**2 + (pt_y - snk_pos[1])**2)
                    min_dist_to_sink = min(min_dist_to_sink, dist)

                # Set boundary potential based on which is closer
                if min_dist_to_source < min_dist_to_sink:
                    # Closer to source: low potential
                    boundary_potential = 0.0
                elif min_dist_to_sink < min_dist_to_source:
                    # Closer to sink: high potential
                    boundary_potential = 1.0
                else:
                    # Equidistant: neutral
                    boundary_potential = 0.5

                A[ij, ij] = 1.0
                b[ij] = boundary_potential
            else:
                # Interior point: ∇·(1/c ∇φ) = 0

                # Harmonic average of conductivity (1/cost) at cell faces
                # For conductivity κ = 1/c, harmonic avg is: κ_avg = 2*κ1*κ2/(κ1+κ2)
                # Which simplifies to: κ_avg = 2 / (c1 + c2)

                # Right face (i+1/2, j)
                conductivity_right = 2.0 / (cost_grid[i, j] + cost_grid[i+1, j])

                # Left face (i-1/2, j)
                conductivity_left = 2.0 / (cost_grid[i, j] + cost_grid[i-1, j])

                # Top face (i, j+1/2)
                conductivity_top = 2.0 / (cost_grid[i, j] + cost_grid[i, j+1])

                # Bottom face (i, j-1/2)
                conductivity_bottom = 2.0 / (cost_grid[i, j] + cost_grid[i, j-1])

                # Finite volume flux balance
                # (φ[i+1,j] - φ[i,j])*κ_right/dx² - (φ[i,j] - φ[i-1,j])*κ_left/dx²
                # + (φ[i,j+1] - φ[i,j])*κ_top/dy² - (φ[i,j] - φ[i,j-1])*κ_bottom/dy² = 0

                coeff_center = -(conductivity_right + conductivity_left) / dx**2 - (conductivity_top + conductivity_bottom) / dy**2
                coeff_right = conductivity_right / dx**2
                coeff_left = conductivity_left / dx**2
                coeff_top = conductivity_top / dy**2
                coeff_bottom = conductivity_bottom / dy**2

                A[ij, ij] = coeff_center
                A[ij, idx(i+1, j)] = coeff_right
                A[ij, idx(i-1, j)] = coeff_left
                A[ij, idx(i, j+1)] = coeff_top
                A[ij, idx(i, j-1)] = coeff_bottom

                b[ij] = 0.0

    return csr_matrix(A), b, x, y, cost_grid
