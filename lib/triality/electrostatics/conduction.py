"""Conductive Media Solver

Solves steady-state current flow in conductive media:
    ∇⋅(σ∇V) = 0

Where:
    σ = conductivity [S/m] (can be position-dependent)
    V = electric potential [V]
    J = -σ∇V = current density [A/m²]

Applications:
- Power electronics layout
- Busbar design
- Grounding systems
- PCB copper pour (DC/low-frequency)
- Hotspot detection
"""

import numpy as np
from scipy import sparse
from typing import Callable, Optional, Tuple, Dict, List
from dataclasses import dataclass

from triality.solvers.linear import solve_linear
from .field_solver import BoundaryCondition, BoundaryType


@dataclass
class Material:
    """Conductive material properties"""
    name: str
    conductivity: float  # σ [S/m]
    region: Optional[Callable[[float, float], bool]] = None  # (x, y) -> bool

    # Optional thermal properties for hotspot analysis
    thermal_conductivity: Optional[float] = None  # k [W/(m·K)]
    max_temperature: Optional[float] = None       # T_max [°C]

    def __repr__(self):
        return f"Material({self.name}, σ={self.conductivity:.2e} S/m)"


class ConductiveSolver:
    """
    Steady-state conduction solver for 2D domains.

    Solves: ∇⋅(σ∇V) = 0

    Where σ can vary spatially (different materials, temperature-dependent, etc.)

    Example:
        >>> solver = ConductiveSolver()
        >>> solver.set_domain((0, 0.1), (0, 0.05))  # 10cm × 5cm
        >>> solver.set_resolution(100)
        >>>
        >>> # Copper busbar
        >>> copper = Material('Copper', conductivity=5.96e7)
        >>> solver.add_material(copper, region=lambda x, y: 0.01 < x < 0.09)
        >>>
        >>> # Voltage source at left
        >>> solver.add_boundary(BoundaryCondition(BoundaryType.DIRICHLET, value=12.0,
        ...                     region=lambda x, y: x < 0.001))
        >>> # Ground at right
        >>> solver.add_boundary(BoundaryCondition(BoundaryType.GROUNDED,
        ...                     region=lambda x, y: x > 0.099))
        >>>
        >>> result = solver.solve()
        >>> J_x, J_y = result.current_density(0.05, 0.025)
        >>> hotspots = result.find_hotspots(max_power_density=1e8)
    """

    def __init__(self):
        self.domain_x = None
        self.domain_y = None
        self.resolution = 50

        self.materials: List[Material] = []
        self.boundaries: List[BoundaryCondition] = []
        self.default_conductivity = 1e-10  # Very low (insulator)

        self.potential = None
        self.grid_x = None
        self.grid_y = None
        self.conductivity_grid = None

    def set_domain(self, x_range: Tuple[float, float], y_range: Tuple[float, float]):
        """Set spatial domain [xmin, xmax] × [ymin, ymax]"""
        self.domain_x = x_range
        self.domain_y = y_range

    def set_resolution(self, n: int):
        """Set grid resolution (n × n points)"""
        self.resolution = n

    def add_material(self, material: Material, region: Optional[Callable] = None):
        """
        Add conductive material.

        Args:
            material: Material with conductivity properties
            region: Optional region function (x, y) -> bool
                   If None, uses material.region
        """
        if region is not None:
            material.region = region
        self.materials.append(material)

    def add_boundary(self, bc: BoundaryCondition):
        """Add boundary condition (voltage source or ground)"""
        self.boundaries.append(bc)

    def solve(self, method='gmres', verbose=False) -> 'ConductionResult':
        """
        Solve the conduction problem.

        Args:
            method: Linear solver method ('gmres', 'bicgstab', 'direct')
            verbose: Print diagnostics

        Returns:
            ConductionResult with potential, current density, and power density
        """
        if self.domain_x is None or self.domain_y is None:
            raise ValueError("Domain not set - call set_domain() first")

        xmin, xmax = self.domain_x
        ymin, ymax = self.domain_y
        n = self.resolution

        # Create grid
        self.grid_x = np.linspace(xmin, xmax, n)
        self.grid_y = np.linspace(ymin, ymax, n)
        dx = self.grid_x[1] - self.grid_x[0]
        dy = self.grid_y[1] - self.grid_y[0]

        X, Y = np.meshgrid(self.grid_x, self.grid_y, indexing='ij')

        if verbose:
            print(f"Conduction solver: {n}×{n} grid, dx={dx:.3e}, dy={dy:.3e}")

        # Build conductivity field
        self.conductivity_grid = self._build_conductivity_field(X, Y, verbose)

        # Build discrete variable-coefficient operator
        A, b = self._build_system(X, Y, dx, dy, verbose)

        if verbose:
            print(f"Solving linear system ({A.shape[0]} unknowns)...")

        # Solve Ax = b with relaxed tolerance for variable-coefficient problems
        result = solve_linear(A, b, method=method, precond='jacobi', tol=1e-6, verbose=verbose)

        if not result.converged:
            # Try with direct solver as fallback
            if verbose:
                print("  GMRES failed, trying direct solver...")
            result = solve_linear(A, b, method='direct', verbose=verbose)

        if not result.converged:
            raise RuntimeError(
                f"Conduction solver did not converge (residual={result.residual:.2e})"
            )

        # Reshape to 2D grid
        self.potential = result.x.reshape(n, n)

        if verbose:
            V_min, V_max = np.min(self.potential), np.max(self.potential)
            print(f"✓ Solution: V ∈ [{V_min:.2e}, {V_max:.2e}] V")

        # Build material map for result
        material_map = self._build_material_map(X, Y)

        return ConductionResult(
            potential=self.potential,
            conductivity=self.conductivity_grid,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            materials=self.materials,
            material_map=material_map,
        )

    def _build_conductivity_field(self, X, Y, verbose):
        """Build conductivity field σ(x, y)"""
        n = len(self.grid_x)
        sigma = np.full((n, n), self.default_conductivity)

        for material in self.materials:
            if material.region is None:
                continue
            for i in range(n):
                for j in range(n):
                    if material.region(X[i, j], Y[i, j]):
                        sigma[i, j] = material.conductivity

        if verbose:
            sigma_min, sigma_max = np.min(sigma), np.max(sigma)
            print(f"Conductivity: σ ∈ [{sigma_min:.2e}, {sigma_max:.2e}] S/m")
            print(f"Materials: {len(self.materials)}")

        return sigma

    def _build_material_map(self, X, Y) -> Dict[Tuple[int, int], Material]:
        """Build mapping from grid indices to materials"""
        n = len(self.grid_x)
        material_map = {}

        for i in range(n):
            for j in range(n):
                for material in self.materials:
                    if material.region and material.region(X[i, j], Y[i, j]):
                        material_map[(i, j)] = material
                        break

        return material_map

    def _build_system(self, X, Y, dx, dy, verbose):
        """
        Build linear system for variable-coefficient equation: ∇⋅(σ∇V) = 0

        Using finite volume method with harmonic averaging of conductivity
        at cell faces (similar to cost_aware_solver in spatial_flow).
        """
        n = len(self.grid_x)
        N = n * n

        def idx(i, j):
            return i * n + j

        row_indices = []
        col_indices = []
        values = []
        rhs = np.zeros(N)

        sigma = self.conductivity_grid

        for i in range(n):
            for j in range(n):
                k = idx(i, j)
                x, y = X[i, j], Y[i, j]

                # Check for boundary conditions
                bc_applied = False
                for bc in self.boundaries:
                    if bc.region is not None and bc.region(x, y):
                        if bc.type == BoundaryType.DIRICHLET or bc.type == BoundaryType.GROUNDED:
                            row_indices.append(k)
                            col_indices.append(k)
                            values.append(1.0)
                            rhs[k] = bc.value
                            bc_applied = True
                            break

                if bc_applied:
                    continue

                # Interior point - finite volume discretization
                # ∇⋅(σ∇V) ≈ [σ_e(V_E - V_P)/dx - σ_w(V_P - V_W)/dx]/dx
                #          + [σ_n(V_N - V_P)/dy - σ_s(V_P - V_S)/dy]/dy

                # Harmonic mean for conductivity at faces
                # σ_face = 2 * σ_1 * σ_2 / (σ_1 + σ_2)

                coeff_center = 0.0

                # East face (i+1/2, j)
                if i < n - 1:
                    sigma_e = 2 * sigma[i, j] * sigma[i + 1, j] / (sigma[i, j] + sigma[i + 1, j] + 1e-30)
                    coeff_e = sigma_e / (dx * dx)
                    row_indices.append(k)
                    col_indices.append(idx(i + 1, j))
                    values.append(coeff_e)
                    coeff_center -= coeff_e

                # West face (i-1/2, j)
                if i > 0:
                    sigma_w = 2 * sigma[i, j] * sigma[i - 1, j] / (sigma[i, j] + sigma[i - 1, j] + 1e-30)
                    coeff_w = sigma_w / (dx * dx)
                    row_indices.append(k)
                    col_indices.append(idx(i - 1, j))
                    values.append(coeff_w)
                    coeff_center -= coeff_w

                # North face (i, j+1/2)
                if j < n - 1:
                    sigma_n = 2 * sigma[i, j] * sigma[i, j + 1] / (sigma[i, j] + sigma[i, j + 1] + 1e-30)
                    coeff_n = sigma_n / (dy * dy)
                    row_indices.append(k)
                    col_indices.append(idx(i, j + 1))
                    values.append(coeff_n)
                    coeff_center -= coeff_n

                # South face (i, j-1/2)
                if j > 0:
                    sigma_s = 2 * sigma[i, j] * sigma[i, j - 1] / (sigma[i, j] + sigma[i, j - 1] + 1e-30)
                    coeff_s = sigma_s / (dy * dy)
                    row_indices.append(k)
                    col_indices.append(idx(i, j - 1))
                    values.append(coeff_s)
                    coeff_center -= coeff_s

                # Center coefficient
                row_indices.append(k)
                col_indices.append(k)
                values.append(coeff_center)

                # RHS is zero for steady-state (no sources/sinks)
                rhs[k] = 0.0

        A = sparse.coo_matrix((values, (row_indices, col_indices)), shape=(N, N))
        A = A.tocsr()

        if verbose:
            print(f"Matrix: {A.shape}, nnz={A.nnz}, sparsity={100 * A.nnz / (N * N):.1f}%")

        return A, rhs


@dataclass
class ConductionResult:
    """Results from conduction solve"""
    potential: np.ndarray           # V[i, j] potential field
    conductivity: np.ndarray        # σ[i, j] conductivity field
    grid_x: np.ndarray              # x coordinates
    grid_y: np.ndarray              # y coordinates
    materials: List[Material]       # Material list
    material_map: Dict[Tuple[int, int], Material]  # Grid -> material mapping

    def current_density(self, x: float, y: float) -> Tuple[float, float]:
        """
        Compute current density J = -σ∇V at point (x, y).

        Returns:
            (J_x, J_y) current density [A/m²]
        """
        dx = self.grid_x[1] - self.grid_x[0]
        dy = self.grid_y[1] - self.grid_y[0]

        # Find grid cell
        i = np.searchsorted(self.grid_x, x) - 1
        j = np.searchsorted(self.grid_y, y) - 1

        i = np.clip(i, 0, len(self.grid_x) - 2)
        j = np.clip(j, 0, len(self.grid_y) - 2)

        # Central difference for gradient
        if i > 0 and i < len(self.grid_x) - 1:
            dV_dx = (self.potential[i + 1, j] - self.potential[i - 1, j]) / (2 * dx)
        else:
            dV_dx = (self.potential[i + 1, j] - self.potential[i, j]) / dx

        if j > 0 and j < len(self.grid_y) - 1:
            dV_dy = (self.potential[i, j + 1] - self.potential[i, j - 1]) / (2 * dy)
        else:
            dV_dy = (self.potential[i, j + 1] - self.potential[i, j]) / dy

        sigma = self.conductivity[i, j]

        # J = -σ∇V
        return (-sigma * dV_dx, -sigma * dV_dy)

    def current_density_magnitude(self, x: float, y: float) -> float:
        """
        Compute |J| at point (x, y).

        Returns:
            |J| current density magnitude [A/m²]
        """
        J_x, J_y = self.current_density(x, y)
        return np.sqrt(J_x**2 + J_y**2)

    def power_density(self, x: float, y: float) -> float:
        """
        Compute power density P = J²/σ at point (x, y).

        This is the Joule heating power per unit volume.

        Returns:
            P power density [W/m³]
        """
        J_x, J_y = self.current_density(x, y)
        J_mag = np.sqrt(J_x**2 + J_y**2)

        # Find conductivity
        i = np.searchsorted(self.grid_x, x) - 1
        j = np.searchsorted(self.grid_y, y) - 1
        i = np.clip(i, 0, len(self.grid_x) - 2)
        j = np.clip(j, 0, len(self.grid_y) - 2)

        sigma = self.conductivity[i, j]

        # P = J²/σ = |J|²/σ
        if sigma < 1e-20:
            return 0.0
        return J_mag**2 / sigma

    def power_density_grid(self) -> np.ndarray:
        """
        Compute power density on entire grid.

        Returns:
            P[i, j] power density [W/m³]
        """
        n = len(self.grid_x)
        P = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                x, y = self.grid_x[i], self.grid_y[j]
                P[i, j] = self.power_density(x, y)

        return P

    def find_hotspots(self, max_power_density: float) -> List[Tuple[float, float, float, Optional[str]]]:
        """
        Find regions exceeding power density threshold (hotspot detection).

        Args:
            max_power_density: Threshold [W/m³]

        Returns:
            List of (x, y, P, material_name) for hotspot locations
        """
        P_grid = self.power_density_grid()
        hotspots = []

        for i in range(len(self.grid_x)):
            for j in range(len(self.grid_y)):
                if P_grid[i, j] > max_power_density:
                    x, y = self.grid_x[i], self.grid_y[j]
                    material = self.material_map.get((i, j))
                    mat_name = material.name if material else None
                    hotspots.append((x, y, P_grid[i, j], mat_name))

        return hotspots

    def total_current(self, boundary_region: Callable[[float, float], bool]) -> float:
        """
        Compute total current through a boundary.

        Args:
            boundary_region: Function (x, y) -> bool defining boundary

        Returns:
            I total current [A] (assumes 2D cross-section with unit depth)
        """
        dx = self.grid_x[1] - self.grid_x[0]
        dy = self.grid_y[1] - self.grid_y[0]

        total_current = 0.0

        for i in range(len(self.grid_x)):
            for j in range(len(self.grid_y)):
                x, y = self.grid_x[i], self.grid_y[j]
                if boundary_region(x, y):
                    J_x, J_y = self.current_density(x, y)
                    # Assume current flows in +x direction through boundary
                    # Integrate J·n over boundary (simplified)
                    total_current += J_x * dy

        return total_current


class CurrentDensityField:
    """Current density field for visualization and analysis"""
    def __init__(self, result: ConductionResult):
        self.result = result

    def get_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get current density magnitude on grid.

        Returns:
            (X, Y, |J|) meshgrid arrays
        """
        n = len(self.result.grid_x)
        X, Y = np.meshgrid(self.result.grid_x, self.result.grid_y, indexing='ij')
        J_mag = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                x, y = self.result.grid_x[i], self.result.grid_y[j]
                J_mag[i, j] = self.result.current_density_magnitude(x, y)

        return X, Y, J_mag


class PowerDensityField:
    """Power density field for hotspot analysis"""
    def __init__(self, result: ConductionResult):
        self.result = result

    def get_grid(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get power density on grid.

        Returns:
            (X, Y, P) meshgrid arrays [W/m³]
        """
        X, Y = np.meshgrid(self.result.grid_x, self.result.grid_y, indexing='ij')
        P = self.result.power_density_grid()
        return X, Y, P

    def max_power(self) -> Tuple[float, float, float]:
        """
        Find maximum power density location.

        Returns:
            (x, y, P_max) location and value
        """
        P = self.result.power_density_grid()
        i_max, j_max = np.unravel_index(np.argmax(P), P.shape)
        x = self.result.grid_x[i_max]
        y = self.result.grid_y[j_max]
        return (x, y, P[i_max, j_max])
