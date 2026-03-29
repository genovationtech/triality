"""Electrostatic Field Solver

Solves Laplace and Poisson equations for electrostatic potential:
- Laplace: ∇²V = 0 (charge-free regions)
- Poisson: ∇²V = -ρ/ε (regions with charge density)

Uses finite difference discretization on uniform grids.
"""

import numpy as np
from scipy import sparse
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from triality.solvers.linear import solve_linear


class BoundaryType(Enum):
    """Boundary condition types"""
    DIRICHLET = 'dirichlet'  # Fixed potential (voltage)
    NEUMANN = 'neumann'      # Fixed normal field (∂V/∂n)
    GROUNDED = 'grounded'    # V = 0 (special case of Dirichlet)
    FLOATING = 'floating'    # Conductor at unknown potential


@dataclass
class BoundaryCondition:
    """Boundary condition specification"""
    type: BoundaryType
    value: float = 0.0
    region: Optional[Callable[[float, float], bool]] = None  # (x, y) -> bool

    def __repr__(self):
        return f"BC({self.type.value}, V={self.value})"


@dataclass
class ChargeDistribution:
    """Charge density distribution ρ(x, y)"""
    density_func: Callable[[float, float], float]  # (x, y) -> charge density [C/m³]
    region: Optional[Callable[[float, float], bool]] = None  # (x, y) -> bool

    def __call__(self, x: float, y: float) -> float:
        """Evaluate charge density at point"""
        if self.region is not None and not self.region(x, y):
            return 0.0
        return self.density_func(x, y)


class ElectrostaticSolver:
    """
    Electrostatic field solver for 2D domains.

    Solves:
        Laplace:  ∇²V = 0
        Poisson:  ∇²V = -ρ/ε

    Where:
        V = electric potential [V]
        ρ = charge density [C/m³]
        ε = permittivity [F/m]

    Example:
        >>> solver = ElectrostaticSolver()
        >>> solver.set_domain((0, 10), (0, 10))
        >>> solver.set_resolution(50)
        >>> solver.add_boundary(BoundaryCondition(BoundaryType.DIRICHLET, value=100.0,
        ...                     region=lambda x, y: x < 0.1))
        >>> solver.add_boundary(BoundaryCondition(BoundaryType.GROUNDED,
        ...                     region=lambda x, y: x > 9.9))
        >>> result = solver.solve()
        >>> E_x, E_y = result.electric_field(5.0, 5.0)
    """

    def __init__(self):
        self.domain_x = None
        self.domain_y = None
        self.resolution = 50
        self.permittivity = 8.854e-12  # Free space ε₀ [F/m]

        self.boundaries: List[BoundaryCondition] = []
        self.charge_distributions: List[ChargeDistribution] = []

        self.potential = None
        self.grid_x = None
        self.grid_y = None

    def set_domain(self, x_range: Tuple[float, float], y_range: Tuple[float, float]):
        """Set spatial domain [xmin, xmax] × [ymin, ymax]"""
        self.domain_x = x_range
        self.domain_y = y_range

    def set_resolution(self, n: int):
        """Set grid resolution (n × n points)"""
        self.resolution = n

    def set_permittivity(self, epsilon: float):
        """Set permittivity [F/m]"""
        self.permittivity = epsilon

    def add_boundary(self, bc: BoundaryCondition):
        """Add boundary condition"""
        self.boundaries.append(bc)

    def add_charge_distribution(self, charge: ChargeDistribution):
        """Add charge density distribution"""
        self.charge_distributions.append(charge)

    def solve(self, method='gmres', verbose=False) -> 'ElectrostaticResult':
        """
        Solve the electrostatic problem.

        Args:
            method: Linear solver method ('gmres', 'bicgstab', 'direct')
            verbose: Print diagnostics

        Returns:
            ElectrostaticResult with potential field and derived quantities
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
            print(f"Electrostatic solver: {n}×{n} grid, dx={dx:.3e}, dy={dy:.3e}")

        # Build discrete Laplacian matrix
        A, b = self._build_system(X, Y, dx, dy, verbose)

        if verbose:
            print(f"Solving linear system ({A.shape[0]} unknowns)...")

        # Solve Ax = b with relaxed tolerance
        result = solve_linear(A, b, method=method, precond='jacobi', tol=1e-6, verbose=verbose)

        if not result.converged:
            # Try direct solver as fallback
            if verbose:
                print("  GMRES failed, trying direct solver...")
            result = solve_linear(A, b, method='direct', verbose=verbose)

        if not result.converged:
            raise RuntimeError(
                f"Electrostatic solver did not converge (residual={result.residual:.2e})"
            )

        # Reshape to 2D grid
        self.potential = result.x.reshape(n, n)

        if verbose:
            V_min, V_max = np.min(self.potential), np.max(self.potential)
            print(f"✓ Solution: V ∈ [{V_min:.2e}, {V_max:.2e}] V")

        return ElectrostaticResult(
            potential=self.potential,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            permittivity=self.permittivity,
        )

    def _build_system(self, X, Y, dx, dy, verbose):
        """Build linear system for Poisson/Laplace equation"""
        n = len(self.grid_x)
        N = n * n

        # Index mapping: (i, j) -> k
        def idx(i, j):
            return i * n + j

        # Build sparse matrix in COO format
        row_indices = []
        col_indices = []
        values = []
        rhs = np.zeros(N)

        # Finite difference stencil for ∇²V
        # (V[i+1,j] - 2V[i,j] + V[i-1,j])/dx² + (V[i,j+1] - 2V[i,j] + V[i,j-1])/dy²
        dx2 = dx * dx
        dy2 = dy * dy
        center_coeff = -2.0 / dx2 - 2.0 / dy2

        for i in range(n):
            for j in range(n):
                k = idx(i, j)
                x, y = X[i, j], Y[i, j]

                # Check for boundary conditions
                bc_applied = False
                for bc in self.boundaries:
                    if bc.region is not None and bc.region(x, y):
                        if bc.type == BoundaryType.DIRICHLET or bc.type == BoundaryType.GROUNDED:
                            # V[i,j] = bc.value
                            row_indices.append(k)
                            col_indices.append(k)
                            values.append(1.0)
                            rhs[k] = bc.value
                            bc_applied = True
                            break

                if bc_applied:
                    continue

                # Interior point - apply Laplacian stencil
                # Center
                row_indices.append(k)
                col_indices.append(k)
                values.append(center_coeff)

                # Left neighbor
                if i > 0:
                    row_indices.append(k)
                    col_indices.append(idx(i - 1, j))
                    values.append(1.0 / dx2)
                else:
                    # Boundary - use one-sided difference or Neumann BC
                    pass

                # Right neighbor
                if i < n - 1:
                    row_indices.append(k)
                    col_indices.append(idx(i + 1, j))
                    values.append(1.0 / dx2)

                # Bottom neighbor
                if j > 0:
                    row_indices.append(k)
                    col_indices.append(idx(i, j - 1))
                    values.append(1.0 / dy2)

                # Top neighbor
                if j < n - 1:
                    row_indices.append(k)
                    col_indices.append(idx(i, j + 1))
                    values.append(1.0 / dy2)

                # RHS: -ρ/ε (Poisson) or 0 (Laplace)
                charge_density = 0.0
                for charge_dist in self.charge_distributions:
                    charge_density += charge_dist(x, y)

                rhs[k] = -charge_density / self.permittivity

        A = sparse.coo_matrix((values, (row_indices, col_indices)), shape=(N, N))
        A = A.tocsr()

        if verbose:
            print(f"Matrix: {A.shape}, nnz={A.nnz}, sparsity={100 * A.nnz / (N * N):.1f}%")
            print(f"Charge distributions: {len(self.charge_distributions)}")
            print(f"Boundary conditions: {len(self.boundaries)}")

        return A, rhs


@dataclass
class ElectrostaticResult:
    """Results from electrostatic solve"""
    potential: np.ndarray        # V[i, j] potential field
    grid_x: np.ndarray           # x coordinates
    grid_y: np.ndarray           # y coordinates
    permittivity: float          # ε [F/m]

    def electric_field(self, x: float, y: float) -> Tuple[float, float]:
        """
        Compute electric field E = -∇V at point (x, y).

        Returns:
            (E_x, E_y) electric field components [V/m]
        """
        # Bilinear interpolation for gradient
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

        # E = -∇V
        return (-dV_dx, -dV_dy)

    def field_magnitude(self, x: float, y: float) -> float:
        """
        Compute |E| at point (x, y).

        Returns:
            |E| electric field magnitude [V/m]
        """
        E_x, E_y = self.electric_field(x, y)
        return np.sqrt(E_x**2 + E_y**2)

    def field_magnitude_grid(self) -> np.ndarray:
        """
        Compute |E| on entire grid.

        Returns:
            |E|[i, j] field magnitude on grid
        """
        dx = self.grid_x[1] - self.grid_x[0]
        dy = self.grid_y[1] - self.grid_y[0]

        # Compute gradients with central differences
        dV_dx = np.zeros_like(self.potential)
        dV_dy = np.zeros_like(self.potential)

        # Interior points
        dV_dx[1:-1, :] = (self.potential[2:, :] - self.potential[:-2, :]) / (2 * dx)
        dV_dy[:, 1:-1] = (self.potential[:, 2:] - self.potential[:, :-2]) / (2 * dy)

        # Boundary points (one-sided)
        dV_dx[0, :] = (self.potential[1, :] - self.potential[0, :]) / dx
        dV_dx[-1, :] = (self.potential[-1, :] - self.potential[-2, :]) / dx
        dV_dy[:, 0] = (self.potential[:, 1] - self.potential[:, 0]) / dy
        dV_dy[:, -1] = (self.potential[:, -1] - self.potential[:, -2]) / dy

        # |E| = |∇V|
        return np.sqrt(dV_dx**2 + dV_dy**2)

    def max_field_regions(self, threshold_percentile=90) -> List[Tuple[float, float, float]]:
        """
        Find regions with high electric field (crowding zones).

        Args:
            threshold_percentile: Percentile for "high" field (default 90%)

        Returns:
            List of (x, y, |E|) for high-field points
        """
        E_mag = self.field_magnitude_grid()
        threshold = np.percentile(E_mag, threshold_percentile)

        high_field = []
        for i in range(len(self.grid_x)):
            for j in range(len(self.grid_y)):
                if E_mag[i, j] >= threshold:
                    high_field.append((self.grid_x[i], self.grid_y[j], E_mag[i, j]))

        return high_field
