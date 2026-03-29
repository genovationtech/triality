"""Electromagnetic Field Solver for Field-Aware Routing

Solves 2D Laplace/Poisson equations on a rectangular domain using
finite-difference discretization with scipy.sparse, producing field
solutions that feed into the existing cost-field builders and coupling
analyzers.

Physics:
    -div(epsilon * grad(V)) = rho / epsilon_0    (Poisson)
    -div(sigma * grad(V))   = 0                  (Steady conduction)

Discretization uses 5-point stencil on a uniform grid.  Boundary
conditions may be Dirichlet (fixed potential) or Neumann (zero normal
flux).  Material properties (permittivity, conductivity) are specified
per-cell.

Typical workflow:
    1. Create FieldSolverConfig with domain, resolution, BCs, sources
    2. Instantiate EMFieldSolver(config)
    3. Call solver.solve()  -> EMFieldSolverResult
    4. Feed result into ElectricFieldCostBuilder, ClearanceCostBuilder, etc.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
EPSILON_0 = 8.854187817e-12   # Vacuum permittivity [F/m]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
class BCType(Enum):
    """Boundary condition type."""
    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"


@dataclass
class BoundaryCondition:
    """Boundary condition specification for one edge.

    For Dirichlet: value is the fixed potential [V].
    For Neumann:   value is dV/dn [V/m] (0 for insulating boundary).
    """
    bc_type: BCType = BCType.NEUMANN
    value: float = 0.0


@dataclass
class DomainConfig:
    """Rectangular domain specification."""
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    nx: int = 64
    ny: int = 64


@dataclass
class ConductorSpec:
    """Fixed-potential conductor inside the domain.

    region_func(x, y) -> bool identifies which cells belong to the
    conductor.  All interior nodes that satisfy the predicate are
    pinned to ``potential`` via Dirichlet constraints.
    """
    name: str
    potential: float
    region_func: Callable[[float, float], bool]


@dataclass
class ChargeRegion:
    """Volumetric charge density source (for Poisson problems).

    charge_func(x, y) -> rho [C/m^3] returns the free-charge density
    at each point.
    """
    charge_func: Callable[[float, float], float]


@dataclass
class FieldSolverConfig:
    """Complete configuration for the electromagnetic field solver.

    Attributes:
        domain:       Rectangular domain definition.
        bc_left:      Left   (x = x_min) boundary condition.
        bc_right:     Right  (x = x_max) boundary condition.
        bc_bottom:    Bottom (y = y_min) boundary condition.
        bc_top:       Top    (y = y_max) boundary condition.
        conductors:   List of internal conductor specifications.
        charge_regions: List of volumetric charge sources.
        epsilon_r:    Relative permittivity field.  May be a scalar
                      (uniform) or a callable (x, y) -> float.
        sigma:        Conductivity field [S/m].  Scalar or callable.
                      If > 0 the solver treats the problem as steady
                      conduction rather than electrostatics.
        mode:         ``'electrostatic'`` (Poisson/Laplace) or
                      ``'conduction'`` (steady current flow).
    """
    domain: DomainConfig = field(default_factory=DomainConfig)
    bc_left: BoundaryCondition = field(default_factory=BoundaryCondition)
    bc_right: BoundaryCondition = field(default_factory=BoundaryCondition)
    bc_bottom: BoundaryCondition = field(default_factory=BoundaryCondition)
    bc_top: BoundaryCondition = field(default_factory=BoundaryCondition)
    conductors: List[ConductorSpec] = field(default_factory=list)
    charge_regions: List[ChargeRegion] = field(default_factory=list)
    epsilon_r: object = 1.0          # scalar or callable(x,y)->float
    sigma: object = 0.0             # scalar or callable(x,y)->float
    mode: str = "electrostatic"


@dataclass
class EMFieldSolverResult:
    """Result container for the electromagnetic field solver.

    Attributes:
        potential:       2-D array of electric potential V(x, y) [V].
                         Shape (nx, ny).
        grid_x:          1-D array of x-coordinates [m].
        grid_y:          1-D array of y-coordinates [m].
        E_x:             x-component of electric field [V/m].
        E_y:             y-component of electric field [V/m].
        E_magnitude:     |E| field magnitude [V/m].
        J_x:             x-component of current density [A/m^2]
                         (conduction mode only, else zeros).
        J_y:             y-component of current density [A/m^2].
        J_magnitude:     |J| current density magnitude [A/m^2].
        power_density:   Ohmic power dissipation sigma*|E|^2 [W/m^3].
        energy_density:  Electrostatic energy 0.5*eps*|E|^2 [J/m^3].
        total_energy:    Integrated electrostatic energy [J].
        config:          The solver configuration that produced this result.
        converged:       Whether the linear solve succeeded.
        residual_norm:   L2 norm of the residual Ax - b.
    """
    potential: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    E_x: np.ndarray
    E_y: np.ndarray
    E_magnitude: np.ndarray
    J_x: np.ndarray
    J_y: np.ndarray
    J_magnitude: np.ndarray
    power_density: np.ndarray
    energy_density: np.ndarray
    total_energy: float
    config: FieldSolverConfig
    converged: bool
    residual_norm: float

    # ------------------------------------------------------------------
    # Convenience query methods (compatible with cost-field builder API)
    # ------------------------------------------------------------------
    def field_magnitude(self, x: float, y: float) -> float:
        """Interpolate |E| at an arbitrary (x, y) point."""
        return self._interp(self.E_magnitude, x, y)

    def field_magnitude_grid(self) -> np.ndarray:
        """Return the full |E| grid (alias used by coupling_analysis)."""
        return self.E_magnitude

    def current_density_magnitude(self, x: float, y: float) -> float:
        """Interpolate |J| at an arbitrary (x, y) point."""
        return self._interp(self.J_magnitude, x, y)

    def current_density(self, x: float, y: float) -> Tuple[float, float]:
        """Return (Jx, Jy) at an arbitrary (x, y) point."""
        return (self._interp(self.J_x, x, y),
                self._interp(self.J_y, x, y))

    def power_density_grid(self) -> np.ndarray:
        """Return full power-density grid (used by ThermalRiskCostBuilder)."""
        return self.power_density

    # ------------------------------------------------------------------
    # Internal interpolation helper
    # ------------------------------------------------------------------
    def _interp(self, arr: np.ndarray, x: float, y: float) -> float:
        """Bilinear interpolation on the solution grid."""
        gx, gy = self.grid_x, self.grid_y
        if x <= gx[0] or x >= gx[-1] or y <= gy[0] or y >= gy[-1]:
            return 0.0
        i = np.searchsorted(gx, x) - 1
        j = np.searchsorted(gy, y) - 1
        i = np.clip(i, 0, len(gx) - 2)
        j = np.clip(j, 0, len(gy) - 2)

        # Bilinear weights
        tx = (x - gx[i]) / (gx[i + 1] - gx[i])
        ty = (y - gy[j]) / (gy[j + 1] - gy[j])

        val = ((1 - tx) * (1 - ty) * arr[i, j]
               + tx * (1 - ty) * arr[i + 1, j]
               + (1 - tx) * ty * arr[i, j + 1]
               + tx * ty * arr[i + 1, j + 1])
        return float(val)


@dataclass
class PhysicsOptimalResult:
    """Result container for the Level 3 physics-informed optimal routing solver.

    Contains the optimised route path, all intermediate field solutions
    (Laplace guidance, Poisson EMI, steady-state thermal), the composite
    cost field, multi-objective Pareto front, and convergence diagnostics.

    Attributes:
        optimised_path:         (N, 2) array of (x, y) waypoints for the
                                optimised route through the potential field.
        path_length:            Total Euclidean length of the optimised path.
        path_emi_exposure:      Mean normalised EMI field along the path [0-1].
        path_thermal_max:       Maximum temperature encountered along path [C].
        thermal_feasible:       True if path_thermal_max <= max_temperature.
        laplace_potential:      2-D Laplace guidance potential phi(x,y).
        emi_field:              2-D EMI field magnitude |grad(psi)| [V/m].
        emi_cost_field:         Normalised EMI cost [0-1] on the grid.
        poisson_potential:      2-D Poisson potential psi(x,y) for EMI.
        temperature_field:      2-D steady-state temperature T(x,y) [C].
        thermal_cost_field:     Normalised thermal cost on the grid.
        thermal_violation_mask: Boolean grid, True where T > T_max.
        composite_cost_field:   Weighted sum of EMI + thermal + penalty.
        heat_source_field:      Volumetric heat generation Q(x,y) [W/m^3].
        pareto_front:           List of non-dominated Pareto solutions, each
                                a dict with keys 'weights', 'emi_exposure',
                                'thermal_max', 'path_length', 'path'.
        gd_cost_history:        Cost vs iteration for the primary path GD.
        gd_iterations_run:      Number of gradient descent iterations executed.
        laplace_converged:      Whether the Laplace Jacobi solver converged.
        laplace_iterations:     Jacobi iterations used for Laplace solve.
        poisson_converged:      Whether the Poisson Jacobi solver converged.
        poisson_iterations:     Jacobi iterations used for Poisson solve.
        thermal_converged:      Whether the thermal Jacobi solver converged.
        thermal_iterations:     Jacobi iterations used for thermal solve.
        grid_x:                 1-D x-coordinate array.
        grid_y:                 1-D y-coordinate array.
        config:                 The FieldSolverConfig used.
    """
    optimised_path: np.ndarray
    path_length: float
    path_emi_exposure: float
    path_thermal_max: float
    thermal_feasible: bool
    laplace_potential: np.ndarray
    emi_field: np.ndarray
    emi_cost_field: np.ndarray
    poisson_potential: np.ndarray
    temperature_field: np.ndarray
    thermal_cost_field: np.ndarray
    thermal_violation_mask: np.ndarray
    composite_cost_field: np.ndarray
    heat_source_field: np.ndarray
    pareto_front: List[Dict]
    gd_cost_history: List[float]
    gd_iterations_run: int
    laplace_converged: bool
    laplace_iterations: int
    poisson_converged: bool
    poisson_iterations: int
    thermal_converged: bool
    thermal_iterations: int
    grid_x: np.ndarray
    grid_y: np.ndarray
    config: FieldSolverConfig


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class EMFieldSolver:
    """2-D Electromagnetic Field Solver (Laplace / Poisson / Conduction).

    Discretises the governing PDE on a uniform rectangular grid using
    a standard 5-point finite-difference stencil and solves the
    resulting sparse linear system with scipy.sparse.linalg.spsolve.

    Electrostatic mode solves:
        -div(epsilon_r * grad(V)) = rho / epsilon_0

    Conduction mode solves:
        -div(sigma * grad(V)) = 0

    Parameters
    ----------
    config : FieldSolverConfig
        Complete problem specification (domain, BCs, materials, sources).
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(self, config: FieldSolverConfig):
        self.config = config
        d = config.domain
        self.nx = d.nx
        self.ny = d.ny
        self.grid_x = np.linspace(d.x_min, d.x_max, d.nx)
        self.grid_y = np.linspace(d.y_min, d.y_max, d.ny)
        self.dx = self.grid_x[1] - self.grid_x[0]
        self.dy = self.grid_y[1] - self.grid_y[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self) -> EMFieldSolverResult:
        """Assemble and solve the finite-difference system.

        Returns
        -------
        EMFieldSolverResult
            Potential, electric field, current density, derived
            quantities, and convergence information.
        """
        N = self.nx * self.ny
        cfg = self.config

        # Material property grids
        eps_grid = self._build_property_grid(cfg.epsilon_r)
        sig_grid = self._build_property_grid(cfg.sigma)

        # Pick the diffusion coefficient
        if cfg.mode == "conduction":
            coeff = sig_grid
        else:
            coeff = eps_grid  # epsilon_r (actual eps = eps_r * eps_0)

        # Build sparse system  A V = b
        A, b = self._assemble_system(coeff, cfg)

        # Apply conductor Dirichlet constraints
        is_conductor = np.zeros(N, dtype=bool)
        for cond in cfg.conductors:
            for i in range(self.nx):
                for j in range(self.ny):
                    if cond.region_func(self.grid_x[i], self.grid_y[j]):
                        k = i * self.ny + j
                        is_conductor[k] = True
                        # Zero the row, set diagonal to 1, rhs to potential
                        A[k, :] = 0
                        A[k, k] = 1.0
                        b[k] = cond.potential

        # Convert to CSC for direct solve
        A_csc = A.tocsc()

        # Solve
        V_flat = spsolve(A_csc, b)
        V = V_flat.reshape((self.nx, self.ny))

        # Residual
        residual = A_csc @ V_flat - b
        residual_norm = float(np.linalg.norm(residual))
        converged = residual_norm < 1e-8 * max(np.linalg.norm(b), 1.0)

        # Derived fields
        E_x, E_y, E_mag = self._compute_electric_field(V)
        J_x, J_y, J_mag = self._compute_current_density(E_x, E_y, sig_grid)
        power_dens = sig_grid * E_mag ** 2
        energy_dens = 0.5 * eps_grid * EPSILON_0 * E_mag ** 2
        total_energy = float(np.sum(energy_dens) * self.dx * self.dy)

        return EMFieldSolverResult(
            potential=V,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            E_x=E_x,
            E_y=E_y,
            E_magnitude=E_mag,
            J_x=J_x,
            J_y=J_y,
            J_magnitude=J_mag,
            power_density=power_dens,
            energy_density=energy_dens,
            total_energy=total_energy,
            config=self.config,
            converged=converged,
            residual_norm=residual_norm,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_property_grid(self, prop) -> np.ndarray:
        """Evaluate a material property (scalar or callable) on the grid."""
        if callable(prop):
            grid = np.empty((self.nx, self.ny))
            for i in range(self.nx):
                for j in range(self.ny):
                    grid[i, j] = prop(self.grid_x[i], self.grid_y[j])
            return grid
        return np.full((self.nx, self.ny), float(prop))

    def _assemble_system(self, coeff: np.ndarray,
                         cfg: FieldSolverConfig) -> Tuple[sparse.lil_matrix, np.ndarray]:
        """Build the sparse coefficient matrix and RHS vector.

        Uses a 5-point stencil with harmonic averaging of the
        diffusion coefficient at cell interfaces to handle material
        discontinuities correctly.

        Boundary conditions are applied row-by-row.
        """
        nx, ny = self.nx, self.ny
        N = nx * ny
        dx2 = self.dx ** 2
        dy2 = self.dy ** 2

        A = sparse.lil_matrix((N, N))
        b = np.zeros(N)

        def idx(i, j):
            return i * ny + j

        for i in range(nx):
            for j in range(ny):
                k = idx(i, j)

                # --- Boundary nodes ---
                on_left   = (i == 0)
                on_right  = (i == nx - 1)
                on_bottom = (j == 0)
                on_top    = (j == ny - 1)

                # Dirichlet boundary overrides
                if on_left and cfg.bc_left.bc_type == BCType.DIRICHLET:
                    A[k, k] = 1.0
                    b[k] = cfg.bc_left.value
                    continue
                if on_right and cfg.bc_right.bc_type == BCType.DIRICHLET:
                    A[k, k] = 1.0
                    b[k] = cfg.bc_right.value
                    continue
                if on_bottom and cfg.bc_bottom.bc_type == BCType.DIRICHLET:
                    A[k, k] = 1.0
                    b[k] = cfg.bc_bottom.value
                    continue
                if on_top and cfg.bc_top.bc_type == BCType.DIRICHLET:
                    A[k, k] = 1.0
                    b[k] = cfg.bc_top.value
                    continue

                # --- Interior / Neumann stencil ---
                # Harmonic average of coefficient at each interface
                c_here = coeff[i, j]
                diag = 0.0

                # West (i-1, j)
                if i > 0:
                    c_w = 2.0 * c_here * coeff[i - 1, j] / (c_here + coeff[i - 1, j] + 1e-30)
                    A[k, idx(i - 1, j)] = c_w / dx2
                    diag -= c_w / dx2
                else:
                    # Neumann: ghost node mirrors interior
                    # dV/dn = bc_value  =>  V_ghost = V_1 - 2*dx*bc_value
                    # The coefficient of V_1 absorbs the ghost contribution
                    pass

                # East (i+1, j)
                if i < nx - 1:
                    c_e = 2.0 * c_here * coeff[i + 1, j] / (c_here + coeff[i + 1, j] + 1e-30)
                    A[k, idx(i + 1, j)] = c_e / dx2
                    diag -= c_e / dx2
                else:
                    pass

                # South (i, j-1)
                if j > 0:
                    c_s = 2.0 * c_here * coeff[i, j - 1] / (c_here + coeff[i, j - 1] + 1e-30)
                    A[k, idx(i, j - 1)] = c_s / dy2
                    diag -= c_s / dy2
                else:
                    pass

                # North (i, j+1)
                if j < ny - 1:
                    c_n = 2.0 * c_here * coeff[i, j + 1] / (c_here + coeff[i, j + 1] + 1e-30)
                    A[k, idx(i, j + 1)] = c_n / dy2
                    diag -= c_n / dy2
                else:
                    pass

                A[k, k] = diag

                # RHS: charge density (electrostatic mode only)
                if cfg.mode == "electrostatic":
                    rho = 0.0
                    for cr in cfg.charge_regions:
                        rho += cr.charge_func(self.grid_x[i], self.grid_y[j])
                    b[k] = -rho / EPSILON_0

        return A, b

    def _compute_electric_field(self, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """E = -grad(V) via central differences (forward/backward at edges)."""
        Ex = np.zeros_like(V)
        Ey = np.zeros_like(V)

        # Central differences for interior
        Ex[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2.0 * self.dx)
        Ey[:, 1:-1] = -(V[:, 2:] - V[:, :-2]) / (2.0 * self.dy)

        # Forward/backward at boundaries
        Ex[0, :]  = -(V[1, :] - V[0, :]) / self.dx
        Ex[-1, :] = -(V[-1, :] - V[-2, :]) / self.dx
        Ey[:, 0]  = -(V[:, 1] - V[:, 0]) / self.dy
        Ey[:, -1] = -(V[:, -1] - V[:, -2]) / self.dy

        E_mag = np.sqrt(Ex ** 2 + Ey ** 2)
        return Ex, Ey, E_mag

    def _compute_current_density(self, Ex: np.ndarray, Ey: np.ndarray,
                                  sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """J = sigma * E (Ohm's law)."""
        Jx = sigma * Ex
        Jy = sigma * Ey
        J_mag = np.sqrt(Jx ** 2 + Jy ** 2)
        return Jx, Jy, J_mag

    def export_state(self, result: EMFieldSolverResult) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="field_aware_routing")
        state.set_field("electric_potential", result.potential, "V")
        state.set_field("electric_field_x", result.E_x, "V/m")
        state.set_field("electric_field_y", result.E_y, "V/m")
        state.set_field("electric_field_magnitude", result.E_magnitude, "V/m")
        if np.any(result.power_density > 0):
            state.set_field("power_density", result.power_density, "W/m^3")
        state.metadata["total_energy"] = result.total_energy
        state.metadata["converged"] = result.converged
        state.metadata["residual_norm"] = result.residual_norm
        return state

    # ------------------------------------------------------------------
    # Level 3: Physics-informed optimal routing
    # ------------------------------------------------------------------
    def solve_physics_optimal(
        self,
        route_start: Tuple[float, float] = (0.1, 0.5),
        route_end: Tuple[float, float] = (0.9, 0.5),
        heat_sources: Optional[List[Tuple[float, float, float]]] = None,
        thermal_conductivity: float = 150.0,
        max_temperature: float = 85.0,
        ambient_temperature: float = 25.0,
        emi_weight: float = 1.0,
        thermal_weight: float = 1.0,
        length_weight: float = 1.0,
        n_path_points: int = 40,
        gd_learning_rate: float = 0.002,
        gd_iterations: int = 500,
        pareto_samples: int = 30,
        poisson_source_func: Optional[Callable[[float, float], float]] = None,
    ) -> "PhysicsOptimalResult":
        """Physics-informed optimal routing with potential fields, thermal
        constraints, and multi-objective Pareto optimisation.

        Solves three coupled 2-D PDEs on the domain grid:

        1. **Laplace equation** for routing cost potential:
           Laplacian(phi) = 0 with Dirichlet BCs at source/sink to
           create a smooth guidance field.  Solved via Jacobi iteration.

        2. **Poisson equation** for EMI field strength:
           Laplacian(psi) = -f(x,y) where f encodes conductor/source
           charge distributions.  Solved via FDM with 5-point stencil.

        3. **Steady-state heat equation** (Poisson form):
           k * Laplacian(T) = -Q(x,y) for thermal constraint evaluation.
           Heat sources are modelled as Gaussian blobs.

        The routing cost field is a weighted combination of the EMI field
        magnitude, thermal proximity penalty, and Euclidean path length.
        Gradient descent on this composite cost field yields an optimised
        path.  A Pareto front is generated by sweeping the weight space.

        Parameters
        ----------
        route_start : tuple
            (x, y) coordinates of the route origin.
        route_end : tuple
            (x, y) coordinates of the route destination.
        heat_sources : list of (x, y, power) tuples, optional
            Gaussian heat source locations and power [W/m].
            Defaults to domain-centre source if None.
        thermal_conductivity : float
            Isotropic thermal conductivity k [W/(m*K)].
        max_temperature : float
            Maximum permissible temperature [C] for constraint.
        ambient_temperature : float
            Boundary / ambient temperature [C].
        emi_weight : float
            Weight for EMI cost component in composite objective.
        thermal_weight : float
            Weight for thermal proximity penalty.
        length_weight : float
            Weight for path length component.
        n_path_points : int
            Number of interior waypoints in the optimised path.
        gd_learning_rate : float
            Step size for gradient descent path optimisation.
        gd_iterations : int
            Number of gradient descent iterations.
        pareto_samples : int
            Number of weight combinations for Pareto front sampling.
        poisson_source_func : callable, optional
            Source term f(x,y) for the EMI Poisson equation.
            Defaults to conductor-derived source if None.

        Returns
        -------
        PhysicsOptimalResult
            Optimised path, field solutions, Pareto front, and diagnostics.
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        gx, gy = self.grid_x, self.grid_y

        if heat_sources is None:
            cx = 0.5 * (gx[0] + gx[-1])
            cy = 0.5 * (gy[0] + gy[-1])
            heat_sources = [(cx, cy, 50.0)]

        # ==============================================================
        # Step 1: Solve Laplace equation for routing guidance potential
        #         Laplacian(phi) = 0, Dirichlet at start/end
        # ==============================================================
        phi = np.zeros((nx, ny))

        # Set Dirichlet BCs: phi=0 at start region, phi=1 at end region
        start_i = np.argmin(np.abs(gx - route_start[0]))
        start_j = np.argmin(np.abs(gy - route_start[1]))
        end_i = np.argmin(np.abs(gx - route_end[0]))
        end_j = np.argmin(np.abs(gy - route_end[1]))

        # Mask for fixed nodes (source/sink pads, 3x3 stencil)
        fixed = np.zeros((nx, ny), dtype=bool)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                si, sj = start_i + di, start_j + dj
                ei, ej = end_i + di, end_j + dj
                if 0 <= si < nx and 0 <= sj < ny:
                    fixed[si, sj] = True
                    phi[si, sj] = 0.0
                if 0 <= ei < nx and 0 <= ej < ny:
                    fixed[ei, ej] = True
                    phi[ei, ej] = 1.0

        # Jacobi iteration for Laplace equation
        rx = dy * dy / (2.0 * (dx * dx + dy * dy))
        ry = dx * dx / (2.0 * (dx * dx + dy * dy))
        laplace_max_iter = 2000
        laplace_tol = 1e-6

        for iteration in range(laplace_max_iter):
            phi_old = phi.copy()
            # 5-point Jacobi update
            phi[1:-1, 1:-1] = (
                rx * (phi_old[2:, 1:-1] + phi_old[:-2, 1:-1])
                + ry * (phi_old[1:-1, 2:] + phi_old[1:-1, :-2])
            )
            # Neumann (zero-flux) on domain boundary
            phi[0, :] = phi[1, :]
            phi[-1, :] = phi[-2, :]
            phi[:, 0] = phi[:, 1]
            phi[:, -1] = phi[:, -2]
            # Re-impose Dirichlet constraints
            phi[fixed] = phi_old[fixed]

            residual = np.max(np.abs(phi - phi_old))
            if residual < laplace_tol:
                break

        laplace_converged = (residual < laplace_tol)
        laplace_iterations = iteration + 1

        # ==============================================================
        # Step 2: Solve Poisson equation for EMI field
        #         Laplacian(psi) = -f(x,y) via Jacobi iteration
        # ==============================================================
        if poisson_source_func is not None:
            source_grid = np.zeros((nx, ny))
            for i in range(nx):
                for j in range(ny):
                    source_grid[i, j] = poisson_source_func(gx[i], gy[j])
        else:
            # Default: derive source from conductors or use Gaussian blobs
            source_grid = np.zeros((nx, ny))
            if self.config.conductors:
                for cond in self.config.conductors:
                    for i in range(nx):
                        for j in range(ny):
                            if cond.region_func(gx[i], gy[j]):
                                source_grid[i, j] = cond.potential * 10.0
            else:
                # Fallback: place source blobs at heat source locations
                for (hx, hy, hp) in heat_sources:
                    xx = gx[:, None] - hx
                    yy = gy[None, :] - hy
                    sigma_blob = 0.05 * (gx[-1] - gx[0])
                    source_grid += hp * np.exp(
                        -(xx ** 2 + yy ** 2) / (2.0 * sigma_blob ** 2)
                    )

        psi = np.zeros((nx, ny))
        poisson_max_iter = 3000
        poisson_tol = 1e-5

        for iteration in range(poisson_max_iter):
            psi_old = psi.copy()
            psi[1:-1, 1:-1] = (
                rx * (psi_old[2:, 1:-1] + psi_old[:-2, 1:-1])
                + ry * (psi_old[1:-1, 2:] + psi_old[1:-1, :-2])
                + 0.5 * dx * dy * source_grid[1:-1, 1:-1]
            )
            # Zero-potential boundary
            psi[0, :] = 0.0
            psi[-1, :] = 0.0
            psi[:, 0] = 0.0
            psi[:, -1] = 0.0

            res_poisson = np.max(np.abs(psi - psi_old))
            if res_poisson < poisson_tol:
                break

        poisson_converged = (res_poisson < poisson_tol)
        poisson_iterations = iteration + 1

        # EMI field = |grad(psi)|
        emi_Ex = np.zeros((nx, ny))
        emi_Ey = np.zeros((nx, ny))
        emi_Ex[1:-1, :] = (psi[2:, :] - psi[:-2, :]) / (2.0 * dx)
        emi_Ey[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dy)
        emi_Ex[0, :] = (psi[1, :] - psi[0, :]) / dx
        emi_Ex[-1, :] = (psi[-1, :] - psi[-2, :]) / dx
        emi_Ey[:, 0] = (psi[:, 1] - psi[:, 0]) / dy
        emi_Ey[:, -1] = (psi[:, -1] - psi[:, -2]) / dy
        emi_magnitude = np.sqrt(emi_Ex ** 2 + emi_Ey ** 2)

        # Normalise to [0, 1]
        emi_max = np.max(emi_magnitude)
        if emi_max > 0:
            emi_cost = emi_magnitude / emi_max
        else:
            emi_cost = np.zeros((nx, ny))

        # ==============================================================
        # Step 3: Solve steady-state heat equation
        #         k * Laplacian(T) = -Q(x,y)
        #         Dirichlet BC: T = T_ambient on all boundaries
        # ==============================================================
        Q_grid = np.zeros((nx, ny))
        for (hx, hy, hp) in heat_sources:
            xx = gx[:, None] - hx
            yy = gy[None, :] - hy
            sigma_heat = 0.03 * (gx[-1] - gx[0])
            Q_grid += hp / (2.0 * np.pi * sigma_heat ** 2) * np.exp(
                -(xx ** 2 + yy ** 2) / (2.0 * sigma_heat ** 2)
            )

        T = np.full((nx, ny), ambient_temperature)
        T[0, :] = ambient_temperature
        T[-1, :] = ambient_temperature
        T[:, 0] = ambient_temperature
        T[:, -1] = ambient_temperature

        thermal_max_iter = 3000
        thermal_tol = 1e-5
        k_inv = 1.0 / thermal_conductivity

        for iteration in range(thermal_max_iter):
            T_old = T.copy()
            T[1:-1, 1:-1] = (
                rx * (T_old[2:, 1:-1] + T_old[:-2, 1:-1])
                + ry * (T_old[1:-1, 2:] + T_old[1:-1, :-2])
                + 0.5 * dx * dy * k_inv * Q_grid[1:-1, 1:-1]
            )
            # Re-impose Dirichlet boundary
            T[0, :] = ambient_temperature
            T[-1, :] = ambient_temperature
            T[:, 0] = ambient_temperature
            T[:, -1] = ambient_temperature

            res_thermal = np.max(np.abs(T - T_old))
            if res_thermal < thermal_tol:
                break

        thermal_converged = (res_thermal < thermal_tol)
        thermal_iterations = iteration + 1

        # Thermal cost: penalty proportional to (T - T_ambient) / (T_max - T_ambient)
        T_range = max_temperature - ambient_temperature
        thermal_cost = np.clip(
            (T - ambient_temperature) / T_range, 0.0, 2.0
        )

        # Thermal constraint mask: True where T exceeds max
        thermal_violation = T > max_temperature

        # ==============================================================
        # Step 4: Build composite routing cost field
        # ==============================================================
        composite_cost = (
            emi_weight * emi_cost
            + thermal_weight * thermal_cost
            + 5.0 * thermal_violation.astype(float)  # hard penalty
        )

        # ==============================================================
        # Step 5: Gradient descent path optimisation on cost field
        # ==============================================================
        def _interpolate_field(field_arr: np.ndarray,
                               x: float, y: float) -> float:
            """Bilinear interpolation on the grid."""
            fi = (x - gx[0]) / dx
            fj = (y - gy[0]) / dy
            i0 = int(np.clip(np.floor(fi), 0, nx - 2))
            j0 = int(np.clip(np.floor(fj), 0, ny - 2))
            tx = fi - i0
            ty = fj - j0
            tx = np.clip(tx, 0.0, 1.0)
            ty = np.clip(ty, 0.0, 1.0)
            return float(
                (1 - tx) * (1 - ty) * field_arr[i0, j0]
                + tx * (1 - ty) * field_arr[i0 + 1, j0]
                + (1 - tx) * ty * field_arr[i0, j0 + 1]
                + tx * ty * field_arr[i0 + 1, j0 + 1]
            )

        def _field_gradient(field_arr: np.ndarray,
                            x: float, y: float) -> Tuple[float, float]:
            """Numerical gradient of an interpolated field."""
            eps = 0.5 * min(dx, dy)
            x_lo = max(gx[0], x - eps)
            x_hi = min(gx[-1], x + eps)
            y_lo = max(gy[0], y - eps)
            y_hi = min(gy[-1], y + eps)
            dfdx = (_interpolate_field(field_arr, x_hi, y)
                    - _interpolate_field(field_arr, x_lo, y)) / (x_hi - x_lo)
            dfdy = (_interpolate_field(field_arr, x, y_hi)
                    - _interpolate_field(field_arr, x, y_lo)) / (y_hi - y_lo)
            return dfdx, dfdy

        # Initialise path as straight line
        sx, sy = route_start
        ex, ey = route_end
        path_x = np.linspace(sx, ex, n_path_points + 2)
        path_y = np.linspace(sy, ey, n_path_points + 2)

        # Gradient descent: optimise interior waypoints
        gd_cost_history = []
        for gd_iter in range(gd_iterations):
            total_cost = 0.0
            grad_x = np.zeros(n_path_points)
            grad_y = np.zeros(n_path_points)

            for p in range(n_path_points):
                pi = p + 1  # index in full path (skip start)
                px, py_ = path_x[pi], path_y[pi]

                # Cost field contribution
                c_val = _interpolate_field(composite_cost, px, py_)
                gc_x, gc_y = _field_gradient(composite_cost, px, py_)
                total_cost += c_val

                # Length regularisation: penalise deviation from neighbors
                prev_x, prev_y = path_x[pi - 1], path_y[pi - 1]
                next_x, next_y = path_x[pi + 1], path_y[pi + 1]
                seg_grad_x = length_weight * (
                    2.0 * px - prev_x - next_x
                )
                seg_grad_y = length_weight * (
                    2.0 * py_ - prev_y - next_y
                )

                grad_x[p] = gc_x + seg_grad_x
                grad_y[p] = gc_y + seg_grad_y

            # Normalise gradient to avoid instability
            grad_norm = np.sqrt(np.sum(grad_x ** 2 + grad_y ** 2))
            if grad_norm > 1e-12:
                grad_x /= grad_norm
                grad_y /= grad_norm

            # Update interior waypoints
            for p in range(n_path_points):
                pi = p + 1
                path_x[pi] -= gd_learning_rate * grad_x[p]
                path_y[pi] -= gd_learning_rate * grad_y[p]
                # Clamp to domain
                path_x[pi] = np.clip(path_x[pi], gx[0], gx[-1])
                path_y[pi] = np.clip(path_y[pi], gy[0], gy[-1])

            gd_cost_history.append(total_cost)

        # Compute final path metrics
        optimised_path = np.column_stack([path_x, path_y])
        segments = np.diff(optimised_path, axis=0)
        segment_lengths = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2)
        path_length = float(np.sum(segment_lengths))

        # EMI exposure along path
        path_emi = 0.0
        for k in range(len(path_x)):
            path_emi += _interpolate_field(emi_cost, path_x[k], path_y[k])
        path_emi /= len(path_x)

        # Thermal exposure along path
        path_thermal_max = 0.0
        path_thermal_values = []
        for k in range(len(path_x)):
            t_val = _interpolate_field(T, path_x[k], path_y[k])
            path_thermal_values.append(t_val)
            path_thermal_max = max(path_thermal_max, t_val)
        thermal_feasible = path_thermal_max <= max_temperature

        # ==============================================================
        # Step 6: Multi-objective Pareto front
        # ==============================================================
        pareto_front = []

        for s_idx in range(pareto_samples):
            # Sample weight combinations on simplex
            alpha = s_idx / max(pareto_samples - 1, 1)
            w_emi = 1.0 - alpha
            w_therm = alpha
            w_len = 0.3 + 0.7 * abs(0.5 - alpha)

            trial_cost = w_emi * emi_cost + w_therm * thermal_cost
            trial_cost += 5.0 * thermal_violation.astype(float)

            # Quick gradient descent for this weight combo
            t_path_x = np.linspace(sx, ex, n_path_points + 2)
            t_path_y = np.linspace(sy, ey, n_path_points + 2)

            for _ in range(min(gd_iterations // 2, 200)):
                for p in range(n_path_points):
                    pi = p + 1
                    tpx, tpy = t_path_x[pi], t_path_y[pi]
                    gc_x, gc_y = _field_gradient(trial_cost, tpx, tpy)
                    prev_x, prev_y = t_path_x[pi - 1], t_path_y[pi - 1]
                    next_x, next_y = t_path_x[pi + 1], t_path_y[pi + 1]
                    gx_total = gc_x + w_len * (2.0 * tpx - prev_x - next_x)
                    gy_total = gc_y + w_len * (2.0 * tpy - prev_y - next_y)
                    g_norm = np.sqrt(gx_total ** 2 + gy_total ** 2) + 1e-15
                    t_path_x[pi] -= gd_learning_rate * gx_total / g_norm
                    t_path_y[pi] -= gd_learning_rate * gy_total / g_norm
                    t_path_x[pi] = np.clip(t_path_x[pi], gx[0], gx[-1])
                    t_path_y[pi] = np.clip(t_path_y[pi], gy[0], gy[-1])

            # Evaluate three objectives for this path
            p_segs = np.diff(
                np.column_stack([t_path_x, t_path_y]), axis=0
            )
            p_len = float(np.sum(np.sqrt(p_segs[:, 0] ** 2 + p_segs[:, 1] ** 2)))

            p_emi = 0.0
            p_tmax = 0.0
            for k in range(len(t_path_x)):
                p_emi += _interpolate_field(emi_cost, t_path_x[k], t_path_y[k])
                t_val = _interpolate_field(T, t_path_x[k], t_path_y[k])
                p_tmax = max(p_tmax, t_val)
            p_emi /= len(t_path_x)

            pareto_front.append({
                "weights": (w_emi, w_therm, w_len),
                "emi_exposure": p_emi,
                "thermal_max": p_tmax,
                "path_length": p_len,
                "path": np.column_stack([t_path_x, t_path_y]),
            })

        # Filter to non-dominated solutions
        dominated = [False] * len(pareto_front)
        for i in range(len(pareto_front)):
            for j in range(len(pareto_front)):
                if i == j:
                    continue
                pi, pj = pareto_front[i], pareto_front[j]
                if (pj["emi_exposure"] <= pi["emi_exposure"]
                        and pj["thermal_max"] <= pi["thermal_max"]
                        and pj["path_length"] <= pi["path_length"]
                        and (pj["emi_exposure"] < pi["emi_exposure"]
                             or pj["thermal_max"] < pi["thermal_max"]
                             or pj["path_length"] < pi["path_length"])):
                    dominated[i] = True
                    break
        pareto_nondominated = [
            pareto_front[i] for i in range(len(pareto_front))
            if not dominated[i]
        ]

        return PhysicsOptimalResult(
            optimised_path=optimised_path,
            path_length=path_length,
            path_emi_exposure=path_emi,
            path_thermal_max=path_thermal_max,
            thermal_feasible=thermal_feasible,
            laplace_potential=phi,
            emi_field=emi_magnitude,
            emi_cost_field=emi_cost,
            poisson_potential=psi,
            temperature_field=T,
            thermal_cost_field=thermal_cost,
            thermal_violation_mask=thermal_violation,
            composite_cost_field=composite_cost,
            heat_source_field=Q_grid,
            pareto_front=pareto_nondominated,
            gd_cost_history=gd_cost_history,
            gd_iterations_run=gd_iterations,
            laplace_converged=laplace_converged,
            laplace_iterations=laplace_iterations,
            poisson_converged=poisson_converged,
            poisson_iterations=poisson_iterations,
            thermal_converged=thermal_converged,
            thermal_iterations=thermal_iterations,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
            config=self.config,
        )


# ========================================================================
# Level 3 2-D: Diffusion-based field-aware routing solver
# ========================================================================

@dataclass
class FieldAwareRouting2DResult:
    """Result container for 2-D cost-field routing solver.

    Attributes
    ----------
    cost_field : np.ndarray
        Diffusion-based navigation cost field, shape (ny, nx).
    path_x : np.ndarray
        Optimal path x-coordinates.
    path_y : np.ndarray
        Optimal path y-coordinates.
    obstacle_mask : np.ndarray
        Boolean obstacle/threat mask, shape (ny, nx).
    gradient_x : np.ndarray
        x-component of cost gradient (descent direction), shape (ny, nx).
    gradient_y : np.ndarray
        y-component of cost gradient, shape (ny, nx).
    x : np.ndarray
        x-coordinates (nx,).
    y : np.ndarray
        y-coordinates (ny,).
    path_length : float
        Total Euclidean path length.
    path_cost : float
        Integrated cost along path.
    n_iterations : int
        Diffusion solver iterations.
    converged : bool
        Whether diffusion solve converged.
    """
    cost_field: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    path_x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    path_y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    obstacle_mask: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    gradient_x: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    gradient_y: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    path_length: float = 0.0
    path_cost: float = 0.0
    n_iterations: int = 0
    converged: bool = False


class FieldAwareRouting2DSolver:
    """2-D cost-field routing solver using diffusion-based path planning.

    Solves a Laplace/diffusion equation on a 2-D domain with obstacles
    and threat regions acting as high-cost barriers. The cost field is
    computed via Jacobi relaxation with Dirichlet conditions at the goal
    (phi=0) and Neumann boundaries elsewhere. A steepest-descent trace
    from the start to the goal yields the optimal path.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    Lx, Ly : float
        Domain size [m].
    start : tuple
        (x, y) start position [m].
    goal : tuple
        (x, y) goal position [m].
    obstacles : list of tuples
        List of (cx, cy, radius) circular obstacles [m].
    threats : list of tuples
        List of (cx, cy, radius, strength) threat zones.
    diffusivity : float
        Base diffusion coefficient.
    max_iterations : int
        Maximum Jacobi iterations for cost field.
    tolerance : float
        Convergence tolerance.
    path_step : float
        Step size for gradient-descent path tracing [m].
    """

    fidelity_tier = FidelityTier.HIGH_FIDELITY
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        nx: int = 80,
        ny: int = 80,
        Lx: float = 10.0,
        Ly: float = 10.0,
        start: Tuple[float, float] = (0.5, 0.5),
        goal: Tuple[float, float] = (9.5, 9.5),
        obstacles: Optional[List[Tuple[float, float, float]]] = None,
        threats: Optional[List[Tuple[float, float, float, float]]] = None,
        diffusivity: float = 1.0,
        max_iterations: int = 5000,
        tolerance: float = 1e-5,
        path_step: float = 0.05,
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.start = start
        self.goal = goal
        self.obstacles = obstacles if obstacles is not None else []
        self.threats = threats if threats is not None else []
        self.diffusivity = diffusivity
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.path_step = path_step

    def solve(self) -> FieldAwareRouting2DResult:
        """Solve the diffusion-based routing problem.

        Returns
        -------
        FieldAwareRouting2DResult
        """
        nx, ny = self.nx, self.ny
        dx = self.Lx / (nx - 1)
        dy = self.Ly / (ny - 1)
        x = np.linspace(0.0, self.Lx, nx)
        y = np.linspace(0.0, self.Ly, ny)
        X, Y = np.meshgrid(x, y)  # (ny, nx)

        # Build obstacle mask and conductivity field
        obstacle_mask = np.zeros((ny, nx), dtype=bool)
        conductivity = np.full((ny, nx), self.diffusivity)

        for (cx, cy, r) in self.obstacles:
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            obstacle_mask[dist < r] = True
            conductivity[dist < r] = 1e-6  # near-zero diffusion in obstacles

        # Threat zones: reduce conductivity (increases cost)
        for (cx, cy, r, strength) in self.threats:
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            threat_factor = strength * np.exp(-dist ** 2 / (2.0 * r ** 2))
            conductivity = conductivity / (1.0 + threat_factor)

        # Initialise cost field: large everywhere, 0 at goal
        phi = np.ones((ny, nx)) * 1e6

        # Find goal cell
        gi = np.argmin(np.abs(x - self.goal[0]))
        gj = np.argmin(np.abs(y - self.goal[1]))
        goal_mask = np.zeros((ny, nx), dtype=bool)
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ii, jj = gi + di, gj + dj
                if 0 <= jj < ny and 0 <= ii < nx:
                    goal_mask[jj, ii] = True
                    phi[jj, ii] = 0.0

        # Jacobi iteration for diffusion equation
        dx2 = dx ** 2
        dy2 = dy ** 2
        rx = 1.0 / dx2
        ry = 1.0 / dy2

        converged = False
        n_iter = 0

        for iteration in range(self.max_iterations):
            phi_old = phi.copy()

            # 5-point Jacobi with variable conductivity
            phi[1:-1, 1:-1] = (
                rx * conductivity[1:-1, 1:-1] * (phi_old[1:-1, 2:] + phi_old[1:-1, :-2])
                + ry * conductivity[1:-1, 1:-1] * (phi_old[2:, 1:-1] + phi_old[:-2, 1:-1])
            ) / (2.0 * conductivity[1:-1, 1:-1] * (rx + ry) + 1e-30)

            # Neumann BCs (zero gradient at domain edges)
            phi[0, :] = phi[1, :]
            phi[-1, :] = phi[-2, :]
            phi[:, 0] = phi[:, 1]
            phi[:, -1] = phi[:, -2]

            # Enforce Dirichlet at goal
            phi[goal_mask] = 0.0

            # Obstacles: set high cost
            phi[obstacle_mask] = 1e6

            residual = np.max(np.abs(phi[1:-1, 1:-1] - phi_old[1:-1, 1:-1]))
            n_iter = iteration + 1
            if residual < self.tolerance:
                converged = True
                break

        # Cost gradient for path tracing
        grad_x = np.zeros((ny, nx))
        grad_y = np.zeros((ny, nx))
        grad_x[:, 1:-1] = (phi[:, 2:] - phi[:, :-2]) / (2.0 * dx)
        grad_x[:, 0] = (phi[:, 1] - phi[:, 0]) / dx
        grad_x[:, -1] = (phi[:, -1] - phi[:, -2]) / dx
        grad_y[1:-1, :] = (phi[2:, :] - phi[:-2, :]) / (2.0 * dy)
        grad_y[0, :] = (phi[1, :] - phi[0, :]) / dy
        grad_y[-1, :] = (phi[-1, :] - phi[-2, :]) / dy

        # Trace path from start to goal via steepest descent
        px, py = self.start
        path_x_list = [px]
        path_y_list = [py]
        step = self.path_step
        max_path_steps = 10 * (nx + ny)

        for _ in range(max_path_steps):
            # Bilinear interpolation of gradient
            fi = (px - x[0]) / dx
            fj = (py - y[0]) / dy
            i0 = int(np.clip(np.floor(fi), 0, nx - 2))
            j0 = int(np.clip(np.floor(fj), 0, ny - 2))
            tx = np.clip(fi - i0, 0.0, 1.0)
            ty = np.clip(fj - j0, 0.0, 1.0)

            gx_val = ((1 - tx) * (1 - ty) * grad_x[j0, i0]
                       + tx * (1 - ty) * grad_x[j0, i0 + 1]
                       + (1 - tx) * ty * grad_x[j0 + 1, i0]
                       + tx * ty * grad_x[j0 + 1, i0 + 1])
            gy_val = ((1 - tx) * (1 - ty) * grad_y[j0, i0]
                       + tx * (1 - ty) * grad_y[j0, i0 + 1]
                       + (1 - tx) * ty * grad_y[j0 + 1, i0]
                       + tx * ty * grad_y[j0 + 1, i0 + 1])

            g_mag = np.sqrt(gx_val ** 2 + gy_val ** 2)
            if g_mag < 1e-12:
                break

            px -= step * gx_val / g_mag
            py -= step * gy_val / g_mag

            px = np.clip(px, x[0], x[-1])
            py = np.clip(py, y[0], y[-1])
            path_x_list.append(px)
            path_y_list.append(py)

            # Check if reached goal
            if np.sqrt((px - self.goal[0]) ** 2 + (py - self.goal[1]) ** 2) < 2.0 * step:
                break

        path_x_arr = np.array(path_x_list)
        path_y_arr = np.array(path_y_list)

        # Path metrics
        segments = np.sqrt(np.diff(path_x_arr) ** 2 + np.diff(path_y_arr) ** 2)
        path_length = float(np.sum(segments))

        # Integrated cost along path
        path_cost = 0.0
        for k in range(len(path_x_arr)):
            fi = (path_x_arr[k] - x[0]) / dx
            fj = (path_y_arr[k] - y[0]) / dy
            i0 = int(np.clip(fi, 0, nx - 1))
            j0 = int(np.clip(fj, 0, ny - 1))
            path_cost += phi[j0, i0]
        path_cost /= max(len(path_x_arr), 1)

        return FieldAwareRouting2DResult(
            cost_field=phi,
            path_x=path_x_arr,
            path_y=path_y_arr,
            obstacle_mask=obstacle_mask,
            gradient_x=grad_x,
            gradient_y=grad_y,
            x=x,
            y=y,
            path_length=path_length,
            path_cost=path_cost,
            n_iterations=n_iter,
            converged=converged,
        )

    def export_state(self) -> PhysicsState:
        """Run solver and export as PhysicsState."""
        result = self.solve()
        state = PhysicsState(solver_name="field_aware_routing_2d")
        state.set_field("cost_field", result.cost_field, "1")
        state.set_field("gradient_x", result.gradient_x, "1/m")
        state.set_field("gradient_y", result.gradient_y, "1/m")
        state.metadata["path_length"] = result.path_length
        state.metadata["path_cost"] = result.path_cost
        state.metadata["converged"] = result.converged
        state.metadata["n_iterations"] = result.n_iterations
        return state
