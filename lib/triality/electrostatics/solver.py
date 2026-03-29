"""
Electrostatics Numerical Solver

Integrates ElectrostaticSolver, ConductiveSolver, and derived-quantity
analysis tools into a unified solver for 2-D electrostatic and conduction
problems.

Solves:
    Laplace:    div(grad V) = 0            (charge-free)
    Poisson:    div(grad V) = -rho / eps   (with charge)
    Conduction: div(sigma grad V) = 0      (current flow)

and computes derived quantities:
    E = -grad V                      (electric field)
    J = sigma E                      (current density)
    P = J^2 / sigma                  (Joule heating)

References:
    Jackson, "Classical Electrodynamics"
    Griffiths, "Introduction to Electrodynamics"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple

from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from .field_solver import (
    ElectrostaticSolver,
    ElectrostaticResult,
    BoundaryCondition,
    BoundaryType,
    ChargeDistribution,
)
from .conduction import (
    ConductiveSolver,
    ConductionResult,
    Material,
)
from .derived_quantities import (
    ElectricField,
    ElectricFieldData,
    FieldMagnitude,
    GradientAnalysis,
    HotspotDetector,
)


@dataclass
class ElectrostaticsResult:
    """Result container for electrostatics simulation.

    Attributes
    ----------
    potential : np.ndarray
        2-D potential field V[i,j] [V].
    grid_x : np.ndarray
        x-coordinates [m].
    grid_y : np.ndarray
        y-coordinates [m].
    E_x : np.ndarray
        x-component of electric field [V/m].
    E_y : np.ndarray
        y-component of electric field [V/m].
    E_magnitude : np.ndarray
        Electric field magnitude [V/m].
    max_field_location : Tuple[float, float, float]
        (x, y, |E|_max) location and value of peak field.
    field_stats : Dict[str, float]
        Field magnitude statistics (min, max, mean, p95, p99).
    high_field_zones : List[Tuple[float, float, float]]
        Points where |E| exceeds the 90th percentile.
    conductivity : Optional[np.ndarray]
        Conductivity field [S/m] (if conduction problem).
    current_density_x : Optional[np.ndarray]
        J_x field [A/m^2] (if conduction problem).
    current_density_y : Optional[np.ndarray]
        J_y field [A/m^2] (if conduction problem).
    power_density : Optional[np.ndarray]
        Joule heating P [W/m^3] (if conduction problem).
    hotspots : Optional[List]
        Hotspot locations (if conduction with threshold).
    mode : str
        Problem type: 'electrostatic' or 'conduction'.
    """
    potential: np.ndarray
    grid_x: np.ndarray
    grid_y: np.ndarray
    E_x: np.ndarray
    E_y: np.ndarray
    E_magnitude: np.ndarray
    max_field_location: Tuple[float, float, float]
    field_stats: Dict[str, float]
    high_field_zones: List[Tuple[float, float, float]]
    conductivity: Optional[np.ndarray] = None
    current_density_x: Optional[np.ndarray] = None
    current_density_y: Optional[np.ndarray] = None
    power_density: Optional[np.ndarray] = None
    hotspots: Optional[List] = None
    mode: str = "electrostatic"


class ElectrostaticsSolver:
    """Unified electrostatics and conduction solver.

    Provides a single interface for both charge-free / Poisson electrostatics
    and steady-state conduction problems. Automatically computes electric
    field, field statistics, and (for conduction) current density and Joule
    heating.

    Parameters
    ----------
    x_range : Tuple[float, float]
        Domain x-extent (x_min, x_max) [m].
    y_range : Tuple[float, float]
        Domain y-extent (y_min, y_max) [m].
    resolution : int
        Grid resolution n (produces n x n grid) (default 50).
    permittivity : float
        Permittivity [F/m] (default eps_0 = 8.854e-12).
    mode : str
        'electrostatic' for Laplace/Poisson, 'conduction' for current flow.
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        x_range: Tuple[float, float] = (0.0, 0.1),
        y_range: Tuple[float, float] = (0.0, 0.1),
        resolution: int = 50,
        permittivity: float = 8.854e-12,
        mode: str = "electrostatic",
    ):
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.permittivity = permittivity
        self.mode = mode

        self._boundaries: List[BoundaryCondition] = []
        self._charges: List[ChargeDistribution] = []
        self._materials: List[Tuple[Material, Optional[Callable]]] = []

    def add_boundary(
        self,
        bc_type: str,
        value: float = 0.0,
        region: Optional[Callable[[float, float], bool]] = None,
    ):
        """Add a boundary condition.

        Parameters
        ----------
        bc_type : str
            'dirichlet', 'grounded', or 'neumann'.
        value : float
            Boundary value [V] (default 0).
        region : callable or None
            Region function (x, y) -> bool.
        """
        type_map = {
            "dirichlet": BoundaryType.DIRICHLET,
            "grounded": BoundaryType.GROUNDED,
            "neumann": BoundaryType.NEUMANN,
            "floating": BoundaryType.FLOATING,
        }
        bt = type_map.get(bc_type.lower(), BoundaryType.DIRICHLET)
        self._boundaries.append(BoundaryCondition(type=bt, value=value, region=region))

    def add_charge(
        self,
        density_func: Callable[[float, float], float],
        region: Optional[Callable[[float, float], bool]] = None,
    ):
        """Add a charge distribution (Poisson source).

        Parameters
        ----------
        density_func : callable
            Charge density function (x, y) -> rho [C/m^3].
        region : callable or None
            Region function (x, y) -> bool.
        """
        self._charges.append(ChargeDistribution(density_func=density_func, region=region))

    def add_material(
        self,
        name: str,
        conductivity: float,
        region: Optional[Callable[[float, float], bool]] = None,
    ):
        """Add a conductive material (only used in 'conduction' mode).

        Parameters
        ----------
        name : str
            Material name.
        conductivity : float
            Conductivity [S/m].
        region : callable or None
            Region function (x, y) -> bool.
        """
        mat = Material(name=name, conductivity=conductivity, region=region)
        self._materials.append((mat, region))

    def solve(self, method: str = "gmres", hotspot_threshold: Optional[float] = None) -> ElectrostaticsResult:
        """Solve the electrostatics or conduction problem.

        Parameters
        ----------
        method : str
            Linear solver: 'gmres', 'bicgstab', or 'direct'.
        hotspot_threshold : float or None
            Power density threshold [W/m^3] for hotspot detection
            (conduction mode only).

        Returns
        -------
        ElectrostaticsResult
            Full solution with derived quantities.
        """
        if self.mode == "conduction":
            return self._solve_conduction(method, hotspot_threshold)
        else:
            return self._solve_electrostatic(method)

    def _solve_electrostatic(self, method: str) -> ElectrostaticsResult:
        """Solve Laplace / Poisson electrostatic problem."""
        solver = ElectrostaticSolver()
        solver.set_domain(self.x_range, self.y_range)
        solver.set_resolution(self.resolution)
        solver.set_permittivity(self.permittivity)

        for bc in self._boundaries:
            solver.add_boundary(bc)
        for cd in self._charges:
            solver.add_charge_distribution(cd)

        es_result = solver.solve(method=method)

        # Derived quantities
        field_data = ElectricField.from_result(es_result)
        stats = FieldMagnitude.analyze(field_data)
        max_loc = field_data.max_field()
        high_zones = GradientAnalysis.high_gradient_zones(field_data, threshold_percentile=90)

        return ElectrostaticsResult(
            potential=es_result.potential,
            grid_x=es_result.grid_x,
            grid_y=es_result.grid_y,
            E_x=field_data.E_x,
            E_y=field_data.E_y,
            E_magnitude=field_data.E_magnitude,
            max_field_location=max_loc,
            field_stats={
                "min": stats.min,
                "max": stats.max,
                "mean": stats.mean,
                "std": stats.std,
                "p95": stats.percentile_95,
                "p99": stats.percentile_99,
            },
            high_field_zones=high_zones,
            mode="electrostatic",
        )

    def _solve_conduction(self, method: str, hotspot_threshold: Optional[float]) -> ElectrostaticsResult:
        """Solve steady-state conduction problem."""
        solver = ConductiveSolver()
        solver.set_domain(self.x_range, self.y_range)
        solver.set_resolution(self.resolution)

        for bc in self._boundaries:
            solver.add_boundary(bc)
        for mat, region in self._materials:
            solver.add_material(mat, region=region)

        cond_result = solver.solve(method=method)

        # Compute electric field from potential
        n = self.resolution
        dx = cond_result.grid_x[1] - cond_result.grid_x[0]
        dy = cond_result.grid_y[1] - cond_result.grid_y[0]

        dV_dx = np.zeros((n, n))
        dV_dy = np.zeros((n, n))
        dV_dx[1:-1, :] = (cond_result.potential[2:, :] - cond_result.potential[:-2, :]) / (2 * dx)
        dV_dx[0, :] = (cond_result.potential[1, :] - cond_result.potential[0, :]) / dx
        dV_dx[-1, :] = (cond_result.potential[-1, :] - cond_result.potential[-2, :]) / dx
        dV_dy[:, 1:-1] = (cond_result.potential[:, 2:] - cond_result.potential[:, :-2]) / (2 * dy)
        dV_dy[:, 0] = (cond_result.potential[:, 1] - cond_result.potential[:, 0]) / dy
        dV_dy[:, -1] = (cond_result.potential[:, -1] - cond_result.potential[:, -2]) / dy

        E_x = -dV_dx
        E_y = -dV_dy
        E_mag = np.sqrt(E_x**2 + E_y**2)

        # Current density J = sigma * E
        sigma = cond_result.conductivity
        J_x = sigma * E_x
        J_y = sigma * E_y

        # Power density P = J^2 / sigma
        J_mag2 = J_x**2 + J_y**2
        P = np.where(sigma > 1e-20, J_mag2 / sigma, 0.0)

        # Field statistics
        e_max_idx = np.unravel_index(np.argmax(E_mag), E_mag.shape)
        max_loc = (
            cond_result.grid_x[e_max_idx[0]],
            cond_result.grid_y[e_max_idx[1]],
            float(E_mag[e_max_idx]),
        )

        # Hotspots
        hotspots = None
        if hotspot_threshold is not None:
            hotspots = cond_result.find_hotspots(hotspot_threshold)

        return ElectrostaticsResult(
            potential=cond_result.potential,
            grid_x=cond_result.grid_x,
            grid_y=cond_result.grid_y,
            E_x=E_x,
            E_y=E_y,
            E_magnitude=E_mag,
            max_field_location=max_loc,
            field_stats={
                "min": float(np.min(E_mag)),
                "max": float(np.max(E_mag)),
                "mean": float(np.mean(E_mag)),
                "std": float(np.std(E_mag)),
                "p95": float(np.percentile(E_mag, 95)),
                "p99": float(np.percentile(E_mag, 99)),
            },
            high_field_zones=[],
            conductivity=sigma,
            current_density_x=J_x,
            current_density_y=J_y,
            power_density=P,
            hotspots=hotspots,
            mode="conduction",
        )

    def export_state(self, result: ElectrostaticsResult) -> PhysicsState:
        """Export result as a canonically labeled electrostatic PhysicsState."""
        state = PhysicsState(solver_name="electrostatics")
        state.set_field("electric_potential", result.potential, "V")
        state.set_field("electric_field_x", result.E_x, "V/m")
        state.set_field("electric_field_y", result.E_y, "V/m")
        state.set_field("electric_field", result.E_magnitude, "V/m")

        if result.current_density_x is not None:
            state.set_field("current_density_x", result.current_density_x, "A/m^2")
        if result.current_density_y is not None:
            state.set_field("current_density_y", result.current_density_y, "A/m^2")
        if result.power_density is not None:
            state.set_field("heat_source", result.power_density, "W/m^3")

        state.metadata["max_field_value"] = result.max_field_location[2]
        state.metadata["max_field_x"] = result.max_field_location[0]
        state.metadata["max_field_y"] = result.max_field_location[1]
        state.metadata["max_field_value_V_per_m"] = result.max_field_location[2]
        state.metadata["max_field_x_m"] = result.max_field_location[0]
        state.metadata["max_field_y_m"] = result.max_field_location[1]
        state.metadata["field_stats"] = result.field_stats
        state.metadata["high_field_zone_count"] = len(result.high_field_zones)
        state.metadata["mode"] = result.mode
        if result.hotspots is not None:
            state.metadata["hotspot_count"] = len(result.hotspots)
        return state


# ========================================================================
# Level 3 2-D: Electrostatic Poisson solver
# ========================================================================

@dataclass
class Electrostatics2DResult:
    """Result container for the 2-D electrostatic Poisson solver.

    Attributes
    ----------
    potential : np.ndarray
        Electrostatic potential V(y, x), shape (ny, nx) [V].
    E_x : np.ndarray
        x-component of electric field, shape (ny, nx) [V/m].
    E_y : np.ndarray
        y-component of electric field, shape (ny, nx) [V/m].
    E_magnitude : np.ndarray
        Electric field magnitude, shape (ny, nx) [V/m].
    charge_density : np.ndarray
        Source charge distribution, shape (ny, nx) [C/m^3].
    x : np.ndarray
        x-coordinates (nx,) [m].
    y : np.ndarray
        y-coordinates (ny,) [m].
    peak_potential : float
        Maximum absolute potential [V].
    peak_field : float
        Maximum electric field magnitude [V/m].
    total_energy : float
        Integrated electrostatic energy 0.5*eps*|E|^2 [J/m].
    n_iterations : int
        Number of SOR iterations to convergence.
    converged : bool
        Whether the solver converged within tolerance.
    """
    potential: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    E_x: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    E_y: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    E_magnitude: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    charge_density: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    peak_potential: float = 0.0
    peak_field: float = 0.0
    total_energy: float = 0.0
    n_iterations: int = 0
    converged: bool = False


class Electrostatics2DSolver:
    """2-D electrostatic Poisson solver using SOR relaxation.

    Solves the Poisson equation on a rectangular domain:

        lap(V) = -rho / eps

    with Dirichlet boundary conditions (V = 0 on all edges by default).
    The electric field is recovered as E = -grad(V).

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    Lx, Ly : float
        Domain size [m].
    permittivity : float
        Permittivity [F/m] (default vacuum).
    bc_left, bc_right, bc_top, bc_bottom : float
        Dirichlet boundary potential [V] on each edge.
    charge_func : callable or None
        Source charge density function (x, y) -> rho [C/m^3].
        If None, a default point-charge-like Gaussian is placed at the centre.
    max_iterations : int
        Maximum SOR iterations.
    tolerance : float
        Convergence tolerance on max residual.
    omega : float
        SOR over-relaxation parameter (1 < omega < 2).
    """

    fidelity_tier = FidelityTier.HIGH_FIDELITY
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        nx: int = 80,
        ny: int = 80,
        Lx: float = 1.0,
        Ly: float = 1.0,
        permittivity: float = 8.854e-12,
        bc_left: float = 0.0,
        bc_right: float = 0.0,
        bc_top: float = 0.0,
        bc_bottom: float = 0.0,
        charge_func: Optional[Callable] = None,
        max_iterations: int = 5000,
        tolerance: float = 1e-6,
        omega: float = 1.7,
        quasi_3d: bool = False,
        z_length: float = 1.0,
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.permittivity = permittivity
        self.bc_left = bc_left
        self.bc_right = bc_right
        self.bc_top = bc_top
        self.bc_bottom = bc_bottom
        self.charge_func = charge_func
        self.quasi_3d = quasi_3d
        self.z_half = z_length / 2.0
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.omega = omega

    def solve(self) -> Electrostatics2DResult:
        """Solve the 2-D Poisson equation for electrostatic potential.

        Returns
        -------
        Electrostatics2DResult
        """
        nx, ny = self.nx, self.ny
        dx = self.Lx / (nx - 1)
        dy = self.Ly / (ny - 1)
        x = np.linspace(0.0, self.Lx, nx)
        y = np.linspace(0.0, self.Ly, ny)
        X, Y = np.meshgrid(x, y)  # (ny, nx)

        eps = self.permittivity

        # Build charge density field
        rho = np.zeros((ny, nx))
        if self.charge_func is not None:
            for j in range(ny):
                for i in range(nx):
                    rho[j, i] = self.charge_func(x[i], y[j])
        else:
            # Default: Gaussian charge blob at centre
            cx, cy = 0.5 * self.Lx, 0.5 * self.Ly
            sigma_q = 0.05 * min(self.Lx, self.Ly)
            rho = 1.0 * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma_q ** 2))

        # Initialise potential with boundary conditions
        V = np.zeros((ny, nx))
        V[:, 0] = self.bc_left
        V[:, -1] = self.bc_right
        V[0, :] = self.bc_bottom
        V[-1, :] = self.bc_top

        # Precompute SOR coefficients
        dx2 = dx**2
        dy2 = dy**2
        denom = 2.0 * (1.0 / dx2 + 1.0 / dy2)
        # Quasi-3D: z-direction field leakage adds a loss term to the Laplacian
        # ∇²V ≈ d²V/dx² + d²V/dy² - V/z_half² = -ρ/ε
        # The -V/z_half² models potential decay through z-thickness
        if self.quasi_3d:
            dz_h2 = self.z_half ** 2
            denom += 1.0 / dz_h2
        omega = self.omega

        converged = False
        n_iter = 0

        for iteration in range(self.max_iterations):
            max_residual = 0.0
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    lap = ((V[j, i + 1] + V[j, i - 1]) / dx2
                           + (V[j + 1, i] + V[j - 1, i]) / dy2)
                    V_gs = (lap + rho[j, i] / eps) / denom
                    residual = V_gs - V[j, i]
                    V[j, i] += omega * residual
                    max_residual = max(max_residual, abs(residual))

            n_iter = iteration + 1
            if max_residual < self.tolerance:
                converged = True
                break

        # Electric field: E = -grad(V)
        E_x = np.zeros((ny, nx))
        E_y = np.zeros((ny, nx))
        E_x[:, 1:-1] = -(V[:, 2:] - V[:, :-2]) / (2.0 * dx)
        E_x[:, 0] = -(V[:, 1] - V[:, 0]) / dx
        E_x[:, -1] = -(V[:, -1] - V[:, -2]) / dx
        E_y[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2.0 * dy)
        E_y[0, :] = -(V[1, :] - V[0, :]) / dy
        E_y[-1, :] = -(V[-1, :] - V[-2, :]) / dy
        E_mag = np.sqrt(E_x**2 + E_y**2)

        # Energy: integral of 0.5 * eps * |E|^2
        total_energy = float(0.5 * eps * np.sum(E_mag**2) * dx * dy)

        return Electrostatics2DResult(
            potential=V,
            E_x=E_x,
            E_y=E_y,
            E_magnitude=E_mag,
            charge_density=rho,
            x=x,
            y=y,
            peak_potential=float(np.max(np.abs(V))),
            peak_field=float(np.max(E_mag)),
            total_energy=total_energy,
            n_iterations=n_iter,
            converged=converged,
        )

    def export_state(self) -> PhysicsState:
        """Run solver and export as PhysicsState."""
        result = self.solve()
        state = PhysicsState(solver_name="electrostatics_2d")
        state.set_field("potential", result.potential, "V")
        state.set_field("E_x", result.E_x, "V/m")
        state.set_field("E_y", result.E_y, "V/m")
        state.set_field("E_magnitude", result.E_magnitude, "V/m")
        state.metadata["peak_potential"] = result.peak_potential
        state.metadata["peak_field"] = result.peak_field
        state.metadata["total_energy"] = result.total_energy
        state.metadata["converged"] = result.converged
        state.metadata["n_iterations"] = result.n_iterations
        state.metadata["quasi_3d"] = self.quasi_3d
        if self.quasi_3d:
            state.metadata["z_length"] = self.z_half * 2.0
        return state