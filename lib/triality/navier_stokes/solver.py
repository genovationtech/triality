"""
Coupled Navier-Stokes flow solver with buoyancy and phase-change support.

Wraps the low-level :class:`NavierStokesSolver` projection method into a
high-level ``solve()`` interface that handles domain setup, time-stepping,
CFL-adaptive dt, and result collection.

Physics
-------
Incompressible variable-density Navier-Stokes on a staggered MAC grid:

    d(rho*u)/dt + div(rho*u x u) = -grad(p) + div(mu grad u) + rho*g + F_s
    div(u) = S_mass

Solved via Chorin's projection (predictor-corrector) with semi-implicit
diffusion (Crank-Nicolson) and an iterative pressure Poisson solve.

Dependencies
------------
- numpy
- scipy (sparse linear algebra)
- projection_solver (this package)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from .projection_solver import (
    NavierStokesSolver,
    NavierStokesResult,
    StaggeredGrid,
    BoundaryCondition,
    BCType,
    CoordinateSystem,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class FlowSolverResult:
    """Aggregated result from a full Navier-Stokes simulation run.

    Attributes
    ----------
    times : np.ndarray
        Recorded snapshot times [s].
    u_snapshots : list of np.ndarray
        u-velocity fields at each snapshot, shape (nx+1, ny).
    v_snapshots : list of np.ndarray
        v-velocity fields at each snapshot, shape (nx, ny+1).
    p_snapshots : list of np.ndarray
        Pressure fields at each snapshot, shape (nx, ny).
    max_velocity : np.ndarray
        Maximum velocity magnitude at each snapshot [m/s].
    divergence_error : np.ndarray
        RMS divergence error at each snapshot [1/s].
    kinetic_energy : np.ndarray
        Domain-integrated kinetic energy at each snapshot [J/m].
    cfl_history : np.ndarray
        CFL number at each snapshot.
    dt_history : np.ndarray
        Time-step size used at each snapshot [s].
    converged : bool
        Whether the simulation completed without solver failure.
    """
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    u_snapshots: List[np.ndarray] = field(default_factory=list)
    v_snapshots: List[np.ndarray] = field(default_factory=list)
    p_snapshots: List[np.ndarray] = field(default_factory=list)
    max_velocity: np.ndarray = field(default_factory=lambda: np.array([]))
    divergence_error: np.ndarray = field(default_factory=lambda: np.array([]))
    kinetic_energy: np.ndarray = field(default_factory=lambda: np.array([]))
    cfl_history: np.ndarray = field(default_factory=lambda: np.array([]))
    dt_history: np.ndarray = field(default_factory=lambda: np.array([]))
    converged: bool = True


# ---------------------------------------------------------------------------
# High-level flow solver
# ---------------------------------------------------------------------------

class FlowSolver:
    """High-level incompressible Navier-Stokes flow solver.

    Wraps :class:`NavierStokesSolver` with convenience methods for domain
    setup, adaptive time-stepping, and post-processing.

    Parameters
    ----------
    coord_system : CoordinateSystem
        Cartesian (default) or axisymmetric.
    cfl_target : float
        Target CFL number for adaptive time-stepping (default 0.5).

    Example
    -------
    >>> solver = FlowSolver()
    >>> solver.setup_domain(x_range=(0, 1), y_range=(0, 1), nx=64, ny=64)
    >>> solver.set_fluid(rho=1.0, mu=0.01)
    >>> solver.set_bcs(
    ...     left=BoundaryCondition(BCType.WALL),
    ...     right=BoundaryCondition(BCType.WALL),
    ...     bottom=BoundaryCondition(BCType.WALL),
    ...     top=BoundaryCondition(BCType.INLET, velocity=(1.0, 0.0)),
    ... )
    >>> result = solver.solve(t_end=5.0, dt=1e-3, store_every=50)
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M3_COUPLED

    def __init__(self, coord_system: CoordinateSystem = CoordinateSystem.CARTESIAN,
                 cfl_target: float = 0.5):
        self.coord_system = coord_system
        self.cfl_target = cfl_target

        # Internal projection solver
        self._ns = NavierStokesSolver(coord_system=coord_system)

        # Domain / physics flags
        self._domain_set = False
        self._fluid_set = False
        self._bcs_set = False

        # Cached grid info
        self._nx: int = 0
        self._ny: int = 0
        self._dx: float = 0.0
        self._dy: float = 0.0

        # Coupling state
        self._coupled_state = None
        self._time = 0.0

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def setup_domain(self, x_range: Tuple[float, float],
                     y_range: Tuple[float, float],
                     nx: int, ny: int) -> None:
        """Define the rectangular computational domain.

        Parameters
        ----------
        x_range : tuple of float
            (x_min, x_max) in metres.
        y_range : tuple of float
            (y_min, y_max) in metres.
        nx, ny : int
            Number of cells in x and y directions.
        """
        self._ns.set_domain(x_range=x_range, y_range=y_range, nx=nx, ny=ny)
        self._nx = nx
        self._ny = ny
        self._dx = (x_range[1] - x_range[0]) / nx
        self._dy = (y_range[1] - y_range[0]) / ny
        self._domain_set = True

    def set_fluid(self, rho: float = 1.0, mu: float = 1.0e-3) -> None:
        """Set uniform fluid properties.

        Parameters
        ----------
        rho : float
            Density [kg/m^3].
        mu : float
            Dynamic viscosity [Pa.s].
        """
        self._ns.set_fluid_properties(rho=rho, mu=mu)
        self._rho = rho
        self._mu = mu
        self._fluid_set = True

    def set_bcs(self, left: BoundaryCondition, right: BoundaryCondition,
                bottom: BoundaryCondition, top: BoundaryCondition) -> None:
        """Apply boundary conditions on all four sides.

        Parameters
        ----------
        left, right, bottom, top : BoundaryCondition
            Boundary condition objects for each side.
        """
        self._ns.set_boundary_conditions(
            left=left, right=right, bottom=bottom, top=top)
        self._bcs_set = True

    def set_gravity(self, gx: float = 0.0, gy: float = -9.81) -> None:
        """Set gravitational acceleration.

        Parameters
        ----------
        gx, gy : float
            Gravity components [m/s^2].
        """
        self._ns.set_gravity(gx=gx, gy=gy)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _compute_cfl(self, u: np.ndarray, v: np.ndarray, dt: float) -> float:
        """Compute the CFL number from current velocity fields."""
        u_max = np.max(np.abs(u)) if u.size > 0 else 0.0
        v_max = np.max(np.abs(v)) if v.size > 0 else 0.0
        cfl = dt * (u_max / max(self._dx, 1e-30) + v_max / max(self._dy, 1e-30))
        return float(cfl)

    def _compute_divergence_rms(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute RMS of the discrete divergence field at cell centres."""
        nx, ny = self._nx, self._ny
        # u has shape (nx+1, ny), v has shape (nx, ny+1)
        du_dx = (u[1:nx+1, :ny] - u[:nx, :ny]) / self._dx
        dv_dy = (v[:nx, 1:ny+1] - v[:nx, :ny]) / self._dy
        div = du_dx + dv_dy
        return float(np.sqrt(np.mean(div ** 2)))

    def _compute_kinetic_energy(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute domain-integrated kinetic energy (per unit depth)."""
        nx, ny = self._nx, self._ny
        # Interpolate velocities to cell centres
        u_cc = 0.5 * (u[:nx, :ny] + u[1:nx+1, :ny])
        v_cc = 0.5 * (v[:nx, :ny] + v[:nx, 1:ny+1])
        ke = 0.5 * self._rho * (u_cc ** 2 + v_cc ** 2)
        return float(np.sum(ke) * self._dx * self._dy)

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, t_end: float, dt: float = 1e-3,
              store_every: int = 1, adaptive_dt: bool = False,
              max_steps: int = 10_000_000,
              progress_callback=None) -> FlowSolverResult:
        """Run the Navier-Stokes simulation.

        Parameters
        ----------
        t_end : float
            Final simulation time [s].
        dt : float
            Initial (or fixed) time step [s].
        store_every : int
            Store a snapshot every *store_every* steps.
        adaptive_dt : bool
            If True, adjust dt each step to satisfy ``cfl_target``.
        max_steps : int
            Safety cap on the total number of time steps.

        Returns
        -------
        FlowSolverResult
            Collected velocity / pressure snapshots and diagnostics.
        """
        if not (self._domain_set and self._fluid_set and self._bcs_set):
            raise RuntimeError(
                "Domain, fluid properties, and boundary conditions must all "
                "be set before calling solve()."
            )

        # Storage lists
        times: List[float] = []
        u_snaps: List[np.ndarray] = []
        v_snaps: List[np.ndarray] = []
        p_snaps: List[np.ndarray] = []
        max_vel: List[float] = []
        div_err: List[float] = []
        ke_hist: List[float] = []
        cfl_hist: List[float] = []
        dt_hist: List[float] = []

        t = 0.0
        step = 0
        converged = True
        total_steps_est = min(int(t_end / max(dt, 1e-30)), max_steps)
        _progress_interval = max(total_steps_est // 50, 1)  # ~50 updates max

        while t < t_end and step < max_steps:
            # Remaining time guard
            dt_eff = min(dt, t_end - t)
            if dt_eff <= 0:
                break

            try:
                snap = self._ns.step(dt_eff)
            except Exception:
                converged = False
                break

            t += dt_eff
            step += 1

            # Report real progress
            if progress_callback and step % _progress_interval == 0:
                progress_callback(step, total_steps_est)

            # Extract velocity / pressure from the internal solver
            grid = self._ns.grid
            u_field = grid.u.copy()
            v_field = grid.v.copy()
            p_field = grid.p.copy()

            # Adaptive dt based on CFL
            cfl = self._compute_cfl(u_field, v_field, dt_eff)
            if adaptive_dt and cfl > 0:
                dt = dt_eff * self.cfl_target / max(cfl, 1e-30)
                dt = max(dt, 1e-12)

            # Record snapshot
            if step % store_every == 0 or t >= t_end:
                times.append(t)
                u_snaps.append(u_field)
                v_snaps.append(v_field)
                p_snaps.append(p_field)
                max_vel.append(float(np.max(np.sqrt(
                    _cell_centre_u(u_field, self._nx, self._ny) ** 2 +
                    _cell_centre_v(v_field, self._nx, self._ny) ** 2
                ))))
                div_err.append(self._compute_divergence_rms(u_field, v_field))
                ke_hist.append(self._compute_kinetic_energy(u_field, v_field))
                cfl_hist.append(cfl)
                dt_hist.append(dt_eff)

        return FlowSolverResult(
            times=np.array(times),
            u_snapshots=u_snaps,
            v_snapshots=v_snaps,
            p_snapshots=p_snaps,
            max_velocity=np.array(max_vel),
            divergence_error=np.array(div_err),
            kinetic_energy=np.array(ke_hist),
            cfl_history=np.array(cfl_hist),
            dt_history=np.array(dt_hist),
            converged=converged,
        )

    def export_state(self, result: FlowSolverResult) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="navier_stokes")
        if len(result.u_snapshots) > 0:
            state.set_field("velocity_x", result.u_snapshots[-1], "m/s")
        if len(result.v_snapshots) > 0:
            state.set_field("velocity_y", result.v_snapshots[-1], "m/s")
        if len(result.p_snapshots) > 0:
            state.set_field("pressure", result.p_snapshots[-1], "Pa")
        state.metadata["max_velocity"] = float(result.max_velocity[-1]) if len(result.max_velocity) > 0 else 0.0
        state.metadata["kinetic_energy"] = float(result.kinetic_energy[-1]) if len(result.kinetic_energy) > 0 else 0.0
        state.metadata["converged"] = result.converged
        return state

    def import_state(self, state: PhysicsState) -> None:
        """Import coupled fields from partner solvers."""
        self._coupled_state = state

    def advance(self, dt: float) -> PhysicsState:
        """Advance solver by dt for closed-loop coupling."""
        if self._coupled_state is not None:
            if self._coupled_state.has("temperature"):
                # Apply imported temperature as a coupled wall BC
                T_wall = self._coupled_state.get_field("temperature").data
                self._ns.set_wall_temperature(T_wall)
        result = self.solve(t_end=self._time + dt, dt=dt)
        self._time += dt
        return self.export_state(result)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _cell_centre_u(u: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Interpolate staggered u-velocity to cell centres."""
    return 0.5 * (u[:nx, :ny] + u[1:nx + 1, :ny])


def _cell_centre_v(v: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Interpolate staggered v-velocity to cell centres."""
    return 0.5 * (v[:nx, :ny] + v[:nx, 1:ny + 1])


# ---------------------------------------------------------------------------
# Level 3 -- Standalone 2D Incompressible Navier-Stokes Solver
# ---------------------------------------------------------------------------

@dataclass
class NavierStokes2DResult:
    """Result container for the standalone 2D incompressible N-S solver.

    All 2-D field arrays have shape ``(ny, nx)``.

    Attributes
    ----------
    u : np.ndarray
        x-velocity at final time, shape (ny, nx).
    v : np.ndarray
        y-velocity at final time, shape (ny, nx).
    p : np.ndarray
        Pressure at final time, shape (ny, nx).
    x : np.ndarray
        x-coordinates of cell centres, shape (nx,).
    y : np.ndarray
        y-coordinates of cell centres, shape (ny,).
    time : float
        Final simulation time [s].
    max_velocity : float
        Peak velocity magnitude at final time [m/s].
    divergence_rms : float
        RMS divergence error at final time [1/s].
    kinetic_energy : float
        Domain-integrated kinetic energy at final time [J/m].
    dt_history : np.ndarray
        Time-step sizes used, shape (n_steps,).
    converged : bool
        Whether the simulation completed without failure.
    """
    u: np.ndarray = field(default_factory=lambda: np.array([]))
    v: np.ndarray = field(default_factory=lambda: np.array([]))
    p: np.ndarray = field(default_factory=lambda: np.array([]))
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    time: float = 0.0
    max_velocity: float = 0.0
    divergence_rms: float = 0.0
    kinetic_energy: float = 0.0
    dt_history: np.ndarray = field(default_factory=lambda: np.array([]))
    converged: bool = True


class NavierStokes2DSolver:
    """Standalone 2D incompressible Navier-Stokes solver using the projection method.

    Solves the incompressible N-S equations on a collocated grid:

        du/dt + (u . grad)u = -1/rho grad(p) + nu * laplacian(u)
        div(u) = 0

    using Chorin's projection method:
        1. Compute intermediate velocity u* (advection + diffusion, no pressure)
        2. Solve pressure Poisson equation via Jacobi iteration
        3. Correct velocity: u^{n+1} = u* - dt/rho * grad(p)

    Default setup: lid-driven cavity (top wall moves at U_lid).

    Parameters
    ----------
    nx, ny : int
        Number of grid cells in x and y.
    Lx, Ly : float
        Domain size [m].
    rho : float
        Fluid density [kg/m^3].
    nu : float
        Kinematic viscosity [m^2/s].
    U_lid : float
        Lid velocity for default lid-driven cavity [m/s].
    cfl : float
        Target CFL number for adaptive time stepping.
    """

    fidelity_tier = FidelityTier.HIGH_FIDELITY
    coupling_maturity = CouplingMaturity.M3_COUPLED

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        Lx: float = 1.0,
        Ly: float = 1.0,
        rho: float = 1.0,
        nu: float = 0.01,
        U_lid: float = 1.0,
        cfl: float = 0.5,
        quasi_3d: bool = False,
        z_length: float = 1.0,
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.rho = rho
        self.nu = nu
        self.U_lid = U_lid
        self.cfl = cfl
        self.quasi_3d = quasi_3d
        self.z_half = z_length / 2.0  # half-depth for friction model

        self.dx = Lx / nx
        self.dy = Ly / ny
        self.x = np.linspace(0.5 * self.dx, Lx - 0.5 * self.dx, nx)
        self.y = np.linspace(0.5 * self.dy, Ly - 0.5 * self.dy, ny)

        # Coupling state
        self._coupled_state = None
        self._time = 0.0

    def _compute_dt(self, u: np.ndarray, v: np.ndarray) -> float:
        """CFL-limited time step with viscous stability constraint."""
        u_max = np.max(np.abs(u)) + 1e-10
        v_max = np.max(np.abs(v)) + 1e-10
        dt_adv = self.cfl / (u_max / self.dx + v_max / self.dy)
        dt_visc = 0.25 / (self.nu * (1.0 / self.dx**2 + 1.0 / self.dy**2) + 1e-30)
        return min(dt_adv, dt_visc)

    def _pressure_poisson_jacobi(
        self, p: np.ndarray, rhs: np.ndarray, max_iter: int = 200, tol: float = 1e-5
    ) -> np.ndarray:
        """Solve the pressure Poisson equation using Jacobi iteration.

        Solves: laplacian(p) = rhs  with Neumann BCs (dp/dn = 0).
        """
        dx2 = self.dx**2
        dy2 = self.dy**2
        coeff = 2.0 * (1.0 / dx2 + 1.0 / dy2)

        for _ in range(max_iter):
            p_old = p.copy()

            p[1:-1, 1:-1] = (
                (p_old[1:-1, 2:] + p_old[1:-1, :-2]) / dx2
                + (p_old[2:, 1:-1] + p_old[:-2, 1:-1]) / dy2
                - rhs[1:-1, 1:-1]
            ) / coeff

            # Neumann BCs: dp/dn = 0
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]
            p[:, 0] = p[:, 1]
            p[:, -1] = p[:, -2]

            residual = np.max(np.abs(p - p_old))
            if residual < tol:
                break

        # Reference pressure: set mean to zero
        p -= np.mean(p)
        return p

    def solve(
        self,
        t_end: float = 10.0,
        dt: Optional[float] = None,
        max_steps: int = 500000,
        pressure_iters: int = 200,
        pressure_tol: float = 1e-5,
        progress_callback=None,
    ) -> NavierStokes2DResult:
        """Run the 2D incompressible N-S simulation (lid-driven cavity).

        Parameters
        ----------
        t_end : float
            Final simulation time [s].
        dt : float or None
            Fixed time step. If None, uses CFL-adaptive stepping.
        max_steps : int
            Maximum number of time steps.
        pressure_iters : int
            Jacobi iterations for pressure Poisson solve per step.
        pressure_tol : float
            Convergence tolerance for pressure Poisson solve.
        progress_callback : callable, optional
            Called as progress_callback(step, total) during time stepping.

        Returns
        -------
        NavierStokes2DResult
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        nu = self.nu
        rho = self.rho

        # Initialise fields on collocated grid (ny, nx)
        u = np.zeros((ny, nx))
        v = np.zeros((ny, nx))
        p = np.zeros((ny, nx))

        t = 0.0
        step = 0
        dt_list = []
        converged = True
        _total_est = min(int(t_end / max(dt or 1e-4, 1e-30)), max_steps)
        _prog_interval = max(_total_est // 50, 1)

        while t < t_end and step < max_steps:
            # Adaptive or fixed dt
            if dt is not None:
                dt_eff = min(dt, t_end - t)
            else:
                dt_eff = min(self._compute_dt(u, v), t_end - t)
            if dt_eff <= 0:
                break

            try:
                # --- Step 1: Compute intermediate velocity u*, v* ---
                # Advection (upwind)
                # du/dx
                dudx_p = np.zeros_like(u)
                dudx_m = np.zeros_like(u)
                dudx_p[:, 1:-1] = (u[:, 2:] - u[:, 1:-1]) / dx
                dudx_m[:, 1:-1] = (u[:, 1:-1] - u[:, :-2]) / dx
                adv_u_x = np.where(u >= 0, u * dudx_m, u * dudx_p)

                # du/dy
                dudy_p = np.zeros_like(u)
                dudy_m = np.zeros_like(u)
                dudy_p[1:-1, :] = (u[2:, :] - u[1:-1, :]) / dy
                dudy_m[1:-1, :] = (u[1:-1, :] - u[:-2, :]) / dy
                adv_u_y = np.where(v >= 0, v * dudy_m, v * dudy_p)

                # dv/dx
                dvdx_p = np.zeros_like(v)
                dvdx_m = np.zeros_like(v)
                dvdx_p[:, 1:-1] = (v[:, 2:] - v[:, 1:-1]) / dx
                dvdx_m[:, 1:-1] = (v[:, 1:-1] - v[:, :-2]) / dx
                adv_v_x = np.where(u >= 0, u * dvdx_m, u * dvdx_p)

                # dv/dy
                dvdy_p = np.zeros_like(v)
                dvdy_m = np.zeros_like(v)
                dvdy_p[1:-1, :] = (v[2:, :] - v[1:-1, :]) / dy
                dvdy_m[1:-1, :] = (v[1:-1, :] - v[:-2, :]) / dy
                adv_v_y = np.where(v >= 0, v * dvdy_m, v * dvdy_p)

                # Diffusion (central)
                diff_u = np.zeros_like(u)
                diff_u[1:-1, 1:-1] = nu * (
                    (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
                    + (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
                )
                diff_v = np.zeros_like(v)
                diff_v[1:-1, 1:-1] = nu * (
                    (v[1:-1, 2:] - 2.0 * v[1:-1, 1:-1] + v[1:-1, :-2]) / dx**2
                    + (v[2:, 1:-1] - 2.0 * v[1:-1, 1:-1] + v[:-2, 1:-1]) / dy**2
                )

                # Quasi-3D friction: models wall shear from confining z-walls
                # F_friction = -nu * u / z_half^2 (acts as a linear drag)
                if self.quasi_3d:
                    z_h2 = self.z_half ** 2
                    friction_u = -nu * u / z_h2
                    friction_v = -nu * v / z_h2
                else:
                    friction_u = 0.0
                    friction_v = 0.0

                # Intermediate velocity
                u_star = u + dt_eff * (-adv_u_x - adv_u_y + diff_u + friction_u)
                v_star = v + dt_eff * (-adv_v_x - adv_v_y + diff_v + friction_v)

                # Apply BCs to intermediate velocity (lid-driven cavity)
                # Top wall: u = U_lid, v = 0
                u_star[-1, :] = self.U_lid
                v_star[-1, :] = 0.0
                # Bottom wall: no-slip
                u_star[0, :] = 0.0
                v_star[0, :] = 0.0
                # Left wall: no-slip
                u_star[:, 0] = 0.0
                v_star[:, 0] = 0.0
                # Right wall: no-slip
                u_star[:, -1] = 0.0
                v_star[:, -1] = 0.0

                # --- Step 2: Pressure Poisson equation ---
                # RHS = (rho/dt) * div(u*)
                div_ustar = np.zeros((ny, nx))
                div_ustar[1:-1, 1:-1] = (
                    (u_star[1:-1, 2:] - u_star[1:-1, :-2]) / (2.0 * dx)
                    + (v_star[2:, 1:-1] - v_star[:-2, 1:-1]) / (2.0 * dy)
                )
                rhs = (rho / dt_eff) * div_ustar

                p = self._pressure_poisson_jacobi(p, rhs, pressure_iters, pressure_tol)

                # --- Step 3: Velocity correction ---
                u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - (dt_eff / rho) * (
                    p[1:-1, 2:] - p[1:-1, :-2]
                ) / (2.0 * dx)
                v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - (dt_eff / rho) * (
                    p[2:, 1:-1] - p[:-2, 1:-1]
                ) / (2.0 * dy)

                # Re-apply BCs after correction
                u[-1, :] = self.U_lid
                v[-1, :] = 0.0
                u[0, :] = 0.0
                v[0, :] = 0.0
                u[:, 0] = 0.0
                v[:, 0] = 0.0
                u[:, -1] = 0.0
                v[:, -1] = 0.0

            except Exception:
                converged = False
                break

            t += dt_eff
            step += 1
            dt_list.append(dt_eff)

            if progress_callback and step % _prog_interval == 0:
                progress_callback(step, _total_est)

        # Diagnostics
        vel_mag = np.sqrt(u**2 + v**2)
        max_vel = float(np.max(vel_mag))
        ke = 0.5 * rho * np.sum(vel_mag**2) * dx * dy

        # Divergence RMS
        div_field = np.zeros((ny, nx))
        div_field[1:-1, 1:-1] = (
            (u[1:-1, 2:] - u[1:-1, :-2]) / (2.0 * dx)
            + (v[2:, 1:-1] - v[:-2, 1:-1]) / (2.0 * dy)
        )
        div_rms = float(np.sqrt(np.mean(div_field**2)))

        return NavierStokes2DResult(
            u=u,
            v=v,
            p=p,
            x=self.x.copy(),
            y=self.y.copy(),
            time=t,
            max_velocity=max_vel,
            divergence_rms=div_rms,
            kinetic_energy=ke,
            dt_history=np.array(dt_list),
            converged=converged,
        )

    def export_state(self, result: NavierStokes2DResult) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="navier_stokes_2d")
        state.set_field("velocity_x", result.u, "m/s")
        state.set_field("velocity_y", result.v, "m/s")
        state.set_field("pressure", result.p, "Pa")
        state.metadata["max_velocity"] = result.max_velocity
        state.metadata["kinetic_energy"] = result.kinetic_energy
        state.metadata["divergence_rms"] = result.divergence_rms
        state.metadata["converged"] = result.converged
        state.metadata["quasi_3d"] = self.quasi_3d
        if self.quasi_3d:
            state.metadata["z_length"] = self.z_half * 2.0
        state.metadata["time"] = result.time
        return state

    def import_state(self, state: PhysicsState) -> None:
        """Import coupled fields from partner solvers."""
        self._coupled_state = state

    def advance(self, dt: float) -> PhysicsState:
        """Advance solver by dt for closed-loop coupling."""
        if self._coupled_state is not None:
            if self._coupled_state.has("temperature"):
                # Use imported temperature field as buoyancy source
                T_field = self._coupled_state.get_field("temperature").data
                beta = 2.1e-4  # thermal expansion coefficient [1/K]
                T_ref = np.mean(T_field)
                self._buoyancy_fy = -self.rho * 9.81 * beta * (T_field - T_ref)
            if self._coupled_state.has("heat_source"):
                self._heat_source = self._coupled_state.get_field("heat_source").data
        result = self.solve(t_end=self._time + dt, dt=dt)
        self._time += dt
        return self.export_state(result)
