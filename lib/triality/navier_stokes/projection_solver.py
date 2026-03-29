"""
Incompressible Navier-Stokes solver using the projection method (Chorin's splitting).

Supports variable-density two-phase flows for nucleate boiling simulations
on staggered (MAC) grids in 2D Cartesian and 2D axisymmetric coordinates.

Physics Basis
=============
Governing Equations (variable-density incompressible Navier-Stokes):

    Momentum:
        d(rho*u)/dt + div(rho*u (x) u) = -grad(p) + div(mu*(grad(u) + grad(u)^T)) + rho*g + F_surface

    Continuity (with phase-change mass source):
        div(u) = S_mass

    where:
        rho   = density field [kg/m^3]  (varies across liquid-vapor interface)
        mu    = dynamic viscosity [Pa.s] (varies across liquid-vapor interface)
        u     = velocity vector (u, v) [m/s]
        p     = pressure [Pa]
        g     = gravitational acceleration vector [m/s^2]
        F_surface = surface tension body force (e.g., CSF model) [N/m^3]
        S_mass = mass source / divergence source from phase change [1/s]

Projection Method (Chorin's Splitting)
======================================
The time integration is split into three sub-steps each time step:

    Step 1 - Predictor (advection + diffusion + body forces):
        Advection: Forward Euler (explicit)
        Diffusion: Crank-Nicolson (semi-implicit, theta=0.5)

        rho^n * (u* - u^n)/dt = -[div(rho*u (x) u)]^n
                                 + theta * div(mu * (grad(u*) + grad(u*^T)))
                                 + (1-theta) * div(mu * (grad(u^n) + grad(u^n^T)))
                                 + rho^n * g + F_body

    Step 2 - Pressure Poisson equation:
        div( (1/rho^n) * grad(p^{n+1}) ) = (div(u*) - S_mass) / dt

    Step 3 - Correction:
        u^{n+1} = u* - (dt / rho^n) * grad(p^{n+1})

Spatial Discretization
======================
Staggered grid (Marker-and-Cell / MAC method):
    - Pressure p(i,j): cell centers
    - u-velocity u(i,j): vertical cell faces (i+1/2, j)
    - v-velocity v(i,j): horizontal cell faces (i, j+1/2)

This arrangement naturally avoids the checkerboard pressure instability
and enforces discrete incompressibility at cell centers.

Coordinate Systems
==================
- Cartesian 2D: standard (x, y)
- Axisymmetric 2D: (r, z) where r is the radial coordinate.
  Extra 1/r terms appear in the divergence and Laplacian operators.

Boundary Conditions
===================
- Wall (no-slip): u = 0, v = 0 at wall; dp/dn = 0
- Inlet: prescribed velocity; dp/dn = 0
- Outlet: du/dn = 0; p = p_out (or dp/dn = 0 for zero-gradient)
- Symmetry (axis): u_normal = 0, d(u_tangential)/dn = 0; dp/dn = 0

Dependencies
============
- numpy: array operations
- scipy: sparse matrix construction and linear solvers (pressure Poisson)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Callable, Union
from enum import Enum

import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CoordinateSystem(Enum):
    """Coordinate system for the simulation domain."""
    CARTESIAN = "cartesian"
    AXISYMMETRIC = "axisymmetric"


class BCType(Enum):
    """Boundary condition types for the Navier-Stokes solver."""
    WALL = "wall"             # No-slip wall: u = 0
    INLET = "inlet"           # Prescribed velocity
    OUTLET = "outlet"         # Zero-gradient velocity, fixed or zero-gradient pressure
    SYMMETRY = "symmetry"     # Symmetry / axis of symmetry


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BoundaryCondition:
    """Specification for a single boundary.

    Attributes:
        bc_type: Type of boundary condition.
        velocity: Prescribed velocity (u, v) for INLET type.  Ignored otherwise.
        pressure: Prescribed pressure for OUTLET type.  ``None`` means
            zero-gradient pressure at the outlet.
    """
    bc_type: BCType
    velocity: Tuple[float, float] = (0.0, 0.0)
    pressure: Optional[float] = None


@dataclass
class StaggeredGrid:
    """Marker-and-Cell (MAC) staggered grid for 2D incompressible flow.

    Layout (for an nx-by-ny grid of cells):

        u-velocity on vertical faces:  shape (nx+1, ny)
            u[i, j] lives at (xf[i], yc[j])

        v-velocity on horizontal faces: shape (nx, ny+1)
            v[i, j] lives at (xc[i], yf[j])

        p, rho, mu at cell centers:    shape (nx, ny)
            p[i, j] lives at (xc[i], yc[j])

    For axisymmetric problems, x -> r (radial) and y -> z (axial).

    Attributes:
        x_range: (x_min, x_max) domain extent in x.
        y_range: (y_min, y_max) domain extent in y.
        nx: Number of cells in x.
        ny: Number of cells in y.
        dx: Cell width in x.
        dy: Cell width in y.
        xc: 1-D array of cell-center x coordinates, shape (nx,).
        yc: 1-D array of cell-center y coordinates, shape (ny,).
        xf: 1-D array of face x coordinates (u-locations), shape (nx+1,).
        yf: 1-D array of face y coordinates (v-locations), shape (ny+1,).
    """
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    nx: int
    ny: int

    # Derived (set in __post_init__)
    dx: float = field(init=False)
    dy: float = field(init=False)
    xc: np.ndarray = field(init=False, repr=False)
    yc: np.ndarray = field(init=False, repr=False)
    xf: np.ndarray = field(init=False, repr=False)
    yf: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.dx = (self.x_range[1] - self.x_range[0]) / self.nx
        self.dy = (self.y_range[1] - self.y_range[0]) / self.ny
        self.xc = np.linspace(
            self.x_range[0] + 0.5 * self.dx,
            self.x_range[1] - 0.5 * self.dx,
            self.nx,
        )
        self.yc = np.linspace(
            self.y_range[0] + 0.5 * self.dy,
            self.y_range[1] - 0.5 * self.dy,
            self.ny,
        )
        self.xf = np.linspace(self.x_range[0], self.x_range[1], self.nx + 1)
        self.yf = np.linspace(self.y_range[0], self.y_range[1], self.ny + 1)


@dataclass
class NavierStokesResult:
    """Container for Navier-Stokes solution data.

    Attributes:
        u: x-velocity on u-faces, shape (nx+1, ny).
        v: y-velocity on v-faces, shape (nx, ny+1).
        p: Pressure at cell centers, shape (nx, ny).
        rho: Density at cell centers, shape (nx, ny).
        mu: Viscosity at cell centers, shape (nx, ny).
        time: Current simulation time [s].
        grid: The :class:`StaggeredGrid` used.
        time_history: List of time values at which snapshots were stored.
        u_history: List of u-field snapshots.
        v_history: List of v-field snapshots.
        p_history: List of pressure snapshots.
        dt_history: List of time-step sizes used.
    """
    u: np.ndarray
    v: np.ndarray
    p: np.ndarray
    rho: np.ndarray
    mu: np.ndarray
    time: float
    grid: StaggeredGrid
    time_history: List[float] = field(default_factory=list)
    u_history: List[np.ndarray] = field(default_factory=list)
    v_history: List[np.ndarray] = field(default_factory=list)
    p_history: List[np.ndarray] = field(default_factory=list)
    dt_history: List[float] = field(default_factory=list)

    # -- convenience accessors ------------------------------------------------

    def velocity_magnitude_at_centers(self) -> np.ndarray:
        """Interpolate velocity to cell centers and return magnitude, shape (nx, ny)."""
        nx, ny = self.grid.nx, self.grid.ny
        uc = 0.5 * (self.u[:nx, :] + self.u[1:nx + 1, :])
        vc = 0.5 * (self.v[:, :ny] + self.v[:, 1:ny + 1])
        return np.sqrt(uc**2 + vc**2)

    def u_at_centers(self) -> np.ndarray:
        """Interpolate u to cell centers, shape (nx, ny)."""
        nx = self.grid.nx
        return 0.5 * (self.u[:nx, :] + self.u[1:nx + 1, :])

    def v_at_centers(self) -> np.ndarray:
        """Interpolate v to cell centers, shape (nx, ny)."""
        ny = self.grid.ny
        return 0.5 * (self.v[:, :ny] + self.v[:, 1:ny + 1])

    def max_velocity(self) -> float:
        """Return the maximum velocity magnitude on the grid."""
        return float(np.max(self.velocity_magnitude_at_centers()))

    def max_cfl(self, dt: float) -> float:
        """Return the maximum CFL number for a given time step."""
        dx = self.grid.dx
        dy = self.grid.dy
        uc = np.abs(self.u_at_centers())
        vc = np.abs(self.v_at_centers())
        return float(np.max(uc / dx + vc / dy) * dt)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

class NavierStokesSolver:
    """Incompressible variable-density Navier-Stokes solver using the
    projection method on a staggered (MAC) grid.

    Intended for two-phase flow simulations (e.g., nucleate boiling) where
    density and viscosity vary sharply across a liquid-vapor interface.

    Typical workflow::

        solver = NavierStokesSolver(coord_system=CoordinateSystem.AXISYMMETRIC)
        solver.set_domain(x_range=(0, 0.01), y_range=(0, 0.02), nx=64, ny=128)
        solver.set_fluid_properties(rho=958.0, mu=2.82e-4)
        solver.set_gravity(gx=0.0, gy=-9.81)
        solver.set_boundary_conditions(
            left=BoundaryCondition(BCType.SYMMETRY),
            right=BoundaryCondition(BCType.WALL),
            bottom=BoundaryCondition(BCType.WALL),
            top=BoundaryCondition(BCType.OUTLET, pressure=0.0),
        )
        result = solver.solve(t_end=0.1, dt=1e-5)

    Parameters:
        coord_system: :class:`CoordinateSystem` (default ``CARTESIAN``).
    """

    def __init__(self, coord_system: CoordinateSystem = CoordinateSystem.CARTESIAN):
        self.coord_system = coord_system

        # Grid
        self.grid: Optional[StaggeredGrid] = None

        # Fields -- allocated by ``set_domain``
        self.u: Optional[np.ndarray] = None   # (nx+1, ny)
        self.v: Optional[np.ndarray] = None   # (nx, ny+1)
        self.p: Optional[np.ndarray] = None   # (nx, ny)
        self.rho: Optional[np.ndarray] = None # (nx, ny)  density at cell centers
        self.mu: Optional[np.ndarray] = None  # (nx, ny)  viscosity at cell centers

        # Gravity
        self._gx: float = 0.0
        self._gy: float = 0.0

        # Body forces (cell-center arrays, lazily accumulated)
        self._body_fx: Optional[np.ndarray] = None  # (nx, ny)
        self._body_fy: Optional[np.ndarray] = None  # (nx, ny)

        # Mass source for phase-change divergence constraint
        self._mass_source: Optional[np.ndarray] = None  # (nx, ny)

        # Boundary conditions (set by set_boundary_conditions)
        self._bc_left: BoundaryCondition = BoundaryCondition(BCType.WALL)
        self._bc_right: BoundaryCondition = BoundaryCondition(BCType.WALL)
        self._bc_bottom: BoundaryCondition = BoundaryCondition(BCType.WALL)
        self._bc_top: BoundaryCondition = BoundaryCondition(BCType.WALL)

        # Simulation clock
        self._time: float = 0.0

        # Crank-Nicolson parameter (0.5 = CN, 1.0 = fully implicit, 0.0 = explicit)
        self.theta_diffusion: float = 0.5

        # Pressure Poisson tolerance
        self.poisson_tol: float = 1e-8
        self.poisson_maxiter: int = 5000

        # Internal cache for the pressure Laplacian operator
        self._L_pressure: Optional[sp.csr_matrix] = None

    # ------------------------------------------------------------------
    # Setup methods
    # ------------------------------------------------------------------

    def set_domain(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        nx: int,
        ny: int,
    ) -> None:
        """Create the staggered grid and allocate field arrays.

        Args:
            x_range: (x_min, x_max) physical extent.
            y_range: (y_min, y_max) physical extent.
            nx: Number of cells in x.
            ny: Number of cells in y.
        """
        self.grid = StaggeredGrid(x_range=x_range, y_range=y_range, nx=nx, ny=ny)
        self.u = np.zeros((nx + 1, ny))
        self.v = np.zeros((nx, ny + 1))
        self.p = np.zeros((nx, ny))
        self.rho = np.ones((nx, ny))
        self.mu = np.ones((nx, ny)) * 1e-3
        self._body_fx = np.zeros((nx, ny))
        self._body_fy = np.zeros((nx, ny))
        self._mass_source = np.zeros((nx, ny))
        self._L_pressure = None

    def set_fluid_properties(
        self,
        rho: Union[float, np.ndarray],
        mu: Union[float, np.ndarray],
    ) -> None:
        """Set density and dynamic viscosity.

        Args:
            rho: Density. Scalar (uniform) or 2-D array shape (nx, ny).
            mu: Dynamic viscosity. Scalar (uniform) or 2-D array shape (nx, ny).
        """
        if self.grid is None:
            raise RuntimeError("Call set_domain() before set_fluid_properties().")
        nx, ny = self.grid.nx, self.grid.ny
        if np.isscalar(rho):
            self.rho = np.full((nx, ny), float(rho))
        else:
            rho = np.asarray(rho, dtype=float)
            if rho.shape != (nx, ny):
                raise ValueError(f"rho shape {rho.shape} != grid shape ({nx}, {ny})")
            self.rho = rho.copy()

        if np.isscalar(mu):
            self.mu = np.full((nx, ny), float(mu))
        else:
            mu = np.asarray(mu, dtype=float)
            if mu.shape != (nx, ny):
                raise ValueError(f"mu shape {mu.shape} != grid shape ({nx}, {ny})")
            self.mu = mu.copy()

        self._L_pressure = None

    def set_gravity(self, gx: float = 0.0, gy: float = 0.0) -> None:
        """Set gravitational acceleration components.

        Args:
            gx: Gravity in x-direction [m/s^2].
            gy: Gravity in y-direction [m/s^2].
        """
        self._gx = gx
        self._gy = gy

    def set_boundary_conditions(
        self,
        left: Optional[BoundaryCondition] = None,
        right: Optional[BoundaryCondition] = None,
        bottom: Optional[BoundaryCondition] = None,
        top: Optional[BoundaryCondition] = None,
    ) -> None:
        """Set boundary conditions on the four domain boundaries.

        Args:
            left:   BC at x = x_min.
            right:  BC at x = x_max.
            bottom: BC at y = y_min.
            top:    BC at y = y_max.
        """
        if left is not None:
            self._bc_left = left
        if right is not None:
            self._bc_right = right
        if bottom is not None:
            self._bc_bottom = bottom
        if top is not None:
            self._bc_top = top
        self._L_pressure = None

    def add_body_force(
        self,
        fx: Union[float, np.ndarray],
        fy: Union[float, np.ndarray],
    ) -> None:
        """Add a body-force field (e.g., surface tension from CSF model).

        Forces are *accumulated* -- call multiple times to superpose forces.
        They are reset to zero after each ``step()`` call so the caller must
        re-add forces each time step if they persist.

        Args:
            fx: x-component of body force per unit volume [N/m^3].
                Scalar or array shape (nx, ny).
            fy: y-component of body force per unit volume [N/m^3].
                Scalar or array shape (nx, ny).
        """
        if self.grid is None:
            raise RuntimeError("Call set_domain() first.")
        nx, ny = self.grid.nx, self.grid.ny
        if np.isscalar(fx):
            self._body_fx += float(fx)
        else:
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (nx, ny):
                raise ValueError(f"fx shape {fx.shape} != ({nx}, {ny})")
            self._body_fx += fx

        if np.isscalar(fy):
            self._body_fy += float(fy)
        else:
            fy = np.asarray(fy, dtype=float)
            if fy.shape != (nx, ny):
                raise ValueError(f"fy shape {fy.shape} != ({nx}, {ny})")
            self._body_fy += fy

    def add_mass_source(self, S: Union[float, np.ndarray]) -> None:
        """Set the divergence source term for phase-change volume expansion.

        In boiling flows the evaporation causes a local volume source:
            div(u) = S_mass = dot{m}'' * (1/rho_v - 1/rho_l)

        The source is *accumulated* and reset after each ``step()`` call.

        Args:
            S: Mass (divergence) source at cell centers [1/s].
                Scalar or array shape (nx, ny).
        """
        if self.grid is None:
            raise RuntimeError("Call set_domain() first.")
        nx, ny = self.grid.nx, self.grid.ny
        if np.isscalar(S):
            self._mass_source += float(S)
        else:
            S_arr = np.asarray(S, dtype=float)
            if S_arr.shape != (nx, ny):
                raise ValueError(f"S shape {S_arr.shape} != ({nx}, {ny})")
            self._mass_source += S_arr

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def set_initial_velocity(
        self,
        u: Union[float, np.ndarray],
        v: Union[float, np.ndarray],
    ) -> None:
        """Set initial velocity field.

        Args:
            u: u-velocity. Scalar or array shape (nx+1, ny).
            v: v-velocity. Scalar or array shape (nx, ny+1).
        """
        if self.grid is None:
            raise RuntimeError("Call set_domain() first.")
        nx, ny = self.grid.nx, self.grid.ny
        if np.isscalar(u):
            self.u = np.full((nx + 1, ny), float(u))
        else:
            u = np.asarray(u, dtype=float)
            if u.shape != (nx + 1, ny):
                raise ValueError(f"u shape {u.shape} != ({nx + 1}, {ny})")
            self.u = u.copy()
        if np.isscalar(v):
            self.v = np.full((nx, ny + 1), float(v))
        else:
            v = np.asarray(v, dtype=float)
            if v.shape != (nx, ny + 1):
                raise ValueError(f"v shape {v.shape} != ({nx}, {ny + 1})")
            self.v = v.copy()

    # ------------------------------------------------------------------
    # Internal: ghost cell construction for boundary conditions
    # ------------------------------------------------------------------

    def _u_ghost_bottom(self) -> np.ndarray:
        """Return ghost row of u below the domain (j = -1), shape (nx+1,).

        u lives at (xf[i], yc[j]).  The domain boundary is at y_min,
        which is at dy/2 below yc[0].  The ghost value is at dy/2 below
        the boundary, i.e. at yc[-1] = yc[0] - dy.

        For no-slip wall:  u_wall = 0 => u_ghost = -u[:,0]
        For inlet:         u_wall = u_in => u_ghost = 2*u_in - u[:,0]
        For outlet:        du/dy = 0 => u_ghost = u[:,0]
        For symmetry:      du/dy = 0 => u_ghost = u[:,0]
        """
        bc = self._bc_bottom
        u0 = self.u[:, 0]
        if bc.bc_type == BCType.WALL:
            u_wall = bc.velocity[0] if bc.velocity is not None else 0.0
            return 2.0 * u_wall - u0
        elif bc.bc_type == BCType.INLET:
            return 2.0 * bc.velocity[0] - u0
        elif bc.bc_type == BCType.OUTLET:
            return u0.copy()
        elif bc.bc_type == BCType.SYMMETRY:
            return u0.copy()
        return u0.copy()

    def _u_ghost_top(self) -> np.ndarray:
        """Return ghost row of u above the domain (j = ny), shape (nx+1,)."""
        bc = self._bc_top
        u_last = self.u[:, -1]
        if bc.bc_type == BCType.WALL:
            # No-slip: ghost = 2*U_wall - u_interior (supports moving walls)
            u_wall = bc.velocity[0] if bc.velocity is not None else 0.0
            return 2.0 * u_wall - u_last
        elif bc.bc_type == BCType.INLET:
            return 2.0 * bc.velocity[0] - u_last
        elif bc.bc_type == BCType.OUTLET:
            return u_last.copy()
        elif bc.bc_type == BCType.SYMMETRY:
            return u_last.copy()
        return u_last.copy()

    def _v_ghost_left(self) -> np.ndarray:
        """Return ghost column of v to the left of the domain (i = -1), shape (ny+1,)."""
        bc = self._bc_left
        v0 = self.v[0, :]
        if bc.bc_type == BCType.WALL:
            v_wall = bc.velocity[1] if bc.velocity is not None else 0.0
            return 2.0 * v_wall - v0
        elif bc.bc_type == BCType.INLET:
            return 2.0 * bc.velocity[1] - v0
        elif bc.bc_type == BCType.OUTLET:
            return v0.copy()
        elif bc.bc_type == BCType.SYMMETRY:
            return v0.copy()
        return v0.copy()

    def _v_ghost_right(self) -> np.ndarray:
        """Return ghost column of v to the right of the domain (i = nx), shape (ny+1,)."""
        bc = self._bc_right
        v_last = self.v[-1, :]
        if bc.bc_type == BCType.WALL:
            v_wall = bc.velocity[1] if bc.velocity is not None else 0.0
            return 2.0 * v_wall - v_last
        elif bc.bc_type == BCType.INLET:
            return 2.0 * bc.velocity[1] - v_last
        elif bc.bc_type == BCType.OUTLET:
            return v_last.copy()
        elif bc.bc_type == BCType.SYMMETRY:
            return v_last.copy()
        return v_last.copy()

    # ------------------------------------------------------------------
    # Internal: interpolation helpers for the staggered grid
    # ------------------------------------------------------------------

    def _rho_at_u_faces(self) -> np.ndarray:
        """Interpolate density to u-face locations, shape (nx+1, ny)."""
        nx, ny = self.grid.nx, self.grid.ny
        rho_u = np.zeros((nx + 1, ny))
        rho_u[1:nx, :] = 0.5 * (self.rho[:nx - 1, :] + self.rho[1:nx, :])
        rho_u[0, :] = self.rho[0, :]
        rho_u[nx, :] = self.rho[nx - 1, :]
        return rho_u

    def _rho_at_v_faces(self) -> np.ndarray:
        """Interpolate density to v-face locations, shape (nx, ny+1)."""
        nx, ny = self.grid.nx, self.grid.ny
        rho_v = np.zeros((nx, ny + 1))
        rho_v[:, 1:ny] = 0.5 * (self.rho[:, :ny - 1] + self.rho[:, 1:ny])
        rho_v[:, 0] = self.rho[:, 0]
        rho_v[:, ny] = self.rho[:, ny - 1]
        return rho_v

    def _mu_at_u_faces(self) -> np.ndarray:
        """Interpolate viscosity to u-face locations, shape (nx+1, ny)."""
        nx, ny = self.grid.nx, self.grid.ny
        mu_u = np.zeros((nx + 1, ny))
        mu_u[1:nx, :] = 0.5 * (self.mu[:nx - 1, :] + self.mu[1:nx, :])
        mu_u[0, :] = self.mu[0, :]
        mu_u[nx, :] = self.mu[nx - 1, :]
        return mu_u

    def _mu_at_v_faces(self) -> np.ndarray:
        """Interpolate viscosity to v-face locations, shape (nx, ny+1)."""
        nx, ny = self.grid.nx, self.grid.ny
        mu_v = np.zeros((nx, ny + 1))
        mu_v[:, 1:ny] = 0.5 * (self.mu[:, :ny - 1] + self.mu[:, 1:ny])
        mu_v[:, 0] = self.mu[:, 0]
        mu_v[:, ny] = self.mu[:, ny - 1]
        return mu_v

    # ------------------------------------------------------------------
    # Internal: advection (explicit, Forward Euler) -- vectorized
    # ------------------------------------------------------------------

    def _compute_advection(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the advection terms at face locations (vectorized).

        For the u-momentum at interior u-faces (i=1..nx-1, j=0..ny-1):
            adv_u = -d(uu)/dx - d(uv)/dy

        For the v-momentum at interior v-faces (i=0..nx-1, j=1..ny-1):
            adv_v = -d(uv)/dx - d(vv)/dy

        Here the convective fluxes are computed using second-order
        centered interpolation on the staggered grid.

        Returns:
            (adv_u, adv_v): advection acceleration (already divided by rho)
                multiplied back by rho_face for the momentum form.
                Shapes (nx+1, ny) and (nx, ny+1).  Only interior face values
                are filled; boundary faces remain zero.
        """
        nx, ny = self.grid.nx, self.grid.ny
        dx, dy = self.grid.dx, self.grid.dy
        u = self.u  # (nx+1, ny)
        v = self.v  # (nx, ny+1)

        adv_u = np.zeros_like(u)
        adv_v = np.zeros_like(v)

        # --- u-momentum advection at interior u-faces (i=1..nx-1) ---
        if nx > 1:
            # d(u*u)/dx at u-face (i, j):
            # Interpolate u to cell centers on either side and compute flux
            # u at cell center (i, j) ~ 0.5*(u[i,j] + u[i+1,j])
            # Flux at right cell boundary = rho_R * u_R * u_R
            # Here we compute fluxes at cell centers surrounding each u-face.

            # For u-face at index i (between cells i-1 and i):
            #   Right flux at cell center i:  0.5*(u[i]+u[i+1]) squared * rho[i]
            #   Left flux at cell center i-1: 0.5*(u[i-1]+u[i]) squared * rho[i-1]
            for i in range(1, nx):
                # uu flux in x
                if i < nx:
                    u_R = 0.5 * (u[i, :] + u[i + 1, :])  # u at cell center i
                else:
                    u_R = u[i, :]
                if i > 0:
                    u_L = 0.5 * (u[i - 1, :] + u[i, :])  # u at cell center i-1
                else:
                    u_L = u[i, :]

                rho_R = self.rho[min(i, nx - 1), :]
                rho_L = self.rho[max(i - 1, 0), :]

                flux_uu_x = (rho_R * u_R * u_R - rho_L * u_L * u_L) / dx

                # uv flux in y:
                # Need v at u-face location: average v from cells on either side
                # and u at top/bottom of the cell containing the u-face.

                # Top face: v at (i, j+1) interpolated to u-face x-position
                # v_top ~ 0.5*(v[i-1, j+1] + v[i, j+1]) but need to handle boundaries
                # u_top ~ 0.5*(u[i, j] + u[i, j+1])

                # Build arrays for j-direction flux
                # v interpolated to u-face locations, at j+1/2 and j-1/2 faces
                i_L = max(i - 1, 0)
                i_R = min(i, nx - 1)

                # v at top of each j-row (j+1/2 face)
                v_top = 0.5 * (v[i_L, 1:ny + 1] + v[i_R, 1:ny + 1])  # (ny,)
                v_bot = 0.5 * (v[i_L, :ny] + v[i_R, :ny])              # (ny,)

                # u at top/bottom (interpolated in y)
                u_ghost_top = self._u_ghost_top()
                u_ghost_bot = self._u_ghost_bottom()

                # u at j+1/2 face
                u_top = np.empty(ny)
                u_top[:ny - 1] = 0.5 * (u[i, :ny - 1] + u[i, 1:ny])
                u_top[ny - 1] = 0.5 * (u[i, ny - 1] + u_ghost_top[i])

                u_bot = np.empty(ny)
                u_bot[0] = 0.5 * (u_ghost_bot[i] + u[i, 0])
                u_bot[1:ny] = 0.5 * (u[i, :ny - 1] + u[i, 1:ny])

                # rho at the y-faces
                rho_top = np.empty(ny)
                rho_top[:ny - 1] = 0.5 * (self.rho[i_L, :ny - 1] + self.rho[i_L, 1:ny]
                                           + self.rho[i_R, :ny - 1] + self.rho[i_R, 1:ny]) / 2.0
                rho_top[ny - 1] = 0.5 * (self.rho[i_L, ny - 1] + self.rho[i_R, ny - 1])
                rho_bot = np.empty(ny)
                rho_bot[0] = 0.5 * (self.rho[i_L, 0] + self.rho[i_R, 0])
                rho_bot[1:ny] = 0.5 * (self.rho[i_L, :ny - 1] + self.rho[i_L, 1:ny]
                                        + self.rho[i_R, :ny - 1] + self.rho[i_R, 1:ny]) / 2.0

                flux_uv_y = (rho_top * u_top * v_top - rho_bot * u_bot * v_bot) / dy

                adv_u[i, :] = -(flux_uu_x + flux_uv_y)

        # --- v-momentum advection at interior v-faces (j=1..ny-1) ---
        if ny > 1:
            for j in range(1, ny):
                # vv flux in y
                j_B = max(j - 1, 0)
                j_T = min(j, ny - 1)

                if j < ny:
                    v_T = 0.5 * (v[:, j] + v[:, j + 1])
                else:
                    v_T = v[:, j]
                if j > 0:
                    v_B = 0.5 * (v[:, j - 1] + v[:, j])
                else:
                    v_B = v[:, j]

                rho_T = self.rho[:, j_T]
                rho_B = self.rho[:, j_B]

                flux_vv_y = (rho_T * v_T * v_T - rho_B * v_B * v_B) / dy

                # uv flux in x
                v_ghost_left = self._v_ghost_left()
                v_ghost_right = self._v_ghost_right()

                # u interpolated to v-face y-location
                u_right = 0.5 * (u[1:nx + 1, j_B] + u[1:nx + 1, j_T])  # (nx,)
                u_left = 0.5 * (u[:nx, j_B] + u[:nx, j_T])              # (nx,)

                # v interpolated to x-faces
                v_right = np.empty(nx)
                v_right[:nx - 1] = 0.5 * (v[:nx - 1, j] + v[1:nx, j])
                v_right[nx - 1] = 0.5 * (v[nx - 1, j] + v_ghost_right[j])

                v_left = np.empty(nx)
                v_left[0] = 0.5 * (v_ghost_left[j] + v[0, j])
                v_left[1:nx] = 0.5 * (v[:nx - 1, j] + v[1:nx, j])

                rho_right = np.empty(nx)
                rho_right[:nx - 1] = 0.5 * (self.rho[:nx - 1, j_B] + self.rho[:nx - 1, j_T]
                                             + self.rho[1:nx, j_B] + self.rho[1:nx, j_T]) / 2.0
                rho_right[nx - 1] = 0.5 * (self.rho[nx - 1, j_B] + self.rho[nx - 1, j_T])
                rho_left = np.empty(nx)
                rho_left[0] = 0.5 * (self.rho[0, j_B] + self.rho[0, j_T])
                rho_left[1:nx] = 0.5 * (self.rho[:nx - 1, j_B] + self.rho[:nx - 1, j_T]
                                         + self.rho[1:nx, j_B] + self.rho[1:nx, j_T]) / 2.0

                flux_uv_x = (rho_right * u_right * v_right - rho_left * u_left * v_left) / dx

                adv_v[:, j] = -(flux_vv_y + flux_uv_x)

        return adv_u, adv_v

    # ------------------------------------------------------------------
    # Internal: diffusion (explicit part)
    # ------------------------------------------------------------------

    def _compute_diffusion_explicit(self) -> Tuple[np.ndarray, np.ndarray]:
        r"""Compute the explicit viscous diffusion: div(mu * (grad u + grad u^T)).

        Uses ghost cells for proper boundary treatment.  For simplicity and
        stability, the cross-derivative terms are neglected in this explicit
        evaluation (they contribute at second order and are typically small
        for the problems of interest).  The dominant diagonal Laplacian
        terms are retained:

            diff_u ~ d/dx(2*mu * du/dx) + d/dy(mu * du/dy)
            diff_v ~ d/dx(mu * dv/dx) + d/dy(2*mu * dv/dy)

        Returns:
            (diff_u, diff_v): viscous stress divergence at face locations.
        """
        nx, ny = self.grid.nx, self.grid.ny
        dx, dy = self.grid.dx, self.grid.dy
        u = self.u
        v = self.v

        diff_u = np.zeros_like(u)
        diff_v = np.zeros_like(v)

        # Ghost cells for u in y-direction
        u_ghost_bot = self._u_ghost_bottom()  # (nx+1,)
        u_ghost_top = self._u_ghost_top()      # (nx+1,)

        # Ghost cells for v in x-direction
        v_ghost_left = self._v_ghost_left()    # (ny+1,)
        v_ghost_right = self._v_ghost_right()  # (ny+1,)

        # --- u-diffusion at interior u-faces (i=1..nx-1) ---
        for i in range(1, nx):
            mu_R = self.mu[min(i, nx - 1), :]    # mu at cell center to the right
            mu_L = self.mu[max(i - 1, 0), :]      # mu at cell center to the left

            # d/dx(2*mu * du/dx)
            dudx_R = (u[min(i + 1, nx), :] - u[i, :]) / dx
            dudx_L = (u[i, :] - u[max(i - 1, 0), :]) / dx
            d2u_x = (2.0 * mu_R * dudx_R - 2.0 * mu_L * dudx_L) / dx

            # d/dy(mu * du/dy) with ghost cells
            mu_face = 0.5 * (mu_R + mu_L)

            # Extend u in y with ghosts
            u_col = np.empty(ny + 2)
            u_col[0] = u_ghost_bot[i]
            u_col[1:ny + 1] = u[i, :]
            u_col[ny + 1] = u_ghost_top[i]

            dudy_top = (u_col[2:ny + 2] - u_col[1:ny + 1]) / dy
            dudy_bot = (u_col[1:ny + 1] - u_col[0:ny]) / dy
            d2u_y = mu_face * (dudy_top - dudy_bot) / dy

            diff_u[i, :] = d2u_x + d2u_y

            # Axisymmetric hoop stress: -2*mu*u/r^2
            if self.coord_system == CoordinateSystem.AXISYMMETRIC:
                r = self.grid.xf[i]
                if r > 1e-30:
                    diff_u[i, :] -= 2.0 * mu_face * u[i, :] / (r * r)

        # --- v-diffusion at interior v-faces (j=1..ny-1) ---
        for j in range(1, ny):
            mu_T = self.mu[:, min(j, ny - 1)]
            mu_B = self.mu[:, max(j - 1, 0)]

            # d/dy(2*mu * dv/dy)
            dvdy_T = (v[:, min(j + 1, ny)] - v[:, j]) / dy
            dvdy_B = (v[:, j] - v[:, max(j - 1, 0)]) / dy
            d2v_y = (2.0 * mu_T * dvdy_T - 2.0 * mu_B * dvdy_B) / dy

            # d/dx(mu * dv/dx) with ghost cells
            mu_face = 0.5 * (mu_T + mu_B)

            v_row = np.empty(nx + 2)
            v_row[0] = v_ghost_left[j]
            v_row[1:nx + 1] = v[:, j]
            v_row[nx + 1] = v_ghost_right[j]

            dvdx_R = (v_row[2:nx + 2] - v_row[1:nx + 1]) / dx
            dvdx_L = (v_row[1:nx + 1] - v_row[0:nx]) / dx
            d2v_x = mu_face * (dvdx_R - dvdx_L) / dx

            diff_v[:, j] = d2v_y + d2v_x

            # Axisymmetric: additional 1/r * d/dr term
            if self.coord_system == CoordinateSystem.AXISYMMETRIC:
                for i_ax in range(nx):
                    r = self.grid.xc[i_ax]
                    if r > 1e-30:
                        diff_v[i_ax, j] += mu_face[i_ax] * (dvdx_R[i_ax] + dvdx_L[i_ax]) / (2.0 * r)

        return diff_u, diff_v

    # ------------------------------------------------------------------
    # Internal: implicit diffusion (Crank-Nicolson)
    # ------------------------------------------------------------------

    def _apply_diffusion_implicit(
        self,
        u_star: np.ndarray,
        v_star: np.ndarray,
        dt: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""Solve the implicit part of Crank-Nicolson diffusion.

        Solves component-by-component:
            (I - theta*dt/(rho) * mu * L) * u_new = u_star_rhs

        where L is the scalar Laplacian.  The cross-derivative terms
        are treated explicitly (standard approximate factorisation).

        Ghost cell values for boundaries that prescribe wall velocity
        (WALL, INLET) are incorporated into the RHS of the linear system.

        Args:
            u_star: Predicted u after explicit terms, shape (nx+1, ny).
            v_star: Predicted v after explicit terms, shape (nx, ny+1).
            dt: Time step.

        Returns:
            (u_new, v_new): Velocity after implicit diffusion solve.
        """
        if self.theta_diffusion < 1e-14:
            return u_star.copy(), v_star.copy()

        nx, ny = self.grid.nx, self.grid.ny
        dx, dy = self.grid.dx, self.grid.dy
        theta = self.theta_diffusion
        rho_u = self._rho_at_u_faces()
        mu_u = self._mu_at_u_faces()

        # --- Solve for u interior faces (i=1..nx-1, all j) ----------------
        n_u = (nx - 1) * ny

        if n_u > 0:
            def _u_idx(ii, jj):
                return (ii - 1) * ny + jj

            rows, cols, vals = [], [], []
            rhs = np.zeros(n_u)

            # Ghost values for y-BCs
            u_ghost_bot = self._u_ghost_bottom()
            u_ghost_top = self._u_ghost_top()

            for i in range(1, nx):
                for j in range(ny):
                    idx = _u_idx(i, j)
                    rhs[idx] = u_star[i, j]

                    rho_here = max(rho_u[i, j], 1e-30)
                    mu_here = mu_u[i, j]
                    alpha = theta * dt * mu_here / rho_here

                    # Axisymmetric hoop stress in implicit part
                    hoop = 0.0
                    if self.coord_system == CoordinateSystem.AXISYMMETRIC:
                        r = self.grid.xf[i]
                        if r > 1e-30:
                            hoop = 2.0 * alpha / (r * r)

                    diag = 1.0 + alpha * (2.0 / dx**2 + 2.0 / dy**2) + hoop

                    rows.append(idx); cols.append(idx); vals.append(diag)

                    # x-neighbours
                    if i - 1 >= 1:
                        n_idx = _u_idx(i - 1, j)
                        rows.append(idx); cols.append(n_idx); vals.append(-alpha / dx**2)
                    else:
                        # i-1 = 0 is a boundary face; u_star[0,j] is prescribed by BC
                        rhs[idx] += alpha / dx**2 * u_star[0, j]

                    if i + 1 <= nx - 1:
                        n_idx = _u_idx(i + 1, j)
                        rows.append(idx); cols.append(n_idx); vals.append(-alpha / dx**2)
                    else:
                        rhs[idx] += alpha / dx**2 * u_star[nx, j]

                    # y-neighbours (using ghost cells for BCs)
                    if j - 1 >= 0:
                        n_idx = _u_idx(i, j - 1)
                        rows.append(idx); cols.append(n_idx); vals.append(-alpha / dy**2)
                    else:
                        # Ghost below: u_ghost_bot[i]
                        # For the implicit solve we need the ghost of u_star,
                        # not of the old u. For wall BCs the ghost is -u_star
                        # at the boundary, which we approximate using the
                        # explicit ghost (based on u^n). This is fine because
                        # the implicit correction is typically small.
                        rhs[idx] += alpha / dy**2 * u_ghost_bot[i]

                    if j + 1 <= ny - 1:
                        n_idx = _u_idx(i, j + 1)
                        rows.append(idx); cols.append(n_idx); vals.append(-alpha / dy**2)
                    else:
                        rhs[idx] += alpha / dy**2 * u_ghost_top[i]

            A_u = sp.csr_matrix((vals, (rows, cols)), shape=(n_u, n_u))
            u_sol = spla.spsolve(A_u, rhs)
            u_new = u_star.copy()
            for i in range(1, nx):
                u_new[i, :] = u_sol[(i - 1) * ny: i * ny]
        else:
            u_new = u_star.copy()

        # --- Solve for v interior faces (all i, j=1..ny-1) ----------------
        rho_v = self._rho_at_v_faces()
        mu_v = self._mu_at_v_faces()
        n_v = nx * (ny - 1)

        if n_v > 0:
            def _v_idx(ii, jj):
                return ii * (ny - 1) + (jj - 1)

            rows, cols, vals = [], [], []
            rhs = np.zeros(n_v)

            v_ghost_left = self._v_ghost_left()
            v_ghost_right = self._v_ghost_right()

            for i in range(nx):
                for j in range(1, ny):
                    idx = _v_idx(i, j)
                    rhs[idx] = v_star[i, j]

                    rho_here = max(rho_v[i, j], 1e-30)
                    mu_here = mu_v[i, j]
                    alpha = theta * dt * mu_here / rho_here

                    diag = 1.0 + alpha * (2.0 / dx**2 + 2.0 / dy**2)
                    rows.append(idx); cols.append(idx); vals.append(diag)

                    # y-neighbours
                    if j - 1 >= 1:
                        n_idx = _v_idx(i, j - 1)
                        rows.append(idx); cols.append(n_idx); vals.append(-alpha / dy**2)
                    else:
                        rhs[idx] += alpha / dy**2 * v_star[i, 0]

                    if j + 1 <= ny - 1:
                        n_idx = _v_idx(i, j + 1)
                        rows.append(idx); cols.append(n_idx); vals.append(-alpha / dy**2)
                    else:
                        rhs[idx] += alpha / dy**2 * v_star[i, ny]

                    # x-neighbours (using ghost cells for BCs)
                    if i - 1 >= 0:
                        n_idx = _v_idx(i - 1, j)
                        rows.append(idx); cols.append(n_idx); vals.append(-alpha / dx**2)
                    else:
                        rhs[idx] += alpha / dx**2 * v_ghost_left[j]

                    if i + 1 <= nx - 1:
                        n_idx = _v_idx(i + 1, j)
                        rows.append(idx); cols.append(n_idx); vals.append(-alpha / dx**2)
                    else:
                        rhs[idx] += alpha / dx**2 * v_ghost_right[j]

            A_v = sp.csr_matrix((vals, (rows, cols)), shape=(n_v, n_v))
            v_sol = spla.spsolve(A_v, rhs)
            v_new = v_star.copy()
            for i in range(nx):
                v_new[i, 1:ny] = v_sol[i * (ny - 1):(i + 1) * (ny - 1)]
        else:
            v_new = v_star.copy()

        return u_new, v_new

    # ------------------------------------------------------------------
    # Internal: pressure Poisson equation
    # ------------------------------------------------------------------

    def _build_pressure_laplacian(self) -> sp.csr_matrix:
        r"""Build the sparse matrix for the pressure Poisson equation:

            div( (1/rho) * grad(p) ) = RHS

        on the cell-centered grid with appropriate boundary conditions.

        For axisymmetric coordinates:
            (1/r) d/dr( r/rho * dp/dr ) + d/dz( 1/rho * dp/dz )

        Returns:
            Sparse matrix of shape (nx*ny, nx*ny).
        """
        nx, ny = self.grid.nx, self.grid.ny
        dx, dy = self.grid.dx, self.grid.dy
        N = nx * ny

        def _idx(i, j):
            return i * ny + j

        rows, cols, vals = [], [], []

        for i in range(nx):
            for j in range(ny):
                idx = _idx(i, j)
                diag = 0.0

                # --- x-direction ---
                # Left face (between cells i-1 and i)
                if i > 0:
                    rho_face = 2.0 / (1.0 / self.rho[i - 1, j] + 1.0 / self.rho[i, j])
                    coeff = 1.0 / (rho_face * dx * dx)
                    if self.coord_system == CoordinateSystem.AXISYMMETRIC:
                        r_c = self.grid.xc[i]
                        r_f = self.grid.xf[i]
                        if r_c > 1e-30:
                            coeff *= r_f / r_c
                    rows.append(idx); cols.append(_idx(i - 1, j)); vals.append(coeff)
                    diag -= coeff
                else:
                    bc = self._bc_left
                    if bc.bc_type == BCType.OUTLET and bc.pressure is not None:
                        rho_face = self.rho[0, j]
                        coeff = 1.0 / (rho_face * dx * dx)
                        if self.coord_system == CoordinateSystem.AXISYMMETRIC:
                            r_c = self.grid.xc[0]
                            r_f = self.grid.xf[0]
                            if r_c > 1e-30:
                                coeff *= r_f / r_c
                        diag -= 2.0 * coeff
                    # else: Neumann (wall/inlet/symmetry) => ghost = interior => no extra term

                # Right face (between cells i and i+1)
                if i < nx - 1:
                    rho_face = 2.0 / (1.0 / self.rho[i, j] + 1.0 / self.rho[i + 1, j])
                    coeff = 1.0 / (rho_face * dx * dx)
                    if self.coord_system == CoordinateSystem.AXISYMMETRIC:
                        r_c = self.grid.xc[i]
                        r_f = self.grid.xf[i + 1]
                        if r_c > 1e-30:
                            coeff *= r_f / r_c
                    rows.append(idx); cols.append(_idx(i + 1, j)); vals.append(coeff)
                    diag -= coeff
                else:
                    bc = self._bc_right
                    if bc.bc_type == BCType.OUTLET and bc.pressure is not None:
                        rho_face = self.rho[nx - 1, j]
                        coeff = 1.0 / (rho_face * dx * dx)
                        if self.coord_system == CoordinateSystem.AXISYMMETRIC:
                            r_c = self.grid.xc[nx - 1]
                            r_f = self.grid.xf[nx]
                            if r_c > 1e-30:
                                coeff *= r_f / r_c
                        diag -= 2.0 * coeff

                # --- y-direction ---
                if j > 0:
                    rho_face = 2.0 / (1.0 / self.rho[i, j - 1] + 1.0 / self.rho[i, j])
                    coeff = 1.0 / (rho_face * dy * dy)
                    rows.append(idx); cols.append(_idx(i, j - 1)); vals.append(coeff)
                    diag -= coeff
                else:
                    bc = self._bc_bottom
                    if bc.bc_type == BCType.OUTLET and bc.pressure is not None:
                        rho_face = self.rho[i, 0]
                        coeff = 1.0 / (rho_face * dy * dy)
                        diag -= 2.0 * coeff

                if j < ny - 1:
                    rho_face = 2.0 / (1.0 / self.rho[i, j] + 1.0 / self.rho[i, j + 1])
                    coeff = 1.0 / (rho_face * dy * dy)
                    rows.append(idx); cols.append(_idx(i, j + 1)); vals.append(coeff)
                    diag -= coeff
                else:
                    bc = self._bc_top
                    if bc.bc_type == BCType.OUTLET and bc.pressure is not None:
                        rho_face = self.rho[i, ny - 1]
                        coeff = 1.0 / (rho_face * dy * dy)
                        diag -= 2.0 * coeff

                rows.append(idx); cols.append(idx); vals.append(diag)

        L = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

        # If no Dirichlet BC the system is singular; pin pressure at (0,0)
        has_dirichlet = any(
            bc.bc_type == BCType.OUTLET and bc.pressure is not None
            for bc in [self._bc_left, self._bc_right, self._bc_bottom, self._bc_top]
        )
        if not has_dirichlet:
            pin = _idx(0, 0)
            L = L.tolil()
            L[pin, :] = 0.0
            L[pin, pin] = 1.0
            L = L.tocsr()

        return L

    def _divergence_of_velocity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute div(u) at cell centers, shape (nx, ny).

        For axisymmetric: div = (1/r) d(r*u)/dr + dv/dz
        For Cartesian:    div = du/dx + dv/dy
        """
        nx, ny = self.grid.nx, self.grid.ny
        dx, dy = self.grid.dx, self.grid.dy

        if self.coord_system == CoordinateSystem.AXISYMMETRIC:
            rf = self.grid.xf
            rc = self.grid.xc
            ru_right = rf[1:nx + 1, np.newaxis] * u[1:nx + 1, :]
            ru_left = rf[:nx, np.newaxis] * u[:nx, :]
            rc_safe = np.where(rc[:, np.newaxis] > 1e-30, rc[:, np.newaxis], 1e-30)
            div_r = (ru_right - ru_left) / (rc_safe * dx)
            div_z = (v[:, 1:ny + 1] - v[:, :ny]) / dy
            return div_r + div_z
        else:
            dudx = (u[1:nx + 1, :] - u[:nx, :]) / dx
            dvdy = (v[:, 1:ny + 1] - v[:, :ny]) / dy
            return dudx + dvdy

    def _solve_pressure(self, u_star: np.ndarray, v_star: np.ndarray, dt: float) -> np.ndarray:
        """Solve the pressure Poisson equation.

            L * p = (div(u*) - S_mass) / dt

        Returns pressure at cell centers, shape (nx, ny).
        """
        nx, ny = self.grid.nx, self.grid.ny
        dx, dy = self.grid.dx, self.grid.dy

        if self._L_pressure is None:
            self._L_pressure = self._build_pressure_laplacian()

        div = self._divergence_of_velocity(u_star, v_star)
        rhs_2d = (div - self._mass_source) / dt

        def _idx(i, j):
            return i * ny + j

        rhs = rhs_2d.ravel().copy()

        # Dirichlet pressure BC contributions
        for side, bc, boundary_cells in [
            ("left", self._bc_left,
             [(0, jj) for jj in range(ny)]),
            ("right", self._bc_right,
             [(nx - 1, jj) for jj in range(ny)]),
            ("bottom", self._bc_bottom,
             [(ii, 0) for ii in range(nx)]),
            ("top", self._bc_top,
             [(ii, ny - 1) for ii in range(nx)]),
        ]:
            if bc.bc_type == BCType.OUTLET and bc.pressure is not None:
                p_bc = bc.pressure
                for (ci, cj) in boundary_cells:
                    rho_f = self.rho[ci, cj]
                    if side in ("left", "right"):
                        coeff = 1.0 / (rho_f * dx * dx)
                        if self.coord_system == CoordinateSystem.AXISYMMETRIC:
                            r_c = self.grid.xc[ci]
                            fi = 0 if side == "left" else nx
                            r_f = self.grid.xf[fi]
                            if r_c > 1e-30:
                                coeff *= r_f / r_c
                    else:
                        coeff = 1.0 / (rho_f * dy * dy)
                    rhs[_idx(ci, cj)] -= 2.0 * coeff * p_bc

        # Pin fix
        has_dirichlet = any(
            bc.bc_type == BCType.OUTLET and bc.pressure is not None
            for bc in [self._bc_left, self._bc_right, self._bc_bottom, self._bc_top]
        )
        if not has_dirichlet:
            rhs[_idx(0, 0)] = 0.0

        # Solve with iterative solver, fall back to direct if needed
        p_flat, info = spla.bicgstab(
            self._L_pressure, rhs,
            x0=self.p.ravel(),
            rtol=self.poisson_tol,
            maxiter=self.poisson_maxiter,
        )
        if info != 0:
            try:
                p_flat = spla.spsolve(self._L_pressure, rhs)
            except Exception:
                p_flat = spla.lsqr(self._L_pressure, rhs)[0]

        return p_flat.reshape((nx, ny))

    # ------------------------------------------------------------------
    # Internal: pressure correction
    # ------------------------------------------------------------------

    def _correct_velocity(self, u_star: np.ndarray, v_star: np.ndarray,
                          p: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        r"""Apply the pressure-gradient correction:

            u^{n+1} = u* - (dt / rho_face) * dp/dx
            v^{n+1} = v* - (dt / rho_face) * dp/dy
        """
        nx, ny = self.grid.nx, self.grid.ny
        dx, dy = self.grid.dx, self.grid.dy
        rho_u = self._rho_at_u_faces()
        rho_v = self._rho_at_v_faces()

        u_new = u_star.copy()
        v_new = v_star.copy()

        # Interior u-faces (i=1..nx-1)
        safe_rho = np.where(rho_u[1:nx, :] > 1e-30, rho_u[1:nx, :], 1e-30)
        dpdx = (p[1:nx, :] - p[:nx - 1, :]) / dx
        u_new[1:nx, :] -= dt / safe_rho * dpdx

        # Interior v-faces (j=1..ny-1)
        safe_rho_v = np.where(rho_v[:, 1:ny] > 1e-30, rho_v[:, 1:ny], 1e-30)
        dpdy = (p[:, 1:ny] - p[:, :ny - 1]) / dy
        v_new[:, 1:ny] -= dt / safe_rho_v * dpdy

        return u_new, v_new

    # ------------------------------------------------------------------
    # Internal: boundary condition enforcement
    # ------------------------------------------------------------------

    def _apply_velocity_bcs(self, u: np.ndarray, v: np.ndarray) -> None:
        """Enforce velocity boundary conditions on the staggered grid (in-place).

        On the MAC grid:
            - u lives on vertical faces: u[0,:] is the left boundary face,
              u[nx,:] is the right boundary face.
            - v lives on horizontal faces: v[:,0] is the bottom boundary face,
              v[:,ny] is the top boundary face.

        For tangential components that do not have a face on the boundary
        (e.g., v at the left/right walls), the ghost cell approach is used
        in the advection and diffusion routines instead.
        """
        nx, ny = self.grid.nx, self.grid.ny

        # Left (x = x_min): u-face i=0
        bc = self._bc_left
        if bc.bc_type == BCType.WALL:
            u[0, :] = 0.0
        elif bc.bc_type == BCType.INLET:
            u[0, :] = bc.velocity[0]
        elif bc.bc_type == BCType.OUTLET:
            u[0, :] = u[1, :]
        elif bc.bc_type == BCType.SYMMETRY:
            u[0, :] = 0.0

        # Right (x = x_max): u-face i=nx
        bc = self._bc_right
        if bc.bc_type == BCType.WALL:
            u[nx, :] = 0.0
        elif bc.bc_type == BCType.INLET:
            u[nx, :] = bc.velocity[0]
        elif bc.bc_type == BCType.OUTLET:
            u[nx, :] = u[nx - 1, :]
        elif bc.bc_type == BCType.SYMMETRY:
            u[nx, :] = 0.0

        # Bottom (y = y_min): v-face j=0
        bc = self._bc_bottom
        if bc.bc_type == BCType.WALL:
            v[:, 0] = 0.0
        elif bc.bc_type == BCType.INLET:
            v[:, 0] = bc.velocity[1]
        elif bc.bc_type == BCType.OUTLET:
            v[:, 0] = v[:, 1]
        elif bc.bc_type == BCType.SYMMETRY:
            v[:, 0] = 0.0

        # Top (y = y_max): v-face j=ny
        bc = self._bc_top
        if bc.bc_type == BCType.WALL:
            v[:, ny] = 0.0
        elif bc.bc_type == BCType.INLET:
            v[:, ny] = bc.velocity[1]
        elif bc.bc_type == BCType.OUTLET:
            v[:, ny] = v[:, ny - 1]
        elif bc.bc_type == BCType.SYMMETRY:
            v[:, ny] = 0.0

    # ------------------------------------------------------------------
    # Public: time stepping
    # ------------------------------------------------------------------

    def step(self, dt: float) -> NavierStokesResult:
        """Advance the solution by one time step using the projection method.

        Projection (Chorin's splitting):
            1. Predictor: compute intermediate velocity u* from advection,
               diffusion, gravity, and body forces.
            2. Pressure solve: solve Poisson equation for pressure.
            3. Corrector: project u* onto divergence-free (or S_mass) space.

        Args:
            dt: Time step size [s].

        Returns:
            :class:`NavierStokesResult` with the updated fields.
        """
        if self.grid is None:
            raise RuntimeError("Call set_domain() before stepping.")

        nx, ny = self.grid.nx, self.grid.ny

        rho_u = self._rho_at_u_faces()
        rho_v = self._rho_at_v_faces()
        safe_rho_u = np.where(rho_u > 1e-30, rho_u, 1e-30)
        safe_rho_v = np.where(rho_v > 1e-30, rho_v, 1e-30)

        # ============================================================
        # STEP 1: Predictor
        # ============================================================

        # Advection (explicit Forward Euler)
        adv_u, adv_v = self._compute_advection()

        # Diffusion (explicit part)
        diff_u_exp, diff_v_exp = self._compute_diffusion_explicit()

        # Gravity at faces
        grav_u = self._gx * rho_u
        grav_v = self._gy * rho_v

        # Body forces interpolated to faces
        bf_u = np.zeros((nx + 1, ny))
        bf_v = np.zeros((nx, ny + 1))
        if nx > 1:
            bf_u[1:nx, :] = 0.5 * (self._body_fx[:nx - 1, :] + self._body_fx[1:nx, :])
        bf_u[0, :] = self._body_fx[0, :]
        bf_u[nx, :] = self._body_fx[nx - 1, :]
        if ny > 1:
            bf_v[:, 1:ny] = 0.5 * (self._body_fy[:, :ny - 1] + self._body_fy[:, 1:ny])
        bf_v[:, 0] = self._body_fy[:, 0]
        bf_v[:, ny] = self._body_fy[:, ny - 1]

        # Build predictor velocity
        u_star = self.u + dt * (
            adv_u / safe_rho_u
            + (1.0 - self.theta_diffusion) * diff_u_exp / safe_rho_u
            + grav_u / safe_rho_u
            + bf_u / safe_rho_u
        )

        v_star = self.v + dt * (
            adv_v / safe_rho_v
            + (1.0 - self.theta_diffusion) * diff_v_exp / safe_rho_v
            + grav_v / safe_rho_v
            + bf_v / safe_rho_v
        )

        self._apply_velocity_bcs(u_star, v_star)

        # Implicit diffusion solve (Crank-Nicolson)
        if self.theta_diffusion > 1e-14:
            u_star, v_star = self._apply_diffusion_implicit(u_star, v_star, dt)
            self._apply_velocity_bcs(u_star, v_star)

        # ============================================================
        # STEP 2: Pressure Poisson solve
        # ============================================================
        self.p = self._solve_pressure(u_star, v_star, dt)

        # ============================================================
        # STEP 3: Correction
        # ============================================================
        self.u, self.v = self._correct_velocity(u_star, v_star, self.p, dt)
        self._apply_velocity_bcs(self.u, self.v)

        # Update time
        self._time += dt

        # Reset transient forces and mass source
        self._body_fx[:] = 0.0
        self._body_fy[:] = 0.0
        self._mass_source[:] = 0.0

        return self._build_result()

    def solve(
        self,
        t_end: float,
        dt: float,
        callback: Optional[Callable[['NavierStokesResult', int], None]] = None,
        store_every: int = 0,
    ) -> NavierStokesResult:
        """Integrate from current time to *t_end*.

        Args:
            t_end: Target end time [s].
            dt: Time step size [s].
            callback: Optional function ``callback(result, step_number)`` called
                after every time step.  Can be used for diagnostics, adding
                time-varying forces, or adaptive property updates.
            store_every: If > 0, store field snapshots every *store_every*
                steps into the result history lists.

        Returns:
            :class:`NavierStokesResult` at the final time.
        """
        if self.grid is None:
            raise RuntimeError("Call set_domain() before solving.")

        time_history: List[float] = []
        u_history: List[np.ndarray] = []
        v_history: List[np.ndarray] = []
        p_history: List[np.ndarray] = []
        dt_history: List[float] = []

        step_count = 0
        while self._time < t_end - 1e-14 * dt:
            dt_actual = min(dt, t_end - self._time)
            result = self.step(dt_actual)
            step_count += 1
            dt_history.append(dt_actual)

            if store_every > 0 and step_count % store_every == 0:
                time_history.append(self._time)
                u_history.append(self.u.copy())
                v_history.append(self.v.copy())
                p_history.append(self.p.copy())

            if callback is not None:
                callback(result, step_count)

        result = self._build_result()
        result.time_history = time_history
        result.u_history = u_history
        result.v_history = v_history
        result.p_history = p_history
        result.dt_history = dt_history
        return result

    # ------------------------------------------------------------------
    # Internal: result builder
    # ------------------------------------------------------------------

    def _build_result(self) -> NavierStokesResult:
        """Construct a :class:`NavierStokesResult` from the current state."""
        return NavierStokesResult(
            u=self.u.copy(),
            v=self.v.copy(),
            p=self.p.copy(),
            rho=self.rho.copy(),
            mu=self.mu.copy(),
            time=self._time,
            grid=self.grid,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def compute_divergence(self) -> np.ndarray:
        """Return div(u) at cell centers for the current velocity field.

        Useful for verifying that the projection step enforced the
        divergence constraint to within solver tolerance.
        """
        return self._divergence_of_velocity(self.u, self.v)

    def cfl_number(self, dt: float) -> float:
        """Compute the maximum CFL number for the current velocity field.

        CFL = max( |u|/dx + |v|/dy ) * dt
        """
        result = self._build_result()
        return result.max_cfl(dt)
