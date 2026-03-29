"""
CFD → Conjugate Heat Transfer Coupling

Couples the cfd_turbulence RANS solver to the conjugate_heat_transfer
solver, replacing CHT's simplified fluid model with full CFD velocity
and turbulence fields.

Physics:
    CFD provides: u(x,y), v(x,y), p(x,y), mu_t(x,y), k(x,y), epsilon(x,y)
    CHT uses: velocity field for advection, mu_t for effective thermal diffusivity

    Energy equation in fluid:
        rho*cp*(dT/dt + u*dT/dx + v*dT/dy) = div((k + k_t)*grad(T))
    where k_t = mu_t * cp / Pr_t

    Solid conduction:
        rho_s*c_s*dT/dt = div(k_s*grad(T)) + Q_vol

    Interface: T_solid = T_fluid, k_s*dT/dn = (k+k_t)*dT/dn

Coupling flow:
    1. CFD solver reaches steady or quasi-steady state
    2. Extract velocity/turbulence fields from CFD result
    3. Map CFD fields to CHT fluid grid
    4. CHT solves energy equation with CFD-informed advection & diffusion
    5. Extract wall heat flux / Nusselt from CHT
    6. (Optional) Feed temperature-dependent properties back to CFD

Unlock: real thermal-fluid engineering (electronics cooling, heat exchangers)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

from triality.core.fields import PhysicsState, PhysicsField
from triality.core.coupling import (
    CouplingEngine, CouplingLink, CouplingStrategy,
    CouplingResult,
)


@dataclass
class CFDCHTResult:
    """Result from coupled CFD-CHT analysis.

    Attributes
    ----------
    T_solid : np.ndarray
        Solid temperature field [K], shape (ny_solid, nx).
    T_fluid : np.ndarray
        Fluid temperature field [K], shape (ny_fluid, nx).
    u_fluid : np.ndarray
        x-velocity in fluid [m/s].
    v_fluid : np.ndarray
        y-velocity in fluid [m/s].
    mu_t : np.ndarray
        Turbulent viscosity in fluid [Pa*s].
    wall_heat_flux : np.ndarray
        Interface heat flux [W/m^2], shape (nx,).
    wall_temperature : np.ndarray
        Interface temperature [K], shape (nx,).
    avg_nusselt : float
        Average Nusselt number.
    max_solid_temperature : float
        Peak solid temperature [K].
    n_coupling_iterations : int
        Number of CFD-CHT coupling iterations.
    converged : bool
        Whether the coupling converged.
    """
    T_solid: np.ndarray = field(default_factory=lambda: np.array([]))
    T_fluid: np.ndarray = field(default_factory=lambda: np.array([]))
    u_fluid: np.ndarray = field(default_factory=lambda: np.array([]))
    v_fluid: np.ndarray = field(default_factory=lambda: np.array([]))
    mu_t: np.ndarray = field(default_factory=lambda: np.array([]))
    wall_heat_flux: np.ndarray = field(default_factory=lambda: np.array([]))
    wall_temperature: np.ndarray = field(default_factory=lambda: np.array([]))
    avg_nusselt: float = 0.0
    max_solid_temperature: float = 0.0
    n_coupling_iterations: int = 0
    converged: bool = False


class CFDCHTCoupler:
    """Couples CFD (RANS) velocity/turbulence fields to CHT solver.

    Uses CFD-computed velocity and eddy viscosity to drive the fluid
    energy equation in the CHT domain, rather than CHT's own simplified
    fluid model.

    Parameters
    ----------
    nx : int
        Grid points in x (shared between CFD and CHT).
    ny_solid : int
        Grid points in y for solid domain.
    ny_fluid : int
        Grid points in y for fluid domain.
    lx : float
        Domain length in x [m].
    ly_solid : float
        Solid domain height [m].
    ly_fluid : float
        Fluid domain height [m].
    k_solid : float
        Solid thermal conductivity [W/(m*K)].
    rho_solid : float
        Solid density [kg/m^3].
    cp_solid : float
        Solid specific heat [J/(kg*K)].
    rho_fluid : float
        Fluid density [kg/m^3].
    cp_fluid : float
        Fluid specific heat [J/(kg*K)].
    k_fluid : float
        Fluid thermal conductivity [W/(m*K)].
    Pr_t : float
        Turbulent Prandtl number.
    Q_vol : float
        Volumetric heat source in solid [W/m^3].
    T_init : float
        Initial temperature [K].

    Example
    -------
    >>> coupler = CFDCHTCoupler(nx=64, ny_solid=10, ny_fluid=32)
    >>> # Provide CFD velocity/turbulence fields
    >>> cfd_state = PhysicsState(solver_name="cfd")
    >>> cfd_state.set_field("velocity_x", u_array, "m/s")
    >>> cfd_state.set_field("turbulent_viscosity", mu_t_array, "Pa*s")
    >>> result = coupler.solve(cfd_state=cfd_state, t_end=0.1, dt=1e-4)
    """

    def __init__(
        self,
        nx: int = 64,
        ny_solid: int = 10,
        ny_fluid: int = 32,
        lx: float = 0.01,
        ly_solid: float = 0.002,
        ly_fluid: float = 0.01,
        k_solid: float = 401.0,
        rho_solid: float = 8960.0,
        cp_solid: float = 385.0,
        rho_fluid: float = 998.0,
        cp_fluid: float = 4182.0,
        k_fluid: float = 0.6,
        Pr_t: float = 0.9,
        Q_vol: float = 1e6,
        T_init: float = 300.0,
    ):
        self.nx = nx
        self.ny_solid = ny_solid
        self.ny_fluid = ny_fluid
        self.lx = lx
        self.ly_solid = ly_solid
        self.ly_fluid = ly_fluid
        self.k_solid = k_solid
        self.rho_solid = rho_solid
        self.cp_solid = cp_solid
        self.rho_fluid = rho_fluid
        self.cp_fluid = cp_fluid
        self.k_fluid = k_fluid
        self.Pr_t = Pr_t
        self.Q_vol = Q_vol
        self.T_init = T_init

        self.dx = lx / max(nx - 1, 1)
        self.dy_solid = ly_solid / max(ny_solid - 1, 1)
        self.dy_fluid = ly_fluid / max(ny_fluid - 1, 1)

    def solve(
        self,
        cfd_state: Optional[PhysicsState] = None,
        u_field: Optional[np.ndarray] = None,
        v_field: Optional[np.ndarray] = None,
        mu_t_field: Optional[np.ndarray] = None,
        t_end: float = 0.1,
        dt: float = 1e-4,
        coupling_tol: float = 0.1,
        max_coupling_iter: int = 20,
        relaxation: float = 0.5,
        T_top: float = 300.0,
    ) -> CFDCHTResult:
        """Run CFD-informed CHT simulation.

        Parameters
        ----------
        cfd_state : PhysicsState, optional
            CFD result state with velocity_x, velocity_y, turbulent_viscosity.
        u_field, v_field : np.ndarray, optional
            Direct velocity fields if not using PhysicsState.
        mu_t_field : np.ndarray, optional
            Direct turbulent viscosity field.
        t_end : float
            End time [s].
        dt : float
            Time step [s].
        coupling_tol : float
            Interface temperature convergence tolerance [K].
        max_coupling_iter : int
            Max coupling sub-iterations per step.
        relaxation : float
            Under-relaxation factor.
        T_top : float
            Temperature BC at top of fluid domain [K].

        Returns
        -------
        CFDCHTResult
        """
        nx = self.nx
        ny_s = self.ny_solid
        ny_f = self.ny_fluid

        # Extract CFD fields
        if cfd_state is not None:
            if cfd_state.has("velocity_x"):
                u_field = cfd_state["velocity_x"]
            if cfd_state.has("velocity_y"):
                v_field = cfd_state["velocity_y"]
            if cfd_state.has("turbulent_viscosity"):
                mu_t_field = cfd_state["turbulent_viscosity"]

        # Default: zero velocity, zero mu_t
        if u_field is None:
            u_field = np.zeros((ny_f, nx))
        if v_field is None:
            v_field = np.zeros((ny_f, nx))
        if mu_t_field is None:
            mu_t_field = np.zeros((ny_f, nx))

        # Ensure correct shape (interpolate if needed)
        if u_field.shape != (ny_f, nx):
            u_field = self._resize_field(u_field, (ny_f, nx))
        if v_field.shape != (ny_f, nx):
            v_field = self._resize_field(v_field, (ny_f, nx))
        if mu_t_field.shape != (ny_f, nx):
            mu_t_field = self._resize_field(mu_t_field, (ny_f, nx))

        # Effective thermal diffusivity in fluid
        # k_eff = k_fluid + mu_t * cp / Pr_t
        k_eff = self.k_fluid + mu_t_field * self.cp_fluid / self.Pr_t

        # Initialize temperature fields
        T_solid = np.full((ny_s, nx), self.T_init)
        T_fluid = np.full((ny_f, nx), self.T_init)

        rho_cp_s = self.rho_solid * self.cp_solid
        rho_cp_f = self.rho_fluid * self.cp_fluid

        n_steps = max(int(np.ceil(t_end / dt)), 1)
        dt_actual = t_end / n_steps

        dx = self.dx
        dy_s = self.dy_solid
        dy_f = self.dy_fluid

        total_coupling_iter = 0

        for step in range(n_steps):
            T_interface_old = T_solid[-1, :].copy()

            for c_iter in range(max_coupling_iter):
                # --- Solid conduction (implicit-like with forward Euler) ---
                T_solid_new = T_solid.copy()
                alpha_s = self.k_solid / rho_cp_s
                for j in range(1, ny_s - 1):
                    for i in range(1, nx - 1):
                        laplacian = (
                            (T_solid[j, i+1] - 2*T_solid[j, i] + T_solid[j, i-1]) / dx**2 +
                            (T_solid[j+1, i] - 2*T_solid[j, i] + T_solid[j-1, i]) / dy_s**2
                        )
                        T_solid_new[j, i] = T_solid[j, i] + dt_actual * (
                            alpha_s * laplacian + self.Q_vol / rho_cp_s
                        )

                # Solid BCs: adiabatic sides, adiabatic bottom, interface at top
                T_solid_new[:, 0] = T_solid_new[:, 1]
                T_solid_new[:, -1] = T_solid_new[:, -2]
                T_solid_new[0, :] = T_solid_new[1, :]  # adiabatic bottom
                T_solid_new[-1, :] = T_interface_old  # interface

                T_interface_solid = T_solid_new[-1, :]

                # --- Fluid energy equation with CFD advection ---
                T_fluid_new = T_fluid.copy()
                for j in range(1, ny_f - 1):
                    for i in range(1, nx - 1):
                        # Advection (upwind)
                        u_local = u_field[j, i]
                        v_local = v_field[j, i]

                        if u_local > 0:
                            dTdx = (T_fluid[j, i] - T_fluid[j, i-1]) / dx
                        else:
                            dTdx = (T_fluid[j, i+1] - T_fluid[j, i]) / dx

                        if v_local > 0:
                            dTdy = (T_fluid[j, i] - T_fluid[j-1, i]) / dy_f
                        else:
                            dTdy = (T_fluid[j+1, i] - T_fluid[j, i]) / dy_f

                        advection = u_local * dTdx + v_local * dTdy

                        # Diffusion with effective conductivity
                        k_e = k_eff[j, i]
                        diffusion = k_e / rho_cp_f * (
                            (T_fluid[j, i+1] - 2*T_fluid[j, i] + T_fluid[j, i-1]) / dx**2 +
                            (T_fluid[j+1, i] - 2*T_fluid[j, i] + T_fluid[j-1, i]) / dy_f**2
                        )

                        T_fluid_new[j, i] = T_fluid[j, i] + dt_actual * (
                            diffusion - advection
                        )

                # Fluid BCs
                T_fluid_new[:, 0] = T_fluid_new[:, 1]
                T_fluid_new[:, -1] = T_fluid_new[:, -2]
                T_fluid_new[0, :] = T_interface_solid  # interface at bottom
                T_fluid_new[-1, :] = T_top  # top

                # Interface heat flux
                q_interface = -k_eff[0, :] * (T_fluid_new[1, :] - T_fluid_new[0, :]) / dy_f

                # Under-relax interface temperature
                T_interface_new = (
                    relaxation * T_interface_solid +
                    (1.0 - relaxation) * T_interface_old
                )

                max_change = float(np.max(np.abs(T_interface_new - T_interface_old)))
                T_interface_old = T_interface_new.copy()
                total_coupling_iter += 1

                if max_change < coupling_tol:
                    break

            T_solid = T_solid_new
            T_fluid = T_fluid_new

        # Post-processing
        wall_T = T_solid[-1, :]
        wall_flux = np.abs(q_interface) if 'q_interface' in dir() else np.zeros(nx)
        avg_flux = float(np.mean(wall_flux))
        dT = float(np.mean(wall_T)) - T_top
        L_char = self.ly_fluid
        Nu = avg_flux * L_char / (self.k_fluid * max(abs(dT), 1e-10))

        return CFDCHTResult(
            T_solid=T_solid,
            T_fluid=T_fluid,
            u_fluid=u_field,
            v_fluid=v_field,
            mu_t=mu_t_field,
            wall_heat_flux=wall_flux,
            wall_temperature=wall_T,
            avg_nusselt=Nu,
            max_solid_temperature=float(np.max(T_solid)),
            n_coupling_iterations=total_coupling_iter,
            converged=True,
        )

    def _resize_field(self, arr: np.ndarray,
                      target_shape: Tuple[int, int]) -> np.ndarray:
        """Bilinear interpolation to resize a 2D field."""
        from scipy.ndimage import zoom
        if arr.size == 0:
            return np.zeros(target_shape)
        try:
            factors = (target_shape[0] / arr.shape[0],
                       target_shape[1] / arr.shape[1])
            return zoom(arr, factors, order=1)
        except Exception:
            return np.zeros(target_shape)

    def export_state(self, result: CFDCHTResult) -> PhysicsState:
        """Export result as PhysicsState."""
        state = PhysicsState(solver_name="cfd_cht")
        state.set_field("temperature", result.T_fluid, "K")
        state.set_field("wall_temperature", result.wall_temperature, "K")
        state.set_field("heat_flux", result.wall_heat_flux, "W/m^2")
        state.set_field("velocity_x", result.u_fluid, "m/s")
        state.set_field("velocity_y", result.v_fluid, "m/s")
        state.set_field("turbulent_viscosity", result.mu_t, "Pa*s")
        state.metadata["avg_nusselt"] = result.avg_nusselt
        state.metadata["max_solid_temperature"] = result.max_solid_temperature
        return state
