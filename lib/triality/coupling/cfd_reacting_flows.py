"""
CFD Turbulence → Reacting Flows Coupling

Couples the cfd_turbulence RANS solver to the reacting_flows solver,
providing turbulent velocity and mixing fields that drive species transport
and combustion.

Physics:
    CFD → Reacting Flows:
        u(x,y), v(x,y) → advection velocity for species transport
        mu_t(x,y) → turbulent diffusivity: D_t = mu_t / (rho * Sc_t)
        k(x,y), epsilon(x,y) → turbulent time scale: tau_t = k / epsilon
            used for turbulence-chemistry interaction

    Reacting Flows → CFD (optional feedback):
        T(x,y) → density via equation of state: rho = p / (R_mix * T)
        Species → molecular weight, viscosity updates

    Turbulence-Chemistry Interaction:
        Damkohler number: Da = tau_t / tau_c
        Eddy dissipation model: reaction rate ~ min(Arrhenius, mixing-limited)

Coupling flow:
    1. CFD RANS solver → steady velocity/turbulence fields
    2. Map CFD fields to reacting flow grid
    3. Compute turbulent diffusivity from mu_t
    4. Run reacting flows with CFD-informed advection + diffusion
    5. (Optional) Feed temperature/density back to CFD

Unlock: turbulent combustion, pollutant transport, flame stabilization
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple

from triality.core.fields import PhysicsState, PhysicsField


@dataclass
class CFDReactingResult:
    """Result from coupled CFD-reacting flows analysis.

    Attributes
    ----------
    x : np.ndarray
        x-coordinates [m].
    y : np.ndarray
        y-coordinates [m].
    temperature : np.ndarray
        Temperature field [K], shape (ny, nx).
    species : Dict[str, np.ndarray]
        Species mass fraction fields, each shape (ny, nx).
    velocity_x : np.ndarray
        x-velocity from CFD [m/s].
    velocity_y : np.ndarray
        y-velocity from CFD [m/s].
    reaction_rate : np.ndarray
        Volumetric reaction rate [1/s].
    heat_release_rate : np.ndarray
        Volumetric heat release [W/m^3].
    damkohler_number : np.ndarray
        Damkohler number field.
    turbulent_diffusivity : np.ndarray
        Turbulent species diffusivity [m^2/s].
    max_temperature : float
        Peak temperature [K].
    avg_heat_release : float
        Volume-averaged heat release rate [W/m^3].
    n_time_steps : int
        Number of time steps taken.
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature: np.ndarray = field(default_factory=lambda: np.array([]))
    species: Dict[str, np.ndarray] = field(default_factory=dict)
    velocity_x: np.ndarray = field(default_factory=lambda: np.array([]))
    velocity_y: np.ndarray = field(default_factory=lambda: np.array([]))
    reaction_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    heat_release_rate: np.ndarray = field(default_factory=lambda: np.array([]))
    damkohler_number: np.ndarray = field(default_factory=lambda: np.array([]))
    turbulent_diffusivity: np.ndarray = field(default_factory=lambda: np.array([]))
    max_temperature: float = 0.0
    avg_heat_release: float = 0.0
    n_time_steps: int = 0


class CFDReactingFlowsCoupler:
    """Couples CFD turbulence fields to reacting flows solver.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions.
    lx, ly : float
        Domain size [m].
    rho : float
        Density [kg/m^3].
    cp : float
        Specific heat [J/(kg*K)].
    k_thermal : float
        Molecular thermal conductivity [W/(m*K)].
    D_mol : float
        Molecular species diffusivity [m^2/s].
    Sc_t : float
        Turbulent Schmidt number.
    Pr_t : float
        Turbulent Prandtl number.
    A_arrhenius : float
        Arrhenius pre-exponential factor [1/s].
    Ea : float
        Activation energy [J/mol].
    Q_rxn : float
        Heat of reaction [J/kg].
    T_init : float
        Initial temperature [K].
    fuel_init : float
        Initial fuel mass fraction.

    Example
    -------
    >>> coupler = CFDReactingFlowsCoupler(nx=64, ny=32)
    >>> # Get CFD fields
    >>> cfd_state = PhysicsState(solver_name="cfd")
    >>> cfd_state.set_field("velocity_x", u_arr, "m/s")
    >>> cfd_state.set_field("turbulent_viscosity", mu_t_arr, "Pa*s")
    >>> result = coupler.solve(cfd_state=cfd_state, t_end=0.01)
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 32,
        lx: float = 1.0,
        ly: float = 0.5,
        rho: float = 1.2,
        cp: float = 1000.0,
        k_thermal: float = 0.025,
        D_mol: float = 2e-5,
        Sc_t: float = 0.7,
        Pr_t: float = 0.85,
        A_arrhenius: float = 1e10,
        Ea: float = 80000.0,
        Q_rxn: float = 2.5e6,
        T_init: float = 300.0,
        fuel_init: float = 0.06,
    ):
        self.nx, self.ny = nx, ny
        self.lx, self.ly = lx, ly
        self.dx = lx / max(nx - 1, 1)
        self.dy = ly / max(ny - 1, 1)
        self.rho = rho
        self.cp = cp
        self.k_thermal = k_thermal
        self.D_mol = D_mol
        self.Sc_t = Sc_t
        self.Pr_t = Pr_t
        self.A = A_arrhenius
        self.Ea = Ea
        self.Q_rxn = Q_rxn
        self.T_init = T_init
        self.fuel_init = fuel_init
        self.R_universal = 8.314  # J/(mol*K)

    def solve(
        self,
        cfd_state: Optional[PhysicsState] = None,
        u_field: Optional[np.ndarray] = None,
        v_field: Optional[np.ndarray] = None,
        mu_t_field: Optional[np.ndarray] = None,
        k_field: Optional[np.ndarray] = None,
        eps_field: Optional[np.ndarray] = None,
        t_end: float = 0.01,
        dt: Optional[float] = None,
        T_inlet: float = 300.0,
        fuel_inlet: float = 0.06,
        ignition_zone: Optional[Tuple[float, float, float, float]] = None,
    ) -> CFDReactingResult:
        """Run coupled CFD-reacting flows simulation.

        Parameters
        ----------
        cfd_state : PhysicsState, optional
            CFD result with velocity/turbulence fields.
        u_field, v_field, mu_t_field, k_field, eps_field : np.ndarray, optional
            Direct field inputs.
        t_end : float
            Simulation time [s].
        dt : float, optional
            Time step (auto-computed from CFL if None).
        T_inlet : float
            Inlet temperature [K].
        fuel_inlet : float
            Inlet fuel mass fraction.
        ignition_zone : tuple, optional
            (x_min, x_max, y_min, y_max) for initial hot zone.

        Returns
        -------
        CFDReactingResult
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        # Extract CFD fields
        if cfd_state is not None:
            if cfd_state.has("velocity_x"):
                u_field = cfd_state["velocity_x"]
            if cfd_state.has("velocity_y"):
                v_field = cfd_state["velocity_y"]
            if cfd_state.has("turbulent_viscosity"):
                mu_t_field = cfd_state["turbulent_viscosity"]
            if cfd_state.has("turbulent_kinetic_energy"):
                k_field = cfd_state["turbulent_kinetic_energy"]
            if cfd_state.has("turbulent_dissipation"):
                eps_field = cfd_state["turbulent_dissipation"]

        if u_field is None:
            u_field = np.ones((ny, nx)) * 1.0
        if v_field is None:
            v_field = np.zeros((ny, nx))
        if mu_t_field is None:
            mu_t_field = np.ones((ny, nx)) * 1e-3

        # Resize if needed
        if u_field.shape != (ny, nx):
            u_field = np.resize(u_field, (ny, nx))
        if v_field.shape != (ny, nx):
            v_field = np.resize(v_field, (ny, nx))
        if mu_t_field.shape != (ny, nx):
            mu_t_field = np.resize(mu_t_field, (ny, nx))

        # Turbulent diffusivity
        D_t = mu_t_field / (self.rho * self.Sc_t)
        D_eff = self.D_mol + D_t

        # Effective thermal diffusivity
        k_t = mu_t_field * self.cp / self.Pr_t
        alpha_eff = (self.k_thermal + k_t) / (self.rho * self.cp)

        # Turbulent time scale for Da number
        if k_field is not None and eps_field is not None:
            if k_field.shape != (ny, nx):
                k_field = np.resize(k_field, (ny, nx))
            if eps_field.shape != (ny, nx):
                eps_field = np.resize(eps_field, (ny, nx))
            tau_t = np.clip(k_field, 1e-10, None) / np.clip(eps_field, 1e-10, None)
        else:
            tau_t = np.ones((ny, nx)) * 1e-3

        # CFL-based time step
        u_max = max(float(np.max(np.abs(u_field))), 0.01)
        v_max = max(float(np.max(np.abs(v_field))), 0.01)
        D_max = max(float(np.max(D_eff)), 1e-10)

        dt_adv = 0.5 * min(dx / u_max, dy / v_max)
        dt_diff = 0.25 * min(dx**2, dy**2) / D_max
        if dt is None:
            dt = 0.5 * min(dt_adv, dt_diff)

        n_steps = max(int(np.ceil(t_end / dt)), 1)
        dt = t_end / n_steps

        # Initialize fields
        T = np.full((ny, nx), self.T_init)
        Y_fuel = np.full((ny, nx), self.fuel_init)
        Y_prod = np.zeros((ny, nx))

        # Ignition zone
        if ignition_zone is not None:
            x_coords = np.linspace(0, self.lx, nx)
            y_coords = np.linspace(0, self.ly, ny)
            xmin, xmax, ymin, ymax = ignition_zone
            for j in range(ny):
                for i in range(nx):
                    if xmin <= x_coords[i] <= xmax and ymin <= y_coords[j] <= ymax:
                        T[j, i] = 1500.0

        # Time integration
        for step in range(n_steps):
            T_new = T.copy()
            Y_fuel_new = Y_fuel.copy()

            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    u = u_field[j, i]
                    v = v_field[j, i]

                    # Upwind advection for T
                    if u > 0:
                        dTdx = (T[j, i] - T[j, i-1]) / dx
                        dYdx = (Y_fuel[j, i] - Y_fuel[j, i-1]) / dx
                    else:
                        dTdx = (T[j, i+1] - T[j, i]) / dx
                        dYdx = (Y_fuel[j, i+1] - Y_fuel[j, i]) / dx

                    if v > 0:
                        dTdy = (T[j, i] - T[j-1, i]) / dy
                        dYdy = (Y_fuel[j, i] - Y_fuel[j-1, i]) / dy
                    else:
                        dTdy = (T[j+1, i] - T[j, i]) / dy
                        dYdy = (Y_fuel[j+1, i] - Y_fuel[j, i]) / dy

                    # Diffusion
                    diff_T = alpha_eff[j, i] * (
                        (T[j, i+1] - 2*T[j, i] + T[j, i-1]) / dx**2 +
                        (T[j+1, i] - 2*T[j, i] + T[j-1, i]) / dy**2
                    )
                    diff_Y = D_eff[j, i] * (
                        (Y_fuel[j, i+1] - 2*Y_fuel[j, i] + Y_fuel[j, i-1]) / dx**2 +
                        (Y_fuel[j+1, i] - 2*Y_fuel[j, i] + Y_fuel[j-1, i]) / dy**2
                    )

                    # Arrhenius reaction rate
                    T_local = max(T[j, i], 300.0)
                    omega = self.A * Y_fuel[j, i] * self.rho * np.exp(
                        -self.Ea / (self.R_universal * T_local)
                    )
                    omega = max(omega, 0.0)

                    # Eddy dissipation limit (mixing-limited)
                    omega_edc = self.rho * Y_fuel[j, i] / max(tau_t[j, i], 1e-10)
                    omega_eff = min(omega, omega_edc)

                    # Update
                    T_new[j, i] = T[j, i] + dt * (
                        diff_T - u * dTdx - v * dTdy +
                        omega_eff * self.Q_rxn / (self.rho * self.cp)
                    )
                    Y_fuel_new[j, i] = Y_fuel[j, i] + dt * (
                        diff_Y - u * dYdx - v * dYdy - omega_eff / self.rho
                    )

            # Clamp
            T_new = np.clip(T_new, 200.0, 5000.0)
            Y_fuel_new = np.clip(Y_fuel_new, 0.0, 1.0)

            # BCs
            T_new[:, 0] = T_inlet
            Y_fuel_new[:, 0] = fuel_inlet
            T_new[0, :] = T_new[1, :]
            T_new[-1, :] = T_new[-2, :]
            Y_fuel_new[0, :] = Y_fuel_new[1, :]
            Y_fuel_new[-1, :] = Y_fuel_new[-2, :]

            T = T_new
            Y_fuel = Y_fuel_new

        # Compute output fields
        reaction_rate = np.zeros((ny, nx))
        heat_release = np.zeros((ny, nx))
        Da = np.zeros((ny, nx))
        for j in range(ny):
            for i in range(nx):
                T_local = max(T[j, i], 300.0)
                omega = self.A * Y_fuel[j, i] * self.rho * np.exp(
                    -self.Ea / (self.R_universal * T_local)
                )
                tau_c = 1.0 / max(omega / max(self.rho * Y_fuel[j, i], 1e-30), 1e-30)
                reaction_rate[j, i] = omega
                heat_release[j, i] = omega * self.Q_rxn
                Da[j, i] = tau_t[j, i] / max(tau_c, 1e-30)

        Y_prod = 1.0 - Y_fuel - (1.0 - self.fuel_init)

        x_coords = np.linspace(0, self.lx, nx)
        y_coords = np.linspace(0, self.ly, ny)

        return CFDReactingResult(
            x=x_coords,
            y=y_coords,
            temperature=T,
            species={"fuel": Y_fuel, "product": np.clip(Y_prod, 0, 1)},
            velocity_x=u_field,
            velocity_y=v_field,
            reaction_rate=reaction_rate,
            heat_release_rate=heat_release,
            damkohler_number=Da,
            turbulent_diffusivity=D_t,
            max_temperature=float(np.max(T)),
            avg_heat_release=float(np.mean(heat_release)),
            n_time_steps=n_steps,
        )

    def export_state(self, result: CFDReactingResult) -> PhysicsState:
        """Export as PhysicsState."""
        state = PhysicsState(solver_name="cfd_reacting_flows")
        state.set_field("temperature", result.temperature, "K")
        state.set_field("velocity_x", result.velocity_x, "m/s")
        state.set_field("velocity_y", result.velocity_y, "m/s")
        state.set_field("species_mass_fraction", result.species.get("fuel", np.array([])), "1")
        state.set_field("heat_source", result.heat_release_rate, "W/m^3")
        state.set_field("reaction_rate", result.reaction_rate, "Hz")
        return state
