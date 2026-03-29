"""
Spacecraft Thermal System Solver

Integrated transient thermal solver for multi-node spacecraft thermal
analysis.  Wires together the existing radiative transfer, heat pipe,
thermal loop, heater control, spacecraft environment, and nonlinear
radiative solver modules into a single simulation framework.

Each node obeys the lumped-capacitance energy balance:

    m_i * c_i * dT_i/dt = Q_solar,i + Q_albedo,i + Q_IR,i
                          + Q_heater,i + Q_internal,i
                          - Q_rad,i(T)
                          + sum_j G_cond,ij * (T_j - T_i)
                          + sum_hp Q_hp(T_evap, T_cond)

where the radiative term is nonlinear (sigma * T^4) and is solved
iteratively at each time step.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable

from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from triality.spacecraft_thermal.radiative_transfer import (
    ViewFactorCalculator,
    RadiativeExchange,
    BlackBodyRadiation,
    Surface,
)
from triality.spacecraft_thermal.heat_pipes import (
    HeatPipe,
    HeatPipeRegime,
)
from triality.spacecraft_thermal.thermal_loops import (
    PumpedLoop,
    HeatExchanger,
    NTU_Method,
)
from triality.spacecraft_thermal.spacecraft_environment import (
    SpacecraftEnvironment,
    Orbit,
    SolarFlux,
    PlanetaryIR,
)
from triality.spacecraft_thermal.heater_control import (
    HeaterController,
    ControlMode,
    PIDController,
)
from triality.spacecraft_thermal.nonlinear_radiative_solver import (
    IterativeRadiositySolver,
    RadiativeSurface,
    CoupledRadiationConductionSolver,
    ConvergenceHistory,
)

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m^2*K^4)]


@dataclass
class ThermalNode:
    """A lumped thermal node on the spacecraft.

    Parameters
    ----------
    name : str
        Node label (e.g. "+X_panel", "electronics_box").
    mass : float
        Thermal mass [kg].
    specific_heat : float
        Specific heat [J/(kg*K)].
    surface : Surface
        Radiating surface properties.
    internal_power : float
        Internal dissipation [W] (electronics, payloads).
    heater : HeaterController or None
        Attached heater controller.
    T_init : float
        Initial temperature [K].
    """
    name: str
    mass: float
    specific_heat: float
    surface: Surface
    internal_power: float = 0.0
    heater: Optional[HeaterController] = None
    T_init: float = 293.0


@dataclass
class ConductiveLink:
    """Conductive coupling between two thermal nodes.

    Parameters
    ----------
    node_a : int
        Index of first node.
    node_b : int
        Index of second node.
    conductance : float
        Thermal conductance G = k*A/L [W/K].
    """
    node_a: int
    node_b: int
    conductance: float


@dataclass
class HeatPipeLink:
    """Heat pipe connection between an evaporator node and a condenser node.

    Parameters
    ----------
    evaporator_node : int
        Index of the hot (evaporator) node.
    condenser_node : int
        Index of the cold (condenser) node.
    heat_pipe : HeatPipe
        HeatPipe model instance.
    max_power : float
        Maximum heat transport [W] (capillary limit).
    """
    evaporator_node: int
    condenser_node: int
    heat_pipe: HeatPipe
    max_power: float = 100.0


@dataclass
class SpacecraftThermalResult:
    """Results from the spacecraft thermal simulation.

    Attributes
    ----------
    times : np.ndarray
        Time vector [s].
    temperatures : np.ndarray
        Node temperatures vs time [K], shape (n_times, n_nodes).
    heat_fluxes : dict
        Per-node breakdown of heat fluxes at final time [W].
    heater_duty_cycles : np.ndarray
        Heater power vs time [W], shape (n_times, n_nodes).
    radiative_heat_loss : np.ndarray
        Radiation to space per node vs time [W], shape (n_times, n_nodes).
    max_temperature : float
        Peak temperature [K].
    min_temperature : float
        Minimum temperature [K].
    thermal_margins : Dict[str, Tuple[float, float]]
        (T_min_margin, T_max_margin) for each node relative to typical
        spacecraft limits [-40 degC, +60 degC].
    converged : bool
        True if the nonlinear radiation iteration converged every step.
    node_names : list of str
        Node names.
    """
    times: np.ndarray
    temperatures: np.ndarray
    heat_fluxes: Dict
    heater_duty_cycles: np.ndarray
    radiative_heat_loss: np.ndarray
    max_temperature: float
    min_temperature: float
    thermal_margins: Dict[str, Tuple[float, float]]
    converged: bool
    node_names: List[str]


@dataclass
class Panel2D:
    """Definition of a spacecraft panel for 2-D thermal analysis.

    Parameters
    ----------
    name : str
        Panel identifier (e.g. "+X", "-Z").
    Lx : float
        Panel dimension in x [m].
    Ly : float
        Panel dimension in y [m].
    nx : int
        Number of grid points in x.
    ny : int
        Number of grid points in y.
    thickness : float
        Panel thickness [m].
    density : float
        Panel material density [kg/m^3].
    specific_heat : float
        Specific heat [J/(kg*K)].
    conductivity : float
        Thermal conductivity [W/(m*K)].
    emissivity : float
        Surface emissivity for radiation.
    absorptivity : float
        Solar absorptivity.
    T_init : float
        Initial uniform temperature [K].
    q_internal : Optional[np.ndarray]
        Internal heat generation [W/m^2], shape (nx, ny). None = 0.
    heater_mask : Optional[np.ndarray]
        Boolean mask of heater coverage, shape (nx, ny).
    heater_power_density : float
        Heater power density [W/m^2] when active.
    heater_setpoint : float
        Heater ON below this temperature [K].
    """
    name: str
    Lx: float = 1.0
    Ly: float = 1.0
    nx: int = 20
    ny: int = 20
    thickness: float = 0.003
    density: float = 2700.0
    specific_heat: float = 900.0
    conductivity: float = 167.0
    emissivity: float = 0.85
    absorptivity: float = 0.3
    T_init: float = 293.0
    q_internal: Optional[np.ndarray] = None
    heater_mask: Optional[np.ndarray] = None
    heater_power_density: float = 50.0
    heater_setpoint: float = 263.0  # -10 degC


@dataclass
class SpacecraftThermal2DResult:
    """Results from the 2-D per-panel transient spacecraft thermal simulation.

    Attributes
    ----------
    panel_names : list of str
        Names of the panels.
    times : np.ndarray
        Recorded time points [s], shape (n_t,).
    temperatures : dict
        Mapping panel_name -> np.ndarray of shape (n_t, nx, ny) giving
        the temperature field at each recorded time.
    max_temperature : float
        Peak temperature anywhere/anytime [K].
    min_temperature : float
        Minimum temperature anywhere/anytime [K].
    thermal_gradients_per_panel : dict
        Mapping panel_name -> np.ndarray of shape (n_t,) giving the
        maximum spatial thermal gradient [K/m] at each recorded time.
    total_heater_energy_J : float
        Total energy delivered by heaters over the simulation [J].
    converged : bool
        Whether the Newton-Raphson iteration converged every step.
    """
    panel_names: List[str]
    times: np.ndarray
    temperatures: Dict[str, np.ndarray]
    max_temperature: float
    min_temperature: float
    thermal_gradients_per_panel: Dict[str, np.ndarray]
    total_heater_energy_J: float
    converged: bool = True


class SpacecraftThermalSolver:
    """Integrated transient spacecraft thermal solver.

    Combines radiative exchange, conduction, heat pipes, environmental
    heating, and heater control into a unified time-stepping framework.

    Parameters
    ----------
    nodes : list of ThermalNode
        Spacecraft thermal nodes.
    conductive_links : list of ConductiveLink
        Conductive couplings between nodes.
    heat_pipe_links : list of HeatPipeLink
        Heat pipe connections.
    environment : SpacecraftEnvironment
        Orbital environment model.
    view_factors : np.ndarray or None
        Inter-node radiative view factors (n_nodes x n_nodes).
        If None, nodes only radiate to deep space (no inter-node exchange).
    T_space : float
        Deep space sink temperature [K].
    T_min_limit : float
        Lower survival temperature limit [K].
    T_max_limit : float
        Upper survival temperature limit [K].
    """

    fidelity_tier = FidelityTier.REDUCED_ORDER
    coupling_maturity = CouplingMaturity.M3_COUPLED

    def __init__(
        self,
        nodes: List[ThermalNode],
        conductive_links: List[ConductiveLink],
        heat_pipe_links: Optional[List[HeatPipeLink]] = None,
        environment: Optional[SpacecraftEnvironment] = None,
        view_factors: Optional[np.ndarray] = None,
        T_space: float = 4.0,
        T_min_limit: float = 233.0,   # -40 degC
        T_max_limit: float = 333.0,   # +60 degC
    ):
        self._coupled_state = None
        self._time = 0.0
        self.nodes = list(nodes)
        self.n_nodes = len(nodes)
        self.conductive_links = list(conductive_links)
        self.heat_pipe_links = list(heat_pipe_links or [])
        self.environment = environment or SpacecraftEnvironment(orbit=Orbit.LEO)
        self.view_factors = view_factors
        self.T_space = T_space
        self.T_min_limit = T_min_limit
        self.T_max_limit = T_max_limit

        # Thermal capacitances [J/K]
        self._C = np.array([n.mass * n.specific_heat for n in nodes])

        # Conductance matrix
        self._G = np.zeros((self.n_nodes, self.n_nodes))
        for link in self.conductive_links:
            self._G[link.node_a, link.node_b] += link.conductance
            self._G[link.node_b, link.node_a] += link.conductance

    def _environmental_heating(self, t: float, sun_angle: float = 0.0) -> np.ndarray:
        """Compute environmental heat loads per node [W].

        Includes direct solar, albedo (simplified), and planetary IR.
        """
        Q_env = np.zeros(self.n_nodes)
        for i, node in enumerate(self.nodes):
            A = node.surface.area
            # Solar heating on sun-facing surfaces
            Q_solar = self.environment.solar_heating(A, sun_angle)
            # Planetary IR (simplified: half the nodes see the planet)
            if self.environment.orbit in (Orbit.LEO, Orbit.GEO):
                planet_ir = PlanetaryIR(planet_temp=255.0, view_factor=0.3)
                Q_ir = planet_ir.heat_flux() * A * node.surface.emissivity
            else:
                Q_ir = 0.0
            Q_env[i] = Q_solar + Q_ir
        return Q_env

    def _radiation_to_space(self, T: np.ndarray) -> np.ndarray:
        """Radiative heat loss to deep space per node [W].

        Q_rad,i = eps_i * sigma * A_i * (T_i^4 - T_space^4)
        """
        Q_rad = np.zeros(self.n_nodes)
        for i, node in enumerate(self.nodes):
            eps = node.surface.emissivity
            A = node.surface.area
            Q_rad[i] = eps * SIGMA * A * (T[i] ** 4 - self.T_space ** 4)
        return Q_rad

    def _inter_node_radiation(self, T: np.ndarray) -> np.ndarray:
        """Net radiative exchange between nodes [W] (positive = heat gain).

        Uses view factor matrix if available.
        """
        Q_rad_net = np.zeros(self.n_nodes)
        if self.view_factors is None:
            return Q_rad_net

        F = self.view_factors
        for i in range(self.n_nodes):
            eps_i = self.nodes[i].surface.emissivity
            A_i = self.nodes[i].surface.area
            for j in range(self.n_nodes):
                if i == j:
                    continue
                eps_j = self.nodes[j].surface.emissivity
                # Simplified gray-body exchange
                eps_eff = 1.0 / (1.0 / eps_i + 1.0 / eps_j - 1.0) if (eps_i < 0.999 or eps_j < 0.999) else 1.0
                q_ij = eps_eff * SIGMA * A_i * F[i, j] * (T[j] ** 4 - T[i] ** 4)
                Q_rad_net[i] += q_ij
        return Q_rad_net

    def _conduction(self, T: np.ndarray) -> np.ndarray:
        """Net conductive heat gain per node [W]."""
        return self._G.dot(T) - np.sum(self._G, axis=1) * T

    def _heat_pipe_transport(self, T: np.ndarray) -> np.ndarray:
        """Net heat gain per node from heat pipes [W]."""
        Q_hp = np.zeros(self.n_nodes)
        for link in self.heat_pipe_links:
            T_evap = T[link.evaporator_node]
            T_cond = T[link.condenser_node]
            if T_evap <= T_cond:
                continue  # Heat pipes are diodes; no reverse flow
            # Effective conductance from heat pipe
            k_eff = link.heat_pipe.effective_conductivity(0.5 * (T_evap + T_cond))
            A_hp = np.pi * (link.heat_pipe.diameter / 2) ** 2
            L_hp = link.heat_pipe.length
            Q = k_eff * A_hp / L_hp * (T_evap - T_cond)
            Q = min(Q, link.max_power)  # Cap at capillary limit
            Q_hp[link.evaporator_node] -= Q
            Q_hp[link.condenser_node] += Q
        return Q_hp

    def _heater_power(self, T: np.ndarray, dt: float) -> np.ndarray:
        """Heater power per node [W]."""
        Q_htr = np.zeros(self.n_nodes)
        for i, node in enumerate(self.nodes):
            if node.heater is not None:
                Q_htr[i] = node.heater.control_output(T[i], dt)
        return Q_htr

    def _internal_dissipation(self) -> np.ndarray:
        """Internal power dissipation per node [W]."""
        return np.array([n.internal_power for n in self.nodes])

    def solve(
        self,
        t_end: float,
        dt: float = 1.0,
        sun_angle_func=None,
        progress_callback=None,
    ) -> SpacecraftThermalResult:
        """Run the transient spacecraft thermal simulation.

        Parameters
        ----------
        t_end : float
            End time [s].
        dt : float
            Time step [s].
        sun_angle_func : callable or None
            Function sun_angle_func(t) -> float returning the sun angle
            [rad] as a function of time.  Constant 0 if None.

        Returns
        -------
        SpacecraftThermalResult
        """
        n_steps = max(int(np.ceil(t_end / dt)), 1)
        dt_actual = t_end / n_steps

        T = np.array([n.T_init for n in self.nodes], dtype=float)

        times = np.zeros(n_steps + 1)
        T_hist = np.zeros((n_steps + 1, self.n_nodes))
        Q_htr_hist = np.zeros((n_steps + 1, self.n_nodes))
        Q_rad_hist = np.zeros((n_steps + 1, self.n_nodes))

        T_hist[0] = T.copy()
        all_converged = True

        _prog_interval = max(n_steps // 50, 1)
        for step in range(1, n_steps + 1):
            if progress_callback and step % _prog_interval == 0:
                progress_callback(step, n_steps)
            t = (step - 1) * dt_actual
            sun_angle = sun_angle_func(t) if sun_angle_func else 0.0

            # Heat loads
            Q_env = self._environmental_heating(t, sun_angle)
            Q_int = self._internal_dissipation()
            Q_htr = self._heater_power(T, dt_actual)
            Q_cond = self._conduction(T)
            Q_hp = self._heat_pipe_transport(T)
            Q_rad_space = self._radiation_to_space(T)
            Q_rad_inter = self._inter_node_radiation(T)

            # Energy balance: C * dT/dt = Q_in - Q_out
            Q_net = (
                Q_env + Q_int + Q_htr + Q_cond + Q_hp
                + Q_rad_inter - Q_rad_space
            )

            dTdt = Q_net / self._C
            T = T + dTdt * dt_actual

            times[step] = step * dt_actual
            T_hist[step] = T.copy()
            Q_htr_hist[step] = Q_htr
            Q_rad_hist[step] = Q_rad_space

        # Post-processing -------------------------------------------------
        max_T = float(np.max(T_hist))
        min_T = float(np.min(T_hist))

        margins = {}
        for i, node in enumerate(self.nodes):
            T_final = T_hist[-1, i]
            margins[node.name] = (
                T_final - self.T_min_limit,
                self.T_max_limit - T_final,
            )

        # Heat flux breakdown at final time
        t_final = t_end
        sun_angle_final = sun_angle_func(t_final) if sun_angle_func else 0.0
        heat_fluxes = {
            "environmental": self._environmental_heating(t_final, sun_angle_final),
            "internal": self._internal_dissipation(),
            "heater": Q_htr_hist[-1],
            "radiation_to_space": Q_rad_hist[-1],
        }

        return SpacecraftThermalResult(
            times=times,
            temperatures=T_hist,
            heat_fluxes=heat_fluxes,
            heater_duty_cycles=Q_htr_hist,
            radiative_heat_loss=Q_rad_hist,
            max_temperature=max_T,
            min_temperature=min_T,
            thermal_margins=margins,
            converged=all_converged,
            node_names=[n.name for n in self.nodes],
        )

    def export_state(self, result: SpacecraftThermalResult) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="spacecraft_thermal")
        state.set_field("temperature", result.temperatures[-1], "K")
        state.set_field("heat_flux", result.radiative_heat_loss[-1], "W/m^2")
        state.metadata["max_temperature"] = result.max_temperature
        state.metadata["min_temperature"] = result.min_temperature
        state.metadata["converged"] = result.converged
        state.metadata["node_names"] = result.node_names
        return state

    def import_state(self, state: PhysicsState) -> None:
        """Import coupled fields from partner solvers."""
        self._coupled_state = state

    def advance(self, dt: float) -> PhysicsState:
        """Advance spacecraft thermal solver by dt for closed-loop coupling."""
        if self._coupled_state is not None:
            if self._coupled_state.has("heat_flux"):
                q_ext = self._coupled_state.get("heat_flux")
                for i, node in enumerate(self.nodes):
                    if i < len(q_ext):
                        node.internal_power += float(q_ext[i])
        result = self.solve(t_end=dt, dt=min(dt, 1.0))
        if self._coupled_state is not None and self._coupled_state.has("heat_flux"):
            q_ext = self._coupled_state.get("heat_flux")
            for i, node in enumerate(self.nodes):
                if i < len(q_ext):
                    node.internal_power -= float(q_ext[i])
        self._time += dt
        return self.export_state(result)

    # ==================================================================
    # Level 3: 2-D per-panel transient thermal with view-factor
    # radiation coupling and orbital transient
    # (implicit Euler + Newton-Raphson for T^4 nonlinearity)
    # ==================================================================

    def solve_2d(
        self,
        panels: List[Panel2D],
        t_end: float,
        dt: float = 1.0,
        sun_angle_func: Optional[Callable[[float], float]] = None,
        panel_view_factors: Optional[np.ndarray] = None,
        record_interval: int = 10,
        newton_tol: float = 1e-3,
        newton_max_iter: int = 20,
        solar_constant: float = 1361.0,
    ) -> SpacecraftThermal2DResult:
        """Run a 2-D per-panel transient thermal simulation (Level 3).

        Each panel is discretised on an (nx, ny) 2-D grid and the
        transient heat equation is solved with implicit Euler.  The
        nonlinear radiative emission term (eps*sigma*T^4) is handled via
        Newton-Raphson linearisation at each time step.

        Inter-panel coupling uses a radiative view-factor matrix with
        iterative update for the nonlinear T^4 terms.  Solar flux varies
        with orbit position via ``sun_angle_func(t)`` and has spatial
        variation across each panel surface.

        Parameters
        ----------
        panels : list of Panel2D
            Panel geometry / material / heater definitions.
        t_end : float
            Simulation end time [s].
        dt : float
            Time step [s].
        sun_angle_func : callable or None
            Returns the sun angle [rad] as a function of time.  Each
            panel is offset by ``panel_index * pi / n_panels`` to model
            different spacecraft facet orientations.  None -> constant 0.
        panel_view_factors : np.ndarray or None
            (n_panels, n_panels) view-factor matrix F[i,j].  None means
            no inter-panel radiative exchange.
        record_interval : int
            Save full 2-D fields every this many steps.
        newton_tol : float
            Max absolute temperature correction [K] for NR convergence.
        newton_max_iter : int
            Maximum Newton-Raphson iterations per time step.
        solar_constant : float
            Solar irradiance [W/m^2].

        Returns
        -------
        SpacecraftThermal2DResult
        """
        if sun_angle_func is None:
            sun_angle_func = lambda t: 0.0

        n_panels = len(panels)
        n_steps = max(int(np.ceil(t_end / dt)), 1)
        dt_actual = t_end / n_steps
        n_records = n_steps // record_interval + 2

        T_space = self.T_space

        # ---- Pre-build per-panel sparse Laplacian & state vectors ----
        panel_state = []
        for ip, pan in enumerate(panels):
            nx, ny = pan.nx, pan.ny
            N = nx * ny
            dx = pan.Lx / max(nx - 1, 1)
            dy = pan.Ly / max(ny - 1, 1)
            rho_cp_h = pan.density * pan.specific_heat * pan.thickness

            # 2-D conduction coefficients [W/m^2 per K] through panel
            kx = pan.conductivity * pan.thickness / (dx ** 2)
            ky = pan.conductivity * pan.thickness / (dy ** 2)

            # Build sparse Laplacian (5-point stencil, Neumann BCs)
            def _idx(i, j, _ny=ny):
                return i * _ny + j

            rows, cols, vals = [], [], []
            for i in range(nx):
                for j in range(ny):
                    n_id = _idx(i, j)
                    diag = 0.0
                    if i > 0:
                        rows.append(n_id); cols.append(_idx(i - 1, j))
                        vals.append(kx); diag -= kx
                    if i < nx - 1:
                        rows.append(n_id); cols.append(_idx(i + 1, j))
                        vals.append(kx); diag -= kx
                    if j > 0:
                        rows.append(n_id); cols.append(_idx(i, j - 1))
                        vals.append(ky); diag -= ky
                    if j < ny - 1:
                        rows.append(n_id); cols.append(_idx(i, j + 1))
                        vals.append(ky); diag -= ky
                    rows.append(n_id); cols.append(n_id)
                    vals.append(diag)

            Lap = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))

            # Internal heat generation [W/m^2] (flattened)
            q_int = (pan.q_internal.ravel().astype(float)
                     if pan.q_internal is not None
                     else np.zeros(N))

            # Heater mask (flattened boolean)
            h_mask = (pan.heater_mask.ravel().astype(bool)
                      if pan.heater_mask is not None
                      else np.zeros(N, dtype=bool))

            panel_state.append({
                "nx": nx, "ny": ny, "N": N,
                "dx": dx, "dy": dy,
                "rho_cp_h": rho_cp_h,
                "Lap": Lap,
                "q_int": q_int,
                "h_mask": h_mask,
                "T": np.full(N, pan.T_init, dtype=float),
                "kx": kx, "ky": ky,
            })

        # ---- Recording storage ----
        T_records = {pan.name: np.zeros((n_records, pan.nx, pan.ny))
                     for pan in panels}
        time_rec = np.zeros(n_records)
        rec_idx = 0
        all_converged = True
        total_heater_energy = 0.0

        def _compute_gradient(T2d, dx_val, dy_val):
            """Max spatial temperature gradient magnitude [K/m]."""
            nx_l, ny_l = T2d.shape
            dTdx = np.zeros_like(T2d)
            dTdy = np.zeros_like(T2d)
            if nx_l > 1:
                dTdx[1:-1, :] = (T2d[2:, :] - T2d[:-2, :]) / (2.0 * dx_val)
                dTdx[0, :] = (T2d[1, :] - T2d[0, :]) / dx_val
                dTdx[-1, :] = (T2d[-1, :] - T2d[-2, :]) / dx_val
            if ny_l > 1:
                dTdy[:, 1:-1] = (T2d[:, 2:] - T2d[:, :-2]) / (2.0 * dy_val)
                dTdy[:, 0] = (T2d[:, 1] - T2d[:, 0]) / dy_val
                dTdy[:, -1] = (T2d[:, -1] - T2d[:, -2]) / dy_val
            return float(np.max(np.sqrt(dTdx ** 2 + dTdy ** 2)))

        def _record(t_val):
            nonlocal rec_idx
            if rec_idx >= n_records:
                return
            time_rec[rec_idx] = t_val
            for ip2, pan2 in enumerate(panels):
                ps2 = panel_state[ip2]
                T_records[pan2.name][rec_idx] = ps2["T"].reshape(
                    ps2["nx"], ps2["ny"])
            rec_idx += 1

        _record(0.0)

        # ---- Main time-stepping loop ----
        for step in range(1, n_steps + 1):
            t = step * dt_actual
            sun_angle = sun_angle_func(t)

            # Average panel temperatures for inter-panel radiation
            T_avg = np.array([ps["T"].mean() for ps in panel_state])

            # Inter-panel radiative exchange [W/m^2] per panel
            # (distributed uniformly over each panel)
            Q_inter_per_area = np.zeros(n_panels)
            if panel_view_factors is not None:
                F = panel_view_factors
                for i in range(n_panels):
                    eps_i = panels[i].emissivity
                    A_i = panels[i].Lx * panels[i].Ly
                    for j in range(n_panels):
                        if i == j:
                            continue
                        eps_j = panels[j].emissivity
                        eps_eff = (
                            1.0 / (1.0 / eps_i + 1.0 / eps_j - 1.0)
                            if (eps_i < 0.999 or eps_j < 0.999)
                            else 1.0
                        )
                        q_ij = eps_eff * SIGMA * F[i, j] * (
                            T_avg[j] ** 4 - T_avg[i] ** 4
                        )
                        # q_ij is [W/m^2] (view factor already area-weighted)
                        Q_inter_per_area[i] += q_ij

            # ---- Per-panel implicit solve ----
            for ip in range(n_panels):
                ps = panel_state[ip]
                pan = panels[ip]
                N = ps["N"]
                T_old = ps["T"].copy()
                rho_cp_h = ps["rho_cp_h"]
                Lap = ps["Lap"]
                eps = pan.emissivity
                alpha_s = pan.absorptivity

                # --- Solar flux with spatial variation ---
                panel_angle_offset = ip * np.pi / max(n_panels, 1)
                cos_sun = np.cos(sun_angle + panel_angle_offset)
                in_sun = cos_sun > 0.0

                if in_sun:
                    # Cosine taper: center of panel gets slightly more flux
                    ix = np.arange(ps["nx"])
                    jy = np.arange(ps["ny"])
                    II, JJ = np.meshgrid(ix, jy, indexing="ij")
                    cx = (ps["nx"] - 1) / 2.0
                    cy = (ps["ny"] - 1) / 2.0
                    spatial_mod = (
                        0.9
                        + 0.1
                        * np.cos(np.pi * (II - cx) / max(ps["nx"] - 1, 1))
                        * np.cos(np.pi * (JJ - cy) / max(ps["ny"] - 1, 1))
                    )
                    q_solar = (
                        alpha_s * solar_constant * cos_sun
                        * spatial_mod.ravel()
                    )
                else:
                    q_solar = np.zeros(N)

                # --- Heater control ---
                q_heater = np.zeros(N)
                if np.any(ps["h_mask"]):
                    cold = T_old[ps["h_mask"]] < pan.heater_setpoint
                    if np.any(cold):
                        q_heater[ps["h_mask"]] = pan.heater_power_density
                        # Accumulate heater energy [J]
                        cell_area = ps["dx"] * ps["dy"]
                        total_heater_energy += (
                            float(np.sum(q_heater)) * cell_area * dt_actual
                        )

                # --- Total external source [W/m^2] per node ---
                q_src = (
                    q_solar
                    + ps["q_int"]
                    + Q_inter_per_area[ip]
                    + q_heater
                )

                # --- Newton-Raphson implicit Euler ---
                # Energy balance per unit area at each node:
                #   rho_cp_h * (T - T_old)/dt = Lap*T + q_src
                #                               - eps*sigma*(T^4 - T_space^4)
                # Residual R(T) = rho_cp_h*(T-T_old)/dt - Lap*T - q_src
                #                 + eps*sigma*T^4 - eps*sigma*T_space^4
                # Jacobian J = rho_cp_h/dt * I - Lap + 4*eps*sigma*diag(T^3)
                # Solve: J * dT = -R;  T <- T + dT

                T_iter = T_old.copy()
                step_converged = False
                inv_dt = rho_cp_h / dt_actual
                rad_space = eps * SIGMA * T_space ** 4

                for _nr in range(newton_max_iter):
                    T4 = T_iter ** 4
                    T3 = T_iter ** 3

                    R = (
                        inv_dt * (T_iter - T_old)
                        - Lap.dot(T_iter)
                        - q_src
                        + eps * SIGMA * T4
                        - rad_space
                    )

                    J_diag = inv_dt + 4.0 * eps * SIGMA * T3
                    J = sparse.diags(J_diag, 0, shape=(N, N),
                                     format="csr") - Lap

                    delta_T = spsolve(J, -R)
                    T_iter += delta_T
                    T_iter = np.maximum(T_iter, 2.7)

                    if np.max(np.abs(delta_T)) < newton_tol:
                        step_converged = True
                        break

                if not step_converged:
                    all_converged = False

                ps["T"] = T_iter

            # --- Record ---
            if step % record_interval == 0 or step == n_steps:
                _record(t)

        # ---- Trim and post-process ----
        for name in T_records:
            T_records[name] = T_records[name][:rec_idx]
        time_rec = time_rec[:rec_idx]

        global_max = -np.inf
        global_min = np.inf
        grad_per_panel: Dict[str, np.ndarray] = {}

        for ip, pan in enumerate(panels):
            T_arr = T_records[pan.name]
            global_max = max(global_max, float(np.max(T_arr)))
            global_min = min(global_min, float(np.min(T_arr)))

            ps = panel_state[ip]
            n_t_rec = T_arr.shape[0]
            grads = np.zeros(n_t_rec)
            for k in range(n_t_rec):
                grads[k] = _compute_gradient(T_arr[k], ps["dx"], ps["dy"])
            grad_per_panel[pan.name] = grads

        return SpacecraftThermal2DResult(
            panel_names=[pan.name for pan in panels],
            times=time_rec,
            temperatures=T_records,
            max_temperature=global_max,
            min_temperature=global_min,
            thermal_gradients_per_panel=grad_per_panel,
            total_heater_energy_J=total_heater_energy,
            converged=all_converged,
        )


# ===========================================================================
# Level 3: 2D Spacecraft Panel Thermal Solver (standalone)
# ===========================================================================

class SpacecraftThermal2DSolver:
    """2D spacecraft panel thermal solver with solar, albedo, IR inputs and radiation to space.

    Solves the transient 2D heat equation on a single spacecraft panel:

        rho*cp*h * dT/dt = k*h*(d^2T/dx^2 + d^2T/dy^2)
                           + alpha_s * q_solar(x,y,t)
                           + q_albedo + q_IR
                           - eps * sigma * (T^4 - T_space^4)
                           + q_internal

    Uses explicit time-stepping with CFL-limited dt for the thermal
    diffusion. Boundary conditions are Neumann (zero-flux) on all edges.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    Lx, Ly : float
        Panel dimensions [m].
    thickness : float
        Panel thickness [m].
    conductivity : float
        Thermal conductivity [W/(m*K)].
    density : float
        Material density [kg/m^3].
    specific_heat : float
        Specific heat [J/(kg*K)].
    emissivity : float
        Surface emissivity.
    absorptivity : float
        Solar absorptivity.
    T_init : float
        Initial temperature [K].
    T_space : float
        Deep space sink temperature [K].
    solar_flux : float
        Solar constant [W/m^2].
    albedo_flux : float
        Planetary albedo flux [W/m^2].
    planetary_ir : float
        Planetary IR flux [W/m^2].
    q_internal : float
        Internal dissipation [W/m^2].
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M3_COUPLED

    def __init__(
        self,
        nx: int = 40,
        ny: int = 40,
        Lx: float = 1.0,
        Ly: float = 1.0,
        thickness: float = 0.003,
        conductivity: float = 167.0,
        density: float = 2700.0,
        specific_heat: float = 900.0,
        emissivity: float = 0.85,
        absorptivity: float = 0.3,
        T_init: float = 293.0,
        T_space: float = 4.0,
        solar_flux: float = 1361.0,
        albedo_flux: float = 50.0,
        planetary_ir: float = 230.0,
        q_internal: float = 0.0,
    ):
        self._coupled_state = None
        self._time = 0.0
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.thickness = thickness
        self.conductivity = conductivity
        self.density = density
        self.specific_heat = specific_heat
        self.emissivity = emissivity
        self.absorptivity = absorptivity
        self.T_init = T_init
        self.T_space = T_space
        self.solar_flux = solar_flux
        self.albedo_flux = albedo_flux
        self.planetary_ir = planetary_ir
        self.q_internal = q_internal

        self.dx = Lx / max(nx - 1, 1)
        self.dy = Ly / max(ny - 1, 1)
        self.x = np.linspace(0.0, Lx, nx)
        self.y = np.linspace(0.0, Ly, ny)

        self.rho_cp_h = density * specific_heat * thickness
        self.alpha_th = conductivity / (density * specific_heat)

    def solve(
        self,
        t_end: float = 5400.0,
        dt: Optional[float] = None,
        sun_angle_func: Optional[Callable[[float], float]] = None,
        record_interval: int = 10,
    ) -> SpacecraftThermal2DResult:
        """Run the 2D panel thermal simulation.

        Parameters
        ----------
        t_end : float
            End time [s] (one orbit period is ~5400 s for LEO).
        dt : float, optional
            Time step. If None, computed from CFL.
        sun_angle_func : callable, optional
            Returns sun angle [rad] as function of time. None = constant 0.
        record_interval : int
            Save fields every this many steps.

        Returns
        -------
        SpacecraftThermal2DResult
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy

        # CFL for explicit diffusion
        dt_cfl = 0.25 * min(dx, dy) ** 2 / self.alpha_th
        if dt is None:
            dt = dt_cfl
        dt = min(dt, dt_cfl, t_end)

        if sun_angle_func is None:
            sun_angle_func = lambda t: 0.0

        n_steps = max(int(np.ceil(t_end / dt)), 1)
        dt_actual = t_end / n_steps

        T = np.full((ny, nx), self.T_init)
        sigma_sb = SIGMA

        # Storage
        n_records = n_steps // record_interval + 2
        T_records = np.zeros((n_records, ny, nx))
        time_rec = np.zeros(n_records)
        grad_rec = np.zeros(n_records)
        rec_idx = 0
        total_heater_energy = 0.0
        all_converged = True

        # Record initial state
        T_records[rec_idx] = T.copy()
        time_rec[rec_idx] = 0.0
        rec_idx += 1

        for step in range(1, n_steps + 1):
            t = step * dt_actual
            sun_angle = sun_angle_func(t)

            # Solar flux with spatial cosine taper
            cos_sun = np.cos(sun_angle)
            in_sun = cos_sun > 0.0

            if in_sun:
                ix = np.arange(nx)
                jy = np.arange(ny)
                II, JJ = np.meshgrid(ix, jy)
                cx_grid = (nx - 1) / 2.0
                cy_grid = (ny - 1) / 2.0
                spatial_mod = 0.9 + 0.1 * np.cos(
                    np.pi * (II - cx_grid) / max(nx - 1, 1)) * np.cos(
                    np.pi * (JJ - cy_grid) / max(ny - 1, 1))
                q_solar = self.absorptivity * self.solar_flux * cos_sun * spatial_mod
            else:
                q_solar = np.zeros((ny, nx))

            # Albedo + planetary IR (uniform)
            q_albedo = self.absorptivity * self.albedo_flux * max(cos_sun, 0.0)
            q_ir_planet = self.emissivity * self.planetary_ir

            # Total source [W/m^2]
            q_src = q_solar + q_albedo + q_ir_planet + self.q_internal

            # Radiation to space [W/m^2]
            q_rad = self.emissivity * sigma_sb * (T ** 4 - self.T_space ** 4)

            # Diffusion (explicit, Neumann BCs via edge padding)
            T_pad = np.pad(T, 1, mode='edge')
            lap = self.conductivity * self.thickness * (
                (T_pad[1:-1, 2:] - 2.0 * T_pad[1:-1, 1:-1] + T_pad[1:-1, :-2]) / dx ** 2 +
                (T_pad[2:, 1:-1] - 2.0 * T_pad[1:-1, 1:-1] + T_pad[:-2, 1:-1]) / dy ** 2
            )

            # Energy balance: rho*cp*h * dT/dt = lap + q_src - q_rad
            dTdt = (lap + q_src - q_rad) / self.rho_cp_h
            T = T + dTdt * dt_actual
            T = np.maximum(T, 2.7)  # cannot go below cosmic background

            # Record
            if step % record_interval == 0 or step == n_steps:
                if rec_idx < n_records:
                    T_records[rec_idx] = T.copy()
                    time_rec[rec_idx] = t
                    # Gradient
                    dTdx = np.zeros_like(T)
                    dTdy = np.zeros_like(T)
                    if nx > 2:
                        dTdx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2.0 * dx)
                    if ny > 2:
                        dTdy[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * dy)
                    grad_rec[rec_idx] = float(np.max(np.sqrt(dTdx ** 2 + dTdy ** 2)))
                    rec_idx += 1

        # Trim
        T_records = T_records[:rec_idx]
        time_rec = time_rec[:rec_idx]
        grad_rec = grad_rec[:rec_idx]

        panel_name = "panel_0"
        return SpacecraftThermal2DResult(
            panel_names=[panel_name],
            times=time_rec,
            temperatures={panel_name: T_records},
            max_temperature=float(np.max(T_records)),
            min_temperature=float(np.min(T_records)),
            thermal_gradients_per_panel={panel_name: grad_rec},
            total_heater_energy_J=total_heater_energy,
            converged=all_converged,
        )

    def export_state(self, result: SpacecraftThermal2DResult) -> PhysicsState:
        """Export 2D spacecraft thermal result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="spacecraft_thermal_2d")
        panel_name = result.panel_names[0] if result.panel_names else "panel_0"
        T_final = result.temperatures[panel_name][-1] if panel_name in result.temperatures else np.array([])
        state.set_field("temperature", T_final, "K")
        state.metadata["max_temperature"] = result.max_temperature
        state.metadata["min_temperature"] = result.min_temperature
        state.metadata["converged"] = result.converged
        state.metadata["total_heater_energy_J"] = result.total_heater_energy_J
        return state

    def import_state(self, state: PhysicsState) -> None:
        """Import coupled fields from partner solvers."""
        self._coupled_state = state

    def advance(self, dt: float) -> PhysicsState:
        """Advance 2D spacecraft thermal solver by dt for closed-loop coupling."""
        q_ext = 0.0
        if self._coupled_state is not None:
            if self._coupled_state.has("heat_source"):
                q_ext = float(np.mean(self._coupled_state.get("heat_source")))
        old_q = self.q_internal
        self.q_internal += q_ext
        result = self.solve(t_end=dt, dt=None)
        self.q_internal = old_q
        self._time += dt
        return self.export_state(result)
