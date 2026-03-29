"""
Automotive Thermal System Solver

Multi-component transient thermal solver for automotive power electronics
that wires together the existing TransientHeatSolver1D, ThermalMaterial,
and HeatSource models into a unified system-level simulation.

The solver models an assembly of thermally coupled components (busbars,
IGBTs, thermal pads, coolant channels) with individual materials and
heat sources, connected through thermal contact resistances.

Governing equation per component node (Level 1, 0-D lumped):
    m_i * c_i * dT_i/dt = Q_gen,i + sum_j (T_j - T_i) / R_ij
                          - h_i * A_i * (T_i - T_ambient)

Level 3 adds a 2-D FEM transient solver (solve_2d) that discretises
each component on an (nx, ny) grid and solves:
    rho*cp*dT/dt = d/dx(k*dT/dx) + d/dy(k*dT/dy) + q'''(x,y,t)
                   - h*(T - T_amb)   [surface nodes]
using implicit Euler with ADI (Alternating Direction Implicit) splitting
and Thomas-algorithm tridiagonal solves.

Discretisation: explicit forward-Euler with adaptive sub-stepping when
the thermal Fourier number exceeds stability limits (Level 1).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable

from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from triality.automotive_thermal.transient_heat import (
    TransientHeatSolver1D,
    ThermalMaterial,
    HeatSource,
    Hotspot,
    CoolingMode,
    COPPER,
    ALUMINUM,
    SILICON,
    THERMAL_PAD,
    COOLANT,
)


@dataclass
class ComponentNode:
    """A lumped thermal node representing one component in the assembly.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. "IGBT_1", "Busbar_top").
    material : ThermalMaterial
        Thermal properties of the component.
    mass : float
        Component mass [kg].
    surface_area : float
        External surface area exposed to convection [m^2].
    heat_sources : list of HeatSource
        Internal heat generation sources attached to this node.
    cooling_mode : CoolingMode
        Active cooling mechanism.
    h_convection : float
        Convection coefficient on the exposed surface [W/(m^2*K)].
    T_init : float
        Initial temperature [K].
    """
    name: str
    material: ThermalMaterial
    mass: float
    surface_area: float
    heat_sources: List[HeatSource] = field(default_factory=list)
    cooling_mode: CoolingMode = CoolingMode.NATURAL_CONVECTION
    h_convection: float = 10.0
    T_init: float = 298.0


@dataclass
class ThermalContact:
    """Thermal contact between two component nodes.

    Parameters
    ----------
    node_a : int
        Index of first node.
    node_b : int
        Index of second node.
    conductance : float
        Thermal conductance G = k*A/L [W/K] through the contact.
    """
    node_a: int
    node_b: int
    conductance: float


@dataclass
class AutomotiveThermalResult:
    """Results from the automotive system-level thermal solver.

    Attributes
    ----------
    times : np.ndarray
        Time vector [s], shape (n_steps+1,).
    temperatures : np.ndarray
        Node temperatures vs time [K], shape (n_steps+1, n_nodes).
    heat_generation : np.ndarray
        Total heat generated per node vs time [W], shape (n_steps+1, n_nodes).
    hotspots : list of Hotspot
        Detected hotspots at the final time.
    max_temperature : float
        Peak temperature across all nodes and times [K].
    thermal_margins : np.ndarray
        Margin to material limit for each node at end of simulation [K].
    cooling_adequate : bool
        True if every node stays below its material limit.
    node_names : list of str
        Names corresponding to each column of *temperatures*.
    """
    times: np.ndarray
    temperatures: np.ndarray
    heat_generation: np.ndarray
    hotspots: List[Hotspot]
    max_temperature: float
    thermal_margins: np.ndarray
    cooling_adequate: bool
    node_names: List[str]


@dataclass
class ComponentGrid2D:
    """2-D grid specification for a single component in the 2-D FEM solver.

    Parameters
    ----------
    node_index : int
        Index of the corresponding ComponentNode in the solver's node list.
    Lx : float
        Component length in x [m].
    Ly : float
        Component length in y [m].
    nx : int
        Number of grid points in x (must be >= 3).
    ny : int
        Number of grid points in y (must be >= 3).
    q_func : callable(x, y, t) -> float or None
        Volumetric heat generation rate [W/m^3].
        If None, total HeatSource power is distributed uniformly.
    surface_nodes : str
        Which boundary faces are exposed to convection.
        Comma-separated subset of "left,right,top,bottom,all".
        Default is "all".
    """
    node_index: int
    Lx: float
    Ly: float
    nx: int = 20
    ny: int = 20
    q_func: Optional[Callable] = None
    surface_nodes: str = "all"


@dataclass
class InterComponentCoupling2D:
    """Contact conductance coupling between two component grids.

    Boundary nodes on the specified edge of component A are coupled
    to the opposing edge of component B via a contact conductance.

    Parameters
    ----------
    comp_a : int
        Index into the grids list for component A.
    edge_a : str
        Edge of A that is in contact: "left", "right", "top", "bottom".
    comp_b : int
        Index into the grids list for component B.
    edge_b : str
        Edge of B that is in contact.
    conductance_per_area : float
        Contact conductance per unit area [W/(m^2*K)].
    """
    comp_a: int
    edge_a: str
    comp_b: int
    edge_b: str
    conductance_per_area: float = 1000.0


@dataclass
class Component2DFieldResult:
    """Per-component 2-D thermal field results.

    Attributes
    ----------
    name : str
        Component name.
    temperature_field : np.ndarray
        Final 2D temperature field [K], shape (ny, nx).
    temperature_history : np.ndarray
        Temperature fields at saved time steps [K], shape (n_save, ny, nx).
    peak_temperature : float
        Peak temperature across all times [K].
    thermal_gradient : np.ndarray
        Magnitude of temperature gradient at final time [K/m], shape (ny, nx).
    hotspot_location : Tuple[float, float]
        (x, y) location of the peak temperature in the final field [m].
    """
    name: str
    temperature_field: np.ndarray
    temperature_history: np.ndarray
    peak_temperature: float
    thermal_gradient: np.ndarray
    hotspot_location: Tuple[float, float]


@dataclass
class AutomotiveThermal2DResult:
    """Results from the 2-D FEM automotive thermal solver.

    Attributes
    ----------
    times : np.ndarray
        Saved time vector [s], shape (n_save,).
    components : list of Component2DFieldResult
        Per-component 2-D field results.
    peak_temperatures : np.ndarray
        Peak temperature per component [K], shape (n_comp,).
    max_temperature : float
        Global peak temperature [K].
    hotspots : list of Hotspot
        Detected hotspots across all components at final time.
    cooling_adequate : bool
        True if no component exceeds its material limit.
    """
    times: np.ndarray
    components: List[Component2DFieldResult]
    peak_temperatures: np.ndarray
    max_temperature: float
    hotspots: List[Hotspot]
    cooling_adequate: bool


class AutomotiveThermalSolver:
    """System-level transient thermal solver for automotive assemblies.

    Wires together multiple ComponentNode objects connected by
    ThermalContact resistances and solves the coupled ODE system
    using forward-Euler integration with adaptive sub-stepping.

    Parameters
    ----------
    nodes : list of ComponentNode
        Thermal nodes in the assembly.
    contacts : list of ThermalContact
        Thermal connections between nodes.
    T_ambient : float
        Ambient / coolant sink temperature [K].
    """

    fidelity_tier = FidelityTier.REDUCED_ORDER
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        nodes: List[ComponentNode],
        contacts: List[ThermalContact],
        T_ambient: float = 298.0,
    ):
        self.nodes = list(nodes)
        self.contacts = list(contacts)
        self.T_ambient = T_ambient
        self.n_nodes = len(nodes)

        # Pre-compute lumped capacitances  C_i = m_i * c_i  [J/K]
        self._capacitance = np.array(
            [n.mass * n.material.specific_heat for n in self.nodes]
        )

        # Build conductance matrix (symmetric)
        self._G = np.zeros((self.n_nodes, self.n_nodes))
        for c in self.contacts:
            self._G[c.node_a, c.node_b] += c.conductance
            self._G[c.node_b, c.node_a] += c.conductance

        # Convective loss coefficient per node  h*A  [W/K]
        self._hA = np.array(
            [n.h_convection * n.surface_area for n in self.nodes]
        )

    def _heat_generation(self, t: float) -> np.ndarray:
        """Total heat generation per node at time *t* [W]."""
        Q = np.zeros(self.n_nodes)
        for i, node in enumerate(self.nodes):
            for src in node.heat_sources:
                Q[i] += src.power_at_time(t)
        return Q

    def _rhs(self, T: np.ndarray, t: float) -> np.ndarray:
        """Right-hand side dT/dt for the lumped ODE system.

        dT_i/dt = [Q_gen,i + sum_j G_ij*(T_j - T_i)
                   - h_i*A_i*(T_i - T_amb)] / C_i
        """
        Q_gen = self._heat_generation(t)

        # Inter-node conduction: sum_j G_ij * (T_j - T_i)
        Q_cond = self._G.dot(T) - np.sum(self._G, axis=1) * T

        # Convection to ambient
        Q_conv = self._hA * (T - self.T_ambient)

        dTdt = (Q_gen + Q_cond - Q_conv) / self._capacitance
        return dTdt

    def _max_stable_dt(self) -> float:
        """Estimate the maximum stable explicit time step.

        For each node the effective thermal time constant is
        tau_i = C_i / (sum_j G_ij + hA_i).  Stability requires
        dt < tau_min.
        """
        total_loss = np.sum(self._G, axis=1) + self._hA
        # Avoid division by zero for perfectly insulated nodes
        total_loss = np.maximum(total_loss, 1e-30)
        tau = self._capacitance / total_loss
        return float(np.min(tau))

    def solve(
        self,
        t_end: float,
        dt: float = 0.1,
        save_every: int = 1,
        progress_callback=None,
    ) -> AutomotiveThermalResult:
        """Run the transient thermal simulation.

        Parameters
        ----------
        t_end : float
            End time [s].
        dt : float
            Requested time step [s].  Will be reduced automatically
            if it exceeds the explicit stability limit.
        save_every : int
            Store results every *save_every* steps to limit memory.
        progress_callback : callable, optional
            Called as progress_callback(step, total) during time stepping.

        Returns
        -------
        AutomotiveThermalResult
        """
        # Adaptive sub-stepping for stability
        dt_stable = self._max_stable_dt() * 0.9  # 90 % of limit
        dt_actual = min(dt, dt_stable) if dt_stable > 0 else dt

        n_steps = max(int(np.ceil(t_end / dt_actual)), 1)
        dt_actual = t_end / n_steps

        # Allocate storage
        save_indices = list(range(0, n_steps + 1, save_every))
        if n_steps not in save_indices:
            save_indices.append(n_steps)
        n_save = len(save_indices)

        times = np.zeros(n_save)
        T_hist = np.zeros((n_save, self.n_nodes))
        Q_hist = np.zeros((n_save, self.n_nodes))

        # Initial conditions
        T = np.array([n.T_init for n in self.nodes], dtype=float)
        save_idx = 0
        if 0 in save_indices:
            times[save_idx] = 0.0
            T_hist[save_idx] = T.copy()
            Q_hist[save_idx] = self._heat_generation(0.0)
            save_idx += 1

        # Time integration (forward Euler)
        _prog_interval = max(n_steps // 50, 1)
        for step in range(1, n_steps + 1):
            t = (step - 1) * dt_actual
            dTdt = self._rhs(T, t)
            T = T + dTdt * dt_actual
            if progress_callback and step % _prog_interval == 0:
                progress_callback(step, n_steps)

            if step in save_indices:
                times[save_idx] = step * dt_actual
                T_hist[save_idx] = T.copy()
                Q_hist[save_idx] = self._heat_generation(step * dt_actual)
                save_idx += 1

        # Post-processing ------------------------------------------------
        # Hotspot detection at final time
        hotspots: List[Hotspot] = []
        for i, node in enumerate(self.nodes):
            T_final = T[i]
            margin = node.material.max_temperature - T_final
            if margin < 0:
                risk = "Critical"
            elif margin < 20:
                risk = "Warning"
            else:
                risk = "Safe"

            if risk != "Safe":
                hotspots.append(
                    Hotspot(
                        position=(float(i), 0.0, 0.0),
                        temperature=T_final,
                        margin_to_limit=margin,
                        power_density=float(Q_hist[-1, i]),
                        risk_level=risk,
                    )
                )

        thermal_margins = np.array(
            [n.material.max_temperature - T[i] for i, n in enumerate(self.nodes)]
        )

        return AutomotiveThermalResult(
            times=times[:save_idx],
            temperatures=T_hist[:save_idx],
            heat_generation=Q_hist[:save_idx],
            hotspots=hotspots,
            max_temperature=float(np.max(T_hist[:save_idx])),
            thermal_margins=thermal_margins,
            cooling_adequate=bool(np.all(thermal_margins > 0)),
            node_names=[n.name for n in self.nodes],
        )

    def export_state(self, result: AutomotiveThermalResult) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="automotive_thermal")
        state.set_field("temperature", result.temperatures[-1, :], "K")
        state.set_field("heat_source", result.heat_generation[-1, :], "W/m^3")
        state.metadata["max_temperature"] = result.max_temperature
        state.metadata["cooling_adequate"] = result.cooling_adequate
        state.metadata["thermal_margins"] = result.thermal_margins
        state.metadata["node_names"] = result.node_names
        return state

    # ------------------------------------------------------------------
    # Level 3: 2-D FEM transient thermal solver (ADI splitting)
    # ------------------------------------------------------------------

    @staticmethod
    def _thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                      d: np.ndarray) -> np.ndarray:
        """Solve a tridiagonal system using the Thomas algorithm.

        Solves  A x = d  where A is tridiagonal with:
            a = sub-diagonal   (length n, a[0] unused)
            b = main diagonal  (length n)
            c = super-diagonal (length n, c[-1] unused)
            d = right-hand side (length n)

        Returns x (length n).  Modifies copies, not originals.
        """
        n = len(d)
        # Work on copies
        bc = b.copy()
        dc = d.copy()
        ac = a.copy()
        cc = c.copy()

        # Forward elimination
        for i in range(1, n):
            if abs(bc[i - 1]) < 1e-30:
                m = 0.0
            else:
                m = ac[i] / bc[i - 1]
            bc[i] -= m * cc[i - 1]
            dc[i] -= m * dc[i - 1]

        # Back substitution
        x = np.zeros(n)
        if abs(bc[-1]) > 1e-30:
            x[-1] = dc[-1] / bc[-1]
        for i in range(n - 2, -1, -1):
            if abs(bc[i]) > 1e-30:
                x[i] = (dc[i] - cc[i] * x[i + 1]) / bc[i]
        return x

    @staticmethod
    def _k_at_T(k0: float, T: float, T_ref: float = 298.0,
                dk_dT: float = 0.0) -> float:
        """Temperature-dependent thermal conductivity (linear model).

        k(T) = k0 + dk_dT * (T - T_ref)
        """
        return k0 + dk_dT * (T - T_ref)

    @staticmethod
    def _cp_at_T(cp0: float, T: float, T_ref: float = 298.0,
                 dcp_dT: float = 0.0) -> float:
        """Temperature-dependent specific heat (linear model).

        cp(T) = cp0 + dcp_dT * (T - T_ref)
        """
        return cp0 + dcp_dT * (T - T_ref)

    def solve_2d(
        self,
        grids: List[ComponentGrid2D],
        t_end: float,
        dt: float = 0.01,
        save_every: int = 10,
        couplings: Optional[List[InterComponentCoupling2D]] = None,
        dk_dT: Optional[Dict[int, float]] = None,
        dcp_dT: Optional[Dict[int, float]] = None,
    ) -> AutomotiveThermal2DResult:
        """Run a 2-D FEM transient thermal simulation using ADI splitting.

        Each component is discretised on its own (nx, ny) grid.  The 2-D
        heat equation is solved with implicit Euler using ADI (Alternating
        Direction Implicit) splitting:

        Half-step (implicit in x, explicit in y):
            rho*cp*(T* - T^n)/dt = d/dx(k dT*/dx) + d/dy(k dT^n/dy) + q'''

        Full-step (explicit in x, implicit in y):
            rho*cp*(T^{n+1} - T*)/dt = d/dx(k dT*/dx) + d/dy(k dT^{n+1}/dy) + q'''

        Tridiagonal systems are solved with the Thomas algorithm.

        Parameters
        ----------
        grids : list of ComponentGrid2D
            Grid specification for each component.
        t_end : float
            End time [s].
        dt : float
            Time step [s].
        save_every : int
            Save results every N steps.
        couplings : list of InterComponentCoupling2D or None
            Inter-component contact couplings.
        dk_dT : dict {node_index: dk/dT} or None
            Linear temperature coefficient for thermal conductivity
            [W/(m*K^2)] per component.  Defaults to 0 (constant k).
        dcp_dT : dict {node_index: dcp/dT} or None
            Linear temperature coefficient for specific heat
            [J/(kg*K^2)] per component.  Defaults to 0 (constant cp).

        Returns
        -------
        AutomotiveThermal2DResult
        """
        if couplings is None:
            couplings = []
        if dk_dT is None:
            dk_dT = {}
        if dcp_dT is None:
            dcp_dT = {}

        n_comp = len(grids)
        n_steps = max(int(np.ceil(t_end / dt)), 1)
        dt_actual = t_end / n_steps

        # Build save schedule
        save_indices = list(range(0, n_steps + 1, save_every))
        if n_steps not in save_indices:
            save_indices.append(n_steps)
        save_set = set(save_indices)
        n_save = len(save_indices)

        # Initialise temperature fields and grid metrics per component
        T_fields = []       # list of (ny, nx) arrays
        dx_arr = []         # dx per component
        dy_arr = []         # dy per component
        save_hist = []      # list of (n_save, ny, nx) arrays

        for g in grids:
            node = self.nodes[g.node_index]
            T0 = np.full((g.ny, g.nx), node.T_init)
            T_fields.append(T0)
            dx_arr.append(g.Lx / max(g.nx - 1, 1))
            dy_arr.append(g.Ly / max(g.ny - 1, 1))
            save_hist.append(np.zeros((n_save, g.ny, g.nx)))

        # Save initial state
        save_idx = 0
        times_save = np.zeros(n_save)
        if 0 in save_set:
            times_save[save_idx] = 0.0
            for ci in range(n_comp):
                save_hist[ci][save_idx] = T_fields[ci].copy()
            save_idx += 1

        # Helper: determine which boundary faces are "surface" nodes
        def _parse_surfaces(surface_str: str):
            s = surface_str.lower().strip()
            if s == "all":
                return {"left", "right", "top", "bottom"}
            return set(x.strip() for x in s.split(","))

        surface_sets = [_parse_surfaces(g.surface_nodes) for g in grids]

        # Helper: get volumetric heat source for a component at (i,j,t)
        def _q_vol(ci: int, ix: int, iy: int, t: float) -> float:
            g = grids[ci]
            if g.q_func is not None:
                x = ix * dx_arr[ci]
                y = iy * dy_arr[ci]
                return g.q_func(x, y, t)
            # Uniform distribution of total HeatSource power
            node = self.nodes[g.node_index]
            total_power = 0.0
            for src in node.heat_sources:
                total_power += src.power_at_time(t)
            # Distribute uniformly; volume per grid cell
            vol_cell = dx_arr[ci] * dy_arr[ci] * 1.0  # unit depth
            # Total volume of component
            total_vol = g.Lx * g.Ly * 1.0
            if total_vol < 1e-30:
                return 0.0
            return total_power / total_vol

        # ---------------------------------------------------------------
        # Time integration with ADI splitting
        # ---------------------------------------------------------------
        for step in range(1, n_steps + 1):
            t_n = (step - 1) * dt_actual
            t_half = t_n + 0.5 * dt_actual
            t_np1 = step * dt_actual

            for ci in range(n_comp):
                g = grids[ci]
                node = self.nodes[g.node_index]
                mat = node.material
                nx, ny = g.nx, g.ny
                dx = dx_arr[ci]
                dy = dy_arr[ci]
                T = T_fields[ci]  # (ny, nx)

                rho = mat.density
                k0 = mat.thermal_conductivity
                cp0 = mat.specific_heat
                h_conv = node.h_convection
                T_amb = self.T_ambient

                dkdT = dk_dT.get(g.node_index, 0.0)
                dcpdT = dcp_dT.get(g.node_index, 0.0)
                surfaces = surface_sets[ci]

                # -------------------------------------------------------
                # Pre-compute Ly(T^n) explicitly for the Douglas-Gunn
                # ADI splitting.
                #
                # Douglas-Gunn implicit Euler ADI:
                #   Step 1 (x-sweep):
                #     (rho_cp/dt - Lx) T* = rho_cp/dt * T^n + Ly(T^n) + S
                #   Step 2 (y-sweep):
                #     (rho_cp/dt - Ly) T^{n+1} = rho_cp/dt * T* - Ly(T^n)
                #
                # where S = q''' - h_loss*(T - T_amb) evaluated at T^n
                # and Ly(T^n) is subtracted in step 2 to prevent
                # double-counting (it was already added in step 1).
                # -------------------------------------------------------

                # Compute Ly(T^n) for each grid point
                Ly_Tn = np.zeros((ny, nx))
                for jy in range(ny):
                    for ix in range(nx):
                        T_ij = T[jy, ix]
                        if jy > 0 and jy < ny - 1:
                            k_jp = self._k_at_T(k0, 0.5*(T_ij + T[jy+1, ix]),
                                                 dk_dT=dkdT)
                            k_jm = self._k_at_T(k0, 0.5*(T_ij + T[jy-1, ix]),
                                                 dk_dT=dkdT)
                            Ly_Tn[jy, ix] = (k_jp*(T[jy+1, ix] - T_ij)
                                              - k_jm*(T_ij - T[jy-1, ix])) / (dy*dy)
                        elif jy == 0:
                            # Neumann at bottom: dT/dy = 0 => T_{-1} = T_0
                            if ny > 1:
                                k_jp = self._k_at_T(k0, 0.5*(T_ij + T[1, ix]),
                                                     dk_dT=dkdT)
                                Ly_Tn[jy, ix] = k_jp*(T[1, ix] - T_ij) / (dy*dy)
                            # else single row: Ly = 0
                        else:  # jy == ny - 1
                            k_jm = self._k_at_T(k0, 0.5*(T_ij + T[jy-1, ix]),
                                                 dk_dT=dkdT)
                            Ly_Tn[jy, ix] = -k_jm*(T_ij - T[jy-1, ix]) / (dy*dy)

                # -------------------------------------------------------
                # Step 1: x-sweep (implicit in x, row by row)
                # (rho_cp/dt + Ax) T* = rho_cp/dt * T^n + Ly(T^n) + S
                # where Ax represents the implicit x-diffusion operator
                # -------------------------------------------------------
                T_star = np.zeros_like(T)

                for jy in range(ny):
                    a_sub = np.zeros(nx)
                    b_main = np.zeros(nx)
                    c_sup = np.zeros(nx)
                    rhs_row = np.zeros(nx)

                    for ix in range(nx):
                        T_ij = T[jy, ix]
                        k_ij = self._k_at_T(k0, T_ij, dk_dT=dkdT)
                        cp_ij = self._cp_at_T(cp0, T_ij, dcp_dT=dcpdT)
                        rho_cp = rho * cp_ij

                        # Source terms
                        q_vol = _q_vol(ci, ix, jy, t_np1)

                        # Convection loss coefficient for surface nodes
                        h_loss = 0.0
                        if ix == 0 and "left" in surfaces:
                            h_loss += h_conv / dx
                        if ix == nx - 1 and "right" in surfaces:
                            h_loss += h_conv / dx
                        if jy == 0 and "bottom" in surfaces:
                            h_loss += h_conv / dy
                        if jy == ny - 1 and "top" in surfaces:
                            h_loss += h_conv / dy

                        # RHS: rho_cp/dt * T^n + Ly(T^n) + q - h_loss*T^n + h_loss*T_amb
                        rhs_row[ix] = (rho_cp / dt_actual * T_ij
                                       + Ly_Tn[jy, ix]
                                       + q_vol
                                       + h_loss * T_amb)

                        # Implicit x-diffusion coefficients
                        if ix > 0:
                            k_im = self._k_at_T(k0, 0.5*(T_ij + T[jy, ix-1]),
                                                 dk_dT=dkdT)
                        else:
                            k_im = k_ij
                        if ix < nx - 1:
                            k_ip = self._k_at_T(k0, 0.5*(T_ij + T[jy, ix+1]),
                                                 dk_dT=dkdT)
                        else:
                            k_ip = k_ij

                        coeff_im = k_im / (dx * dx)
                        coeff_ip = k_ip / (dx * dx)

                        # Main diagonal: rho_cp/dt + coeff_left + coeff_right + h_loss
                        # At boundaries (ix=0 or ix=nx-1), the missing
                        # neighbour is handled by Neumann (dT/dx=0):
                        # ghost node T_{-1} = T_0, so the coeff_im
                        # contribution cancels.  Since a_sub[0]=0 the
                        # coeff_im term in b_main just represents the
                        # Neumann reflection — we should NOT include it.
                        center = rho_cp / dt_actual + h_loss
                        if ix > 0:
                            center += coeff_im
                            a_sub[ix] = -coeff_im
                        if ix < nx - 1:
                            center += coeff_ip
                            c_sup[ix] = -coeff_ip

                        b_main[ix] = center

                    T_star[jy, :] = self._thomas_solve(a_sub, b_main, c_sup, rhs_row)

                # -------------------------------------------------------
                # Step 2: y-sweep (implicit in y, column by column)
                # (rho_cp/dt + Ay) T^{n+1} = rho_cp/dt * T* - Ly(T^n)
                #
                # Note: Ly(T^n) is subtracted because it was already
                # included in step 1's RHS.
                # -------------------------------------------------------
                T_new = np.zeros_like(T)

                for ix in range(nx):
                    a_sub = np.zeros(ny)
                    b_main = np.zeros(ny)
                    c_sup = np.zeros(ny)
                    rhs_col = np.zeros(ny)

                    for jy in range(ny):
                        T_s_ij = T_star[jy, ix]
                        k_ij = self._k_at_T(k0, T_s_ij, dk_dT=dkdT)
                        cp_ij = self._cp_at_T(cp0, T_s_ij, dcp_dT=dcpdT)
                        rho_cp = rho * cp_ij

                        # RHS: rho_cp/dt * T* - Ly(T^n)
                        rhs_col[jy] = (rho_cp / dt_actual * T_s_ij
                                        - Ly_Tn[jy, ix])

                        # Implicit y-diffusion coefficients
                        if jy > 0:
                            k_jm = self._k_at_T(k0, 0.5*(T_s_ij + T_star[jy-1, ix]),
                                                 dk_dT=dkdT)
                        else:
                            k_jm = k_ij
                        if jy < ny - 1:
                            k_jp = self._k_at_T(k0, 0.5*(T_s_ij + T_star[jy+1, ix]),
                                                 dk_dT=dkdT)
                        else:
                            k_jp = k_ij

                        coeff_jm = k_jm / (dy * dy)
                        coeff_jp = k_jp / (dy * dy)

                        center = rho_cp / dt_actual
                        if jy > 0:
                            center += coeff_jm
                            a_sub[jy] = -coeff_jm
                        if jy < ny - 1:
                            center += coeff_jp
                            c_sup[jy] = -coeff_jp

                        b_main[jy] = center

                    T_new[:, ix] = self._thomas_solve(a_sub, b_main, c_sup, rhs_col)

                # -------------------------------------------------------
                # Inter-component coupling (contact conductance)
                # Applied as explicit source/sink at boundary nodes
                # -------------------------------------------------------
                T_fields[ci] = T_new

            # Apply inter-component coupling after all components are updated
            for coup in couplings:
                ca, cb = coup.comp_a, coup.comp_b
                ga, gb = grids[ca], grids[cb]
                Ta, Tb = T_fields[ca], T_fields[cb]
                h_c = coup.conductance_per_area

                # Get edge temperatures
                edge_a = self._get_edge_temps(Ta, coup.edge_a)
                edge_b = self._get_edge_temps(Tb, coup.edge_b)

                # Interpolate if edge lengths differ
                n_a = len(edge_a)
                n_b = len(edge_b)
                if n_a != n_b:
                    # Interpolate edge_b to match edge_a length
                    x_b = np.linspace(0, 1, n_b)
                    x_a = np.linspace(0, 1, n_a)
                    edge_b_interp = np.interp(x_a, x_b, edge_b)
                    x_b2 = np.linspace(0, 1, n_b)
                    x_a2 = np.linspace(0, 1, n_a)
                    edge_a_interp = np.interp(x_b2, x_a2, edge_a)
                else:
                    edge_b_interp = edge_b
                    edge_a_interp = edge_a

                # Heat flux from A edge to B edge: q = h_c * (T_b - T_a)
                # Apply as temperature correction
                dT_a = edge_b_interp - edge_a
                dT_b = edge_a_interp - edge_b

                # Compute effective correction
                node_a = self.nodes[ga.node_index]
                node_b = self.nodes[gb.node_index]
                rho_a = node_a.material.density
                cp_a = node_a.material.specific_heat
                rho_b = node_b.material.density
                cp_b = node_b.material.specific_heat

                # Correction for edge A
                if coup.edge_a in ("left", "right"):
                    cell_depth_a = dy_arr[ca]
                else:
                    cell_depth_a = dx_arr[ca]
                corr_a = h_c * dt_actual / (rho_a * cp_a * cell_depth_a)

                if coup.edge_b in ("left", "right"):
                    cell_depth_b = dy_arr[cb]
                else:
                    cell_depth_b = dx_arr[cb]
                corr_b = h_c * dt_actual / (rho_b * cp_b * cell_depth_b)

                # Clamp correction factor
                corr_a = min(corr_a, 0.5)
                corr_b = min(corr_b, 0.5)

                self._apply_edge_correction(T_fields[ca], coup.edge_a,
                                            corr_a * dT_a)
                self._apply_edge_correction(T_fields[cb], coup.edge_b,
                                            corr_b * dT_b)

            # Save if needed
            if step in save_set:
                times_save[save_idx] = step * dt_actual
                for ci in range(n_comp):
                    save_hist[ci][save_idx] = T_fields[ci].copy()
                save_idx += 1

        # ---------------------------------------------------------------
        # Post-processing
        # ---------------------------------------------------------------
        times_out = times_save[:save_idx]
        comp_results = []
        peak_temps = np.zeros(n_comp)
        all_hotspots: List[Hotspot] = []
        cooling_ok = True

        for ci in range(n_comp):
            g = grids[ci]
            node = self.nodes[g.node_index]
            T_final = T_fields[ci]
            hist = save_hist[ci][:save_idx]
            dx = dx_arr[ci]
            dy = dy_arr[ci]

            # Peak temperature across all saved times
            peak_T = float(np.max(hist))
            peak_temps[ci] = peak_T

            # Thermal gradient at final time (magnitude)
            grad_x = np.zeros_like(T_final)
            grad_y = np.zeros_like(T_final)
            # Central differences, forward/backward at boundaries
            if g.nx > 1:
                grad_x[:, 1:-1] = (T_final[:, 2:] - T_final[:, :-2]) / (2*dx)
                grad_x[:, 0] = (T_final[:, 1] - T_final[:, 0]) / dx
                grad_x[:, -1] = (T_final[:, -1] - T_final[:, -2]) / dx
            if g.ny > 1:
                grad_y[1:-1, :] = (T_final[2:, :] - T_final[:-2, :]) / (2*dy)
                grad_y[0, :] = (T_final[1, :] - T_final[0, :]) / dy
                grad_y[-1, :] = (T_final[-1, :] - T_final[-2, :]) / dy
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)

            # Hotspot location
            idx_flat = np.argmax(T_final)
            jy_hot, ix_hot = np.unravel_index(idx_flat, T_final.shape)
            x_hot = ix_hot * dx
            y_hot = jy_hot * dy
            T_hot = float(T_final[jy_hot, ix_hot])

            # Check against material limit
            margin = node.material.max_temperature - peak_T
            if margin < 0:
                cooling_ok = False
                risk = "Critical"
            elif margin < 20:
                risk = "Warning"
            else:
                risk = "Safe"

            if risk != "Safe":
                all_hotspots.append(Hotspot(
                    position=(x_hot, y_hot, 0.0),
                    temperature=T_hot,
                    margin_to_limit=margin,
                    power_density=_q_vol(ci, ix_hot, jy_hot, t_end),
                    risk_level=risk,
                ))

            comp_results.append(Component2DFieldResult(
                name=node.name,
                temperature_field=T_final,
                temperature_history=hist,
                peak_temperature=peak_T,
                thermal_gradient=grad_mag,
                hotspot_location=(x_hot, y_hot),
            ))

        return AutomotiveThermal2DResult(
            times=times_out,
            components=comp_results,
            peak_temperatures=peak_temps,
            max_temperature=float(np.max(peak_temps)),
            hotspots=all_hotspots,
            cooling_adequate=cooling_ok,
        )

    @staticmethod
    def _get_edge_temps(T: np.ndarray, edge: str) -> np.ndarray:
        """Extract temperature values along a specified edge of a 2-D field.

        Parameters
        ----------
        T : (ny, nx) array
        edge : "left", "right", "top", or "bottom"
        """
        if edge == "left":
            return T[:, 0].copy()
        elif edge == "right":
            return T[:, -1].copy()
        elif edge == "bottom":
            return T[0, :].copy()
        elif edge == "top":
            return T[-1, :].copy()
        else:
            raise ValueError(f"Unknown edge: {edge}")

    @staticmethod
    def _apply_edge_correction(T: np.ndarray, edge: str,
                               dT: np.ndarray) -> None:
        """Apply a temperature correction along a specified edge in-place.

        Parameters
        ----------
        T : (ny, nx) array, modified in place
        edge : "left", "right", "top", or "bottom"
        dT : 1-D correction array matching the edge length
        """
        if edge == "left":
            T[:, 0] += dT
        elif edge == "right":
            T[:, -1] += dT
        elif edge == "bottom":
            T[0, :] += dT
        elif edge == "top":
            T[-1, :] += dT
        else:
            raise ValueError(f"Unknown edge: {edge}")


# ======================================================================
# Level 3: Standalone 2D Battery Pack Thermal Solver
# ======================================================================

@dataclass
class AutomotiveThermal2DSolverResult:
    """Result of the standalone 2D battery pack thermal solver.

    Attributes
    ----------
    temperature : np.ndarray
        Final temperature field [K], shape (ny, nx).
    temperature_history : np.ndarray
        Temperature snapshots [K], shape (n_saves, ny, nx).
    x : np.ndarray
        x coordinates [m], shape (nx,).
    y : np.ndarray
        y coordinates [m], shape (ny,).
    times : np.ndarray
        Snapshot times [s].
    max_temperature : float
        Peak temperature [K].
    mean_temperature : float
        Mean temperature at final time [K].
    thermal_gradient : np.ndarray
        Temperature gradient magnitude at final time [K/m], shape (ny, nx).
    hotspot_location : Tuple[float, float]
        (x, y) of peak temperature [m].
    cooling_adequate : bool
        True if max temperature stays below limit.
    """
    temperature: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature_history: np.ndarray = field(default_factory=lambda: np.array([]))
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    times: np.ndarray = field(default_factory=lambda: np.array([]))
    max_temperature: float = 0.0
    mean_temperature: float = 0.0
    thermal_gradient: np.ndarray = field(default_factory=lambda: np.array([]))
    hotspot_location: Tuple[float, float] = (0.0, 0.0)
    cooling_adequate: bool = True


class AutomotiveThermal2DSolver:
    """Standalone 2D thermal management solver for battery pack cross-section.

    Solves the 2D transient heat equation:

        rho * cp * dT/dt = k * (d^2T/dx^2 + d^2T/dy^2) + q_gen(x,y)

    with heat generation zones (battery cells), coolant channel BCs
    (Dirichlet or Robin), and convective boundaries.

    Uses explicit forward Euler with CFL-limited time stepping.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    Lx, Ly : float
        Domain size [m] (pack cross-section).
    k : float
        Thermal conductivity [W/(m K)].
    rho : float
        Density [kg/m^3].
    cp : float
        Specific heat [J/(kg K)].
    T_init : float
        Initial temperature [K].
    T_coolant : float
        Coolant temperature [K].
    h_conv : float
        Convection coefficient at boundaries [W/(m^2 K)].
    q_gen : float
        Volumetric heat generation in cells [W/m^3].
    T_limit : float
        Maximum allowable temperature [K].
    n_cells_x, n_cells_y : int
        Number of battery cells in x and y directions.
    coolant_channels : str
        Coolant channel configuration: 'bottom', 'top_bottom', 'interleaved'.
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        nx: int = 80,
        ny: int = 60,
        Lx: float = 0.3,
        Ly: float = 0.2,
        k: float = 30.0,
        rho: float = 2500.0,
        cp: float = 1000.0,
        T_init: float = 298.0,
        T_coolant: float = 293.0,
        h_conv: float = 500.0,
        q_gen: float = 5e4,
        T_limit: float = 333.0,
        n_cells_x: int = 4,
        n_cells_y: int = 3,
        coolant_channels: str = "bottom",
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.k = k
        self.rho = rho
        self.cp = cp
        self.T_init = T_init
        self.T_coolant = T_coolant
        self.h_conv = h_conv
        self.q_gen = q_gen
        self.T_limit = T_limit
        self.n_cells_x = n_cells_x
        self.n_cells_y = n_cells_y
        self.coolant_channels = coolant_channels
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)

    def _build_heat_source(self) -> np.ndarray:
        """Build the volumetric heat source field for battery cells."""
        q = np.zeros((self.ny, self.nx))
        cell_w = self.Lx / (self.n_cells_x * 2)
        cell_h = self.Ly / (self.n_cells_y * 2)
        for cy in range(self.n_cells_y):
            y_center = (cy + 0.5) * self.Ly / self.n_cells_y
            for cx in range(self.n_cells_x):
                x_center = (cx + 0.5) * self.Lx / self.n_cells_x
                for j in range(self.ny):
                    for i in range(self.nx):
                        if (abs(self.x[i] - x_center) < cell_w and
                                abs(self.y[j] - y_center) < cell_h):
                            q[j, i] = self.q_gen
        return q

    def _build_coolant_mask(self) -> np.ndarray:
        """Build a boolean mask for coolant channel locations."""
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        channel_thickness = max(2, self.ny // 20)
        if self.coolant_channels in ("bottom", "top_bottom"):
            mask[:channel_thickness, :] = True
        if self.coolant_channels in ("top_bottom",):
            mask[-channel_thickness:, :] = True
        if self.coolant_channels == "interleaved":
            mask[:channel_thickness, :] = True
            mid = self.ny // 2
            mask[mid - channel_thickness // 2:mid + channel_thickness // 2, :] = True
            mask[-channel_thickness:, :] = True
        return mask

    def solve(
        self,
        t_end: float = 60.0,
        dt: Optional[float] = None,
        n_snapshots: int = 30,
    ) -> AutomotiveThermal2DSolverResult:
        """Run the 2D transient thermal simulation.

        Parameters
        ----------
        t_end : float
            End time [s].
        dt : float or None
            Time step [s]. If None, computed from CFL.
        n_snapshots : int
            Number of snapshots to save.

        Returns
        -------
        AutomotiveThermal2DSolverResult
        """
        alpha = self.k / (self.rho * self.cp)
        h_min = min(self.dx, self.dy)

        # CFL stability: dt < h^2 / (4*alpha) for 2D explicit
        if dt is None:
            dt = 0.4 * h_min**2 / (4.0 * alpha + 1e-30)

        n_steps = int(np.ceil(t_end / dt))
        dt = t_end / n_steps

        Fx = alpha * dt / self.dx**2
        Fy = alpha * dt / self.dy**2

        # Heat source and coolant
        q_source = self._build_heat_source()
        coolant_mask = self._build_coolant_mask()
        q_dt = q_source * dt / (self.rho * self.cp)

        # Biot number for convective BC
        Bi_x = self.h_conv * self.dx / self.k
        Bi_y = self.h_conv * self.dy / self.k

        T = np.full((self.ny, self.nx), self.T_init)

        snapshot_interval = max(n_steps // n_snapshots, 1)
        snapshots = []
        snap_times = []
        max_temp = self.T_init

        for n in range(n_steps + 1):
            t = n * dt

            if n % snapshot_interval == 0 or n == n_steps:
                snapshots.append(T.copy())
                snap_times.append(t)

            t_max = np.max(T)
            if t_max > max_temp:
                max_temp = t_max

            if n == n_steps:
                break

            # Explicit update interior
            T_new = T.copy()
            T_new[1:-1, 1:-1] = (
                T[1:-1, 1:-1]
                + Fx * (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2])
                + Fy * (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1])
                + q_dt[1:-1, 1:-1]
            )

            # Convective BCs (Robin): -k dT/dn = h*(T - T_amb)
            # Bottom
            T_new[0, 1:-1] = T_new[1, 1:-1] / (1.0 + Bi_y) + Bi_y * self.T_coolant / (1.0 + Bi_y)
            # Top
            T_new[-1, 1:-1] = T_new[-2, 1:-1] / (1.0 + Bi_y) + Bi_y * self.T_coolant / (1.0 + Bi_y)
            # Left
            T_new[1:-1, 0] = T_new[1:-1, 1] / (1.0 + Bi_x) + Bi_x * self.T_coolant / (1.0 + Bi_x)
            # Right
            T_new[1:-1, -1] = T_new[1:-1, -2] / (1.0 + Bi_x) + Bi_x * self.T_coolant / (1.0 + Bi_x)
            # Corners
            T_new[0, 0] = 0.5 * (T_new[0, 1] + T_new[1, 0])
            T_new[0, -1] = 0.5 * (T_new[0, -2] + T_new[1, -1])
            T_new[-1, 0] = 0.5 * (T_new[-1, 1] + T_new[-2, 0])
            T_new[-1, -1] = 0.5 * (T_new[-1, -2] + T_new[-2, -1])

            # Enforce coolant channel temperature (Dirichlet)
            T_new[coolant_mask] = self.T_coolant

            T = T_new

        # Post-processing
        grad_x = np.zeros_like(T)
        grad_y = np.zeros_like(T)
        grad_x[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2.0 * self.dx)
        grad_y[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * self.dy)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        idx_flat = np.argmax(T)
        jy_hot, ix_hot = np.unravel_index(idx_flat, T.shape)
        hotspot = (float(self.x[ix_hot]), float(self.y[jy_hot]))

        return AutomotiveThermal2DSolverResult(
            temperature=T,
            temperature_history=np.array(snapshots),
            x=self.x,
            y=self.y,
            times=np.array(snap_times),
            max_temperature=float(max_temp),
            mean_temperature=float(np.mean(T)),
            thermal_gradient=grad_mag,
            hotspot_location=hotspot,
            cooling_adequate=bool(max_temp < self.T_limit),
        )

    def export_state(
        self, result: AutomotiveThermal2DSolverResult
    ) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="automotive_thermal_2d")
        state.set_field("temperature", result.temperature, "K")
        state.metadata["max_temperature"] = result.max_temperature
        state.metadata["mean_temperature"] = result.mean_temperature
        state.metadata["cooling_adequate"] = result.cooling_adequate
        state.metadata["hotspot_location"] = result.hotspot_location
        return state
