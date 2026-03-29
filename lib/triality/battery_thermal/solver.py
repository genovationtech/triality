"""
Battery Thermal Management Solver

System-level solver that wires together BatteryPackThermalModel,
PackConfiguration, and BatteryCell to perform multi-scenario battery
thermal analysis including drive-cycle simulation, cooling system
sizing, and thermal runaway risk assessment.

The solver orchestrates:
1. Time-varying current profiles (drive cycles)
2. Pack thermal model (cell energy balance with I^2R heating)
3. Cooling effectiveness evaluation
4. Thermal runaway propagation tracking
5. Post-processed safety metrics

Governing equation per cell:
    m*c*dT_i/dt = I^2*R_i + Q_cal
                  - h*A*(T_i - T_coolant)
                  - sum_j k_eff*A_c*(T_i - T_j)/d_ij
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from triality.battery_thermal.pack_thermal import (
    BatteryPackThermalModel,
    BatteryCell,
    PackConfiguration,
    ThermalRunawayEvent,
    PackThermalResult,
    CellChemistry,
    CoolingType,
    evaluate_cooling_adequacy,
)


@dataclass
class DriveCycleSegment:
    """A segment of a drive-cycle current profile.

    Parameters
    ----------
    duration : float
        Segment duration [s].
    current : float
        Constant current during this segment [A] (positive = discharge).
    label : str
        Human-readable label (e.g. "acceleration", "cruise", "regen").
    """
    duration: float
    current: float
    label: str = ""


@dataclass
class BatteryThermalSolverResult:
    """Aggregated results from the battery thermal solver.

    Attributes
    ----------
    pack_result : PackThermalResult
        Raw result from the underlying pack model.
    drive_cycle_labels : list of str
        Labels for each drive-cycle segment.
    segment_boundaries : list of float
        Time boundaries between drive-cycle segments [s].
    max_cell_temperature : float
        Peak temperature across all cells and times [K].
    min_cell_temperature : float
        Minimum temperature across all cells at end [K].
    temperature_spread : float
        Max - min cell temperature at end of simulation [K].
    cooling_power_W : float
        Estimated average cooling power removed [W].
    runaway_risk : bool
        True if any cell exceeded runaway threshold.
    safety_score : float
        0-1 score (1 = safe).  Based on margin to runaway threshold.
    """
    pack_result: PackThermalResult
    drive_cycle_labels: List[str]
    segment_boundaries: List[float]
    max_cell_temperature: float
    min_cell_temperature: float
    temperature_spread: float
    cooling_power_W: float
    runaway_risk: bool
    safety_score: float


class BatteryThermalSolver:
    """High-level battery thermal management solver.

    Wraps BatteryPackThermalModel and provides drive-cycle simulation,
    cooling sizing sweeps, and safety assessment utilities.

    Parameters
    ----------
    config : PackConfiguration
        Pack-level configuration (cell count, spacing, cooling).
    cell_chemistry : CellChemistry
        Chemistry type for runaway threshold selection.
    T_init : float
        Initial uniform cell temperature [K].
    """

    fidelity_tier = FidelityTier.REDUCED_ORDER
    coupling_maturity = CouplingMaturity.M3_COUPLED

    # Thermal runaway onset temperatures by chemistry [K]
    _RUNAWAY_THRESHOLDS = {
        CellChemistry.NMC: 403.0,   # 130 degC
        CellChemistry.NCA: 403.0,   # 130 degC
        CellChemistry.LFP: 423.0,   # 150 degC
    }

    def __init__(
        self,
        config: PackConfiguration,
        cell_chemistry: CellChemistry = CellChemistry.NMC,
        T_init: float = 298.0,
    ):
        self.config = config
        self.cell_chemistry = cell_chemistry
        self.T_init = T_init
        self._coupled_state = None
        self._time = 0.0
        self._runaway_threshold = self._RUNAWAY_THRESHOLDS.get(
            cell_chemistry, 403.0
        )

    def _build_current_profile(
        self, segments: List[DriveCycleSegment], dt: float
    ) -> Tuple[np.ndarray, List[str], List[float]]:
        """Convert drive-cycle segments to a sampled current array.

        Returns
        -------
        current_profile : np.ndarray
            Sampled current values [A].
        labels : list of str
            Segment labels.
        boundaries : list of float
            Cumulative time boundaries [s].
        """
        labels = [s.label for s in segments]
        boundaries = [0.0]
        samples: List[float] = []
        for seg in segments:
            n_samples = max(int(seg.duration / dt), 1)
            samples.extend([seg.current] * n_samples)
            boundaries.append(boundaries[-1] + seg.duration)
        return np.array(samples, dtype=float), labels, boundaries

    def solve(
        self,
        segments: List[DriveCycleSegment],
        dt: float = 0.1,
        progress_callback=None,
    ) -> BatteryThermalSolverResult:
        """Run the drive-cycle thermal simulation.

        Parameters
        ----------
        segments : list of DriveCycleSegment
            Ordered drive-cycle segments.
        dt : float
            Integration time step [s].

        Returns
        -------
        BatteryThermalSolverResult
        """
        t_end = sum(s.duration for s in segments)
        current_profile, labels, boundaries = self._build_current_profile(
            segments, dt
        )

        # Build fresh pack model with initial temperature
        pack = BatteryPackThermalModel(self.config)
        for cell in pack.cells:
            cell.temperature = self.T_init

        # Run transient
        pack_result = pack.solve_transient(
            current_profile=current_profile,
            t_end=t_end,
            dt=dt,
            progress_callback=progress_callback,
        )

        # Post-process ---------------------------------------------------
        T_final = pack_result.temperatures[-1, :]
        max_T = float(pack_result.max_temperature)
        min_T = float(np.min(T_final))
        spread = float(np.max(T_final) - np.min(T_final))

        # Estimate average cooling power (h*A*deltaT summed over cells)
        avg_T = float(np.mean(pack_result.temperatures[-1, :]))
        cooling_power = (
            self.config.h_convection
            * pack.cells[0].surface_area
            * self.config.n_cells
            * max(avg_T - self.config.T_coolant, 0.0)
        )

        runaway_risk = max_T >= self._runaway_threshold
        margin = self._runaway_threshold - max_T
        safety_score = float(np.clip(margin / 100.0, 0.0, 1.0))

        return BatteryThermalSolverResult(
            pack_result=pack_result,
            drive_cycle_labels=labels,
            segment_boundaries=boundaries,
            max_cell_temperature=max_T,
            min_cell_temperature=min_T,
            temperature_spread=spread,
            cooling_power_W=cooling_power,
            runaway_risk=runaway_risk,
            safety_score=safety_score,
        )

    def export_state(self, result: BatteryThermalSolverResult) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="battery_thermal")
        state.set_field("temperature", result.pack_result.temperatures[-1, :], "K")
        state.metadata["max_cell_temperature"] = result.max_cell_temperature
        state.metadata["min_cell_temperature"] = result.min_cell_temperature
        state.metadata["temperature_spread"] = result.temperature_spread
        state.metadata["cooling_power_W"] = result.cooling_power_W
        state.metadata["runaway_risk"] = result.runaway_risk
        state.metadata["safety_score"] = result.safety_score
        return state

    def import_state(self, state: PhysicsState) -> None:
        """Import coupled fields from electrochemistry solver (heat source)."""
        self._coupled_state = state

    def advance(self, dt: float) -> PhysicsState:
        """Advance battery thermal solver by dt for closed-loop coupling."""
        current = 50.0  # default discharge current [A]
        if self._coupled_state is not None:
            if self._coupled_state.has("heat_source"):
                Q = self._coupled_state.get_field("heat_source").data
                # Estimate equivalent current from heat generation: Q ~ I^2*R
                R_est = 0.01
                Q_mean = float(np.mean(np.abs(Q)))
                current = np.sqrt(max(Q_mean * 4e-4 / R_est, 0.0))
        segments = [DriveCycleSegment(duration=dt, current=current,
                                      label="coupled")]
        result = self.solve(segments=segments, dt=min(dt, 0.1))
        self._time += dt
        return self.export_state(result)

    def sweep_cooling(
        self,
        current_A: float,
        duration_s: float,
        h_values: List[float],
    ) -> List[Dict]:
        """Sweep convection coefficient to size the cooling system.

        For each h value, runs a constant-current simulation and
        evaluates cooling adequacy.

        Parameters
        ----------
        current_A : float
            Constant discharge current [A].
        duration_s : float
            Simulation duration [s].
        h_values : list of float
            Convection coefficients to test [W/(m^2*K)].

        Returns
        -------
        list of dict
            One result dict per h value with temperature and pass/fail.
        """
        results = []
        for h in h_values:
            cfg = PackConfiguration(
                n_cells=self.config.n_cells,
                cell_spacing=self.config.cell_spacing,
                cooling_type=self.config.cooling_type,
                h_convection=h,
                T_coolant=self.config.T_coolant,
                thermal_barriers=self.config.thermal_barriers,
            )
            pack = BatteryPackThermalModel(cfg)
            for cell in pack.cells:
                cell.temperature = self.T_init

            n_steps = int(duration_s / 0.1)
            profile = np.ones(n_steps) * current_A
            res = pack.solve_transient(profile, duration_s, dt=0.1)

            results.append({
                "h_convection": h,
                "max_temperature_K": res.max_temperature,
                "max_temperature_C": res.max_temperature - 273.15,
                "temp_delta_K": res.max_temp_delta,
                "cooling_adequate": res.cooling_adequate,
            })
        return results


# ======================================================================
# 2D Result Dataclass
# ======================================================================

@dataclass
class BatteryThermal2DResult:
    """Container for 2D spatially-resolved battery thermal results.

    Represents a cross-section of a battery cell or module with internal
    heat generation from electrochemical discharge.

    Attributes
    ----------
    time : np.ndarray
        Time stamps [s], shape (n_save,).
    temperature : np.ndarray
        Temperature field [K], shape (n_save, ny, nx).
    heat_generation : np.ndarray
        Volumetric heat generation field [W/m^3], shape (n_save, ny, nx).
    max_temperature : float
        Peak temperature across all space and time [K].
    min_temperature : float
        Minimum temperature at final time [K].
    temperature_spread : float
        Max - min temperature at final time [K].
    x : np.ndarray
    y : np.ndarray
    """
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature: np.ndarray = field(default_factory=lambda: np.array([]))
    heat_generation: np.ndarray = field(default_factory=lambda: np.array([]))
    max_temperature: float = 0.0
    min_temperature: float = 0.0
    temperature_spread: float = 0.0
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))


# ======================================================================
# 2D Spatially-Resolved Battery Thermal Solver  (Level 3)
# ======================================================================

class BatteryThermal2DSolver:
    """2D spatially-resolved thermal solver for a battery cross-section.

    Solves the 2D heat equation on a rectangular domain representing a
    battery cell/module cross-section with internal volumetric heat
    generation from I^2*R Joule heating and entropic heating.

    Governing PDE:

        rho * cp * dT/dt = kx * d^2T/dx^2 + ky * d^2T/dy^2 + Q_gen(x,y,t)

    Anisotropic conductivity is supported (kx != ky) to model the
    layered electrode/separator structure.

    Boundary conditions: convective cooling on all four sides with
    independent heat transfer coefficients.

    Parameters
    ----------
    Lx, Ly : float
        Domain extents [m]. E.g. cell width x cell height.
    nx, ny : int
        Grid nodes in x and y.
    rho : float
        Effective density [kg/m^3].
    cp : float
        Effective specific heat [J/(kg K)].
    kx, ky : float
        Thermal conductivity in x and y [W/(m K)].
    T_init : float
        Initial temperature [K].
    T_coolant : float
        Coolant temperature [K].
    h_conv : float
        Convective HTC on boundaries [W/(m^2 K)].
    I_discharge : float
        Discharge current [A].
    R_internal : float
        Internal resistance [Ohm].
    cell_volume : float
        Cell volume [m^3] for volumetric heat generation normalisation.
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M3_COUPLED

    def __init__(
        self,
        Lx: float = 0.20,
        Ly: float = 0.10,
        nx: int = 41,
        ny: int = 21,
        rho: float = 2500.0,
        cp: float = 1000.0,
        kx: float = 30.0,
        ky: float = 1.0,
        T_init: float = 298.0,
        T_coolant: float = 293.0,
        h_conv: float = 50.0,
        I_discharge: float = 100.0,
        R_internal: float = 0.010,
        cell_volume: float = 4.0e-4,
    ):
        self._coupled_state = None
        self._time = 0.0
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.rho = rho
        self.cp = cp
        self.kx = kx
        self.ky = ky
        self.T_init = T_init
        self.T_coolant = T_coolant
        self.h_conv = h_conv
        self.I_discharge = I_discharge
        self.R_internal = R_internal
        self.cell_volume = cell_volume

        self.dx = Lx / max(nx - 1, 1)
        self.dy = Ly / max(ny - 1, 1)
        self.x = np.linspace(0.0, Lx, nx)
        self.y = np.linspace(0.0, Ly, ny)

        # Effective thermal diffusivity (use max for CFL)
        self.alpha_max = max(kx, ky) / (rho * cp)

    # ------------------------------------------------------------------
    def _cfl_dt(self, safety: float = 0.4) -> float:
        return safety / (self.alpha_max * (1.0 / self.dx**2 + 1.0 / self.dy**2))

    # ------------------------------------------------------------------
    def _heat_generation(self, T: np.ndarray, t: float) -> np.ndarray:
        """Compute volumetric heat generation [W/m^3].

        Includes I^2*R Joule heating with temperature-dependent resistance
        (linear increase ~0.5%/K) distributed uniformly, plus a localised
        hot-spot contribution near the electrode tabs (top centre).
        """
        # Temperature-dependent resistance
        R_eff = self.R_internal * (1.0 + 0.005 * (T - 298.0))
        Q_joule = self.I_discharge**2 * R_eff / self.cell_volume  # W/m^3

        # Tab heating hot-spot (Gaussian near top-centre)
        ny, nx = T.shape
        cx = nx // 2
        tab_boost = np.zeros_like(T)
        for j in range(ny):
            for i in range(nx):
                r2 = ((i - cx) * self.dx)**2 + ((j - (ny - 1)) * self.dy)**2
                sigma2 = (0.1 * self.Lx)**2
                tab_boost[j, i] = 0.5 * Q_joule[j, i] * np.exp(-r2 / (2.0 * sigma2))

        return Q_joule + tab_boost

    # ------------------------------------------------------------------
    def _apply_convective_bc(self, T: np.ndarray) -> np.ndarray:
        """Apply convective (Robin) BCs on all four sides using ghost nodes.

        -k dT/dn = h*(T - T_coolant) approximated as:
        T_boundary = (T_interior + Bi*T_coolant) / (1 + Bi)
        """
        Bi_x = self.h_conv * self.dx / self.kx
        Bi_y = self.h_conv * self.dy / self.ky

        # Left / right
        T[:, 0] = (T[:, 1] + Bi_x * self.T_coolant) / (1.0 + Bi_x)
        T[:, -1] = (T[:, -2] + Bi_x * self.T_coolant) / (1.0 + Bi_x)
        # Bottom / top
        T[0, :] = (T[1, :] + Bi_y * self.T_coolant) / (1.0 + Bi_y)
        T[-1, :] = (T[-2, :] + Bi_y * self.T_coolant) / (1.0 + Bi_y)
        return T

    # ------------------------------------------------------------------
    def solve(
        self,
        t_final: float,
        dt: Optional[float] = None,
        save_every: int = 10,
        current_profile: Optional[np.ndarray] = None,
    ) -> BatteryThermal2DResult:
        """Run the 2D battery thermal simulation.

        Parameters
        ----------
        t_final : float
            End time [s].
        dt : float or None
            Time step. If None, chosen from CFL.
        save_every : int
            Store output every *save_every* steps.
        current_profile : np.ndarray, optional
            Time-varying current [A]. If provided, linearly interpolated.

        Returns
        -------
        BatteryThermal2DResult
        """
        ny, nx = self.ny, self.nx
        dx, dy = self.dx, self.dy

        cfl_dt = self._cfl_dt(safety=0.4)
        if dt is None:
            dt = cfl_dt
        dt = min(dt, cfl_dt, t_final)

        T = np.full((ny, nx), self.T_init)
        T = self._apply_convective_bc(T)

        time_list, T_list, Qgen_list = [], [], []
        n_steps = int(np.ceil(t_final / dt))
        t = 0.0

        for step in range(n_steps + 1):
            if step % save_every == 0:
                time_list.append(t)
                T_list.append(T.copy())
                Qgen_list.append(self._heat_generation(T, t))

            if step == n_steps:
                break

            dt_eff = min(dt, t_final - t)
            if dt_eff <= 0:
                break

            # Update current if profile provided
            if current_profile is not None and len(current_profile) > 1:
                frac = t / t_final
                idx_f = frac * (len(current_profile) - 1)
                idx_lo = int(idx_f)
                idx_hi = min(idx_lo + 1, len(current_profile) - 1)
                w = idx_f - idx_lo
                self.I_discharge = (1.0 - w) * current_profile[idx_lo] + w * current_profile[idx_hi]

            # Heat generation
            Q_gen = self._heat_generation(T, t)

            # Explicit diffusion
            lap = np.zeros_like(T)
            lap[:, 1:-1] += self.kx * (T[:, 2:] - 2.0 * T[:, 1:-1] + T[:, :-2]) / dx**2
            lap[1:-1, :] += self.ky * (T[2:, :] - 2.0 * T[1:-1, :] + T[:-2, :]) / dy**2

            dTdt = (lap + Q_gen) / (self.rho * self.cp)
            T += dTdt * dt_eff

            # BCs
            T = self._apply_convective_bc(T)
            T = np.clip(T, 200.0, 600.0)
            t += dt_eff

        T_final = T_list[-1] if T_list else T
        return BatteryThermal2DResult(
            time=np.array(time_list),
            temperature=np.array(T_list),
            heat_generation=np.array(Qgen_list),
            max_temperature=float(np.max(np.array(T_list))),
            min_temperature=float(np.min(T_final)),
            temperature_spread=float(np.max(T_final) - np.min(T_final)),
            x=self.x,
            y=self.y,
        )

    def export_state(self, result: BatteryThermal2DResult) -> PhysicsState:
        """Export 2D result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="battery_thermal_2d")
        state.set_field("temperature", result.temperature[-1] if len(result.temperature) > 0 else np.full((self.ny, self.nx), self.T_init), "K")
        state.metadata["max_temperature"] = result.max_temperature
        state.metadata["min_temperature"] = result.min_temperature
        state.metadata["temperature_spread"] = result.temperature_spread
        return state

    def import_state(self, state: PhysicsState) -> None:
        """Import coupled fields from electrochemistry solver (heat source)."""
        self._coupled_state = state

    def advance(self, dt: float) -> PhysicsState:
        """Advance 2D battery thermal solver by dt for closed-loop coupling."""
        if self._coupled_state is not None:
            if self._coupled_state.has("heat_source"):
                Q_ext = self._coupled_state.get_field("heat_source").data
                # Scale discharge current to match imported heat source
                Q_mean = float(np.mean(np.abs(Q_ext)))
                self.I_discharge = np.sqrt(max(Q_mean * self.cell_volume / self.R_internal, 0.0))
        result = self.solve(t_final=dt)
        self._time += dt
        return self.export_state(result)
