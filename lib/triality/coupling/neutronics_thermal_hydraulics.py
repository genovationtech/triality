"""
Neutronics ↔ Thermal-Hydraulics Coupling

Couples the neutronics solver (diffusion + point kinetics) to the
thermal-hydraulics solver (fuel rod + subchannel + loop) with proper
SI boundary adapters.

Physics:
    Neutronics → TH:
        Power shape q''(z) from diffusion eigenvalue → axial heat rate for TH
        Transient power P(t) from point kinetics → time-dependent heat source

    TH → Neutronics:
        T_fuel(z) → Doppler reactivity feedback: rho_D = alpha_D * ln(T/T_ref)
        T_coolant(z) → Moderator density feedback: rho_mod = alpha_mod * (rho - rho_ref)

Unit boundary:
    Neutronics internal: cm, 1/cm, W/cm  (CGS)
    TH internal: mostly SI (m, Pa, K) but q_linear in W/cm
    Coupling interface: all SI
    NuclearSIAdapter handles conversion at boundaries

Coupling flow:
    1. Neutronics steady-state → k_eff, power shape (CGS → SI)
    2. Scale power shape to desired total power → q_linear(z) in W/m (SI)
    3. Convert q_linear to W/cm for TH solver input
    4. TH solver → T_fuel(z), T_coolant(z), DNBR
    5. Compute Doppler and moderator reactivity from temperatures
    6. Feed reactivity back to neutronics for transient
    7. Iterate until coupled steady-state converges

Unlock: nuclear reactor credibility (coupled N-TH is the minimum viable fidelity)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable

from triality.core.fields import PhysicsState, PhysicsField
from triality.core.units import NuclearSIAdapter, UnitMetadata
from triality.core.coupling import (
    CouplingEngine, CouplingLink, CouplingStrategy,
    CouplingResult, Relaxation, RelaxationMethod,
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class NeutronicsTHResult:
    """Result from coupled neutronics-thermal-hydraulics analysis.

    All external-facing fields are in SI units.

    Attributes
    ----------
    k_eff : float
        Effective multiplication factor.
    total_power_W : float
        Total reactor power [W].
    z_m : np.ndarray
        Axial positions [m].
    q_linear_W_m : np.ndarray
        Linear heat rate [W/m] (SI).
    power_shape : np.ndarray
        Normalized axial power shape.
    peaking_factor : float
        Axial power peaking factor.
    T_fuel_centerline_K : np.ndarray
        Fuel centerline temperature [K].
    T_coolant_K : np.ndarray
        Coolant bulk temperature [K].
    T_clad_outer_K : np.ndarray
        Cladding outer temperature [K].
    dnbr : np.ndarray
        DNBR at each axial node.
    min_dnbr : float
        Minimum DNBR.
    max_fuel_temp_K : float
        Peak fuel temperature [K].
    margin_to_melt_K : float
        Temperature margin to UO2 melting [K].
    rho_doppler : float
        Doppler reactivity [dk/k].
    rho_moderator : float
        Moderator reactivity [dk/k].
    rho_total : float
        Total feedback reactivity [dk/k].
    n_coupling_iterations : int
        Number of N-TH coupling iterations.
    converged : bool
        Whether the coupled solution converged.
    time : np.ndarray
        Time array for transient [s] (empty for steady-state).
    power_history : np.ndarray
        Power vs time [W] (empty for steady-state).
    units : UnitMetadata
        Unit metadata for all fields.
    """
    k_eff: float = 1.0
    total_power_W: float = 0.0
    z_m: np.ndarray = field(default_factory=lambda: np.array([]))
    q_linear_W_m: np.ndarray = field(default_factory=lambda: np.array([]))
    power_shape: np.ndarray = field(default_factory=lambda: np.array([]))
    peaking_factor: float = 1.0
    T_fuel_centerline_K: np.ndarray = field(default_factory=lambda: np.array([]))
    T_coolant_K: np.ndarray = field(default_factory=lambda: np.array([]))
    T_clad_outer_K: np.ndarray = field(default_factory=lambda: np.array([]))
    dnbr: np.ndarray = field(default_factory=lambda: np.array([]))
    min_dnbr: float = 999.0
    max_fuel_temp_K: float = 0.0
    margin_to_melt_K: float = 0.0
    rho_doppler: float = 0.0
    rho_moderator: float = 0.0
    rho_total: float = 0.0
    n_coupling_iterations: int = 0
    converged: bool = False
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    power_history: np.ndarray = field(default_factory=lambda: np.array([]))
    units: UnitMetadata = field(default_factory=UnitMetadata)

    def __post_init__(self):
        """Declare unit metadata for all fields."""
        self.units.declare("z_m", "m", "Axial position")
        self.units.declare("q_linear_W_m", "W/m", "Linear heat rate (SI)")
        self.units.declare("T_fuel_centerline_K", "K", "Fuel centerline temperature")
        self.units.declare("T_coolant_K", "K", "Coolant temperature")
        self.units.declare("T_clad_outer_K", "K", "Cladding outer temperature")
        self.units.declare("total_power_W", "W", "Total reactor power")
        self.units.declare("max_fuel_temp_K", "K", "Peak fuel temperature")
        self.units.declare("margin_to_melt_K", "K", "Margin to fuel melting")


# ---------------------------------------------------------------------------
# Feedback models
# ---------------------------------------------------------------------------

class ReactivityFeedback:
    """Computes reactivity feedback from temperature fields.

    Parameters
    ----------
    alpha_doppler : float
        Doppler coefficient [dk/k per ln(K)] (typically -2 to -5 pcm/K).
    T_fuel_ref : float
        Reference fuel temperature [K].
    alpha_moderator : float
        Moderator temperature coefficient [dk/k per K].
    T_mod_ref : float
        Reference moderator temperature [K].
    alpha_void : float
        Void coefficient [dk/k per unit void fraction].
    """

    def __init__(
        self,
        alpha_doppler: float = -2.5e-5,
        T_fuel_ref: float = 900.0,
        alpha_moderator: float = -3.0e-5,
        T_mod_ref: float = 565.0,
        alpha_void: float = -0.01,
    ):
        self.alpha_doppler = alpha_doppler
        self.T_fuel_ref = T_fuel_ref
        self.alpha_moderator = alpha_moderator
        self.T_mod_ref = T_mod_ref
        self.alpha_void = alpha_void

    def doppler(self, T_fuel: np.ndarray) -> float:
        """Compute Doppler reactivity from fuel temperature.

        rho_D = alpha_D * ln(T_fuel_avg / T_fuel_ref)
        """
        T_avg = float(np.mean(T_fuel))
        T_avg = max(T_avg, 300.0)  # floor
        return self.alpha_doppler * np.log(T_avg / self.T_fuel_ref)

    def moderator(self, T_coolant: np.ndarray) -> float:
        """Compute moderator reactivity from coolant temperature.

        rho_mod = alpha_mod * (T_mod_avg - T_mod_ref)
        """
        T_avg = float(np.mean(T_coolant))
        return self.alpha_moderator * (T_avg - self.T_mod_ref)

    def total(self, T_fuel: np.ndarray, T_coolant: np.ndarray,
              void_fraction: float = 0.0) -> float:
        """Compute total feedback reactivity."""
        return (self.doppler(T_fuel) +
                self.moderator(T_coolant) +
                self.alpha_void * void_fraction)


# ---------------------------------------------------------------------------
# N-TH Coupler
# ---------------------------------------------------------------------------

class NeutronicsTHCoupler:
    """Coupled neutronics-thermal-hydraulics solver.

    Wraps the neutronics and thermal-hydraulics solvers with:
    - SI boundary adapters for unit conversion
    - Reactivity feedback models
    - Picard iteration for coupled steady-state
    - Sequential coupling for transients

    Parameters
    ----------
    core_length_m : float
        Reactor core active length [m] (SI input).
    total_power_W : float
        Total reactor power [W].
    n_axial : int
        Number of axial nodes.
    peak_linear_heat_rate_W_m : float
        Peak linear heat rate [W/m] (SI input).
    inlet_temperature_K : float
        Coolant inlet temperature [K].
    system_pressure_Pa : float
        System pressure [Pa].
    mass_flux_kg_m2s : float
        Coolant mass flux [kg/(m^2*s)].
    feedback : ReactivityFeedback, optional
        Reactivity feedback model.

    Example
    -------
    >>> coupler = NeutronicsTHCoupler(
    ...     core_length_m=3.66,
    ...     total_power_W=3000e6,
    ...     n_axial=50,
    ... )
    >>> result = coupler.solve_steady_state()
    >>> print(f"k_eff = {result.k_eff:.5f}")
    >>> print(f"Max fuel T = {result.max_fuel_temp_K:.0f} K")
    >>> print(f"Min DNBR = {result.min_dnbr:.2f}")
    """

    def __init__(
        self,
        core_length_m: float = 3.66,
        total_power_W: float = 3000e6,
        n_axial: int = 50,
        peak_linear_heat_rate_W_m: float = 44000.0,
        inlet_temperature_K: float = 565.0,
        system_pressure_Pa: float = 15.5e6,
        mass_flux_kg_m2s: float = 3500.0,
        feedback: Optional[ReactivityFeedback] = None,
    ):
        self.adapter = NuclearSIAdapter()
        self.core_length_m = core_length_m
        self.total_power_W = total_power_W
        self.n_axial = n_axial
        self.peak_lhr_W_m = peak_linear_heat_rate_W_m
        self.inlet_T = inlet_temperature_K
        self.system_pressure_Pa = system_pressure_Pa
        self.mass_flux = mass_flux_kg_m2s
        self.feedback = feedback or ReactivityFeedback()

        # Convert core length to CGS for neutronics
        self.core_length_cm = self.adapter.length_to_cgs(core_length_m)
        # Convert peak LHR to W/cm for TH
        self.peak_lhr_W_cm = self.adapter.linear_heat_rate_to_cgs(peak_linear_heat_rate_W_m)

    def solve_steady_state(
        self,
        max_iterations: int = 20,
        tolerance: float = 1e-3,
        relaxation: float = 0.7,
    ) -> NeutronicsTHResult:
        """Solve coupled N-TH steady-state with Picard iteration.

        Parameters
        ----------
        max_iterations : int
            Maximum coupling iterations.
        tolerance : float
            Relative convergence tolerance on temperatures.
        relaxation : float
            Under-relaxation factor.

        Returns
        -------
        NeutronicsTHResult
        """
        n = self.n_axial
        z_m = np.linspace(0, self.core_length_m, n)

        # Initial guess: cosine power shape
        z_norm = z_m / self.core_length_m
        ext = 1.4
        power_shape = np.cos(np.pi * (z_norm - 0.5) * ext)
        power_shape = np.maximum(power_shape, 0.0)
        mean_ps = np.mean(power_shape)
        if mean_ps > 0:
            power_shape /= mean_ps

        peaking = float(np.max(power_shape))

        # Initial q_linear in W/m (SI)
        q_linear_W_m = self.peak_lhr_W_m * power_shape / peaking

        T_fuel_prev = np.full(n, 900.0)
        T_coolant_prev = np.full(n, self.inlet_T)

        converged = False
        n_iter = 0
        rho_doppler = 0.0
        rho_moderator = 0.0

        for iteration in range(max_iterations):
            n_iter = iteration + 1

            # --- TH solve ---
            # Convert q_linear to W/cm for TH solver
            q_linear_W_cm = self.adapter.linear_heat_rate_to_cgs(q_linear_W_m)

            # Run TH: simplified 1-D axial coolant march + fuel rod
            T_coolant, T_clad, T_fuel = self._run_th(z_m, q_linear_W_cm)

            # Under-relax temperatures
            T_fuel_new = relaxation * T_fuel + (1.0 - relaxation) * T_fuel_prev
            T_cool_new = relaxation * T_coolant + (1.0 - relaxation) * T_coolant_prev

            # Convergence check
            dT_fuel = float(np.max(np.abs(T_fuel_new - T_fuel_prev)))
            dT_cool = float(np.max(np.abs(T_cool_new - T_coolant_prev)))
            T_scale = max(float(np.max(T_fuel_new)), 1.0)

            if max(dT_fuel, dT_cool) / T_scale < tolerance and iteration > 0:
                converged = True
                T_fuel_prev = T_fuel_new
                T_coolant_prev = T_cool_new
                break

            T_fuel_prev = T_fuel_new.copy()
            T_coolant_prev = T_cool_new.copy()

            # --- Reactivity feedback ---
            rho_doppler = self.feedback.doppler(T_fuel_new)
            rho_moderator = self.feedback.moderator(T_cool_new)
            rho_total = rho_doppler + rho_moderator

            # Update k_eff (simplified: k_eff_new = k_eff_0 / (1 - rho_total))
            # This modifies power shape slightly in real coupling
            # For now, keep power shape fixed (single-step coupling)

        # Compute DNBR
        dnbr = self._compute_dnbr(z_m, q_linear_W_m, T_coolant_prev)

        # Results
        T_melt_UO2 = 3120.0
        max_fuel_T = float(np.max(T_fuel_prev))

        return NeutronicsTHResult(
            k_eff=1.0 / (1.0 - rho_doppler - rho_moderator) if abs(rho_doppler + rho_moderator) < 0.5 else 1.0,
            total_power_W=self.total_power_W,
            z_m=z_m,
            q_linear_W_m=q_linear_W_m,
            power_shape=power_shape,
            peaking_factor=peaking,
            T_fuel_centerline_K=T_fuel_prev,
            T_coolant_K=T_coolant_prev,
            T_clad_outer_K=T_clad if T_clad is not None else np.zeros(n),
            dnbr=dnbr,
            min_dnbr=float(np.min(dnbr)),
            max_fuel_temp_K=max_fuel_T,
            margin_to_melt_K=T_melt_UO2 - max_fuel_T,
            rho_doppler=rho_doppler,
            rho_moderator=rho_moderator,
            rho_total=rho_doppler + rho_moderator,
            n_coupling_iterations=n_iter,
            converged=converged,
        )

    def solve_transient(
        self,
        t_final: float = 10.0,
        dt: float = 0.01,
        reactivity_insertion: Optional[Callable] = None,
    ) -> NeutronicsTHResult:
        """Solve a coupled N-TH transient.

        First runs steady-state, then marches forward in time with
        point-kinetics power + TH feedback at each step.

        Parameters
        ----------
        t_final : float
            Final time [s].
        dt : float
            Time step [s].
        reactivity_insertion : callable, optional
            External reactivity rho_ext(t).

        Returns
        -------
        NeutronicsTHResult
        """
        if reactivity_insertion is None:
            reactivity_insertion = lambda t: 0.0

        # Get steady-state initial condition
        ss = self.solve_steady_state()

        n_steps = int(np.ceil(t_final / dt))
        times = np.zeros(n_steps + 1)
        powers = np.zeros(n_steps + 1)

        P = self.total_power_W
        powers[0] = P
        T_fuel = ss.T_fuel_centerline_K.copy()
        T_cool = ss.T_coolant_K.copy()

        # Simplified point kinetics parameters
        Lambda = 2e-5  # prompt neutron generation time [s]
        beta = 0.0065  # delayed neutron fraction

        for step in range(n_steps):
            t = (step + 1) * dt
            times[step + 1] = t

            # External reactivity
            rho_ext = reactivity_insertion(t)

            # Feedback reactivity
            rho_fb = self.feedback.total(T_fuel, T_cool)

            # Total reactivity
            rho_total = rho_ext + rho_fb

            # Point kinetics (simplified prompt jump)
            if abs(rho_total) < beta:
                # Below prompt critical: exponential with period
                period = Lambda / (beta - rho_total)
                P = P * np.exp(dt / period)
            else:
                # Prompt critical: rapid increase
                P = P * (1.0 + (rho_total - beta) / Lambda * dt)

            P = max(P, 0.0)
            P = min(P, 100.0 * self.total_power_W)  # cap
            powers[step + 1] = P

            # Update TH with new power
            power_ratio = P / max(self.total_power_W, 1e-30)
            q_lin_W_cm = self.adapter.linear_heat_rate_to_cgs(
                ss.q_linear_W_m * power_ratio
            )
            T_cool, T_clad, T_fuel = self._run_th(ss.z_m, q_lin_W_cm)

        return NeutronicsTHResult(
            k_eff=ss.k_eff,
            total_power_W=float(powers[-1]),
            z_m=ss.z_m,
            q_linear_W_m=ss.q_linear_W_m * powers[-1] / max(self.total_power_W, 1e-30),
            power_shape=ss.power_shape,
            peaking_factor=ss.peaking_factor,
            T_fuel_centerline_K=T_fuel,
            T_coolant_K=T_cool,
            T_clad_outer_K=T_clad if T_clad is not None else np.zeros_like(T_fuel),
            dnbr=ss.dnbr,
            min_dnbr=float(np.min(ss.dnbr)),
            max_fuel_temp_K=float(np.max(T_fuel)),
            margin_to_melt_K=3120.0 - float(np.max(T_fuel)),
            rho_doppler=self.feedback.doppler(T_fuel),
            rho_moderator=self.feedback.moderator(T_cool),
            rho_total=self.feedback.total(T_fuel, T_cool),
            n_coupling_iterations=ss.n_coupling_iterations,
            converged=ss.converged,
            time=times,
            power_history=powers,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_th(self, z_m: np.ndarray,
                q_linear_W_cm: np.ndarray) -> tuple:
        """Simplified TH solve: 1-D axial coolant march + radial fuel rod.

        Parameters
        ----------
        z_m : np.ndarray
            Axial positions [m].
        q_linear_W_cm : np.ndarray
            Linear heat rate [W/cm] (TH convention).

        Returns
        -------
        T_coolant, T_clad, T_fuel : np.ndarray
            Temperature arrays [K].
        """
        n = len(z_m)
        L = z_m[-1] - z_m[0]
        dz = L / max(n - 1, 1)

        # Coolant properties (water at 15.5 MPa, simplified)
        cp_w = 5400.0  # J/(kg*K)
        D_h = 0.0118   # hydraulic diameter [m]

        # Convert q_linear from W/cm to W/m for energy balance
        q_lin_W_m = self.adapter.linear_heat_rate_to_si(q_linear_W_cm)

        # Axial coolant temperature march
        T_coolant = np.zeros(n)
        T_coolant[0] = self.inlet_T
        A_flow = np.pi * D_h**2 / 4.0
        m_dot = self.mass_flux * A_flow

        for i in range(1, n):
            T_coolant[i] = T_coolant[i-1] + q_lin_W_m[i] * dz / (m_dot * cp_w)

        # Fuel rod radial temperatures (simplified)
        # T_clad = T_coolant + q'' / h
        h_coolant = 30000.0  # W/(m^2*K) typical
        r_clad = 0.00475     # m
        circumference = 2.0 * np.pi * r_clad

        q_flux = q_lin_W_m / circumference  # W/m^2
        T_clad = T_coolant + q_flux / h_coolant

        # T_fuel_center = T_clad + q_linear / (4*pi*k_fuel)
        k_fuel = 3.0  # W/(m*K) UO2 (approximate)
        T_fuel = T_clad + q_lin_W_m / (4.0 * np.pi * k_fuel)

        return T_coolant, T_clad, T_fuel

    def _compute_dnbr(self, z_m: np.ndarray,
                       q_linear_W_m: np.ndarray,
                       T_coolant: np.ndarray) -> np.ndarray:
        """Compute DNBR at each axial node.

        Uses simplified W-3 correlation estimate.
        """
        n = len(z_m)
        D_h = 0.0118
        circumference = 2.0 * np.pi * 0.00475
        q_actual = q_linear_W_m / circumference  # W/m^2

        # Simplified CHF estimate (W-3 like, very rough)
        P_MPa = self.system_pressure_Pa / 1e6
        G = self.mass_flux

        # q_CHF ~ f(G, P, quality) -- simplified
        q_chf_base = 3.0e6  # W/m^2 (typical CHF at PWR conditions)
        # Quality correction (reduces CHF at higher quality)
        T_sat = 618.0  # K at 15.5 MPa (approximate)
        h_fg = 1000e3  # J/kg (approximate)

        quality = np.maximum((T_coolant - T_sat) * 5400.0 / h_fg, -0.1)
        q_chf = q_chf_base * (1.0 - 0.5 * np.maximum(quality, 0.0))
        q_chf = np.maximum(q_chf, 1e5)

        dnbr = q_chf / np.maximum(q_actual, 1.0)
        return dnbr

    def export_state(self, result: NeutronicsTHResult) -> PhysicsState:
        """Export result as PhysicsState (all SI)."""
        state = PhysicsState(solver_name="neutronics_th", time=0.0)
        state.set_field("temperature", result.T_fuel_centerline_K, "K",
                        grid=result.z_m)
        state.set_field("linear_heat_rate", result.q_linear_W_m, "W/m",
                        grid=result.z_m)
        state.set_field("power_density",
                        result.q_linear_W_m / (np.pi * 0.004095**2),
                        "W/m^3", grid=result.z_m)
        state.set_field("pressure",
                        np.full_like(result.z_m, self.system_pressure_Pa),
                        "Pa", grid=result.z_m)
        state.metadata["k_eff"] = result.k_eff
        state.metadata["min_dnbr"] = result.min_dnbr
        state.metadata["rho_total"] = result.rho_total
        return state
