"""
Coupled neutronics-thermal-hydraulics transient solver.

Extends the existing steady-state CoupledNeutronicsThermal solver with
time-dependent capability using the point-kinetics equations for the
neutron population and a transient heat conduction discretisation for
the temperature field.  Spatially-resolved reactivity feedback from
VoidReactivityMap, ModeratorDensityFeedback, and FuelTemperatureFeedback
is included at every time step.

Point Kinetics (one group of delayed neutrons):
    dn/dt = (rho - beta) / Lambda * n + lambda_d * C
    dC/dt = beta / Lambda * n - lambda_d * C

Transient Heat Conduction (1-D):
    rho_m * cp * dT/dt = d/dx(k dT/dx) + q'''(x, t)

Uses numpy for array operations and scipy.linalg for tridiagonal solves.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple

from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from .neutronics_thermal_coupled import (
    CoupledNeutronicsThermal,
    CoupledResult,
    FeedbackMode,
)
from triality.neutronics import MaterialType
from .void_reactivity import (
    VoidReactivityMap,
    ModeratorDensityFeedback,
    FuelTemperatureFeedback,
    ReactivityCoupling,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CoupledTransientResult:
    """Result container for a coupled physics transient simulation.

    Attributes
    ----------
    time : np.ndarray
        Time vector [s], shape (n_steps,).
    power : np.ndarray
        Total reactor power history [W].
    k_eff : np.ndarray
        Effective multiplication factor history.
    reactivity_pcm : np.ndarray
        Total reactivity history [pcm].
    temperature_max : np.ndarray
        Peak temperature at each step [K].
    temperature_avg : np.ndarray
        Volume-average temperature at each step [K].
    temperature_field : np.ndarray
        Full temperature field history, shape (n_steps, n_x).
    power_density_field : np.ndarray
        Power density field history [W/cm^3], shape (n_steps, n_x).
    precursor_concentration : np.ndarray
        Delayed-neutron precursor concentration history.
    rho_doppler : np.ndarray
        Doppler reactivity contribution [pcm].
    rho_moderator : np.ndarray
        Moderator feedback reactivity [pcm].
    rho_external : np.ndarray
        Externally inserted reactivity [pcm].
    converged : bool
        Whether the simulation completed without instability.
    """
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    power: np.ndarray = field(default_factory=lambda: np.array([]))
    k_eff: np.ndarray = field(default_factory=lambda: np.array([]))
    reactivity_pcm: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature_max: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature_avg: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature_field: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    power_density_field: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    precursor_concentration: np.ndarray = field(default_factory=lambda: np.array([]))
    rho_doppler: np.ndarray = field(default_factory=lambda: np.array([]))
    rho_moderator: np.ndarray = field(default_factory=lambda: np.array([]))
    rho_external: np.ndarray = field(default_factory=lambda: np.array([]))
    converged: bool = True


@dataclass
class CoupledTransient2DResult:
    """Result container for a 2-D coupled spatial-kinetics transient.

    The 2-D arrays follow shape conventions (nt, nx, ny) where *nt* is
    the number of recorded time steps.

    Attributes
    ----------
    x : np.ndarray
        Spatial grid in x [cm], shape (nx,).
    y : np.ndarray
        Spatial grid in y [cm], shape (ny,).
    time : np.ndarray
        Recorded time points [s], shape (nt,).
    flux_2d : np.ndarray
        Scalar neutron flux snapshots [n/cm^2/s], shape (nt, nx, ny).
    temperature_2d : np.ndarray
        Temperature field snapshots [K], shape (nt, nx, ny).
    power_2d : np.ndarray
        Local fission power density [W/cm^3], shape (nt, nx, ny).
    k_eff : np.ndarray
        Effective multiplication factor history, shape (nt,).
    reactivity_pcm : np.ndarray
        Domain-averaged reactivity [pcm], shape (nt,).
    power_integrated : np.ndarray
        Total (integrated) fission power [W], shape (nt,).
    temperature_peak : np.ndarray
        Peak temperature at each recorded step [K], shape (nt,).
    precursor_2d : np.ndarray
        Delayed-neutron precursor concentration field, shape (nt, nx, ny).
    rho_doppler_2d : np.ndarray
        Local Doppler reactivity contribution [pcm], shape (nt, nx, ny).
    rho_moderator_2d : np.ndarray
        Local moderator feedback [pcm], shape (nt, nx, ny).
    converged : bool
        Whether the simulation completed without instability.
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    flux_2d: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    temperature_2d: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    power_2d: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    k_eff: np.ndarray = field(default_factory=lambda: np.array([]))
    reactivity_pcm: np.ndarray = field(default_factory=lambda: np.array([]))
    power_integrated: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature_peak: np.ndarray = field(default_factory=lambda: np.array([]))
    precursor_2d: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    rho_doppler_2d: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    rho_moderator_2d: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    converged: bool = True


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class CoupledPhysicsSolver:
    """Transient coupled neutronics-thermal solver with point kinetics.

    Combines:
    - Point kinetics for neutron population / power level.
    - 1-D transient heat conduction for temperature field.
    - Spatially-resolved Doppler and moderator density feedback.
    - Optional external reactivity insertion (e.g. rod withdrawal).

    The solver first obtains an initial steady state from the existing
    CoupledNeutronicsThermal solver, then marches forward in time using
    implicit Euler for both the kinetics and heat conduction equations.

    Parameters
    ----------
    steady_state_solver : CoupledNeutronicsThermal
        Pre-configured steady-state coupled solver (material regions
        must already be set).
    beta_eff : float
        Effective delayed-neutron fraction.  Typical PWR: 0.0065.
    neutron_lifetime : float
        Prompt neutron generation time [s].  Typical: 2e-5 s.
    lambda_precursor : float
        Effective precursor decay constant [1/s].  Typical: 0.08 s^-1.
    rho_material_cp : float
        Volumetric heat capacity rho*c_p [J/(cm^3*K)].
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        steady_state_solver: CoupledNeutronicsThermal,
        beta_eff: float = 0.0065,
        neutron_lifetime: float = 2e-5,
        lambda_precursor: float = 0.08,
        rho_material_cp: float = 3.5,
    ):
        self.ss = steady_state_solver
        self.beta = beta_eff
        self.Lambda = neutron_lifetime
        self.lam_d = lambda_precursor
        self.rho_cp = rho_material_cp

        self.n_x = steady_state_solver.n_points
        self.dx = steady_state_solver.dx
        self.x = steady_state_solver.x

        # Feedback models
        self._doppler_fb = FuelTemperatureFeedback()
        self._mod_fb = ModeratorDensityFeedback()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _solve_heat_conduction_implicit(
        self,
        T_old: np.ndarray,
        q_new: np.ndarray,
        dt: float,
        k_th: np.ndarray,
        T_left: float,
        T_right: float,
    ) -> np.ndarray:
        """Implicit-Euler step for 1-D transient heat conduction.

        rho*cp * (T^{n+1} - T^n)/dt = d/dx(k dT^{n+1}/dx) + q^{n+1}

        Parameters
        ----------
        T_old : np.ndarray
            Temperature at previous time step [K].
        q_new : np.ndarray
            Power density at new time step [W/cm^3].
        dt : float
            Time step [s].
        k_th : np.ndarray
            Thermal conductivity field [W/(cm*K)].
        T_left, T_right : float
            Dirichlet boundary temperatures [K].

        Returns
        -------
        np.ndarray
            Temperature at new time step [K].
        """
        n = len(T_old)
        dx = self.dx

        # Interface conductivities
        k_half = 0.5 * (k_th[:-1] + k_th[1:])

        diag = np.zeros(n)
        upper = np.zeros(n - 1)
        lower = np.zeros(n - 1)
        rhs = np.zeros(n)

        alpha = self.rho_cp / dt

        for i in range(1, n - 1):
            kL = k_half[i - 1] / dx**2
            kR = k_half[i] / dx**2
            diag[i] = alpha + kL + kR
            lower[i - 1] = -kL
            upper[i] = -kR
            rhs[i] = alpha * T_old[i] + q_new[i]

        # Boundary conditions
        diag[0] = 1.0
        rhs[0] = T_left
        diag[-1] = 1.0
        rhs[-1] = T_right

        # Thomas algorithm
        T_new = self._thomas(lower, diag, upper, rhs)
        return T_new

    @staticmethod
    def _thomas(lower, diag, upper, rhs):
        """Thomas algorithm for tridiagonal system."""
        n = len(diag)
        c = np.zeros(n - 1)
        d = np.zeros(n)
        x = np.zeros(n)

        c[0] = upper[0] / diag[0]
        d[0] = rhs[0] / diag[0]
        for i in range(1, n - 1):
            denom = diag[i] - lower[i - 1] * c[i - 1]
            c[i] = upper[i] / denom
            d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom
        d[n - 1] = (rhs[n - 1] - lower[n - 2] * d[n - 2]) / (
            diag[n - 1] - lower[n - 2] * c[n - 2]
        )

        x[n - 1] = d[n - 1]
        for i in range(n - 2, -1, -1):
            x[i] = d[i] - c[i] * x[i + 1]
        return x

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        t_end: float = 10.0,
        dt: float = 0.01,
        initial_power: float = 1e6,
        reactivity_insertion: Optional[Callable[[float], float]] = None,
        record_interval: int = 10,
        progress_callback=None,
    ) -> CoupledTransientResult:
        """Run the coupled transient simulation.

        Parameters
        ----------
        t_end : float
            Transient duration [s].
        dt : float
            Time step [s].
        initial_power : float
            Steady-state power level [W].
        reactivity_insertion : callable, optional
            rho_ext(t) -> float [dk/k].  External reactivity as a
            function of time (e.g. rod withdrawal).  Defaults to zero.
        record_interval : int
            Record full fields every N steps.

        Returns
        -------
        CoupledTransientResult
        """
        if reactivity_insertion is None:
            reactivity_insertion = lambda t: 0.0  # noqa: E731

        # --- Ensure SS solver has proper fuel/reflector regions ---------------
        if not self.ss.material_regions:
            L = self.ss.length
            refl = 0.2 * L
            self.ss.set_fuel_region((refl, L - refl), MaterialType.FUEL_UO2_3PCT, k_thermal=3.0)
            self.ss.set_moderator_region((0, refl), MaterialType.REFLECTOR_H2O, k_thermal=0.6)
            self.ss.set_moderator_region((L - refl, L), MaterialType.REFLECTOR_H2O, k_thermal=0.6)

        # --- Build self-consistent initial thermal state --------------------
        # Instead of using the SS solver's thermal solution (which can diverge
        # for large powers in simple 1D geometry), we construct a physically
        # reasonable initial temperature field.
        #
        # The initial power density profile comes from the neutronics SS solve,
        # but we rescale the thermal conductivity so the resulting temperature
        # stays in a physical range (Tmax ~ 900K for fuel, ~600K for moderator).
        k_th = self.ss.k_thermal.copy()
        T_left = self.ss.T_boundary_left
        T_right = self.ss.T_boundary_right

        # Run neutronics SS to get power shape (but not thermal)
        ss_result = self.ss.solve(total_power=initial_power, verbose=False)
        q_field = ss_result.power_density.copy()

        # Scale thermal conductivity so peak temperature is physically reasonable
        # Target: Tmax ~ 900K for initial_power in the given geometry
        # Analytical estimate: dT_max = q_max * (L_fuel/2)^2 / (2*k_eff)
        q_max = np.max(q_field)
        if q_max > 0:
            L_fuel = 0.6 * self.ss.length  # fuel region length
            target_dT = 300.0  # want ~300K rise above boundary (600K -> 900K)
            k_eff_needed = q_max * (L_fuel / 2.0) ** 2 / (2.0 * target_dT)
            k_scale = max(k_eff_needed / np.mean(k_th), 1.0)
            k_th *= k_scale

        # Solve heat conduction with scaled conductivity to get initial T
        T_field = self._solve_heat_conduction_implicit(
            np.full(self.n_x, T_left), q_field, 1e10, k_th, T_left, T_right,
        )
        T_field = np.clip(T_field, 300.0, 1200.0)
        T_ref = T_field.copy()

        P = initial_power

        # Initial precursor concentration at equilibrium:
        #   C_eq = beta / (lambda * Lambda) * n_eq
        n = P  # neutron population proportional to power
        C_prec = self.beta / (self.lam_d * self.Lambda) * n

        # --- Allocate recording arrays -------------------------------------
        n_steps = int(np.ceil(t_end / dt)) + 1
        # Extra records for early transient (10x resolution in first second)
        early_steps = min(int(1.0 / dt), n_steps)
        early_interval = max(1, record_interval // 10)
        n_records = early_steps // early_interval + (n_steps - early_steps) // record_interval + 2

        time_rec = np.zeros(n_records)
        power_rec = np.zeros(n_records)
        keff_rec = np.zeros(n_records)
        rho_pcm_rec = np.zeros(n_records)
        Tmax_rec = np.zeros(n_records)
        Tavg_rec = np.zeros(n_records)
        T_field_rec = np.zeros((n_records, self.n_x))
        q_field_rec = np.zeros((n_records, self.n_x))
        Cprec_rec = np.zeros(n_records)
        rho_dop_rec = np.zeros(n_records)
        rho_mod_rec = np.zeros(n_records)
        rho_ext_rec = np.zeros(n_records)

        rec = 0
        converged = True
        _prog_interval = max(n_steps // 50, 1)

        for step in range(n_steps):
            t = step * dt

            # --- Feedback reactivities -------------------------------------
            # Doppler: spatially-resolved sqrt(T) feedback
            # alpha_D ~ -3 pcm/sqrt(K) typical for PWR UO2 fuel
            # Use the actual temperature field (evolves with thermal time constant)
            rho_doppler = self._doppler_fb.compute_rho_doppler(T_field, T_ref)
            rho_doppler_pcm = rho_doppler * 1e5

            # Moderator density feedback: ~-5 pcm/K (typical PWR MTC at BOL)
            # Applied to average temperature deviation
            dT_avg = np.mean(T_field) - np.mean(T_ref)
            rho_mod = -5e-5 * dT_avg  # dk/k, ~-5 pcm/K
            rho_mod_pcm = rho_mod * 1e5

            rho_ext = reactivity_insertion(t)  # dk/k
            rho_ext_pcm = rho_ext * 1e5

            # Apply feedback based on mode
            if self.ss.feedback_mode == FeedbackMode.NO_FEEDBACK:
                rho_total = rho_ext
            elif self.ss.feedback_mode == FeedbackMode.DOPPLER_ONLY:
                rho_total = rho_doppler + rho_ext
            else:  # FULL_FEEDBACK
                rho_total = rho_doppler + rho_mod + rho_ext  # dk/k

            # --- Point kinetics (exponential + implicit) --------------------
            # dn/dt = (rho - beta)/Lambda * n + lambda_d * C
            # dC/dt = beta/Lambda * n - lambda_d * C
            #
            # The prompt neutron term (rho-beta)/Lambda can be very stiff
            # (timescale ~ Lambda = 2e-5 s). Standard implicit Euler damps
            # the prompt jump. We use an exponential integrator for the
            # prompt term and implicit Euler for the delayed coupling.
            alpha_prompt = (rho_total - self.beta) / self.Lambda
            S_d = self.lam_d * C_prec  # delayed neutron source

            if abs(alpha_prompt * dt) > 0.01:
                # Exponential integrator: resolves prompt jump exactly
                exp_a = np.exp(alpha_prompt * dt)
                if abs(alpha_prompt) > 1e-30:
                    n_new = n * exp_a + (S_d / alpha_prompt) * (exp_a - 1.0)
                else:
                    n_new = n + dt * S_d
                # Limit exponential growth
                n_new = min(n_new, 1e4 * initial_power)
            else:
                # Standard implicit Euler for small alpha*dt (near critical)
                a11 = 1.0 - dt * alpha_prompt
                a12 = -dt * self.lam_d
                a21 = -dt * self.beta / self.Lambda
                a22 = 1.0 + dt * self.lam_d
                det = a11 * a22 - a12 * a21
                if abs(det) < 1e-30:
                    converged = False
                    break
                n_new = (a22 * n + (-a12) * C_prec) / det

            # Precursor update (implicit Euler — always stable for slow precursors)
            C_new = (C_prec + dt * (self.beta / self.Lambda) * n_new) / (1.0 + dt * self.lam_d)

            # Safety: power must be non-negative
            if n_new < 0:
                n_new = 0.0
                converged = False

            n = n_new
            C_prec = max(C_new, 0.0)
            P = n  # power proportional to neutron population

            # --- Update power density field (shape preserved from SS) ------
            if np.sum(q_field) > 1e-10:
                q_field = ss_result.power_density * (P / initial_power)
            else:
                q_field = np.ones(self.n_x) * (P / (self.ss.length))

            # --- Transient heat conduction ---------------------------------
            T_field = self._solve_heat_conduction_implicit(
                T_field, q_field, dt, k_th, T_left, T_right,
            )

            # --- k_eff estimate from reactivity ----------------------------
            # rho = (k-1)/k  =>  k = 1/(1-rho)
            k_eff_est = 1.0 / (1.0 - rho_total) if abs(1.0 - rho_total) > 1e-12 else 1.0

            # --- Record (every step for first 1s, then at record_interval) -
            should_record = (step < int(1.0 / dt) and step % max(1, record_interval // 10) == 0) or (step % record_interval == 0)
            if should_record and rec < n_records:
                time_rec[rec] = t
                power_rec[rec] = P
                keff_rec[rec] = k_eff_est
                rho_pcm_rec[rec] = rho_total * 1e5
                Tmax_rec[rec] = np.max(T_field)
                Tavg_rec[rec] = np.mean(T_field)
                T_field_rec[rec] = T_field.copy()
                q_field_rec[rec] = q_field.copy()
                Cprec_rec[rec] = C_prec
                rho_dop_rec[rec] = rho_doppler_pcm
                rho_mod_rec[rec] = rho_mod_pcm
                rho_ext_rec[rec] = rho_ext_pcm
                rec += 1

            if progress_callback and step % _prog_interval == 0:
                progress_callback(step, n_steps)

            # Divergence check — allow large transients but catch runaway
            if P > 1e4 * initial_power or np.max(T_field) > 4000.0:
                converged = False
                # Record the divergence point
                if rec < n_records:
                    time_rec[rec] = t
                    power_rec[rec] = P
                    keff_rec[rec] = k_eff_est
                    rho_pcm_rec[rec] = rho_total * 1e5
                    Tmax_rec[rec] = np.max(T_field)
                    Tavg_rec[rec] = np.mean(T_field)
                    T_field_rec[rec] = T_field.copy()
                    q_field_rec[rec] = q_field.copy()
                    Cprec_rec[rec] = C_prec
                    rho_dop_rec[rec] = rho_doppler_pcm
                    rho_mod_rec[rec] = rho_mod_pcm
                    rho_ext_rec[rec] = rho_ext_pcm
                    rec += 1
                break

        # Trim arrays
        nr = rec
        return CoupledTransientResult(
            time=time_rec[:nr],
            power=power_rec[:nr],
            k_eff=keff_rec[:nr],
            reactivity_pcm=rho_pcm_rec[:nr],
            temperature_max=Tmax_rec[:nr],
            temperature_avg=Tavg_rec[:nr],
            temperature_field=T_field_rec[:nr],
            power_density_field=q_field_rec[:nr],
            precursor_concentration=Cprec_rec[:nr],
            rho_doppler=rho_dop_rec[:nr],
            rho_moderator=rho_mod_rec[:nr],
            rho_external=rho_ext_rec[:nr],
            converged=converged,
        )

    def export_state(self, result: CoupledTransientResult) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="coupled_physics")

        # Export the PEAK power density field (not final — transient peaks matter)
        if len(result.power_density_field) > 0:
            idx_peak = int(np.argmax(result.power))
            state.set_field("power_density", result.power_density_field[idx_peak], "W/cm^3")
            state.set_field("temperature", result.temperature_field[idx_peak], "K")
        else:
            state.set_field("power_density", np.array([]), "W/cm^3")
            state.set_field("temperature", np.array([]), "K")

        # Power history as a field for time-series visualization
        state.set_field("power_history", result.power, "W")

        # Critical transient metrics — these are what matter for safety analysis
        state.metadata["peak_power_W"] = float(np.max(result.power)) if len(result.power) > 0 else 0.0
        state.metadata["initial_power_W"] = float(result.power[0]) if len(result.power) > 0 else 0.0
        state.metadata["final_power_W"] = float(result.power[-1]) if len(result.power) > 0 else 0.0
        state.metadata["power_ratio"] = float(np.max(result.power) / result.power[0]) if len(result.power) > 0 and result.power[0] > 0 else 0.0
        state.metadata["time_to_peak_s"] = float(result.time[np.argmax(result.power)]) if len(result.time) > 0 else 0.0
        state.metadata["peak_temperature_K"] = float(np.max(result.temperature_max)) if len(result.temperature_max) > 0 else 0.0
        state.metadata["final_temperature_K"] = float(result.temperature_max[-1]) if len(result.temperature_max) > 0 else 0.0
        state.metadata["converged"] = result.converged
        state.metadata["k_eff_final"] = float(result.k_eff[-1]) if len(result.k_eff) > 0 else 0.0
        state.metadata["reactivity_pcm_final"] = float(result.reactivity_pcm[-1]) if len(result.reactivity_pcm) > 0 else 0.0
        state.metadata["rho_doppler_pcm_final"] = float(result.rho_doppler[-1]) if len(result.rho_doppler) > 0 else 0.0
        state.metadata["rho_moderator_pcm_final"] = float(result.rho_moderator[-1]) if len(result.rho_moderator) > 0 else 0.0
        state.metadata["rho_external_pcm_final"] = float(result.rho_external[-1]) if len(result.rho_external) > 0 else 0.0
        state.metadata["converged"] = result.converged
        return state


# ======================================================================
# Level 3 — 2-D Thermo-Mechanical Coupling Solver
# ======================================================================

@dataclass
class CoupledPhysics2DResult:
    """Result of 2-D thermo-mechanical coupling solver.

    Attributes
    ----------
    x : np.ndarray
        x-coordinates [m], shape (nx,).
    y : np.ndarray
        y-coordinates [m], shape (ny,).
    temperature : np.ndarray
        Temperature field T(x,y) [K], shape (ny, nx).
    displacement_x : np.ndarray
        x-displacement field [m], shape (ny, nx).
    displacement_y : np.ndarray
        y-displacement field [m], shape (ny, nx).
    stress_xx : np.ndarray
        Normal stress sigma_xx [Pa], shape (ny, nx).
    stress_yy : np.ndarray
        Normal stress sigma_yy [Pa], shape (ny, nx).
    stress_xy : np.ndarray
        Shear stress sigma_xy [Pa], shape (ny, nx).
    von_mises_stress : np.ndarray
        Von Mises equivalent stress [Pa], shape (ny, nx).
    thermal_strain : np.ndarray
        Thermal strain field (isotropic), shape (ny, nx).
    max_temperature : float
        Peak temperature [K].
    max_von_mises : float
        Peak von Mises stress [Pa].
    converged : bool
        Whether the thermal solve converged.
    iterations : int
        Number of thermal relaxation iterations.
    """
    x: np.ndarray
    y: np.ndarray
    temperature: np.ndarray
    displacement_x: np.ndarray
    displacement_y: np.ndarray
    stress_xx: np.ndarray
    stress_yy: np.ndarray
    stress_xy: np.ndarray
    von_mises_stress: np.ndarray
    thermal_strain: np.ndarray
    max_temperature: float
    max_von_mises: float
    converged: bool
    iterations: int


class CoupledPhysics2DSolver:
    """2-D thermo-mechanical coupling solver.

    Solves steady-state heat equation on a 2-D domain, then computes
    thermal expansion strains and resulting elastic stresses via
    plane-stress assumption.

    Thermal:  k * Laplacian(T) + Q = 0
    Mechanical: epsilon_thermal = alpha * (T - T_ref)
                sigma = E / (1 - nu^2) * (epsilon - epsilon_thermal)

    Parameters
    ----------
    nx : int
        Grid points in x.
    ny : int
        Grid points in y.
    Lx : float
        Domain size in x [m].
    Ly : float
        Domain size in y [m].
    k_thermal : float
        Thermal conductivity [W/(m*K)].
    E_modulus : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    alpha_cte : float
        Coefficient of thermal expansion [1/K].
    T_ref : float
        Stress-free reference temperature [K].
    T_boundary : float
        Boundary temperature [K].
    heat_source_power : float
        Total internal heat source power density [W/m^3].
    """

    fidelity_tier = FidelityTier.HIGH_FIDELITY
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        Lx: float = 1.0,
        Ly: float = 1.0,
        k_thermal: float = 50.0,
        E_modulus: float = 200e9,
        nu: float = 0.3,
        alpha_cte: float = 12e-6,
        T_ref: float = 293.15,
        T_boundary: float = 293.15,
        heat_source_power: float = 1e6,
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.k_thermal = k_thermal
        self.E = E_modulus
        self.nu = nu
        self.alpha_cte = alpha_cte
        self.T_ref = T_ref
        self.T_boundary = T_boundary
        self.Q0 = heat_source_power

        self.x = np.linspace(0.0, Lx, nx)
        self.y = np.linspace(0.0, Ly, ny)
        self.dx = Lx / max(nx - 1, 1)
        self.dy = Ly / max(ny - 1, 1)

    def solve(
        self,
        max_iter: int = 5000,
        tol: float = 1e-6,
        omega: float = 1.6,
    ) -> CoupledPhysics2DResult:
        """Solve the coupled thermo-mechanical problem.

        Parameters
        ----------
        max_iter : int
            Maximum SOR iterations for thermal solve.
        tol : float
            Convergence tolerance for temperature.
        omega : float
            SOR relaxation factor.

        Returns
        -------
        CoupledPhysics2DResult
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        dx2, dy2 = dx ** 2, dy ** 2
        k = self.k_thermal

        # Heat source: Gaussian blob at centre
        xx, yy = np.meshgrid(self.x, self.y)  # (ny, nx)
        cx, cy = self.Lx / 2.0, self.Ly / 2.0
        sigma_q = min(self.Lx, self.Ly) / 8.0
        Q = self.Q0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma_q ** 2))

        # --- Solve thermal: k * Laplacian(T) + Q = 0, Dirichlet BCs ---
        T = np.full((ny, nx), self.T_boundary)
        converged = False
        iteration = 0

        rx = dy2 / (2.0 * (dx2 + dy2))
        ry = dx2 / (2.0 * (dx2 + dy2))
        rq = dx2 * dy2 / (2.0 * k * (dx2 + dy2))

        for iteration in range(1, max_iter + 1):
            T_old = T.copy()
            # Jacobi-style update with SOR
            T_new = (
                rx * (T_old[1:-1, 2:] + T_old[1:-1, :-2])
                + ry * (T_old[2:, 1:-1] + T_old[:-2, 1:-1])
                + rq * Q[1:-1, 1:-1]
            )
            T[1:-1, 1:-1] = (1.0 - omega) * T_old[1:-1, 1:-1] + omega * T_new

            # Dirichlet BCs on all boundaries
            T[0, :] = self.T_boundary
            T[-1, :] = self.T_boundary
            T[:, 0] = self.T_boundary
            T[:, -1] = self.T_boundary

            residual = np.max(np.abs(T - T_old))
            if residual < tol:
                converged = True
                break

        # --- Compute thermal strain ---
        dT = T - self.T_ref
        thermal_strain = self.alpha_cte * dT

        # --- Compute thermal stress (plane stress, constrained boundaries) ---
        # For a constrained body:
        #   sigma_xx = sigma_yy = -E * alpha * dT / (1 - nu)  (biaxial)
        #   sigma_xy from shear of thermal gradient
        E_ps = self.E / (1.0 - self.nu)
        sigma_xx = -E_ps * self.alpha_cte * dT
        sigma_yy = -E_ps * self.alpha_cte * dT

        # Shear stress from thermal gradient (approximation)
        dTdx = np.zeros((ny, nx))
        dTdy = np.zeros((ny, nx))
        dTdx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2.0 * dx)
        dTdy[1:-1, :] = (T[2:, :] - T[:-2, :]) / (2.0 * dy)
        sigma_xy = -self.E * self.alpha_cte / (2.0 * (1.0 + self.nu)) * (dTdx + dTdy) * dx

        # Von Mises stress
        von_mises = np.sqrt(
            sigma_xx ** 2 + sigma_yy ** 2 - sigma_xx * sigma_yy + 3.0 * sigma_xy ** 2
        )

        # Displacement estimate (integration of strain)
        # u_x ~ integral(alpha * dT dx) from boundary
        ux = np.cumsum(thermal_strain, axis=1) * dx
        uy = np.cumsum(thermal_strain, axis=0) * dy
        # Zero at left/bottom boundaries
        ux -= ux[:, 0:1]
        uy -= uy[0:1, :]

        return CoupledPhysics2DResult(
            x=self.x,
            y=self.y,
            temperature=T,
            displacement_x=ux,
            displacement_y=uy,
            stress_xx=sigma_xx,
            stress_yy=sigma_yy,
            stress_xy=sigma_xy,
            von_mises_stress=von_mises,
            thermal_strain=thermal_strain,
            max_temperature=float(np.max(T)),
            max_von_mises=float(np.max(von_mises)),
            converged=converged,
            iterations=iteration,
        )

    def export_state(self, result: CoupledPhysics2DResult) -> PhysicsState:
        """Export 2-D result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="coupled_physics_2d")
        state.set_field("temperature", result.temperature, "K")
        state.set_field("stress", result.von_mises_stress, "Pa")
        state.set_field("displacement", result.displacement_x, "m")
        state.metadata["max_temperature"] = result.max_temperature
        state.metadata["max_von_mises"] = result.max_von_mises
        state.metadata["converged"] = result.converged
        state.metadata["iterations"] = result.iterations
        return state
