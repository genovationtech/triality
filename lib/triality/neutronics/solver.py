"""
Coupled Neutronics Solver -- Diffusion + Point Kinetics + Precursors

Integrates the multi-group diffusion solver, point kinetics engine,
and precursor tracking into a unified solver for reactor physics
analysis: steady-state eigenvalue problems and time-dependent transients.

Physics:
    Steady-state: Two-group diffusion eigenvalue problem
        -div(D_g grad(phi_g)) + Sigma_r,g phi_g = (1/k) chi_g * sum(nu*Sigma_f * phi)
        Solved via power iteration with GMRES inner solves.

    Transient: Point kinetics with six-group delayed neutrons
        dP/dt = [(rho - beta)/Lambda] * P + sum(lambda_i * C_i)
        dC_i/dt = (beta_i/Lambda) * P - lambda_i * C_i

    The diffusion solution provides the initial power shape; the point
    kinetics engine then tracks the time-dependent power response to
    reactivity insertions.

Dependencies:
    numpy, scipy.sparse
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

from triality.core.units import NuclearSIAdapter, UnitMetadata
from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from triality.neutronics.diffusion_solver import (
    MultiGroupDiffusion1D,
    NeutronicsResult,
    MaterialType,
    CrossSectionSet,
)
from triality.neutronics.precursors import (
    PrecursorField,
    DelayedNeutronGroup,
)
from triality.neutronics.point_kinetics import (
    PointKineticsState,
    PointKineticsEngine,
    SpatialKineticsAdapter,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NeutronicsSolverResult:
    """Results from the coupled neutronics solver.

    Attributes
    ----------
    k_eff : float
        Effective multiplication factor from diffusion eigenvalue.
    converged : bool
        Whether the eigenvalue problem converged.
    diffusion_result : NeutronicsResult
        Full diffusion solver output (flux, power density, etc.).
    time : np.ndarray
        Time array for transient [s] (empty if steady-state only).
    power : np.ndarray
        Power history [W] (transient).
    reactivity : np.ndarray
        Reactivity history [dk/k] (transient).
    period : np.ndarray
        Reactor period history [s] (transient).
    precursor_source : np.ndarray
        Delayed neutron source history (transient).
    peak_power : float
        Peak power during transient [W].
    is_prompt_critical : bool
        Whether prompt criticality was reached during transient.
    power_shape : np.ndarray
        Normalised axial power shape from diffusion.
    peaking_factor : float
        Axial power peaking factor.
    states : List[PointKineticsState]
        Snapshots of the point kinetics state (transient).
    """
    k_eff: float
    converged: bool
    diffusion_result: Optional[NeutronicsResult]
    time: np.ndarray
    power: np.ndarray
    reactivity: np.ndarray
    period: np.ndarray
    precursor_source: np.ndarray
    peak_power: float
    is_prompt_critical: bool
    power_shape: np.ndarray
    peaking_factor: float
    states: List[PointKineticsState] = field(default_factory=list)
    units: UnitMetadata = field(default_factory=UnitMetadata)

    def __post_init__(self):
        """Declare unit metadata for all fields."""
        self.units.declare("time", "s", "Time")
        self.units.declare("power", "W", "Reactor power")
        self.units.declare("peak_power", "W", "Peak reactor power")
        self.units.declare("reactivity", "1", "Reactivity dk/k")
        self.units.declare("period", "s", "Reactor period")
        self.units.declare("power_shape", "1", "Normalized axial power shape")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class NeutronicsSolver:
    """Coupled neutronics solver: diffusion eigenvalue + point kinetics transient.

    Parameters
    ----------
    core_length : float
        Reactor core length [cm].
    n_spatial : int
        Number of spatial mesh points.
    fuel_material : MaterialType
        Fuel material type.
    reflector_material : MaterialType
        Reflector material type.
    reflector_thickness : float
        Reflector thickness on each side [cm].
    fuel_type : str
        Fuel isotope for delayed neutron data ('U235', 'Pu239').
    generation_time : float
        Prompt neutron generation time [s].
    initial_power : float
        Initial reactor power [W].

    Example
    -------
    >>> solver = NeutronicsSolver(
    ...     core_length=200.0, n_spatial=100,
    ...     fuel_material=MaterialType.FUEL_UO2_3PCT,
    ...     initial_power=3000e6,
    ... )
    >>> result = solver.solve()
    >>> print(f"k_eff = {result.k_eff:.5f}")
    >>> # Run a transient with 100 pcm step insertion
    >>> result_t = solver.solve_transient(
    ...     t_final=5.0, dt=0.01,
    ...     reactivity_func=lambda t: 100e-5 if t > 0.5 else 0.0
    ... )
    >>> print(f"Peak power = {result_t.peak_power:.4e} W")
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M3_COUPLED

    def __init__(
        self,
        core_length: float = 200.0,
        n_spatial: int = 100,
        fuel_material: MaterialType = MaterialType.FUEL_UO2_3PCT,
        reflector_material: MaterialType = MaterialType.REFLECTOR_H2O,
        reflector_thickness: float = 30.0,
        fuel_type: str = 'U235',
        generation_time: float = 2e-5,
        initial_power: float = 3000e6,
    ):
        self.core_length = core_length
        self.n_spatial = n_spatial
        self.fuel_material = fuel_material
        self.reflector_material = reflector_material
        self.reflector_thickness = reflector_thickness
        self.fuel_type = fuel_type
        self.generation_time = generation_time
        self.initial_power = initial_power

        # Total domain length
        self.total_length = core_length + 2 * reflector_thickness

        # Build the diffusion solver
        self._diffusion = MultiGroupDiffusion1D(
            length=self.total_length,
            n_points=n_spatial,
        )
        # Set materials
        self._diffusion.set_material(
            region=(0, reflector_thickness),
            material=reflector_material,
        )
        self._diffusion.set_material(
            region=(reflector_thickness, reflector_thickness + core_length),
            material=fuel_material,
        )
        self._diffusion.set_material(
            region=(reflector_thickness + core_length, self.total_length),
            material=reflector_material,
        )

        # Precursor and kinetics objects (initialised on first solve)
        self._precursors: Optional[PrecursorField] = None
        self._engine: Optional[PointKineticsEngine] = None
        self._adapter: Optional[SpatialKineticsAdapter] = None

        # Coupling state
        self._coupled_state = None
        self._time = 0.0

    def solve(self, max_iterations: int = 100,
              tolerance: float = 1e-5) -> NeutronicsSolverResult:
        """Solve the steady-state diffusion eigenvalue problem.

        Returns
        -------
        NeutronicsSolverResult
            k-eff, flux distributions, and power shape.
        """
        diff_result = self._diffusion.solve(
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

        # Extract power shape from thermal flux in fuel region
        power_shape = diff_result.power_density.copy()
        power_shape = np.maximum(power_shape, 0.0)
        mean_p = np.mean(power_shape)
        if mean_p > 0:
            power_shape_norm = power_shape / mean_p
        else:
            power_shape_norm = np.ones_like(power_shape)

        peaking = float(np.max(power_shape_norm))

        return NeutronicsSolverResult(
            k_eff=diff_result.k_eff,
            converged=diff_result.converged,
            diffusion_result=diff_result,
            time=np.array([]),
            power=np.array([self.initial_power]),
            reactivity=np.array([0.0]),
            period=np.array([np.inf]),
            precursor_source=np.array([0.0]),
            peak_power=self.initial_power,
            is_prompt_critical=False,
            power_shape=power_shape_norm,
            peaking_factor=peaking,
        )

    def export_state(self, result: NeutronicsSolverResult) -> PhysicsState:
        """Export neutronics result as SI PhysicsState for coupling."""
        adapter = NuclearSIAdapter()
        state = PhysicsState(solver_name="neutronics", time=0.0)

        if result.diffusion_result is not None:
            z_m = adapter.length_to_si(result.diffusion_result.x)
            total_flux_si = adapter.flux_to_si(
                result.diffusion_result.phi_fast + result.diffusion_result.phi_thermal
            )
            power_density_si = adapter.power_density_to_si(result.diffusion_result.power_density)
            state.set_field("neutron_flux", total_flux_si, "n/(m^2*s)", grid=z_m)
            state.set_field("power_density", power_density_si, "W/m^3", grid=z_m)
        else:
            n = self.n_spatial
            z_cm = np.linspace(0, self.total_length, n)
            z_m = adapter.length_to_si(z_cm)
            state.set_field("neutron_flux", np.zeros(n), "n/(m^2*s)", grid=z_m)
            state.set_field("power_density", np.zeros(n), "W/m^3", grid=z_m)

        state.metadata["k_eff"] = result.k_eff
        state.metadata["converged"] = result.converged
        state.metadata["peaking_factor"] = result.peaking_factor
        state.metadata["peak_power_W"] = result.peak_power
        state.metadata["core_length_m"] = float(adapter.length_to_si(self.core_length))
        state.metadata["power_shape"] = result.power_shape

        if len(result.time) > 0:
            state.time = float(result.time[-1])
            state.metadata["power_history_W"] = result.power
            state.metadata["reactivity_history"] = result.reactivity
            state.metadata["period_history_s"] = result.period

        return state

    def import_state(self, state: PhysicsState) -> None:
        """Import coupled fields from partner solvers (e.g. thermal-hydraulics)."""
        self._coupled_state = state

    def advance(self, dt: float) -> PhysicsState:
        """Advance transient neutronics by dt with coupled temperature feedback."""
        reactivity_func = lambda t: 0.0
        if self._coupled_state is not None:
            if self._coupled_state.has("temperature"):
                T = self._coupled_state.get_field("temperature").data
                T_ref = 565.0  # reference coolant temperature [K]
                alpha_T = -3.0e-5  # temperature reactivity coefficient [dk/k/K]
                dT_avg = float(np.mean(T)) - T_ref
                rho_feedback = alpha_T * dT_avg
                reactivity_func = lambda t, rho=rho_feedback: rho
        result = self.solve_transient(
            t_final=self._time + dt, dt=min(dt / 10.0, 0.01),
            reactivity_func=reactivity_func,
        )
        self._time += dt
        return self.export_state(result)

    def solve_transient(
        self,
        t_final: float = 10.0,
        dt: float = 0.01,
        reactivity_func=None,
    ) -> NeutronicsSolverResult:
        """Solve a reactor transient using point kinetics.

        First runs the diffusion eigenvalue to get the initial power shape,
        then marches the point kinetics equations forward in time.

        Parameters
        ----------
        t_final : float
            Final simulation time [s].
        dt : float
            Time step [s].
        reactivity_func : callable, optional
            Function rho(t) returning external reactivity [dk/k].
            Default: zero reactivity (steady state).

        Returns
        -------
        NeutronicsSolverResult
        """
        if reactivity_func is None:
            reactivity_func = lambda t: 0.0

        # First get steady-state solution
        ss = self.solve()

        # Initialise precursors and engine
        self._precursors = PrecursorField(n_groups=6, fuel_type=self.fuel_type)
        self._engine = PointKineticsEngine(
            precursors=self._precursors,
            generation_time=self.generation_time,
            initial_power=self.initial_power,
        )

        n_steps = int(np.ceil(t_final / dt))
        times = np.zeros(n_steps + 1)
        powers = np.zeros(n_steps + 1)
        reactivities = np.zeros(n_steps + 1)
        periods = np.zeros(n_steps + 1)
        delayed_sources = np.zeros(n_steps + 1)
        states: List[PointKineticsState] = []

        # Initial conditions
        times[0] = 0.0
        powers[0] = self.initial_power
        reactivities[0] = reactivity_func(0.0)
        periods[0] = np.inf
        delayed_sources[0] = self._engine.delayed_neutron_source()
        states.append(self._engine.get_state())

        is_prompt_critical = False

        for step in range(n_steps):
            t_new = (step + 1) * dt
            rho = reactivity_func(t_new)

            self._engine.step(dt=dt, rho_total=rho)

            times[step + 1] = t_new
            powers[step + 1] = self._engine.power
            reactivities[step + 1] = rho
            periods[step + 1] = self._engine.period()
            delayed_sources[step + 1] = self._engine.delayed_neutron_source()
            states.append(self._engine.get_state())

            if self._engine.is_prompt_critical():
                is_prompt_critical = True

        peak_power = float(np.max(powers))

        return NeutronicsSolverResult(
            k_eff=ss.k_eff,
            converged=ss.converged,
            diffusion_result=ss.diffusion_result,
            time=times,
            power=powers,
            reactivity=reactivities,
            period=periods,
            precursor_source=delayed_sources,
            peak_power=peak_power,
            is_prompt_critical=is_prompt_critical,
            power_shape=ss.power_shape,
            peaking_factor=ss.peaking_factor,
            states=states,
        )


# ---------------------------------------------------------------------------
# Level 3 -- 2D Multigroup Neutron Diffusion Solver
# ---------------------------------------------------------------------------

@dataclass
class Neutronics2DResult:
    """Result container for the 2D multigroup neutron diffusion solver.

    Attributes
    ----------
    k_eff : float
        Effective multiplication factor.
    converged : bool
        Whether the power iteration converged.
    phi_fast : np.ndarray
        Fast-group flux, shape (ny, nx).
    phi_thermal : np.ndarray
        Thermal-group flux, shape (ny, nx).
    power_density : np.ndarray
        Fission power density (normalised), shape (ny, nx).
    x : np.ndarray
        x-grid [cm], shape (nx,).
    y : np.ndarray
        y-grid [cm], shape (ny,).
    peaking_factor : float
        Maximum / mean power density.
    iteration_count : int
        Number of outer power iterations.
    k_history : np.ndarray
        k-effective convergence history.
    """
    k_eff: float = 1.0
    converged: bool = False
    phi_fast: np.ndarray = field(default_factory=lambda: np.array([]))
    phi_thermal: np.ndarray = field(default_factory=lambda: np.array([]))
    power_density: np.ndarray = field(default_factory=lambda: np.array([]))
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    peaking_factor: float = 1.0
    iteration_count: int = 0
    k_history: np.ndarray = field(default_factory=lambda: np.array([]))


class Neutronics2DSolver:
    """2D two-group neutron diffusion eigenvalue solver (Level 3).

    Solves the two-group steady-state neutron diffusion equations on a
    rectangular domain using power iteration with inner Jacobi solves:

        Group 1 (fast):
            -D1 * laplacian(phi1) + Sigma_r1 * phi1 = (1/k) * chi1 * (nu*Sigma_f1*phi1 + nu*Sigma_f2*phi2)

        Group 2 (thermal):
            -D2 * laplacian(phi2) + Sigma_a2 * phi2 = Sigma_s12 * phi1

    Boundary conditions: zero flux (Dirichlet) on all sides.

    Parameters
    ----------
    nx, ny : int
        Number of grid cells in x and y.
    Lx, Ly : float
        Domain size [cm].
    D1, D2 : float
        Diffusion coefficients for fast and thermal groups [cm].
    Sigma_a1, Sigma_a2 : float
        Absorption cross sections [1/cm].
    Sigma_s12 : float
        Scattering cross section from group 1 to group 2 [1/cm].
    nu_Sigma_f1, nu_Sigma_f2 : float
        Nu * fission cross sections [1/cm].
    chi1 : float
        Fission spectrum fraction born in fast group.
    """

    fidelity_tier = FidelityTier.HIGH_FIDELITY
    coupling_maturity = CouplingMaturity.M3_COUPLED

    def __init__(
        self,
        nx: int = 60,
        ny: int = 60,
        Lx: float = 200.0,
        Ly: float = 200.0,
        D1: float = 1.5,
        D2: float = 0.4,
        Sigma_a1: float = 0.01,
        Sigma_a2: float = 0.08,
        Sigma_s12: float = 0.02,
        nu_Sigma_f1: float = 0.005,
        nu_Sigma_f2: float = 0.1,
        chi1: float = 1.0,
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0.0, Lx, nx)
        self.y = np.linspace(0.0, Ly, ny)

        self.D1 = D1
        self.D2 = D2
        self.Sigma_a1 = Sigma_a1
        self.Sigma_a2 = Sigma_a2
        self.Sigma_r1 = Sigma_a1 + Sigma_s12  # removal from group 1
        self.Sigma_s12 = Sigma_s12
        self.nu_Sigma_f1 = nu_Sigma_f1
        self.nu_Sigma_f2 = nu_Sigma_f2
        self.chi1 = chi1

        # Coupling state
        self._coupled_state = None
        self._time = 0.0

    def _jacobi_diffusion_solve(
        self, phi: np.ndarray, D: float, Sigma: float,
        source: np.ndarray, n_iter: int = 50
    ) -> np.ndarray:
        """Jacobi iterative solve for:  -D * laplacian(phi) + Sigma * phi = source
        with zero-flux BCs."""
        dx2 = self.dx**2
        dy2 = self.dy**2
        coeff = 2.0 * D * (1.0 / dx2 + 1.0 / dy2) + Sigma

        for _ in range(n_iter):
            phi_new = np.zeros_like(phi)
            phi_new[1:-1, 1:-1] = (
                D * (phi[1:-1, 2:] + phi[1:-1, :-2]) / dx2
                + D * (phi[2:, 1:-1] + phi[:-2, 1:-1]) / dy2
                + source[1:-1, 1:-1]
            ) / coeff
            # Zero-flux BCs (Dirichlet phi=0 at boundary)
            phi = phi_new

        return phi

    def solve(
        self,
        max_outer: int = 500,
        tol: float = 1e-6,
        inner_iters: int = 50,
    ) -> Neutronics2DResult:
        """Solve the 2D two-group diffusion eigenvalue problem via power iteration.

        Parameters
        ----------
        max_outer : int
            Maximum power iterations.
        tol : float
            Convergence tolerance on k-effective.
        inner_iters : int
            Jacobi iterations per inner solve.

        Returns
        -------
        Neutronics2DResult
        """
        nx, ny = self.nx, self.ny

        # Initial guess: flat flux
        phi1 = np.ones((ny, nx))
        phi2 = np.ones((ny, nx))
        k = 1.0
        k_history = []
        converged = False

        for outer in range(max_outer):
            # Fission source
            fission_source = self.nu_Sigma_f1 * phi1 + self.nu_Sigma_f2 * phi2

            # Solve group 1: -D1*lap(phi1) + Sigma_r1*phi1 = (chi1/k)*fission_source
            source1 = (self.chi1 / k) * fission_source
            phi1_new = self._jacobi_diffusion_solve(
                phi1, self.D1, self.Sigma_r1, source1, inner_iters
            )

            # Solve group 2: -D2*lap(phi2) + Sigma_a2*phi2 = Sigma_s12*phi1_new
            source2 = self.Sigma_s12 * phi1_new
            phi2_new = self._jacobi_diffusion_solve(
                phi2, self.D2, self.Sigma_a2, source2, inner_iters
            )

            # Update k-effective
            fission_source_new = (
                self.nu_Sigma_f1 * phi1_new + self.nu_Sigma_f2 * phi2_new
            )
            f_old = np.sum(fission_source)
            f_new = np.sum(fission_source_new)
            k_new = k * f_new / max(f_old, 1e-30)

            # Normalise fluxes
            norm = np.max(phi1_new) + 1e-30
            phi1 = phi1_new / norm
            phi2 = phi2_new / norm

            k_history.append(k_new)

            if outer > 2 and abs(k_new - k) < tol:
                converged = True
                k = k_new
                break

            k = k_new

        # Power density proportional to fission source
        power = self.nu_Sigma_f1 * phi1 + self.nu_Sigma_f2 * phi2
        power = np.maximum(power, 0.0)
        mean_p = np.mean(power)
        if mean_p > 0:
            power_norm = power / mean_p
        else:
            power_norm = np.ones_like(power)

        peaking = float(np.max(power_norm))

        return Neutronics2DResult(
            k_eff=k,
            converged=converged,
            phi_fast=phi1,
            phi_thermal=phi2,
            power_density=power_norm,
            x=self.x.copy(),
            y=self.y.copy(),
            peaking_factor=peaking,
            iteration_count=outer + 1,
            k_history=np.array(k_history),
        )

    def export_state(self, result: Neutronics2DResult) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="neutronics_2d")
        state.set_field("neutron_flux_fast", result.phi_fast, "1/cm^2/s")
        state.set_field("neutron_flux_thermal", result.phi_thermal, "1/cm^2/s")
        state.set_field("power_density", result.power_density, "W/cm^3")
        state.metadata["k_eff"] = result.k_eff
        state.metadata["converged"] = result.converged
        state.metadata["peaking_factor"] = result.peaking_factor
        state.metadata["iterations"] = result.iteration_count
        return state

    def import_state(self, state: PhysicsState) -> None:
        """Import coupled fields from partner solvers."""
        self._coupled_state = state

    def advance(self, dt: float) -> PhysicsState:
        """Re-solve eigenvalue problem with temperature-updated cross-sections."""
        if self._coupled_state is not None:
            if self._coupled_state.has("temperature"):
                T = self._coupled_state.get_field("temperature").data
                # Doppler broadening: absorption XS increases with sqrt(T)
                T_ref = 600.0  # reference temperature [K]
                T_avg = float(np.mean(T))
                doppler_factor = np.sqrt(T_avg / T_ref)
                self.Sigma_a2 = 0.08 * doppler_factor
                self.Sigma_r1 = self.Sigma_a1 + self.Sigma_s12
        result = self.solve()
        self._time += dt
        return self.export_state(result)
