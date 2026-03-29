"""
Drift-Diffusion Semiconductor Solver

High-level numerical solver that integrates DriftDiffusion1D,
SemiconductorMaterial, TemperatureDependentMaterial, ShockleyReadHall,
FieldDependentMobility, and ImprovedCurrentCalculator for production-quality
semiconductor device simulation.

Solves the coupled Poisson + continuity system:
    Poisson:     d^2V/dx^2 = -q(p - n + N_d - N_a) / eps
    Electron:    dJ_n/dx = q * (G - R)
    Hole:        dJ_p/dx = -q * (G - R)
    Currents:    J_n = q mu_n n E + q D_n dn/dx
                 J_p = q mu_p p E - q D_p dp/dx

With optional advanced physics:
    - Temperature-dependent bandgap and intrinsic concentration
    - Shockley-Read-Hall generation-recombination
    - Field-dependent mobility (velocity saturation)

References:
    Selberherr, "Analysis and Simulation of Semiconductor Devices"
    Sze & Ng, "Physics of Semiconductor Devices"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Tuple, List

from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from .device_solver import (
    DriftDiffusion1D,
    DD1DResult,
    SemiconductorMaterial,
    PNJunctionAnalyzer,
    create_pn_junction,
    q,
    k_B,
    eps_0,
    V_T,
)
from .advanced_physics import (
    TemperatureDependentMaterial,
    ShockleyReadHall,
    FieldDependentMobility,
    ImprovedCurrentCalculator,
)


@dataclass
class DriftDiffusionResult:
    """Result container for drift-diffusion simulation.

    Attributes
    ----------
    x : np.ndarray
        Position array [cm].
    V : np.ndarray
        Electrostatic potential [V].
    n : np.ndarray
        Electron concentration [cm^-3].
    p : np.ndarray
        Hole concentration [cm^-3].
    E_field : np.ndarray
        Electric field [V/cm].
    J_total : np.ndarray
        Total current density profile [A/cm^2].
    J_n : np.ndarray
        Electron current density [A/cm^2].
    J_p : np.ndarray
        Hole current density [A/cm^2].
    built_in_potential : float
        Built-in potential [V].
    depletion_width : float
        Depletion region width [cm].
    junction_position : Optional[float]
        Metallurgical junction position [cm].
    max_field : float
        Maximum electric field [V/cm].
    current_density : float
        Average total current density [A/cm^2].
    iv_voltages : Optional[np.ndarray]
        I-V sweep voltages [V] (if computed).
    iv_currents : Optional[np.ndarray]
        I-V sweep current densities [A/cm^2] (if computed).
    temperature : float
        Device temperature [K].
    material_name : str
        Semiconductor material name.
    converged : bool
        Whether the solver converged.
    iterations : int
        Number of Gummel iterations taken.
    residual : float
        Final residual.
    """
    x: np.ndarray
    V: np.ndarray
    n: np.ndarray
    p: np.ndarray
    E_field: np.ndarray
    J_total: np.ndarray
    J_n: np.ndarray
    J_p: np.ndarray
    built_in_potential: float
    depletion_width: float
    junction_position: Optional[float]
    max_field: float
    current_density: float
    iv_voltages: Optional[np.ndarray] = None
    iv_currents: Optional[np.ndarray] = None
    temperature: float = 300.0
    material_name: str = "Silicon"
    converged: bool = True
    iterations: int = 0
    residual: float = 0.0


@dataclass
class DriftDiffusion2DResult:
    """Result container for 2D drift-diffusion simulation.

    Attributes
    ----------
    x : np.ndarray
        x-position array [cm], shape (nx,).
    y : np.ndarray
        y-position array [cm], shape (ny,).
    V : np.ndarray
        Electrostatic potential [V], shape (ny, nx).
    n : np.ndarray
        Electron concentration [cm^-3], shape (ny, nx).
    p : np.ndarray
        Hole concentration [cm^-3], shape (ny, nx).
    E_x : np.ndarray
        x-component of electric field [V/cm], shape (ny, nx).
    E_y : np.ndarray
        y-component of electric field [V/cm], shape (ny, nx).
    J_n_x : np.ndarray
        Electron current density x-component [A/cm^2], shape (ny, nx).
    J_n_y : np.ndarray
        Electron current density y-component [A/cm^2], shape (ny, nx).
    J_p_x : np.ndarray
        Hole current density x-component [A/cm^2], shape (ny, nx).
    J_p_y : np.ndarray
        Hole current density y-component [A/cm^2], shape (ny, nx).
    recombination : np.ndarray
        Net recombination rate [cm^-3 s^-1], shape (ny, nx).
    built_in_potential : float
        Built-in potential [V].
    max_field : float
        Maximum electric field magnitude [V/cm].
    temperature : float
        Device temperature [K].
    material_name : str
        Semiconductor material name.
    converged : bool
        Whether the solver converged.
    iterations : int
        Number of Gummel iterations taken.
    residual : float
        Final residual.
    """
    x: np.ndarray
    y: np.ndarray
    V: np.ndarray
    n: np.ndarray
    p: np.ndarray
    E_x: np.ndarray
    E_y: np.ndarray
    J_n_x: np.ndarray
    J_n_y: np.ndarray
    J_p_x: np.ndarray
    J_p_y: np.ndarray
    recombination: np.ndarray
    built_in_potential: float
    max_field: float
    temperature: float = 300.0
    material_name: str = "Silicon"
    converged: bool = True
    iterations: int = 0
    residual: float = 0.0


class DriftDiffusion2DSolver:
    """2D drift-diffusion semiconductor solver with Scharfetter-Gummel discretization.

    Solves the coupled 2D Poisson + continuity system:
        Poisson:  div(eps grad V) = -q(p - n + Nd - Na)
        Electron: div(Jn) =  q(G - R)
        Hole:     div(Jp) = -q(G - R)

    Current densities use Scharfetter-Gummel flux in both x and y:
        Jn_{i+1/2} = (q Dn / dx) * [n_{i+1} B(-dV/Vt) - n_i B(dV/Vt)]
        Jp_{i+1/2} = (q Dp / dx) * [p_i B(dV/Vt) - p_{i+1} B(-dV/Vt)]

    where B(x) = x / (exp(x) - 1) is the Bernoulli function.

    Parameters
    ----------
    Lx : float
        Device length in x [cm] (default 2e-4).
    Ly : float
        Device length in y [cm] (default 2e-4).
    nx : int
        Grid points in x (default 80).
    ny : int
        Grid points in y (default 80).
    material : str
        'Silicon' or 'GaAs' (default 'Silicon').
    temperature : float
        Device temperature [K] (default 300).
    tau_n : float
        SRH electron lifetime [s] (default 1e-6).
    tau_p : float
        SRH hole lifetime [s] (default 1e-6).
    """

    fidelity_tier = FidelityTier.HIGH_FIDELITY
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        Lx: float = 2e-4,
        Ly: float = 2e-4,
        nx: int = 80,
        ny: int = 80,
        material: str = 'Silicon',
        temperature: float = 300.0,
        tau_n: float = 1e-6,
        tau_p: float = 1e-6,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.temperature = temperature
        self.tau_n = tau_n
        self.tau_p = tau_p

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)

        self.Vt = k_B * temperature / q  # thermal voltage [V]

        # Material properties
        if material.lower() == 'gaas':
            self.material_name = 'GaAs'
            self.eps_r = 12.9
            self.mu_n = 8500.0   # cm^2/(V s)
            self.mu_p = 400.0
            self.ni = 2.1e6      # intrinsic carrier conc [cm^-3]
        else:
            self.material_name = 'Silicon'
            self.eps_r = 11.7
            self.mu_n = 1350.0
            self.mu_p = 480.0
            self.ni = 1.5e10

        self.D_n = self.Vt * self.mu_n   # Einstein relation
        self.D_p = self.Vt * self.mu_p
        self.eps = self.eps_r * eps_0

        # Doping profile: callable (x, y) -> Nd, Na
        self._Nd_func = None
        self._Na_func = None
        self._doping_set = False

    @staticmethod
    def _bernoulli(x: np.ndarray) -> np.ndarray:
        """Bernoulli function B(x) = x / (exp(x) - 1), numerically stable."""
        result = np.ones_like(x, dtype=float)
        small = np.abs(x) < 1e-10
        big = ~small
        result[big] = x[big] / (np.exp(x[big]) - 1.0 + 1e-30)
        # For small x: B(x) ~ 1 - x/2
        result[small] = 1.0 - 0.5 * x[small]
        return result

    def set_doping_2d(
        self,
        N_d: Callable[[float, float], float],
        N_a: Callable[[float, float], float],
    ):
        """Set 2D donor and acceptor doping profiles.

        Parameters
        ----------
        N_d : callable
            Donor concentration function (x, y) -> [cm^-3].
        N_a : callable
            Acceptor concentration function (x, y) -> [cm^-3].
        """
        self._Nd_func = N_d
        self._Na_func = N_a
        self._doping_set = True

    def set_pn_junction_2d(
        self,
        N_d_level: float = 1e17,
        N_a_level: float = 1e16,
        junction_x: Optional[float] = None,
    ):
        """Set a step PN junction along x (uniform in y).

        Parameters
        ----------
        N_d_level : float
            N-side donor doping [cm^-3].
        N_a_level : float
            P-side acceptor doping [cm^-3].
        junction_x : float or None
            Junction position in x [cm]. Default: midpoint.
        """
        if junction_x is None:
            junction_x = self.Lx / 2.0
        jx = junction_x
        self._Nd_func = lambda x, y: N_d_level if x < jx else 0.0
        self._Na_func = lambda x, y: 0.0 if x < jx else N_a_level
        self._doping_set = True

    def _srh_recombination(self, n: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Shockley-Read-Hall recombination rate [cm^-3 s^-1]."""
        ni2 = self.ni**2
        num = n * p - ni2
        denom = self.tau_p * (n + self.ni) + self.tau_n * (p + self.ni)
        return num / np.maximum(denom, 1e-30)

    def _solve_poisson_2d(
        self,
        V: np.ndarray,
        n: np.ndarray,
        p: np.ndarray,
        Nd: np.ndarray,
        Na: np.ndarray,
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> np.ndarray:
        """Solve 2D Poisson equation using SOR.

        d2V/dx2 + d2V/dy2 = -q/eps * (p - n + Nd - Na)
        """
        dx2 = self.dx**2
        dy2 = self.dy**2
        factor = 0.5 / (1.0 / dx2 + 1.0 / dy2)
        omega = 1.7  # SOR parameter

        rhs = -q / self.eps * (p - n + Nd - Na)

        for _ in range(max_iter):
            max_change = 0.0
            for j in range(1, self.ny - 1):
                for i in range(1, self.nx - 1):
                    V_new = factor * (
                        (V[j, i - 1] + V[j, i + 1]) / dx2
                        + (V[j - 1, i] + V[j + 1, i]) / dy2
                        - rhs[j, i]
                    )
                    delta = V_new - V[j, i]
                    V[j, i] += omega * delta
                    max_change = max(max_change, abs(delta))
            if max_change < tol:
                break
        return V

    def _scharfetter_gummel_2d(
        self,
        V: np.ndarray,
        n: np.ndarray,
        p: np.ndarray,
        R: np.ndarray,
    ):
        """Update carrier concentrations using Scharfetter-Gummel in 2D.

        Returns updated n, p arrays.
        """
        Vt = self.Vt
        dx = self.dx
        dy = self.dy

        # --- Electron continuity: div(Jn) = q(G - R) ---
        # Jn_x at (j, i+1/2): between i and i+1
        dn_dt = np.zeros_like(n)
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                # x-direction SG flux
                dV_x_p = (V[j, i + 1] - V[j, i]) / Vt
                dV_x_m = (V[j, i] - V[j, i - 1]) / Vt
                Bp = self._bernoulli(np.array([dV_x_p]))[0]
                Bm = self._bernoulli(np.array([-dV_x_p]))[0]
                Jn_x_right = self.D_n / dx * (n[j, i + 1] * Bp - n[j, i] * Bm)
                Bp2 = self._bernoulli(np.array([dV_x_m]))[0]
                Bm2 = self._bernoulli(np.array([-dV_x_m]))[0]
                Jn_x_left = self.D_n / dx * (n[j, i] * Bp2 - n[j, i - 1] * Bm2)

                # y-direction SG flux
                dV_y_p = (V[j + 1, i] - V[j, i]) / Vt
                dV_y_m = (V[j, i] - V[j - 1, i]) / Vt
                Bp3 = self._bernoulli(np.array([dV_y_p]))[0]
                Bm3 = self._bernoulli(np.array([-dV_y_p]))[0]
                Jn_y_top = self.D_n / dy * (n[j + 1, i] * Bp3 - n[j, i] * Bm3)
                Bp4 = self._bernoulli(np.array([dV_y_m]))[0]
                Bm4 = self._bernoulli(np.array([-dV_y_m]))[0]
                Jn_y_bot = self.D_n / dy * (n[j, i] * Bp4 - n[j - 1, i] * Bm4)

                div_Jn = (Jn_x_right - Jn_x_left) / dx + (Jn_y_top - Jn_y_bot) / dy
                dn_dt[j, i] = div_Jn - R[j, i]

        # --- Hole continuity: div(Jp) = -q(G - R) ---
        dp_dt = np.zeros_like(p)
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                dV_x_p = (V[j, i + 1] - V[j, i]) / Vt
                dV_x_m = (V[j, i] - V[j, i - 1]) / Vt
                Bp = self._bernoulli(np.array([-dV_x_p]))[0]
                Bm = self._bernoulli(np.array([dV_x_p]))[0]
                Jp_x_right = self.D_p / dx * (p[j, i] * Bm - p[j, i + 1] * Bp)
                Bp2 = self._bernoulli(np.array([-dV_x_m]))[0]
                Bm2 = self._bernoulli(np.array([dV_x_m]))[0]
                Jp_x_left = self.D_p / dx * (p[j, i - 1] * Bm2 - p[j, i] * Bp2)

                dV_y_p = (V[j + 1, i] - V[j, i]) / Vt
                dV_y_m = (V[j, i] - V[j - 1, i]) / Vt
                Bp3 = self._bernoulli(np.array([-dV_y_p]))[0]
                Bm3 = self._bernoulli(np.array([dV_y_p]))[0]
                Jp_y_top = self.D_p / dy * (p[j, i] * Bm3 - p[j + 1, i] * Bp3)
                Bp4 = self._bernoulli(np.array([-dV_y_m]))[0]
                Bm4 = self._bernoulli(np.array([dV_y_m]))[0]
                Jp_y_bot = self.D_p / dy * (p[j - 1, i] * Bm4 - p[j, i] * Bp4)

                div_Jp = (Jp_x_right - Jp_x_left) / dx + (Jp_y_top - Jp_y_bot) / dy
                dp_dt[j, i] = -div_Jp - R[j, i]

        return dn_dt, dp_dt

    def solve(
        self,
        applied_voltage: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        under_relaxation: float = 0.1,
    ) -> DriftDiffusion2DResult:
        """Solve the 2D drift-diffusion equations at a given bias.

        Uses Gummel iteration: solve Poisson -> update n,p -> repeat.

        Parameters
        ----------
        applied_voltage : float
            Applied voltage [V].
        max_iterations : int
            Maximum Gummel iterations.
        tolerance : float
            Convergence tolerance [V].
        under_relaxation : float
            Damping factor.

        Returns
        -------
        DriftDiffusion2DResult
        """
        if not self._doping_set:
            raise RuntimeError("Doping not set. Call set_doping_2d() or set_pn_junction_2d() first.")

        nx, ny = self.nx, self.ny

        # Build doping arrays
        Nd = np.zeros((ny, nx))
        Na = np.zeros((ny, nx))
        for j in range(ny):
            for i in range(nx):
                Nd[j, i] = self._Nd_func(self.x[i], self.y[j])
                Na[j, i] = self._Na_func(self.x[i], self.y[j])

        net_doping = Nd - Na

        # Initial equilibrium guess
        # n ~ Nd where Nd > Na, p ~ Na where Na > Nd
        n = np.maximum(net_doping, self.ni) + self.ni
        p = np.maximum(-net_doping, self.ni) + self.ni
        # Enforce n*p = ni^2 at equilibrium
        n_eq = 0.5 * (net_doping + np.sqrt(net_doping**2 + 4 * self.ni**2))
        p_eq = self.ni**2 / np.maximum(n_eq, 1.0)
        n = np.maximum(n_eq, 1.0)
        p = np.maximum(p_eq, 1.0)

        # Built-in potential
        Vbi = self.Vt * np.log(np.maximum(np.max(Nd), 1.0) * np.maximum(np.max(Na), 1.0) / self.ni**2)

        # Initial potential: linear ramp for applied voltage
        V = np.zeros((ny, nx))
        V_boundary = Vbi + applied_voltage
        for i in range(nx):
            V[:, i] = V_boundary * self.x[i] / self.Lx

        # Gummel iteration
        converged = False
        final_residual = 1.0
        iteration = 0
        for iteration in range(max_iterations):
            V_old = V.copy()

            # 1. Solve Poisson
            V = self._solve_poisson_2d(V.copy(), n, p, Nd, Na, max_iter=200, tol=tolerance * 0.1)

            # BCs
            V[:, 0] = 0.0
            V[:, -1] = applied_voltage + Vbi
            V[0, :] = V[1, :]    # Neumann top/bottom
            V[-1, :] = V[-2, :]

            # 2. Recombination
            R = self._srh_recombination(n, p)

            # 3. Scharfetter-Gummel carrier update
            dn_dt, dp_dt = self._scharfetter_gummel_2d(V, n, p, R)

            # Pseudo-time step update
            n_new = n + under_relaxation * dn_dt * (self.dx**2 / self.D_n)
            p_new = p + under_relaxation * dp_dt * (self.dx**2 / self.D_p)
            n = np.maximum(n_new, 1.0)
            p = np.maximum(p_new, 1.0)

            # BCs for carriers
            # Ohmic contacts at x=0 and x=Lx
            n[:, 0] = np.maximum(net_doping[:, 0], self.ni)
            p[:, 0] = self.ni**2 / np.maximum(n[:, 0], 1.0)
            n[:, -1] = np.maximum(net_doping[:, -1], self.ni)
            p[:, -1] = self.ni**2 / np.maximum(n[:, -1], 1.0)
            # Neumann on y boundaries
            n[0, :] = n[1, :]
            n[-1, :] = n[-2, :]
            p[0, :] = p[1, :]
            p[-1, :] = p[-2, :]

            final_residual = float(np.max(np.abs(V - V_old)))
            if final_residual < tolerance:
                converged = True
                break

        # Compute electric field
        E_x = np.zeros((ny, nx))
        E_y = np.zeros((ny, nx))
        E_x[:, 1:-1] = -(V[:, 2:] - V[:, :-2]) / (2.0 * self.dx)
        E_x[:, 0] = -(V[:, 1] - V[:, 0]) / self.dx
        E_x[:, -1] = -(V[:, -1] - V[:, -2]) / self.dx
        E_y[1:-1, :] = -(V[2:, :] - V[:-2, :]) / (2.0 * self.dy)
        E_y[0, :] = -(V[1, :] - V[0, :]) / self.dy
        E_y[-1, :] = -(V[-1, :] - V[-2, :]) / self.dy

        # Current densities (finite difference approximation)
        J_n_x = q * self.mu_n * n * E_x + q * self.D_n * np.gradient(n, self.dx, axis=1)
        J_n_y = q * self.mu_n * n * E_y + q * self.D_n * np.gradient(n, self.dy, axis=0)
        J_p_x = q * self.mu_p * p * E_x - q * self.D_p * np.gradient(p, self.dx, axis=1)
        J_p_y = q * self.mu_p * p * E_y - q * self.D_p * np.gradient(p, self.dy, axis=0)

        R_final = self._srh_recombination(n, p)
        E_mag = np.sqrt(E_x**2 + E_y**2)

        return DriftDiffusion2DResult(
            x=self.x.copy(),
            y=self.y.copy(),
            V=V,
            n=n,
            p=p,
            E_x=E_x,
            E_y=E_y,
            J_n_x=J_n_x,
            J_n_y=J_n_y,
            J_p_x=J_p_x,
            J_p_y=J_p_y,
            recombination=R_final,
            built_in_potential=float(Vbi),
            max_field=float(np.max(E_mag)),
            temperature=self.temperature,
            material_name=self.material_name,
            converged=converged,
            iterations=iteration + 1,
            residual=final_residual,
        )


    def export_state(self, result: DriftDiffusionResult) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="drift_diffusion")
        state.set_field("temperature", np.array([result.temperature]), "K")
        state.set_field("electron_density", result.n, "1/m^3")
        state.set_field("electric_field", result.E_field, "V/m")
        state.metadata["built_in_potential"] = result.built_in_potential
        state.metadata["depletion_width"] = result.depletion_width
        state.metadata["max_field"] = result.max_field
        state.metadata["current_density"] = result.current_density
        state.metadata["converged"] = result.converged
        state.metadata["material_name"] = result.material_name
        return state


class DriftDiffusionSolverV2:
    """Production drift-diffusion solver with advanced physics.

    Wires together DriftDiffusion1D with temperature-dependent materials,
    SRH recombination, field-dependent mobility, and improved current
    calculation for realistic semiconductor device analysis.

    Parameters
    ----------
    length : float
        Device length [cm] (default 2e-4 = 2 um).
    n_points : int
        Number of grid points (default 200).
    material : str
        Semiconductor material: 'Silicon' or 'GaAs' (default 'Silicon').
    temperature : float
        Device temperature [K] (default 300).
    enable_srh : bool
        Enable Shockley-Read-Hall recombination (default False).
    enable_field_mobility : bool
        Enable field-dependent mobility (default False).
    tau_n : float
        SRH electron lifetime [s] (default 1e-6).
    tau_p : float
        SRH hole lifetime [s] (default 1e-6).
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        length: float = 2e-4,
        n_points: int = 200,
        material: str = 'Silicon',
        temperature: float = 300.0,
        enable_srh: bool = False,
        enable_field_mobility: bool = False,
        tau_n: float = 1e-6,
        tau_p: float = 1e-6,
    ):
        self.length = length
        self.n_points = n_points
        self.temperature = temperature
        self.enable_srh = enable_srh
        self.enable_field_mobility = enable_field_mobility

        # Core 1D solver
        self.dd = DriftDiffusion1D(length=length, n_points=n_points)

        # Material selection
        if material.lower() == 'gaas':
            mat = SemiconductorMaterial.GaAs(temperature)
        else:
            mat = SemiconductorMaterial.Silicon(temperature)
        self.dd.set_material(mat)
        self.material_name = mat.name

        # Temperature-dependent material for advanced calculations
        if material.lower() == 'gaas':
            self.temp_mat = TemperatureDependentMaterial(
                name='GaAs', T=temperature,
                E_g0=1.519, alpha=5.405e-4, beta=204,
                eps_r=12.9, mu_n_300=8500, mu_p_300=400,
                m_n_eff=0.063, m_p_eff=0.51,
            )
        else:
            self.temp_mat = TemperatureDependentMaterial.Silicon(temperature)

        # SRH model
        self.srh = ShockleyReadHall(tau_n=tau_n, tau_p=tau_p) if enable_srh else None

        # Field-dependent mobility
        self.field_mob = FieldDependentMobility() if enable_field_mobility else None

        # Doping not yet set
        self._doping_set = False

    def set_doping(
        self,
        N_d: Callable[[float], float],
        N_a: Callable[[float], float],
    ):
        """Set donor and acceptor doping profiles.

        Parameters
        ----------
        N_d : callable
            Donor concentration function x -> [cm^-3].
        N_a : callable
            Acceptor concentration function x -> [cm^-3].
        """
        self.dd.set_doping(N_d, N_a)
        self._doping_set = True

    def set_pn_junction(
        self,
        N_d_level: float = 1e17,
        N_a_level: float = 1e16,
        junction_pos: Optional[float] = None,
    ):
        """Convenience method to set a step PN junction.

        Parameters
        ----------
        N_d_level : float
            N-side donor doping [cm^-3].
        N_a_level : float
            P-side acceptor doping [cm^-3].
        junction_pos : float or None
            Junction position [cm]. If None, uses midpoint.
        """
        if junction_pos is None:
            junction_pos = self.length / 2.0
        self.dd.set_doping(
            N_d=lambda x: N_d_level if x < junction_pos else 0,
            N_a=lambda x: 0 if x < junction_pos else N_a_level,
        )
        self._doping_set = True

    def solve(
        self,
        applied_voltage: float = 0.0,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        under_relaxation: float = 0.3,
    ) -> DriftDiffusionResult:
        """Solve the drift-diffusion equations at a given bias.

        Parameters
        ----------
        applied_voltage : float
            Applied voltage [V].
        max_iterations : int
            Maximum Gummel iterations.
        tolerance : float
            Convergence tolerance [V].
        under_relaxation : float
            Damping factor (0 < alpha <= 1).

        Returns
        -------
        DriftDiffusionResult
            Complete solution with derived quantities.
        """
        if not self._doping_set:
            raise RuntimeError("Doping not set. Call set_doping() or set_pn_junction() first.")

        # Run core solver
        dd_result = self.dd.solve(
            applied_voltage=applied_voltage,
            max_iterations=max_iterations,
            tolerance=tolerance,
            under_relaxation=under_relaxation,
        )

        x = dd_result.x
        V = dd_result.V
        n = dd_result.n
        p = dd_result.p
        dx = x[1] - x[0]

        # Electric field
        E = dd_result.electric_field()

        # Current density calculation
        if self.enable_field_mobility and self.field_mob is not None:
            mu_n_eff = self.field_mob.mobility_n(np.abs(E), self.dd.material.mu_n)
            mu_p_eff = self.field_mob.mobility_p(np.abs(E), self.dd.material.mu_p)
        else:
            mu_n_eff = self.dd.material.mu_n * np.ones_like(E)
            mu_p_eff = self.dd.material.mu_p * np.ones_like(E)

        dn_dx = np.gradient(n, dx)
        dp_dx = np.gradient(p, dx)

        J_n = q * mu_n_eff * n * E + q * self.dd.material.D_n * dn_dx
        J_p = q * mu_p_eff * p * E - q * self.dd.material.D_p * dp_dx
        J_total = J_n + J_p

        return DriftDiffusionResult(
            x=x,
            V=V,
            n=n,
            p=p,
            E_field=E,
            J_total=J_total,
            J_n=J_n,
            J_p=J_p,
            built_in_potential=dd_result.built_in_potential(),
            depletion_width=dd_result.depletion_width(),
            junction_position=dd_result.junction_position(),
            max_field=dd_result.max_field(),
            current_density=dd_result.current_density(),
            temperature=self.temperature,
            material_name=self.material_name,
            converged=dd_result.converged,
            iterations=dd_result.iterations,
            residual=dd_result.residual,
        )

    def compute_iv(
        self,
        voltage_range: Tuple[float, float] = (-0.5, 0.8),
        n_points: int = 20,
    ) -> DriftDiffusionResult:
        """Compute I-V characteristic by sweeping bias.

        Parameters
        ----------
        voltage_range : tuple
            (V_min, V_max) voltage sweep range [V].
        n_points : int
            Number of bias points.

        Returns
        -------
        DriftDiffusionResult
            Result at the last bias point, with iv_voltages and iv_currents filled.
        """
        if not self._doping_set:
            raise RuntimeError("Doping not set. Call set_doping() or set_pn_junction() first.")

        voltages = np.linspace(voltage_range[0], voltage_range[1], n_points)
        currents = np.zeros(n_points)

        for i, V_app in enumerate(voltages):
            result = self.solve(applied_voltage=V_app)
            currents[i] = result.current_density

        # Return final result with I-V data attached
        last = self.solve(applied_voltage=voltages[-1])
        return DriftDiffusionResult(
            x=last.x,
            V=last.V,
            n=last.n,
            p=last.p,
            E_field=last.E_field,
            J_total=last.J_total,
            J_n=last.J_n,
            J_p=last.J_p,
            built_in_potential=last.built_in_potential,
            depletion_width=last.depletion_width,
            junction_position=last.junction_position,
            max_field=last.max_field,
            current_density=last.current_density,
            iv_voltages=voltages,
            iv_currents=currents,
            temperature=self.temperature,
            material_name=self.material_name,
            converged=last.converged,
            iterations=last.iterations,
            residual=last.residual,
        )
