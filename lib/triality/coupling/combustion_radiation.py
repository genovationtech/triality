"""
Combustion → Soot → Radiation Transport Chain Coupling

Couples three physics solvers into a deep chain:
    combustion_chemistry → soot_model → radiation_transport

Physics chain:
    1. Combustion chemistry provides:
       - Temperature field T(x)
       - Species concentrations (CO2, H2O, CH4, etc.)
       - Local equivalence ratio

    2. Soot model receives from combustion:
       - Temperature T(x) → soot formation/oxidation rates
       - Species (C2H2 as soot precursor, O2 for oxidation)
       Produces: soot volume fraction fv(x), soot number density

    3. Radiation transport receives from combustion + soot:
       - T(x) → Planck emission source
       - CO2, H2O concentrations → gas absorption (WSGG/Planck mean)
       - fv(x) → soot absorption: kappa_soot = C * fv * T
       Produces: radiative heat flux q_rad(x), div(q_rad)

    4. Feedback to combustion:
       - div(q_rad) → energy source/sink in energy equation

Coupling flow:
    1. Run combustion chemistry → T, species
    2. Pass T, species to soot model → fv, Nd
    3. Compute total absorption: kappa = kappa_gas(T, species) + kappa_soot(fv, T)
    4. Run radiation transport → q_rad, div(q_rad)
    5. Feed div(q_rad) back to combustion as radiative source term
    6. Iterate until converged

Unlock: radiative heat transfer in flames, soot-radiation coupling, TRI effects
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from triality.core.fields import PhysicsState, PhysicsField
from triality.core.units import UnitMetadata


_SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann [W/(m^2*K^4)]


@dataclass
class CombustionRadiationResult:
    """Result from the combustion-soot-radiation chain.

    Attributes
    ----------
    x : np.ndarray
        Spatial coordinate [m].
    temperature : np.ndarray
        Temperature profile [K].
    species : Dict[str, np.ndarray]
        Species mass fractions.
    soot_volume_fraction : np.ndarray
        Soot volume fraction [-].
    soot_number_density : np.ndarray
        Soot number density [1/m^3].
    kappa_gas : np.ndarray
        Gas absorption coefficient [1/m].
    kappa_soot : np.ndarray
        Soot absorption coefficient [1/m].
    kappa_total : np.ndarray
        Total absorption coefficient [1/m].
    q_rad : np.ndarray
        Radiative heat flux [W/m^2].
    div_q_rad : np.ndarray
        Divergence of radiative heat flux [W/m^3].
    q_wall : float
        Radiative flux at wall [W/m^2].
    soot_emission_fraction : float
        Fraction of total emission from soot (vs gas).
    n_coupling_iterations : int
        Number of chain coupling iterations.
    converged : bool
        Whether the chain converged.
    """
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    temperature: np.ndarray = field(default_factory=lambda: np.array([]))
    species: Dict[str, np.ndarray] = field(default_factory=dict)
    soot_volume_fraction: np.ndarray = field(default_factory=lambda: np.array([]))
    soot_number_density: np.ndarray = field(default_factory=lambda: np.array([]))
    kappa_gas: np.ndarray = field(default_factory=lambda: np.array([]))
    kappa_soot: np.ndarray = field(default_factory=lambda: np.array([]))
    kappa_total: np.ndarray = field(default_factory=lambda: np.array([]))
    q_rad: np.ndarray = field(default_factory=lambda: np.array([]))
    div_q_rad: np.ndarray = field(default_factory=lambda: np.array([]))
    q_wall: float = 0.0
    soot_emission_fraction: float = 0.0
    n_coupling_iterations: int = 0
    converged: bool = False


class CombustionRadiationCoupler:
    """Couples combustion chemistry, soot, and radiation transport.

    Implements the full chain:
        Combustion → Soot → Radiation → (feedback to combustion)

    Parameters
    ----------
    n_cells : int
        Number of spatial cells.
    length : float
        Domain length [m].
    pressure : float
        Pressure [Pa].
    T_wall : float
        Wall temperature [K] (radiation BC).
    T_ambient : float
        Ambient temperature [K].
    soot_yield_factor : float
        Soot yield calibration factor.
    C_soot_abs : float
        Soot absorption constant [1/(m*K)]: kappa_soot = C * fv * T.

    Example
    -------
    >>> coupler = CombustionRadiationCoupler(n_cells=100, length=0.1)
    >>> T = np.linspace(300, 2000, 100)
    >>> Y_fuel = np.linspace(0.06, 0.0, 100)
    >>> result = coupler.solve(T, species={"fuel": Y_fuel})
    """

    def __init__(
        self,
        n_cells: int = 100,
        length: float = 0.1,
        pressure: float = 101325.0,
        T_wall: float = 300.0,
        T_ambient: float = 300.0,
        soot_yield_factor: float = 1.0,
        C_soot_abs: float = 1862.0,
    ):
        self.n_cells = n_cells
        self.length = length
        self.pressure = pressure
        self.T_wall = T_wall
        self.T_ambient = T_ambient
        self.soot_yield_factor = soot_yield_factor
        self.C_soot_abs = C_soot_abs

    def solve(
        self,
        temperature: np.ndarray,
        species: Optional[Dict[str, np.ndarray]] = None,
        velocity: Optional[np.ndarray] = None,
        combustion_state: Optional[PhysicsState] = None,
        max_iterations: int = 10,
        tolerance: float = 1e-3,
        relaxation: float = 0.7,
    ) -> CombustionRadiationResult:
        """Run the combustion-soot-radiation chain.

        Parameters
        ----------
        temperature : np.ndarray
            Temperature profile from combustion solver [K].
        species : dict, optional
            Species mass fractions. Expected keys: 'fuel', 'CO2', 'H2O', 'O2'.
        velocity : np.ndarray, optional
            Velocity field [m/s] for soot advection.
        combustion_state : PhysicsState, optional
            Alternative input via PhysicsState.
        max_iterations : int
            Max chain coupling iterations.
        tolerance : float
            Convergence tolerance on temperature change.
        relaxation : float
            Under-relaxation factor.

        Returns
        -------
        CombustionRadiationResult
        """
        n = self.n_cells
        L = self.length
        dx = L / max(n - 1, 1)
        x = np.linspace(0, L, n)

        # Extract from PhysicsState if provided
        if combustion_state is not None:
            if combustion_state.has("temperature"):
                temperature = combustion_state["temperature"]
            if species is None:
                species = {}
                for sp_name in ["fuel", "CO2", "H2O", "O2"]:
                    if combustion_state.has(f"species_{sp_name}"):
                        species[sp_name] = combustion_state[f"species_{sp_name}"]

        T = np.asarray(temperature[:n], dtype=float)
        if species is None:
            species = {}

        # Default species if not provided
        Y_fuel = species.get("fuel", np.zeros(n))
        Y_CO2 = species.get("CO2", np.maximum(0.1 * (1.0 - Y_fuel / max(np.max(Y_fuel), 1e-10)), 0))
        Y_H2O = species.get("H2O", Y_CO2 * 0.5)
        Y_O2 = species.get("O2", np.maximum(0.21 - 3.5 * Y_fuel, 0))

        if velocity is None:
            velocity = np.ones(n) * 1.0

        T_prev = T.copy()
        converged = False
        n_iter = 0

        for iteration in range(max_iterations):
            n_iter = iteration + 1

            # ---- Step 1: Soot model ----
            fv, Nd = self._compute_soot(T, Y_fuel, Y_O2, velocity, dx)

            # ---- Step 2: Gas absorption (WSGG simplified) ----
            kappa_gas = self._compute_gas_absorption(T, Y_CO2, Y_H2O)

            # ---- Step 3: Soot absorption ----
            kappa_soot = self.C_soot_abs * fv * T

            # ---- Step 4: Total absorption ----
            kappa_total = kappa_gas + kappa_soot

            # ---- Step 5: Radiation transport (1-D tangent slab) ----
            q_rad, div_q_rad, q_wall = self._solve_radiation(
                x, T, kappa_total
            )

            # ---- Step 6: Update temperature with radiative source ----
            # T_new accounts for radiative cooling/heating
            rho = self.pressure / (287.0 * T)  # ideal gas
            cp = 1000.0  # simplified
            # Steady-state energy balance: modify T by div(q_rad)
            # This is a simplified update (not full energy equation)
            dT_rad = -div_q_rad / (rho * cp) * 0.001  # small pseudo-time step
            T_new = T + dT_rad
            T_new = np.clip(T_new, 200.0, 5000.0)

            # Under-relax
            T_relaxed = relaxation * T_new + (1.0 - relaxation) * T

            # Check convergence
            dT = float(np.max(np.abs(T_relaxed - T_prev)))
            T_scale = max(float(np.max(T_prev)), 1.0)
            if dT / T_scale < tolerance and iteration > 0:
                converged = True
                T = T_relaxed
                break

            T_prev = T_relaxed.copy()
            T = T_relaxed

        # Soot emission fraction
        emission_soot = np.sum(kappa_soot * _SIGMA_SB * T**4)
        emission_gas = np.sum(kappa_gas * _SIGMA_SB * T**4)
        emission_total = emission_soot + emission_gas
        soot_frac = emission_soot / max(emission_total, 1e-30)

        return CombustionRadiationResult(
            x=x,
            temperature=T,
            species={"fuel": Y_fuel, "CO2": Y_CO2, "H2O": Y_H2O, "O2": Y_O2},
            soot_volume_fraction=fv,
            soot_number_density=Nd,
            kappa_gas=kappa_gas,
            kappa_soot=kappa_soot,
            kappa_total=kappa_total,
            q_rad=q_rad,
            div_q_rad=div_q_rad,
            q_wall=q_wall,
            soot_emission_fraction=float(soot_frac),
            n_coupling_iterations=n_iter,
            converged=converged,
        )

    def _compute_soot(self, T: np.ndarray, Y_fuel: np.ndarray,
                       Y_O2: np.ndarray, u: np.ndarray,
                       dx: float) -> tuple:
        """Simplified two-equation soot model.

        Computes soot volume fraction and number density from:
        - Nucleation: from fuel pyrolysis products (acetylene proxy)
        - Surface growth: HACA mechanism (simplified)
        - Oxidation: O2 attack
        - Coagulation: number density reduction

        Returns
        -------
        fv : np.ndarray  Soot volume fraction
        Nd : np.ndarray  Soot number density [1/m^3]
        """
        n = len(T)
        fv = np.zeros(n)
        Nd = np.zeros(n)

        rho_soot = 1800.0  # soot density [kg/m^3]
        d_min = 1e-9        # minimum soot particle diameter [m]

        for i in range(1, n):
            T_local = max(T[i], 300.0)

            # Nucleation rate (Arrhenius from fuel proxy)
            R_nuc = (4e12 * Y_fuel[i] * np.exp(-21000.0 / T_local)
                     * self.soot_yield_factor)
            R_nuc = max(R_nuc, 0.0)

            # Surface growth rate
            S_growth = 1e4 * np.sqrt(fv[i-1]) * np.exp(-12000.0 / T_local)
            S_growth = max(S_growth, 0.0)

            # Oxidation rate
            R_ox = 1e5 * Y_O2[i] * fv[i-1] * np.exp(-19000.0 / T_local)
            R_ox = max(R_ox, 0.0)

            # Update volume fraction (advection + source)
            dfv_dt = (R_nuc + S_growth - R_ox) * 1e-6  # scale factor
            fv[i] = max(fv[i-1] + dfv_dt * dx / max(abs(u[i]), 0.01), 0.0)
            fv[i] = min(fv[i], 1e-4)  # physical cap

            # Number density
            if fv[i] > 0:
                d_p = max((6.0 * fv[i] / (np.pi * max(Nd[i-1], 1.0)))**(1.0/3.0), d_min)
            else:
                d_p = d_min
            Nd[i] = max(6.0 * fv[i] / (np.pi * d_p**3), 0.0)

        return fv, Nd

    def _compute_gas_absorption(self, T: np.ndarray,
                                  Y_CO2: np.ndarray,
                                  Y_H2O: np.ndarray) -> np.ndarray:
        """Compute gas absorption using Planck-mean approximation.

        Simplified WSGG-like model:
            kappa_gas = a_CO2 * p_CO2 + a_H2O * p_H2O
        where partial pressures are estimated from mass fractions.
        """
        # Partial pressures (ideal gas approximation)
        M_CO2 = 44.0
        M_H2O = 18.0
        M_air = 29.0

        x_CO2 = (Y_CO2 / M_CO2) / (1.0 / M_air)  # mole fraction (approximate)
        x_H2O = (Y_H2O / M_H2O) / (1.0 / M_air)

        p_CO2 = x_CO2 * self.pressure  # Pa
        p_H2O = x_H2O * self.pressure

        # Planck-mean absorption coefficients [1/(m*atm)]
        # Temperature-dependent fit (simplified)
        a_CO2 = np.where(T > 500, 0.3 * (T / 1000.0)**(-0.5), 0.01)
        a_H2O = np.where(T > 500, 0.2 * (T / 1000.0)**(-0.3), 0.01)

        kappa = (a_CO2 * p_CO2 + a_H2O * p_H2O) / 101325.0  # convert to 1/m
        return np.maximum(kappa, 1e-6)

    def _solve_radiation(self, x: np.ndarray, T: np.ndarray,
                          kappa: np.ndarray) -> tuple:
        """1-D radiation transport (tangent-slab, S2 discrete ordinates).

        Returns
        -------
        q_rad : np.ndarray  Radiative heat flux [W/m^2]
        div_q_rad : np.ndarray  Divergence [W/m^3]
        q_wall : float  Wall heat flux [W/m^2]
        """
        n = len(x)
        dx = x[1] - x[0] if n > 1 else 1.0

        # S2 discrete ordinates: 2 directions (mu = +/- 0.5774)
        mu_pos = 0.5773502692
        mu_neg = -mu_pos
        w = 1.0  # weight per direction (half-range)

        I_pos = np.zeros(n)  # forward intensity
        I_neg = np.zeros(n)  # backward intensity

        # Blackbody source
        I_b = _SIGMA_SB * T**4 / np.pi  # isotropic intensity

        # Forward sweep (left to right)
        I_pos[0] = _SIGMA_SB * self.T_wall**4 / np.pi  # wall emission
        for i in range(1, n):
            tau = kappa[i] * dx / mu_pos
            if tau < 0.01:
                I_pos[i] = I_pos[i-1] + dx / mu_pos * kappa[i] * (I_b[i] - I_pos[i-1])
            else:
                I_pos[i] = I_pos[i-1] * np.exp(-tau) + I_b[i] * (1.0 - np.exp(-tau))

        # Backward sweep (right to left)
        I_neg[-1] = _SIGMA_SB * self.T_ambient**4 / np.pi
        for i in range(n - 2, -1, -1):
            tau = kappa[i] * dx / mu_pos
            if tau < 0.01:
                I_neg[i] = I_neg[i+1] + dx / mu_pos * kappa[i] * (I_b[i] - I_neg[i+1])
            else:
                I_neg[i] = I_neg[i+1] * np.exp(-tau) + I_b[i] * (1.0 - np.exp(-tau))

        # Heat flux: q = 2*pi * integral(I * mu * dmu)
        # For S2: q = 2*pi * w * (mu_pos * I_pos + mu_neg * I_neg)
        q_rad = 2.0 * np.pi * w * mu_pos * (I_pos - I_neg)

        # Divergence: div(q_rad) = kappa * (4*sigma*T^4 - G)
        G = 2.0 * np.pi * w * (I_pos + I_neg)
        div_q_rad = kappa * (4.0 * _SIGMA_SB * T**4 - G)

        q_wall = float(q_rad[0])

        return q_rad, div_q_rad, q_wall

    def export_state(self, result: CombustionRadiationResult) -> PhysicsState:
        """Export as PhysicsState."""
        state = PhysicsState(solver_name="combustion_radiation")
        state.set_field("temperature", result.temperature, "K", grid=result.x)
        state.set_field("radiative_heat_flux", result.q_rad, "W/m^2", grid=result.x)
        state.set_field("heat_source", result.div_q_rad, "W/m^3", grid=result.x)
        state.set_field("absorption_coefficient", result.kappa_total, "1/m", grid=result.x)
        state.metadata["soot_emission_fraction"] = result.soot_emission_fraction
        state.metadata["q_wall"] = result.q_wall
        return state
