"""
Advanced Physics Models for Production Device Simulation

Adds temperature dependence, generation-recombination, and field-dependent effects
to make Layer 3 truly powerful for production use.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

# Physical constants
q = 1.602e-19      # Elementary charge [C]
k_B = 1.381e-23    # Boltzmann constant [J/K]
h = 6.626e-34      # Planck constant [J·s]
m_0 = 9.109e-31    # Electron mass [kg]


@dataclass
class TemperatureDependentMaterial:
    """
    Temperature-dependent semiconductor material properties.

    Includes models for:
    - Bandgap narrowing with temperature
    - Intrinsic carrier concentration vs T
    - Mobility vs temperature
    - Thermal velocity
    """
    name: str
    T: float = 300  # Temperature [K]

    # Silicon parameters at 300K (reference)
    E_g0: float = 1.12  # Bandgap at 0K [eV]
    alpha: float = 4.73e-4  # Bandgap temperature coefficient [eV/K]
    beta: float = 636  # Bandgap temperature parameter [K]

    eps_r: float = 11.7  # Relative permittivity

    # Mobility parameters
    mu_n_300: float = 1400  # Electron mobility at 300K [cm²/(V·s)]
    mu_p_300: float = 450   # Hole mobility at 300K [cm²/(V·s)]
    mobility_exp: float = -2.42  # Temperature exponent for mobility

    # Effective masses (in units of m_0)
    m_n_eff: float = 1.08  # Electron effective mass
    m_p_eff: float = 0.56  # Hole effective mass

    @staticmethod
    def Silicon(T: float = 300):
        """Create Silicon material at specified temperature [K]."""
        return TemperatureDependentMaterial(name='Silicon', T=T)

    @property
    def E_g(self) -> float:
        """
        Bandgap energy at temperature T [eV].

        Uses Varshni model: E_g(T) = E_g0 - alpha*T²/(T + beta)
        """
        return self.E_g0 - self.alpha * self.T**2 / (self.T + self.beta)

    @property
    def n_i(self) -> float:
        """
        Intrinsic carrier concentration at temperature T [cm^-3].

        Uses: n_i² = N_c * N_v * exp(-E_g / kT)
        Where N_c, N_v are effective density of states.
        """
        # Effective density of states
        N_c = 2 * (2 * np.pi * self.m_n_eff * m_0 * k_B * self.T / h**2)**(3/2) * 1e-6  # [cm^-3]
        N_v = 2 * (2 * np.pi * self.m_p_eff * m_0 * k_B * self.T / h**2)**(3/2) * 1e-6  # [cm^-3]

        # Intrinsic concentration
        n_i_squared = N_c * N_v * np.exp(-self.E_g * q / (k_B * self.T))
        return np.sqrt(n_i_squared)

    @property
    def V_T(self) -> float:
        """Thermal voltage at temperature T [V]."""
        return k_B * self.T / q

    @property
    def mu_n(self) -> float:
        """
        Electron mobility at temperature T [cm²/(V·s)].

        Uses: mu(T) = mu_300 * (T/300)^exp
        """
        return self.mu_n_300 * (self.T / 300) ** self.mobility_exp

    @property
    def mu_p(self) -> float:
        """Hole mobility at temperature T [cm²/(V·s)]."""
        return self.mu_p_300 * (self.T / 300) ** self.mobility_exp

    @property
    def D_n(self) -> float:
        """Electron diffusion coefficient [cm²/s]."""
        return self.mu_n * self.V_T

    @property
    def D_p(self) -> float:
        """Hole diffusion coefficient [cm²/s]."""
        return self.mu_p * self.V_T

    @property
    def v_th_n(self) -> float:
        """Electron thermal velocity [cm/s]."""
        return np.sqrt(3 * k_B * self.T / (self.m_n_eff * m_0)) * 100  # m/s -> cm/s

    @property
    def v_th_p(self) -> float:
        """Hole thermal velocity [cm/s]."""
        return np.sqrt(3 * k_B * self.T / (self.m_p_eff * m_0)) * 100  # m/s -> cm/s


class ShockleyReadHall:
    """
    Shockley-Read-Hall generation-recombination model.

    U_SRH = (n*p - n_i²) / (tau_p*(n + n_1) + tau_n*(p + p_1))

    Where:
    - tau_n, tau_p are minority carrier lifetimes
    - n_1 = n_i * exp(E_trap / kT)
    - p_1 = n_i * exp(-E_trap / kT)
    - E_trap is trap energy level
    """

    def __init__(self, tau_n: float = 1e-6, tau_p: float = 1e-6,
                 E_trap: float = 0.0):
        """
        Initialize SRH model.

        Args:
            tau_n: Electron lifetime [s]
            tau_p: Hole lifetime [s]
            E_trap: Trap energy relative to intrinsic level [eV]
        """
        self.tau_n = tau_n
        self.tau_p = tau_p
        self.E_trap = E_trap

    def recombination_rate(self, n: np.ndarray, p: np.ndarray,
                          n_i: float, V_T: float) -> np.ndarray:
        """
        Calculate SRH recombination rate [cm^-3/s].

        Args:
            n: Electron concentration [cm^-3]
            p: Hole concentration [cm^-3]
            n_i: Intrinsic carrier concentration [cm^-3]
            V_T: Thermal voltage [V]

        Returns:
            U_SRH: Recombination rate [cm^-3/s]
        """
        # Trap level carrier concentrations
        n_1 = n_i * np.exp(self.E_trap / V_T)
        p_1 = n_i * np.exp(-self.E_trap / V_T)

        # SRH recombination rate
        numerator = n * p - n_i**2
        denominator = self.tau_p * (n + n_1) + self.tau_n * (p + p_1)

        U_SRH = numerator / np.maximum(denominator, 1e-30)
        return U_SRH

    def lifetime_effective(self, n: np.ndarray, p: np.ndarray,
                          n_i: float, V_T: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate effective carrier lifetimes.

        Returns:
            (tau_n_eff, tau_p_eff): Effective lifetimes [s]
        """
        n_1 = n_i * np.exp(self.E_trap / V_T)
        p_1 = n_i * np.exp(-self.E_trap / V_T)

        tau_n_eff = self.tau_n * (1 + p / p_1)
        tau_p_eff = self.tau_p * (1 + n / n_1)

        return tau_n_eff, tau_p_eff


class FieldDependentMobility:
    """
    Field-dependent mobility with velocity saturation.

    mu_eff(E) = mu_0 / (1 + (mu_0 * E / v_sat))

    At low fields: mu_eff ≈ mu_0
    At high fields: mu_eff ≈ v_sat / E (velocity saturation)
    """

    def __init__(self, v_sat_n: float = 1e7, v_sat_p: float = 8e6):
        """
        Initialize field-dependent mobility model.

        Args:
            v_sat_n: Electron saturation velocity [cm/s]
            v_sat_p: Hole saturation velocity [cm/s]
        """
        self.v_sat_n = v_sat_n
        self.v_sat_p = v_sat_p

    def mobility_n(self, E: np.ndarray, mu_0_n: float) -> np.ndarray:
        """
        Calculate field-dependent electron mobility.

        Args:
            E: Electric field magnitude [V/cm]
            mu_0_n: Low-field mobility [cm²/(V·s)]

        Returns:
            mu_n_eff: Effective mobility [cm²/(V·s)]
        """
        return mu_0_n / (1 + mu_0_n * np.abs(E) / self.v_sat_n)

    def mobility_p(self, E: np.ndarray, mu_0_p: float) -> np.ndarray:
        """
        Calculate field-dependent hole mobility.

        Args:
            E: Electric field magnitude [V/cm]
            mu_0_p: Low-field mobility [cm²/(V·s)]

        Returns:
            mu_p_eff: Effective mobility [cm²/(V·s)]
        """
        return mu_0_p / (1 + mu_0_p * np.abs(E) / self.v_sat_p)

    def velocity_n(self, E: np.ndarray, mu_0_n: float) -> np.ndarray:
        """Calculate electron drift velocity [cm/s]."""
        mu_n_eff = self.mobility_n(E, mu_0_n)
        return mu_n_eff * np.abs(E)

    def velocity_p(self, E: np.ndarray, mu_0_p: float) -> np.ndarray:
        """Calculate hole drift velocity [cm/s]."""
        mu_p_eff = self.mobility_p(E, mu_0_p)
        return mu_p_eff * np.abs(E)


class ImprovedCurrentCalculator:
    """
    Improved current density calculation for I-V characteristics.

    Uses proper drift-diffusion current:
    J_n = q * mu_n * n * E + q * D_n * dn/dx
    J_p = q * mu_p * p * E - q * D_p * dp/dx

    Total: J = J_n + J_p
    """

    @staticmethod
    def calculate_current_density(x: np.ndarray, V: np.ndarray,
                                  n: np.ndarray, p: np.ndarray,
                                  material: TemperatureDependentMaterial,
                                  use_field_dependence: bool = False) -> np.ndarray:
        """
        Calculate total current density [A/cm²].

        Args:
            x: Position array [cm]
            V: Potential [V]
            n: Electron concentration [cm^-3]
            p: Hole concentration [cm^-3]
            material: Material properties
            use_field_dependence: Use field-dependent mobility

        Returns:
            J: Current density [A/cm²]
        """
        dx = x[1] - x[0]

        # Electric field E = -dV/dx
        E = -np.gradient(V, dx)

        # Carrier gradients
        dn_dx = np.gradient(n, dx)
        dp_dx = np.gradient(p, dx)

        # Mobility (field-dependent or constant)
        if use_field_dependence:
            field_model = FieldDependentMobility()
            mu_n = field_model.mobility_n(E, material.mu_n)
            mu_p = field_model.mobility_p(E, material.mu_p)
        else:
            mu_n = material.mu_n * np.ones_like(E)
            mu_p = material.mu_p * np.ones_like(E)

        # Current densities
        J_n_drift = q * mu_n * n * E
        J_n_diff = q * material.D_n * dn_dx
        J_n = J_n_drift + J_n_diff

        J_p_drift = q * mu_p * p * E
        J_p_diff = -q * material.D_p * dp_dx
        J_p = J_p_drift + J_p_diff

        # Total current
        J_total = J_n + J_p

        return J_total

    @staticmethod
    def integrate_current(x: np.ndarray, J: np.ndarray, area: float = 1e-8) -> float:
        """
        Integrate current density to get total current [A].

        Args:
            x: Position [cm]
            J: Current density [A/cm²]
            area: Device cross-sectional area [cm²]

        Returns:
            I: Total current [A]
        """
        # Use current at center of device (should be constant in 1D steady-state)
        J_center = J[len(J) // 2]
        I = J_center * area
        return I


# Export key classes
__all__ = [
    'TemperatureDependentMaterial',
    'ShockleyReadHall',
    'FieldDependentMobility',
    'ImprovedCurrentCalculator',
]
