"""
Newtonian impact theory for hypersonic aerodynamics.

For hypersonic flow (M >> 1), pressure distribution can be approximated
using Newtonian impact theory:

    Cp = 2·sin²(θ)

where θ is the local surface angle to freestream.

Includes:
- Modified Newtonian theory
- Pressure coefficient distribution
- Normal and tangential force coefficients
- Simple shock relations
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class FreestreamConditions:
    """Freestream flow conditions"""
    velocity: float  # m/s
    pressure: float  # Pa
    temperature: float  # K
    density: float  # kg/m³
    gamma: float = 1.4  # Specific heat ratio

    @property
    def mach_number(self) -> float:
        """Compute Mach number"""
        R = 287.05  # Gas constant for air [J/(kg·K)]
        a = np.sqrt(self.gamma * R * self.temperature)
        return self.velocity / a

    @property
    def dynamic_pressure(self) -> float:
        """Dynamic pressure q = ½ρV²"""
        return 0.5 * self.density * self.velocity**2


class NewtonianFlow:
    """
    Newtonian impact theory for hypersonic flow.

    Assumptions:
    - Hypersonic flow (M >> 1)
    - Inviscid
    - Particles strike surface and lose normal momentum
    """

    @staticmethod
    def pressure_coefficient(theta: float, Cp_max: float = 2.0) -> float:
        """
        Newtonian pressure coefficient.

        Cp = Cp_max · sin²(θ)

        Args:
            theta: Surface angle to freestream [rad]
            Cp_max: Maximum Cp (stagnation value, typically 2.0)

        Returns:
            Pressure coefficient
        """
        return Cp_max * np.sin(theta)**2

    @staticmethod
    def modified_newtonian_Cp_max(M_inf: float, gamma: float = 1.4) -> float:
        """
        Modified Newtonian theory: compute Cp_max from Mach number.

        Uses exact stagnation pressure relation:
            Cp_max = (2/(γ·M²)) · [((γ+1)²·M²/(4γ·M² - 2(γ-1)))^(γ/(γ-1)) ·
                     ((1 - γ + 2γ·M²)/(γ+1)) - 1]

        Simplified for high Mach:
            Cp_max ≈ 2  (for M >> 1)

        Args:
            M_inf: Freestream Mach number
            gamma: Specific heat ratio

        Returns:
            Maximum pressure coefficient
        """
        if M_inf < 1.0:
            raise ValueError("Newtonian theory requires supersonic/hypersonic flow")

        # For hypersonic (M > 5), Cp_max ≈ 2
        if M_inf > 5.0:
            return 2.0

        # More accurate for lower supersonic Mach
        # Using normal shock relations
        M2 = M_inf**2
        numerator = (gamma + 1)**2 * M2
        denominator = 4*gamma*M2 - 2*(gamma - 1)

        pressure_ratio = (numerator / denominator)**(gamma/(gamma-1))
        pressure_ratio *= (1 - gamma + 2*gamma*M2) / (gamma + 1)

        Cp_max = (2 / (gamma * M2)) * (pressure_ratio - 1)

        return Cp_max

    @staticmethod
    def pressure(theta: float, freestream: FreestreamConditions) -> float:
        """
        Surface pressure using Newtonian theory.

        p = p_∞ + q_∞·Cp

        Args:
            theta: Surface angle [rad]
            freestream: Freestream conditions

        Returns:
            Surface pressure [Pa]
        """
        Cp_max = NewtonianFlow.modified_newtonian_Cp_max(
            freestream.mach_number, freestream.gamma
        )
        Cp = NewtonianFlow.pressure_coefficient(theta, Cp_max)

        p = freestream.pressure + freestream.dynamic_pressure * Cp

        return p

    @staticmethod
    def normal_force_coefficient(theta_array: np.ndarray, dA_array: np.ndarray,
                                 A_ref: float, Cp_max: float = 2.0) -> float:
        """
        Integrate normal force coefficient over surface.

        CN = (1/A_ref) ∫ Cp·cos(θ)·dA

        Args:
            theta_array: Surface angles [rad]
            dA_array: Differential areas [m²]
            A_ref: Reference area [m²]
            Cp_max: Maximum Cp

        Returns:
            Normal force coefficient
        """
        Cp_array = Cp_max * np.sin(theta_array)**2
        CN = np.sum(Cp_array * np.cos(theta_array) * dA_array) / A_ref

        return CN

    @staticmethod
    def axial_force_coefficient(theta_array: np.ndarray, dA_array: np.ndarray,
                               A_ref: float, Cp_max: float = 2.0) -> float:
        """
        Integrate axial force coefficient over surface.

        CA = (1/A_ref) ∫ Cp·sin(θ)·dA

        Args:
            theta_array: Surface angles [rad]
            dA_array: Differential areas [m²]
            A_ref: Reference area [m²]
            Cp_max: Maximum Cp

        Returns:
            Axial force coefficient
        """
        Cp_array = Cp_max * np.sin(theta_array)**2
        CA = np.sum(Cp_array * np.sin(theta_array) * dA_array) / A_ref

        return CA


class SimpleShapes:
    """
    Newtonian flow solutions for simple geometric shapes.
    """

    @staticmethod
    def flat_plate(alpha: float, Cp_max: float = 2.0) -> Tuple[float, float, float]:
        """
        Flat plate at angle of attack.

        Args:
            alpha: Angle of attack [rad]
            Cp_max: Maximum Cp

        Returns:
            (CN, CA, CL) - normal, axial, and lift coefficients
        """
        # Windward side
        CN_windward = Cp_max * np.sin(alpha)**2 * np.cos(alpha)

        # Leeward side (vacuum or base pressure)
        # Assuming vacuum: Cp_leeward = 0
        CN_leeward = 0.0

        CN = CN_windward - CN_leeward
        CA = Cp_max * np.sin(alpha)**3

        # Lift and drag
        CL = CN * np.cos(alpha) - CA * np.sin(alpha)

        return CN, CA, CL

    @staticmethod
    def cone(theta_c: float, Cp_max: float = 2.0) -> Tuple[float, float]:
        """
        Cone at zero angle of attack.

        Args:
            theta_c: Cone half-angle [rad]
            Cp_max: Maximum Cp

        Returns:
            (CD, CL) - drag and lift coefficients
        """
        # For cone at α=0, only axial force
        CD = Cp_max * np.sin(theta_c)**3

        # No lift at α=0
        CL = 0.0

        return CD, CL

    @staticmethod
    def sphere(Cp_max: float = 2.0) -> float:
        """
        Sphere drag coefficient.

        CD = (2/3) · Cp_max

        Args:
            Cp_max: Maximum Cp (at stagnation point)

        Returns:
            Drag coefficient
        """
        CD = (2.0 / 3.0) * Cp_max

        return CD

    @staticmethod
    def cylinder(Cp_max: float = 2.0) -> float:
        """
        Infinite cylinder drag coefficient (cross-flow).

        CD = Cp_max

        Args:
            Cp_max: Maximum Cp

        Returns:
            Drag coefficient (per unit length)
        """
        return Cp_max


class PrandtlMeyerExpansion:
    """
    Prandtl-Meyer expansion for supersonic flow around corners.

    Used for flow expansion on leeward surfaces.
    """

    @staticmethod
    def prandtl_meyer_function(M: float, gamma: float = 1.4) -> float:
        """
        Prandtl-Meyer function ν(M).

        ν(M) = √((γ+1)/(γ-1)) · arctan(√((γ-1)/(γ+1)·(M²-1))) - arctan(√(M²-1))

        Args:
            M: Mach number
            gamma: Specific heat ratio

        Returns:
            Prandtl-Meyer angle [rad]
        """
        if M < 1.0:
            raise ValueError("Prandtl-Meyer expansion requires supersonic flow")

        if M > 40.0:
            raise ValueError("Prandtl-Meyer theory invalid for M > 40 (use hypersonic methods)")

        # Clamp to avoid numerical overflow
        M_safe = min(M, 30.0)

        sqrt_term = np.sqrt((gamma + 1) / (gamma - 1))
        M2_minus_1 = M_safe**2 - 1.0

        # Avoid sqrt of negative numbers due to roundoff
        inner_term = max(0.0, (gamma - 1) / (gamma + 1) * M2_minus_1)
        outer_term = max(0.0, M2_minus_1)

        inner_sqrt = np.sqrt(inner_term)
        outer_sqrt = np.sqrt(outer_term)

        nu = sqrt_term * np.arctan(inner_sqrt) - np.arctan(outer_sqrt)

        return nu

    @staticmethod
    def mach_from_nu(nu: float, gamma: float = 1.4, M_guess: float = 2.0) -> float:
        """
        Find Mach number from Prandtl-Meyer angle (inverse function).

        Uses Newton iteration.

        Args:
            nu: Prandtl-Meyer angle [rad]
            gamma: Specific heat ratio
            M_guess: Initial Mach number guess

        Returns:
            Mach number
        """
        M = max(M_guess, 1.01)  # Ensure supersonic

        for _ in range(20):  # Newton iterations
            # Clamp M to valid range
            M = np.clip(M, 1.01, 30.0)

            nu_current = PrandtlMeyerExpansion.prandtl_meyer_function(M, gamma)
            error = nu - nu_current

            if abs(error) < 1e-8:
                break

            # Derivative dν/dM with safe denominators
            M2 = M**2
            M2_minus_1 = max(M2 - 1.0, 0.01)  # Avoid division by zero

            sqrt_M2_minus_1 = np.sqrt(M2_minus_1)
            inner_term = max((gamma - 1) / (gamma + 1) * M2_minus_1, 1e-10)

            term1 = np.sqrt((gamma + 1) / (gamma - 1))
            term2 = 1.0 / (1.0 + (gamma - 1) / (gamma + 1) * M2_minus_1)
            term3 = (gamma - 1) / (gamma + 1) * M / np.sqrt(inner_term)
            term4 = M / M2_minus_1

            d_nu_dM = term1 * term2 * term3 - term4 / sqrt_M2_minus_1

            # Newton update with safeguards
            if abs(d_nu_dM) < 1e-10:
                break  # Derivative too small, stop iteration

            M = M - error / d_nu_dM

        return np.clip(M, 1.01, 30.0)

    @staticmethod
    def expansion_pressure_ratio(M1: float, theta: float, gamma: float = 1.4) -> float:
        """
        Pressure ratio across Prandtl-Meyer expansion.

        Args:
            M1: Upstream Mach number (must be > 1)
            theta: Deflection angle [rad] (must be > 0 for expansion)
            gamma: Specific heat ratio

        Returns:
            p2/p1 (< 1 for expansion)
        """
        if M1 < 1.0:
            raise ValueError("Upstream Mach must be supersonic (M1 > 1)")

        if theta < 0:
            raise ValueError("Deflection angle must be positive for expansion")

        # Maximum theoretical deflection for Prandtl-Meyer
        nu_max = (np.sqrt((gamma + 1) / (gamma - 1)) * np.pi / 2 -
                  np.pi / 2) * 0.9  # 90% of theoretical max

        nu1 = PrandtlMeyerExpansion.prandtl_meyer_function(M1, gamma)

        if nu1 + theta > nu_max:
            raise ValueError(f"Deflection too large: exceeds Prandtl-Meyer limit")

        nu2 = nu1 + theta

        M2 = PrandtlMeyerExpansion.mach_from_nu(nu2, gamma, M_guess=min(M1 * 1.5, 25.0))

        # Isentropic relation with safe exponentiation
        M1_safe = min(M1, 30.0)
        M2_safe = min(M2, 30.0)

        num = 1.0 + (gamma - 1) / 2.0 * M1_safe**2
        den = 1.0 + (gamma - 1) / 2.0 * M2_safe**2

        pressure_ratio = (num / den)**(gamma / (gamma - 1))

        return pressure_ratio
