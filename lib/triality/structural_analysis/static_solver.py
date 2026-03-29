"""
Static structural analysis for beams and plates.

Includes:
- Euler-Bernoulli beam theory
- Plate bending (Kirchhoff theory)
- Stress analysis
- Margin of safety calculations
- Load case combinations
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class LoadType(Enum):
    """Load case type"""
    ULTIMATE = 'ultimate'
    LIMIT = 'limit'
    YIELD = 'yield'


@dataclass
class Material:
    """Structural material properties"""
    name: str
    youngs_modulus: float  # E [Pa]
    shear_modulus: float  # G [Pa]
    poissons_ratio: float  # ν
    density: float  # ρ [kg/m³]
    yield_strength: float  # σ_y [Pa]
    ultimate_strength: float  # σ_ult [Pa]

    # Optional: orthotropic properties
    E_longitudinal: Optional[float] = None
    E_transverse: Optional[float] = None
    G_lt: Optional[float] = None

    def is_orthotropic(self) -> bool:
        """Check if material is orthotropic"""
        return self.E_longitudinal is not None


class EulerBernoulliBeam:
    """
    Euler-Bernoulli beam analysis.

    Governing equation:
        EI·d⁴w/dx⁴ = q(x)

    where w = deflection, q = distributed load
    """

    def __init__(self, length: float, E: float, I: float, A: float):
        """
        Initialize beam.

        Args:
            length: Beam length [m]
            E: Young's modulus [Pa]
            I: Second moment of area [m⁴]
            A: Cross-sectional area [m²]
        """
        self.L = length
        self.E = E
        self.I = I
        self.A = A

    def cantilever_tip_deflection(self, P: float) -> float:
        """
        Deflection at tip of cantilever beam with point load.

        δ = PL³ / (3EI)

        Args:
            P: Point load at tip [N]

        Returns:
            Deflection [m]
        """
        delta = (P * self.L**3) / (3 * self.E * self.I)
        return delta

    def simply_supported_center_deflection(self, P: float) -> float:
        """
        Deflection at center of simply supported beam with center load.

        δ = PL³ / (48EI)

        Args:
            P: Point load at center [N]

        Returns:
            Deflection [m]
        """
        delta = (P * self.L**3) / (48 * self.E * self.I)
        return delta

    def cantilever_max_stress(self, P: float, c: float) -> float:
        """
        Maximum bending stress in cantilever beam.

        σ_max = M·c / I  where M = P·L

        Args:
            P: Point load at tip [N]
            c: Distance from neutral axis to outer fiber [m]

        Returns:
            Maximum stress [Pa]
        """
        M_max = P * self.L  # Maximum moment at fixed end
        sigma_max = M_max * c / self.I
        return sigma_max

    def uniform_load_deflection(self, w: float, x: float) -> float:
        """
        Deflection of simply supported beam with uniform load.

        δ(x) = (wx/24EI)·(L³ - 2Lx² + x³)

        Args:
            w: Uniform load [N/m]
            x: Position along beam [m]

        Returns:
            Deflection at x [m]
        """
        delta = (w * x / (24 * self.E * self.I)) * (
            self.L**3 - 2*self.L*x**2 + x**3
        )
        return delta

    def axial_stress(self, P_axial: float) -> float:
        """
        Axial stress.

        σ = P / A

        Args:
            P_axial: Axial load [N]

        Returns:
            Axial stress [Pa]
        """
        return P_axial / self.A


class PlateAnalysis:
    """
    Thin plate bending analysis (Kirchhoff theory).

    For isotropic plates:
        D·∇⁴w = q

    where D = Et³/(12(1-ν²)) is flexural rigidity
    """

    def __init__(self, a: float, b: float, thickness: float, E: float, nu: float):
        """
        Initialize rectangular plate.

        Args:
            a: Plate dimension in x [m]
            b: Plate dimension in y [m]
            thickness: Plate thickness [m]
            E: Young's modulus [Pa]
            nu: Poisson's ratio
        """
        self.a = a
        self.b = b
        self.t = thickness
        self.E = E
        self.nu = nu

        # Flexural rigidity
        self.D = (E * thickness**3) / (12 * (1 - nu**2))

    def simply_supported_uniform_load_deflection(self, q: float) -> float:
        """
        Maximum deflection of simply supported rectangular plate under uniform load.

        w_max ≈ α·q·a⁴/D

        where α depends on aspect ratio b/a

        Args:
            q: Uniform pressure [Pa]

        Returns:
            Maximum deflection [m]
        """
        # Aspect ratio
        beta = self.b / self.a

        # Coefficient (approximate)
        if beta >= 1.0:
            alpha = 0.0138  # For square or b > a
        else:
            alpha = 0.0138 * (1 + beta**4)

        w_max = alpha * q * self.a**4 / self.D

        return w_max

    def max_bending_stress(self, q: float) -> float:
        """
        Maximum bending stress in plate.

        σ_max = β·q·a²/t²

        Args:
            q: Uniform pressure [Pa]

        Returns:
            Maximum stress [Pa]
        """
        # Stress coefficient (depends on boundary conditions and aspect ratio)
        # For simply supported plate
        beta_stress = 0.5  # Approximate

        sigma_max = beta_stress * q * self.a**2 / self.t**2

        return sigma_max


class MarginOfSafety:
    """
    Margin of safety calculations per aerospace standards.

    MS = (Allowable / Applied) - 1

    MS > 0: Safe
    MS = 0: At limit
    MS < 0: Failure
    """

    @staticmethod
    def compute_ms(allowable: float, applied: float) -> float:
        """
        Compute margin of safety.

        MS = (F_allowable / F_applied) - 1

        Args:
            allowable: Allowable stress/load
            applied: Applied stress/load

        Returns:
            Margin of safety
        """
        if applied <= 0:
            return float('inf')  # No load applied

        ms = (allowable / applied) - 1.0

        return ms

    @staticmethod
    def compute_ms_with_factors(allowable: float, applied: float,
                                factor_of_safety: float = 1.5,
                                knockdown_factor: float = 1.0) -> float:
        """
        Compute margin with safety factors.

        MS = (Allowable · KD / (Applied · FS)) - 1

        Args:
            allowable: Allowable stress/load
            applied: Applied stress/load
            factor_of_safety: Safety factor (typically 1.5 for ultimate)
            knockdown_factor: Material knockdown (< 1.0 for damage, etc.)

        Returns:
            Margin of safety
        """
        effective_allowable = allowable * knockdown_factor
        effective_applied = applied * factor_of_safety

        return MarginOfSafety.compute_ms(effective_allowable, effective_applied)

    @staticmethod
    def is_safe(ms: float) -> bool:
        """Check if margin of safety is positive (safe)"""
        return ms >= 0.0

    @staticmethod
    def combined_loading_ms(sigma_axial: float, sigma_bending: float,
                           sigma_allowable: float, interaction: str = 'linear') -> float:
        """
        Margin of safety for combined loading.

        Args:
            sigma_axial: Axial stress
            sigma_bending: Bending stress
            sigma_allowable: Allowable stress
            interaction: 'linear' or 'sreiner' (interaction formula)

        Returns:
            Combined margin of safety
        """
        if interaction == 'linear':
            # Linear interaction
            sigma_combined = abs(sigma_axial) + abs(sigma_bending)
        elif interaction == 'sreiner':
            # Sreiner interaction (for buckling)
            R_axial = abs(sigma_axial) / sigma_allowable
            R_bending = abs(sigma_bending) / sigma_allowable
            R_combined = R_axial + R_bending  # Simplified
            sigma_combined = R_combined * sigma_allowable
        else:
            raise ValueError(f"Unknown interaction type: {interaction}")

        ms = MarginOfSafety.compute_ms(sigma_allowable, sigma_combined)

        return ms


class LoadCaseCombination:
    """
    Load case combination analysis.

    Combines multiple load cases with load factors.
    """

    def __init__(self):
        self.load_cases: List[dict] = []

    def add_load_case(self, name: str, loads: dict, load_factor: float = 1.0):
        """
        Add load case.

        Args:
            name: Load case name
            loads: Dictionary of load components
            load_factor: Load multiplication factor
        """
        self.load_cases.append({
            'name': name,
            'loads': loads,
            'factor': load_factor
        })

    def combine_loads(self, case_indices: List[int], combination_factors: List[float]) -> dict:
        """
        Combine multiple load cases.

        Args:
            case_indices: Indices of load cases to combine
            combination_factors: Factors for each case

        Returns:
            Combined load dictionary
        """
        combined = {}

        for idx, factor in zip(case_indices, combination_factors):
            case = self.load_cases[idx]
            case_factor = case['factor'] * factor

            for load_name, load_value in case['loads'].items():
                if load_name not in combined:
                    combined[load_name] = 0.0
                combined[load_name] += case_factor * load_value

        return combined

    def envelope_analysis(self) -> dict:
        """
        Find envelope (maximum) of all load cases.

        Returns:
            Dictionary with maximum value for each load component
        """
        envelope = {}

        for case in self.load_cases:
            for load_name, load_value in case['loads'].items():
                factored_value = abs(load_value * case['factor'])

                if load_name not in envelope:
                    envelope[load_name] = factored_value
                else:
                    envelope[load_name] = max(envelope[load_name], factored_value)

        return envelope


# Pre-defined materials
MATERIALS = {
    'AL7075-T6': Material(
        name='Aluminum 7075-T6',
        youngs_modulus=71.7e9,  # Pa
        shear_modulus=26.9e9,
        poissons_ratio=0.33,
        density=2810,  # kg/m³
        yield_strength=503e6,  # Pa
        ultimate_strength=572e6
    ),
    'AL6061-T6': Material(
        name='Aluminum 6061-T6',
        youngs_modulus=68.9e9,
        shear_modulus=26.0e9,
        poissons_ratio=0.33,
        density=2700,
        yield_strength=276e6,
        ultimate_strength=310e6
    ),
    'Ti-6Al-4V': Material(
        name='Titanium Ti-6Al-4V',
        youngs_modulus=113.8e9,
        shear_modulus=44.0e9,
        poissons_ratio=0.342,
        density=4430,
        yield_strength=880e6,
        ultimate_strength=950e6
    ),
    'Steel-4130': Material(
        name='Steel 4130',
        youngs_modulus=205e9,
        shear_modulus=80e9,
        poissons_ratio=0.29,
        density=7850,
        yield_strength=460e6,
        ultimate_strength=560e6
    )
}
