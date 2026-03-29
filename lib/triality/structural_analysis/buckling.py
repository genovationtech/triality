"""
Buck buckling analysis for columns and panels.

Includes:
- Euler column buckling
- Panel buckling (isotropic and orthotropic)
- Plate buckling coefficients
- Crippling stress
- Interaction equations for combined loading
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class BucklingMode(Enum):
    """Buckling failure mode"""
    COLUMN = 'column'
    PANEL = 'panel'
    CRIPPLING = 'crippling'
    SHEAR = 'shear'


class EndCondition(Enum):
    """Column end condition"""
    PINNED_PINNED = 'pinned-pinned'  # K = 1.0
    FIXED_FREE = 'fixed-free'  # K = 0.25
    FIXED_PINNED = 'fixed-pinned'  # K = 2.0
    FIXED_FIXED = 'fixed-fixed'  # K = 4.0


class EulerColumnBuckling:
    """
    Euler column buckling analysis.

    Critical load:
        P_cr = (π²·E·I) / (K·L²)

    where K = effective length factor (depends on end conditions)
    """

    # Effective length factors
    K_FACTORS = {
        EndCondition.PINNED_PINNED: 1.0,
        EndCondition.FIXED_FREE: 0.25,
        EndCondition.FIXED_PINNED: 2.0,
        EndCondition.FIXED_FIXED: 4.0
    }

    @staticmethod
    def critical_load(E: float, I: float, L: float, end_condition: EndCondition) -> float:
        """
        Compute Euler critical buckling load.

        P_cr = (π²·E·I) / (KL)²

        Args:
            E: Young's modulus [Pa]
            I: Second moment of area [m⁴]
            L: Column length [m]
            end_condition: End condition type

        Returns:
            Critical load [N]
        """
        K = EulerColumnBuckling.K_FACTORS[end_condition]
        Le = L / np.sqrt(K)  # Effective length

        P_cr = (np.pi**2 * E * I) / Le**2

        return P_cr

    @staticmethod
    def critical_stress(E: float, I: float, A: float, L: float,
                       end_condition: EndCondition) -> float:
        """
        Compute Euler critical buckling stress.

        σ_cr = P_cr / A = (π²·E) / (L/r)²

        where r = √(I/A) is radius of gyration

        Args:
            E: Young's modulus [Pa]
            I: Second moment of area [m⁴]
            A: Cross-sectional area [m²]
            L: Column length [m]
            end_condition: End condition type

        Returns:
            Critical stress [Pa]
        """
        P_cr = EulerColumnBuckling.critical_load(E, I, L, end_condition)
        sigma_cr = P_cr / A

        return sigma_cr

    @staticmethod
    def slenderness_ratio(L: float, r: float, end_condition: EndCondition) -> float:
        """
        Compute slenderness ratio.

        λ = L_e / r

        where L_e = K·L (effective length)
              r = √(I/A) (radius of gyration)

        Args:
            L: Column length [m]
            r: Radius of gyration [m]
            end_condition: End condition type

        Returns:
            Slenderness ratio
        """
        K = EulerColumnBuckling.K_FACTORS[end_condition]
        Le = L / np.sqrt(K)

        lambda_ratio = Le / r

        return lambda_ratio

    @staticmethod
    def is_long_column(E: float, sigma_y: float, L: float, r: float,
                      end_condition: EndCondition) -> bool:
        """
        Check if column is "long" (Euler buckling applies).

        Long column if: (L/r) > π·√(E / σ_y)

        Args:
            E: Young's modulus [Pa]
            sigma_y: Yield strength [Pa]
            L: Length [m]
            r: Radius of gyration [m]
            end_condition: End condition

        Returns:
            True if long column (Euler formula applies)
        """
        lambda_ratio = EulerColumnBuckling.slenderness_ratio(L, r, end_condition)
        lambda_transition = np.pi * np.sqrt(E / sigma_y)

        return lambda_ratio > lambda_transition


class PanelBuckling:
    """
    Panel buckling analysis for thin-walled structures.

    For isotropic panels:
        σ_cr = k·(π²·E / (12(1-ν²)))·(t/b)²

    where k = buckling coefficient (depends on loading and edge conditions)
    """

    @staticmethod
    def critical_stress_compression(E: float, nu: float, t: float, b: float,
                                   a: Optional[float] = None,
                                   edge_condition: str = 'simply_supported') -> float:
        """
        Critical buckling stress for panel in compression.

        σ_cr = k·π²·E / (12(1-ν²))·(t/b)²

        Args:
            E: Young's modulus [Pa]
            nu: Poisson's ratio
            t: Panel thickness [m]
            b: Panel width (unsupported dimension) [m]
            a: Panel length [m] (optional, for aspect ratio)
            edge_condition: 'simply_supported' or 'clamped'

        Returns:
            Critical buckling stress [Pa]
        """
        # Buckling coefficient (depends on boundary conditions and aspect ratio)
        if a is not None:
            aspect_ratio = a / b
        else:
            aspect_ratio = float('inf')  # Very long panel

        k = PanelBuckling._compression_buckling_coefficient(aspect_ratio, edge_condition)

        # Critical stress
        sigma_cr = k * (np.pi**2 * E) / (12 * (1 - nu**2)) * (t / b)**2

        return sigma_cr

    @staticmethod
    def _compression_buckling_coefficient(aspect_ratio: float, edge_condition: str) -> float:
        """
        Buckling coefficient for uniaxial compression.

        For simply supported edges:
            k = (m·b/a + a/(m·b))² where m is number of half-waves

        Args:
            aspect_ratio: a/b (length/width)
            edge_condition: Boundary condition

        Returns:
            Buckling coefficient k
        """
        if edge_condition == 'simply_supported':
            # For long panels (a/b > 1), k → 4.0
            # For square panels (a/b = 1), k ≈ 4.0
            # Minimum k occurs at optimal m

            if aspect_ratio >= 1.0:
                # Long panel: k ≈ 4.0
                k = 4.0
            else:
                # Short panel
                m = 1  # Single half-wave
                k = (m / aspect_ratio + aspect_ratio / m)**2

        elif edge_condition == 'clamped':
            # Clamped edges: higher buckling coefficient
            k = 6.97  # For square plate
            if aspect_ratio > 1.0:
                k = 7.0  # Long panel

        else:
            # Default: simply supported
            k = 4.0

        return k

    @staticmethod
    def critical_stress_shear(G: float, nu: float, t: float, b: float,
                             a: Optional[float] = None) -> float:
        """
        Critical shear buckling stress.

        τ_cr = k_s·π²·E / (12(1-ν²))·(t/b)²

        Args:
            G: Shear modulus [Pa]
            nu: Poisson's ratio
            t: Thickness [m]
            b: Width [m]
            a: Length [m] (optional)

        Returns:
            Critical shear stress [Pa]
        """
        # Shear buckling coefficient (depends on aspect ratio)
        if a is not None:
            aspect_ratio = a / b
        else:
            aspect_ratio = 1.0

        # For simply supported edges
        if aspect_ratio >= 1.0:
            k_s = 5.34 + 4.0 / aspect_ratio**2
        else:
            k_s = 5.34 + 4.0 * aspect_ratio**2

        # Use equivalent modulus for shear
        E_eq = 2 * G * (1 + nu)

        tau_cr = k_s * (np.pi**2 * E_eq) / (12 * (1 - nu**2)) * (t / b)**2

        return tau_cr


class OrthotropicPanelBuckling:
    """
    Buckling analysis for orthotropic panels (composites).

    Uses orthotropic plate theory with D11, D12, D22, D66 stiffness terms.
    """

    @staticmethod
    def critical_stress_compression(D11: float, D12: float, D22: float, D66: float,
                                   b: float, aspect_ratio: float = 1.0) -> float:
        """
        Critical buckling stress for orthotropic panel.

        Simplified formula for uniaxial compression:
            σ_cr = (π²/b²)·√(√(D11·D22))

        Args:
            D11: Bending stiffness in x-direction [N·m]
            D12: Coupling bending stiffness [N·m]
            D22: Bending stiffness in y-direction [N·m]
            D66: Twisting stiffness [N·m]
            b: Panel width [m]
            aspect_ratio: a/b

        Returns:
            Critical stress [Pa]
        """
        # Effective bending stiffness
        D_eff = (D11 * D22**3)**0.25

        # Buckling coefficient (depends on orthotropy ratio)
        k = 2 * (np.sqrt(D11/D22) + 2*D12/(np.sqrt(D11*D22)) + 2*np.sqrt(D66/(np.sqrt(D11*D22))))

        # Critical load per unit width
        N_cr = k * (np.pi**2 / b**2) * D_eff

        # Note: This is load per unit width [N/m]
        # To get stress, need to divide by thickness (not included in D formulation)
        # Return as representative buckling load

        return N_cr


class CripplingAnalysis:
    """
    Local crippling analysis for thin-walled sections.

    Crippling is local buckling of thin elements (flanges, webs).
    """

    @staticmethod
    def crippling_stress_flat_plate(E: float, sigma_y: float, t: float, b: float) -> float:
        """
        Crippling stress for flat plate element.

        Gerard-Becker formula:
            σ_cr = C·E·(t/b)^n

        Args:
            E: Young's modulus [Pa]
            sigma_y: Yield strength [Pa]
            t: Thickness [m]
            b: Width [m]

        Returns:
            Crippling stress [Pa]
        """
        # Gerard-Becker coefficients (approximate)
        C = 0.316
        n = 1.15

        sigma_cr = C * E * (t / b)**n

        # Cannot exceed yield
        sigma_cr = min(sigma_cr, sigma_y)

        return sigma_cr


class BucklingInteraction:
    """
    Interaction equations for combined loading (compression + shear, etc.)
    """

    @staticmethod
    def compression_shear_interaction(sigma_applied: float, sigma_cr_compression: float,
                                     tau_applied: float, tau_cr_shear: float) -> float:
        """
        Interaction equation for combined compression and shear.

        (σ/σ_cr)² + (τ/τ_cr)² ≤ 1.0

        Args:
            sigma_applied: Applied compressive stress [Pa]
            sigma_cr_compression: Critical compression buckling stress [Pa]
            tau_applied: Applied shear stress [Pa]
            tau_cr_shear: Critical shear buckling stress [Pa]

        Returns:
            Interaction ratio (< 1.0 is safe)
        """
        R_sigma = abs(sigma_applied) / sigma_cr_compression
        R_tau = abs(tau_applied) / tau_cr_shear

        interaction_ratio = R_sigma**2 + R_tau**2

        return interaction_ratio

    @staticmethod
    def is_safe(interaction_ratio: float) -> bool:
        """Check if interaction ratio is safe"""
        return interaction_ratio <= 1.0

    @staticmethod
    def margin_of_safety(interaction_ratio: float) -> float:
        """Compute margin of safety from interaction ratio"""
        if interaction_ratio <= 0:
            return float('inf')

        ms = (1.0 / interaction_ratio) - 1.0

        return ms
