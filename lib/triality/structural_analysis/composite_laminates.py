"""
Composite laminate analysis using Classical Lamination Theory (CLT).

Includes:
- Ply-level stress-strain (orthotropic)
- Laminate stiffness matrices [A], [B], [D]
- Stress/strain through thickness
- Failure criteria (Tsai-Wu, Tsai-Hill, max stress)
- First-ply failure analysis
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FailureCriterion(Enum):
    """Composite failure criterion"""
    MAX_STRESS = 'max_stress'
    MAX_STRAIN = 'max_strain'
    TSAI_HILL = 'tsai_hill'
    TSAI_WU = 'tsai_wu'


@dataclass
class OrthotropicPly:
    """
    Single orthotropic ply properties.

    Principal material axes: 1 = fiber direction, 2 = transverse
    """
    # Elastic properties
    E1: float  # Longitudinal modulus [Pa]
    E2: float  # Transverse modulus [Pa]
    G12: float  # In-plane shear modulus [Pa]
    nu12: float  # Major Poisson's ratio

    # Strengths (ultimate)
    X_t: float  # Longitudinal tensile strength [Pa]
    X_c: float  # Longitudinal compressive strength [Pa]
    Y_t: float  # Transverse tensile strength [Pa]
    Y_c: float  # Transverse compressive strength [Pa]
    S: float  # In-plane shear strength [Pa]

    # Ply thickness
    thickness: float  # [m]

    def nu21(self) -> float:
        """Minor Poisson's ratio from reciprocal relation"""
        return self.nu12 * self.E2 / self.E1

    def Q_matrix(self) -> np.ndarray:
        """
        Reduced stiffness matrix [Q] in principal coordinates.

        [Q] relates stress to strain in material coordinates:
            {σ} = [Q]{ε}

        Returns:
            3×3 reduced stiffness matrix
        """
        nu21 = self.nu21()
        denom = 1 - self.nu12 * nu21

        Q11 = self.E1 / denom
        Q12 = self.nu12 * self.E2 / denom
        Q22 = self.E2 / denom
        Q66 = self.G12

        Q = np.array([
            [Q11, Q12, 0],
            [Q12, Q22, 0],
            [0, 0, Q66]
        ])

        return Q

    def Q_bar_matrix(self, theta: float) -> np.ndarray:
        """
        Transformed reduced stiffness matrix [Q̄] at angle θ.

        Transforms from material coordinates to laminate coordinates.

        Args:
            theta: Ply angle [radians] (0 = fibers aligned with x-axis)

        Returns:
            3×3 transformed stiffness matrix
        """
        Q = self.Q_matrix()

        c = np.cos(theta)
        s = np.sin(theta)

        # Transformation matrix
        T = np.array([
            [c**2, s**2, 2*s*c],
            [s**2, c**2, -2*s*c],
            [-s*c, s*c, c**2 - s**2]
        ])

        # Reuter matrix (accounts for engineering shear strain)
        R = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 2]
        ])

        # Q̄ = T⁻¹ Q T R⁻¹
        Q_bar = np.linalg.inv(T) @ Q @ T @ np.linalg.inv(R)

        return Q_bar


@dataclass
class LaminatePly:
    """Single ply in a laminate"""
    material: OrthotropicPly
    angle: float  # Ply orientation [degrees]
    z_bottom: float  # Bottom z-coordinate [m]
    z_top: float  # Top z-coordinate [m]

    @property
    def z_mid(self) -> float:
        """Mid-plane z-coordinate"""
        return (self.z_bottom + self.z_top) / 2

    @property
    def thickness(self) -> float:
        """Ply thickness"""
        return self.z_top - self.z_bottom


class Laminate:
    """
    Composite laminate with Classical Lamination Theory (CLT).

    ABD matrices relate forces/moments to mid-plane strains/curvatures:
        {N, M} = [A, B; B, D]{ε⁰, κ}
    """

    def __init__(self, plies: List[LaminatePly]):
        """
        Initialize laminate.

        Args:
            plies: List of plies (ordered from bottom to top)
        """
        self.plies = plies
        self.n_plies = len(plies)

        # Compute ABD matrices
        self.A, self.B, self.D = self._compute_ABD()

    def _compute_ABD(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ABD stiffness matrices.

        [A] = Σ [Q̄]_k · (z_k - z_{k-1})        (extensional stiffness)
        [B] = ½ Σ [Q̄]_k · (z_k² - z_{k-1}²)   (coupling stiffness)
        [D] = ⅓ Σ [Q̄]_k · (z_k³ - z_{k-1}³)   (bending stiffness)

        Returns:
            (A, B, D) matrices (each 3×3)
        """
        A = np.zeros((3, 3))
        B = np.zeros((3, 3))
        D = np.zeros((3, 3))

        for ply in self.plies:
            theta_rad = np.deg2rad(ply.angle)
            Q_bar = ply.material.Q_bar_matrix(theta_rad)

            z_k = ply.z_top
            z_k1 = ply.z_bottom

            # Extensional stiffness
            A += Q_bar * (z_k - z_k1)

            # Coupling stiffness
            B += 0.5 * Q_bar * (z_k**2 - z_k1**2)

            # Bending stiffness
            D += (1/3) * Q_bar * (z_k**3 - z_k1**3)

        return A, B, D

    def is_symmetric(self, tol: float = 1e-6) -> bool:
        """
        Check if laminate is symmetric (B matrix should be near zero).

        Args:
            tol: Tolerance for zero check

        Returns:
            True if symmetric
        """
        return np.all(np.abs(self.B) < tol)

    def compute_strains_curvatures(self, N: np.ndarray, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute mid-plane strains and curvatures from forces and moments.

        {ε⁰, κ} = [ABD]⁻¹ {N, M}

        Args:
            N: Resultant forces per unit width [N/m] (3×1: Nx, Ny, Nxy)
            M: Resultant moments per unit width [N·m/m] (3×1: Mx, My, Mxy)

        Returns:
            (epsilon_0, kappa) - mid-plane strains and curvatures
        """
        # Build ABD matrix
        ABD = np.block([
            [self.A, self.B],
            [self.B, self.D]
        ])

        # Force/moment vector
        NM = np.concatenate([N, M])

        # Solve for strains/curvatures
        epsilon_kappa = np.linalg.solve(ABD, NM)

        epsilon_0 = epsilon_kappa[:3]
        kappa = epsilon_kappa[3:]

        return epsilon_0, kappa

    def compute_ply_stresses(self, epsilon_0: np.ndarray, kappa: np.ndarray,
                            ply_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute stresses in a specific ply.

        ε(z) = ε⁰ + z·κ
        σ = [Q̄]·ε

        Args:
            epsilon_0: Mid-plane strains
            kappa: Curvatures
            ply_index: Ply index (0-based)

        Returns:
            (sigma_bottom, sigma_top) - stresses at ply bottom and top
        """
        ply = self.plies[ply_index]

        # Strains at bottom and top of ply
        epsilon_bottom = epsilon_0 + ply.z_bottom * kappa
        epsilon_top = epsilon_0 + ply.z_top * kappa

        # Transformed stiffness matrix
        theta_rad = np.deg2rad(ply.angle)
        Q_bar = ply.material.Q_bar_matrix(theta_rad)

        # Stresses
        sigma_bottom = Q_bar @ epsilon_bottom
        sigma_top = Q_bar @ epsilon_top

        return sigma_bottom, sigma_top

    def first_ply_failure(self, N: np.ndarray, M: np.ndarray,
                         criterion: FailureCriterion = FailureCriterion.TSAI_HILL) -> Tuple[int, float]:
        """
        Determine first-ply failure load and location.

        Args:
            N: Applied resultant forces [N/m]
            M: Applied resultant moments [N·m/m]
            criterion: Failure criterion to use

        Returns:
            (ply_index, failure_load_multiplier)
        """
        # Compute strains/curvatures
        epsilon_0, kappa = self.compute_strains_curvatures(N, M)

        min_failure_multiplier = float('inf')
        critical_ply = -1

        for i, ply in enumerate(self.plies):
            # Get stresses
            sigma_bottom, sigma_top = self.compute_ply_stresses(epsilon_0, kappa, i)

            # Check failure at both bottom and top of ply
            for sigma in [sigma_bottom, sigma_top]:
                # Transform to material coordinates
                theta_rad = np.deg2rad(ply.angle)
                sigma_material = self._transform_stress_to_material(sigma, theta_rad)

                # Apply failure criterion
                if criterion == FailureCriterion.TSAI_HILL:
                    failure_index = self._tsai_hill_criterion(sigma_material, ply.material)
                elif criterion == FailureCriterion.MAX_STRESS:
                    failure_index = self._max_stress_criterion(sigma_material, ply.material)
                else:
                    failure_index = 1.0

                # Failure load multiplier
                if failure_index > 0:
                    load_multiplier = 1.0 / np.sqrt(failure_index)

                    if load_multiplier < min_failure_multiplier:
                        min_failure_multiplier = load_multiplier
                        critical_ply = i

        return critical_ply, min_failure_multiplier

    def _transform_stress_to_material(self, sigma_xy: np.ndarray, theta: float) -> np.ndarray:
        """Transform stress from laminate to material coordinates"""
        c = np.cos(theta)
        s = np.sin(theta)

        sigma_x, sigma_y, tau_xy = sigma_xy

        sigma_1 = c**2*sigma_x + s**2*sigma_y + 2*s*c*tau_xy
        sigma_2 = s**2*sigma_x + c**2*sigma_y - 2*s*c*tau_xy
        tau_12 = -s*c*sigma_x + s*c*sigma_y + (c**2 - s**2)*tau_xy

        return np.array([sigma_1, sigma_2, tau_12])

    def _tsai_hill_criterion(self, sigma: np.ndarray, material: OrthotropicPly) -> float:
        """
        Tsai-Hill failure criterion.

        (σ₁/X)² - (σ₁·σ₂/X²) + (σ₂/Y)² + (τ₁₂/S)² ≤ 1

        Args:
            sigma: Stress in material coordinates [sigma_1, sigma_2, tau_12]
            material: Ply material

        Returns:
            Failure index (> 1 means failure)
        """
        sigma_1, sigma_2, tau_12 = sigma

        # Select appropriate strength based on sign
        X = material.X_t if sigma_1 >= 0 else material.X_c
        Y = material.Y_t if sigma_2 >= 0 else material.Y_c
        S = material.S

        failure_index = (
            (sigma_1 / X)**2
            - (sigma_1 * sigma_2 / X**2)
            + (sigma_2 / Y)**2
            + (tau_12 / S)**2
        )

        return failure_index

    def _max_stress_criterion(self, sigma: np.ndarray, material: OrthotropicPly) -> float:
        """
        Maximum stress failure criterion.

        max(σ₁/X, σ₂/Y, τ₁₂/S) ≤ 1

        Args:
            sigma: Stress in material coordinates
            material: Ply material

        Returns:
            Failure index
        """
        sigma_1, sigma_2, tau_12 = sigma

        X = material.X_t if sigma_1 >= 0 else material.X_c
        Y = material.Y_t if sigma_2 >= 0 else material.Y_c
        S = material.S

        failure_index = max(
            abs(sigma_1) / X,
            abs(sigma_2) / Y,
            abs(tau_12) / S
        )

        return failure_index


# Pre-defined composite materials
COMPOSITE_MATERIALS = {
    'IM7-8552': OrthotropicPly(
        E1=171e9,  # Pa
        E2=9.08e9,
        G12=5.29e9,
        nu12=0.32,
        X_t=2326e6,  # Pa
        X_c=1200e6,
        Y_t=62.3e6,
        Y_c=199.8e6,
        S=92.3e6,
        thickness=0.000125  # 125 microns
    ),
    'T300-5208': OrthotropicPly(
        E1=132e9,
        E2=10.8e9,
        G12=5.65e9,
        nu12=0.24,
        X_t=1900e6,
        X_c=1100e6,
        Y_t=48e6,
        Y_c=200e6,
        S=80e6,
        thickness=0.000125
    )
}
