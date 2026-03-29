"""
Radiative heat transfer for spacecraft thermal analysis.

View factors and radiative exchange between surfaces.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum


# Stefan-Boltzmann constant
SIGMA = 5.670374419e-8  # W/(m²·K⁴)


class BlackBodyRadiation:
    """Black body radiation calculations"""

    @staticmethod
    def emissive_power(temperature: float) -> float:
        """
        Calculate black body emissive power.

        E = σ·T⁴  [W/m²]

        Args:
            temperature: Temperature [K]

        Returns:
            Emissive power [W/m²]
        """
        return SIGMA * temperature**4

    @staticmethod
    def heat_flux_between_surfaces(T1: float, T2: float, emissivity1: float = 1.0,
                                   emissivity2: float = 1.0) -> float:
        """
        Heat flux between two surfaces.

        For gray bodies:
            q = σ·(T₁⁴ - T₂⁴) / (1/ε₁ + 1/ε₂ - 1)

        Simplified for parallel infinite plates.

        Args:
            T1: Temperature of surface 1 [K]
            T2: Temperature of surface 2 [K]
            emissivity1: Emissivity of surface 1
            emissivity2: Emissivity of surface 2

        Returns:
            Heat flux [W/m²]
        """
        # Effective emissivity
        if emissivity1 >= 0.999 and emissivity2 >= 0.999:
            eps_eff = 1.0
        else:
            eps_eff = 1.0 / (1.0/emissivity1 + 1.0/emissivity2 - 1.0)

        q = eps_eff * SIGMA * (T1**4 - T2**4)

        return q

    @staticmethod
    def radiation_to_space(temperature: float, emissivity: float,
                          T_space: float = 4.0) -> float:
        """
        Radiation from surface to deep space.

        q = ε·σ·(T⁴ - T_space⁴)

        Args:
            temperature: Surface temperature [K]
            emissivity: Surface emissivity
            T_space: Space temperature [K] (default 4 K)

        Returns:
            Heat flux [W/m²]
        """
        q = emissivity * SIGMA * (temperature**4 - T_space**4)

        return q


class ViewFactorCalculator:
    """
    Calculate view factors for simple geometries.

    View factor F_12: fraction of radiation leaving surface 1
    that directly reaches surface 2.

    Reciprocity: A₁·F_12 = A₂·F_21
    Summation: Σ F_1j = 1 (for enclosure)
    """

    @staticmethod
    def parallel_plates(width: float, length: float, separation: float) -> float:
        """
        View factor between two identical parallel rectangular plates.

        Args:
            width: Plate width [m]
            length: Plate length [m]
            separation: Separation distance [m]

        Returns:
            View factor F_12
        """
        # Dimensionless ratios
        X = width / separation
        Y = length / separation

        # Hottel's crossed-string formula for rectangles
        # F_12 = (2/(π·X·Y)) · [ln(√((1+X²)·(1+Y²)/(1+X²+Y²)))
        #                       + X·√(1+Y²)·atan(X/√(1+Y²))
        #                       + Y·√(1+X²)·atan(Y/√(1+X²))
        #                       - X·atan(X) - Y·atan(Y)]

        # Simplified for far-field (separation >> size): F ≈ A/(π·d²)
        if separation > 3 * max(width, length):
            A = width * length
            F = A / (np.pi * separation**2)
        else:
            # Numerical approximation for near-field
            # For identical parallel plates, use correlation:
            term1 = np.log(np.sqrt((1 + X**2) * (1 + Y**2) / (1 + X**2 + Y**2)))
            term2 = X * np.sqrt(1 + Y**2) * np.arctan(X / np.sqrt(1 + Y**2))
            term3 = Y * np.sqrt(1 + X**2) * np.arctan(Y / np.sqrt(1 + X**2))
            term4 = X * np.arctan(X)
            term5 = Y * np.arctan(Y)

            F = (2.0 / (np.pi * X * Y)) * (term1 + term2 + term3 - term4 - term5)

        return F

    @staticmethod
    def perpendicular_plates(width1: float, width2: float, common_edge: float) -> float:
        """
        View factor between two perpendicular plates with common edge.

        Args:
            width1: Width of plate 1 [m]
            width2: Width of plate 2 [m]
            common_edge: Length of common edge [m]

        Returns:
            View factor F_12
        """
        # Dimensionless
        W = width2 / width1

        # Hottel formula for perpendicular rectangles
        F = (1.0 / (np.pi * W)) * (W * np.arctan(1.0/W) + np.arctan(W) - np.sqrt(1 + W**2)
                                    + 0.25 * np.log((1 + W**2) * W**2 / ((1 + W**2 + W**2) * W**2)))

        # Multiply by common edge factor (simplified)
        return F

    @staticmethod
    def sphere_to_sphere(R1: float, R2: float, distance: float) -> float:
        """
        View factor from sphere 1 to sphere 2.

        Args:
            R1: Radius of sphere 1 [m]
            R2: Radius of sphere 2 [m]
            distance: Center-to-center distance [m]

        Returns:
            View factor F_12
        """
        if distance <= R1 + R2:
            # Spheres overlap or touch
            return 0.0

        # View factor from sphere to sphere
        # F_12 = R₂² / d²  (for R2 << d)

        F = R2**2 / distance**2

        return min(F, 1.0)

    @staticmethod
    def disk_to_parallel_disk(R1: float, R2: float, separation: float) -> float:
        """
        View factor between two parallel coaxial disks.

        Args:
            R1: Radius of disk 1 [m]
            R2: Radius of disk 2 [m]
            separation: Separation distance [m]

        Returns:
            View factor F_12
        """
        # Dimensionless
        R_ratio = R2 / R1
        S = separation / R1

        # Hottel formula
        X = 1 + (1 + R_ratio**2) / S**2

        F = 0.5 * (X - np.sqrt(X**2 - 4 * R_ratio**2))

        return F


@dataclass
class Surface:
    """Radiating surface properties"""
    area: float  # m²
    emissivity: float  # dimensionless
    absorptivity: float  # dimensionless (for solar)
    temperature: float  # K
    name: str = "surface"


class RadiativeExchange:
    """
    Multi-surface radiative exchange solver.

    Uses radiosity method for enclosures.

    For n surfaces:
        q_i = Σ A_i·F_ij·(J_i - J_j)

    where J = radiosity [W/m²]

    For gray diffuse surfaces:
        J_i = ε_i·σ·T_i⁴ + (1-ε_i)·H_i

    where H = irradiation [W/m²]
    """

    def __init__(self, surfaces: List[Surface], view_factors: np.ndarray):
        """
        Initialize radiative exchange solver.

        Args:
            surfaces: List of Surface objects
            view_factors: Matrix of view factors [n×n], F_ij
        """
        self.surfaces = surfaces
        self.n = len(surfaces)
        self.F = view_factors

        # Validate view factors
        self._validate_view_factors()

    def _validate_view_factors(self):
        """Check view factor reciprocity and summation"""
        for i in range(self.n):
            for j in range(self.n):
                # Reciprocity: A_i·F_ij = A_j·F_ji
                if i != j:
                    lhs = self.surfaces[i].area * self.F[i, j]
                    rhs = self.surfaces[j].area * self.F[j, i]
                    if abs(lhs - rhs) / max(lhs, rhs, 1e-10) > 0.01:
                        print(f"Warning: View factor reciprocity violated for surfaces {i},{j}")

    def solve_radiosity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve for radiosity and heat flux at each surface.

        Returns:
            (radiosity [W/m²], net_heat_flux [W])
        """
        # Build radiosity matrix
        # (I - (1-ε)·F)·J = ε·σ·T⁴

        I = np.eye(self.n)
        eps = np.array([s.emissivity for s in self.surfaces])
        A = np.array([s.area for s in self.surfaces])
        T = np.array([s.temperature for s in self.surfaces])

        # Matrix: [I - (1-ε)·F]
        M = I - np.outer(1 - eps, np.ones(self.n)) * self.F

        # RHS: ε·σ·T⁴
        E = eps * SIGMA * T**4

        # Solve for radiosity
        J = np.linalg.solve(M, E)

        # Net heat flux for each surface
        # q_i = A_i·(E_i - ε_i·J_i)/(1-ε_i)  where E_i = ε_i·σT_i⁴
        # Sign convention: negative = heat out (loss), positive = heat in (gain)
        q_net = np.zeros(self.n)
        for i in range(self.n):
            q_net[i] = -A[i] * (E[i] - eps[i] * J[i]) / (1 - eps[i] + 1e-10)

        return J, q_net

    def heat_flux_between(self, i: int, j: int) -> float:
        """
        Calculate heat flux from surface i to surface j.

        Args:
            i: Surface index i
            j: Surface index j

        Returns:
            Heat flux from i to j [W]
        """
        J, _ = self.solve_radiosity()

        # q_i→j = A_i·F_ij·(J_i - J_j)
        q = self.surfaces[i].area * self.F[i, j] * (J[i] - J[j])

        return q
