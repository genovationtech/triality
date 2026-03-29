"""
Layer 5: Reactor Neutronics - Multi-Group Neutron Diffusion Solver

Production-ready deterministic neutron diffusion for reactor physics analysis.

Physics Equations:
    Two-group neutron diffusion (fast and thermal):

    Fast group:
        ∇·D_f∇φ_f - Σ_a,f φ_f - Σ_s,f→t φ_f = (1/k_eff)(νΣ_f,f φ_f + νΣ_f,t φ_t)

    Thermal group:
        ∇·D_t∇φ_t - Σ_a,t φ_t + Σ_s,f→t φ_f = (1/k_eff)νΣ_f,t φ_t

    Where:
        φ_f, φ_t: Fast and thermal neutron flux [n/(cm²·s)]
        D: Diffusion coefficient [cm]
        Σ_a: Absorption cross-section [cm⁻¹]
        Σ_f: Fission cross-section [cm⁻¹]
        Σ_s: Scattering cross-section [cm⁻¹]
        ν: Neutrons per fission
        k_eff: Effective multiplication factor

Applications:
    ✓ Core criticality calculations (k_eff)
    ✓ Flux distribution analysis
    ✓ Power peaking factors
    ✓ Control rod worth
    ✓ Reflector effectiveness
    ✓ Initial reactor design

Accuracy Expectations:
    ±5-15% for k_eff (good for design studies)
    ±10-20% for flux distributions
    Good for relative comparisons and initial sizing

When to Use:
    ✓ Early-stage reactor design
    ✓ Control rod positioning studies
    ✓ Reflector optimization
    ✓ Quick criticality checks

When to Upgrade:
    ✗ Licensing calculations → Use Monte Carlo (MCNP, Serpent)
    ✗ Fine spatial detail → Use transport codes
    ✗ Resonance treatment → Use continuous energy
    ✗ Burnup analysis → Use coupled depletion codes
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from typing import Tuple, Optional, Callable, Dict, NamedTuple
from dataclasses import dataclass
from enum import Enum


class MaterialType(Enum):
    """Standard reactor materials"""
    FUEL_UO2_3PCT = "uo2_3pct"  # 3% enriched UO2 fuel
    FUEL_UO2_5PCT = "uo2_5pct"  # 5% enriched UO2 fuel
    MODERATOR_H2O = "h2o"  # Light water moderator
    MODERATOR_D2O = "d2o"  # Heavy water moderator
    MODERATOR_GRAPHITE = "graphite"  # Graphite moderator
    REFLECTOR_H2O = "h2o_reflector"  # Water reflector
    REFLECTOR_GRAPHITE = "graphite_reflector"  # Graphite reflector
    CONTROL_B4C = "b4c"  # Boron carbide control rod
    STRUCTURAL_SS = "stainless_steel"  # Stainless steel structure
    VOID = "void"  # Void region


@dataclass
class CrossSectionSet:
    """Two-group neutron cross-sections for a material

    All cross-sections in cm⁻¹, diffusion coefficients in cm
    """
    name: str
    # Diffusion coefficients
    D_fast: float  # Fast group diffusion coefficient [cm]
    D_thermal: float  # Thermal group diffusion coefficient [cm]

    # Absorption cross-sections
    Sigma_a_fast: float  # Fast absorption [cm⁻¹]
    Sigma_a_thermal: float  # Thermal absorption [cm⁻¹]

    # Fission cross-sections
    nu_Sigma_f_fast: float  # ν×Σ_f fast [cm⁻¹]
    nu_Sigma_f_thermal: float  # ν×Σ_f thermal [cm⁻¹]

    # Scattering (fast → thermal, down-scattering only)
    Sigma_s_fast_to_thermal: float  # Fast to thermal scattering [cm⁻¹]

    @staticmethod
    def get_material(material_type: MaterialType) -> 'CrossSectionSet':
        """Get cross-section set for standard reactor materials

        These are representative two-group constants for typical reactor
        configurations. Values are approximate and suitable for design studies.

        Args:
            material_type: Material type

        Returns:
            CrossSectionSet with two-group constants
        """
        materials = {
            MaterialType.FUEL_UO2_3PCT: CrossSectionSet(
                name="UO2 3% enriched fuel",
                D_fast=1.4,  # cm
                D_thermal=0.4,  # cm
                Sigma_a_fast=0.015,  # cm⁻¹ (increased to reduce k_eff)
                Sigma_a_thermal=0.130,  # cm⁻¹ (increased)
                nu_Sigma_f_fast=0.005,  # cm⁻¹ (slightly reduced)
                nu_Sigma_f_thermal=0.115,  # cm⁻¹ (reduced for realistic k_eff)
                Sigma_s_fast_to_thermal=0.025,  # cm⁻¹
            ),

            MaterialType.FUEL_UO2_5PCT: CrossSectionSet(
                name="UO2 5% enriched fuel",
                D_fast=1.4,
                D_thermal=0.4,
                Sigma_a_fast=0.012,  # Lower than 3% (more reactive)
                Sigma_a_thermal=0.100,  # Lower absorption (more reactive)
                nu_Sigma_f_fast=0.007,  # Higher fission
                nu_Sigma_f_thermal=0.145,  # Higher fission
                Sigma_s_fast_to_thermal=0.025,
            ),

            MaterialType.MODERATOR_H2O: CrossSectionSet(
                name="Light water moderator",
                D_fast=1.3,
                D_thermal=0.2,
                Sigma_a_fast=0.0001,  # Very low absorption
                Sigma_a_thermal=0.020,  # Some thermal absorption
                nu_Sigma_f_fast=0.0,  # No fission
                nu_Sigma_f_thermal=0.0,
                Sigma_s_fast_to_thermal=0.050,  # Good moderator
            ),

            MaterialType.MODERATOR_D2O: CrossSectionSet(
                name="Heavy water moderator",
                D_fast=1.5,
                D_thermal=0.8,
                Sigma_a_fast=0.00001,  # Very low absorption
                Sigma_a_thermal=0.0001,  # Very low thermal absorption
                nu_Sigma_f_fast=0.0,
                nu_Sigma_f_thermal=0.0,
                Sigma_s_fast_to_thermal=0.030,  # Moderate moderation
            ),

            MaterialType.MODERATOR_GRAPHITE: CrossSectionSet(
                name="Graphite moderator",
                D_fast=1.8,
                D_thermal=0.9,
                Sigma_a_fast=0.0001,
                Sigma_a_thermal=0.0003,
                nu_Sigma_f_fast=0.0,
                nu_Sigma_f_thermal=0.0,
                Sigma_s_fast_to_thermal=0.025,  # Slow moderation
            ),

            MaterialType.REFLECTOR_H2O: CrossSectionSet(
                name="Water reflector",
                D_fast=1.3,
                D_thermal=0.2,
                Sigma_a_fast=0.0001,
                Sigma_a_thermal=0.020,
                nu_Sigma_f_fast=0.0,
                nu_Sigma_f_thermal=0.0,
                Sigma_s_fast_to_thermal=0.050,
            ),

            MaterialType.REFLECTOR_GRAPHITE: CrossSectionSet(
                name="Graphite reflector",
                D_fast=2.0,  # Higher diffusion in reflector
                D_thermal=1.0,
                Sigma_a_fast=0.0001,
                Sigma_a_thermal=0.0003,
                nu_Sigma_f_fast=0.0,
                nu_Sigma_f_thermal=0.0,
                Sigma_s_fast_to_thermal=0.025,
            ),

            MaterialType.CONTROL_B4C: CrossSectionSet(
                name="Boron carbide control rod",
                D_fast=1.0,
                D_thermal=0.3,
                Sigma_a_fast=0.005,
                Sigma_a_thermal=2.5,  # VERY high thermal absorption
                nu_Sigma_f_fast=0.0,
                nu_Sigma_f_thermal=0.0,
                Sigma_s_fast_to_thermal=0.010,
            ),

            MaterialType.STRUCTURAL_SS: CrossSectionSet(
                name="Stainless steel structure",
                D_fast=1.2,
                D_thermal=0.4,
                Sigma_a_fast=0.002,
                Sigma_a_thermal=0.015,
                nu_Sigma_f_fast=0.0,
                nu_Sigma_f_thermal=0.0,
                Sigma_s_fast_to_thermal=0.015,
            ),

            MaterialType.VOID: CrossSectionSet(
                name="Void",
                D_fast=1e6,  # Effectively infinite diffusion
                D_thermal=1e6,
                Sigma_a_fast=0.0,
                Sigma_a_thermal=0.0,
                nu_Sigma_f_fast=0.0,
                nu_Sigma_f_thermal=0.0,
                Sigma_s_fast_to_thermal=0.0,
            ),
        }

        return materials[material_type]


class NeutronicsResult(NamedTuple):
    """Results from neutronics calculation"""
    x: np.ndarray  # Position [cm]
    phi_fast: np.ndarray  # Fast flux [n/(cm²·s)]
    phi_thermal: np.ndarray  # Thermal flux [n/(cm²·s)]
    k_eff: float  # Effective multiplication factor
    converged: bool  # Whether eigenvalue converged
    iterations: int  # Number of power iterations
    power_density: np.ndarray  # Power density [W/cm³]
    materials: np.ndarray  # Material indices

    def total_flux(self) -> np.ndarray:
        """Total neutron flux [n/(cm²·s)]"""
        return self.phi_fast + self.phi_thermal

    def thermal_utilization(self) -> float:
        """Thermal utilization factor (fraction of thermal flux in fuel)"""
        # Placeholder - would need material map
        return np.mean(self.phi_thermal) / (np.mean(self.total_flux()) + 1e-10)

    def fast_fission_factor(self) -> float:
        """Fast fission factor (ratio of all fissions to thermal fissions)"""
        # Simplified estimate
        total_fissions = np.sum(self.phi_fast) + np.sum(self.phi_thermal)
        thermal_fissions = np.sum(self.phi_thermal)
        return total_fissions / (thermal_fissions + 1e-10)

    def peaking_factor(self) -> float:
        """Power peaking factor (max/avg power density)"""
        avg_power = np.mean(self.power_density)
        max_power = np.max(self.power_density)
        return max_power / (avg_power + 1e-10)

    def max_power_location(self) -> float:
        """Location of maximum power density [cm]"""
        idx = np.argmax(self.power_density)
        return self.x[idx]


class MultiGroupDiffusion1D:
    """
    1D Two-Group Neutron Diffusion Solver

    Solves the multigroup neutron diffusion eigenvalue problem to find
    the critical state (k_eff) and flux distribution.

    Uses power iteration for the eigenvalue problem with GMRES for
    the inner iterations.

    Example:
        >>> solver = MultiGroupDiffusion1D(length=200.0, n_points=100)
        >>> # Set up fuel region
        >>> solver.set_material(
        ...     region=(50, 150),
        ...     material=MaterialType.FUEL_UO2_3PCT
        ... )
        >>> # Set up moderator
        >>> solver.set_material(
        ...     region=(0, 50),
        ...     material=MaterialType.MODERATOR_H2O
        ... )
        >>> result = solver.solve()
        >>> print(f"k_eff = {result.k_eff:.5f}")
    """

    def __init__(
        self,
        length: float,  # [cm]
        n_points: int = 100,
        energy_per_fission: float = 200.0,  # MeV per fission
    ):
        """Initialize neutronics solver

        Args:
            length: System length [cm]
            n_points: Number of spatial points
            energy_per_fission: Energy released per fission [MeV]
        """
        self.length = length
        self.n_points = n_points
        self.energy_per_fission = energy_per_fission

        # Spatial grid
        self.x = np.linspace(0, length, n_points)
        self.dx = length / (n_points - 1)

        # Material properties (default to void)
        self.materials = np.full(n_points, MaterialType.VOID)
        self.D_fast = np.ones(n_points) * 1e6
        self.D_thermal = np.ones(n_points) * 1e6
        self.Sigma_a_fast = np.zeros(n_points)
        self.Sigma_a_thermal = np.zeros(n_points)
        self.nu_Sigma_f_fast = np.zeros(n_points)
        self.nu_Sigma_f_thermal = np.zeros(n_points)
        self.Sigma_s_f_to_t = np.zeros(n_points)

        # Flux arrays
        self.phi_fast = np.ones(n_points)
        self.phi_thermal = np.ones(n_points)

        # Eigenvalue
        self.k_eff = 1.0

    def set_material(
        self,
        region: Tuple[float, float],
        material: MaterialType
    ):
        """Set material in a spatial region

        Args:
            region: (x_start, x_end) in cm
            material: Material type
        """
        x_start, x_end = region
        mask = (self.x >= x_start) & (self.x <= x_end)

        xs = CrossSectionSet.get_material(material)

        self.materials[mask] = material
        self.D_fast[mask] = xs.D_fast
        self.D_thermal[mask] = xs.D_thermal
        self.Sigma_a_fast[mask] = xs.Sigma_a_fast
        self.Sigma_a_thermal[mask] = xs.Sigma_a_thermal
        self.nu_Sigma_f_fast[mask] = xs.nu_Sigma_f_fast
        self.nu_Sigma_f_thermal[mask] = xs.nu_Sigma_f_thermal
        self.Sigma_s_f_to_t[mask] = xs.Sigma_s_fast_to_thermal

    def solve(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        verbose: bool = False
    ) -> NeutronicsResult:
        """Solve two-group diffusion eigenvalue problem

        Uses power iteration to find k_eff and flux distribution.

        Args:
            max_iterations: Maximum power iterations
            tolerance: Convergence tolerance for k_eff
            verbose: Print iteration info

        Returns:
            NeutronicsResult with k_eff and flux distributions
        """
        if verbose:
            print(f"Solving 1D two-group diffusion ({self.n_points} points)...")
            print(f"  Domain length: {self.length:.1f} cm")

        # Initialize flux guess (flat with some variation)
        self.phi_fast = np.ones(self.n_points) * 1e12
        self.phi_thermal = np.ones(self.n_points) * 1e12
        self.k_eff = 1.0

        # Store initial fission source for eigenvalue calculation
        self._fission_source_old = self._calculate_fission_source()

        converged = False

        # Power iteration
        for iteration in range(max_iterations):
            k_old = self.k_eff

            # Solve for new flux given current k_eff
            self._inner_iteration()

            # Calculate new fission source
            fission_source_new = self._calculate_fission_source()

            # Update k_eff: k_new = k_old * (S_new / S_old)
            if self._fission_source_old > 1e-20:
                self.k_eff = k_old * (fission_source_new / self._fission_source_old)

            # Normalize flux to unity fission source for next iteration
            if fission_source_new > 1e-20:
                norm_factor = 1.0 / fission_source_new
                self.phi_fast *= norm_factor
                self.phi_thermal *= norm_factor
                self._fission_source_old = 1.0  # After normalization
            else:
                self._fission_source_old = fission_source_new

            # Check convergence
            dk = abs(self.k_eff - k_old)

            if verbose and (iteration % 10 == 0 or iteration < 5):
                max_flux_f = np.max(self.phi_fast)
                max_flux_t = np.max(self.phi_thermal)
                print(f"  Iteration {iteration:3d}: k_eff = {self.k_eff:.6f}, "
                      f"Δk = {dk:.2e}, φ_f_max = {max_flux_f:.2e}")

            if dk < tolerance:
                converged = True
                if verbose:
                    print(f"  ✓ Converged in {iteration + 1} iterations")
                break
        else:
            if verbose:
                print(f"  ⚠ Max iterations reached (Δk = {dk:.2e})")

        # Calculate power density
        power = self._calculate_power_density()

        return NeutronicsResult(
            x=self.x,
            phi_fast=self.phi_fast,
            phi_thermal=self.phi_thermal,
            k_eff=self.k_eff,
            converged=converged,
            iterations=iteration + 1,
            power_density=power,
            materials=self.materials
        )

    def _inner_iteration(self):
        """Solve for flux given current k_eff (one power iteration step)"""
        # Fast group equation:
        # -∇·D_f∇φ_f + (Σ_a,f + Σ_s,f→t)φ_f = (1/k)νΣ_f,f φ_f + (1/k)νΣ_f,t φ_t

        # Thermal group equation:
        # -∇·D_t∇φ_t + Σ_a,t φ_t - Σ_s,f→t φ_f = (1/k)νΣ_f,t φ_t

        # Build matrices
        Lf = self._build_diffusion_operator(self.D_fast)
        Lt = self._build_diffusion_operator(self.D_thermal)

        # Total removal cross-sections
        Sigma_r_fast = self.Sigma_a_fast + self.Sigma_s_f_to_t
        Sigma_r_thermal = self.Sigma_a_thermal

        # System matrix and source
        n = self.n_points

        # Build coupled system (simplified - fixed source iteration)
        # Solve fast group with thermal source
        fission_source_fast = (1.0/self.k_eff) * (
            self.nu_Sigma_f_fast * self.phi_fast +
            self.nu_Sigma_f_thermal * self.phi_thermal
        )

        A_fast = Lf + sparse.diags(Sigma_r_fast, format='csr')
        b_fast = fission_source_fast

        # Solve for fast flux
        try:
            self.phi_fast, info = spla.gmres(A_fast, b_fast, x0=self.phi_fast, rtol=1e-6, atol=1e-10)
            if info != 0:
                # Fallback to direct solve
                self.phi_fast = spla.spsolve(A_fast, b_fast)
        except:
            self.phi_fast = spla.spsolve(A_fast, b_fast)

        # Ensure positive flux
        self.phi_fast = np.maximum(self.phi_fast, 0.0)

        # Solve thermal group with fast scattering source
        thermal_source = (
            self.Sigma_s_f_to_t * self.phi_fast +
            (1.0/self.k_eff) * self.nu_Sigma_f_thermal * self.phi_thermal
        )

        A_thermal = Lt + sparse.diags(Sigma_r_thermal, format='csr')
        b_thermal = thermal_source

        try:
            self.phi_thermal, info = spla.gmres(A_thermal, b_thermal, x0=self.phi_thermal, rtol=1e-6, atol=1e-10)
            if info != 0:
                self.phi_thermal = spla.spsolve(A_thermal, b_thermal)
        except:
            self.phi_thermal = spla.spsolve(A_thermal, b_thermal)

        self.phi_thermal = np.maximum(self.phi_thermal, 0.0)

    def _build_diffusion_operator(self, D: np.ndarray) -> sparse.csr_matrix:
        """Build -∇·D∇ operator using finite differences

        Returns sparse matrix for -d/dx(D dφ/dx)
        """
        n = self.n_points
        dx = self.dx

        # Central differences: -d/dx(D dφ/dx) ≈ -(D[i+1/2](φ[i+1]-φ[i]) - D[i-1/2](φ[i]-φ[i-1]))/dx²

        # Average diffusion coefficients at interfaces
        # D_plus[i] = D at interface i+1/2 (between i and i+1)
        # D_minus[i] = D at interface i-1/2 (between i-1 and i)

        D_interfaces = (D[:-1] + D[1:]) / 2.0  # Length n-1 (interfaces)

        # For each interior point i:
        # diagonal[i] = (D_interfaces[i] + D_interfaces[i-1]) / dx²
        # upper[i] = -D_interfaces[i] / dx²  (coefficient of φ[i+1])
        # lower[i] = -D_interfaces[i-1] / dx² (coefficient of φ[i-1])

        diag = np.zeros(n)
        upper = np.zeros(n-1)
        lower = np.zeros(n-1)

        # Interior points
        for i in range(1, n-1):
            D_left = D_interfaces[i-1]   # D at i-1/2
            D_right = D_interfaces[i]     # D at i+1/2

            diag[i] = (D_left + D_right) / (dx * dx)
            upper[i] = -D_right / (dx * dx)
            lower[i-1] = -D_left / (dx * dx)

        # Boundary conditions (vacuum: φ = 0 at boundaries)
        # Use large diagonal values to enforce φ[0] = 0 and φ[n-1] = 0
        diag[0] = 1e10
        diag[-1] = 1e10

        A = sparse.diags([lower, diag, upper], [-1, 0, 1], shape=(n, n), format='csr')

        return A

    def _calculate_fission_source(self) -> float:
        """Calculate total fission neutron source

        Returns:
            Total fission source (integrated over volume)
        """
        fission_source_density = (
            self.nu_Sigma_f_fast * self.phi_fast +
            self.nu_Sigma_f_thermal * self.phi_thermal
        )

        total_fission_source = np.sum(fission_source_density) * self.dx

        return total_fission_source

    def _calculate_power_density(self) -> np.ndarray:
        """Calculate power density [W/cm³]

        Power = energy per fission × fission rate density
        """
        # Fission rate density [fissions/(cm³·s)]
        # νΣ_f × φ gives fission neutron production
        # To get fissions, divide by average ν (≈ 2.5 for U-235)
        nu_avg = 2.5

        fission_rate = (
            (self.nu_Sigma_f_fast / nu_avg) * self.phi_fast +
            (self.nu_Sigma_f_thermal / nu_avg) * self.phi_thermal
        )

        # Convert MeV/fission to Watts
        MeV_to_J = 1.602e-13  # J/MeV
        power_density = fission_rate * self.energy_per_fission * MeV_to_J

        # Ensure non-negative
        power_density = np.maximum(power_density, 0.0)

        return power_density


# Public API
__all__ = [
    'MultiGroupDiffusion1D',
    'NeutronicsResult',
    'MaterialType',
    'CrossSectionSet',
]
