"""
Finite-Difference Time-Domain (FDTD) method for electromagnetic simulation.

Includes:
- Yee cell (staggered grid)
- Maxwell's equations discretization
- Courant stability criterion
- PML (Perfectly Matched Layer) boundary conditions
- Update equations
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class BoundaryCondition(Enum):
    """Boundary condition types"""
    PEC = 'PEC'  # Perfect Electric Conductor
    PMC = 'PMC'  # Perfect Magnetic Conductor
    PML = 'PML'  # Perfectly Matched Layer
    PERIODIC = 'PERIODIC'
    ABC = 'ABC'  # Absorbing Boundary Condition (Mur)


@dataclass
class FDTDParameters:
    """
    FDTD simulation parameters.

    Attributes:
        dx, dy, dz: Spatial step sizes [m]
        dt: Time step [s]
        nx, ny, nz: Number of cells in each direction
        epsilon_r: Relative permittivity
        mu_r: Relative permeability
        sigma: Conductivity [S/m]
    """
    dx: float
    dy: float
    dz: float
    dt: float
    nx: int
    ny: int
    nz: int
    epsilon_r: float = 1.0
    mu_r: float = 1.0
    sigma: float = 0.0


class FDTD:
    """
    Finite-Difference Time-Domain electromagnetic solver.

    Physics Basis:
    --------------
    Maxwell's Equations (Time Domain):
        ∇×E = -∂B/∂t
        ∇×H = ∂D/∂t + J

    Constitutive Relations:
        D = ε·E
        B = μ·H
        J = σ·E

    Yee Cell:
        - E and H components staggered in space and time
        - E(i,j,k,n) and H(i+½,j+½,k+½,n+½)
        - Leap-frog time stepping

    Update Equations (1D, simplified):
        E^(n+1) = E^n + (Δt/ε)·(∂H^(n+½)/∂x - σ·E^n)
        H^(n+½) = H^(n-½) + (Δt/μ)·∂E^n/∂x

    Courant Stability Criterion:
        Δt ≤ 1/(c·√(1/Δx² + 1/Δy² + 1/Δz²))

    where c = 1/√(με) = speed of light in medium

    PML Absorbing Boundary:
        - Artificial anisotropic material
        - Matches impedance at interface
        - Absorbs outgoing waves without reflection
    """

    # Constants
    c = 2.99792458e8    # Speed of light [m/s]
    mu_0 = 4*np.pi*1e-7 # Permeability [H/m]
    epsilon_0 = 8.854187817e-12  # Permittivity [F/m]

    @staticmethod
    def courant_stability_criterion(dx: float, dy: float, dz: float,
                                    epsilon_r: float = 1.0, 
                                    mu_r: float = 1.0) -> float:
        """
        Calculate maximum stable time step.

        Δt_max = S/(c·√(1/Δx² + 1/Δy² + 1/Δz²))

        where S = Courant number (typically 0.5-1.0)
              c = 1/√(με)

        Args:
            dx, dy, dz: Spatial step sizes [m]
            epsilon_r: Relative permittivity
            mu_r: Relative permeability

        Returns:
            Maximum time step [s]
        """
        # Speed of light in medium
        c_medium = FDTD.c / np.sqrt(epsilon_r * mu_r)
        
        # Courant number (conservative: use 0.5 for stability margin)
        S = 0.5
        
        # Stability limit
        dt_max = S / (c_medium * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2))
        
        return dt_max

    @staticmethod
    def courant_number(dt: float, dx: float, dy: float, dz: float,
                      epsilon_r: float = 1.0, mu_r: float = 1.0) -> float:
        """
        Calculate Courant number for given parameters.

        S = c·Δt·√(1/Δx² + 1/Δy² + 1/Δz²)

        Must satisfy S < 1 for stability.

        Args:
            dt: Time step [s]
            dx, dy, dz: Spatial step sizes [m]
            epsilon_r: Relative permittivity
            mu_r: Relative permeability

        Returns:
            Courant number (must be < 1)
        """
        c_medium = FDTD.c / np.sqrt(epsilon_r * mu_r)
        S = c_medium * dt * np.sqrt(1/dx**2 + 1/dy**2 + 1/dz**2)
        
        return S

    @staticmethod
    def update_coefficient_E(epsilon_r: float, sigma: float, dt: float) -> Tuple[float, float]:
        """
        Calculate E-field update coefficients.

        E^(n+1) = C_a·E^n + C_b·∇×H^(n+½)

        where:
            C_a = (1 - σΔt/2ε) / (1 + σΔt/2ε)
            C_b = (Δt/ε) / (1 + σΔt/2ε)

        Args:
            epsilon_r: Relative permittivity
            sigma: Conductivity [S/m]
            dt: Time step [s]

        Returns:
            Tuple (C_a, C_b)
        """
        epsilon = epsilon_r * FDTD.epsilon_0
        
        if sigma == 0:
            # Lossless case
            C_a = 1.0
            C_b = dt / epsilon
        else:
            # Lossy case
            C_a = (1 - sigma*dt/(2*epsilon)) / (1 + sigma*dt/(2*epsilon))
            C_b = (dt/epsilon) / (1 + sigma*dt/(2*epsilon))
        
        return C_a, C_b

    @staticmethod
    def update_coefficient_H(mu_r: float, sigma_m: float, dt: float) -> Tuple[float, float]:
        """
        Calculate H-field update coefficients.

        H^(n+½) = D_a·H^(n-½) + D_b·∇×E^n

        where:
            D_a = (1 - σ_m·Δt/2μ) / (1 + σ_m·Δt/2μ)
            D_b = (Δt/μ) / (1 + σ_m·Δt/2μ)

        Args:
            mu_r: Relative permeability
            sigma_m: Magnetic loss [Ω/m]
            dt: Time step [s]

        Returns:
            Tuple (D_a, D_b)
        """
        mu = mu_r * FDTD.mu_0
        
        if sigma_m == 0:
            # Lossless case
            D_a = 1.0
            D_b = dt / mu
        else:
            # Lossy case
            D_a = (1 - sigma_m*dt/(2*mu)) / (1 + sigma_m*dt/(2*mu))
            D_b = (dt/mu) / (1 + sigma_m*dt/(2*mu))
        
        return D_a, D_b

    @staticmethod
    def dispersion_relation(dx: float, dt: float, wavelength: float) -> float:
        """
        Calculate numerical dispersion error.

        Numerical phase velocity differs from true value:
            v_num/v_true ≈ 1 - (1/6)·(dx/λ)² for Δx << λ

        Args:
            dx: Spatial step [m]
            dt: Time step [s]
            wavelength: Wavelength [m]

        Returns:
            Relative error in phase velocity
        """
        # Rule of thumb: need at least 10-20 cells per wavelength
        cells_per_wavelength = wavelength / dx
        
        if cells_per_wavelength < 10:
            # Significant dispersion
            error = 0.1  # 10% error (rough estimate)
        else:
            # Low dispersion
            error = (1.0/6.0) * (dx/wavelength)**2
        
        return error

    @staticmethod
    def pml_parameters(thickness: int, R_0: float = 1e-6) -> np.ndarray:
        """
        Calculate PML (Perfectly Matched Layer) conductivity profile.

        Graded conductivity for minimal reflections:
            σ(x) = σ_max · ((x - x_0)/d)^m

        where d = PML thickness
              m = grading exponent (typically 3-4)
              σ_max chosen to achieve reflection coefficient R_0

        Args:
            thickness: Number of PML cells
            R_0: Desired reflection coefficient (e.g., 1e-6)

        Returns:
            Array of conductivity values
        """
        m = 3.0  # Grading exponent
        
        # Maximum conductivity
        sigma_max = -(m+1) * np.log(R_0) / (2 * thickness)
        
        # Graded profile
        x = np.arange(thickness)
        sigma = sigma_max * (x / thickness)**m
        
        return sigma

    @staticmethod
    def yee_cell_positions() -> Dict[str, Tuple[float, float, float]]:
        """
        Get Yee cell component positions relative to cell center.

        Returns:
            Dictionary with offsets for E and H components
        """
        # E-field components (on edges)
        E_x_offset = (0.0, 0.5, 0.5)
        E_y_offset = (0.5, 0.0, 0.5)
        E_z_offset = (0.5, 0.5, 0.0)
        
        # H-field components (on faces)
        H_x_offset = (0.5, 0.0, 0.0)
        H_y_offset = (0.0, 0.5, 0.0)
        H_z_offset = (0.0, 0.0, 0.5)
        
        return {
            'Ex': E_x_offset,
            'Ey': E_y_offset,
            'Ez': E_z_offset,
            'Hx': H_x_offset,
            'Hy': H_y_offset,
            'Hz': H_z_offset
        }

    @staticmethod
    def grid_resolution_wavelength(wavelength: float, 
                                   cells_per_wavelength: int = 20) -> float:
        """
        Calculate required grid resolution.

        Rule of thumb: 10-20 cells per wavelength for accuracy.

        Args:
            wavelength: Wavelength [m]
            cells_per_wavelength: Desired resolution (default: 20)

        Returns:
            Maximum cell size [m]
        """
        dx = wavelength / cells_per_wavelength
        return dx

    @staticmethod
    def minimum_wavelength(frequency_max: float, epsilon_r: float = 1.0) -> float:
        """
        Calculate minimum wavelength in simulation.

        λ_min = c/(f_max·√ε_r)

        Args:
            frequency_max: Maximum frequency [Hz]
            epsilon_r: Maximum relative permittivity

        Returns:
            Minimum wavelength [m]
        """
        c_medium = FDTD.c / np.sqrt(epsilon_r)
        lambda_min = c_medium / frequency_max
        
        return lambda_min


class MurABC:
    """
    Mur Absorbing Boundary Condition (1st order).

    Simple ABC for terminating FDTD grid.
    """

    @staticmethod
    def coefficient(c: float, dt: float, dx: float) -> float:
        """
        Calculate Mur ABC coefficient.

        E^(n+1)(boundary) = E^n(boundary-1) + 
                            ((c·Δt - Δx)/(c·Δt + Δx)) · [E^(n+1)(boundary-1) - E^n(boundary)]

        Args:
            c: Speed of light in medium [m/s]
            dt: Time step [s]
            dx: Spatial step [m]

        Returns:
            Mur coefficient
        """
        coeff = (c*dt - dx) / (c*dt + dx)
        return coeff


class GaussianPulse:
    """
    Gaussian pulse source for FDTD.
    """

    @staticmethod
    def pulse(t: float, t_0: float, tau: float, f_0: float = 0.0) -> float:
        """
        Gaussian pulse with optional carrier.

        E(t) = exp(-((t - t_0)/τ)²) · cos(2π·f_0·t)

        Args:
            t: Time [s]
            t_0: Pulse center time [s]
            tau: Pulse width [s]
            f_0: Carrier frequency [Hz] (0 for baseband)

        Returns:
            Pulse amplitude
        """
        envelope = np.exp(-((t - t_0)/tau)**2)
        
        if f_0 > 0:
            carrier = np.cos(2*np.pi*f_0*t)
            pulse = envelope * carrier
        else:
            pulse = envelope
        
        return pulse

    @staticmethod
    def pulse_spectrum_width(tau: float) -> float:
        """
        Calculate spectral width of Gaussian pulse.

        Δf ≈ 1/(π·τ)  (FWHM)

        Args:
            tau: Pulse width [s]

        Returns:
            Spectral width [Hz]
        """
        delta_f = 1.0 / (np.pi * tau)
        return delta_f
