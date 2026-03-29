"""
Dipole antenna analysis and radiation patterns.

Includes:
- Hertzian dipole (infinitesimal)
- Half-wave dipole (λ/2)
- Quarter-wave monopole (λ/4)
- Radiation patterns
- Gain and directivity
- Input impedance
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class DipoleType(Enum):
    """Types of dipole antennas"""
    HERTZIAN = 'hertzian'          # Infinitesimal (L << λ)
    SHORT = 'short'                # L < λ/10
    HALF_WAVE = 'half_wave'        # L = λ/2
    FULL_WAVE = 'full_wave'        # L = λ
    MONOPOLE = 'monopole'          # L = λ/4 (over ground plane)


@dataclass
class AntennaParameters:
    """
    Antenna physical parameters.

    Attributes:
        length: Antenna length [m]
        frequency: Operating frequency [Hz]
        wavelength: Wavelength [m]
        radius: Wire radius [m]
        efficiency: Radiation efficiency (0-1)
    """
    length: float
    frequency: float
    wavelength: float
    radius: float = 0.001  # m (1 mm wire)
    efficiency: float = 1.0


class DipoleAntenna:
    """
    Dipole antenna analysis.

    Physics Basis:
    --------------
    Wavelength:
        λ = c/f
        where c = 3×10^8 m/s

    Wave Number:
        k = 2π/λ [rad/m]

    Hertzian Dipole Radiation Pattern:
        E_θ = (j·k·I_0·L)/(4πr) · exp(-jkr) · sin(θ)
        
    Far-Field Power Density:
        S = |E|²/(2·η₀)
        where η₀ = 377 Ω (free space impedance)

    Radiation Resistance (Hertzian):
        R_rad = 80·π² · (L/λ)²  [Ω]

    Half-Wave Dipole:
        R_rad = 73 Ω
        Directivity = 1.64 (2.15 dBi)
        
    Gain:
        G = e_cd · D
        where e_cd = radiation efficiency
              D = directivity
    """

    # Constants
    c = 2.99792458e8    # Speed of light [m/s]
    eta_0 = 376.730     # Free space impedance [Ω]
    mu_0 = 4*np.pi*1e-7 # Permeability [H/m]
    epsilon_0 = 8.854187817e-12  # Permittivity [F/m]

    def __init__(self, length: float, frequency: float, 
                 radius: float = 0.001, efficiency: float = 1.0):
        """
        Initialize dipole antenna.

        Args:
            length: Physical length [m]
            frequency: Operating frequency [Hz]
            radius: Wire radius [m]
            efficiency: Radiation efficiency (0-1)
        """
        self.length = length
        self.frequency = frequency
        self.radius = radius
        self.efficiency = efficiency
        
        # Derived parameters
        self.wavelength = DipoleAntenna.c / frequency
        self.k = 2*np.pi / self.wavelength  # Wave number [rad/m]
        
        # Electrical length
        self.L_lambda = length / self.wavelength

    def radiation_resistance(self) -> float:
        """
        Calculate radiation resistance.

        For short dipole (L << λ):
            R_rad = 80·π² · (L/λ)²

        For half-wave (L = λ/2):
            R_rad ≈ 73 Ω

        Returns:
            Radiation resistance [Ω]
        """
        if self.L_lambda < 0.1:
            # Short dipole approximation
            R_rad = 80 * np.pi**2 * self.L_lambda**2
        elif abs(self.L_lambda - 0.5) < 0.05:
            # Half-wave dipole
            R_rad = 73.1
        else:
            # General case (approximate)
            R_rad = 80 * np.pi**2 * self.L_lambda**2
            
        return R_rad

    def input_impedance(self) -> complex:
        """
        Calculate input impedance.

        Half-wave dipole:
            Z_in ≈ 73 + j42.5 Ω

        Returns:
            Complex impedance [Ω]
        """
        R_rad = self.radiation_resistance()
        
        if abs(self.L_lambda - 0.5) < 0.05:
            # Half-wave dipole
            Z_in = 73.1 + 42.5j
        elif self.L_lambda < 0.5:
            # Capacitive (shorter than resonance)
            X = -100 * (0.5 - self.L_lambda)
            Z_in = R_rad + X*1j
        else:
            # Inductive (longer than resonance)
            X = 100 * (self.L_lambda - 0.5)
            Z_in = R_rad + X*1j
            
        return Z_in

    def directivity(self) -> float:
        """
        Calculate directivity.

        Hertzian dipole: D = 1.5 (1.76 dBi)
        Half-wave dipole: D = 1.64 (2.15 dBi)

        Returns:
            Directivity (linear)
        """
        if self.L_lambda < 0.1:
            # Hertzian dipole
            D = 1.5
        elif abs(self.L_lambda - 0.5) < 0.05:
            # Half-wave dipole
            D = 1.64
        else:
            # Approximate
            D = 1.5 + 0.14 * min(self.L_lambda, 0.5)
            
        return D

    def gain_dbi(self) -> float:
        """
        Calculate gain in dBi.

        G_dBi = 10·log₁₀(e_cd · D)

        where e_cd = radiation efficiency
              D = directivity

        Returns:
            Gain [dBi]
        """
        D = self.directivity()
        G = self.efficiency * D
        G_dBi = 10 * np.log10(G)
        
        return G_dBi

    def radiation_pattern(self, theta: np.ndarray) -> np.ndarray:
        """
        Calculate normalized radiation pattern.

        Hertzian dipole:
            F(θ) = sin(θ)

        Half-wave dipole:
            F(θ) = cos(π/2·cos(θ)) / sin(θ)

        Args:
            theta: Elevation angle [radians] (0 = z-axis)

        Returns:
            Normalized field pattern
        """
        if self.L_lambda < 0.1:
            # Hertzian dipole
            F = np.abs(np.sin(theta))
        elif abs(self.L_lambda - 0.5) < 0.05:
            # Half-wave dipole
            with np.errstate(divide='ignore', invalid='ignore'):
                F = np.abs(np.cos(np.pi/2 * np.cos(theta)) / np.sin(theta))
                F[np.isnan(F)] = 0.0
                F[np.isinf(F)] = 0.0
        else:
            # General case (simplified)
            F = np.abs(np.sin(theta))
        
        # Normalize
        F_max = np.max(F)
        if F_max > 0:
            F = F / F_max
            
        return F

    def beamwidth_3db(self) -> float:
        """
        Calculate 3-dB beamwidth (half-power beamwidth).

        Half-wave dipole: ~78° in E-plane

        Returns:
            Beamwidth [degrees]
        """
        if abs(self.L_lambda - 0.5) < 0.05:
            return 78.0  # degrees
        else:
            # Approximate
            return 90.0  # degrees

    def effective_area(self) -> float:
        """
        Calculate effective aperture area.

        A_eff = (λ²/(4π)) · G

        Args:
            None

        Returns:
            Effective area [m²]
        """
        D = self.directivity()
        G = self.efficiency * D
        A_eff = (self.wavelength**2 / (4*np.pi)) * G
        
        return A_eff

    @staticmethod
    def friis_transmission(P_t: float, G_t: float, G_r: float, 
                          distance: float, wavelength: float) -> float:
        """
        Friis transmission equation for free space.

        P_r = P_t · G_t · G_r · (λ/(4πd))²

        Args:
            P_t: Transmit power [W]
            G_t: Transmit antenna gain (linear)
            G_r: Receive antenna gain (linear)
            distance: Separation distance [m]
            wavelength: Wavelength [m]

        Returns:
            Received power [W]
        """
        if distance <= 0:
            return 0.0
        
        # Free space path loss
        FSPL = (wavelength / (4*np.pi*distance))**2
        
        P_r = P_t * G_t * G_r * FSPL
        
        return P_r


class MonopoleAntenna:
    """
    Quarter-wave monopole antenna (over ground plane).

    Equivalent to half-wave dipole with ground plane.
    """

    @staticmethod
    def input_impedance() -> float:
        """
        Quarter-wave monopole input impedance.

        Z_in ≈ 36.5 Ω  (half of dipole, due to ground plane)

        Returns:
            Input resistance [Ω]
        """
        return 36.5

    @staticmethod
    def radiation_resistance() -> float:
        """
        Quarter-wave monopole radiation resistance.

        Returns:
            Radiation resistance [Ω]
        """
        return 36.5

    @staticmethod
    def gain_dbi() -> float:
        """
        Quarter-wave monopole gain.

        G = 5.15 dBi  (3 dB higher than dipole due to ground plane)

        Returns:
            Gain [dBi]
        """
        return 5.15


class WireDipoleArray:
    """
    Array of dipole antennas.
    """

    @staticmethod
    def array_factor_2element(d: float, wavelength: float, 
                              delta: float = 0.0) -> np.ndarray:
        """
        Array factor for 2-element array.

        AF = cos(kd·cos(θ) + δ)

        Args:
            d: Element spacing [m]
            wavelength: Wavelength [m]
            delta: Phase difference [radians]

        Returns:
            Array factor (normalized)
        """
        theta = np.linspace(0, np.pi, 181)
        k = 2*np.pi / wavelength
        
        AF = np.cos(k * d * np.cos(theta) + delta)
        AF_norm = np.abs(AF / np.max(np.abs(AF)))
        
        return AF_norm
