"""
Rectangular waveguide analysis and mode propagation.

Includes:
- TE and TM modes
- Cutoff frequency
- Propagation constant
- Wave impedance
- Attenuation
- Standard waveguide dimensions (WR-XX)
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ModeType(Enum):
    """Waveguide mode types"""
    TE = 'TE'  # Transverse Electric (H modes)
    TM = 'TM'  # Transverse Magnetic (E modes)
    TEM = 'TEM'  # Transverse Electromagnetic


@dataclass
class WaveguideGeometry:
    """
    Rectangular waveguide dimensions.

    Attributes:
        a: Broad dimension (width) [m]
        b: Narrow dimension (height) [m]
        length: Waveguide length [m]
        conductivity: Wall conductivity [S/m]
    """
    a: float
    b: float
    length: float = 1.0
    conductivity: float = 5.8e7  # Copper [S/m]


class WaveguideMode:
    """
    Rectangular waveguide mode analysis.

    Physics Basis:
    --------------
    Cutoff Frequency (TE_mn or TM_mn):
        f_c = (c/2) · √((m/a)² + (n/b)²)

        where m, n = mode indices (m,n = 0,1,2,...)
              a, b = waveguide dimensions
              c = speed of light

    TE modes: E_z = 0  (m,n not both zero)
    TM modes: H_z = 0  (m,n both nonzero)

    Dominant Mode (Rectangular):
        TE₁₀ (if a > b)
        f_c = c/(2a)

    Propagation Constant:
        β = √(k² - k_c²)
        where k = ω/c
              k_c = cutoff wave number

    Wave Impedance:
        TE: Z_TE = η₀ / √(1 - (f_c/f)²)
        TM: Z_TM = η₀ · √(1 - (f_c/f)²)
        where η₀ = 377 Ω

    Phase Velocity:
        v_p = c / √(1 - (f_c/f)²)  > c  (phase velocity exceeds c)

    Group Velocity:
        v_g = c · √(1 - (f_c/f)²)  < c  (information travels slower)

    Guide Wavelength:
        λ_g = λ₀ / √(1 - (f_c/f)²)  > λ₀
    """

    # Constants
    c = 2.99792458e8    # Speed of light [m/s]
    mu_0 = 4*np.pi*1e-7 # Permeability [H/m]
    epsilon_0 = 8.854187817e-12  # Permittivity [F/m]
    eta_0 = 376.730     # Free space impedance [Ω]

    @staticmethod
    def cutoff_frequency_rectangular(a: float, b: float, mode: str = 'TE10') -> float:
        """
        Calculate cutoff frequency for rectangular waveguide.

        f_c = (c/2) · √((m/a)² + (n/b)²)

        Args:
            a: Broad dimension [m]
            b: Narrow dimension [m]
            mode: Mode designation (e.g., 'TE10', 'TE20', 'TM11')

        Returns:
            Cutoff frequency [Hz]
        """
        # Parse mode string
        mode_type = mode[:2]  # 'TE' or 'TM'
        m = int(mode[2])
        n = int(mode[3]) if len(mode) > 3 else 0
        
        # Cutoff frequency
        f_c = (WaveguideMode.c / 2.0) * np.sqrt((m/a)**2 + (n/b)**2)
        
        return f_c

    @staticmethod
    def propagation_constant(frequency: float, f_c: float) -> complex:
        """
        Calculate propagation constant β.

        For f > f_c (propagating):
            β = (2π/c) · √(f² - f_c²)  [real, propagating]

        For f < f_c (evanescent):
            β = j·(2π/c) · √(f_c² - f²)  [imaginary, decaying]

        Args:
            frequency: Operating frequency [Hz]
            f_c: Cutoff frequency [Hz]

        Returns:
            Propagation constant β [rad/m] (complex)
        """
        k = 2*np.pi*frequency / WaveguideMode.c  # Free space wave number
        k_c = 2*np.pi*f_c / WaveguideMode.c      # Cutoff wave number
        
        if frequency > f_c:
            # Propagating mode
            beta = np.sqrt(k**2 - k_c**2)
        else:
            # Evanescent mode (decays exponentially)
            beta = 1j * np.sqrt(k_c**2 - k**2)
        
        return beta

    @staticmethod
    def wave_impedance_TE(frequency: float, f_c: float) -> float:
        """
        Calculate wave impedance for TE mode.

        Z_TE = η₀ / √(1 - (f_c/f)²)

        Args:
            frequency: Operating frequency [Hz]
            f_c: Cutoff frequency [Hz]

        Returns:
            Wave impedance [Ω]
        """
        if frequency <= f_c:
            return np.inf
        
        Z_TE = WaveguideMode.eta_0 / np.sqrt(1 - (f_c/frequency)**2)
        
        return Z_TE

    @staticmethod
    def wave_impedance_TM(frequency: float, f_c: float) -> float:
        """
        Calculate wave impedance for TM mode.

        Z_TM = η₀ · √(1 - (f_c/f)²)

        Args:
            frequency: Operating frequency [Hz]
            f_c: Cutoff frequency [Hz]

        Returns:
            Wave impedance [Ω]
        """
        if frequency <= f_c:
            return 0.0
        
        Z_TM = WaveguideMode.eta_0 * np.sqrt(1 - (f_c/frequency)**2)
        
        return Z_TM

    @staticmethod
    def guide_wavelength(wavelength: float, f_c: float, frequency: float) -> float:
        """
        Calculate guide wavelength.

        λ_g = λ₀ / √(1 - (f_c/f)²)

        Args:
            wavelength: Free space wavelength [m]
            f_c: Cutoff frequency [Hz]
            frequency: Operating frequency [Hz]

        Returns:
            Guide wavelength [m]
        """
        if frequency <= f_c:
            return np.inf
        
        lambda_g = wavelength / np.sqrt(1 - (f_c/frequency)**2)
        
        return lambda_g

    @staticmethod
    def group_velocity(frequency: float, f_c: float) -> float:
        """
        Calculate group velocity.

        v_g = c · √(1 - (f_c/f)²)

        Args:
            frequency: Operating frequency [Hz]
            f_c: Cutoff frequency [Hz]

        Returns:
            Group velocity [m/s]
        """
        if frequency <= f_c:
            return 0.0
        
        v_g = WaveguideMode.c * np.sqrt(1 - (f_c/frequency)**2)
        
        return v_g

    @staticmethod
    def phase_velocity(frequency: float, f_c: float) -> float:
        """
        Calculate phase velocity.

        v_p = c / √(1 - (f_c/f)²)

        Args:
            frequency: Operating frequency [Hz]
            f_c: Cutoff frequency [Hz]

        Returns:
            Phase velocity [m/s]
        """
        if frequency <= f_c:
            return np.inf
        
        v_p = WaveguideMode.c / np.sqrt(1 - (f_c/frequency)**2)
        
        return v_p

    @staticmethod
    def attenuation_conductor(frequency: float, f_c: float, 
                             a: float, b: float, 
                             conductivity: float = 5.8e7,
                             mode: str = 'TE10') -> float:
        """
        Calculate conductor attenuation (ohmic losses).

        For TE₁₀ mode:
            α_c = (R_s/(b·η₀·√(1-(f_c/f)²))) · [1 + (2b/a)·(f_c/f)²]

        where R_s = √(π·f·μ₀/σ) = surface resistance

        Args:
            frequency: Operating frequency [Hz]
            f_c: Cutoff frequency [Hz]
            a, b: Waveguide dimensions [m]
            conductivity: Wall conductivity [S/m] (default: copper)
            mode: Mode designation

        Returns:
            Attenuation constant [Np/m]
        """
        if frequency <= f_c:
            return np.inf
        
        # Surface resistance
        R_s = np.sqrt(np.pi * frequency * WaveguideMode.mu_0 / conductivity)
        
        # TE10 mode attenuation (simplified)
        if mode == 'TE10':
            term1 = R_s / (b * WaveguideMode.eta_0 * np.sqrt(1 - (f_c/frequency)**2))
            term2 = 1 + (2*b/a) * (f_c/frequency)**2
            alpha_c = term1 * term2
        else:
            # Approximate for other modes
            alpha_c = R_s / (b * WaveguideMode.eta_0)
        
        return alpha_c  # Np/m (to convert to dB/m: multiply by 8.686)


class StandardWaveguides:
    """
    Standard rectangular waveguide dimensions (EIA/WR-XX).
    """

    # WR-XX designations (XX = broad dimension in mils/10)
    WAVEGUIDES = {
        'WR-2300': {'a': 58.42e-2, 'b': 29.21e-2, 'f_min': 0.32e9, 'f_max': 0.49e9},
        'WR-2100': {'a': 53.34e-2, 'b': 26.67e-2, 'f_min': 0.35e9, 'f_max': 0.53e9},
        'WR-1800': {'a': 45.72e-2, 'b': 22.86e-2, 'f_min': 0.41e9, 'f_max': 0.63e9},
        'WR-1500': {'a': 38.10e-2, 'b': 19.05e-2, 'f_min': 0.49e9, 'f_max': 0.75e9},
        'WR-1150': {'a': 29.21e-2, 'b': 14.61e-2, 'f_min': 0.64e9, 'f_max': 0.98e9},
        'WR-975': {'a': 24.77e-2, 'b': 12.39e-2, 'f_min': 0.76e9, 'f_max': 1.15e9},
        'WR-770': {'a': 19.55e-2, 'b': 9.78e-2, 'f_min': 0.96e9, 'f_max': 1.45e9},
        'WR-650': {'a': 16.51e-2, 'b': 8.26e-2, 'f_min': 1.14e9, 'f_max': 1.73e9},
        'WR-510': {'a': 12.95e-2, 'b': 6.48e-2, 'f_min': 1.45e9, 'f_max': 2.20e9},
        'WR-430': {'a': 10.92e-2, 'b': 5.46e-2, 'f_min': 1.72e9, 'f_max': 2.61e9},
        'WR-340': {'a': 8.636e-2, 'b': 4.318e-2, 'f_min': 2.17e9, 'f_max': 3.30e9},
        'WR-284': {'a': 7.214e-2, 'b': 3.404e-2, 'f_min': 2.60e9, 'f_max': 3.95e9},
        'WR-229': {'a': 5.817e-2, 'b': 2.908e-2, 'f_min': 3.22e9, 'f_max': 4.90e9},
        'WR-187': {'a': 4.755e-2, 'b': 2.215e-2, 'f_min': 3.94e9, 'f_max': 5.99e9},
        'WR-159': {'a': 4.039e-2, 'b': 2.019e-2, 'f_min': 4.64e9, 'f_max': 7.05e9},
        'WR-137': {'a': 3.485e-2, 'b': 1.580e-2, 'f_min': 5.38e9, 'f_max': 8.17e9},
        'WR-112': {'a': 2.850e-2, 'b': 1.262e-2, 'f_min': 6.57e9, 'f_max': 10.0e9},
        'WR-90': {'a': 2.286e-2, 'b': 1.016e-2, 'f_min': 8.20e9, 'f_max': 12.5e9},
        'WR-75': {'a': 1.905e-2, 'b': 9.525e-3, 'f_min': 9.84e9, 'f_max': 15.0e9},
        'WR-62': {'a': 1.580e-2, 'b': 7.900e-3, 'f_min': 11.9e9, 'f_max': 18.0e9},
        'WR-51': {'a': 1.295e-2, 'b': 6.477e-3, 'f_min': 14.5e9, 'f_max': 22.0e9},
        'WR-42': {'a': 1.067e-2, 'b': 4.318e-3, 'f_min': 17.6e9, 'f_max': 26.7e9},
        'WR-34': {'a': 8.636e-3, 'b': 4.318e-3, 'f_min': 21.7e9, 'f_max': 33.0e9},
        'WR-28': {'a': 7.112e-3, 'b': 3.556e-3, 'f_min': 26.4e9, 'f_max': 40.0e9},
        'WR-22': {'a': 5.690e-3, 'b': 2.845e-3, 'f_min': 33.0e9, 'f_max': 50.0e9},
        'WR-19': {'a': 4.775e-3, 'b': 2.388e-3, 'f_min': 39.2e9, 'f_max': 59.0e9},
        'WR-15': {'a': 3.759e-3, 'b': 1.880e-3, 'f_min': 49.8e9, 'f_max': 75.0e9},
        'WR-12': {'a': 3.099e-3, 'b': 1.549e-3, 'f_min': 60.5e9, 'f_max': 91.0e9},
        'WR-10': {'a': 2.540e-3, 'b': 1.270e-3, 'f_min': 73.8e9, 'f_max': 112e9},
    }

    @staticmethod
    def get_dimensions(designation: str) -> Optional[Dict]:
        """
        Get waveguide dimensions.

        Args:
            designation: Waveguide designation (e.g., 'WR-90', 'WR-187')

        Returns:
            Dictionary with a, b, f_min, f_max or None
        """
        return StandardWaveguides.WAVEGUIDES.get(designation)

    @staticmethod
    def find_waveguide_for_frequency(frequency: float) -> Optional[str]:
        """
        Find suitable waveguide for given frequency.

        Args:
            frequency: Operating frequency [Hz]

        Returns:
            Waveguide designation or None
        """
        for name, params in StandardWaveguides.WAVEGUIDES.items():
            if params['f_min'] <= frequency <= params['f_max']:
                return name
        
        return None
