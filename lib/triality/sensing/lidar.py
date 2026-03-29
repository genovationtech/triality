"""
Lidar System Convenience Interface

High-level interface for lidar system analysis.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass

from . import signals, targets


@dataclass
class LidarSystem:
    """Lidar system specification"""
    wavelength_nm: float = 1550.0        # Wavelength (nm)
    power_w: float = 0.1                 # Average power (W)
    pulse_energy_uj: float = 100.0       # Pulse energy (μJ)
    aperture_diameter_m: float = 0.1     # Aperture (m)
    range_m: float = 200.0               # Max range (m)


class LidarAnalyzer:
    """Lidar system analyzer"""

    def __init__(self, system: LidarSystem):
        """Initialize analyzer

        Args:
            system: Lidar system specification
        """
        self.system = system
        self.wavelength_m = system.wavelength_nm * 1e-9

    def analyze_resolution(self, range_m: float = 100.0) -> Dict:
        """Analyze resolution

        Args:
            range_m: Range for cross-range calculation

        Returns:
            Resolution dictionary
        """
        # Angular resolution
        beamwidth_rad = signals.BeamCharacteristics.beamwidth_2d(
            self.wavelength_m, self.system.aperture_diameter_m
        )

        # Cross-range resolution
        cross_range_res_m = beamwidth_rad * range_m

        # Range resolution (typical for lidar)
        range_res_m = 0.02  # 2 cm (limited by pulse width)

        return {
            'range_resolution_m': range_res_m,
            'beamwidth_rad': beamwidth_rad,
            'beamwidth_deg': np.degrees(beamwidth_rad),
            'cross_range_resolution_m': cross_range_res_m
        }


# Convenience functions
def lidar_resolution(aperture_m: float = 0.1,
                    wavelength_nm: float = 1550.0,
                    range_m: float = 100.0) -> Dict:
    """Quick lidar resolution calculation

    Args:
        aperture_m: Aperture diameter (m)
        wavelength_nm: Wavelength (nm)
        range_m: Range (m)

    Returns:
        Resolution dictionary
    """
    system = LidarSystem(
        aperture_diameter_m=aperture_m,
        wavelength_nm=wavelength_nm
    )
    analyzer = LidarAnalyzer(system)
    return analyzer.analyze_resolution(range_m)
