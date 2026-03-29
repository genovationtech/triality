"""
Sonar System Convenience Interface

High-level interface for sonar system analysis.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass

from . import signals, acoustic_propagation, targets


@dataclass
class SonarSystem:
    """Sonar system specification"""
    frequency_khz: float = 50.0          # Frequency (kHz)
    source_level_db: float = 220.0       # Source level (dB re 1 μPa @ 1m)
    array_size_m: float = 1.0            # Array size (m)
    bandwidth_khz: float = 10.0          # Bandwidth (kHz)
    is_active: bool = True               # Active (True) or passive (False)


class SonarAnalyzer:
    """Sonar system analyzer"""

    def __init__(self, system: SonarSystem):
        """Initialize analyzer

        Args:
            system: Sonar system specification
        """
        self.system = system

        # Sound speed in water (typical)
        self.sound_speed = 1500.0  # m/s
        self.wavelength = self.sound_speed / (system.frequency_khz * 1000)

    def analyze_performance(self, target_strength_db: float = 10.0,
                          range_km: float = 5.0,
                          water_temp_c: float = 15.0) -> Dict:
        """Analyze sonar performance

        Args:
            target_strength_db: Target strength (dB)
            range_km: Range (km)
            water_temp_c: Water temperature (°C)

        Returns:
            Performance dictionary
        """
        # Transmission loss
        from . import acoustic_propagation

        conditions = acoustic_propagation.WaterConditions(
            temperature_c=water_temp_c,
            salinity_ppt=35.0
        )

        loss_result = acoustic_propagation.PropagationLoss.underwater_transmission_loss(
            self.system.frequency_khz,
            range_km,
            conditions
        )

        # Ambient noise
        noise_db = acoustic_propagation.AmbientNoise.underwater_ambient_noise(
            self.system.frequency_khz,
            sea_state=3
        )

        # Signal excess (active sonar)
        if self.system.is_active:
            signal_excess = acoustic_propagation.SonarEquation.active_sonar_snr(
                self.system.source_level_db,
                loss_result['total_loss_db'] / 2,  # One-way for active
                target_strength_db,
                noise_db,
                directivity_index_db=10
            )
        else:
            signal_excess = acoustic_propagation.SonarEquation.passive_sonar_snr(
                target_strength_db,  # Target radiated noise
                loss_result['total_loss_db'],
                noise_db,
                directivity_index_db=10
            )

        return {
            'transmission_loss_db': loss_result['total_loss_db'],
            'absorption_loss_db': loss_result['absorption_loss_db'],
            'spreading_loss_db': loss_result['spreading_loss_db'],
            'ambient_noise_db': noise_db,
            'signal_excess_db': signal_excess,
            'detection_feasible': signal_excess > 0
        }

    def analyze_resolution(self, range_km: float = 1.0) -> Dict:
        """Analyze resolution

        Args:
            range_km: Range for cross-range calculation

        Returns:
            Resolution dictionary
        """
        # Range resolution
        range_res_m = signals.SignalCharacteristics.range_resolution(
            self.system.bandwidth_khz * 1000, 'water'
        )

        # Angular resolution
        beamwidth_rad = signals.BeamCharacteristics.beamwidth_2d(
            self.wavelength, self.system.array_size_m
        )

        # Cross-range resolution
        cross_range_res_m = beamwidth_rad * range_km * 1000

        return {
            'range_resolution_m': range_res_m,
            'beamwidth_deg': np.degrees(beamwidth_rad),
            'cross_range_resolution_m': cross_range_res_m
        }


# Convenience functions
def sonar_detection_range(source_level_db: float = 220.0,
                         target_strength_db: float = 10.0,
                         frequency_khz: float = 50.0) -> float:
    """Estimate sonar detection range

    Args:
        source_level_db: Source level (dB)
        target_strength_db: Target strength (dB)
        frequency_khz: Frequency (kHz)

    Returns:
        Approximate detection range (km)
    """
    # Simplified calculation
    # Assume detection threshold of 10 dB signal excess

    # Available signal budget
    budget_db = source_level_db + target_strength_db - 10  # 10 dB threshold

    # Estimate range from budget
    # Rough approximation: 60 dB loss at 1 km, increases ~20 log R
    range_km = 10 ** ((budget_db - 60) / 20)

    return range_km
