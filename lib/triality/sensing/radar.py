"""
Radar System Convenience Interface

High-level interface for radar system analysis combining all Triality sensing modules.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Optional, Dict
from dataclasses import dataclass

from . import signals, em_propagation, targets, noise, detection, tradeoffs, feasibility


@dataclass
class RadarSystem:
    """Complete radar system specification"""
    # Transmitter
    frequency_ghz: float = 10.0          # Frequency (GHz)
    power_w: float = 1000.0              # Peak power (W)
    pulse_width_us: float = 1.0          # Pulse width (μs)
    prf_hz: float = 1000.0               # PRF (Hz)

    # Antenna
    aperture_diameter_m: float = 1.0     # Aperture diameter (m)

    # Receiver
    bandwidth_mhz: float = 1.0           # Bandwidth (MHz)
    noise_figure_db: float = 3.0         # Noise figure (dB)

    # Integration
    n_pulses_integrated: int = 10        # Number of pulses integrated

    # System
    losses_db: float = 3.0               # System losses (dB)


class RadarAnalyzer:
    """Comprehensive radar system analyzer"""

    def __init__(self, system: RadarSystem):
        """Initialize analyzer with radar system

        Args:
            system: Radar system specification
        """
        self.system = system

        # Derived parameters
        self.wavelength = 3e8 / (system.frequency_ghz * 1e9)
        self.bandwidth_hz = system.bandwidth_mhz * 1e6

    def analyze_performance(self, target_rcs_m2: float = 1.0,
                          range_km: float = 10.0,
                          weather: str = 'clear') -> Dict:
        """Complete performance analysis

        Args:
            target_rcs_m2: Target RCS in m²
            range_km: Range to target in km
            weather: Weather condition

        Returns:
            Complete performance dictionary
        """
        results = {}

        # 1. Link budget analysis
        results['link_budget'] = self._analyze_link_budget(
            target_rcs_m2, range_km, weather
        )

        # 2. Resolution capabilities
        results['resolution'] = self._analyze_resolution(range_km)

        # 3. Ambiguities
        results['ambiguities'] = self._analyze_ambiguities()

        # 4. Feasibility check
        results['feasibility'] = self._check_feasibility(
            target_rcs_m2, range_km, weather
        )

        return results

    def _analyze_link_budget(self, target_rcs_m2: float,
                            range_km: float, weather: str) -> Dict:
        """Analyze link budget"""
        # Antenna gain
        aperture_area = np.pi * (self.system.aperture_diameter_m / 2) ** 2
        antenna_gain_linear = (4 * np.pi * aperture_area * 0.6) / (self.wavelength ** 2)
        antenna_gain_db = 10 * np.log10(antenna_gain_linear)

        # Calculate max range
        max_range_m = tradeoffs.RadarBudget.max_range(
            self.system.power_w,
            antenna_gain_db,
            self.wavelength,
            target_rcs_m2,
            13.0,  # 13 dB SNR for Pd=0.9, Pfa=1e-6
            self.system.losses_db,
            self.system.noise_figure_db,
            self.bandwidth_hz
        )

        # Propagation losses
        weather_map = {'clear': 0.0, 'light_rain': 2.5,
                      'moderate_rain': 10.0, 'heavy_rain': 50.0}
        rain_rate = weather_map.get(weather, 0.0)

        rain_loss_db = em_propagation.rain_loss(
            self.system.frequency_ghz, range_km, rain_rate
        )

        return {
            'antenna_gain_db': antenna_gain_db,
            'max_range_km': max_range_m / 1000,
            'required_range_km': range_km,
            'margin_db': 20 * np.log10(max_range_m / (range_km * 1000)),
            'rain_loss_db': rain_loss_db,
            'weather_limited': rain_loss_db > 10
        }

    def _analyze_resolution(self, range_km: float) -> Dict:
        """Analyze resolution"""
        # Range resolution
        range_res_m = signals.SignalCharacteristics.range_resolution(
            self.bandwidth_hz, 'vacuum'
        )

        # Angular resolution
        beamwidth_rad = signals.BeamCharacteristics.beamwidth_2d(
            self.wavelength, self.system.aperture_diameter_m
        )
        beamwidth_deg = np.degrees(beamwidth_rad)

        # Cross-range resolution
        cross_range_res_m = beamwidth_rad * range_km * 1000

        # Doppler resolution
        cpi_time = self.system.n_pulses_integrated / self.system.prf_hz
        doppler_res_hz = detection.ResolutionLimits.doppler_resolution(cpi_time)
        velocity_res_ms = detection.ResolutionLimits.velocity_resolution(
            cpi_time, self.wavelength
        )

        return {
            'range_resolution_m': range_res_m,
            'beamwidth_deg': beamwidth_deg,
            'cross_range_resolution_m': cross_range_res_m,
            'doppler_resolution_hz': doppler_res_hz,
            'velocity_resolution_ms': velocity_res_ms
        }

    def _analyze_ambiguities(self) -> Dict:
        """Analyze range and velocity ambiguities"""
        max_unambig_range_m = signals.SignalCharacteristics.unambiguous_range(
            self.system.prf_hz, 'vacuum'
        )

        max_unambig_velocity_ms = signals.SignalCharacteristics.unambiguous_velocity(
            self.system.prf_hz, self.wavelength
        )

        return {
            'max_unambiguous_range_km': max_unambig_range_m / 1000,
            'max_unambiguous_velocity_ms': max_unambig_velocity_ms,
            'prf_hz': self.system.prf_hz
        }

    def _check_feasibility(self, target_rcs_m2: float,
                          range_km: float, weather: str) -> Dict:
        """Check system feasibility"""
        checker = feasibility.FeasibilityChecker('radar')
        report = checker.check_radar_system(
            self.system.frequency_ghz,
            self.system.power_w,
            self.system.aperture_diameter_m,
            range_km,
            target_rcs_m2,
            weather_condition=weather,
            bandwidth_hz=self.bandwidth_hz
        )

        return {
            'is_feasible': report.is_feasible,
            'assessment': report.overall_assessment,
            'num_showstoppers': report.showstoppers,
            'num_critical': report.critical_issues,
            'num_warnings': report.warnings
        }

    def print_summary(self, target_rcs_m2: float = 1.0,
                     range_km: float = 10.0,
                     weather: str = 'clear'):
        """Print performance summary

        Args:
            target_rcs_m2: Target RCS
            range_km: Range
            weather: Weather condition
        """
        results = self.analyze_performance(target_rcs_m2, range_km, weather)

        print("=" * 70)
        print(f"RADAR SYSTEM PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"\nSystem:")
        print(f"  Frequency: {self.system.frequency_ghz} GHz")
        print(f"  Power: {self.system.power_w} W")
        print(f"  Aperture: {self.system.aperture_diameter_m} m")
        print(f"  PRF: {self.system.prf_hz} Hz")

        print(f"\nLink Budget:")
        lb = results['link_budget']
        print(f"  Max range: {lb['max_range_km']:.2f} km")
        print(f"  Margin: {lb['margin_db']:.1f} dB")
        print(f"  Rain loss ({weather}): {lb['rain_loss_db']:.1f} dB")

        print(f"\nResolution:")
        res = results['resolution']
        print(f"  Range: {res['range_resolution_m']:.2f} m")
        print(f"  Cross-range @ {range_km}km: {res['cross_range_resolution_m']:.2f} m")
        print(f"  Velocity: {res['velocity_resolution_ms']:.2f} m/s")

        print(f"\nAmbiguities:")
        amb = results['ambiguities']
        print(f"  Max unambig range: {amb['max_unambiguous_range_km']:.2f} km")
        print(f"  Max unambig velocity: {amb['max_unambiguous_velocity_ms']:.1f} m/s")

        print(f"\nFeasibility:")
        feas = results['feasibility']
        print(f"  {feas['assessment']}")

        print("=" * 70)


# Convenience functions
def quick_radar_analysis(frequency_ghz: float = 10.0,
                        power_w: float = 1000.0,
                        aperture_m: float = 1.0,
                        range_km: float = 10.0,
                        target_rcs_m2: float = 1.0) -> Dict:
    """Quick radar performance analysis

    Args:
        frequency_ghz: Frequency (GHz)
        power_w: Power (W)
        aperture_m: Aperture diameter (m)
        range_km: Range (km)
        target_rcs_m2: Target RCS (m²)

    Returns:
        Performance summary
    """
    system = RadarSystem(
        frequency_ghz=frequency_ghz,
        power_w=power_w,
        aperture_diameter_m=aperture_m
    )

    analyzer = RadarAnalyzer(system)
    return analyzer.analyze_performance(target_rcs_m2, range_km)


def check_radar_feasibility(frequency_ghz: float, power_w: float,
                           aperture_m: float, range_km: float,
                           weather: str = 'moderate_rain') -> bool:
    """Quick feasibility check

    Args:
        frequency_ghz: Frequency (GHz)
        power_w: Power (W)
        aperture_m: Aperture (m)
        range_km: Range (km)
        weather: Weather condition

    Returns:
        True if feasible
    """
    return feasibility.quick_radar_check(
        frequency_ghz, power_w, aperture_m, range_km,
        weather=weather
    )
