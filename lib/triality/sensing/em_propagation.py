"""
Electromagnetic Propagation Physics

Implements medium-aware propagation models for radar and lidar systems:
- ITU-R rain attenuation models
- Atmospheric absorption (oxygen, water vapor)
- Fog, cloud, and aerosol effects
- Ionospheric effects
- Multipath and diffraction

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class AtmosphericConditions:
    """Atmospheric conditions for propagation modeling"""
    temperature_k: float = 288.15      # Temperature (K), default 15°C
    pressure_hpa: float = 1013.25      # Pressure (hPa), default sea level
    water_vapor_density: float = 7.5   # Water vapor density (g/m³)
    rain_rate: float = 0.0             # Rain rate (mm/hr)
    fog_visibility: Optional[float] = None  # Visibility (km), None if no fog
    relative_humidity: float = 50.0    # Relative humidity (%)


class RainAttenuation:
    """ITU-R rain attenuation models (ITU-R P.838)"""

    @staticmethod
    def specific_attenuation(frequency_ghz: float, rain_rate_mmh: float,
                            polarization: str = 'horizontal',
                            elevation_angle: float = 90.0) -> float:
        """Calculate specific rain attenuation (dB/km)

        Based on ITU-R P.838-3 model

        Args:
            frequency_ghz: Frequency in GHz (1-1000 GHz)
            rain_rate_mmh: Rain rate in mm/hr
            polarization: 'horizontal' or 'vertical'
            elevation_angle: Elevation angle in degrees (default 90° = zenith)

        Returns:
            Specific attenuation in dB/km
        """
        if rain_rate_mmh <= 0:
            return 0.0

        # ITU-R P.838-3 coefficients
        # These are simplified fits; full model has frequency-dependent coefficients
        f = frequency_ghz

        if polarization == 'horizontal':
            # Horizontal polarization coefficients (approximate)
            if f < 2.9:
                k = 0.0000387 * f ** 0.912
                alpha = 0.880 * f ** (-0.073)
            elif f < 54:
                k = 0.0001154 * f ** 1.478
                alpha = 1.380 - 0.00117 * f
            else:
                k = 0.0694 * f ** 0.325
                alpha = 2.0 - 0.0081 * f
        else:  # vertical
            # Vertical polarization coefficients (approximate)
            if f < 8.5:
                k = 0.0000868 * f ** 0.855
                alpha = 0.851 * f ** (-0.058)
            elif f < 25:
                k = 0.0004431 * f ** 1.217
                alpha = 1.41 - 0.0079 * f
            else:
                k = 0.0485 * f ** 0.446
                alpha = 1.98 - 0.0054 * f

        # Specific attenuation
        gamma_r = k * (rain_rate_mmh ** alpha)

        # Path reduction factor for elevation angles
        if elevation_angle < 90.0:
            # Simple adjustment for non-zenith paths
            theta_rad = np.radians(elevation_angle)
            if theta_rad > 0:
                path_factor = 1.0 / np.sin(theta_rad)
                # Limit path factor for low elevation angles
                path_factor = min(path_factor, 10.0)
            else:
                path_factor = 10.0
        else:
            path_factor = 1.0

        return gamma_r

    @staticmethod
    def total_attenuation(frequency_ghz: float, rain_rate_mmh: float,
                         path_length_km: float,
                         polarization: str = 'horizontal',
                         elevation_angle: float = 90.0) -> float:
        """Calculate total rain attenuation over path

        Args:
            frequency_ghz: Frequency in GHz
            rain_rate_mmh: Rain rate in mm/hr
            path_length_km: Path length in km
            polarization: 'horizontal' or 'vertical'
            elevation_angle: Elevation angle in degrees

        Returns:
            Total attenuation in dB
        """
        gamma_r = RainAttenuation.specific_attenuation(
            frequency_ghz, rain_rate_mmh, polarization, elevation_angle
        )

        # Effective path length (rain is usually not uniform over entire path)
        # For simplicity, assume uniform rain
        return gamma_r * path_length_km

    @staticmethod
    def rain_rate_from_description(description: str) -> float:
        """Get typical rain rate from description

        Args:
            description: 'light', 'moderate', 'heavy', 'violent'

        Returns:
            Rain rate in mm/hr
        """
        rain_rates = {
            'drizzle': 1.0,
            'light': 2.5,
            'moderate': 10.0,
            'heavy': 50.0,
            'violent': 100.0,
            'cloudburst': 150.0
        }
        return rain_rates.get(description.lower(), 0.0)


class AtmosphericAbsorption:
    """Atmospheric gas absorption (ITU-R P.676)"""

    @staticmethod
    def oxygen_attenuation(frequency_ghz: float, path_length_km: float,
                          temperature_k: float = 288.15,
                          pressure_hpa: float = 1013.25) -> float:
        """Calculate oxygen absorption attenuation

        Based on simplified ITU-R P.676 model

        Args:
            frequency_ghz: Frequency in GHz
            path_length_km: Path length in km
            temperature_k: Temperature in Kelvin
            pressure_hpa: Pressure in hPa

        Returns:
            Attenuation in dB
        """
        f = frequency_ghz

        # Simplified model - full ITU-R model is complex
        # Major oxygen absorption lines near 60 GHz and 118 GHz

        # Sea-level specific attenuation (dB/km) - approximate
        if f < 57:
            gamma_o = 0.0007 * f ** 2
        elif f < 63:
            # Strong absorption band around 60 GHz
            gamma_o = 15.0 * np.exp(-((f - 60.0) ** 2) / 5.0)
        elif f < 100:
            gamma_o = 0.002 * f
        else:
            # Absorption band around 118 GHz
            gamma_o = 0.01 * f + 5.0 * np.exp(-((f - 118.0) ** 2) / 10.0)

        # Pressure and temperature correction
        p_ratio = pressure_hpa / 1013.25
        t_ratio = 288.15 / temperature_k

        gamma_o_corrected = gamma_o * p_ratio * (t_ratio ** 0.8)

        return gamma_o_corrected * path_length_km

    @staticmethod
    def water_vapor_attenuation(frequency_ghz: float, path_length_km: float,
                               water_vapor_density: float = 7.5,
                               temperature_k: float = 288.15,
                               pressure_hpa: float = 1013.25) -> float:
        """Calculate water vapor absorption attenuation

        Based on simplified ITU-R P.676 model

        Args:
            frequency_ghz: Frequency in GHz
            path_length_km: Path length in km
            water_vapor_density: Water vapor density in g/m³
            temperature_k: Temperature in Kelvin
            pressure_hpa: Pressure in hPa

        Returns:
            Attenuation in dB
        """
        f = frequency_ghz
        rho = water_vapor_density

        # Simplified model
        # Major water vapor absorption lines at 22 GHz and 183 GHz

        if f < 20:
            gamma_w = 0.00001 * f ** 2
        elif f < 25:
            # Strong absorption near 22.235 GHz
            gamma_w = 0.2 * np.exp(-((f - 22.235) ** 2) / 2.0)
        elif f < 150:
            gamma_w = 0.0005 * f
        else:
            # Absorption near 183 GHz
            gamma_w = 0.005 * f + 5.0 * np.exp(-((f - 183.0) ** 2) / 20.0)

        # Scale by water vapor density
        gamma_w_total = gamma_w * (rho / 7.5)

        return gamma_w_total * path_length_km

    @staticmethod
    def total_atmospheric_attenuation(frequency_ghz: float, path_length_km: float,
                                    conditions: AtmosphericConditions) -> float:
        """Calculate total atmospheric absorption (oxygen + water vapor)

        Args:
            frequency_ghz: Frequency in GHz
            path_length_km: Path length in km
            conditions: Atmospheric conditions

        Returns:
            Total attenuation in dB
        """
        gamma_o = AtmosphericAbsorption.oxygen_attenuation(
            frequency_ghz, path_length_km,
            conditions.temperature_k, conditions.pressure_hpa
        )

        gamma_w = AtmosphericAbsorption.water_vapor_attenuation(
            frequency_ghz, path_length_km,
            conditions.water_vapor_density,
            conditions.temperature_k, conditions.pressure_hpa
        )

        return gamma_o + gamma_w


class FogCloudAttenuation:
    """Fog and cloud attenuation models"""

    @staticmethod
    def fog_attenuation(frequency_ghz: float, visibility_km: float,
                       path_length_km: float,
                       temperature_k: float = 288.15) -> float:
        """Calculate fog attenuation (Rayleigh approximation)

        Args:
            frequency_ghz: Frequency in GHz
            visibility_km: Visibility in km
            path_length_km: Path length in km
            temperature_k: Temperature in Kelvin

        Returns:
            Attenuation in dB
        """
        if visibility_km <= 0:
            return 1000.0  # Very high attenuation for zero visibility

        # Fog liquid water content from visibility (empirical)
        # M (g/m³) ≈ 0.024 / V^0.75  where V is visibility in km
        M = 0.024 / (visibility_km ** 0.75)

        # Specific attenuation (dB/km) from Rayleigh approximation
        # γ ≈ 0.4 * M * f² / (ε″ + 2)²
        # Simplified: γ ≈ K_l * M * f²

        f = frequency_ghz
        K_l = 0.05  # Empirical coefficient

        gamma_fog = K_l * M * (f ** 2)

        return gamma_fog * path_length_km

    @staticmethod
    def cloud_attenuation(frequency_ghz: float, liquid_water_content: float,
                         path_length_km: float) -> float:
        """Calculate cloud attenuation

        Args:
            frequency_ghz: Frequency in GHz
            liquid_water_content: Liquid water content in g/m³
            path_length_km: Path length through cloud in km

        Returns:
            Attenuation in dB
        """
        f = frequency_ghz
        M = liquid_water_content

        # Rayleigh approximation for cloud droplets
        K_l = 0.06  # Empirical coefficient for clouds

        gamma_cloud = K_l * M * (f ** 2)

        return gamma_cloud * path_length_km


class MultipathFading:
    """Multipath and fading effects"""

    @staticmethod
    def two_ray_path_loss(frequency_hz: float, distance_m: float,
                         tx_height: float, rx_height: float) -> float:
        """Calculate two-ray ground reflection path loss

        Args:
            frequency_hz: Frequency in Hz
            distance_m: Horizontal distance in meters
            tx_height: Transmitter height in meters
            rx_height: Receiver height in meters

        Returns:
            Additional path loss in dB (relative to free space)
        """
        wavelength = 3e8 / frequency_hz

        # Direct path
        d_direct = np.sqrt(distance_m ** 2 + (tx_height - rx_height) ** 2)

        # Reflected path
        d_reflected = np.sqrt(distance_m ** 2 + (tx_height + rx_height) ** 2)

        # Path difference
        delta_d = d_reflected - d_direct

        # Phase difference
        phase_diff = 2.0 * np.pi * delta_d / wavelength

        # Interference factor (assuming ground reflection coefficient ≈ -1)
        # E_total = E_direct - E_reflected
        # Simplified: constructive/destructive interference
        interference_factor = abs(1.0 - np.exp(1j * phase_diff))

        # Additional loss in dB
        if interference_factor > 0:
            loss_db = -20.0 * np.log10(interference_factor)
        else:
            loss_db = 100.0  # Deep null

        return loss_db

    @staticmethod
    def rayleigh_fading_probability(fade_depth_db: float) -> float:
        """Calculate probability of exceeding fade depth in Rayleigh fading

        Args:
            fade_depth_db: Fade depth in dB

        Returns:
            Probability (0-1)
        """
        # P(fade > F) = exp(-F/F_median) where F is in linear
        fade_linear = 10.0 ** (fade_depth_db / 10.0)
        return np.exp(-fade_linear)


class DiffractionLoss:
    """Diffraction loss models"""

    @staticmethod
    def knife_edge_diffraction(frequency_hz: float,
                              d1: float, d2: float, h: float) -> float:
        """Calculate knife-edge diffraction loss

        Args:
            frequency_hz: Frequency in Hz
            d1: Distance from transmitter to obstacle (m)
            d2: Distance from obstacle to receiver (m)
            h: Height of obstacle above line-of-sight (m)

        Returns:
            Diffraction loss in dB
        """
        wavelength = 3e8 / frequency_hz

        # Fresnel-Kirchhoff diffraction parameter
        v = h * np.sqrt(2.0 * (d1 + d2) / (wavelength * d1 * d2))

        # Approximation for diffraction loss
        if v <= -0.7:
            # Line of sight
            loss_db = 0.0
        else:
            # Diffraction loss (Epstein-Peterson approximation)
            loss_db = 6.9 + 20.0 * np.log10(np.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)

        return max(0.0, loss_db)


class PropagationCalculator:
    """High-level propagation calculator combining all effects"""

    @staticmethod
    def total_path_loss(frequency_ghz: float, range_km: float,
                       conditions: AtmosphericConditions,
                       elevation_angle: float = 90.0,
                       polarization: str = 'horizontal') -> Dict[str, float]:
        """Calculate total path loss including all atmospheric effects

        Args:
            frequency_ghz: Frequency in GHz
            range_km: Range in km
            conditions: Atmospheric conditions
            elevation_angle: Elevation angle in degrees
            polarization: Polarization ('horizontal' or 'vertical')

        Returns:
            Dictionary with breakdown of losses in dB
        """
        # Free space path loss (two-way for radar)
        wavelength = 3e8 / (frequency_ghz * 1e9)
        fspl_db = 20.0 * np.log10(4.0 * np.pi * range_km * 1000.0 / wavelength)
        # Two-way: double the one-way loss
        fspl_two_way_db = 2.0 * fspl_db

        # Rain attenuation (two-way)
        rain_db = 2.0 * RainAttenuation.total_attenuation(
            frequency_ghz, conditions.rain_rate, range_km,
            polarization, elevation_angle
        )

        # Atmospheric absorption (two-way)
        atmos_db = 2.0 * AtmosphericAbsorption.total_atmospheric_attenuation(
            frequency_ghz, range_km, conditions
        )

        # Fog attenuation (two-way) if present
        fog_db = 0.0
        if conditions.fog_visibility is not None:
            fog_db = 2.0 * FogCloudAttenuation.fog_attenuation(
                frequency_ghz, conditions.fog_visibility, range_km,
                conditions.temperature_k
            )

        # Total loss
        total_db = fspl_two_way_db + rain_db + atmos_db + fog_db

        return {
            'free_space_loss_db': fspl_two_way_db,
            'rain_loss_db': rain_db,
            'atmospheric_loss_db': atmos_db,
            'fog_loss_db': fog_db,
            'total_loss_db': total_db,
            'breakdown': {
                'FSPL': fspl_two_way_db,
                'Rain': rain_db,
                'Atmosphere': atmos_db,
                'Fog': fog_db
            }
        }

    @staticmethod
    def range_limited_by_attenuation(frequency_ghz: float,
                                     max_allowable_loss_db: float,
                                     conditions: AtmosphericConditions,
                                     polarization: str = 'horizontal') -> float:
        """Find maximum range given loss budget

        Args:
            frequency_ghz: Frequency in GHz
            max_allowable_loss_db: Maximum allowable path loss in dB
            conditions: Atmospheric conditions
            polarization: Polarization

        Returns:
            Maximum range in km
        """
        # Binary search for range
        range_min = 0.1  # km
        range_max = 1000.0  # km

        for _ in range(20):  # Iterations for convergence
            range_test = (range_min + range_max) / 2.0

            losses = PropagationCalculator.total_path_loss(
                frequency_ghz, range_test, conditions, 90.0, polarization
            )

            if losses['total_loss_db'] > max_allowable_loss_db:
                range_max = range_test
            else:
                range_min = range_test

        return (range_min + range_max) / 2.0


# Convenience functions
def rain_loss(frequency_ghz: float, range_km: float, rain_rate_mmh: float) -> float:
    """Quick rain loss calculation (two-way)

    Args:
        frequency_ghz: Frequency in GHz
        range_km: Range in km
        rain_rate_mmh: Rain rate in mm/hr

    Returns:
        Two-way rain loss in dB
    """
    return 2.0 * RainAttenuation.total_attenuation(
        frequency_ghz, rain_rate_mmh, range_km
    )


def atmospheric_loss(frequency_ghz: float, range_km: float) -> float:
    """Quick atmospheric loss calculation (standard conditions, two-way)

    Args:
        frequency_ghz: Frequency in GHz
        range_km: Range in km

    Returns:
        Two-way atmospheric loss in dB
    """
    conditions = AtmosphericConditions()
    return 2.0 * AtmosphericAbsorption.total_atmospheric_attenuation(
        frequency_ghz, range_km, conditions
    )
