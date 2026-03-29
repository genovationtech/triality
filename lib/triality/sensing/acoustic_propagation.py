"""
Acoustic Propagation Physics for Sonar Systems

Implements acoustic propagation models for underwater and airborne sonar:
- Absorption (frequency-dependent, medium-dependent)
- Speed of sound variations
- Refraction and ducting
- Boundary reflections
- Ambient noise models

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class WaterConditions:
    """Water conditions for underwater acoustics"""
    temperature_c: float = 15.0       # Temperature (°C)
    salinity_ppt: float = 35.0        # Salinity (parts per thousand)
    depth_m: float = 0.0              # Depth (m)
    ph: float = 8.0                   # pH
    sea_state: int = 3                # Sea state (0-9)


@dataclass
class AirConditions:
    """Air conditions for airborne acoustics"""
    temperature_c: float = 15.0       # Temperature (°C)
    pressure_hpa: float = 1013.25     # Pressure (hPa)
    relative_humidity: float = 50.0   # Relative humidity (%)


class SoundSpeed:
    """Sound speed calculations for different media"""

    @staticmethod
    def in_water(temperature_c: float, salinity_ppt: float = 35.0,
                 depth_m: float = 0.0) -> float:
        """Calculate sound speed in seawater (Mackenzie equation)

        Args:
            temperature_c: Temperature in °C
            salinity_ppt: Salinity in parts per thousand
            depth_m: Depth in meters

        Returns:
            Sound speed in m/s
        """
        T = temperature_c
        S = salinity_ppt
        D = depth_m

        # Mackenzie (1981) equation - accurate to ±0.07 m/s
        c = (1448.96 + 4.591 * T - 5.304e-2 * T**2 + 2.374e-4 * T**3
             + 1.340 * (S - 35) + 1.630e-2 * D + 1.675e-7 * D**2
             - 1.025e-2 * T * (S - 35) - 7.139e-13 * T * D**3)

        return c

    @staticmethod
    def in_air(temperature_c: float, pressure_hpa: float = 1013.25,
               relative_humidity: float = 50.0) -> float:
        """Calculate sound speed in air

        Args:
            temperature_c: Temperature in °C
            pressure_hpa: Pressure in hPa
            relative_humidity: Relative humidity (%)

        Returns:
            Sound speed in m/s
        """
        # Simplified formula (accurate to ~0.1 m/s for typical conditions)
        c = 331.3 + 0.606 * temperature_c

        return c

    @staticmethod
    def gradient_in_water(surface_temp_c: float, depth_m: float,
                         salinity_ppt: float = 35.0) -> Tuple[float, float]:
        """Calculate sound speed gradient with depth

        Args:
            surface_temp_c: Surface temperature in °C
            depth_m: Depth in meters
            salinity_ppt: Salinity

        Returns:
            Tuple of (sound_speed, gradient_per_meter)
        """
        # Typical thermocline: temperature drops with depth
        # Simplified model
        if depth_m < 100:
            # Mixed layer: relatively constant
            temp_at_depth = surface_temp_c
        else:
            # Thermocline: temperature decreases
            temp_at_depth = surface_temp_c - 0.03 * (depth_m - 100)

        c = SoundSpeed.in_water(temp_at_depth, salinity_ppt, depth_m)

        # Gradient (approximate)
        c_shallow = SoundSpeed.in_water(surface_temp_c, salinity_ppt, depth_m - 1)
        gradient = c - c_shallow  # m/s per meter

        return c, gradient


class UnderwaterAbsorption:
    """Underwater acoustic absorption models"""

    @staticmethod
    def thorp_absorption(frequency_khz: float) -> float:
        """Thorp's absorption formula (simplified, valid for deep ocean)

        Args:
            frequency_khz: Frequency in kHz

        Returns:
            Absorption coefficient in dB/km
        """
        f = frequency_khz

        # Thorp's formula (empirical, for f > 0.4 kHz)
        f2 = f ** 2

        alpha = (0.11 * f2 / (1 + f2) +
                44 * f2 / (4100 + f2) +
                2.75e-4 * f2 + 0.003)

        return alpha  # dB/km

    @staticmethod
    def francois_garrison_absorption(frequency_khz: float,
                                     conditions: WaterConditions) -> float:
        """Francois-Garrison absorption model (more accurate)

        Args:
            frequency_khz: Frequency in kHz
            conditions: Water conditions

        Returns:
            Absorption coefficient in dB/km
        """
        f = frequency_khz
        T = conditions.temperature_c
        S = conditions.salinity_ppt
        D = conditions.depth_m
        pH = conditions.ph

        # Boric acid contribution
        f1 = 0.78 * np.sqrt(S / 35) * np.exp(T / 26)
        A1 = 0.106 * np.exp((pH - 8) / 0.56)
        P1 = 1.0
        alpha_boric = A1 * P1 * f1 * f**2 / (f1**2 + f**2)

        # Magnesium sulfate contribution
        f2 = 42 * np.exp(T / 17)
        A2 = 0.52 * (1 + T / 43) * (S / 35)
        P2 = 1 - 1.37e-4 * D + 6.2e-9 * D**2
        alpha_mgso4 = A2 * P2 * f2 * f**2 / (f2**2 + f**2)

        # Pure water contribution
        if T < 20:
            A3 = 4.9e-4
        else:
            A3 = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T**2 - 6.5e-10 * T**3

        P3 = 1 - 3.83e-5 * D + 4.9e-10 * D**2
        alpha_water = A3 * P3 * f**2

        # Total absorption
        alpha_total = alpha_boric + alpha_mgso4 + alpha_water

        return alpha_total  # dB/km

    @staticmethod
    def total_absorption_loss(frequency_khz: float, range_km: float,
                             conditions: WaterConditions,
                             model: str = 'thorp') -> float:
        """Calculate total absorption loss

        Args:
            frequency_khz: Frequency in kHz
            range_km: Range in km
            conditions: Water conditions
            model: 'thorp' or 'francois_garrison'

        Returns:
            Absorption loss in dB
        """
        if model == 'thorp':
            alpha = UnderwaterAbsorption.thorp_absorption(frequency_khz)
        else:
            alpha = UnderwaterAbsorption.francois_garrison_absorption(
                frequency_khz, conditions
            )

        return alpha * range_km


class AirborneAbsorption:
    """Airborne acoustic absorption (ISO 9613-1)"""

    @staticmethod
    def atmospheric_absorption(frequency_hz: float, range_m: float,
                              conditions: AirConditions) -> float:
        """Calculate atmospheric absorption for airborne sound

        Based on ISO 9613-1

        Args:
            frequency_hz: Frequency in Hz
            range_m: Range in meters
            conditions: Air conditions

        Returns:
            Absorption loss in dB
        """
        T = conditions.temperature_c + 273.15  # Kelvin
        h = conditions.relative_humidity
        p = conditions.pressure_hpa / 1013.25  # Relative pressure

        f = frequency_hz

        # Simplified absorption coefficient (dB/m)
        # Full ISO 9613-1 is complex; this is approximate
        T_ref = 293.15  # 20°C
        T_ratio = T / T_ref

        # Empirical formula (simplified)
        alpha = (1.6e-10 * f**2 / p +
                (1.0 + 0.05 * h) * 2.0e-11 * f**1.5)

        return alpha * range_m


class BoundaryInteraction:
    """Acoustic boundary interactions (reflections)"""

    @staticmethod
    def surface_reflection_loss(frequency_khz: float,
                               grazing_angle_deg: float,
                               sea_state: int = 3) -> float:
        """Calculate surface reflection loss

        Args:
            frequency_khz: Frequency in kHz
            grazing_angle_deg: Grazing angle in degrees
            sea_state: Sea state (0-9)

        Returns:
            Reflection loss in dB
        """
        # At low grazing angles, surface roughness causes scattering loss
        # Simplified model

        theta = grazing_angle_deg
        ss = sea_state

        # At high frequencies and rough seas, significant loss
        if theta < 10:
            loss = 10 + ss * 2 + frequency_khz * 0.5
        else:
            loss = ss * 0.5 + frequency_khz * 0.1

        return min(loss, 30.0)  # Cap at 30 dB

    @staticmethod
    def bottom_reflection_loss(frequency_khz: float,
                              grazing_angle_deg: float,
                              bottom_type: str = 'mud') -> float:
        """Calculate bottom reflection loss

        Args:
            frequency_khz: Frequency in kHz
            grazing_angle_deg: Grazing angle in degrees
            bottom_type: 'mud', 'sand', 'gravel', 'rock'

        Returns:
            Reflection loss in dB
        """
        # Bottom loss depends on sediment type and angle
        # Simplified model

        theta = grazing_angle_deg

        # Typical loss values (empirical)
        if bottom_type == 'mud':
            base_loss = 5.0
        elif bottom_type == 'sand':
            base_loss = 3.0
        elif bottom_type == 'gravel':
            base_loss = 2.0
        elif bottom_type == 'rock':
            base_loss = 1.0
        else:
            base_loss = 5.0

        # Angle dependence
        if theta < 30:
            angle_factor = 2.0
        else:
            angle_factor = 1.0

        return base_loss * angle_factor


class AmbientNoise:
    """Ambient noise models for underwater and airborne environments"""

    @staticmethod
    def underwater_ambient_noise(frequency_khz: float,
                                sea_state: int = 3,
                                shipping_level: str = 'moderate',
                                wind_speed_kts: float = 10.0) -> float:
        """Calculate underwater ambient noise level

        Based on Wenz curves

        Args:
            frequency_khz: Frequency in kHz
            sea_state: Sea state (0-9)
            shipping_level: 'low', 'moderate', 'high'
            wind_speed_kts: Wind speed in knots

        Returns:
            Noise level in dB re 1 μPa/√Hz
        """
        f = frequency_khz

        # Shipping noise (dominant at low frequencies)
        if shipping_level == 'low':
            N_ship = 75 - 20 * np.log10(f)
        elif shipping_level == 'moderate':
            N_ship = 80 - 20 * np.log10(f)
        else:  # high
            N_ship = 85 - 20 * np.log10(f)

        # Wind-driven noise (dominant at mid-high frequencies)
        N_wind = 44 + np.sqrt(wind_speed_kts) + 17 * (3 - np.log10(f)) * (np.log10(f) - 2)

        # Thermal noise (dominant at very high frequencies)
        N_thermal = -15 + 20 * np.log10(f)

        # Combine sources (energetically)
        N_total_linear = (10**(N_ship/10) +
                         10**(N_wind/10) +
                         10**(N_thermal/10))

        N_total = 10 * np.log10(N_total_linear)

        return N_total

    @staticmethod
    def airborne_ambient_noise(environment: str = 'urban') -> float:
        """Typical airborne ambient noise levels

        Args:
            environment: 'quiet', 'suburban', 'urban', 'industrial'

        Returns:
            A-weighted sound pressure level in dBA
        """
        noise_levels = {
            'anechoic': 0.0,
            'quiet': 30.0,
            'suburban': 50.0,
            'urban': 70.0,
            'industrial': 85.0,
            'very_loud': 100.0
        }

        return noise_levels.get(environment, 60.0)


class SonarEquation:
    """Sonar equation calculations"""

    @staticmethod
    def passive_sonar_snr(source_level_db: float,
                         transmission_loss_db: float,
                         noise_level_db: float,
                         directivity_index_db: float = 0.0) -> float:
        """Calculate passive sonar SNR

        SE = SL - TL - (NL - DI)

        Args:
            source_level_db: Source level in dB re 1 μPa @ 1m
            transmission_loss_db: Transmission loss in dB
            noise_level_db: Noise level in dB re 1 μPa/√Hz
            directivity_index_db: Array directivity index in dB

        Returns:
            Signal excess (SNR) in dB
        """
        return source_level_db - transmission_loss_db - (noise_level_db - directivity_index_db)

    @staticmethod
    def active_sonar_snr(source_level_db: float,
                        transmission_loss_db: float,
                        target_strength_db: float,
                        noise_level_db: float,
                        directivity_index_db: float = 0.0) -> float:
        """Calculate active sonar SNR

        SE = SL - 2*TL + TS - (NL - DI)

        Args:
            source_level_db: Source level in dB re 1 μPa @ 1m
            transmission_loss_db: One-way transmission loss in dB
            target_strength_db: Target strength in dB
            noise_level_db: Noise level in dB re 1 μPa/√Hz
            directivity_index_db: Array directivity index in dB

        Returns:
            Signal excess (SNR) in dB
        """
        return (source_level_db - 2 * transmission_loss_db +
                target_strength_db - (noise_level_db - directivity_index_db))


class PropagationLoss:
    """Complete propagation loss calculations for sonar"""

    @staticmethod
    def underwater_transmission_loss(frequency_khz: float, range_km: float,
                                    conditions: WaterConditions,
                                    include_spreading: bool = True,
                                    spreading_law: str = 'spherical') -> Dict[str, float]:
        """Calculate underwater transmission loss

        Args:
            frequency_khz: Frequency in kHz
            range_km: Range in km
            conditions: Water conditions
            include_spreading: Include geometric spreading loss
            spreading_law: 'spherical', 'cylindrical', or 'practical'

        Returns:
            Dictionary with loss breakdown
        """
        # Spreading loss
        if include_spreading:
            if spreading_law == 'spherical':
                spreading_db = 20 * np.log10(range_km * 1000)
            elif spreading_law == 'cylindrical':
                spreading_db = 10 * np.log10(range_km * 1000)
            else:  # practical (mixed)
                # Spherical to 1 km, cylindrical beyond
                if range_km < 1:
                    spreading_db = 20 * np.log10(range_km * 1000)
                else:
                    spreading_db = 60 + 10 * np.log10(range_km)
        else:
            spreading_db = 0.0

        # Absorption loss
        absorption_db = UnderwaterAbsorption.francois_garrison_absorption(
            frequency_khz, conditions
        ) * range_km

        # Total
        total_db = spreading_db + absorption_db

        return {
            'spreading_loss_db': spreading_db,
            'absorption_loss_db': absorption_db,
            'total_loss_db': total_db,
            'breakdown': {
                'Spreading': spreading_db,
                'Absorption': absorption_db
            }
        }

    @staticmethod
    def airborne_transmission_loss(frequency_hz: float, range_m: float,
                                  conditions: AirConditions) -> Dict[str, float]:
        """Calculate airborne transmission loss

        Args:
            frequency_hz: Frequency in Hz
            range_m: Range in meters
            conditions: Air conditions

        Returns:
            Dictionary with loss breakdown
        """
        # Spherical spreading
        if range_m > 0:
            spreading_db = 20 * np.log10(range_m)
        else:
            spreading_db = 0.0

        # Atmospheric absorption
        absorption_db = AirborneAbsorption.atmospheric_absorption(
            frequency_hz, range_m, conditions
        )

        # Total
        total_db = spreading_db + absorption_db

        return {
            'spreading_loss_db': spreading_db,
            'absorption_loss_db': absorption_db,
            'total_loss_db': total_db
        }


# Convenience functions
def underwater_loss(frequency_khz: float, range_km: float,
                   temperature_c: float = 15.0,
                   salinity_ppt: float = 35.0) -> float:
    """Quick underwater transmission loss calculation

    Args:
        frequency_khz: Frequency in kHz
        range_km: Range in km
        temperature_c: Temperature in °C
        salinity_ppt: Salinity in ppt

    Returns:
        Total transmission loss in dB
    """
    conditions = WaterConditions(temperature_c=temperature_c, salinity_ppt=salinity_ppt)
    result = PropagationLoss.underwater_transmission_loss(frequency_khz, range_km, conditions)
    return result['total_loss_db']


def sound_speed_water(temperature_c: float = 15.0, depth_m: float = 0.0) -> float:
    """Quick sound speed calculation in water

    Args:
        temperature_c: Temperature in °C
        depth_m: Depth in meters

    Returns:
        Sound speed in m/s
    """
    return SoundSpeed.in_water(temperature_c, 35.0, depth_m)
