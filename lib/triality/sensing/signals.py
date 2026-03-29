"""
Wave & Signal Fundamentals for Sensing Systems

Provides fundamental wave physics, signal characteristics, and basic calculations
for radar, lidar, and sonar systems.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# Physical constants
C_LIGHT = 299792458.0  # Speed of light (m/s)
C_SOUND_AIR = 343.0    # Speed of sound in air at 20°C (m/s)
C_SOUND_WATER = 1500.0 # Speed of sound in water (m/s)


@dataclass
class WaveParameters:
    """Basic wave parameters for any sensing modality"""
    frequency: float           # Hz
    wavelength: float          # m
    wave_speed: float          # m/s
    period: float              # s

    @classmethod
    def from_frequency(cls, frequency: float, medium: str = 'air', wave_speed: float = None):
        """Create wave parameters from frequency

        Args:
            frequency: Frequency in Hz
            medium: 'air', 'water', or 'vacuum' (for EM waves) - ignored if wave_speed provided
            wave_speed: Override wave speed in m/s (optional). If None, uses medium default.

        Returns:
            WaveParameters object
        """
        if wave_speed is None:
            if medium == 'vacuum' or medium == 'air_em':
                wave_speed = C_LIGHT
            elif medium == 'air':
                wave_speed = C_SOUND_AIR
            elif medium == 'water':
                wave_speed = C_SOUND_WATER
            else:
                raise ValueError(f"Unknown medium: {medium}")

        wavelength = wave_speed / frequency
        period = 1.0 / frequency

        return cls(
            frequency=frequency,
            wavelength=wavelength,
            wave_speed=wave_speed,
            period=period
        )

    @classmethod
    def from_wavelength(cls, wavelength: float, medium: str = 'air', wave_speed: float = None):
        """Create wave parameters from wavelength

        Args:
            wavelength: Wavelength in meters
            medium: 'air', 'water', or 'vacuum' - ignored if wave_speed provided
            wave_speed: Override wave speed in m/s (optional). If None, uses medium default.

        Returns:
            WaveParameters object
        """
        if wave_speed is None:
            if medium == 'vacuum' or medium == 'air_em':
                wave_speed = C_LIGHT
            elif medium == 'air':
                wave_speed = C_SOUND_AIR
            elif medium == 'water':
                wave_speed = C_SOUND_WATER
            else:
                raise ValueError(f"Unknown medium: {medium}")

        frequency = wave_speed / wavelength
        period = 1.0 / frequency

        return cls(
            frequency=frequency,
            wavelength=wavelength,
            wave_speed=wave_speed,
            period=period
        )


class SignalCharacteristics:
    """Signal characteristics for pulse-based sensing systems"""

    @staticmethod
    def pulse_bandwidth(pulse_duration: float) -> float:
        """Calculate bandwidth of a pulse

        Args:
            pulse_duration: Pulse duration in seconds

        Returns:
            Bandwidth in Hz (approximate, assuming rectangular pulse)
        """
        return 1.0 / pulse_duration

    @staticmethod
    def range_resolution(bandwidth: float, medium: str = 'vacuum') -> float:
        """Calculate range resolution from bandwidth

        Args:
            bandwidth: Signal bandwidth in Hz
            medium: Propagation medium ('vacuum', 'air', 'water')

        Returns:
            Range resolution in meters
        """
        if medium == 'vacuum' or medium == 'air_em':
            c = C_LIGHT
        elif medium == 'air':
            c = C_SOUND_AIR
        elif medium == 'water':
            c = C_SOUND_WATER
        else:
            raise ValueError(f"Unknown medium: {medium}")

        return c / (2.0 * bandwidth)

    @staticmethod
    def doppler_shift(velocity: float, frequency: float, medium: str = 'vacuum') -> float:
        """Calculate Doppler shift

        Args:
            velocity: Relative velocity in m/s (positive = approaching)
            frequency: Carrier frequency in Hz
            medium: Propagation medium

        Returns:
            Doppler shift in Hz
        """
        if medium == 'vacuum' or medium == 'air_em':
            c = C_LIGHT
        elif medium == 'air':
            c = C_SOUND_AIR
        elif medium == 'water':
            c = C_SOUND_WATER
        else:
            raise ValueError(f"Unknown medium: {medium}")

        # For velocities << c, use approximation
        return 2.0 * velocity * frequency / c

    @staticmethod
    def unambiguous_range(prf: float, medium: str = 'vacuum') -> float:
        """Calculate maximum unambiguous range

        Args:
            prf: Pulse repetition frequency in Hz
            medium: Propagation medium

        Returns:
            Maximum unambiguous range in meters
        """
        if medium == 'vacuum' or medium == 'air_em':
            c = C_LIGHT
        elif medium == 'air':
            c = C_SOUND_AIR
        elif medium == 'water':
            c = C_SOUND_WATER
        else:
            raise ValueError(f"Unknown medium: {medium}")

        return c / (2.0 * prf)

    @staticmethod
    def unambiguous_velocity(prf: float, wavelength: float) -> float:
        """Calculate maximum unambiguous velocity

        Args:
            prf: Pulse repetition frequency in Hz
            wavelength: Wavelength in meters

        Returns:
            Maximum unambiguous velocity in m/s
        """
        return prf * wavelength / 4.0

    @staticmethod
    def chirp_rate(bandwidth: float, pulse_duration: float) -> float:
        """Calculate chirp rate for linear FM pulse

        Args:
            bandwidth: Chirp bandwidth in Hz
            pulse_duration: Pulse duration in seconds

        Returns:
            Chirp rate in Hz/s
        """
        return bandwidth / pulse_duration

    @staticmethod
    def time_bandwidth_product(bandwidth: float, pulse_duration: float) -> float:
        """Calculate time-bandwidth product (processing gain)

        Args:
            bandwidth: Signal bandwidth in Hz
            pulse_duration: Pulse duration in seconds

        Returns:
            Time-bandwidth product (dimensionless)
        """
        return bandwidth * pulse_duration


class BeamCharacteristics:
    """Antenna/aperture beam characteristics"""

    @staticmethod
    def beamwidth_1d(wavelength: float, aperture_size: float) -> float:
        """Calculate 3dB beamwidth for 1D aperture

        Args:
            wavelength: Wavelength in meters
            aperture_size: Aperture size in meters

        Returns:
            Beamwidth in radians
        """
        # Simplified formula: θ ≈ λ/D (approximation for small angles)
        return wavelength / aperture_size

    @staticmethod
    def beamwidth_2d(wavelength: float, aperture_diameter: float) -> float:
        """Calculate 3dB beamwidth for circular aperture

        Args:
            wavelength: Wavelength in meters
            aperture_diameter: Aperture diameter in meters

        Returns:
            Beamwidth in radians
        """
        # For circular aperture: θ ≈ 1.22 * λ/D
        return 1.22 * wavelength / aperture_diameter

    @staticmethod
    def angular_resolution(wavelength: float, aperture_size: float) -> float:
        """Calculate angular resolution (Rayleigh criterion)

        Args:
            wavelength: Wavelength in meters
            aperture_size: Aperture size in meters

        Returns:
            Angular resolution in radians
        """
        return 1.22 * wavelength / aperture_size

    @staticmethod
    def cross_range_resolution(range_m: float, beamwidth_rad: float) -> float:
        """Calculate cross-range resolution at given range

        Args:
            range_m: Range to target in meters
            beamwidth_rad: Beamwidth in radians

        Returns:
            Cross-range resolution in meters
        """
        return range_m * beamwidth_rad

    @staticmethod
    def antenna_gain(wavelength: float, aperture_area: float, efficiency: float = 0.6) -> float:
        """Calculate antenna gain

        Args:
            wavelength: Wavelength in meters
            aperture_area: Physical aperture area in m²
            efficiency: Aperture efficiency (0-1), typical 0.5-0.7

        Returns:
            Gain (linear, not dB)
        """
        return (4.0 * np.pi * aperture_area * efficiency) / (wavelength ** 2)

    @staticmethod
    def effective_aperture(wavelength: float, gain: float) -> float:
        """Calculate effective aperture from gain

        Args:
            wavelength: Wavelength in meters
            gain: Antenna gain (linear, not dB)

        Returns:
            Effective aperture area in m²
        """
        return (gain * wavelength ** 2) / (4.0 * np.pi)


class RangeEquation:
    """Basic range equation calculations"""

    @staticmethod
    def two_way_time(range_m: float, medium: str = 'vacuum') -> float:
        """Calculate two-way propagation time

        Args:
            range_m: Range in meters
            medium: Propagation medium

        Returns:
            Two-way time in seconds
        """
        if medium == 'vacuum' or medium == 'air_em':
            c = C_LIGHT
        elif medium == 'air':
            c = C_SOUND_AIR
        elif medium == 'water':
            c = C_SOUND_WATER
        else:
            raise ValueError(f"Unknown medium: {medium}")

        return 2.0 * range_m / c

    @staticmethod
    def range_from_time(two_way_time: float, medium: str = 'vacuum') -> float:
        """Calculate range from two-way time

        Args:
            two_way_time: Two-way propagation time in seconds
            medium: Propagation medium

        Returns:
            Range in meters
        """
        if medium == 'vacuum' or medium == 'air_em':
            c = C_LIGHT
        elif medium == 'air':
            c = C_SOUND_AIR
        elif medium == 'water':
            c = C_SOUND_WATER
        else:
            raise ValueError(f"Unknown medium: {medium}")

        return c * two_way_time / 2.0

    @staticmethod
    def free_space_path_loss(range_m: float, wavelength: float) -> float:
        """Calculate free space path loss (two-way)

        Args:
            range_m: Range in meters
            wavelength: Wavelength in meters

        Returns:
            Path loss (linear, not dB)
        """
        # Two-way path loss: L = (4πR/λ)^4
        return ((4.0 * np.pi * range_m) / wavelength) ** 4

    @staticmethod
    def free_space_path_loss_db(range_m: float, wavelength: float) -> float:
        """Calculate free space path loss in dB (two-way)

        Args:
            range_m: Range in meters
            wavelength: Wavelength in meters

        Returns:
            Path loss in dB
        """
        loss_linear = RangeEquation.free_space_path_loss(range_m, wavelength)
        return 10.0 * np.log10(loss_linear)


class UtilityConversions:
    """Utility functions for common conversions"""

    @staticmethod
    def power_to_db(power_linear: float) -> float:
        """Convert linear power to dB"""
        return 10.0 * np.log10(power_linear)

    @staticmethod
    def db_to_power(power_db: float) -> float:
        """Convert dB to linear power"""
        return 10.0 ** (power_db / 10.0)

    @staticmethod
    def voltage_to_db(voltage_linear: float) -> float:
        """Convert linear voltage/field to dB"""
        return 20.0 * np.log10(voltage_linear)

    @staticmethod
    def db_to_voltage(voltage_db: float) -> float:
        """Convert dB to linear voltage/field"""
        return 10.0 ** (voltage_db / 20.0)

    @staticmethod
    def frequency_to_wavelength(frequency: float, medium: str = 'vacuum', wave_speed: float = None) -> float:
        """Convert frequency to wavelength

        Args:
            frequency: Frequency in Hz
            medium: Propagation medium - ignored if wave_speed provided
            wave_speed: Override wave speed in m/s (optional)
        """
        wave_params = WaveParameters.from_frequency(frequency, medium, wave_speed)
        return wave_params.wavelength

    @staticmethod
    def wavelength_to_frequency(wavelength: float, medium: str = 'vacuum', wave_speed: float = None) -> float:
        """Convert wavelength to frequency

        Args:
            wavelength: Wavelength in meters
            medium: Propagation medium - ignored if wave_speed provided
            wave_speed: Override wave speed in m/s (optional)
        """
        wave_params = WaveParameters.from_wavelength(wavelength, medium, wave_speed)
        return wave_params.frequency

    @staticmethod
    def degrees_to_radians(degrees: float) -> float:
        """Convert degrees to radians"""
        return degrees * np.pi / 180.0

    @staticmethod
    def radians_to_degrees(radians: float) -> float:
        """Convert radians to degrees"""
        return radians * 180.0 / np.pi

    @staticmethod
    def ghz_to_hz(ghz: float) -> float:
        """Convert GHz to Hz"""
        return ghz * 1e9

    @staticmethod
    def mhz_to_hz(mhz: float) -> float:
        """Convert MHz to Hz"""
        return mhz * 1e6

    @staticmethod
    def khz_to_hz(khz: float) -> float:
        """Convert kHz to Hz"""
        return khz * 1e3


# Convenience functions for quick calculations
def wavelength(frequency: float, medium: str = 'vacuum', wave_speed: float = None) -> float:
    """Quick wavelength calculation

    Args:
        frequency: Frequency in Hz
        medium: Propagation medium - ignored if wave_speed provided
        wave_speed: Override wave speed in m/s (optional). Allows exploration of "what if" scenarios.
                   Default None uses standard physics (c=3e8 m/s for vacuum, 343 m/s for air, etc.)

    Returns:
        Wavelength in meters

    Examples:
        >>> wavelength(10e9)  # 10 GHz in vacuum (default c=3e8 m/s)
        0.03
        >>> wavelength(10e9, wave_speed=5e8)  # Custom speed of light
        0.05
    """
    return UtilityConversions.frequency_to_wavelength(frequency, medium, wave_speed)


def frequency(wavelength: float, medium: str = 'vacuum', wave_speed: float = None) -> float:
    """Quick frequency calculation

    Args:
        wavelength: Wavelength in meters
        medium: Propagation medium - ignored if wave_speed provided
        wave_speed: Override wave speed in m/s (optional). Allows exploration of "what if" scenarios.
                   Default None uses standard physics (c=3e8 m/s for vacuum, 343 m/s for air, etc.)

    Returns:
        Frequency in Hz

    Examples:
        >>> frequency(0.03)  # Vacuum (default c=3e8 m/s)
        10000000000.0
        >>> frequency(0.03, wave_speed=5e8)  # Custom speed
        16666666666.67
    """
    return UtilityConversions.wavelength_to_frequency(wavelength, medium, wave_speed)


def beamwidth(wavelength: float, aperture: float, aperture_type: str = '1d') -> float:
    """Quick beamwidth calculation

    Args:
        wavelength: Wavelength in meters
        aperture: Aperture size/diameter in meters
        aperture_type: '1d' or '2d' (circular)

    Returns:
        Beamwidth in radians
    """
    if aperture_type == '1d':
        return BeamCharacteristics.beamwidth_1d(wavelength, aperture)
    elif aperture_type == '2d':
        return BeamCharacteristics.beamwidth_2d(wavelength, aperture)
    else:
        raise ValueError(f"Unknown aperture_type: {aperture_type}")


def doppler(velocity: float, frequency: float, medium: str = 'vacuum') -> float:
    """Quick Doppler shift calculation

    Args:
        velocity: Velocity in m/s (positive = approaching)
        frequency: Carrier frequency in Hz
        medium: Propagation medium

    Returns:
        Doppler shift in Hz
    """
    return SignalCharacteristics.doppler_shift(velocity, frequency, medium)
