"""
Noise, Clutter, and Interference Models

Thermal noise, clutter statistics, and interference calculations for
radar, lidar, and sonar systems.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


# Physical constants
BOLTZMANN_K = 1.380649e-23  # Boltzmann constant (J/K)


@dataclass
class NoiseParameters:
    """Noise parameters for sensing systems"""
    temperature_k: float = 290.0      # System noise temperature (K)
    bandwidth_hz: float = 1e6         # Receiver bandwidth (Hz)
    noise_figure_db: float = 3.0      # Receiver noise figure (dB)


class ThermalNoise:
    """Thermal noise calculations"""

    @staticmethod
    def noise_power(temperature_k: float, bandwidth_hz: float) -> float:
        """Calculate thermal noise power (Nyquist-Johnson noise)

        P_n = k * T * B

        Args:
            temperature_k: System temperature in Kelvin
            bandwidth_hz: Bandwidth in Hz

        Returns:
            Noise power in Watts
        """
        return BOLTZMANN_K * temperature_k * bandwidth_hz

    @staticmethod
    def noise_power_dbm(temperature_k: float, bandwidth_hz: float) -> float:
        """Calculate thermal noise power in dBm

        Args:
            temperature_k: System temperature in Kelvin
            bandwidth_hz: Bandwidth in Hz

        Returns:
            Noise power in dBm
        """
        P_watts = ThermalNoise.noise_power(temperature_k, bandwidth_hz)
        P_dbm = 10 * np.log10(P_watts * 1000)  # Convert to dBm
        return P_dbm

    @staticmethod
    def system_noise_temperature(receiver_temp_k: float, noise_figure_db: float) -> float:
        """Calculate effective system noise temperature

        T_sys = T_receiver + T_noise_figure

        where T_noise_figure = T_0 * (F - 1)
        and T_0 = 290 K, F is noise figure (linear)

        Args:
            receiver_temp_k: Receiver physical temperature in Kelvin
            noise_figure_db: Noise figure in dB

        Returns:
            System noise temperature in Kelvin
        """
        T_0 = 290.0  # Reference temperature
        F_linear = 10 ** (noise_figure_db / 10)
        T_nf = T_0 * (F_linear - 1)

        return receiver_temp_k + T_nf

    @staticmethod
    def noise_equivalent_power(bandwidth_hz: float, noise_figure_db: float = 3.0) -> float:
        """Calculate noise equivalent power (NEP)

        Args:
            bandwidth_hz: Bandwidth in Hz
            noise_figure_db: Noise figure in dB

        Returns:
            NEP in Watts
        """
        T_sys = ThermalNoise.system_noise_temperature(290.0, noise_figure_db)
        return ThermalNoise.noise_power(T_sys, bandwidth_hz)


class ClutterModels:
    """Clutter models for radar and lidar"""

    @staticmethod
    def surface_clutter_rcs(area_m2: float, sigma_0_db: float) -> float:
        """Calculate clutter RCS from surface area

        σ_clutter = σ_0 * A

        Args:
            area_m2: Clutter patch area in m²
            sigma_0_db: Normalized radar cross-section (σ₀) in dB

        Returns:
            Clutter RCS in m²
        """
        sigma_0_linear = 10 ** (sigma_0_db / 10)
        return sigma_0_linear * area_m2

    @staticmethod
    def clutter_area(range_m: float, range_resolution_m: float,
                    azimuth_beamwidth_rad: float) -> float:
        """Calculate clutter patch area

        For surface clutter: A = R * ΔR * R * θ_az

        Args:
            range_m: Range to clutter in meters
            range_resolution_m: Range resolution in meters
            azimuth_beamwidth_rad: Azimuth beamwidth in radians

        Returns:
            Clutter area in m²
        """
        return range_m * range_resolution_m * range_m * azimuth_beamwidth_rad

    @staticmethod
    def sigma_0_land(terrain_type: str, frequency_ghz: float,
                    grazing_angle_deg: float = 30.0) -> float:
        """Typical σ₀ values for land clutter

        Args:
            terrain_type: Type of terrain
            frequency_ghz: Frequency in GHz
            grazing_angle_deg: Grazing angle in degrees

        Returns:
            σ₀ in dB
        """
        # Typical values at X-band (10 GHz), 30° grazing angle
        # These vary significantly with frequency and angle

        sigma_0_baseline = {
            'urban': -5.0,
            'suburban': -10.0,
            'farmland': -15.0,
            'forest': -20.0,
            'grassland': -25.0,
            'desert': -30.0,
            'ice': -25.0,
            'snow': -30.0
        }

        baseline = sigma_0_baseline.get(terrain_type, -20.0)

        # Frequency correction (rough)
        freq_factor = 10 * np.log10(frequency_ghz / 10.0)

        # Grazing angle correction (rough)
        if grazing_angle_deg < 10:
            angle_factor = -10
        elif grazing_angle_deg < 30:
            angle_factor = -5
        else:
            angle_factor = 0

        return baseline + freq_factor + angle_factor

    @staticmethod
    def sigma_0_sea(sea_state: int, frequency_ghz: float,
                   grazing_angle_deg: float = 30.0,
                   polarization: str = 'hh') -> float:
        """σ₀ for sea clutter (simplified model)

        Args:
            sea_state: Sea state (0-9)
            frequency_ghz: Frequency in GHz
            grazing_angle_deg: Grazing angle in degrees
            polarization: 'hh' (horizontal-horizontal) or 'vv' (vertical-vertical)

        Returns:
            σ₀ in dB
        """
        # Very simplified model
        # Sea clutter is complex and depends on many factors

        # Baseline at sea state 3, 10 GHz, 30° grazing
        baseline = -40.0

        # Sea state correction
        ss_factor = 3.0 * sea_state

        # Frequency correction
        freq_factor = 10 * np.log10(frequency_ghz / 10.0)

        # Polarization (VV is typically higher than HH)
        if polarization == 'vv':
            pol_factor = 3.0
        else:
            pol_factor = 0.0

        return baseline + ss_factor + freq_factor + pol_factor

    @staticmethod
    def volume_clutter_coefficient(clutter_type: str) -> float:
        """Volume clutter coefficient (η) for rain, chaff, etc.

        Args:
            clutter_type: Type of volume clutter

        Returns:
            η in m⁻¹ (linear)
        """
        # Typical values
        eta_dict = {
            'light_rain': 1e-8,
            'moderate_rain': 1e-7,
            'heavy_rain': 1e-6,
            'chaff': 1e-5,
            'birds': 1e-9,
            'insects': 1e-10
        }

        return eta_dict.get(clutter_type, 1e-9)


class ClutterStatistics:
    """Statistical models for clutter"""

    @staticmethod
    def rayleigh_clutter_pdf(amplitude: float, sigma: float) -> float:
        """Rayleigh probability density function for clutter amplitude

        Used for simple clutter (e.g., sea clutter in certain conditions)

        Args:
            amplitude: Clutter amplitude
            sigma: RMS amplitude

        Returns:
            PDF value
        """
        if amplitude < 0:
            return 0.0
        return (amplitude / sigma**2) * np.exp(-amplitude**2 / (2 * sigma**2))

    @staticmethod
    def weibull_clutter_pdf(amplitude: float, scale: float, shape: float) -> float:
        """Weibull probability density function for clutter

        Used for more complex clutter (land clutter, sea clutter in high sea states)

        Args:
            amplitude: Clutter amplitude
            scale: Scale parameter
            shape: Shape parameter (c)

        Returns:
            PDF value
        """
        if amplitude < 0:
            return 0.0
        c = shape
        a = scale
        return (c / a) * (amplitude / a)**(c - 1) * np.exp(-(amplitude / a)**c)

    @staticmethod
    def k_distribution_clutter(shape_param: float) -> Dict[str, float]:
        """K-distribution parameters for sea clutter

        Args:
            shape_param: Shape parameter (ν)

        Returns:
            Dictionary with distribution parameters
        """
        # K-distribution is commonly used for sea clutter
        # Characterized by shape parameter ν
        # ν → ∞: Rayleigh
        # ν < 1: Very spiky clutter
        # ν = 0.5-2: Typical sea clutter

        return {
            'shape': shape_param,
            'distribution': 'K-distribution',
            'spikiness': 'high' if shape_param < 1 else 'moderate' if shape_param < 5 else 'low'
        }


class InterferenceModels:
    """Interference from other emitters"""

    @staticmethod
    def jamming_power_received(jammer_power_w: float, jammer_gain_db: float,
                              range_km: float, frequency_ghz: float,
                              receiver_gain_db: float) -> float:
        """Calculate received jamming power

        Args:
            jammer_power_w: Jammer transmit power in Watts
            jammer_gain_db: Jammer antenna gain in dB
            range_km: Range to jammer in km
            frequency_ghz: Frequency in GHz
            receiver_gain_db: Receiver antenna gain in dB

        Returns:
            Received jamming power in Watts
        """
        # Free space path loss (one-way)
        wavelength = 3e8 / (frequency_ghz * 1e9)
        fspl = (4 * np.pi * range_km * 1000 / wavelength) ** 2

        # Convert gains to linear
        G_j = 10 ** (jammer_gain_db / 10)
        G_r = 10 ** (receiver_gain_db / 10)

        # Received power
        P_j_received = jammer_power_w * G_j * G_r / fspl

        return P_j_received

    @staticmethod
    def jamming_to_signal_ratio(jammer_power_received: float,
                                signal_power_received: float) -> float:
        """Calculate jamming-to-signal ratio (JSR)

        Args:
            jammer_power_received: Received jammer power in Watts
            signal_power_received: Received signal power in Watts

        Returns:
            JSR in dB
        """
        if signal_power_received <= 0:
            return 100.0  # Very high JSR

        jsr_linear = jammer_power_received / signal_power_received
        return 10 * np.log10(jsr_linear)

    @staticmethod
    def self_interference_power(tx_power_w: float, isolation_db: float) -> float:
        """Calculate self-interference power (e.g., in FMCW radar)

        Args:
            tx_power_w: Transmit power in Watts
            isolation_db: TX-RX isolation in dB

        Returns:
            Self-interference power in Watts
        """
        isolation_linear = 10 ** (isolation_db / 10)
        return tx_power_w / isolation_linear


class SignalToNoiseRatio:
    """SNR calculations including all noise sources"""

    @staticmethod
    def calculate_snr(signal_power_w: float, noise_params: NoiseParameters,
                     clutter_power_w: float = 0.0,
                     interference_power_w: float = 0.0) -> float:
        """Calculate SNR including thermal noise, clutter, and interference

        SNR = S / (N + C + I)

        Args:
            signal_power_w: Signal power in Watts
            noise_params: Noise parameters
            clutter_power_w: Clutter power in Watts
            interference_power_w: Interference power in Watts

        Returns:
            SNR in dB
        """
        # Thermal noise power
        T_sys = ThermalNoise.system_noise_temperature(
            noise_params.temperature_k, noise_params.noise_figure_db
        )
        N = ThermalNoise.noise_power(T_sys, noise_params.bandwidth_hz)

        # Total noise + clutter + interference
        total_noise = N + clutter_power_w + interference_power_w

        if total_noise <= 0:
            return 100.0  # Very high SNR

        snr_linear = signal_power_w / total_noise
        return 10 * np.log10(snr_linear)

    @staticmethod
    def calculate_sinr(signal_power_w: float, noise_power_w: float,
                      interference_power_w: float) -> float:
        """Calculate signal-to-interference-plus-noise ratio (SINR)

        Args:
            signal_power_w: Signal power in Watts
            noise_power_w: Noise power in Watts
            interference_power_w: Interference power in Watts

        Returns:
            SINR in dB
        """
        total = noise_power_w + interference_power_w
        if total <= 0:
            return 100.0

        sinr_linear = signal_power_w / total
        return 10 * np.log10(sinr_linear)

    @staticmethod
    def calculate_scnr(signal_power_w: float, clutter_power_w: float,
                      noise_power_w: float) -> float:
        """Calculate signal-to-clutter-plus-noise ratio (SCNR)

        Args:
            signal_power_w: Signal power in Watts
            clutter_power_w: Clutter power in Watts
            noise_power_w: Noise power in Watts

        Returns:
            SCNR in dB
        """
        total = clutter_power_w + noise_power_w
        if total <= 0:
            return 100.0

        scnr_linear = signal_power_w / total
        return 10 * np.log10(scnr_linear)


class DopplerClutter:
    """Doppler clutter analysis (for MTI/MTD radars)"""

    @staticmethod
    def clutter_doppler_spread(platform_velocity: float, beamwidth_rad: float,
                              wavelength: float) -> float:
        """Calculate clutter Doppler spread

        Args:
            platform_velocity: Platform velocity in m/s
            beamwidth_rad: Antenna beamwidth in radians
            wavelength: Wavelength in meters

        Returns:
            Doppler spread in Hz
        """
        # For airborne/spaceborne radar looking down
        # Δf_d ≈ (2 * v * θ) / λ

        return (2 * platform_velocity * beamwidth_rad) / wavelength

    @staticmethod
    def minimum_detectable_velocity(clutter_attenuation_db: float,
                                   prf: float, wavelength: float) -> float:
        """Calculate minimum detectable velocity (MDV)

        Args:
            clutter_attenuation_db: MTI/MTD clutter attenuation in dB
            prf: Pulse repetition frequency in Hz
            wavelength: Wavelength in meters

        Returns:
            MDV in m/s
        """
        # Simplified: MDV corresponds to clutter notch width
        # Δf ≈ PRF / (CA)  where CA is clutter attenuation (linear)

        CA_linear = 10 ** (clutter_attenuation_db / 10)
        delta_f = prf / CA_linear

        # Convert to velocity
        mdv = (delta_f * wavelength) / 2

        return mdv


# Convenience functions
def thermal_noise_dbm(bandwidth_mhz: float, noise_figure_db: float = 3.0) -> float:
    """Quick thermal noise calculation in dBm

    Args:
        bandwidth_mhz: Bandwidth in MHz
        noise_figure_db: Noise figure in dB

    Returns:
        Noise power in dBm
    """
    T_sys = ThermalNoise.system_noise_temperature(290.0, noise_figure_db)
    return ThermalNoise.noise_power_dbm(T_sys, bandwidth_mhz * 1e6)


def clutter_rcs(terrain_type: str, range_m: float, range_res_m: float,
               beamwidth_deg: float, frequency_ghz: float = 10.0) -> float:
    """Quick clutter RCS calculation

    Args:
        terrain_type: Terrain type
        range_m: Range in meters
        range_res_m: Range resolution in meters
        beamwidth_deg: Beamwidth in degrees
        frequency_ghz: Frequency in GHz

    Returns:
        Clutter RCS in m²
    """
    beamwidth_rad = np.radians(beamwidth_deg)
    area = ClutterModels.clutter_area(range_m, range_res_m, beamwidth_rad)
    sigma_0_db = ClutterModels.sigma_0_land(terrain_type, frequency_ghz)
    return ClutterModels.surface_clutter_rcs(area, sigma_0_db)
