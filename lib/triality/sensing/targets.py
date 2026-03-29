"""
Target Interaction Models

Radar cross-section (RCS), target strength (TS), and optical reflectivity
models for radar, sonar, and lidar systems.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class TargetGeometry:
    """Basic target geometry"""
    shape: str                    # 'sphere', 'cylinder', 'plate', 'complex'
    characteristic_dimension: float  # Characteristic size in meters
    aspect_angle: float = 0.0     # Aspect angle in degrees
    material: str = 'metal'       # Material type


class RadarCrossSection:
    """Radar Cross Section (RCS) models for various shapes"""

    @staticmethod
    def sphere(radius: float, wavelength: float) -> float:
        """RCS of a sphere

        Args:
            radius: Sphere radius in meters
            wavelength: Radar wavelength in meters

        Returns:
            RCS in m² (linear)
        """
        # Mie regime check
        ka = 2 * np.pi * radius / wavelength

        if ka > 1:  # Optical regime (ka >> 1)
            # Geometric optics: σ = π * r²
            return np.pi * radius ** 2
        else:
            # Rayleigh regime (ka << 1)
            # σ = (9π/4) * k⁴ * a⁶
            k = 2 * np.pi / wavelength
            return (9 * np.pi / 4) * (k ** 4) * (radius ** 6)

    @staticmethod
    def flat_plate(width: float, height: float, wavelength: float,
                   incident_angle: float = 0.0) -> float:
        """RCS of a flat rectangular plate (normal incidence)

        Args:
            width: Plate width in meters
            height: Plate height in meters
            wavelength: Radar wavelength in meters
            incident_angle: Angle from normal in degrees

        Returns:
            RCS in m²
        """
        # Area
        A = width * height

        # At normal incidence (or near-normal)
        if abs(incident_angle) < 10:
            # σ = (4π * A²) / λ²
            return (4 * np.pi * A ** 2) / (wavelength ** 2)
        else:
            # Off-normal: reduce by cos²(θ) approximation
            theta_rad = np.radians(incident_angle)
            return ((4 * np.pi * A ** 2) / (wavelength ** 2)) * (np.cos(theta_rad) ** 2)

    @staticmethod
    def cylinder(radius: float, length: float, wavelength: float,
                aspect_angle: float = 90.0) -> float:
        """RCS of a cylinder

        Args:
            radius: Cylinder radius in meters
            length: Cylinder length in meters
            wavelength: Radar wavelength in meters
            aspect_angle: Aspect angle in degrees (0=end-on, 90=broadside)

        Returns:
            RCS in m²
        """
        if abs(aspect_angle - 90) < 10:
            # Broadside: σ ≈ (2π * r * L²) / λ
            return (2 * np.pi * radius * length ** 2) / wavelength
        else:
            # End-on: treat as disk
            return RadarCrossSection.sphere(radius, wavelength)

    @staticmethod
    def corner_reflector(leg_length: float, wavelength: float) -> float:
        """RCS of a trihedral corner reflector

        Args:
            leg_length: Length of corner reflector legs in meters
            wavelength: Radar wavelength in meters

        Returns:
            RCS in m²
        """
        # σ = (12π * a⁴) / λ²  where a is leg length
        return (12 * np.pi * leg_length ** 4) / (wavelength ** 2)

    @staticmethod
    def aircraft_empirical(wingspan: float, length: float,
                          frequency_ghz: float,
                          aspect_angle: float = 0.0) -> float:
        """Empirical RCS estimate for aircraft

        Args:
            wingspan: Wingspan in meters
            length: Aircraft length in meters
            frequency_ghz: Frequency in GHz
            aspect_angle: Aspect angle in degrees (0=nose-on, 90=broadside)

        Returns:
            RCS in m² (approximate)
        """
        # Very rough empirical model
        wavelength = 3e8 / (frequency_ghz * 1e9)

        # Characteristic area
        A_char = wingspan * length

        # Baseline RCS (broadside)
        sigma_broadside = 0.01 * A_char ** 2 / wavelength

        # Aspect angle variation (rough model)
        if abs(aspect_angle) < 30:
            # Nose-on: much smaller
            sigma = sigma_broadside * 0.01
        elif abs(aspect_angle - 90) < 30:
            # Broadside: maximum
            sigma = sigma_broadside
        else:
            # Intermediate
            theta_rad = np.radians(aspect_angle)
            sigma = sigma_broadside * (0.01 + 0.99 * abs(np.sin(theta_rad)))

        return sigma

    @staticmethod
    def vehicle_empirical(length: float, width: float, height: float,
                         frequency_ghz: float) -> float:
        """Empirical RCS estimate for ground vehicles

        Args:
            length: Vehicle length in meters
            width: Vehicle width in meters
            height: Vehicle height in meters
            frequency_ghz: Frequency in GHz

        Returns:
            RCS in m² (approximate)
        """
        # Rough empirical model
        # Typical car/truck RCS ~ 100-200 m² at X-band

        volume = length * width * height
        wavelength = 3e8 / (frequency_ghz * 1e9)

        # Empirical scaling
        sigma = 50 * (volume / 10.0) ** 0.5

        return sigma

    @staticmethod
    def rcs_to_db(rcs_m2: float) -> float:
        """Convert RCS to dBsm (dB relative to 1 m²)

        Args:
            rcs_m2: RCS in m²

        Returns:
            RCS in dBsm
        """
        if rcs_m2 > 0:
            return 10 * np.log10(rcs_m2)
        else:
            return -100.0  # Very small RCS

    @staticmethod
    def db_to_rcs(rcs_dbsm: float) -> float:
        """Convert dBsm to RCS in m²

        Args:
            rcs_dbsm: RCS in dBsm

        Returns:
            RCS in m²
        """
        return 10 ** (rcs_dbsm / 10.0)


class TargetStrength:
    """Target Strength (TS) models for sonar targets"""

    @staticmethod
    def sphere_ts(radius: float, wavelength: float,
                  density_ratio: float = 7.8) -> float:
        """Target strength of a sphere (elastic sphere model)

        Args:
            radius: Sphere radius in meters
            wavelength: Acoustic wavelength in meters
            density_ratio: Density ratio (sphere/water), e.g., 7.8 for steel

        Returns:
            Target strength in dB re 1 m²
        """
        # ka parameter
        ka = 2 * np.pi * radius / wavelength

        if ka < 0.5:
            # Rayleigh scattering regime
            # TS ≈ 20 log(ka) + 20 log(a) - 8 dB (approximate)
            ts = 20 * np.log10(ka) + 20 * np.log10(radius) - 8
        else:
            # Geometric regime (high frequency)
            # TS ≈ 10 log(π * a²) = 10 log(π) + 20 log(a)
            ts = 10 * np.log10(np.pi) + 20 * np.log10(radius)

        return ts

    @staticmethod
    def cylinder_ts(radius: float, length: float, wavelength: float,
                   aspect_angle: float = 90.0) -> float:
        """Target strength of a cylinder

        Args:
            radius: Cylinder radius in meters
            length: Cylinder length in meters
            wavelength: Acoustic wavelength in meters
            aspect_angle: Aspect angle in degrees (90=broadside)

        Returns:
            Target strength in dB re 1 m²
        """
        if abs(aspect_angle - 90) < 10:
            # Broadside incidence
            # TS ≈ 10 log(2π * a * L² / λ)
            rcs_equiv = (2 * np.pi * radius * length ** 2) / wavelength
            ts = 10 * np.log10(rcs_equiv)
        else:
            # End-on: use sphere approximation
            ts = TargetStrength.sphere_ts(radius, wavelength)

        return ts

    @staticmethod
    def submarine_empirical(length: float, frequency_khz: float) -> float:
        """Empirical target strength for submarine

        Args:
            length: Submarine length in meters
            frequency_khz: Frequency in kHz

        Returns:
            Target strength in dB re 1 m²
        """
        # Rough empirical model
        # Typical submarine TS ~ 15-30 dB depending on aspect and frequency

        wavelength = 1500 / (frequency_khz * 1000)  # Assume c = 1500 m/s

        # Baseline: treat as cylinder
        radius_est = length / 10  # Rough estimate
        ts_broadside = TargetStrength.cylinder_ts(radius_est, length, wavelength, 90)

        return ts_broadside

    @staticmethod
    def fish_empirical(length_cm: float, frequency_khz: float) -> float:
        """Empirical target strength for fish

        Based on Love (1971) and Foote (1987) models

        Args:
            length_cm: Fish length in cm
            frequency_khz: Frequency in kHz

        Returns:
            Target strength in dB re 1 m²
        """
        L = length_cm

        # Love's equation (for fish with swim bladder)
        # TS = 19.1 log(L) - 0.9 log(f) - 62
        f = frequency_khz

        ts = 19.1 * np.log10(L) - 0.9 * np.log10(f) - 62

        return ts


class OpticalReflectivity:
    """Optical reflectivity for lidar systems"""

    @staticmethod
    def lambertian_reflectance(albedo: float, incident_angle: float = 0.0) -> float:
        """Lambertian (diffuse) surface reflectance

        Args:
            albedo: Surface albedo (0-1)
            incident_angle: Incident angle from normal in degrees

        Returns:
            Reflectance (0-1)
        """
        # Lambertian surface: reflectance = albedo * cos(θ)
        theta_rad = np.radians(incident_angle)
        return albedo * np.cos(theta_rad)

    @staticmethod
    def specular_reflectance(material: str, incident_angle: float = 0.0,
                            wavelength_nm: float = 1550) -> float:
        """Specular (mirror-like) surface reflectance

        Args:
            material: Material type ('metal', 'glass', 'water')
            incident_angle: Incident angle from normal in degrees
            wavelength_nm: Wavelength in nanometers

        Returns:
            Reflectance (0-1)
        """
        # Fresnel reflection (simplified)
        theta_rad = np.radians(incident_angle)

        # Refractive indices (approximate, wavelength-dependent in reality)
        n_dict = {
            'air': 1.0,
            'water': 1.33,
            'glass': 1.5,
            'metal': 100.0  # High for metals (opaque)
        }

        n2 = n_dict.get(material, 1.5)
        n1 = 1.0  # Air

        # Fresnel equations (simplified for normal incidence)
        if abs(incident_angle) < 5:
            # Normal incidence
            R = ((n1 - n2) / (n1 + n2)) ** 2
        else:
            # Fresnel equations for s-polarization (approximate)
            cos_theta1 = np.cos(theta_rad)
            sin_theta1 = np.sin(theta_rad)
            sin_theta2 = (n1 / n2) * sin_theta1

            if abs(sin_theta2) > 1:
                # Total internal reflection
                R = 1.0
            else:
                cos_theta2 = np.sqrt(1 - sin_theta2 ** 2)
                Rs = ((n1 * cos_theta1 - n2 * cos_theta2) /
                     (n1 * cos_theta1 + n2 * cos_theta2)) ** 2
                R = Rs

        return R

    @staticmethod
    def terrain_albedo(terrain_type: str) -> float:
        """Typical albedo for terrain types

        Args:
            terrain_type: Type of terrain

        Returns:
            Albedo (0-1)
        """
        albedos = {
            'asphalt': 0.12,
            'concrete': 0.30,
            'grass': 0.25,
            'forest': 0.15,
            'snow': 0.80,
            'water': 0.06,
            'desert_sand': 0.40,
            'building': 0.35,
            'metal': 0.70
        }

        return albedos.get(terrain_type, 0.20)

    @staticmethod
    def atmospheric_backscatter_coefficient(visibility_km: float,
                                           wavelength_nm: float = 1550) -> float:
        """Atmospheric backscatter coefficient for lidar

        Args:
            visibility_km: Visibility in km
            wavelength_nm: Wavelength in nanometers

        Returns:
            Backscatter coefficient in m⁻¹ sr⁻¹
        """
        # Simplified model based on visibility
        # β ≈ 3.91 / V * (λ / 550)^(-q)  where q ≈ 0.585 * V^(1/3)

        V = visibility_km
        wavelength_ratio = wavelength_nm / 550.0

        q = 0.585 * (V ** (1.0 / 3.0))

        beta = (3.91 / V) * (wavelength_ratio ** (-q))

        return beta * 1e-3  # Convert to m⁻¹ sr⁻¹


class TargetModels:
    """Combined target models for different sensor types"""

    @staticmethod
    def get_radar_rcs(target_type: str, frequency_ghz: float,
                     aspect_angle: float = 0.0) -> float:
        """Get typical RCS for common targets

        Args:
            target_type: Target type (e.g., 'car', 'truck', 'aircraft', 'person')
            frequency_ghz: Frequency in GHz
            aspect_angle: Aspect angle in degrees

        Returns:
            RCS in m²
        """
        wavelength = 3e8 / (frequency_ghz * 1e9)

        if target_type == 'person':
            return RadarCrossSection.cylinder(0.15, 1.7, wavelength, aspect_angle)
        elif target_type == 'car':
            return RadarCrossSection.vehicle_empirical(4.5, 1.8, 1.5, frequency_ghz)
        elif target_type == 'truck':
            return RadarCrossSection.vehicle_empirical(10, 2.5, 3.0, frequency_ghz)
        elif target_type == 'small_aircraft':
            return RadarCrossSection.aircraft_empirical(10, 8, frequency_ghz, aspect_angle)
        elif target_type == 'large_aircraft':
            return RadarCrossSection.aircraft_empirical(60, 70, frequency_ghz, aspect_angle)
        elif target_type == 'bird':
            return RadarCrossSection.sphere(0.05, wavelength)
        elif target_type == 'drone':
            return RadarCrossSection.sphere(0.2, wavelength)
        else:
            return 1.0  # Default 1 m²

    @staticmethod
    def get_sonar_ts(target_type: str, frequency_khz: float) -> float:
        """Get typical target strength for common sonar targets

        Args:
            target_type: Target type (e.g., 'submarine', 'torpedo', 'mine', 'fish')
            frequency_khz: Frequency in kHz

        Returns:
            Target strength in dB re 1 m²
        """
        wavelength = 1500 / (frequency_khz * 1000)

        if target_type == 'submarine':
            return TargetStrength.submarine_empirical(100, frequency_khz)
        elif target_type == 'torpedo':
            return TargetStrength.cylinder_ts(0.25, 5, wavelength, 90)
        elif target_type == 'mine':
            return TargetStrength.sphere_ts(0.5, wavelength, 7.8)
        elif target_type == 'small_fish':
            return TargetStrength.fish_empirical(10, frequency_khz)  # 10 cm
        elif target_type == 'large_fish':
            return TargetStrength.fish_empirical(50, frequency_khz)  # 50 cm
        elif target_type == 'whale':
            return TargetStrength.fish_empirical(1000, frequency_khz)  # 10 m
        else:
            return 0.0  # Default 0 dB (1 m²)

    @staticmethod
    def get_lidar_reflectance(surface_type: str, incident_angle: float = 0.0) -> float:
        """Get typical reflectance for lidar surfaces

        Args:
            surface_type: Surface type
            incident_angle: Incident angle in degrees

        Returns:
            Reflectance (0-1)
        """
        albedo = OpticalReflectivity.terrain_albedo(surface_type)
        return OpticalReflectivity.lambertian_reflectance(albedo, incident_angle)


# Convenience functions
def rcs(target_type: str, frequency_ghz: float = 10.0) -> float:
    """Quick RCS lookup

    Args:
        target_type: Target type
        frequency_ghz: Frequency in GHz

    Returns:
        RCS in m²
    """
    return TargetModels.get_radar_rcs(target_type, frequency_ghz)


def target_strength(target_type: str, frequency_khz: float = 50.0) -> float:
    """Quick target strength lookup

    Args:
        target_type: Target type
        frequency_khz: Frequency in kHz

    Returns:
        Target strength in dB
    """
    return TargetModels.get_sonar_ts(target_type, frequency_khz)


def reflectance(surface_type: str) -> float:
    """Quick reflectance lookup

    Args:
        surface_type: Surface type

    Returns:
        Reflectance (0-1)
    """
    return TargetModels.get_lidar_reflectance(surface_type)
