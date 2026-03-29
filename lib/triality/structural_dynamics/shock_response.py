"""Shock Response Spectrum (SRS) analysis."""
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class ShockResponseSpectrum:
    """SRS: Maximum response vs natural frequency"""
    frequencies_Hz: np.ndarray
    max_acceleration: np.ndarray  # g or m/s²
    damping_ratio: float = 0.05

class SRSCalculator:
    """Calculate Shock Response Spectrum from time history."""
    
    @staticmethod
    def sdof_response(time: np.ndarray, acceleration: np.ndarray,
                     natural_freq_Hz: float, damping_ratio: float = 0.05) -> float:
        """
        Calculate maximum response of SDOF to shock input.
        
        Solves: ẍ + 2ζω_nẋ + ω_n²x = -a(t)
        
        Args:
            time: Time array [s]
            acceleration: Base acceleration [g or m/s²]
            natural_freq_Hz: Natural frequency [Hz]
            damping_ratio: Damping ratio
        
        Returns:
            Maximum absolute acceleration [same units as input]
        """
        omega_n = 2 * np.pi * natural_freq_Hz
        omega_d = omega_n * np.sqrt(1 - damping_ratio**2)
        
        # Duhamel integral (convolution)
        # x(t) = ∫ a(τ)·h(t-τ) dτ
        # where h(t) = exp(-ζω_n·t)·sin(ω_d·t) / ω_d
        
        dt = time[1] - time[0]
        n = len(time)
        
        # Impulse response
        t_impulse = time
        h = np.exp(-damping_ratio * omega_n * t_impulse) * np.sin(omega_d * t_impulse) / omega_d
        h[0] = 0  # Initial condition
        
        # Convolution
        response = np.convolve(acceleration, h, mode='full')[:n] * dt
        
        # Acceleration response: ẍ = -ω_n²·x - 2ζω_n·ẋ
        # Simplified: take max of relative acceleration
        max_response = np.max(np.abs(response)) * omega_n**2
        
        return max_response
    
    @staticmethod
    def compute_srs(time: np.ndarray, acceleration: np.ndarray,
                   freq_range_Hz: np.ndarray, damping_ratio: float = 0.05) -> ShockResponseSpectrum:
        """
        Compute SRS over frequency range.
        
        Args:
            time: Time array [s]
            acceleration: Base acceleration time history
            freq_range_Hz: Frequencies to evaluate [Hz]
            damping_ratio: Damping ratio
        
        Returns:
            ShockResponseSpectrum object
        """
        srs_values = np.zeros(len(freq_range_Hz))
        
        for i, fn in enumerate(freq_range_Hz):
            srs_values[i] = SRSCalculator.sdof_response(
                time, acceleration, fn, damping_ratio
            )
        
        return ShockResponseSpectrum(
            frequencies_Hz=freq_range_Hz,
            max_acceleration=srs_values,
            damping_ratio=damping_ratio
        )
    
    @staticmethod
    def haversine_pulse(amplitude: float, duration: float, time_array: np.ndarray) -> np.ndarray:
        """
        Generate haversine pulse for testing.
        
        a(t) = A·sin²(πt/T) for 0 ≤ t ≤ T
        """
        pulse = np.zeros_like(time_array)
        mask = (time_array >= 0) & (time_array <= duration)
        pulse[mask] = amplitude * np.sin(np.pi * time_array[mask] / duration)**2
        
        return pulse
