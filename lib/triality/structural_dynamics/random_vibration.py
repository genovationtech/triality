"""Random vibration analysis using PSD."""
import numpy as np
from dataclasses import dataclass

@dataclass
class PSD:
    """Power Spectral Density"""
    frequencies_Hz: np.ndarray
    psd_values: np.ndarray  # g²/Hz or (m/s²)²/Hz
    
    def rms(self) -> float:
        """Calculate RMS from PSD: σ = √(∫ G(f) df)"""
        # Trapezoidal integration
        rms_squared = np.trapezoid(self.psd_values, self.frequencies_Hz)
        return np.sqrt(rms_squared)

class MilesEquation:
    """Miles' equation for random vibration response."""
    
    @staticmethod
    def response_rms(natural_freq_Hz: float, Q_factor: float, input_psd: float) -> float:
        """
        Calculate RMS response using Miles' equation.
        
        g_RMS = √(π/2 · f_n · Q · PSD(f_n))
        
        Args:
            natural_freq_Hz: Natural frequency [Hz]
            Q_factor: Quality factor (Q ≈ 1/(2ζ))
            input_psd: Input PSD at natural frequency [g²/Hz]
        
        Returns:
            RMS acceleration [g]
        """
        g_rms = np.sqrt(np.pi / 2 * natural_freq_Hz * Q_factor * input_psd)
        return g_rms
    
    @staticmethod
    def fatigue_damage(g_rms: float, duration_s: float, material_exponent_b: float = 6.0) -> float:
        """
        Estimate fatigue damage from random vibration.
        
        Simplified: D ∝ g_RMS^b · t
        
        Args:
            g_rms: RMS acceleration [g]
            duration_s: Test/mission duration [s]
            material_exponent_b: S-N curve exponent (6-10 for metals)
        
        Returns:
            Damage parameter
        """
        return (g_rms ** material_exponent_b) * duration_s

class RandomVibrationAnalyzer:
    """Analyze random vibration response."""
    
    def __init__(self, input_psd: PSD):
        self.input_psd = input_psd
    
    def sdof_response(self, natural_freq_Hz: float, damping_ratio: float = 0.05) -> float:
        """
        Calculate SDOF response to random vibration.
        
        Args:
            natural_freq_Hz: Natural frequency [Hz]
            damping_ratio: Damping ratio ζ
        
        Returns:
            RMS response [g]
        """
        # Quality factor
        Q = 1.0 / (2 * damping_ratio)
        
        # Find PSD value at natural frequency
        idx = np.argmin(np.abs(self.input_psd.frequencies_Hz - natural_freq_Hz))
        psd_at_fn = self.input_psd.psd_values[idx]
        
        # Miles' equation
        g_rms = MilesEquation.response_rms(natural_freq_Hz, Q, psd_at_fn)
        
        return g_rms
    
    def transmissibility(self, freq_ratio: float, damping_ratio: float = 0.05) -> float:
        """
        Transmissibility for vibration isolation.
        
        T = 1 / √[(1-r²)² + (2ζr)²]
        
        where r = f/f_n
        """
        r = freq_ratio
        zeta = damping_ratio
        
        T = 1.0 / np.sqrt((1 - r**2)**2 + (2*zeta*r)**2)
        
        return T
