"""Heat pipe models for spacecraft thermal control."""
import numpy as np
from dataclasses import dataclass
from enum import Enum

class HeatPipeRegime(Enum):
    NORMAL = "normal"
    CAPILLARY_LIMITED = "capillary_limited"
    SONIC_LIMITED = "sonic_limited"

@dataclass
class HeatPipe:
    length: float  # m
    diameter: float  # m
    working_fluid: str = "ammonia"
    
    def effective_conductivity(self, temperature: float = 300) -> float:
        """Effective thermal conductivity [W/(m·K)]"""
        # Simplified model: k_eff >> k_metal
        return 50000.0  # Typical for ammonia heat pipe

class CapillaryLimit:
    @staticmethod
    def max_heat_transport(wick_permeability: float, length: float) -> float:
        """Maximum heat transport before capillary limit [W]"""
        # Q_max ≈ (K·ρ·h_fg) / (μ·L)
        return 1000.0  # Placeholder

class SonicLimit:
    @staticmethod
    def check_sonic_limit(heat_flux: float, vapor_velocity: float) -> bool:
        """Check if sonic limit exceeded"""
        return vapor_velocity < 340  # m/s (simplified)
