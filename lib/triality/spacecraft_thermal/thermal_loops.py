"""Pumped thermal loops and heat exchangers."""
import numpy as np
from dataclasses import dataclass

@dataclass
class PumpedLoop:
    length: float  # m
    diameter: float  # m
    mass_flow_rate: float  # kg/s
    
    def heat_transport(self, delta_T: float, c_p: float = 4186) -> float:
        """Heat transport capacity Q = ṁ·c_p·ΔT [W]"""
        return self.mass_flow_rate * c_p * delta_T

@dataclass
class HeatExchanger:
    UA: float  # Overall heat transfer coefficient × area [W/K]
    
    def effectiveness(self, C_min: float, C_max: float) -> float:
        """ε-NTU effectiveness"""
        NTU = self.UA / C_min
        C_r = C_min / C_max
        # Counter-flow
        if C_r < 1.0:
            eps = (1 - np.exp(-NTU*(1-C_r))) / (1 - C_r*np.exp(-NTU*(1-C_r)))
        else:
            eps = NTU / (1 + NTU)
        return min(eps, 1.0)

class NTU_Method:
    @staticmethod
    def heat_transfer(effectiveness: float, C_min: float, T_h_in: float, T_c_in: float) -> float:
        """Q = ε·C_min·(T_h,in - T_c,in) [W]"""
        return effectiveness * C_min * (T_h_in - T_c_in)
