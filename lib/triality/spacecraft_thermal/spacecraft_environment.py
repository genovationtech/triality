"""Spacecraft environmental heating (solar, albedo, planetary IR)."""
import numpy as np
from dataclasses import dataclass
from enum import Enum

class Orbit(Enum):
    LEO = "low_earth_orbit"
    GEO = "geostationary"
    LUNAR = "lunar"
    MARS = "mars"

class SolarFlux:
    SOLAR_CONSTANT = 1367  # W/m² at 1 AU
    
    @staticmethod
    def at_distance(au: float) -> float:
        """Solar flux at distance [AU]"""
        return SolarFlux.SOLAR_CONSTANT / au**2

@dataclass
class PlanetaryIR:
    planet_temp: float  # K
    view_factor: float  # to spacecraft
    
    def heat_flux(self) -> float:
        """IR flux from planet [W/m²]"""
        SIGMA = 5.67e-8
        return self.view_factor * SIGMA * self.planet_temp**4

@dataclass
class SpacecraftEnvironment:
    orbit: Orbit
    solar_absorptivity: float = 0.3
    IR_emissivity: float = 0.8
    
    def solar_heating(self, area: float, sun_angle: float = 0) -> float:
        """Solar heat load Q = G·A·α·cos(θ) [W]"""
        if self.orbit == Orbit.LEO:
            G = 1367
        elif self.orbit == Orbit.MARS:
            G = 590  # 1.52 AU
        else:
            G = 1367
        return G * area * self.solar_absorptivity * np.cos(sun_angle)
    
    def radiation_to_space(self, area: float, temp: float) -> float:
        """Radiation to space Q = ε·σ·A·T⁴ [W]"""
        SIGMA = 5.67e-8
        return self.IR_emissivity * SIGMA * area * temp**4
