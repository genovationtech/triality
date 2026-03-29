"""
Layer 18: Spacecraft Thermal Control Systems

Radiative heat transfer, heat pipes, thermal loops, and thermal control for spacecraft.

Physics Basis:
--------------
Radiative Heat Transfer:
    q_rad = ε·σ·A·(T₁⁴ - T₂⁴)        (Stefan-Boltzmann)

    With view factors:
    q_1→2 = A₁·F_12·σ·(T₁⁴ - T₂⁴)

Heat Pipe Effective Conductivity:
    k_eff = k_pipe + (h_fg·ρ_vapor·A_wick·k_wick) / (μ_vapor·L)

    Typical k_eff ~ 10,000 - 100,000 W/(m·K)

Thermal Loop (Pumped):
    Q = ṁ·c_p·ΔT                    (heat transport)
    ΔP = f·(L/D)·(ρ·V²/2)           (pressure drop)

Heat Exchanger Effectiveness:
    ε = (T_c,out - T_c,in) / (T_h,in - T_c,in)

    NTU method:
    ε = f(NTU, C_r)                  where NTU = UA/(ṁ·c_p)_min

Radiator Sizing:
    Q = ε·σ·A·(T⁴ - T_sink⁴)

    where T_sink = deep space (4 K) or planetary IR

Environmental Heating:
    Q_solar = G·A·α                  (solar absorption)
    Q_albedo = G·a·F·A·α             (planetary reflection)
    Q_IR = ε·σ·F·A·T_planet⁴         (planetary IR)

Features:
---------
1. View factor calculator (simple geometries)
2. Multi-surface radiative exchange (simplified radiosity)
3. Heat pipe effective conductance model
4. Pumped loop thermal transport
5. Heat exchanger (ε-NTU method)
6. Radiator sizing and performance
7. Environmental heat loads (solar, albedo, Earth IR)
8. Heater control logic (bang-bang, PID)
9. Thermal margin and derating

Applications:
------------
- Spacecraft thermal design (LEO, GEO, deep space)
- Radiator sizing for power systems
- Heat pipe placement optimization
- Thermal control system trade studies
- Avionics cold plate design
- Solar array thermal management

Typical Use Cases:
-----------------
- Cubesat thermal design (passive radiation)
- Communication satellite radiators
- Planetary lander thermal protection
- Space station thermal control loops
- Mars rover thermal management
"""

from .radiative_transfer import (
    ViewFactorCalculator,
    RadiativeExchange,
    BlackBodyRadiation,
    Surface
)

from .heat_pipes import (
    HeatPipe,
    HeatPipeRegime,
    CapillaryLimit,
    SonicLimit
)

from .thermal_loops import (
    PumpedLoop,
    HeatExchanger,
    NTU_Method
)

from .spacecraft_environment import (
    SpacecraftEnvironment,
    Orbit,
    SolarFlux,
    PlanetaryIR
)

from .heater_control import (
    HeaterController,
    ControlMode,
    PIDController
)

from .nonlinear_radiative_solver import (
    IterativeRadiositySolver,
    RadiativeSurface,
    CoupledRadiationConductionSolver,
    ConvergenceHistory,
    verify_enclosure_closure
)

from .solver import (
    SpacecraftThermalSolver,
    SpacecraftThermalResult,
    ThermalNode,
    ConductiveLink,
    HeatPipeLink,
)

__all__ = [
    'ViewFactorCalculator',
    'RadiativeExchange',
    'BlackBodyRadiation',
    'Surface',
    'HeatPipe',
    'HeatPipeRegime',
    'CapillaryLimit',
    'SonicLimit',
    'PumpedLoop',
    'HeatExchanger',
    'NTU_Method',
    'SpacecraftEnvironment',
    'Orbit',
    'SolarFlux',
    'PlanetaryIR',
    'HeaterController',
    'ControlMode',
    'PIDController',
    'IterativeRadiositySolver',
    'RadiativeSurface',
    'CoupledRadiationConductionSolver',
    'ConvergenceHistory',
    'verify_enclosure_closure',
    'SpacecraftThermalSolver',
    'SpacecraftThermalResult',
    'ThermalNode',
    'ConductiveLink',
    'HeatPipeLink',
]
