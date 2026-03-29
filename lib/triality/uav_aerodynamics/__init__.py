"""
UAV Aerodynamics Module

Subsonic aerodynamic analysis for fixed-wing and rotary-wing unmanned
aerial vehicles. Provides airfoil lift/drag/moment computation, blade
element momentum theory for rotors and propellers, multirotor hover
and forward-flight power models, and fixed-wing trim analysis.

Physics Basis:
--------------
Thin Airfoil Theory (Prandtl, 1918):
    CL = 2*pi*alpha                       (lift curve slope)

Parabolic Drag Polar:
    CD = CD0 + CL^2 / (pi * e * AR)       (induced drag)

Blade Element Momentum Theory (Glauert, 1926):
    dT = 0.5 * rho * c * W^2 * CL * cos(phi) * dr
    T  = CT * rho * n^2 * D^4             (thrust coefficient form)
    Q  = CQ * rho * n^2 * D^5             (torque coefficient form)

Momentum Theory (Rankine-Froude):
    T = 2 * rho * A * v_i * (V + v_i)     (actuator disk)

Hover Power:
    P_hover = T^(3/2) / sqrt(2 * rho * A)

Maximum L/D:
    (L/D)_max = 0.5 * sqrt(pi * e * AR / CD0)

Features:
---------
1.  Subsonic airfoil CL(alpha), CD(CL), CM(alpha)
2.  Thin airfoil theory with finite-wing correction
3.  Post-stall aerodynamic model
4.  Blade element momentum theory (BEMT) for rotors
5.  Prandtl tip-loss correction
6.  Momentum theory induced velocity
7.  Multirotor hover and forward-flight power
8.  Power decomposition: induced + parasite + profile
9.  Fixed-wing trim solver (alpha, elevator)
10. Stall speed and flight envelope
11. Best endurance and best range speeds
12. NACA 0012, Clark Y, and flat plate presets

Applications:
-------------
- Multirotor UAV sizing and performance
- Fixed-wing UAV preliminary design
- Propeller selection and matching
- Endurance and range estimation
- Flight envelope analysis

References:
-----------
- Prandtl, L. (1918). Lifting-line theory.
- Glauert, H. (1926). Blade element theory.
- Betz, A. (1919). Propeller efficiency limit.
- Rankine, W.J.M. (1865). Actuator disk theory.
- Froude, R.E. (1889). Momentum theory of propellers.
"""

from .aero_models import (
    # Enums
    AirfoilType,
    FlightRegime,
    UAVType,
    # Result dataclasses
    AeroCoefficients,
    RotorPerformance,
    TrimResult,
    MultirotorPerformance,
    # Presets
    AIRFOIL_PRESETS,
    # Classes
    SubsonicAerodynamics,
    BladeElementTheory,
    MultirotorModel,
    FixedWingModel,
)

from .solver import VortexLatticeSolver, VLMResult

__all__ = [
    # Enums
    'AirfoilType',
    'FlightRegime',
    'UAVType',
    # Result dataclasses
    'AeroCoefficients',
    'RotorPerformance',
    'TrimResult',
    'MultirotorPerformance',
    'VLMResult',
    # Presets
    'AIRFOIL_PRESETS',
    # Classes
    'SubsonicAerodynamics',
    'BladeElementTheory',
    'MultirotorModel',
    'FixedWingModel',
    # Solvers
    'VortexLatticeSolver',
]
