"""
Layer 23: Distributed Aero Loads & Heating

Engineering-level aerodynamic load and heating distributions for
hypersonic and high-speed flight vehicles.

Physics Basis:
--------------
Newtonian Impact Theory (Hypersonic):
    Cp = Cp_max · sin²(θ)

    where θ = surface angle to freestream
          Cp_max ≈ 2 for M >> 1

Modified Newtonian (finite Mach):
    Cp_max = f(M_∞, γ)  from shock relations

Flat Plate Heating:
    Laminar:   q ∝ x^(-1/2)
    Turbulent: q ∝ x^(-0.2)

    St = h/(ρ·V·c_p)  (Stanton number)

Leading Edge Heating:
    q_stag = C·√(ρ/R_n)·V³  (Fay-Riddell)

    Distribution: q(s) = q_stag · f(s/R_n)

Load Integration:
    F = ∫ (p - p_∞)·n·dA  (force)
    M = ∫ r × dF         (moment)

Force Coefficients:
    CL = L/(q_∞·S)
    CD = D/(q_∞·S)
    Cm = M/(q_∞·S·c)

Prandtl-Meyer Expansion:
    ν(M) = √((γ+1)/(γ-1))·arctan(√(...)) - arctan(√(M²-1))

Features:
---------
1. Newtonian flow pressure distributions
2. Modified Newtonian (Mach-dependent Cp_max)
3. Simple shape solutions (flat plate, cone, sphere)
4. Prandtl-Meyer expansion for leeward surfaces
5. Distributed heating (flat plate, wedge, cone)
6. Leading edge heating distribution
7. Heat load integration (total Q)
8. Panel-based load integration
9. Force and moment coefficients
10. Center of pressure calculation
11. Structural load distributions (shear, moment)
12. Pressure coefficient analysis
13. Critical Cp for sonic conditions
14. Simplified aeroelastic coupling

Applications:
------------
- Re-entry vehicle heating analysis
- Hypersonic cruise vehicle loads
- Control surface heating and loads
- Wing leading edge design
- Panel-level stress analysis
- Center of pressure tracking
- Thermal protection system design

Typical Use Cases:
-----------------
- Distributed TPS thickness sizing
- Wing panel buckling with aero loads
- Control surface actuator sizing
- Center of pressure vs. center of gravity margin
- Peak heating location identification
- Structural load path analysis
"""

from .newtonian_flow import (
    FreestreamConditions,
    NewtonianFlow,
    SimpleShapes,
    PrandtlMeyerExpansion
)

from .distributed_heating import (
    HeatingConditions,
    DistributedHeating,
    HeatLoadIntegration,
    SurfaceHeatingMap
)

from .load_integration import (
    PressurePanel,
    AeroLoads,
    LoadIntegration,
    StructuralLoadDistribution,
    PressureCoefficientAnalysis,
    AeroelasticLoads
)

__all__ = [
    # Newtonian flow
    'FreestreamConditions',
    'NewtonianFlow',
    'SimpleShapes',
    'PrandtlMeyerExpansion',

    # Distributed heating
    'HeatingConditions',
    'DistributedHeating',
    'HeatLoadIntegration',
    'SurfaceHeatingMap',

    # Load integration
    'PressurePanel',
    'AeroLoads',
    'LoadIntegration',
    'StructuralLoadDistribution',
    'PressureCoefficientAnalysis',
    'AeroelasticLoads',

    # Solver
    'AeroLoadsSolverResult',
    'AeroLoadsSolver',
]

from .solver import AeroLoadsSolverResult, AeroLoadsSolver
