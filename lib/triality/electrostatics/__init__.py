"""Electrostatics and Conduction Module

This module provides physics-based electrostatic field analysis and conductive
media simulation for electrical engineering applications.

Layer 1: Electrostatics & Conduction
- Laplace equation (∇²V = 0) for charge-free regions
- Poisson equation (∇²V = -ρ/ε) for regions with charge density
- Electric field calculation (E = -∇V)
- Conductive media (∇⋅(σ∇V) = 0) for current flow
- Current density (J = -σ∇V) and power density (J²/σ)

Applications:
- Power electronics layout
- Busbar and grounding design
- Clearance & insulation planning
- PCB copper pour analysis (DC/low-frequency)
- High-voltage safety analysis
"""

from .field_solver import (
    ElectrostaticSolver,
    BoundaryCondition,
    BoundaryType,
    ChargeDistribution,
)

from .conduction import (
    ConductiveSolver,
    Material,
    CurrentDensityField,
    PowerDensityField,
)

from .derived_quantities import (
    ElectricField,
    FieldMagnitude,
    GradientAnalysis,
    HotspotDetector,
)

from .solver import (
    ElectrostaticsSolver,
    ElectrostaticsResult,
)

__all__ = [
    # Field solver
    'ElectrostaticSolver',
    'BoundaryCondition',
    'BoundaryType',
    'ChargeDistribution',

    # Conduction
    'ConductiveSolver',
    'Material',
    'CurrentDensityField',
    'PowerDensityField',

    # Derived quantities
    'ElectricField',
    'FieldMagnitude',
    'GradientAnalysis',
    'HotspotDetector',

    # Numerical solver
    'ElectrostaticsSolver',
    'ElectrostaticsResult',
]
