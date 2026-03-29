"""Field-Aware Routing Module

Layer 2: Physics-informed routing using electromagnetic analysis

This module bridges Layer 1 (Electrostatics & Conduction) with the Spatial Flow
Engine, creating physics-aware cost fields for intelligent routing.

Key Capabilities:
- EM-informed cost fields from electric field and current density
- Multi-conductor coupling analysis (quasi-static)
- Return path quality metrics
- Ground impedance awareness
- EMI risk assessment
- Thermal risk from high-current zones

Applications:
- EMI-aware PCB and harness routing
- High-current trace shaping
- Ground return optimization
- Clearance-aware layout
- Crosstalk mitigation

This is the "missing middle layer" between hand calculations and full-wave EM simulation.
"""

from .cost_field_builders import (
    ElectricFieldCostBuilder,
    CurrentDensityCostBuilder,
    EMICostBuilder,
    ThermalRiskCostBuilder,
    ClearanceCostBuilder,
    PhysicsCostField,
)

from .coupling_analysis import (
    MultiConductorCoupling,
    CouplingZone,
    ReturnPathAnalyzer,
    GroundImpedanceMap,
    CrosstalkAnalyzer,
)

from .routing_integration import (
    PhysicsAwareRouter,
    RouteWithPhysics,
    OptimizationObjective,
    MultiRouteOptimizer,
)

from .solver import (
    EMFieldSolver,
    EMFieldSolverResult,
    FieldSolverConfig,
    DomainConfig,
    BoundaryCondition,
    BCType,
    ConductorSpec,
    ChargeRegion,
)

__all__ = [
    # Cost field builders
    'ElectricFieldCostBuilder',
    'CurrentDensityCostBuilder',
    'EMICostBuilder',
    'ThermalRiskCostBuilder',
    'ClearanceCostBuilder',
    'PhysicsCostField',

    # Coupling analysis
    'MultiConductorCoupling',
    'CouplingZone',
    'ReturnPathAnalyzer',
    'GroundImpedanceMap',
    'CrosstalkAnalyzer',

    # Routing integration
    'PhysicsAwareRouter',
    'RouteWithPhysics',
    'OptimizationObjective',
    'MultiRouteOptimizer',

    # Solver
    'EMFieldSolver',
    'EMFieldSolverResult',
    'FieldSolverConfig',
    'DomainConfig',
    'BoundaryCondition',
    'BCType',
    'ConductorSpec',
    'ChargeRegion',
]
