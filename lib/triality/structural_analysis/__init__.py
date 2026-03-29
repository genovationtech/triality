"""
Layer 22: Structural Analysis & Buckling

Static structural analysis, buckling, and composite laminates for aerospace structures.

Physics Basis:
--------------
Beam Bending (Euler-Bernoulli):
    EI·d⁴w/dx⁴ = q(x)              (deflection equation)
    σ = M·c/I                      (bending stress)

Plate Bending (Kirchhoff):
    D·∇⁴w = q                      (thin plate equation)
    D = Et³/(12(1-ν²))            (flexural rigidity)

Euler Column Buckling:
    P_cr = π²·EI/(KL)²            (critical load)
    where K = effective length factor

Panel Buckling (Compression):
    σ_cr = k·π²·E/(12(1-ν²))·(t/b)²

    where k = buckling coefficient (depends on boundary conditions)

Composite Laminates (CLT):
    [A] = Σ [Q̄]ₖ·(zₖ - zₖ₋₁)      (extensional stiffness)
    [B] = ½Σ[Q̄]ₖ·(zₖ² - zₖ₋₁²)    (coupling stiffness)
    [D] = ⅓Σ[Q̄]ₖ·(zₖ³ - zₖ₋₁³)    (bending stiffness)

Failure Criteria:
    Tsai-Hill: (σ₁/X)² - σ₁σ₂/X² + (σ₂/Y)² + (τ/S)² ≤ 1
    Max Stress: max(σ₁/X, σ₂/Y, τ/S) ≤ 1

Margin of Safety:
    MS = (F_allowable/F_applied) - 1

Features:
---------
1. Euler-Bernoulli beam analysis (cantilever, simply supported)
2. Thin plate bending (Kirchhoff theory)
3. Euler column buckling with end conditions
4. Panel buckling (compression and shear)
5. Orthotropic panel buckling
6. Crippling analysis (Gerard-Becker)
7. Buckling interaction equations
8. Classical Lamination Theory (CLT)
9. Composite ply stress analysis
10. First-ply failure prediction
11. Tsai-Hill and max stress failure criteria
12. Margin of safety calculations
13. Load case combinations

Applications:
------------
- Aircraft fuselage and wing structures
- Spacecraft primary structure
- Composite panels and sandwich structures
- Launch vehicle shells
- Pressure vessels
- Stiffened panels

Typical Use Cases:
-----------------
- Wing spar design and verification
- Fuselage skin panel buckling check
- Composite laminate layup optimization
- Margin of safety verification per NASA-STD-5001
- Load case envelope analysis
"""

from .static_solver import (
    EulerBernoulliBeam,
    PlateAnalysis,
    Material,
    MarginOfSafety,
    LoadCaseCombination,
    LoadType,
    MATERIALS
)

from .buckling import (
    EulerColumnBuckling,
    PanelBuckling,
    OrthotropicPanelBuckling,
    CripplingAnalysis,
    BucklingInteraction,
    BucklingMode,
    EndCondition
)

from .composite_laminates import (
    OrthotropicPly,
    LaminatePly,
    Laminate,
    FailureCriterion,
    COMPOSITE_MATERIALS
)

from .fem_solver import (
    StructuralFEMSolver,
    FEMResult,
    ElementResult,
    DOFType,
    euler_bernoulli_stiffness,
    bar_stiffness,
    beam_uniform_load_vector,
)

from .solver import (
    StructuralSolver,
    StructuralAnalysisResult,
    BucklingCheckResult,
    LaminateCheckResult,
)

__all__ = [
    # Static analysis
    'EulerBernoulliBeam',
    'PlateAnalysis',
    'Material',
    'MarginOfSafety',
    'LoadCaseCombination',
    'LoadType',
    'MATERIALS',

    # Buckling
    'EulerColumnBuckling',
    'PanelBuckling',
    'OrthotropicPanelBuckling',
    'CripplingAnalysis',
    'BucklingInteraction',
    'BucklingMode',
    'EndCondition',

    # Composites
    'OrthotropicPly',
    'LaminatePly',
    'Laminate',
    'FailureCriterion',
    'COMPOSITE_MATERIALS',

    # FEM solver
    'StructuralFEMSolver',
    'FEMResult',
    'ElementResult',
    'DOFType',
    'euler_bernoulli_stiffness',
    'bar_stiffness',
    'beam_uniform_load_vector',

    # Integrated solver
    'StructuralSolver',
    'StructuralAnalysisResult',
    'BucklingCheckResult',
    'LaminateCheckResult',
]
