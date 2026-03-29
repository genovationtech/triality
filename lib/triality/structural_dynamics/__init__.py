"""
Layer 19: Structural Dynamics & Vibrations

Modal analysis, random vibration, shock response for aerospace structures.

Physics Basis:
--------------
Equation of Motion:
    M·ẍ + C·ẋ + K·x = F(t)

Modal Analysis (Eigenvalue Problem):
    (K - ω²·M)·φ = 0

    Eigenvalues: ω_n² (natural frequencies squared)
    Eigenvectors: φ_n (mode shapes)

Random Vibration (PSD):
    G_response(f) = |H(f)|²·G_input(f)

    where H(f) = transfer function

    RMS response:
        σ = √(∫ G(f) df)

Shock Response Spectrum (SRS):
    Maximum response of SDOF systems to transient input.

    For SDOF with natural frequency f_n:
        SRS(f_n) = max|x(t)|

Miles' Equation (Random Vibration):
    g_RMS = √(π/2 · f_n · Q · PSD(f_n))

    where Q = quality factor ≈ 1/(2·ζ)

Features:
---------
1. Modal analysis (eigenvalue solver)
2. Mode shapes and participation factors
3. Random vibration response (PSD)
4. Shock response spectrum (SRS)
5. Miles' equation for RMS acceleration
6. Fatigue life from vibration
7. Transmissibility and isolation

Applications:
------------
- Launch vehicle load analysis
- Spacecraft modal testing correlation
- Random vibration qualification
- Shock/pyroshock analysis
- Isolation system design
- Fatigue life prediction

Typical Use Cases:
-----------------
- Launch environment (NASA-STD-7001)
- Spacecraft acceptance testing
- Pyroshock from stage separation
- Reaction wheel vibration isolation
"""

from .modal_analysis import (
    ModalSolver,
    ModeShape,
    StructuralModel
)

from .random_vibration import (
    RandomVibrationAnalyzer,
    PSD,
    MilesEquation
)

from .shock_response import (
    ShockResponseSpectrum,
    SRSCalculator
)

from .solver import (
    StructuralDynamicsSolver,
    StructuralDynamicsResult,
)

__all__ = [
    'ModalSolver',
    'ModeShape',
    'StructuralModel',
    'RandomVibrationAnalyzer',
    'PSD',
    'MilesEquation',
    'ShockResponseSpectrum',
    'SRSCalculator',
    'StructuralDynamicsSolver',
    'StructuralDynamicsResult',
]
