"""
Layer 31: Full-Wave EM Solvers

Complete electromagnetic field simulation methods.

Physics Basis:
--------------
Maxwell's Equations (Frequency Domain):
    ∇×E = -jω·B
    ∇×H = jω·D + J
    ∇·D = ρ
    ∇·B = 0

Maxwell's Equations (Time Domain):
    ∇×E = -∂B/∂t
    ∇×H = ∂D/∂t + J
    
Constitutive Relations:
    D = ε·E
    B = μ·H
    J = σ·E

Wave Equation:
    ∇²E - με·∂²E/∂t² = 0

Helmholtz Equation (frequency domain):
    ∇²E + k²·E = 0
    where k = ω√(με) = wave number

Features:
---------
Dipole Antennas:
1. Hertzian dipole (infinitesimal)
   - Length L << λ
   - Radiation pattern: sin(θ)
   - Radiation resistance: R_rad = 80π²(L/λ)²
2. Half-wave dipole (λ/2)
   - Resonant length
   - R_rad = 73 Ω
   - Directivity = 1.64 (2.15 dBi)
   - Input impedance ≈ 73 + j42.5 Ω
3. Quarter-wave monopole (λ/4)
   - Over ground plane
   - R_rad = 36.5 Ω
   - Gain = 5.15 dBi
4. Radiation patterns
   - E-plane and H-plane
   - 3D patterns
   - Beamwidth (HPBW)
5. Antenna parameters
   - Gain and directivity
   - Effective aperture
   - Radiation efficiency
   - Input impedance
6. Friis transmission equation
   - Free space path loss
   - Link budget

Waveguides:
7. Rectangular waveguide modes
   - TE_mn (transverse electric)
   - TM_mn (transverse magnetic)
   - Dominant mode: TE₁₀
8. Cutoff frequency
   - f_c = (c/2)·√((m/a)² + (n/b)²)
   - Below cutoff: evanescent
   - Above cutoff: propagating
9. Propagation characteristics
   - Propagation constant β
   - Phase velocity v_p = c/√(1-(f_c/f)²)
   - Group velocity v_g = c·√(1-(f_c/f)²)
   - Guide wavelength λ_g = λ₀/√(1-(f_c/f)²)
10. Wave impedance
    - TE: Z_TE = η₀/√(1-(f_c/f)²)
    - TM: Z_TM = η₀·√(1-(f_c/f)²)
11. Attenuation
    - Conductor losses (ohmic)
    - Dielectric losses
12. Standard waveguides
    - WR-XX designations
    - EIA/IEC standards
    - Frequency ranges

FDTD Method:
13. Yee cell (staggered grid)
    - E and H components offset in space
    - Leap-frog time stepping
    - E(n) and H(n+1/2) alternating
14. Update equations
    - E^(n+1) from H^(n+1/2)
    - H^(n+3/2) from E^(n+1)
    - Central differences for spatial derivatives
15. Courant stability criterion
    - Δt ≤ 1/(c·√(1/Δx² + 1/Δy² + 1/Δz²))
    - Courant number S < 1
16. Grid resolution
    - 10-20 cells per wavelength
    - Smaller cells for fine features
17. Boundary conditions
    - PEC (Perfect Electric Conductor)
    - PMC (Perfect Magnetic Conductor)
    - PML (Perfectly Matched Layer)
    - Mur ABC (Absorbing BC)
18. Sources
    - Hard source (E = E_source)
    - Soft source (E += E_source)
    - Gaussian pulse
    - Sinusoidal

Applications:
------------
- Antenna design (dipoles, arrays, microstrip)
- Waveguide components (filters, couplers, transitions)
- Microwave circuits
- Radar cross-section (RCS) calculation
- EMI/EMC analysis
- Photonic devices
- Metamaterials
- Wireless propagation
- Medical imaging (MRI, microwave)
- Remote sensing

Standards & References:
-----------------------
- Balanis, "Antenna Theory: Analysis and Design"
- Pozar, "Microwave Engineering"
- Taflove & Hagness, "Computational Electrodynamics: The FDTD Method"
- Harrington, "Time-Harmonic Electromagnetic Fields"
- Jackson, "Classical Electrodynamics"
- IEEE Std 145 (Antenna definitions)
- EIA RS-261 (Waveguide flanges)

Typical Use Cases:
------------------
1. Dipole antenna at 300 MHz
   - λ = 1 m
   - L = 0.5 m (half-wave)
   - Gain ≈ 2.15 dBi
2. X-band waveguide (WR-90)
   - 8.2-12.4 GHz
   - a = 22.86 mm, b = 10.16 mm
   - TE₁₀ cutoff: 6.56 GHz
3. FDTD simulation of patch antenna
   - Grid: 100×100×50 cells
   - Δx = λ/20
   - PML boundaries
   - Run for 1000 time steps
4. Link budget calculation
   - Tx power: 1 W (30 dBm)
   - Antenna gains: 10 dBi each
   - Distance: 10 km
   - Frequency: 2.4 GHz
   - Received power ≈ -50 dBm

Example Workflow:
-----------------
1. Define antenna geometry (length, frequency)
2. Calculate electrical parameters (Z_in, gain)
3. Compute radiation pattern
4. Design waveguide feed system
5. Set up FDTD simulation
   - Define grid resolution
   - Set boundary conditions
   - Place sources
6. Run time-stepping
7. Extract S-parameters, fields
8. Post-process (FFT for frequency response)

Notes:
------
- FDTD is explicit (conditionally stable)
- PML absorbs > 99.9999% of outgoing waves
- Dispersion error increases with Δx/λ
- Waveguide modes orthogonal (no coupling)
- TE₁₀ has no cutoff in one direction
- Group velocity carries energy (v_g < c)
- Phase velocity can exceed c (no contradiction)
- Antenna Q-factor = ω·stored_energy / radiated_power
"""

from .dipole_antenna import (
    DipoleType,
    AntennaParameters,
    DipoleAntenna,
    MonopoleAntenna,
    WireDipoleArray
)

from .waveguide import (
    ModeType,
    WaveguideGeometry,
    WaveguideMode,
    StandardWaveguides
)

from .fdtd import (
    BoundaryCondition,
    FDTDParameters,
    FDTD,
    MurABC,
    GaussianPulse
)

__all__ = [
    # Dipole antennas - Enums and data structures
    'DipoleType',
    'AntennaParameters',
    
    # Dipole antennas - Analysis classes
    'DipoleAntenna',
    'MonopoleAntenna',
    'WireDipoleArray',
    
    # Waveguides - Enums and data structures
    'ModeType',
    'WaveguideGeometry',
    
    # Waveguides - Analysis classes
    'WaveguideMode',
    'StandardWaveguides',
    
    # FDTD - Enums and data structures
    'BoundaryCondition',
    'FDTDParameters',
    
    # FDTD - Analysis classes
    'FDTD',
    'MurABC',
    'GaussianPulse'
]
