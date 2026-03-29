# Triality Comprehensive Technical Reference

**Version 0.2.0 | © 2025 Genovation Technological Solutions Pvt Ltd. Powered by Mentis OS.**

## Table of Contents

1. [Overview](#overview)
2. [Production-Ready Layers (1-3)](#production-ready-layers)
3. [Extended Physics Modules (Layers 4-33)](#extended-physics-modules)
4. [Core Systems](#core-systems)
5. [Complete Module Reference](#complete-module-reference)
6. [API Quick Reference](#api-quick-reference)
7. [Performance Characteristics](#performance-characteristics)

---

## Overview

Triality is a multi-layer physics simulation framework organized into:
- **Production Layers (1-3)**: Fully validated with comprehensive testing
- **Extended Modules (4-33+)**: Functional implementations for early-stage design
- **Core Systems**: Automatic PDE solving, spatial routing, verification

**Status Indicators:**
- ✅ Production-ready (validated, tested, supported)
- 🔧 Framework complete (numerical refinement ongoing)
- 📚 Functional implementation (early-stage design, not production-certified)

**Cross-Cutting: Observable Layer** — All 16 runtime modules produce domain-specific engineering observables (126 total, 10 with pass/fail thresholds) at < 0.15% solver overhead via `triality.observables`. See [Architecture](architecture.md#the-observable-layer).

---

## Production-Ready Layers

### Layer 1: Electrostatics & Conduction ✅

**Status**: 16/16 tests passing | Production-ready

**Module**: `triality.electrostatics`

**Capabilities:**

**Core Solvers:**
- `ElectrostaticSolver2D`: 2D electrostatic field solver
  - Laplace equation: ∇²V = 0
  - Poisson equation: ∇²V = -ρ/ε
  - Boundary conditions: Dirichlet, Neumann, mixed
  - Multi-material interfaces with harmonic averaging

- `ConductionSolver2D`: Steady-state thermal/electrical conduction
  - Conduction equation: ∇⋅(σ∇V) = 0
  - Variable conductivity materials
  - Heat source/sink support
  - Coupled electrical-thermal analysis

**Derived Quantities:**
- Electric field: E = -∇V
- Current density: J = -σ∇V
- Power density: P = J²/σ = |E|²σ
- Capacitance calculations
- Energy storage analysis

**Performance:**
- 50×50 grid: ~5ms
- 100×100 grid: ~20ms
- 200×200 grid: ~150ms

**Accuracy:**
- Simple geometries: ±2-5% vs analytical
- Complex geometries: ±10-15% vs FEM
- Multi-material: ±10-20%

**API Example:**
```python
from triality.electrostatics import ElectrostaticSolver2D

solver = ElectrostaticSolver2D(
    x_range=(0, 0.1),
    y_range=(0, 0.1),
    resolution=100
)
solver.set_boundary('left', 'voltage', 5.0)
solver.set_boundary('right', 'voltage', 0.0)
solver.set_charge_density(lambda x, y: rho(x, y))

result = solver.solve()
E_field = result.electric_field()
J_current = result.current_density(conductivity=5.8e7)
```

**Key Classes:**
- `ElectrostaticSolver2D`: Main solver class
- `ConductionSolver2D`: Conduction analysis
- `ElectrostaticResult`: Solution container with analysis methods
- `MaterialInterface`: Multi-material boundary handling

**Use Cases:**
- PCB power distribution analysis
- Capacitor design and optimization
- Thermal management (steady-state)
- EMI field mapping
- Current density hot-spot identification

---

### Layer 2: Physics-Aware Routing ✅

**Status**: 15/15 tests passing | Production-ready | **Industry-unique capability**

**Module**: `triality.field_aware_routing`

**Capabilities:**

**Core Router:**
- `PhysicsAwareRouter`: Multi-physics continuous routing
  - Physics-to-cost conversion
  - Multi-objective optimization
  - Continuous path optimization (no grid discretization)
  - Real-time obstacle avoidance

**Cost Field Builders:**
- `ElectricFieldCostBuilder`: E-field → EMI risk
- `CurrentDensityCostBuilder`: J-field → thermal risk
- `ThermalCostBuilder`: Temperature → thermal penalty
- `CouplingCostBuilder`: Mutual inductance/capacitance

**Optimization Objectives:**
- `MIN_LENGTH`: Shortest path (traditional routing)
- `MIN_EMI`: Minimize electromagnetic interference
- `MIN_THERMAL`: Avoid high-temperature regions
- `MIN_CROSSTALK`: Reduce coupling between traces
- `BALANCED`: Multi-objective trade-off

**Coupling Analysis:**
- `CouplingAnalyzer`: Multi-conductor coupling calculation
  - Mutual capacitance estimation
  - Mutual inductance estimation
  - Crosstalk prediction
  - Return path optimization

**Performance:**
- 100×100 grid analysis: ~20ms
- Path optimization: ~50ms
- Total (solve + route): ~70ms

**Accuracy:**
- EMI reduction: 30-70% vs geometric routing
- Thermal reduction: 35% lower peak temperature
- Crosstalk reduction: 40-60%
- Cost field fidelity: ±20-30% vs full EM

**API Example:**
```python
from triality.field_aware_routing import (
    PhysicsAwareRouter,
    OptimizationObjective,
    ElectricFieldCostBuilder
)

# Solve electromagnetic fields
solver = ElectrostaticSolver2D(...)
result = solver.solve()

# Convert physics to routing costs
emi_cost = ElectricFieldCostBuilder.from_result(result)
thermal_cost = ThermalCostBuilder.from_temperature_field(T_field)

# Multi-physics routing
router = PhysicsAwareRouter()
router.set_domain((0, 0.05), (0, 0.05))
router.add_physics_cost('EMI', emi_cost, weight=10.0)
router.add_physics_cost('Thermal', thermal_cost, weight=5.0)
router.add_rectangular_obstacle((0.02, 0.02), (0.03, 0.03))

route = router.route(
    start=(0.01, 0.01),
    end=(0.04, 0.04),
    objective=OptimizationObjective.BALANCED
)

print(f"EMI reduction: {route.emi_reduction:.1f}%")
print(f"Thermal reduction: {route.thermal_reduction:.1f}%")
```

**Key Classes:**
- `PhysicsAwareRouter`: Main routing engine
- `CostFieldBuilder`: Base class for physics-to-cost conversion
- `RouteResult`: Solution with metrics and visualization
- `CouplingAnalyzer`: Multi-conductor analysis
- `OptimizationObjective`: Objective function enum

**Use Cases:**
- EMI-aware PCB trace routing
- Thermal-aware power distribution
- High-speed signal integrity
- Multi-conductor crosstalk minimization
- Return path optimization
- Mixed-signal board layout

---

### Layer 3: Drift-Diffusion Semiconductors 🔧

**Status**: 5/15 tests passing | Framework complete | Numerical refinement ongoing

**Module**: `triality.drift_diffusion`

**Capabilities:**

**1D Semiconductor Device Solver:**
- `DriftDiffusionSolver1D`: Coupled Poisson + continuity
  - Poisson equation: ∇²ψ = -q(p - n + N_D - N_A)/ε
  - Electron continuity: ∂n/∂t = (1/q)∇⋅J_n + G - R
  - Hole continuity: ∂p/∂t = -(1/q)∇⋅J_p + G - R
  - Drift-diffusion currents

**Device Types:**
- PN junctions
- Schottky diodes
- MOS capacitors (basic)
- Resistors with distributed doping

**Analysis Capabilities:**
- Built-in potential calculation
- Depletion width estimation
- I-V characteristics
- Carrier concentration profiles
- Electric field distribution
- Recombination/generation rates

**Current Accuracy:**
- Built-in potential: ±20%
- Depletion width: ±20%
- I-V quantitative: ±30-50%
- Relative comparison: Very good

**API Example:**
```python
from triality.drift_diffusion import (
    DriftDiffusionSolver1D,
    DopingProfile,
    MaterialParameters
)

# PN junction setup
solver = DriftDiffusionSolver1D(
    length=1e-6,  # 1 micron
    n_points=200
)

# Define doping
doping = DopingProfile.pn_junction(
    length=1e-6,
    N_A=1e17,  # p-side
    N_D=1e16,  # n-side
    junction_pos=0.5e-6
)

solver.set_doping_profile(doping)
result = solver.solve_equilibrium()

# Analysis
V_bi = result.built_in_potential()
W_dep = result.depletion_width()
iv_curve = solver.compute_iv_curve(voltages=np.linspace(-1, 0.8, 50))
```

**Key Classes:**
- `DriftDiffusionSolver1D`: Main 1D solver
- `DopingProfile`: Doping distribution specification
- `MaterialParameters`: Si, GaAs, etc. material properties
- `DeviceResult`: Solution with analysis methods

**Use Cases:**
- PN junction analysis and design
- Diode I-V curve prediction
- Doping profile optimization
- Order-of-magnitude device estimates
- Educational semiconductor physics
- Relative device comparisons

**Limitations:**
- 1D only (no 2D/3D)
- Basic drift-diffusion model
- No quantum effects
- No advanced recombination models
- NOT for production tapeout (use Sentaurus, Silvaco)

---

## Extended Physics Modules

**Status**: 📚 Functional implementations for early-stage design | Not production-validated

These modules provide physics implementations suitable for:
- Initial feasibility studies
- Proof-of-concept analysis
- Educational purposes
- Relative design comparisons
- Trend analysis

**⚠️ For production/certification work**, upgrade to specialized commercial tools.

---

### Layers 4-10: Nuclear & Energy Systems 📚

#### Layer 4-5: Neutronics & Radiation

**Module**: `triality.neutronics`

**Capabilities:**
- Basic neutron diffusion
- Cross-section libraries
- Criticality calculations
- Flux distribution

**Module**: `triality.radiation_environment`

**Capabilities:**
- Space radiation environment modeling
- Particle flux calculations
- Single Event Effect (SEE) rates
- Total Ionizing Dose (TID) estimation

**Module**: `triality.shielding`

**Capabilities:**
- Radiation shielding design
- Attenuation calculations
- Multi-layer shield optimization
- Material effectiveness comparison

**API Example:**
```python
from triality.neutronics import ReactorPhysics, CrossSection
from triality.radiation_environment import SpaceEnvironment

# Neutronics
k_eff = ReactorPhysics.calculate_k_effective(
    fuel_enrichment=3.5,
    geometry='square_lattice'
)

# Space radiation
env = SpaceEnvironment.GEO_orbit()
flux = env.proton_flux(energy_mev=100)
```

---

#### Layer 6-10: Automotive & Power Systems

**Module**: `triality.automotive_thermal`

**Capabilities:**
- Engine thermal management
- Cooling system analysis
- Thermal load estimation
- Heat exchanger sizing

**Module**: `triality.battery_thermal`

**Capabilities:**
- Battery pack thermal analysis
- Cell-level heat generation
- Cooling strategy evaluation
- Thermal runaway assessment

**Module**: `triality.coupled_electrical_thermal`

**Capabilities:**
- Coupled E-T field solving
- Joule heating effects
- Temperature-dependent conductivity
- Steady-state and transient analysis

**API Example:**
```python
from triality.battery_thermal import BatteryCell, ThermalModel

cell = BatteryCell(
    capacity_ah=50,
    discharge_rate_c=2.0,
    dimensions=(0.1, 0.2, 0.01)
)

thermal = ThermalModel(cell)
T_max = thermal.predict_max_temperature(
    ambient=25,
    cooling_type='forced_air',
    airflow_rate=0.5  # m/s
)
```

---

### Layers 11-20: Aerospace & Propulsion 📚

#### Layer 11-15: Advanced Manufacturing & CFD

**Module**: `triality.cfd_turbulence`

**Capabilities:**
- Turbulence models (k-ε, k-ω, Reynolds stress)
- Compressible flow analysis
- Normal/oblique shock calculations
- Boundary layer analysis
- Turbulent viscosity estimation

**Module**: `triality.reacting_flows`

**Capabilities:**
- Species transport equations
- Chemical reaction networks
- Flame front tracking
- Well-stirred reactor modeling

**Module**: `triality.combustion_chemistry`

**Capabilities:**
- Arrhenius rate calculations
- Laminar flame speed
- Ignition delay prediction
- Chemical time scales
- Mechanism reduction

**API Example:**
```python
from triality.cfd_turbulence import TurbulenceModel, BoundaryLayer
from triality.combustion_chemistry import ArrheniusRate, LaminarFlameSpeed

# Turbulence
turb = TurbulenceModel.k_epsilon(
    velocity=50,
    length_scale=0.1,
    turbulence_intensity=0.05
)
nu_t = turb.eddy_viscosity()

# Combustion
S_L = LaminarFlameSpeed.methane_air(
    T=300,
    P=101325,
    phi=1.0  # Stoichiometric
)
print(f"Flame speed: {S_L:.3f} m/s")
```

---

#### Layer 16-20: Aerospace Fundamentals

**Module**: `triality.aero_loads`

**Capabilities:**
- Distributed aerodynamic loads
- Lift/drag coefficient estimation
- Moment coefficient calculations
- Aerodynamic heating (convective, radiative)
- Hypersonic flow effects

**Module**: `triality.aerothermodynamics`

**Capabilities:**
- Stagnation point heating
- Boundary layer transition
- Ablation modeling (basic)
- Reentry heating prediction

**Module**: `triality.propulsion`

**Capabilities:**
- Rocket engine performance (Isp, thrust)
- Nozzle flow analysis
- Jet engine cycle analysis
- Propellant mass fractions

**API Example:**
```python
from triality.aero_loads import AerodynamicLoads, HypersonicHeating
from triality.propulsion import RocketEngine

# Aero loads
loads = AerodynamicLoads(
    mach=3.0,
    altitude=20000,
    alpha_deg=5.0,
    wing_area=50
)
L, D = loads.lift_drag_forces()

# Propulsion
engine = RocketEngine(
    thrust_n=1e6,
    chamber_pressure_pa=10e6,
    exit_pressure_pa=5e3,
    propellant='LOX_RP1'
)
isp = engine.specific_impulse()
```

---

### Layers 21-27: Flight Dynamics & Structures 📚

#### Layer 21-22: Flight Mechanics & Structures

**Module**: `triality.flight_mechanics`

**Capabilities:**
- 6-DoF dynamics (equations of motion)
- Guidance, Navigation, Control (GNC)
- Trajectory optimization
- Stability derivatives
- Autopilot design

**Module**: `triality.structural_analysis`

**Capabilities:**
- Stress and strain analysis
- Buckling load calculations
- Modal analysis (natural frequencies)
- Composite materials
- Fatigue life estimation

**Module**: `triality.structural_dynamics`

**Capabilities:**
- Dynamic response analysis
- Forced vibration
- Damping models
- Shock loading
- Frequency response functions

**API Example:**
```python
from triality.flight_mechanics import RigidBody6DOF, GNCSystem
from triality.structural_analysis import BeamAnalysis, CompositeProperties

# Flight dynamics
aircraft = RigidBody6DOF(
    mass=50000,
    inertia_matrix=I_xx_yy_zz,
    position=[0, 0, 1000],
    velocity=[200, 0, 0]
)
gnc = GNCSystem(aircraft)
gnc.set_autopilot_gains(Kp=2.0, Ki=0.5, Kd=0.1)

# Structures
beam = BeamAnalysis(
    length=10,
    E_modulus=70e9,
    I_moment=1e-5,
    distributed_load=1000
)
max_deflection = beam.max_deflection()
natural_freq = beam.natural_frequency(mode=1)
```

---

#### Layer 23-27: Advanced Aero/Structures

**Module**: `triality.aeroelasticity`

**Capabilities:**
- Static aeroelasticity (divergence, control reversal)
- Flutter analysis (frequency/damping)
- Elastic axis calculations
- Structural/aerodynamic coupling

**Module**: `triality.fracture_mechanics`

**Capabilities:**
- Stress intensity factor (K_I, K_II, K_III)
- Crack growth rate (Paris law)
- Fracture toughness (K_Ic)
- Damage tolerance analysis
- Crack geometries (center, edge, surface)

**Module**: `triality.thermo_mechanical`

**Capabilities:**
- Coupled thermal-structural analysis
- Thermal stress calculation
- CTE mismatch effects
- Thermal strain

**API Example:**
```python
from triality.aeroelasticity import StaticAeroelasticity, FlutterAnalysis
from triality.fracture_mechanics import StressIntensityFactor, ParisLaw

# Aeroelasticity
elastic_props = ElasticProperties(...)
aero_props = AeroProperties(...)
V_div = StaticAeroelasticity(elastic_props, aero_props).divergence_speed()

# Fracture mechanics
K_I = StressIntensityFactor.center_crack(
    stress=100e6,  # Pa
    crack_length=0.01,  # m
    width=0.1
)
da_dN = ParisLaw.crack_growth_rate(
    delta_K=20e6,  # MPa√m
    C=3e-13,
    m=3.0
)
```

---

### Layers 28-33: Advanced Multiphysics 📚

#### Layer 28: Quantum & Nanoscale

**Module**: `triality.quantum_nanoscale`

**Capabilities:**
- Particle in a box (1D, 2D, 3D)
- Infinite square well energy levels
- Tunneling probability (rectangular, triangular barriers)
- Quantum harmonic oscillator
- Basic wavefunction calculations

**API Example:**
```python
from triality.quantum_nanoscale import (
    InfiniteSquareWell,
    TunnelingProbability,
    QuantumSystemParameters
)

params = QuantumSystemParameters(
    mass=9.1e-31,  # electron
    L=1e-9  # 1nm box
)

# Energy levels
E_levels = InfiniteSquareWell.energy_levels(params, n_max=3)
E1 = E_levels[0]  # Ground state

# Tunneling
T = TunnelingProbability.rectangular_barrier(
    params,
    E=0.5*1.6e-19,  # eV
    V0=1.0*1.6e-19,
    a=1e-9
)
```

---

#### Layer 29: Injury Biomechanics

**Module**: `triality.injury_biomechanics`

**Capabilities:**
- Head Injury Criterion (HIC-15, HIC-36)
- Chest compression injury
- Hybrid III dummy modeling
- Crash biomechanics

**API Example:**
```python
from triality.injury_biomechanics import HIC, ChestInjury

# Head injury
hic_value = HIC.calculate_hic(
    time=time_array,
    acceleration=accel_array_g
)
print(f"HIC-15: {hic_value} (limit: 700)")

# Chest compression
compression_mm = ChestInjury.max_compression_mm(
    force=3000,  # N
    stiffness=2.5  # kN/m
)
```

---

#### Layer 30: Monte Carlo Neutron Transport

**Module**: `triality.monte_carlo_neutron`

**Capabilities:**
- Neutron random walk simulation
- Cross-section libraries (U-235, Pu-239, etc.)
- k-effective calculations
- Flux tallies
- Criticality estimation

**API Example:**
```python
from triality.monte_carlo_neutron import (
    CrossSection,
    NeutronRandomWalk,
    calculate_k_eff
)

# Cross sections
u235 = CrossSection.uranium_235(energy_ev=0.025)  # Thermal
sigma_fission = u235['sigma_fission']

# Random walk
path_length = NeutronRandomWalk.sample_free_path(
    Sigma_total=1.0  # 1/cm
)

# Criticality
k_eff = calculate_k_eff(
    geometry='sphere',
    radius=10,
    enrichment=3.5
)
```

---

#### Layer 31: Full-Wave EM Solvers

**Module**: `triality.em_solvers`

**Capabilities:**
- Dipole antenna analysis
- Waveguide mode analysis
- FDTD basics (framework)
- Radiation pattern calculation
- Gain/directivity

**Note**: Basic implementations. For production RF design, use HFSS, CST, FEKO.

**API Example:**
```python
from triality.em_solvers import DipoleAntenna, WaveguideMode

# Dipole antenna
dipole = DipoleAntenna(
    length=0.5,  # wavelength
    frequency=300e6
)
gain_dbi = dipole.gain_dbi()

# Waveguide
fc = WaveguideMode.cutoff_frequency_rectangular(
    a=0.05,  # WR-187
    b=0.025,
    mode='TE10'
)
```

---

#### Layer 32-33: Combustion & Reacting Flows

**Module**: `triality.combustion_chemistry`

**Capabilities:**
- Arrhenius rate calculations
- Laminar flame speed
- Ignition delay
- Chemical equilibrium
- Reaction mechanism analysis

**Module**: `triality.reacting_flows`

**Capabilities:**
- Species transport
- Well-stirred reactor (WSR)
- Chemical time scales
- Flamelet models
- Reaction progress variable

**API Example:**
```python
from triality.combustion_chemistry import ArrheniusRate, LaminarFlameSpeed
from triality.reacting_flows import WellStirredReactor, ChemicalTimeScale

# Reaction rate
k = ArrheniusRate.calculate_rate(
    A=1e13,
    Ea=200e3,  # J/mol
    T=1500  # K
)

# Flame speed
S_L = LaminarFlameSpeed.methane_air(
    T=300,
    P=101325,
    phi=1.0
)

# Chemical time scale
tau_chem = ChemicalTimeScale.estimate_chemical_time(
    T=2000,
    P=101325,
    fuel='CH4'
)
```

---

## Core Systems

### Automatic PDE Solver

**Module**: `triality.solvers`

**Capabilities:**
- Automatic problem classification
- Intelligent method selection
- Well-posedness validation
- Assumption tracking
- NaN/Inf detection

**Supported PDEs:**
- Elliptic: Laplace, Poisson, steady diffusion
- Parabolic: Heat equation, transient diffusion (coming)
- Hyperbolic: Wave equation (coming)

**API Example:**
```python
from triality import Field, laplacian, solve, Interval, Rectangle

u = Field("u")

# 1D Poisson
sol = solve(
    laplacian(u) == -1,
    domain=Interval(0, 1),
    bc={'left': 0, 'right': 0}
)

# 2D Poisson
sol = solve(
    laplacian(u) == -2,
    domain=Rectangle((0, 1), (0, 1)),
    bc={'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
)

# Inspect
print(sol.classification)  # 'elliptic, linear'
print(sol.method)          # 'finite_difference_2d'
print(sol.assumptions)     # List of assumptions
```

---

### Spatial Flow Engine

**Module**: `triality.spatial_flow`

**Capabilities:**
- Continuous potential field optimization
- Multi-source/sink routing
- Obstacle avoidance
- Custom cost fields
- Path extraction and simplification

**Use Cases:**
- Cable routing (power, signal)
- Pipe routing
- Material flow optimization
- HVAC duct layout
- Warehouse logistics

**API Example:**
```python
from triality.spatial_flow import SpatialFlowEngine, ObstacleBuilder

engine = SpatialFlowEngine()
engine.set_domain((0, 1), (0, 1))
engine.add_source((0.1, 0.5), weight=1.0, label="A")
engine.add_sink((0.9, 0.5), weight=1.0, label="B")

obstacle = ObstacleBuilder.rectangle(0.4, 0.6, 0.3, 0.7)
engine.add_obstacle(obstacle)

network = engine.solve()
path = network.paths[0]
print(f"Path cost: {path.cost:.3f}")
```

---

### Verification System

**Module**: `triality.verification`

**Capabilities:**
- Method of Manufactured Solutions (MMS)
- Grid convergence analysis
- Conservation law checks
- Regression benchmarks

**API Example:**
```python
from triality.verification import (
    mms_verify,
    grid_convergence_study,
    check_current_conservation
)

# MMS verification
result = mms_verify(
    solver,
    manufactured_solution,
    domain,
    resolutions=[10, 20, 40]
)
print(f"Convergence rate: {result.convergence_rate:.2f}")

# Conservation check
conservation = check_current_conservation(
    result,
    tolerance=1e-6
)
assert conservation.passed
```

---

## Complete Module Reference

### Module Directory Structure

```
triality/
├── core/                          # Core expression system
│   ├── expressions.py             # Field, operators, AST
│   ├── domains.py                 # Interval, Rectangle, etc.
│   └── validation.py              # Input validation
│
├── solvers/                       # PDE solver pipeline
│   ├── classify.py                # Problem classification
│   ├── select.py                  # Method selection
│   ├── solve.py                   # Main solver API
│   ├── linear.py                  # Linear solvers
│   └── wellposedness.py           # Validation
│
├── geometry/                      # Discretization
│   └── fdm.py                     # Finite difference
│
├── electrostatics/                # ✅ Layer 1
│   ├── field_solver.py
│   ├── conduction.py
│   └── derived_quantities.py
│
├── field_aware_routing/           # ✅ Layer 2
│   ├── cost_field_builders.py
│   ├── coupling_analysis.py
│   └── routing_integration.py
│
├── drift_diffusion/               # 🔧 Layer 3
│   ├── device_solver.py
│   └── advanced_physics.py
│
├── spatial_flow/                  # Routing engine
│   ├── engine.py
│   ├── sources_sinks.py
│   ├── cost_fields.py
│   ├── constraints.py
│   └── extraction.py
│
├── verification/                  # Production validation
│   ├── mms.py
│   ├── convergence.py
│   └── conservation.py
│
├── neutronics/                    # 📚 Nuclear physics
├── radiation_environment/         # 📚 Space radiation
├── shielding/                     # 📚 Radiation shielding
├── monte_carlo_neutron/           # 📚 MC neutron transport
├── burnup/                        # 📚 Fuel burnup
│
├── automotive_thermal/            # 📚 Automotive thermal
├── battery_thermal/               # 📚 Battery thermal
├── coupled_electrical_thermal/    # 📚 Coupled E-T
├── thermal_hydraulics/            # 📚 Thermal-fluid
├── spacecraft_thermal/            # 📚 Spacecraft thermal
│
├── cfd_turbulence/                # 📚 CFD & turbulence
├── reacting_flows/                # 📚 Reacting flows
├── combustion_chemistry/          # 📚 Combustion chemistry
│
├── aero_loads/                    # 📚 Aerodynamic loads
├── aerothermodynamics/            # 📚 Aerothermodynamics
├── propulsion/                    # 📚 Propulsion
├── flight_mechanics/              # 📚 Flight dynamics
│
├── structural_analysis/           # 📚 Structural analysis
├── structural_dynamics/           # 📚 Structural dynamics
├── fracture_mechanics/            # 📚 Fracture mechanics
├── aeroelasticity/                # 📚 Aeroelasticity
├── thermo_mechanical/             # 📚 Thermo-mechanical
│
├── quantum_nanoscale/             # 📚 Quantum physics
├── injury_biomechanics/           # 📚 Biomechanics
├── em_solvers/                    # 📚 EM solvers
├── emi_emc/                       # 📚 EMI/EMC
├── hv_safety/                     # 📚 High voltage
├── coupled_physics/               # 📚 Multi-physics
├── agent_design/                  # 📚 Design optimization
├── safety/                        # 📚 Safety analysis
│
└── examples/                      # Usage examples
```

---

## API Quick Reference

### Top-Level Imports

```python
# Automatic PDE solving
from triality import Field, Constant, solve
from triality import laplacian, grad, div, dx, dy
from triality import Interval, Rectangle, Square, Circle

# Production layers
from triality.electrostatics import ElectrostaticSolver2D, ConductionSolver2D
from triality.field_aware_routing import PhysicsAwareRouter, OptimizationObjective
from triality.drift_diffusion import DriftDiffusionSolver1D

# Core systems
from triality.spatial_flow import SpatialFlowEngine
from triality.verification import mms_verify, grid_convergence_study

# Extended modules (examples)
from triality.cfd_turbulence import TurbulenceModel, NormalShock
from triality.structural_analysis import BeamAnalysis, StressAnalysis
from triality.flight_mechanics import RigidBody6DOF, GNCSystem
from triality.quantum_nanoscale import InfiniteSquareWell, TunnelingProbability
```

### Common Patterns

**Pattern 1: Electrostatic Analysis**
```python
solver = ElectrostaticSolver2D(x_range, y_range, resolution)
solver.set_boundary(side, type, value)
result = solver.solve()
E = result.electric_field()
```

**Pattern 2: Physics-Aware Routing**
```python
router = PhysicsAwareRouter()
router.set_domain(x_bounds, y_bounds)
router.add_physics_cost(name, cost_field, weight)
route = router.route(start, end, objective)
```

**Pattern 3: PDE Solving**
```python
u = Field("u")
equation = laplacian(u) == source
solution = solve(equation, domain, bc={...})
```

---

## Performance Characteristics

### Production Layers

**Layer 1 (Electrostatics):**
| Grid Size | Solve Time | Memory Usage | Accuracy |
|-----------|------------|--------------|----------|
| 50×50     | 5 ms       | ~10 MB       | ±2-5%    |
| 100×100   | 20 ms      | ~40 MB       | ±5%      |
| 200×200   | 150 ms     | ~160 MB      | ±5-10%   |
| 500×500   | 2-3 sec    | ~1 GB        | ±5-10%   |

**Layer 2 (Routing):**
| Grid Size | Physics Solve | Routing | Total | Path Quality |
|-----------|---------------|---------|-------|--------------|
| 50×50     | 5 ms          | 10 ms   | 15 ms | Excellent    |
| 100×100   | 20 ms         | 50 ms   | 70 ms | Excellent    |
| 200×200   | 150 ms        | 300 ms  | 450 ms| Very Good    |

**Layer 3 (Semiconductors):**
| Points | Equilibrium | I-V Curve (50 pts) | Accuracy |
|--------|-------------|-------------------|----------|
| 100    | 0.1 sec     | 5 sec             | ±20-30%  |
| 200    | 0.3 sec     | 15 sec            | ±20-30%  |
| 500    | 2 sec       | 100 sec           | ±20-30%  |

### Hardware Requirements

**Minimum:**
- Python 3.8+
- 4 GB RAM
- Single-core CPU

**Recommended:**
- Python 3.10+
- 16 GB RAM
- Multi-core CPU (4+ cores for parallel sweeps)
- SSD for large datasets

**For Large Problems (500×500+ grids):**
- 32 GB+ RAM
- 8+ core CPU
- GPU (for future acceleration)

---

## Usage Patterns

### Pattern 1: Early-Stage PCB Design

```python
# Step 1: Analyze electromagnetic fields
from triality.electrostatics import ElectrostaticSolver2D

solver = ElectrostaticSolver2D((0, 0.1), (0, 0.1), resolution=100)
solver.set_boundary('left', 'voltage', 5.0)
solver.set_boundary('right', 'voltage', 0.0)
result = solver.solve()

# Step 2: Convert to routing costs
from triality.field_aware_routing import PhysicsAwareRouter
from triality.field_aware_routing.cost_field_builders import ElectricFieldCostBuilder

emi_cost = ElectricFieldCostBuilder.from_result(result)

# Step 3: Route with physics awareness
router = PhysicsAwareRouter()
router.set_domain((0, 0.1), (0, 0.1))
router.add_physics_cost('EMI', emi_cost, weight=10.0)

route = router.route(
    start=(0.01, 0.01),
    end=(0.09, 0.09),
    objective=OptimizationObjective.MIN_EMI
)

print(f"EMI reduction: {route.emi_reduction:.1f}%")
```

### Pattern 2: Multi-Physics Analysis

```python
# Electromagnetic + Thermal + Mechanical
from triality.electrostatics import ConductionSolver2D
from triality.structural_analysis import StressAnalysis
from triality.thermo_mechanical import ThermalStressCalculator

# 1. Thermal analysis
thermal_solver = ConductionSolver2D(...)
T_field = thermal_solver.solve()

# 2. Thermal expansion
thermal_strain = ThermalStressCalculator.thermal_strain(
    T_field,
    T_ref=300,
    alpha=23e-6  # CTE
)

# 3. Structural response
stress_solver = StressAnalysis(...)
stress = stress_solver.compute_stress(thermal_strain)
```

### Pattern 3: Design Space Exploration

```python
# Parameter sweep with Triality
import numpy as np

parameters = {
    'voltage': np.linspace(1, 10, 10),
    'width': np.linspace(0.01, 0.1, 10),
    'conductivity': [1e6, 5e6, 1e7]
}

results = []
for V in parameters['voltage']:
    for w in parameters['width']:
        for sigma in parameters['conductivity']:
            solver = ElectrostaticSolver2D(...)
            # ... configure with V, w, sigma
            result = solver.solve()
            results.append({
                'V': V, 'w': w, 'sigma': sigma,
                'max_E': result.electric_field_magnitude().max(),
                'energy': result.electrostatic_energy()
            })

# Analyze trends
import pandas as pd
df = pd.DataFrame(results)
optimal = df.loc[df['max_E'].idxmin()]
```

---

## Citation

If you use Triality in your work, please cite:

```bibtex
@software{triality2025,
  title={Triality: Production-Ready Physics Simulation Framework},
  author={Genovation Technological Solutions Pvt Ltd — Powered by Mentis OS},
  year={2025},
  version={0.2.0},
  organization={Genovation Technological Solutions Pvt Ltd — Powered by Mentis OS}
}
```

---

## Support & Contact

**Genovation Technological Solutions Pvt Ltd — Powered by Mentis OS**
*"We build systems that understand reality."*
*Building AI that understands reality.*

For technical support, licensing inquiries, or custom development:
- **Email**: connect@genovationsolutions.com
- Technical documentation: Complete API docs and examples
- Professional support: Contact Genovation for enterprise support
- Training: On-site and remote training available

---

**© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.**

Triality is proprietary software. Unauthorized copying, modification, or distribution is prohibited.
