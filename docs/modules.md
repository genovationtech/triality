# Modules Reference

Triality contains 40+ specialized physics modules. This document provides an overview of each.

## Observable Layer

All 16 runtime-capable modules have a registered **Observable Layer** that automatically derives domain-specific engineering quantities from solver output. Each module produces 5–12 ranked observables including peak values, safety margins, efficiency ratios, and pass/fail verdicts. Total: 126 observables, 10 with pass/fail thresholds, at 0.027 ms median compute time. See [API Reference](api_reference.md#observable-layer-trialityobservables) for the full list.

## Quick-Reference Table

| Module | Domain | Status | Key Class |
|---|---|---|---|
| `electrostatics` | EM | Production ✅ | `ElectrostaticSolver2D` |
| `field_aware_routing` | Routing | Production ✅ | `FieldAwareRouter` |
| `spatial_flow` | Routing | Production ✅ | `FlowEngine` |
| `em_solvers` | EM | Stable | `EMSolver` |
| `emi_emc` | EM | Stable | `EMIAnalyzer` |
| `hv_safety` | EM/Safety | Stable | `HVSafetyAnalyzer` |
| `rf_jamming` | EM/Defense | Stable | `RFJammingModel` |
| `coupled_electrical_thermal` | Multi-physics | Stable | `CoupledETSolver` |
| `thermal_hydraulics` | Thermal/Fluid | Stable | `ThermalHydraulicsSolver` |
| `battery_thermal` | Thermal | Stable | `BatteryThermalModel` |
| `automotive_thermal` | Thermal | Stable | `AutomotiveThermalModel` |
| `spacecraft_thermal` | Thermal | Stable | `SpacecraftThermalModel` |
| `aerothermodynamics` | Thermal/Aero | Stable | `AerothermodynamicsSolver` |
| `conjugate_heat_transfer` | Thermal/Fluid | Stable | `CHTSolver` |
| `pack_thermal` | Thermal | Stable | `PackThermalModel` |
| `structural_analysis` | Structural | Stable | `StructuralSolver` |
| `structural_dynamics` | Structural | Stable | `ModalAnalysis` |
| `fracture_mechanics` | Structural | Stable | `FractureMechanicsModel` |
| `thermo_mechanical` | Multi-physics | Stable | `ThermoMechanicalSolver` |
| `aeroelasticity` | Multi-physics | Stable | `AeroelasticityModel` |
| `cfd_turbulence` | Fluid | Stable | `CFDTurbulenceSolver` |
| `navier_stokes` | Fluid | Stable | `NavierStokesSolver` |
| `reacting_flows` | Fluid/Chem | Stable | `ReactingFlowSolver` |
| `combustion_chemistry` | Chemistry | Stable | `CombustionChemistrySolver` |
| `multiphase_vof` | Fluid | Stable | `MultiphaseVOFSolver` |
| `aero_loads` | Aerospace | Stable | `AeroLoadsCalculator` |
| `flight_mechanics` | Aerospace | Stable | `FlightMechanicsModel` |
| `propulsion` | Aerospace | Stable | `PropulsionModel` |
| `hypersonic_vehicle_simulation` | Aerospace | Stable | `HypersonicSimulator` |
| `uav_aerodynamics` | Aerospace/UAV | Stable | `UAVAerodynamicsModel` |
| `ballistic_trajectory` | Defense | Stable | `BallisticTrajectory` |
| `reentry_simulation` | Aerospace | Stable | `ReentrySimulator` |
| `monte_carlo_neutron` | Nuclear | Stable | `MonteCarloNeutronTransport` |
| `neutronics` | Nuclear | Stable | `NeutronicsCalculator` |
| `shielding` | Nuclear | Stable | `ShieldingCalculator` |
| `burnup` | Nuclear | Stable | `BurnupCalculator` |
| `reactor_transient` | Nuclear | Stable | `ReactorTransientSolver` |
| `quantum_optimization` | Quantum | Experimental | `QuantumOptimizer` |
| `plasma` | Plasma | Stable | `PlasmaModel` |
| `sensing` | Sensing | Stable | `MultiSensorFusion` |
| `uncertainty_quantification` | Analysis | Stable | `UQAnalysis` |
| `verification` | Testing | Stable | `VerificationSuite` |
| `geospatial` | Geo | Stable | `GeospatialRouter` |

---

## Electromagnetic Modules

### `electrostatics`
Solves the electrostatic Poisson equation on 2D Cartesian grids.

**Use cases:** PCB voltage distribution, capacitor analysis, electrode design, dielectric field analysis.

**Key classes:**
- `ElectrostaticSolver2D(x_range, y_range, resolution, epsilon_r)`
- `ConductionSolver2D(x_range, y_range, resolution, conductivity)`
- `ElectrostaticResult` — voltage, E-field, energy density, visualization

### `em_solvers`
Full electromagnetic solvers including time-domain (FDTD) and frequency-domain (FEM) methods.

**Use cases:** Antenna radiation, waveguide analysis, cavity resonance, EMC pre-compliance.

### `emi_emc`
EMI/EMC analysis: coupling coefficients, common/differential mode currents, shielding effectiveness.

**Use cases:** PCB EMC design, cable harness qualification, regulatory pre-compliance checks.

### `hv_safety`
High-voltage safety analysis: creepage distances, clearances, partial discharge inception, arc flash.

**Use cases:** Power electronics design, EV battery pack safety, industrial HV equipment.

### `rf_jamming`
RF jamming effectiveness modeling: effective radiated power, jamming-to-signal ratio, terrain effects.

### `electronic_countermeasures`
ECM/ECCM system modeling: noise jamming, deceptive jamming, frequency-agile waveforms.

---

## Thermal Modules

### `thermal_hydraulics`
Coupled thermal-hydraulic analysis: 1D/2D pipe networks, heat exchangers, cooling systems.

**Key class:** `ThermalHydraulicsSolver(geometry, fluid, T_inlet, P_inlet, flow_rate)`

### `battery_thermal`
Electrochemical-thermal battery cell and pack models.

**Key class:** `BatteryThermalModel(cell_type, n_cells, capacity_Ah, cooling)`

Supported chemistries: NMC, LFP, NCA, LTO, solid-state.

### `automotive_thermal`
Vehicle-level thermal management: HVAC, powertrain cooling, cabin thermal comfort.

### `spacecraft_thermal`
Orbital thermal analysis: solar flux, albedo, Earth IR, radiator sizing, eclipse cycles.

### `aerothermodynamics`
Aerodynamic heating: stagnation point heat flux, shock-layer radiation, ablation modeling.

### `conjugate_heat_transfer`
Coupled solid-fluid heat transfer: fin analysis, heat sinks, thermal interface materials.

### `pack_thermal`
Battery pack thermal management: cell-level to pack-level thermal modeling, cooling plate design.

---

## Structural Modules

### `structural_analysis`
Linear and nonlinear static structural analysis using FEM.

**Use cases:** Stress/strain fields, deflection, factor of safety analysis.

### `structural_dynamics`
Modal analysis, frequency response, transient dynamics, seismic analysis.

**Key outputs:** Natural frequencies, mode shapes, dynamic amplification factors.

### `fracture_mechanics`
Fracture mechanics: stress intensity factors (K_I, K_II, K_III), J-integral, crack growth (Paris law).

### `thermo_mechanical`
Coupled thermal-structural analysis: thermal stress, thermal fatigue, CTE mismatch.

### `aeroelasticity`
Flutter analysis, divergence speed, gust response, control reversal.

---

## Fluid Dynamics Modules

### `cfd_turbulence`
RANS turbulence modeling: k-ε, k-ω SST. 2D steady-state external/internal flows.

### `navier_stokes`
Incompressible Navier-Stokes: 2D cavity flow, channel flow, flow past obstacles.

### `reacting_flows`
Combustion in flows: premixed and diffusion flames, species transport, heat release.

### `combustion_chemistry`
Reduced kinetic mechanisms: ignition delay, laminar flame speed, extinction.

### `multiphase_vof`
Volume-of-fluid multiphase: free surface flows, sloshing, droplet dynamics.

---

## Aerospace & Propulsion Modules

### `aero_loads`
Aerodynamic load estimation: lift, drag, moment curves; gust loads; maneuver envelope.

### `flight_mechanics`
6-DOF flight dynamics: equations of motion, trim analysis, stability derivatives.

### `propulsion`
Propulsion system analysis: turbofan thermodynamic cycle, electric motor-propeller, rocket nozzle.

### `hypersonic_vehicle_simulation`
Hypersonic vehicle modeling: bow shock, plasma blackout, aerothermodynamic heating, guidance.

### `uav_aerodynamics`
Small UAV aerodynamics: blade element momentum theory, rotor-fuselage interaction, ground effect.

### `ballistic_trajectory`
Exterior ballistics: point-mass trajectory, drag models (G1-G8), wind effects, coriolis correction.

### `reentry_simulation`
Atmospheric re-entry: ablative heat shield, trajectory optimization, landing footprint.

---

## Nuclear & Radiation Modules

### `neutronics`
Multi-group diffusion theory: k-effective, flux distribution, control rod worth.

### `monte_carlo_neutron`
Monte Carlo particle transport: neutron/photon transport, dose calculation, shielding design.

### `shielding`
Analytical/semi-empirical shielding: gamma attenuation, neutron moderation, buildup factors.

### `burnup`
Fuel depletion: Bateman equations for actinide and fission product buildup, depletion chains.

### `reactor_transient`
Point kinetics and space-time kinetics: reactivity insertions, SCRAM analysis, decay heat.

---

## Advanced Modules

### `quantum_optimization`
Quantum-inspired optimization algorithms: QAOA, VQE, quantum annealing simulation.

**Use cases:** Combinatorial design optimization, route optimization, material design.

### `plasma`
Plasma physics: MHD equilibrium, wave dispersion, instability analysis.

### `sensing`
Multi-sensor modeling: radar (RCS), IR signature, acoustic signature, passive detection.

### `geospatial`
Geographic routing: terrain-aware path planning, coordinate transforms, satellite coverage.

### `uncertainty_quantification`
UQ methods: Monte Carlo, Latin hypercube sampling, Sobol sensitivity indices, polynomial chaos.

### `verification`
V&V framework: comparison against analytical solutions, convergence testing, benchmark suite.

---

## Module Development Status

| Status | Meaning |
|---|---|
| Production ✅ | Full test coverage, used in production workflows |
| Stable | Core functionality tested, API stable |
| Experimental | Under active development, API may change |
| Framework | Architecture complete, numerical validation ongoing |
