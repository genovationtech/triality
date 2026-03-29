# Triality Fidelity & Coupling Audit

This is a fresh repo-local re-audit of every importable `triality` module found in the current working tree. The classification is evidence-based and intentionally conservative: when code and docs disagree, the lower-confidence / lower-fidelity interpretation wins unless the repository contains clear benchmark or production-readiness evidence.

> Note: I attempted to refetch `main`, but this clone currently has no configured Git remote, so the audit below reflects the latest code available in this working tree rather than a newly fetched upstream branch.

## Fidelity tiers

- **L0** → heuristic / screening / infrastructure
- **L1** → reduced-order physics
- **L2** → engineering-grade
- **L3** → high-fidelity / near-CFD/FEM
- **L4** → validated / benchmarked

## Coupling maturity rubric

- **M0** → isolated or orchestration-only; no standardized physics exchange
- **M1** → one-way/ad hoc exchange; partial unit/time handling
- **M2** → direct module feedthrough with partial standardization of units/time bases
- **M3** → direct canonical exchange with explicit coupling iteration / stabilization
- **M4** → benchmarked closed-loop multi-physics coupling

## Observable Layer coverage

All 16 runtime-capable modules have a registered `ObservableSet` in `triality/observables.py` (100% coverage). The Observable Layer derives domain-specific engineering quantities (126 total, 5–12 per module) from solved `PhysicsState` fields at negligible cost (0.027 ms median). Observables include 10 pass/fail thresholds with signed safety margins for structural yield, thermal runaway, breakdown clearance, and other critical engineering limits. This layer does not affect fidelity tier classification — it is algebraic post-processing, not additional physics computation.

## Method used for this re-check

The audit combines four evidence sources for each module:

1. `solver.py` / primary Python implementation to see whether the module exports `PhysicsState`, uses canonical field metadata, exposes SI conversion hooks, and contains convergence / relaxation logic.
2. Module README / status markdown files when they exist, especially for documented production layers such as drift-diffusion, HV safety, geospatial, spatial flow, and verification.
3. Cross-cutting framework files in `triality/core/` and `triality/coupling/` that define the real coupling contract.
4. Conservative manual overrides for modules whose names or docs clearly indicate infrastructure-only, reduced-order, or high-fidelity roles.

## Portfolio snapshot

- **L0**: 10 modules (heuristic / screening / infrastructure)
- **L1**: 15 modules (reduced-order physics)
- **L2**: 33 modules (engineering-grade)
- **L3**: 45 modules (high-fidelity / near-CFD/FEM)
- **L4**: 1 modules (validated / benchmarked)

- **M0**: 2 modules (isolated or orchestration-only; no standardized physics exchange)
- **M1**: 55 modules (one-way/ad hoc exchange; partial unit/time handling)
- **M2**: 42 modules (direct module feedthrough with partial standardization of units/time bases)
- **M3**: 5 modules (direct canonical exchange with explicit coupling iteration / stabilization)
- **M4**: 0 modules (benchmarked closed-loop multi-physics coupling)

## Answers to the coupling questions

- **Can outputs feed directly into other modules?** Yes for 96/104 modules. The strongest enabler is the shared `PhysicsState` pattern and the canonical field / coupling helpers in `triality.core` and `triality.coupling`.
- **Are units consistent?** Mostly but not universally. 98/104 modules show either explicit SI/canonical-unit handling or enough metadata to mark them at least **Partial**.
- **Are time scales handled?** 91/104 modules expose transient, timestep, trajectory, or integration language. Multi-rate coordination is still uncommon even where time dependence exists.
- **Is coupling stable?** Only partly. 101/104 modules show explicit iteration / convergence / relaxation hooks, but very few modules demonstrate benchmarked closed-loop multiphysics stability.

## High-level conclusions

- **Best current advantage:** coupling is materially ahead of a typical grab-bag physics repo because many modules emit reusable state objects, and the repo has explicit shared coupling abstractions rather than only ad hoc file/JSON exchange.
- **Main weakness:** there is still a big difference between *connectable* and *robustly co-simulatable*. Many modules can feed outputs downstream, but only a small subset show explicit stabilization logic for tight two-way coupling.
- **Most credible high-fidelity pockets:** CFD / transport / plasma / TPS-style modules, the drift-diffusion layer, and the documented HV safety layer.
- **Validation gap remains:** outside of the HV safety docs and specialized verification / fixture infrastructure, most modules still read as engineering-grade tools for design iteration rather than formally benchmarked end-state solvers.

## Modules grouped by fidelity level

### L0 — heuristic / screening / infrastructure

**Why these modules land here:** support/infrastructure/orchestration rather than forward-physics solvers.

**Modules (10):** `agent_design`, `core`, `coupling`, `diagnostics`, `geometry`, `ground_truth`, `safety_logic`, `sensor_fusion`, `solvers`, `verification`

- `agent_design`: orchestration/optimization layer, not a forward-physics solver; optimizer/orchestration layer; does not expose standardized physics exchange.
- `core`: shared data model, units, and coupling infrastructure; canonical PhysicsState + unit conversion + coupling engine.
- `coupling`: shared coupling infrastructure; shared canonical coupling abstractions.
- `diagnostics`: monitoring/post-processing utilities; consumes or supports other modules without canonical closed-loop exchange.
- `geometry`: geometry/FDM support utilities; support utilities only.
- `ground_truth`: fixtures/reference data rather than forward simulation; direct feedthrough plus partial standardization and convergence controls.
- `safety_logic`: rules/logic wrapper around physics outputs; direct feedthrough plus partial standardization and convergence controls.
- `sensor_fusion`: state-estimation layer, not first-principles physics; direct feedthrough exists, but standardization/stabilization are limited.
- `solvers`: solver-selection/backend infrastructure; consumes or supports other modules without canonical closed-loop exchange.
- `verification`: verification harness, not a physics module; benchmarks consume solver outputs but do not provide reusable runtime coupling; README.md contains benchmark and regression infrastructure, but also notes key verification tests are still failing..

### L1 — reduced-order physics

**Why these modules land here:** reduced-order, surrogate, or system-level physics abstractions.

**Modules (15):** `electrochemistry`, `emi_emc`, `geospatial`, `hypersonic_vehicle_simulation`, `injury_biomechanics`, `quantum_nanoscale`, `quantum_optimization`, `rcs_shaping`, `safety`, `sensing`, `sensors_actuators`, `spacecraft_thermal`, `spatial_flow`, `swarm_dynamics`, `uncertainty`

- `electrochemistry`: cell-level reduced-order battery chemistry; direct feedthrough exists, but standardization/stabilization are limited.
- `emi_emc`: reduced-order or application-level physics logic; consumes or supports other modules without canonical closed-loop exchange.
- `geospatial`: physics-inspired feasibility/routing with optional real-data hooks; direct feedthrough exists, but standardization/stabilization are limited; README.md documents production data integration, but this module is logistics/feasibility rather than first-principles physics..
- `hypersonic_vehicle_simulation`: system-level vehicle synthesis over higher-fidelity submodels; direct feedthrough plus partial standardization and convergence controls.
- `injury_biomechanics`: lumped injury metrics and surrogate injury models; direct feedthrough exists, but standardization/stabilization are limited.
- `quantum_nanoscale`: reduced-order nanoscale / quantum device models; direct feedthrough plus partial standardization and convergence controls.
- `quantum_optimization`: optimization workflow, not direct field solve; direct feedthrough plus partial standardization and convergence controls.
- `rcs_shaping`: shaping/surrogate RCS estimates rather than full-wave solve; direct feedthrough exists, but standardization/stabilization are limited.
- `safety`: system safety / kinetics abstractions; direct feedthrough plus partial standardization and convergence controls.
- `sensing`: trade-study/feasibility architecture rather than full sensor scene simulation; direct feedthrough exists, but standardization/stabilization are limited.
- `sensors_actuators`: component-level reduced-order hardware models; direct feedthrough exists, but standardization/stabilization are limited.
- `spacecraft_thermal`: network-style spacecraft thermal abstractions; direct feedthrough plus partial standardization and convergence controls.
- `spatial_flow`: routing engine built on potential-field abstractions; consumes or supports other modules without canonical closed-loop exchange.
- `swarm_dynamics`: agent/surrogate dynamics; direct feedthrough exists, but standardization/stabilization are limited.
- `uncertainty`: uncertainty wrapper around other physics solvers; direct feedthrough exists, but standardization/stabilization are limited.

### L2 — engineering-grade

**Why these modules land here:** engineering-grade modules that solve useful physics but do not yet show strong evidence of near-CFD/FEM depth or broad validation.

**Modules (33):** `aero_loads`, `aerothermodynamics`, `battery_abuse_simulation`, `battery_thermal`, `combustion_chemistry`, `counter_stealth`, `coupled_electrical_thermal`, `coupled_physics`, `directed_energy`, `electronic_countermeasures`, `field_aware_routing`, `flight_mechanics`, `fracture_mechanics`, `gas_generation`, `ir_signature`, `material_erosion`, `pack_thermal`, `power_processing`, `propulsion`, `radar_waveforms`, `radiation_environment`, `reactivity_feedback`, `reactor_transient`, `reentry_simulation`, `rf_jamming`, `sheath_model`, `signature_simulation`, `thermal_runaway_kinetics`, `thermo_mechanical`, `uav_aerodynamics`, `uav_navigation`, `uav_propulsion`, `warhead_lethality`

- `aero_loads`: forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls.
- `aerothermodynamics`: forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls.
- `battery_abuse_simulation`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `battery_thermal`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `combustion_chemistry`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `counter_stealth`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `coupled_electrical_thermal`: forward-physics solver exports reusable simulation state; direct two-way coupling with convergence / relaxation logic.
- `coupled_physics`: forward-physics solver exports reusable simulation state; explicit coupled multi-solver orchestrator with convergence logic.
- `directed_energy`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `electronic_countermeasures`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `field_aware_routing`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `flight_mechanics`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `fracture_mechanics`: forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls.
- `gas_generation`: forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls.
- `ir_signature`: forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls.
- `material_erosion`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `pack_thermal`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `power_processing`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `propulsion`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `radar_waveforms`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `radiation_environment`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `reactivity_feedback`: forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls.
- `reactor_transient`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `reentry_simulation`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `rf_jamming`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `sheath_model`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `signature_simulation`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `thermal_runaway_kinetics`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `thermo_mechanical`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `uav_aerodynamics`: forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls.
- `uav_navigation`: forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls.
- `uav_propulsion`: forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited.
- `warhead_lethality`: forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls.

### L3 — high-fidelity / near-CFD/FEM

**Why these modules land here:** higher-fidelity numerical methods or explicitly high-fidelity domain solvers.

**Modules (45):** `acoustic_signature`, `aeroelasticity`, `automotive_thermal`, `ballistic_trajectory`, `burnup`, `cfd_turbulence`, `collision_avoidance`, `combustor_simulation`, `conjugate_heat_transfer`, `contact_line`, `counter_uas`, `drift_diffusion`, `electrostatics`, `em_solvers`, `microlayer`, `missile_defense`, `missile_guidance`, `monte_carlo_neutron`, `multiphase_vof`, `navier_stokes`, `neutronics`, `nonequilibrium`, `nucleate_boiling`, `particle_in_cell`, `passive_detection`, `phase_change`, `plasma_fluid`, `plasma_thruster_simulation`, `pollutant_formation`, `propagation`, `radar_absorbing_materials`, `radar_detection`, `radiation_transport`, `reacting_flows`, `reactor_transients`, `shielding`, `soot_model`, `structural_analysis`, `structural_dynamics`, `surface_tension`, `thermal_hydraulics`, `thermal_signature_coupling`, `tps_ablation`, `tracking`, `turbulence_combustion`

- `acoustic_signature`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `aeroelasticity`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `automotive_thermal`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `ballistic_trajectory`: code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited.
- `burnup`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `cfd_turbulence`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `collision_avoidance`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `combustor_simulation`: combustor solver with reacting-flow style iterative coupling; direct feedthrough exists, but standardization/stabilization are limited.
- `conjugate_heat_transfer`: explicit CHT coupling and outer iterations; direct feedthrough plus partial standardization and convergence controls.
- `contact_line`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `counter_uas`: code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited.
- `drift_diffusion`: Layer 3 semiconductor solver with dedicated device physics docs; direct feedthrough plus partial standardization and convergence controls; README_LAYER3.md says 'production-ready' while IMPLEMENTATION_STATUS.md still documents numerical-refinement risk; rated conservatively..
- `electrostatics`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `em_solvers`: code/docs indicate higher-fidelity numerical method family; consumes or supports other modules without canonical closed-loop exchange.
- `microlayer`: code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited.
- `missile_defense`: code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited.
- `missile_guidance`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `monte_carlo_neutron`: Monte Carlo neutronics; direct feedthrough plus partial standardization and convergence controls.
- `multiphase_vof`: VOF multiphase flow; direct feedthrough exists, but standardization/stabilization are limited.
- `navier_stokes`: Navier-Stokes CFD solver; direct feedthrough plus partial standardization and convergence controls.
- `neutronics`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `nonequilibrium`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `nucleate_boiling`: code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited.
- `particle_in_cell`: PIC plasma / charged-particle method; direct feedthrough plus partial standardization and convergence controls.
- `passive_detection`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `phase_change`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `plasma_fluid`: coupled plasma-fluid solver; direct feedthrough exists, but standardization/stabilization are limited.
- `plasma_thruster_simulation`: high-fidelity plasma thruster simulation; direct feedthrough exists, but standardization/stabilization are limited.
- `pollutant_formation`: code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited.
- `propagation`: code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited.
- `radar_absorbing_materials`: code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited.
- `radar_detection`: code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited.
- `radiation_transport`: transport/radiation PDEs with dedicated solver; direct feedthrough plus partial standardization and convergence controls.
- `reacting_flows`: code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited.
- `reactor_transients`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `shielding`: code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited.
- `soot_model`: code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited.
- `structural_analysis`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `structural_dynamics`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `surface_tension`: code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls.
- `thermal_hydraulics`: coupled thermal-fluid engineering solver stack; tight thermal-fluid iteration with SI conversion hooks.
- `thermal_signature_coupling`: explicit thermal/signature coupled analysis; direct feedthrough plus partial standardization and convergence controls.
- `tps_ablation`: TPS ablation / high-enthalpy thermal coupling; direct feedthrough plus partial standardization and convergence controls.
- `tracking`: code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited.
- `turbulence_combustion`: combustion-turbulence coupling; direct feedthrough exists, but standardization/stabilization are limited.

### L4 — validated / benchmarked

**Why these modules land here:** modules with explicit production-ready / benchmarked evidence in repository docs.

**Modules (1):** `hv_safety`

- `hv_safety`: module documentation explicitly claims production-ready / tested operation; direct feedthrough plus partial standardization and convergence controls; README_LAYER4.md explicitly marks the module production-ready with 17/17 tests passing..

## What is left / incomplete / risky

- **No true M4 stack yet.** The repo has meaningful coupling infrastructure, but no module set currently demonstrates benchmarked closed-loop multiphysics coupling with durable regression evidence.
- **A lot of coupling is still immature.** All `M0` and `M1` modules still need clearer boundary contracts, stronger stabilization logic, or better canonical field/unit exchange before they can be treated as robust plug-and-play coupled modules.
- **Contradictory status docs still exist.** `drift_diffusion` contains both a 'production-ready' README and a separate implementation-status file that still describes refinement risk; `verification` contains benchmark infrastructure but also documents failing verification tests.
- **Some modules still look incomplete at the interface level.** The modules below are currently missing at least one desirable audit property such as direct feedthrough, explicit unit handling, time-scale handling, or strong/stable coupling evidence.
- **Modules still needing follow-up (62):** `aero_loads`, `agent_design`, `ballistic_trajectory`, `battery_abuse_simulation`, `battery_thermal`, `combustion_chemistry`, `combustor_simulation`, `counter_stealth`, `counter_uas`, `diagnostics`, `directed_energy`, `electrochemistry`, `electronic_countermeasures`, `electrostatics`, `em_solvers`, `emi_emc`, `field_aware_routing`, `flight_mechanics`, `geometry`, `geospatial`, `injury_biomechanics`, `material_erosion`, `microlayer`, `missile_defense`, `multiphase_vof`, `nucleate_boiling`, `pack_thermal`, `plasma_fluid`, `plasma_thruster_simulation`, `pollutant_formation`, `power_processing`, `propagation`, `propulsion`, `quantum_nanoscale`, `quantum_optimization`, `radar_absorbing_materials`, `radar_detection`, `radar_waveforms`, `radiation_environment`, `rcs_shaping`, `reacting_flows`, `reactor_transient`, `reentry_simulation`, `rf_jamming`, `sensing`, `sensor_fusion`, `sensors_actuators`, `sheath_model`, `shielding`, `signature_simulation`, `solvers`, `soot_model`, `spatial_flow`, `structural_analysis`, `swarm_dynamics`, `thermal_runaway_kinetics`, `thermo_mechanical`, `tracking`, `turbulence_combustion`, `uav_propulsion`, `uncertainty`, `verification`.
- **Validation remains thinner than fidelity.** Several modules legitimately look Level 3 by method family, but they are not automatically Level 4 because benchmark/validation evidence is still sparse.

## Module-by-module assessment

| Module | Fidelity | Coupling maturity | Direct feed? | Units consistent? | Time scales handled? | Coupling stable? | Evidence checked | Basis |
|---|---:|---:|---|---|---|---|---|---|
| `acoustic_signature` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `aero_loads` | L2 | M2 | Yes | Partial | No | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls |
| `aeroelasticity` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `aerothermodynamics` | L2 | M2 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls |
| `agent_design` | L0 | M0 | No | Partial | Yes | No | `__init__.py` | orchestration/optimization layer, not a forward-physics solver; optimizer/orchestration layer; does not expose standardized physics exchange |
| `automotive_thermal` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `ballistic_trajectory` | L3 | M1 | Yes | Partial | Yes | Partial | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited |
| `battery_abuse_simulation` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `battery_thermal` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `burnup` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `cfd_turbulence` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `collision_avoidance` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `combustion_chemistry` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `combustor_simulation` | L3 | M1 | Yes | Partial | Yes | Yes | `solver.py` | combustor solver with reacting-flow style iterative coupling; direct feedthrough exists, but standardization/stabilization are limited |
| `conjugate_heat_transfer` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | explicit CHT coupling and outer iterations; direct feedthrough plus partial standardization and convergence controls |
| `contact_line` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `core` | L0 | M3 | Yes | Partial | Yes | Yes | `__init__.py` | shared data model, units, and coupling infrastructure; canonical PhysicsState + unit conversion + coupling engine |
| `counter_stealth` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `counter_uas` | L3 | M1 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited |
| `coupled_electrical_thermal` | L2 | M3 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct two-way coupling with convergence / relaxation logic |
| `coupled_physics` | L2 | M3 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; explicit coupled multi-solver orchestrator with convergence logic |
| `coupling` | L0 | M3 | Yes | Partial | Yes | Yes | `__init__.py` | shared coupling infrastructure; shared canonical coupling abstractions |
| `diagnostics` | L0 | M1 | Partial | No | Yes | Yes | `__init__.py` | monitoring/post-processing utilities; consumes or supports other modules without canonical closed-loop exchange |
| `directed_energy` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `drift_diffusion` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py, IMPLEMENTATION_STATUS.md, README_LAYER3.md` | Layer 3 semiconductor solver with dedicated device physics docs; direct feedthrough plus partial standardization and convergence controls; README_LAYER3.md says 'production-ready' while IMPLEMENTATION_STATUS.md still documents numerical-refinement risk; rated conservatively. |
| `electrochemistry` | L1 | M1 | Yes | Partial | Yes | Partial | `solver.py` | cell-level reduced-order battery chemistry; direct feedthrough exists, but standardization/stabilization are limited |
| `electronic_countermeasures` | L2 | M1 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `electrostatics` | L3 | M2 | Yes | Partial | No | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `em_solvers` | L3 | M1 | No | No | Yes | Yes | `__init__.py` | code/docs indicate higher-fidelity numerical method family; consumes or supports other modules without canonical closed-loop exchange |
| `emi_emc` | L1 | M1 | No | Partial | Yes | No | `__init__.py` | reduced-order or application-level physics logic; consumes or supports other modules without canonical closed-loop exchange |
| `field_aware_routing` | L2 | M1 | Yes | Partial | No | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `flight_mechanics` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `fracture_mechanics` | L2 | M2 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls |
| `gas_generation` | L2 | M2 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls |
| `geometry` | L0 | M0 | No | No | No | No | `__init__.py` | geometry/FDM support utilities; support utilities only |
| `geospatial` | L1 | M1 | Yes | Partial | Yes | Yes | `solver.py, README.md` | physics-inspired feasibility/routing with optional real-data hooks; direct feedthrough exists, but standardization/stabilization are limited; README.md documents production data integration, but this module is logistics/feasibility rather than first-principles physics. |
| `ground_truth` | L0 | M2 | Yes | Partial | Yes | Yes | `solver.py` | fixtures/reference data rather than forward simulation; direct feedthrough plus partial standardization and convergence controls |
| `hv_safety` | L4 | M2 | Yes | Partial | Yes | Yes | `solver.py, README_LAYER4.md` | module documentation explicitly claims production-ready / tested operation; direct feedthrough plus partial standardization and convergence controls; README_LAYER4.md explicitly marks the module production-ready with 17/17 tests passing. |
| `hypersonic_vehicle_simulation` | L1 | M2 | Yes | Partial | Yes | Yes | `solver.py` | system-level vehicle synthesis over higher-fidelity submodels; direct feedthrough plus partial standardization and convergence controls |
| `injury_biomechanics` | L1 | M1 | Yes | Partial | Yes | Partial | `solver.py` | lumped injury metrics and surrogate injury models; direct feedthrough exists, but standardization/stabilization are limited |
| `ir_signature` | L2 | M2 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls |
| `material_erosion` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `microlayer` | L3 | M1 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited |
| `missile_defense` | L3 | M1 | Yes | Partial | Yes | Partial | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited |
| `missile_guidance` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `monte_carlo_neutron` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | Monte Carlo neutronics; direct feedthrough plus partial standardization and convergence controls |
| `multiphase_vof` | L3 | M1 | Yes | Partial | Yes | Partial | `solver.py` | VOF multiphase flow; direct feedthrough exists, but standardization/stabilization are limited |
| `navier_stokes` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | Navier-Stokes CFD solver; direct feedthrough plus partial standardization and convergence controls |
| `neutronics` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `nonequilibrium` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `nucleate_boiling` | L3 | M1 | Yes | Partial | Yes | Partial | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited |
| `pack_thermal` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `particle_in_cell` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | PIC plasma / charged-particle method; direct feedthrough plus partial standardization and convergence controls |
| `passive_detection` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `phase_change` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `plasma_fluid` | L3 | M1 | Yes | Partial | Yes | Partial | `solver.py` | coupled plasma-fluid solver; direct feedthrough exists, but standardization/stabilization are limited |
| `plasma_thruster_simulation` | L3 | M1 | Yes | Partial | Yes | Yes | `solver.py` | high-fidelity plasma thruster simulation; direct feedthrough exists, but standardization/stabilization are limited |
| `pollutant_formation` | L3 | M1 | Yes | Partial | Yes | Partial | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited |
| `power_processing` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `propagation` | L3 | M1 | Yes | Partial | Yes | Partial | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited |
| `propulsion` | L2 | M1 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `quantum_nanoscale` | L1 | M2 | Yes | Partial | No | Yes | `solver.py` | reduced-order nanoscale / quantum device models; direct feedthrough plus partial standardization and convergence controls |
| `quantum_optimization` | L1 | M2 | Yes | Partial | No | Yes | `solver.py` | optimization workflow, not direct field solve; direct feedthrough plus partial standardization and convergence controls |
| `radar_absorbing_materials` | L3 | M1 | Yes | Partial | Yes | Partial | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited |
| `radar_detection` | L3 | M1 | Yes | Partial | No | Partial | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited |
| `radar_waveforms` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `radiation_environment` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `radiation_transport` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | transport/radiation PDEs with dedicated solver; direct feedthrough plus partial standardization and convergence controls |
| `rcs_shaping` | L1 | M1 | Yes | Partial | No | Partial | `solver.py` | shaping/surrogate RCS estimates rather than full-wave solve; direct feedthrough exists, but standardization/stabilization are limited |
| `reacting_flows` | L3 | M1 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited |
| `reactivity_feedback` | L2 | M2 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls |
| `reactor_transient` | L2 | M1 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `reactor_transients` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `reentry_simulation` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `rf_jamming` | L2 | M1 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `safety` | L1 | M2 | Yes | Partial | Yes | Yes | `solver.py` | system safety / kinetics abstractions; direct feedthrough plus partial standardization and convergence controls |
| `safety_logic` | L0 | M2 | Yes | Partial | Yes | Yes | `solver.py` | rules/logic wrapper around physics outputs; direct feedthrough plus partial standardization and convergence controls |
| `sensing` | L1 | M1 | Yes | Partial | No | Partial | `solver.py` | trade-study/feasibility architecture rather than full sensor scene simulation; direct feedthrough exists, but standardization/stabilization are limited |
| `sensor_fusion` | L0 | M1 | Yes | Partial | Yes | Yes | `solver.py` | state-estimation layer, not first-principles physics; direct feedthrough exists, but standardization/stabilization are limited |
| `sensors_actuators` | L1 | M1 | Yes | Partial | Yes | Partial | `solver.py` | component-level reduced-order hardware models; direct feedthrough exists, but standardization/stabilization are limited |
| `sheath_model` | L2 | M1 | Yes | Partial | No | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `shielding` | L3 | M1 | Yes | Partial | No | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited |
| `signature_simulation` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `solvers` | L0 | M1 | No | No | Yes | Yes | `__init__.py` | solver-selection/backend infrastructure; consumes or supports other modules without canonical closed-loop exchange |
| `soot_model` | L3 | M1 | Yes | Partial | Yes | Partial | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited |
| `spacecraft_thermal` | L1 | M2 | Yes | Partial | Yes | Yes | `solver.py` | network-style spacecraft thermal abstractions; direct feedthrough plus partial standardization and convergence controls |
| `spatial_flow` | L1 | M1 | No | No | No | Yes | `__init__.py, README.md` | routing engine built on potential-field abstractions; consumes or supports other modules without canonical closed-loop exchange |
| `structural_analysis` | L3 | M2 | Yes | Partial | No | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `structural_dynamics` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `surface_tension` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough plus partial standardization and convergence controls |
| `swarm_dynamics` | L1 | M1 | Yes | Partial | Yes | Yes | `solver.py` | agent/surrogate dynamics; direct feedthrough exists, but standardization/stabilization are limited |
| `thermal_hydraulics` | L3 | M3 | Yes | Partial | Yes | Yes | `solver.py` | coupled thermal-fluid engineering solver stack; tight thermal-fluid iteration with SI conversion hooks |
| `thermal_runaway_kinetics` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `thermal_signature_coupling` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | explicit thermal/signature coupled analysis; direct feedthrough plus partial standardization and convergence controls |
| `thermo_mechanical` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `tps_ablation` | L3 | M2 | Yes | Partial | Yes | Yes | `solver.py` | TPS ablation / high-enthalpy thermal coupling; direct feedthrough plus partial standardization and convergence controls |
| `tracking` | L3 | M1 | Yes | Partial | Yes | Yes | `solver.py` | code/docs indicate higher-fidelity numerical method family; direct feedthrough exists, but standardization/stabilization are limited |
| `turbulence_combustion` | L3 | M1 | Yes | Partial | Yes | Partial | `solver.py` | combustion-turbulence coupling; direct feedthrough exists, but standardization/stabilization are limited |
| `uav_aerodynamics` | L2 | M2 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls |
| `uav_navigation` | L2 | M2 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls |
| `uav_propulsion` | L2 | M1 | Yes | Partial | Yes | Partial | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough exists, but standardization/stabilization are limited |
| `uncertainty` | L1 | M1 | Yes | Partial | Yes | Partial | `solver.py` | uncertainty wrapper around other physics solvers; direct feedthrough exists, but standardization/stabilization are limited |
| `verification` | L0 | M1 | No | No | Yes | Yes | `__init__.py, README.md` | verification harness, not a physics module; benchmarks consume solver outputs but do not provide reusable runtime coupling; README.md contains benchmark and regression infrastructure, but also notes key verification tests are still failing. |
| `warhead_lethality` | L2 | M2 | Yes | Partial | Yes | Yes | `solver.py` | forward-physics solver exports reusable simulation state; direct feedthrough plus partial standardization and convergence controls |

## Priority actions to raise coupling maturity

1. **Benchmark the tight-coupling exemplars.** `coupled_electrical_thermal`, `coupled_physics`, `thermal_hydraulics`, `conjugate_heat_transfer`, and `tps_ablation` should be turned into documented regression-coupled stacks with pass/fail tolerances.
2. **Normalize module boundaries to SI/canonical fields.** Several modules expose reusable state but still only achieve **Partial** for units because their boundary contracts are not uniformly explicit.
3. **Add multi-rate coupling policies.** Fast electrical / control modules and slow thermal / structural modules need scheduler-level time-scale handling instead of assuming one shared integration cadence.
4. **Separate 'production-ready' from 'high-fidelity'.** Where module docs claim readiness, back them with benchmark tables and regression artifacts so L4 / M4 can be justified unambiguously.
5. **Resolve contradictory status docs.** `drift_diffusion` and `verification` both contain mixed signals; aligning their status documents will make future audits less subjective.

