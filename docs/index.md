# Triality Documentation

**The Physics Operating System — 2D/3D Solving, Rust-Accelerated Engine, 113 Domain Modules**

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
*"We build systems that understand reality."*
*Building AI that understands reality.*

---

## Table of Contents

| Document | Description |
|---|---|
| [Getting Started](getting_started.md) | Installation, quickstart, and first steps |
| [Whitepaper](whitepaper.md) | Detailed product and technical whitepaper for Triality |
| [Architecture](architecture.md) | Three-layer design, module overview, internals |
| [API Reference](api_reference.md) | Full public API for all modules |
| [Physics Guide](physics_guide.md) | Physics models, assumptions, and limitations |
| [Modules](modules.md) | Per-module documentation (113 specialized domains) |
| [Examples](examples.md) | Annotated real-world examples |
| [Rust Backend](rust_backend.md) | triality_engine Rust extension |
| [Observable Layer](architecture.md#the-observable-layer) | Post-solver engineering quantities, thresholds, margins |
| [Testing & Validation](testing.md) | Test suite, verification, and validation |
| [Contributing](contributing.md) | Development setup, coding standards |
| [Changelog](changelog.md) | Version history and release notes |

---

## What is Triality?

Triality is a multi-physics simulation framework for rapid engineering analysis. It is designed around the philosophy of **"80% answer in 5% effort"** — giving engineers physically grounded results in milliseconds instead of hours.

### The Three-Layer Model

```
Layer 1: Automatic PDE Solving
  └─ Classify problem → select solver → solve → observe → post-process
  └─ Electrostatics, heat conduction, diffusion

Layer 2: Physics-Aware Spatial Routing
  └─ Convert physics fields → cost fields → route paths
  └─ PCB routing, cable routing, corridor planning

Layer 3: Drift-Diffusion Semiconductors
  └─ Full semiconductor device simulation
  └─ Production ready

3D FEM: PoissonSolver3D
  └─ Tet4/Tet10/Hex8 elements, Gmsh import, VTU export
  └─ Powered by Rust engine (triality_engine)

Observable Layer: observables.py
  └─ Fields → engineering quantities (126 observables, 16/16 modules)
  └─ Pass/fail thresholds, safety margins, go/no-go decisions

Rust Engine: triality_engine (PyO3/Maturin)
  └─ Linear solvers, FEM assembly, preconditioners, mesh I/O
  └─ 10-100x acceleration, required for 3D
```

### Supported Physics Domains (113 modules)

- **Electromagnetics**: Electrostatics, EM solvers, EMI/EMC, RF jamming, HV safety
- **Thermal**: Conjugate heat transfer, battery/automotive/spacecraft thermal, aerothermodynamics
- **Structural**: Static/dynamic structural, fracture mechanics, aeroelasticity, contact mechanics
- **Fluid/Combustion**: CFD turbulence, Navier-Stokes, multiphase flow, reacting flows
- **Aerospace**: Aero loads, flight mechanics, propulsion, hypersonic re-entry, UAV systems
- **Nuclear/Radiation**: Neutronics, Monte Carlo particle transport, shielding, burnup, reactor transients
- **Advanced**: Plasma, semiconductors, electrochemistry, quantum optimization, particle-in-cell
- **Sensing/Detection**: Radar, IR, acoustic, passive detection, multi-sensor fusion
- **Defense/Systems**: Ballistics, missile guidance, collision avoidance, counter-UAS, lethality

---

## Quick Navigation

**New user?** → [Getting Started](getting_started.md)

**Want the full platform thesis?** → [Whitepaper](whitepaper.md)

**Looking for a specific module?** → [Modules](modules.md)

**Understanding the internals?** → [Architecture](architecture.md)

**Full API docs?** → [API Reference](api_reference.md)
