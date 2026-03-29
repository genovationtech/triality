# Changelog

All notable changes to Triality are documented here.

---

## [0.2.0] — Current

### Added
- **Field-Aware Routing (Layer 2)** — Production-ready physics-aware spatial routing
  - `FieldAwareRouter` with A* pathfinding on physics cost fields
  - `CostFieldBuilder` for composing multi-physics cost fields
  - `CouplingAnalysis` for cross-domain field interaction analysis
  - Multi-path routing (`route_multi`)
  - Gerber/DXF export for PCB integration
- **Spatial Flow Engine** — Underlying flow solver for routing
  - Source/sink flow field computation
  - Path extraction from flow fields
  - Template configurations for common routing scenarios
- **Runtime SDK** (`runtime.py`, `runtime_graph.py`)
  - Computational graph for coupled multi-physics
  - Staggered and monolithic time integration
  - `RuntimeGraph` for dependency-based execution ordering
- **40+ specialized physics modules** — See [Modules](modules.md)
- **Rust backend** (`triality_engine`) — 10–100x speedup for large grids
  - Finite difference matrix assembly
  - CG/GMRES/BiCGSTAB solvers
  - Crank-Nicolson time integrator
  - ILU0 preconditioner
- **Uncertainty Quantification module** — Monte Carlo, Sobol indices
- **Verification suite** — Systematic comparison vs. analytical solutions
- **14 demonstration scripts** — Kill-switch analysis for aerospace, nuclear, automotive, UAV

### Improved
- Electrostatics solver: 3x speedup via vectorized assembly
- Routing: 50% reduction in memory for large grids
- All solver modules now report accuracy assumptions explicitly

### Fixed
- Neumann BC assembly was incorrect for non-square grids (was: using wrong dx/dy)
- Energy calculation in `ElectrostaticResult` off by factor of 0.5 (fixed)

---

## [0.1.0] — Initial Release

### Added
- **Automatic PDE Solving (Layer 1)** — Production-ready
  - Symbolic `Field` / expression system
  - PDE classifier (elliptic/parabolic/hyperbolic)
  - Intelligent solver selection (direct/iterative based on problem size)
  - `Interval`, `Rectangle`, `Cube` domain types
- **Electrostatics module** — 2D Poisson solver
  - Dirichlet and Neumann boundary conditions
  - Multiple conductor support
  - Electric field, energy density, force field post-processing
- **Conduction module** — 2D steady-state heat/current conduction
- **Drift-Diffusion framework (Layer 3)** — Architecture complete
  - Poisson + electron/hole continuity system
  - Scharfetter-Gummel discretization
  - SRH, radiative, Auger recombination models
  - Caughey-Thomas velocity saturation model
- **Core package** — Fields, domains, units, expressions, validation
- **52 of 53 tests passing** (98%)

---

## Roadmap

### [0.3.0] — Planned
- Observable Layer with per-module engineering quantity derivation
- Layer 3 (Drift-Diffusion): full numerical convergence for HV devices
- 3D electrostatics solver
- FEM-based structural solver
- Export to COMSOL/ANSYS formats

### [0.4.0] — Future
- GPU acceleration (CUDA via CuPy)
- Mesh-based (unstructured) solvers
- Reduced-order model (ROM) capability
- Cloud-based multi-node solving
