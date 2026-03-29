# Architecture

## Overview

Triality is organized as a three-layer progressive framework built on a hybrid Python + Rust stack.

```
triality/
├── core/               # Expressions, fields, domains, units
├── solvers/            # PDE classification and solver selection
├── electrostatics/     # Layer 1: Field solving (production)
├── field_aware_routing/# Layer 2: Physics-aware routing (production)
├── spatial_flow/       # Underlying flow engine for routing
├── triality_engine/    # Rust extension (optional, performance)
├── [40+ domain modules]
├── examples/           # Worked examples
├── tests/              # Test suite
└── docs/               # This documentation
```

---

## The Three Layers

### Layer 1 — Automatic PDE Solving

**Status: Production-Ready** (100% tests passing)

Layer 1 provides a high-level API for formulating and solving PDEs automatically. The user describes the problem symbolically; Triality handles classification, solver selection, assembly, and post-processing.

**Pipeline:**

```
User problem description
        │
        ▼
  Problem Classifier      ← classify.py
  (linear/nonlinear, order, domain shape)
        │
        ▼
  Solver Selector          ← select.py
  (direct, iterative, spectral, FEM)
        │
        ▼
  Matrix Assembler         ← solve.py
  (finite differences or FEM)
        │
        ▼
  Linear/Nonlinear Solve   ← linear.py / nonlinear.py
        │
        ▼
  Solution Object          ← callable, plottable, exportable
        │
        ▼
  Observable Layer         ← observables.py
  (domain-specific engineering quantities, thresholds, margins)
```

**Supported problem types:**
- Elliptic PDEs (Poisson, Laplace, Helmholtz)
- Parabolic PDEs (heat equation, diffusion)
- Hyperbolic PDEs (wave equation)
- Systems of PDEs (multi-physics coupling)

### Layer 2 — Physics-Aware Spatial Routing

**Status: Production-Ready** (100% tests passing)

Layer 2 converts physics field solutions into routing cost fields and then finds optimal paths through the physics landscape. This is Triality's key differentiator — no other pathfinding library integrates EM/thermal physics directly.

**Pipeline:**

```
Physics field (e.g., E-field from Layer 1)
        │
        ▼
  Cost Field Builder       ← cost_field_builders.py
  (physics → per-cell routing cost)
        │
        ▼
  Coupling Analysis        ← coupling_analysis.py
  (cross-domain field interactions)
        │
        ▼
  A*/Dijkstra Router       ← spatial_flow/engine.py
  (physics-aware pathfinding on cost grid)
        │
        ▼
  Path Object              ← with physics stats along route
```

**Applications:**
- PCB signal trace routing (EMI avoidance)
- Cable routing in vehicles (thermal/EMI)
- Pipeline routing (terrain + hazard avoidance)
- Emergency evacuation planning
- Robot navigation

### Layer 3 — Drift-Diffusion Semiconductors

**Status: Framework Complete** (numerical refinement in progress)

Layer 3 implements the Poisson-drift-diffusion system for semiconductor device simulation. The mathematical framework and code architecture are complete; full numerical convergence is being refined.

**Equations solved:**
- Poisson: ∇·(ε∇φ) = -ρ/ε₀
- Electron continuity: ∂n/∂t = (1/q)∇·Jₙ + G - R
- Hole continuity: ∂p/∂t = -(1/q)∇·Jₚ + G - R

### The Observable Layer

**Status: Production-Ready** (100% module coverage)

The Observable Layer sits after solver execution and transforms raw PhysicsState fields into domain-specific engineering quantities — the values that answer design questions.

**Pipeline position:**
```
Intent → Solver → Fields → **Observables** → Interpretation
```

Each physics module registers an `ObservableSet` that derives observables relevant to its domain:

```python
from triality.observables import compute_observables

obs = compute_observables("navier_stokes", state, config)
# Returns ranked list: peak_velocity, vorticity_peak, dead_zone_fraction, reynolds_number, ...
```

**Key properties (benchmarked across all 16 modules, 20 trials each):**

| Metric | Value |
|--------|-------|
| Module coverage | 16/16 (100%) |
| Total observables | 126 (5–12 per module) |
| Scalar observables | 123 (97.6%) |
| Pass/fail thresholds | 10 (structural yield, thermal runaway, breakdown, etc.) |
| Compute time | 0.007 – 0.133 ms per module |
| Median overhead | 0.027 ms (< 0.15% of solver time) |

**Observable types by domain:**

| Module | Observables | Example quantities |
|--------|------------|-------------------|
| `navier_stokes` | 10 | Peak velocity, vorticity, dead zone fraction, Reynolds number, wall shear |
| `coupled_physics` | 12 | Power ratio, peak fuel temperature, Doppler feedback, prompt critical margin |
| `drift_diffusion` | 9 | Built-in potential, depletion width, ideality factor, junction capacitance |
| `electrostatics` | 8 | Peak field strength, breakdown margin, field uniformity, stored energy |
| `structural_analysis` | 8 | Von Mises stress, deflection, buckling margin, stress ratio |

Each observable carries:
- **Rank** (0 = primary design driver, lower = more important)
- **Optional threshold** (pass/fail boundary)
- **Margin** (signed distance to threshold: positive = safe, negative = violated)
- **Relevance** (why it matters for the design question)

---

## Core Package (`triality/core/`)

| File | Lines | Purpose |
|---|---|---|
| `expressions.py` | 224 | Symbolic field expressions and operators |
| `fields.py` | 616 | Field class with units, metadata, interpolation |
| `domains.py` | 61 | Domain types: Interval, Rectangle, Cube, Mesh |
| `units.py` | 501 | SI unit system, conversion, dimensional analysis |
| `validation.py` | 278 | Input validation and error messages |
| `presets.py` | 398 | Predefined material and physics presets |
| `coupling.py` | 635 | Multi-physics coupling interface |
| `adapters.py` | 178 | Adapters between module interfaces |

### Observable Layer (`triality/observables.py`)

| File | Lines | Purpose |
|---|---|---|
| `observables.py` | 1226 | Observable dataclass, registry, 16 per-module ObservableSets |

### Expression System

The expression system supports standard differential operators:

```python
laplacian(u)          # ∇²u = ∂²u/∂x² + ∂²u/∂y² + ...
gradient(u)           # ∇u
divergence(F)         # ∇·F
curl(F)               # ∇×F
d_dt(u)               # ∂u/∂t
d_dx(u)               # ∂u/∂x
```

Expressions compose naturally:

```python
heat_eq = rho*cp * d_dt(T) - k * laplacian(T) == Q
```

---

## Solver Package (`triality/solvers/`)

| File | Lines | Purpose |
|---|---|---|
| `classify.py` | 104 | Classify PDE type (elliptic/parabolic/hyperbolic) |
| `select.py` | 105 | Select best solver for problem type |
| `solve.py` | 236 | Main `solve()` API |
| `linear.py` | 196 | Linear system solvers (direct + iterative) |
| `assumptions.py` | 210 | Track solver assumptions and warn on violations |
| `wellposedness.py` | 123 | Check problem is well-posed before solving |
| `rust_backend.py` | 327 | Interface to Rust `triality_engine` extension |

---

## Rust Backend (`triality/triality_engine/`)

The `triality_engine` is a Rust extension compiled with PyO3/Maturin. It provides:

- **Finite difference assembly**: Fast sparse matrix construction
- **Time integration**: Explicit/implicit/Crank-Nicolson steppers
- **Sparse linear algebra**: Custom CG/GMRES with preconditioners
- **Grid management**: Structured and unstructured grid types

See [Rust Backend](rust_backend.md) for full details.

---

## Multi-Physics Runtime

`runtime.py` and `runtime_graph.py` implement a computational graph for coupled multi-physics simulations:

```python
from triality.runtime import PhysicsRuntime

rt = PhysicsRuntime()
rt.add_module('thermal', ThermalModule(...))
rt.add_module('structural', StructuralModule(...))
rt.couple('thermal.temperature', 'structural.thermal_load')
rt.run(t_end=10.0, dt=0.1)
```

The runtime handles:
- Dependency resolution between coupled modules
- Staggered vs. monolithic time integration
- Data exchange at coupling interfaces
- Convergence checking for iterative coupling
- Observable derivation via the Observable Layer after each solve

---

## Design Principles

1. **Speed over completeness** — 50×50 grid solves in 5 ms; 200×200 in 150 ms
2. **Honest limitations** — every solver reports its accuracy assumptions
3. **Progressive fidelity** — use Triality for early design, then export to FEM/CFD
4. **Modular** — each physics domain is independent; import only what you need
5. **Hybrid stack** — Python API for expressiveness, Rust for performance-critical paths

---

## Performance Characteristics

| Task | Grid | Time |
|---|---|---|
| Poisson solve (1D) | N=1000 | < 1 ms |
| Electrostatics (2D) | 50×50 | 5 ms |
| Electrostatics (2D) | 100×100 | 20 ms |
| Electrostatics (2D) | 200×200 | 150 ms |
| Field-aware routing | 100×100 | ~70 ms total |
| Coupled analysis + route | 100×100 | ~200 ms total |

Accuracy vs. full FEM/CFD:
- Electrostatics: ±2–5% for simple geometries, ±10–15% for complex
- Routing cost fields: ±20–30% vs. full EM
- Thermal (conduction): ±3–8%
