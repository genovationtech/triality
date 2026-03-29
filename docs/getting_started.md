# Getting Started with Triality

## Installation

### Prerequisites

- Python 3.8 or newer
- pip

### Install from source

```bash
git clone <repository>
cd triality
pip install -e .
```

### Optional: Rust backend (performance)

The Rust backend (`triality_engine`) provides 10-100x speedups for large grids. It requires Rust and Maturin:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
pip install maturin
cd triality/triality_engine
maturin develop --release
```

If the Rust backend is not installed, Triality falls back to pure-Python NumPy/SciPy implementations automatically.

### Dependencies

| Package | Version | Purpose |
|---|---|---|
| numpy | ≥ 1.20 | Arrays and numerical operations |
| scipy | ≥ 1.7 | Sparse solvers, optimization |
| matplotlib | ≥ 3.4 | Visualization (optional) |

---

## Your First Simulation

### Example 1: Solve a PDE in 3 lines

```python
from triality import Field, solve, laplacian, Interval

u = Field("u")
sol = solve(laplacian(u) == 1, Interval(0, 1), bc={'left': 0, 'right': 0})
print(f"u(0.5) = {sol(0.5):.6f}")   # -0.125000
sol.plot()
```

Triality automatically classifies this as a 1D Poisson problem, selects a direct tridiagonal solver, and returns a callable solution object.

### Example 2: Electrostatic field analysis

```python
from triality.electrostatics import ElectrostaticSolver2D

solver = ElectrostaticSolver2D(x_range=(0, 0.1), y_range=(0, 0.1), resolution=50)
solver.set_boundary('left',  'voltage', 5.0)
solver.set_boundary('right', 'voltage', 0.0)
solver.set_boundary('top',   'neumann', 0.0)
solver.set_boundary('bottom','neumann', 0.0)

result = solver.solve()
print(f"Max E-field: {result.electric_field_magnitude().max():.1f} V/m")
result.plot_voltage()
result.plot_electric_field()
```

### Example 3: Physics-aware PCB routing

```python
from triality.field_aware_routing import FieldAwareRouter
from triality.electrostatics import ElectrostaticSolver2D

# 1. Solve the physics
esolver = ElectrostaticSolver2D(x_range=(0,0.1), y_range=(0,0.1), resolution=100)
esolver.set_component('source', position=(0.02, 0.05), voltage=3.3)
esolver.set_component('gnd',    position=(0.08, 0.05), voltage=0.0)
efield = esolver.solve()

# 2. Route a signal trace avoiding high-field regions
router = FieldAwareRouter(efield)
router.set_start((0.01, 0.02))
router.set_end((0.09, 0.08))
router.add_constraint('max_field', 1000)   # V/m limit

path = router.route()
print(f"Path length: {path.length:.4f} m")
print(f"Max field along path: {path.max_field:.1f} V/m")
path.plot()
```

### Example 4: Observable Layer — Engineering quantities from solver output

```python
from triality import load_module
from triality.observables import compute_observables

# Solve a Navier-Stokes lid-driven cavity
solver = load_module("navier_stokes").from_demo_case()
result = solver.solve()

# Derive domain-specific engineering observables
obs = compute_observables("navier_stokes", result.generated_state, {})
for o in obs:
    print(f"{o.name}: {o.value:.4g} {o.unit} — {o.description}")
    if o.margin is not None:
        status = "PASS" if o.margin >= 0 else "FAIL"
        print(f"  margin: {o.margin:+.3g} [{status}]")
# Output: peak_velocity, vorticity_peak, dead_zone_fraction, reynolds_number, ...
```

---

## Core Concepts

### Fields

A `Field` is a symbolic variable used to define PDEs:

```python
from triality import Field
u = Field("u")
T = Field("T", units="K")
```

### Expressions

Build PDE expressions using operators:

```python
from triality import laplacian, gradient, divergence, curl

expr = laplacian(u) + 2*u - 1   # ∇²u + 2u = 1
```

### Domains

Define spatial domains:

```python
from triality import Interval, Rectangle, Cube

d1 = Interval(0, 1)              # 1D
d2 = Rectangle(0, 1, 0, 1)      # 2D
d3 = Cube(0, 1, 0, 1, 0, 1)     # 3D
```

### Boundary Conditions

```python
bc = {
    'left':   0.0,          # Dirichlet: u = 0
    'right':  0.0,
    'top':    ('neumann', 0.0),   # Neumann: du/dn = 0
    'bottom': ('robin', 1.0, 2.0) # Robin: a*u + b*du/dn = 0
}
```

### Solve

```python
from triality import solve
sol = solve(laplacian(u) == source_term, domain, bc=bc)
```

---

## Next Steps

- [Architecture](architecture.md) — understand how Triality is structured
- [Modules](modules.md) — browse the 40+ physics modules
- [Examples](examples.md) — see full worked examples
- [API Reference](api_reference.md) — complete function/class documentation
