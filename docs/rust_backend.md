# Rust Backend (`triality_engine`)

The `triality_engine` is a compiled Rust extension that provides high-performance numerical routines. It is **optional** — Triality falls back to pure Python/NumPy implementations if it is not installed.

---

## When to Use the Rust Backend

| Scenario | Python only | With Rust backend |
|---|---|---|
| 50×50 grid Poisson | 5 ms | 0.5 ms |
| 100×100 grid Poisson | 20 ms | 2 ms |
| 200×200 grid Poisson | 150 ms | 12 ms |
| 500×500 grid Poisson | ~5 s | ~200 ms |
| Time-stepping (1000 steps) | ~1 min | ~5 s |

The Rust backend is recommended for:
- Grids larger than 100×100
- Time-dependent simulations with many time steps
- Production batch processing
- Real-time embedded applications

---

## Building from Source

### Prerequisites

```bash
# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install maturin (Python-Rust bridge)
pip install maturin
```

### Build and install

```bash
cd triality/triality_engine
maturin develop --release   # development install
# or
maturin build --release     # build wheel
pip install target/wheels/triality_engine-*.whl
```

### Verify installation

```python
import triality_engine
print(triality_engine.__version__)
# Should print: 0.2.0

# Benchmark
import time
from triality_engine import fdm_poisson_2d
import numpy as np

n = 200
rhs = np.ones((n, n))
t0 = time.perf_counter()
sol = fdm_poisson_2d(rhs, dx=1.0/n)
print(f"200×200 Poisson: {(time.perf_counter()-t0)*1000:.1f} ms")
```

---

## Module Structure

```
triality_engine/
├── Cargo.toml          # Rust dependencies and build config
├── pyproject.toml      # Python/maturin build config
├── src/
│   ├── lib.rs          # PyO3 module root, Python bindings
│   ├── fdm.rs          # Finite difference matrix assembly
│   ├── solvers.rs      # Krylov solvers (CG, GMRES, BiCGSTAB)
│   ├── sparse.rs       # CSR sparse matrix operations
│   ├── timestepper.rs  # Time integration schemes
│   ├── grid.rs         # Structured grid management
│   ├── precond.rs      # Preconditioners (Jacobi, ILU0, ILU(k))
│   └── python_bindings.rs  # PyO3 type conversions
```

---

## Rust API (exposed to Python)

### FDM Assembly (`fdm.rs`)

```python
import triality_engine as te

# Assemble 2D Poisson operator: ∇²u = f
A = te.assemble_poisson_2d(nx=100, ny=100, dx=0.01, dy=0.01)
# Returns: scipy.sparse.csr_matrix compatible object

# Assemble with variable coefficients: ∇·(k(x,y)∇u) = f
k_field = np.ones((100, 100))
A = te.assemble_poisson_variable_coeff_2d(k_field, dx=0.01, dy=0.01)
```

### Solvers (`solvers.rs`)

```python
# Conjugate Gradient (symmetric positive definite systems)
x = te.solve_cg(A, b, tol=1e-8, max_iter=1000)

# GMRES (general nonsymmetric systems)
x = te.solve_gmres(A, b, restart=50, tol=1e-8, max_iter=1000)

# BiCGSTAB (alternative for nonsymmetric systems)
x = te.solve_bicgstab(A, b, tol=1e-8, max_iter=1000)
```

### Time Stepping (`timestepper.rs`)

```python
# Forward Euler: u^{n+1} = u^n + dt * f(u^n)
stepper = te.ExplicitStepper(M=mass_matrix, dt=0.001)

# Backward Euler: (M + dt*A) u^{n+1} = M*u^n + dt*f
stepper = te.ImplicitStepper(M=mass_matrix, A=stiffness_matrix, dt=0.001)

# Crank-Nicolson: second-order in time
stepper = te.CrankNicolsonStepper(M=mass_matrix, A=stiffness_matrix, dt=0.001)

u_new = stepper.step(u_old, rhs)
```

### Grid Management (`grid.rs`)

```python
# Create a structured 2D grid
grid = te.StructuredGrid2D(
    nx=100, ny=100,
    x_min=0.0, x_max=1.0,
    y_min=0.0, y_max=1.0
)
# Grid indexing utilities
i, j = grid.index_to_ij(flat_index)
flat = grid.ij_to_index(i, j)
```

### Preconditioners (`precond.rs`)

```python
# Jacobi (diagonal scaling)
P = te.JacobiPreconditioner(A)

# Incomplete LU (ILU0)
P = te.ILU0Preconditioner(A)

# Apply: z = P^{-1} r
z = P.apply(r)
```

---

## Cargo.toml Configuration

Key Rust dependencies:

```toml
[dependencies]
pyo3   = { version = "0.22", features = ["extension-module"] }
ndarray = "0.15"
rayon   = "1.10"     # parallel iteration
num-traits = "0.2"

[profile.release]
opt-level = 3
lto = true           # link-time optimization
codegen-units = 1    # maximum optimization
```

---

## Automatic Fallback

Triality automatically detects whether `triality_engine` is available:

```python
# In triality/solvers/rust_backend.py
try:
    import triality_engine as _te
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    import warnings
    warnings.warn(
        "triality_engine Rust extension not found. "
        "Falling back to Python/NumPy (slower for large grids).",
        RuntimeWarning
    )

def solve_poisson_2d(rhs, dx, dy):
    if RUST_AVAILABLE:
        return _te.fdm_poisson_2d(rhs, dx=dx, dy=dy)
    else:
        return _python_fallback_poisson_2d(rhs, dx, dy)
```

To force Python-only mode (for debugging or testing):

```python
import triality.solvers.rust_backend as rb
rb.RUST_AVAILABLE = False
```
