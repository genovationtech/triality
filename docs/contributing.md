# Contributing

## Development Setup

```bash
git clone <repository>
cd triality

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Optional: build Rust extension
cd triality_engine
maturin develop --release
cd ..
```

## Code Standards

- **Python version**: 3.8+
- **Style**: PEP 8, enforced with `flake8`
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public classes and functions

### Example function signature

```python
def solve_poisson(
    rhs: np.ndarray,
    dx: float,
    dy: float,
    bc: dict[str, float | tuple],
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Solve the 2D Poisson equation ∇²u = f.

    Args:
        rhs: Right-hand side array f, shape (ny, nx).
        dx: Grid spacing in x direction (metres).
        dy: Grid spacing in y direction (metres).
        bc: Boundary conditions dict. Keys are 'left', 'right', 'top',
            'bottom'. Values are floats (Dirichlet) or tuples
            ('neumann', value) for Neumann conditions.
        tolerance: Iterative solver convergence tolerance.

    Returns:
        Solution array u, same shape as rhs.

    Raises:
        ValueError: If rhs is not 2D or bc keys are invalid.
    """
```

## Adding a New Physics Module

1. Create a directory: `triality/<module_name>/`
2. Create `__init__.py` with public exports
3. Implement the solver in `solver.py` or appropriately named files
4. Write tests in `triality/test_<module_name>.py`
5. Add an example in `triality/examples/<module_name>_demo.py`
6. Add an entry to `triality/runtime_templates.py`
7. Document in `docs/modules.md`

### Module interface contract

Every module solver class should implement:

```python
class MySolver:
    def __init__(self, **kwargs): ...
    def solve(self) -> MyResult: ...

class MyResult:
    def plot(self) -> None: ...
    def export(self, filename: str) -> None: ...
```

## Running Checks Before Committing

```bash
# Run full test suite
python -m pytest triality/ -v

# Check code style
flake8 triality/ --max-line-length=100

# Type check
mypy triality/ --ignore-missing-imports
```

## Branch Strategy

- `main` — stable, production-ready
- `develop` — integration branch for features
- `feature/<name>` — individual features
- `fix/<name>` — bug fixes

## Commit Message Format

```
<type>: <short description>

<optional body>
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

Examples:
```
feat: add variable-coefficient Poisson solver
fix: correct Neumann BC assembly for non-square grids
docs: add drift-diffusion physics guide
test: add verification suite for thermal solver
perf: switch 200x200 assembly to Rust backend
```

## License

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.

External contributions are welcome via pull request. By submitting a PR you agree that your contribution may be incorporated under the project's license terms.
