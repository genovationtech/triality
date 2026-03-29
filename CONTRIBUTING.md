# Contributing to Triality

Thank you for your interest in contributing to Triality. This document provides guidelines for contributing to the project.

## Development Setup

```bash
./setup.sh
source .venv/bin/activate
```

## Code Standards

- **Python**: Follow PEP 8. Use type hints for public API functions.
- **Rust**: Follow standard Rust conventions. Run `cargo fmt` and `cargo clippy`.
- **Tests**: Every new feature or bug fix must include tests. Maintain the 98%+ pass rate.
- **Documentation**: Update relevant docs when changing public APIs.
- **Observables**: When adding a new runtime module, register an `ObservableSet` in `triality/observables.py` that derives domain-specific engineering quantities from the solver's output fields.

## Branching Strategy

- `main` — stable, production-ready code
- `develop` — integration branch for features
- `feature/*` — feature branches
- `fix/*` — bug fix branches

## Pull Request Process

1. Create a feature branch from `develop`
2. Write your code with tests
3. Run the full test suite: `pytest lib/triality/ -v`
4. Update documentation if applicable
5. Submit a pull request with a clear description

## Testing

```bash
# Full suite
pytest lib/triality/ -v

# Specific layer
pytest lib/triality/tests/ -v -k "electrostatics"      # Layer 1
pytest lib/triality/tests/ -v -k "field_aware_routing"  # Layer 2
pytest lib/triality/tests/ -v -k "drift_diffusion"      # Layer 3

# With coverage
pytest lib/triality/ --cov=triality --cov-report=html
```

## Physics Module Guidelines

When adding a new domain module:

1. Create the module directory under `lib/triality/`
2. Implement the standard runtime interface (`run(config) -> RuntimeExecutionResult`)
3. Add tests in `lib/triality/tests/`
4. Add a README.md in the module directory
5. Register the module in `runtime.py`
6. Be honest about accuracy limitations in the module docstring

## Reporting Issues

When reporting a bug, include:

- Python version and OS
- Minimal reproduction code
- Expected vs actual behavior
- Full traceback if applicable

---

*Genovation Technological Solutions Pvt Ltd — Powered by Mentis OS*
*"We build systems that understand reality."*
