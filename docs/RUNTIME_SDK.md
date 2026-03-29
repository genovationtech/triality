# Triality Runtime SDK

This file introduces the first standardized runtime execution layer for Triality.

## Goal

Give users, tests, and agents a predictable way to run modules without learning
every solver's constructor shape.

## Locked contract

```python
from triality import load_module, RuntimeExecutionResult, RuntimeDescription

solver = load_module("navier_stokes").from_demo_case()
result: RuntimeExecutionResult = solver.solve()
state = solver.to_state()          # PhysicsState
meta: RuntimeDescription = solver.describe()
```

## Required methods

Every runtime adapter implements:

- `from_demo_case()`
- `demo()`
- `from_config(config)`
- `from_state(state)`
- `solve()`
- `to_state()`
- `get_output()`
- `describe()`

## Exact return types

- `from_demo_case() -> BaseRuntimeSolver`
- `from_config(config: dict | None) -> BaseRuntimeSolver`
- `solve() -> RuntimeExecutionResult`
- `solve_safe() -> RuntimeExecutionResult`
- `to_state() -> PhysicsState`
- `describe() -> RuntimeDescription`

## Machine-usable metadata schema

`describe()` is now intended for orchestration, not prose. It returns a
machine-usable metadata object with these required fields:

- `module_name`
- `domain`
- `fidelity_level`
- `coupling_ready`
- `supports_transient`
- `supports_steady`
- `required_inputs`
- `output_fields`
- `validation_status`
- `contract_version`
- `supports_demo_case`
- `construction_mode`

## Observable Layer Integration

After `solve()` returns a `RuntimeExecutionResult`, the Observable Layer automatically derives
domain-specific engineering quantities from the solver's `PhysicsState`:

```python
from triality.observables import compute_observables

result = solver.solve()
obs = compute_observables(module_name, result.generated_state, config)
# Returns ranked list of Observable dataclasses with:
#   name, value, unit, description, threshold, margin, rank
```

Every runtime module has a registered `ObservableSet` (16/16 = 100% coverage). Observable
computation adds < 0.15% overhead to solver time (median 0.027 ms).

The Triality App automatically computes observables for every `run_module` and `sweep_parameter`
call, including them in the structured result sent to the LLM interpretation layer.

## Field spec format

Each entry in `required_inputs` / `output_fields` is a field spec with:

- `name`
- `units`
- `kind`
- `required`

Example shape:

```python
{
  "module_name": "navier_stokes",
  "domain": "fluid_dynamics",
  "fidelity_level": "L3",
  "coupling_ready": "M2",
  "supports_transient": True,
  "supports_steady": False,
  "supports_demo_case": True,
  "construction_mode": "demo",
  "required_inputs": [],
  "output_fields": [
    {"name": "velocity_x", "units": "m/s", "kind": "field", "required": False},
    {"name": "velocity_y", "units": "m/s", "kind": "field", "required": False},
    {"name": "pressure", "units": "Pa", "kind": "field", "required": False},
  ],
  "validation_status": "demo_smoke_tested",
  "contract_version": "1.0",
}
```


## Production-grade `from_config()`

`from_config()` is now the primary usability path for the runtime SDK.
It supports both:

- nested config sections: `solver`, `solve`, and module-specific sections such as `doping`
- legacy flat config keys for backward compatibility

Validation guarantees:

- non-dict configs are rejected
- unknown keys are rejected
- duplicating the same key both top-level and inside a section is rejected
- module-specific sections are validated before construction

### Config examples

#### Navier-Stokes

```python
solver = load_module("navier_stokes").from_config({
  "solver": {"nx": 20, "ny": 20, "U_lid": 0.2},
  "solve": {"t_end": 0.01, "dt": 0.002, "max_steps": 20},
})
```

#### Thermal hydraulics

```python
solver = load_module("thermal_hydraulics").from_config({
  "solver": {"n_axial": 16, "n_fuel_radial": 8, "mass_flux": 3200.0},
  "solve": {"peak_linear_heat_rate": 150.0, "axial_shape": "uniform"},
})
```

#### Conjugate heat transfer

```python
solver = load_module("conjugate_heat_transfer").from_config({
  "solver": {"nx": 16, "ny_solid": 6, "ny_fluid": 12, "Q_vol": 2e5},
  "solve": {"t_end": 0.005, "dt": 5e-4, "max_coupling_iter": 6},
})
```

#### Drift diffusion

```python
solver = load_module("drift_diffusion").from_config({
  "solver": {"length": 1e-4, "n_points": 60, "temperature": 300.0},
  "doping": {"type": "pn_junction", "N_d_level": 1e17, "N_a_level": 5e16},
  "solve": {"applied_voltage": 0.0, "max_iterations": 50},
})
```

Runtime metadata now separates capability from provenance:

- `supports_demo_case` answers whether the adapter offers a demo constructor
- `construction_mode` records how the current instance was created (`demo`, `config`, `state`, or `direct`)

That lets orchestration code distinguish smoke-test capability from the
provenance of a specific runtime instance.


## Runtime graph / orchestration engine

The runtime SDK now includes a first DAG executor for composing runtime nodes:

- `RuntimeGraph`
- `RuntimeNode`
- `RuntimeLink`
- `RuntimeGraphResult`
- `merge_physics_states()`

This orchestration layer can:

- discover runtime-capable modules through `load_module()`
- validate links against advertised output fields and compatible units
- instantiate nodes from demo or config paths
- propagate linked `PhysicsState` fields downstream
- execute the graph in topological order or staged schedules
- iterate until graph-level convergence criteria are satisfied
- persist and replay graph definitions as JSON
- apply graph-level error policy (`raise` or `continue`)

Example:

```python
from triality import RuntimeGraph

graph = RuntimeGraph()
graph.add_node("flow", "navier_stokes", config={
  "solver": {"nx": 10, "ny": 10},
  "solve": {"t_end": 0.004, "dt": 0.002},
})
graph.add_node("thermal", "thermal_hydraulics", config={
  "solver": {"n_axial": 8, "n_fuel_radial": 6},
  "solve": {"peak_linear_heat_rate": 140.0, "axial_shape": "uniform"},
})
graph.add_link("flow", "thermal", "pressure")
result = graph.run()
```

The graph engine now supports staged scheduling, iterative multi-pass execution,
unit-aware link validation, graph-level convergence checks, graph-definition
persistence/replay, and a graph-level error policy. It is still feed-forward at
the connection level, but it now provides the substrate needed for later
fixed-point and tighter multiphysics coupling loops.


## Template workflows

The runtime/orchestration layer now ships with reusable JSON-backed graph
definitions that can be loaded directly as `RuntimeGraph` instances:

- `fluid_to_thermal`
- `thermal_to_conjugate_heat`
- `drift_diffusion_device`
- `sensing_pipeline`

Example:

```python
from triality import load_runtime_template

graph = load_runtime_template("fluid_to_thermal")
report = graph.explain()
result = graph.run()
```

These templates are intended to become the standard demos/benchmarks/agent
targets for the orchestration layer.

## Failure behavior

- Contract/schema violations raise `RuntimeContractError`
- Runtime solve failures raise `RuntimeExecutionError`
- `to_state()` before a successful `solve()` raises `RuntimeContractError`
- Unknown config keys passed to `from_config()` raise `RuntimeContractError`

## RuntimeExecutionResult

`solve()` now returns a standardized wrapper containing:

- `module_name`
- `success`
- `status`
- `warnings`
- `residuals`
- `convergence`
- `elapsed_time_s`
- `result_payload`
- `generated_state`
- `description`

On failure:

- `solve(strict=True)` raises
- `solve(strict=False)` / `solve_safe()` returns a non-success result object

This means callers can always depend on the same outer return type even when
the underlying solver-native result differs module to module.

## First supported modules

- `navier_stokes`
- `thermal_hydraulics`
- `conjugate_heat_transfer`
- `drift_diffusion`

These were chosen because they represent high-value solver modules with
substantial physics already present in the repository and are strong candidates
for future agent-driven orchestration.

## Next step

The intended expansion path is:

1. add more adapters,
2. move more native solvers toward the same constructor/solve/export contract,
3. wire this runtime layer into execution-depth testing and agent workflows.
