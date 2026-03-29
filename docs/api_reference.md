# API Reference

## Top-Level (`triality`)

These symbols are exported from `triality/__init__.py` and available via `from triality import *`.

### Fields and Expressions

```python
Field(name, units=None)
```
Create a symbolic field variable.

```python
laplacian(field)            # ∇²f
gradient(field)             # ∇f  (returns vector expression)
divergence(field)           # ∇·F
curl(field)                 # ∇×F
d_dt(field)                 # ∂f/∂t
d_dx(field)                 # ∂f/∂x
d_dy(field)                 # ∂f/∂y
d_dz(field)                 # ∂f/∂z
```

### Domains

```python
Interval(x_min, x_max)
Rectangle(x_min, x_max, y_min, y_max)
Cube(x_min, x_max, y_min, y_max, z_min, z_max)
```

### Solver

```python
solve(equation, domain, bc=None, ic=None, params=None) -> Solution
```

**Parameters:**
- `equation` — PDE expressed as `lhs == rhs`
- `domain` — an `Interval`, `Rectangle`, `Cube`, or mesh object
- `bc` — boundary conditions dict; keys are boundary names, values are floats (Dirichlet) or `('neumann', value)` or `('robin', a, b)`
- `ic` — initial conditions for time-dependent problems (callable or array)
- `params` — optional solver parameters (resolution, tolerance, max_iter, etc.)

**Returns:** `Solution` object with:
- `sol(x)` / `sol(x, y)` — evaluate at a point
- `sol.plot()` — matplotlib visualization
- `sol.array` — raw NumPy array
- `sol.gradient()` — gradient field
- `sol.error_estimate` — estimated numerical error

---

## Observable Layer (`triality.observables`)

### compute_observables

```python
from triality.observables import compute_observables

observables = compute_observables(module_name, state, config, native_result=None)
```

Derives domain-specific engineering quantities from solved PhysicsState fields.

**Parameters:**
- `module_name` — name of the physics module (e.g. `"navier_stokes"`)
- `state` — `PhysicsState` from `RuntimeExecutionResult.generated_state`
- `config` — merged solver configuration dict
- `native_result` — optional raw solver result for metadata access

**Returns:** `List[Observable]` sorted by rank (most important first).

### Observable

```python
@dataclass
class Observable:
    name: str                       # e.g. "peak_velocity"
    value: Any                      # float, ndarray, bool, str
    unit: str                       # e.g. "m/s"
    description: str                # human-readable one-liner
    relevance: str = ""             # why it matters
    threshold: Optional[float]      # pass/fail threshold
    margin: Optional[float]         # distance to threshold (positive = safe)
    rank: int = 99                  # lower = more important (0 = primary)
```

**Properties:**
- `is_scalar` — True if value is int, float, or bool
- `to_dict()` — JSON-serializable dict

### OBSERVABLE_REGISTRY

```python
from triality.observables import OBSERVABLE_REGISTRY
# Dict[str, Type[BaseObservableSet]] — maps module names to observable set classes
# 16/16 modules registered
```

### Registered Observable Sets

| Module | Domain | Observables | Example quantities |
|--------|--------|------------|-------------------|
| `navier_stokes` | fluid | 10 | peak_velocity, vorticity_peak, dead_zone_fraction, reynolds_number |
| `coupled_physics` | nuclear | 12 | power_ratio, peak_fuel_temperature, doppler_feedback_pcm |
| `aero_loads` | aerospace | 10 | lift_coefficient, drag_coefficient, peak_heat_flux |
| `drift_diffusion` | semiconductor | 9 | built_in_potential, depletion_width, ideality_factor |
| `electrostatics` | EM | 8 | peak_field_strength, breakdown_margin, field_uniformity |
| `sensing` | sensing | 8 | mean_detection_probability, coverage_fraction_90pct, blind_zone_fraction |
| `structural_analysis` | structures | 8 | max_von_mises_stress, max_deflection, min_buckling_margin |
| `neutronics` | nuclear | 8 | k_effective, excess_reactivity_pcm, power_peaking_factor |
| `geospatial` | logistics | 8 | coverage_fraction, facilities_opened, meets_coverage_target |
| `battery_thermal` | thermal | 7 | peak_cell_temperature, runaway_risk, margin_to_runaway |
| `uav_aerodynamics` | aerospace | 7 | lift_coefficient, lift_to_drag_ratio, span_efficiency |
| `automotive_thermal` | thermal | 6 | junction_temperature_peak, margin_to_175C_limit |
| `spacecraft_thermal` | thermal | 6 | peak_temperature, hot_margin, cold_margin |
| `flight_mechanics` | dynamics | 6 | settling_time, max_angular_rate, fuel_consumed |
| `field_aware_routing` | EM | 8 | peak_field_strength, low_field_corridor_fraction |
| `structural_dynamics` | structures | 5 | peak_displacement, peak_acceleration, crest_factor |

---

## Electrostatics (`triality.electrostatics`)

### ElectrostaticSolver2D

```python
from triality.electrostatics import ElectrostaticSolver2D

solver = ElectrostaticSolver2D(
    x_range=(0, 0.1),    # metres
    y_range=(0, 0.1),
    resolution=50,        # grid points per axis
    epsilon_r=1.0         # relative permittivity
)
```

**Methods:**

```python
solver.set_boundary(side, bc_type, value)
# side: 'left', 'right', 'top', 'bottom'
# bc_type: 'voltage' (Dirichlet) or 'neumann'
# value: float

solver.set_component(name, position, voltage=None, charge=None)
# Add a lumped component (conductor, charge source)

solver.add_dielectric(region, epsilon_r)
# Add dielectric region

result = solver.solve()
```

**ElectrostaticResult:**

```python
result.voltage                          # 2D voltage array (V)
result.electric_field_x                 # Ex component (V/m)
result.electric_field_y                 # Ey component (V/m)
result.electric_field_magnitude()       # |E| array
result.energy_density()                 # ε₀εᵣ|E|²/2 (J/m³)
result.plot_voltage()
result.plot_electric_field()
result.plot_field_lines()
result.export_vtk(filename)
```

### ConductionSolver2D

```python
from triality.electrostatics import ConductionSolver2D

solver = ConductionSolver2D(
    x_range, y_range,
    resolution=50,
    conductivity=1.0      # S/m
)
solver.set_boundary('left', 'current', 1e-3)
solver.set_boundary('right', 'voltage', 0.0)
result = solver.solve()
```

**ConductionResult:**
```python
result.voltage
result.current_density_x
result.current_density_y
result.power_dissipation()   # σ|J|² (W/m³)
result.total_power()         # integrated power (W)
```

---

## Field-Aware Routing (`triality.field_aware_routing`)

### FieldAwareRouter

```python
from triality.field_aware_routing import FieldAwareRouter

router = FieldAwareRouter(
    physics_result,          # ElectrostaticResult or similar
    grid_resolution=100
)
```

**Methods:**

```python
router.set_start(point)               # (x, y) in metres
router.set_end(point)
router.add_waypoint(point)
router.add_obstacle(region)           # no-go zone
router.add_constraint(name, value)    # e.g. 'max_field', 1000

router.set_cost_weights(
    field_weight=1.0,    # penalty for high-field regions
    length_weight=0.1,   # penalty for path length
    bend_weight=0.05     # penalty for direction changes
)

path = router.route()
multi = router.route_multi(n_paths=3)   # find N alternative paths
```

**RoutingPath:**

```python
path.points           # list of (x,y) waypoints
path.length           # total path length (m)
path.max_field        # maximum field value along path
path.mean_field       # average field value along path
path.field_profile()  # field value vs. arc-length array
path.plot()
path.export_gerber(filename)   # PCB export
path.export_dxf(filename)
```

### CostFieldBuilder

```python
from triality.field_aware_routing import CostFieldBuilder

builder = CostFieldBuilder(grid_shape=(100, 100))
builder.add_field(efield_result, weight=1.0, transform='log')
builder.add_thermal(thermal_result, weight=0.5)
builder.add_keep_out(region)

cost_field = builder.build()
```

---

## Spatial Flow (`triality.spatial_flow`)

Low-level flow engine underlying the routing module.

```python
from triality.spatial_flow import FlowEngine, Source, Sink

engine = FlowEngine(grid_shape=(100, 100))
engine.add_source(Source(position=(10, 10), strength=1.0))
engine.add_sink(Sink(position=(90, 90)))
engine.set_cost_field(cost_array)

result = engine.solve()
path = result.extract_path()
```

---

## Thermal Modules

### ThermalHydraulicsSolver

```python
from triality.thermal_hydraulics import ThermalHydraulicsSolver

solver = ThermalHydraulicsSolver(
    geometry=pipe_geometry,
    fluid='water',
    T_inlet=300,    # K
    P_inlet=1e5,    # Pa
    flow_rate=0.01  # kg/s
)
result = solver.solve_steady()
```

### BatteryThermalModel

```python
from triality.battery_thermal import BatteryThermalModel

model = BatteryThermalModel(
    cell_type='NMC',
    n_cells=96,
    capacity_Ah=50,
    cooling='liquid'
)
result = model.simulate_drive_cycle(drive_cycle='WLTP')
print(f"Max cell temp: {result.T_max:.1f} K")
```

---

## Nuclear / Radiation

### NeutronicsCalculator

```python
from triality.neutronics import NeutronicsCalculator

calc = NeutronicsCalculator(
    geometry='cylinder',
    radius=0.5,    # m
    height=1.0,
    material='UO2',
    enrichment=0.04
)
keff = calc.k_effective()
flux = calc.neutron_flux()
```

### ShieldingCalculator

```python
from triality.shielding import ShieldingCalculator

shield = ShieldingCalculator()
shield.add_layer('concrete', thickness=0.5)   # m
shield.add_layer('lead',     thickness=0.05)
attenuation = shield.gamma_attenuation(energy_MeV=1.25)
```

---

## Multi-Physics Runtime (`triality.runtime`)

### PhysicsRuntime

```python
from triality.runtime import PhysicsRuntime

rt = PhysicsRuntime(dt=0.01, t_end=10.0)
rt.add_module('thermal',    thermal_module)
rt.add_module('structural', structural_module)
rt.couple('thermal.T', 'structural.thermal_load')
history = rt.run()
```

### RuntimeGraph

```python
from triality.runtime_graph import RuntimeGraph

graph = RuntimeGraph()
graph.add_node('electrostatics', ElectrostaticSolver2D(...))
graph.add_node('routing',        FieldAwareRouter(...))
graph.add_edge('electrostatics', 'routing', key='field_result')
results = graph.execute()
```

---

## Uncertainty Quantification (`triality.uncertainty_quantification`)

```python
from triality.uncertainty_quantification import UQAnalysis

uq = UQAnalysis(solver_fn)
uq.add_parameter('conductivity', dist='normal', mean=1.0, std=0.1)
uq.add_parameter('temperature',  dist='uniform', low=270, high=330)

results = uq.monte_carlo(n_samples=1000)
print(f"Mean output: {results.mean:.4f}")
print(f"Std output:  {results.std:.4f}")
print(f"P95:         {results.percentile(95):.4f}")

sobol = uq.sobol_indices()
```

---

## Verification Tools (`triality.verification`)

```python
from triality.verification import VerificationSuite

v = VerificationSuite()
v.add_case('poisson_1d', analytical_solution=lambda x: x*(1-x)/2)
v.run_all()
v.print_report()
# Expected output: all cases within tolerance
```
