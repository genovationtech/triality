# Examples

All examples are in `triality/examples/`. Run them with `python triality/examples/<file>.py`.

---

## Basic 1D PDE Solving (`basic_1d.py`)

Solve a 1D Poisson equation analytically and compare with Triality.

```python
from triality import Field, solve, laplacian, Interval

u = Field("u")

# Solve ∇²u = 1 on [0,1] with u(0)=u(1)=0
# Analytical solution: u(x) = x(x-1)/2
sol = solve(laplacian(u) == 1, Interval(0, 1), bc={'left': 0, 'right': 0})

import numpy as np
x = np.linspace(0, 1, 100)
analytical = x * (x - 1) / 2
numerical  = np.array([sol(xi) for xi in x])
error = np.abs(numerical - analytical).max()
print(f"Max error: {error:.2e}")   # < 1e-10
```

---

## Basic 2D PDE Solving (`basic_2d.py`)

Solve the 2D Laplace equation on a unit square.

```python
from triality import Field, solve, laplacian, Rectangle

u = Field("u")
bc = {
    'left': 0, 'right': 1,
    'top': ('neumann', 0), 'bottom': ('neumann', 0)
}
sol = solve(laplacian(u) == 0, Rectangle(0,1,0,1), bc=bc, params={'resolution': 50})
sol.plot(title="2D Laplace — Voltage")
```

---

## Electrostatics Demo (`electrostatics_demo.py`)

Full electrostatic analysis of a parallel-plate region.

```python
from triality.electrostatics import ElectrostaticSolver2D
import matplotlib.pyplot as plt

# 10cm × 10cm domain, 1mm resolution (100×100 grid)
solver = ElectrostaticSolver2D(
    x_range=(0, 0.1), y_range=(0, 0.1),
    resolution=100, epsilon_r=1.0
)

# Voltage boundaries
solver.set_boundary('left',   'voltage',  5.0)
solver.set_boundary('right',  'voltage',  0.0)
solver.set_boundary('top',    'neumann',  0.0)
solver.set_boundary('bottom', 'neumann',  0.0)

result = solver.solve()

print(f"Max |E|: {result.electric_field_magnitude().max():.1f} V/m")
print(f"Total stored energy: {result.stored_energy():.4e} J/m")  # per unit depth

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
result.plot_voltage(ax=axes[0])
result.plot_electric_field(ax=axes[1])
plt.tight_layout()
plt.savefig("electrostatics_result.png", dpi=150)
```

---

## Physics-Aware PCB Routing (`field_aware_routing_demo.py`)

Route a signal trace while avoiding high-field regions around power components.

```python
from triality.electrostatics import ElectrostaticSolver2D
from triality.field_aware_routing import FieldAwareRouter
import numpy as np

# 1. Set up PCB physics (10cm × 10cm board)
esolver = ElectrostaticSolver2D(
    x_range=(0, 0.1), y_range=(0, 0.1), resolution=100
)
# Power rail components
esolver.set_component('VCC',  position=(0.02, 0.08), voltage=5.0)
esolver.set_component('GND1', position=(0.02, 0.02), voltage=0.0)
esolver.set_component('GND2', position=(0.08, 0.05), voltage=0.0)
efield = esolver.solve()

# 2. Route a sensitive signal trace
router = FieldAwareRouter(efield, grid_resolution=100)
router.set_start((0.01, 0.05))
router.set_end((0.09, 0.05))
router.add_constraint('max_field', 500)   # 500 V/m limit for sensitive signals
router.set_cost_weights(field_weight=2.0, length_weight=0.2)

path = router.route()
print(f"Route length:  {path.length*100:.1f} mm")
print(f"Max |E| along route: {path.max_field:.0f} V/m")
print(f"EMI margin: {(500 - path.max_field)/500 * 100:.0f}%")

# Compare with shortest path
straight_path = router.route(physics_weight=0.0)
print(f"\nStraight path max |E|: {straight_path.max_field:.0f} V/m")
print(f"EMI reduction: {(straight_path.max_field - path.max_field)/straight_path.max_field*100:.0f}%")

path.plot(title="Physics-Aware Signal Routing")
```

---

## Datacenter Power Routing (`datacenter_power_routing.py`)

Demonstrates routing power buses in a server rack while managing thermal gradients.

```python
from triality.electrostatics import ConductionSolver2D
from triality.field_aware_routing import FieldAwareRouter, CostFieldBuilder

# 1. Solve current distribution in power plane
cond = ConductionSolver2D(
    x_range=(0, 0.3), y_range=(0, 0.2),
    resolution=150, conductivity=5.8e7  # copper
)
cond.set_boundary('left',  'current',  100.0)  # 100A input
cond.set_boundary('right', 'voltage',  0.0)
result = cond.solve()

# 2. Build cost field from current density (hot spots = high cost)
builder = CostFieldBuilder(grid_shape=(150, 150))
builder.add_field(result, weight=1.0, transform='quadratic')  # I²R heating
cost = builder.build()

# 3. Route additional bus to avoid hot spots
router = FieldAwareRouter(result)
paths = router.route_multi(
    start=(0.01, 0.1),
    end=(0.29, 0.1),
    n_paths=3
)

for i, p in enumerate(paths):
    print(f"Path {i+1}: length={p.length*100:.1f}mm, max_J={p.max_field:.0f} A/m²")
```

---

## Hospital Evacuation Planning (`hospital_evacuation_planning.py`)

Physics-aware evacuation routing that avoids fire-hot zones and smoke-filled corridors.

```python
from triality.field_aware_routing import FieldAwareRouter
from triality.geospatial import BuildingFloorPlan
import numpy as np

# Load floor plan (100×80 grid, each cell = 1m)
plan = BuildingFloorPlan.from_image("hospital_floor_2.png", scale=1.0)

# Define heat/smoke cost field (from fire simulation or sensor data)
heat_field = np.zeros((100, 80))
heat_field[40:60, 30:50] = 800   # fire zone (K above ambient)
heat_field[30:70, 25:55] = 200   # smoke zone

router = FieldAwareRouter.from_cost_array(
    cost_array=heat_field,
    obstacle_mask=plan.walls
)

# Route evacuation paths for all fire exits
exits = [(5, 10), (5, 70), (95, 40)]
start = (50, 40)   # current position (trapped person)

best_path = None
best_cost = float('inf')
for exit_pos in exits:
    path = router.route(start=start, end=exit_pos)
    if path.total_cost < best_cost:
        best_cost = path.total_cost
        best_path = path

print(f"Safest exit: {best_path.end}")
print(f"Max heat exposure: {best_path.max_field:.0f} K")
print(f"Path length: {best_path.length:.0f} m")
best_path.plot(overlay=plan.image)
```

---

## Drift-Diffusion Semiconductor (`drift_diffusion_production.py`)

Simulate a p-n junction diode.

```python
from triality.drift_diffusion import DriftDiffusionSolver

solver = DriftDiffusionSolver(
    length=1e-4,        # 100 µm device
    n_points=200,
    material='silicon',
    temperature=300     # K
)

# Doping profile: p-n junction at midpoint
solver.set_doping_profile(
    n_donor=lambda x: 1e16 if x > 5e-5 else 0,    # n-side: 10^16 cm^-3
    p_acceptor=lambda x: 1e16 if x < 5e-5 else 0   # p-side: 10^16 cm^-3
)

# Forward bias I-V curve
import numpy as np
voltages = np.linspace(-0.5, 0.7, 50)
currents = []
for V in voltages:
    result = solver.solve(applied_voltage=V)
    currents.append(result.terminal_current())

import matplotlib.pyplot as plt
plt.semilogy(voltages, np.abs(currents))
plt.xlabel("Voltage (V)")
plt.ylabel("Current density (A/cm²)")
plt.title("p-n Junction I-V Characteristic")
plt.grid(True)
plt.savefig("pn_junction_iv.png", dpi=150)
```

---

## High Voltage Safety Analysis (`hv_safety_demo.py`)

Check creepage and clearance distances for HV electronics.

```python
from triality.hv_safety import HVSafetyAnalyzer
from triality.electrostatics import ElectrostaticSolver2D

# Solve field around HV bus bar
solver = ElectrostaticSolver2D(
    x_range=(0, 0.05), y_range=(0, 0.05),
    resolution=200
)
solver.set_component('HV_bus', position=(0.025, 0.025), voltage=1000.0)
solver.set_boundary('all', 'voltage', 0.0)
efield = solver.solve()

# Safety analysis
safety = HVSafetyAnalyzer(efield, standard='IEC_60664')
report = safety.analyze()

print(f"Max field: {report.max_field:.0f} V/m")
print(f"Required clearance (IEC): {report.required_clearance*1000:.1f} mm")
print(f"Actual clearance: {report.actual_clearance*1000:.1f} mm")
print(f"Safety margin: {report.safety_factor:.1f}x")
print(f"PD inception voltage: {report.pd_inception_voltage:.0f} V")

if report.passes:
    print("✅ PASSES IEC 60664 requirements")
else:
    print("❌ FAILS — increase clearance or add insulation")
```

---

## Aerospace Kill-Switch Analysis

Quickly identify design showstoppers before committing to detailed simulation.

```python
from triality.flight_mechanics import FlightMechanicsModel
from triality.propulsion import PropulsionModel
from triality.structural_analysis import StructuralSolver

# 1. Check if propulsion meets thrust requirements
prop = PropulsionModel(engine_type='turbofan', bypass_ratio=8)
T_req = 25000  # N (required thrust)
T_avail = prop.max_thrust(altitude=10000, mach=0.82)
if T_avail < T_req:
    print(f"❌ KILL SWITCH: Thrust deficit {T_req - T_avail:.0f} N at cruise")
    print("   Cannot meet mission requirements — redesign propulsion")
    exit(1)

# 2. Check structural margins
struct = StructuralSolver.from_geometry("wing_geometry.stl")
struct.apply_load('2.5g_maneuver')
margins = struct.safety_factors()
if margins.min() < 1.0:
    failed = [(loc, sf) for loc, sf in margins.items() if sf < 1.0]
    print(f"❌ KILL SWITCH: Structural failure at {len(failed)} locations")
    for loc, sf in failed[:3]:
        print(f"   {loc}: SF = {sf:.2f}")
    exit(1)

# 3. All checks passed
print("✅ Concept survives kill-switch screening")
print(f"   Thrust margin: {(T_avail/T_req - 1)*100:.0f}%")
print(f"   Structural margin: {(margins.min() - 1)*100:.0f}%")
```

---

## Observable Layer — Post-Solver Engineering Quantities

Every runtime module produces domain-specific engineering observables from its solver output:

```python
from triality import load_module
from triality.observables import compute_observables

# Run any module
solver = load_module("battery_thermal").from_demo_case()
result = solver.solve()

# Derive engineering observables
obs = compute_observables("battery_thermal", result.generated_state, {})
for o in obs:
    line = f"  {o.name}: {o.value}"
    if o.unit:
        line += f" {o.unit}"
    if o.margin is not None:
        line += f"  [margin: {o.margin:+.1f}]"
    print(line)

# Example output:
#   peak_cell_temperature: 362.4 K
#   runaway_risk: False
#   temperature_spread: 8.2 K
#   margin_to_runaway: +40.6 K  [margin: +40.6]  ← positive = safe
#   safety_score: 0.85
```

The observable layer adds < 0.15% overhead to solver execution time.
