# Triality Spatial Flow Engine

**Continuous routing and distribution problems using physics-based field optimization**

## Overview

The Spatial Flow Engine solves routing and distribution problems in continuous space using physics-based methods rather than traditional graph algorithms. This approach provides:

- **Continuous reasoning**: No discretization artifacts from graph construction
- **Physics-based optimization**: Solves Laplace/Poisson equations, not heuristic search
- **Natural handling of**: Obstacles, cost fields, multi-objective optimization
- **Extractable as**: Individual paths, networks, flow fields

## Key Advantage

```
Traditional Tools: Space → Discrete Graph → Heuristic Search (A*, Dijkstra)
Triality:           Space → Potential Field → Optimal Extraction
```

This is fundamentally more stable and explainable because:
1. Potential fields naturally encode optimal paths
2. No graph discretization errors
3. Obstacle/cost handling is continuous
4. Multiple objectives blend naturally

## Applications

The same engine solves:

| Application | Description | Status |
|------------|-------------|--------|
| Cable routing | Power and signal cable layout | ✅ Implemented |
| Pipe routing | Fluid pipeline networks | 🔜 Template coming |
| HVAC routing | Duct and ventilation layout | 🔜 Template coming |
| Warehouse paths | Logistics corridor planning | 🔜 Template coming |
| Factory flow | Material flow optimization | 🔜 Template coming |
| Evacuation paths | Emergency exit routing | 🔜 Template coming |
| Robot navigation | Path planning with obstacles | 🔜 Template coming |

## Quick Start

### Low-Level API (Full Control)

```python
from triality.spatial_flow import SpatialFlowEngine

# Create engine
engine = SpatialFlowEngine()

# Define problem
engine.set_domain((0, 1), (0, 1))
engine.add_source((0.1, 0.5), label="A")
engine.add_sink((0.9, 0.5), label="B")

# Optional: Add obstacles
from triality.spatial_flow import ObstacleBuilder
obstacle = ObstacleBuilder.rectangle(0.4, 0.6, 0.3, 0.7)
engine.add_obstacle(obstacle)

# Solve
network = engine.solve()

# Results
print(f"Network: {network}")
for path in network.paths:
    print(f"  {path}")
```

### High-Level API (Templates)

```python
from triality.spatial_flow.templates import cable_routing

# Route a cable with obstacles
network = cable_routing.route_cable(
    sources=[(0.1, 0.5)],
    sinks=[(0.9, 0.5)],
    obstacles=[
        {'type': 'rectangle', 'params': [0.4, 0.6, 0.3, 0.7]}
    ],
    resolution=100
)
```

### Advanced: Power Cable with Heat Avoidance

```python
from triality.spatial_flow.templates import cable_routing

network = cable_routing.route_power_cable(
    power_source=(0.1, 0.5),
    devices=[(0.9, 0.3), (0.9, 0.7)],
    heat_sources=[
        ((0.5, 0.5), 5.0)  # (center, intensity)
    ],
    heat_weight=0.8  # How much to avoid heat
)
```

### Advanced: Signal Cable with EMI Avoidance

```python
from triality.spatial_flow.templates import cable_routing

network = cable_routing.route_signal_cable(
    source=(0.1, 0.5),
    destination=(0.9, 0.5),
    emi_sources=[
        ((0.5, 0.5), 8.0)  # (center, intensity)
    ],
    emi_weight=1.0  # How much to avoid EMI
)
```

## Architecture

The module is structured in layers:

```
triality/spatial_flow/
├── engine.py              # Main SpatialFlowEngine
├── sources_sinks.py       # Source/sink definitions
├── cost_fields.py         # Spatial cost functions
├── constraints.py         # Obstacle handling
├── extraction.py          # Path extraction from fields
└── templates/             # High-level APIs
    ├── cable_routing.py   # Cable layout
    └── ...                # More coming
```

### Core Concepts

**1. Sources & Sinks**
- **Source**: Flow origin point (e.g., power supply)
- **Sink**: Flow destination (e.g., device to power)
- Each has position, weight, optional label

**2. Cost Fields**
- Spatial cost function: `(x, y) → cost`
- Built-in types: uniform, gaussian hotspot, linear gradient, radial
- Composable: combine multiple cost factors

**3. Obstacles**
- Hard: Completely forbidden regions
- Soft: High-cost but traversable
- Geometries: rectangle, circle, polygon, union

**4. Extraction**
- Traces paths by following gradient descent
- Simplifies using Douglas-Peucker algorithm
- Returns Network with all paths

## How It Works

The engine uses a 6-step process:

1. **Validate** inputs (sources, sinks, domain, obstacles)
2. **Build PDE**: Laplace equation `∇²φ = 0`
3. **Solve** potential field using Triality PDE solver
4. **Extract** paths via gradient descent
5. **Simplify** paths (optional)
6. **Return** Network object

The potential field `φ` encodes optimal routing:
- Low values at sources
- High values at sinks
- Gradient points toward sinks
- Paths follow negative gradient (steepest descent)

## API Reference

### SpatialFlowEngine

Main class for spatial flow problems.

**Methods:**

- `add_source(position, weight, label)`: Add flow source
- `add_sink(position, weight, label)`: Add flow sink
- `add_obstacle(obstacle)`: Add obstacle constraint
- `set_cost_field(cost_field)`: Set spatial cost function
- `set_domain(x_range, y_range)`: Set spatial domain
- `set_resolution(resolution)`: Set grid resolution
- `solve(verbose, simplify)`: Solve and extract paths
- `visualize(network)`: Plot the solution (requires matplotlib)

### Source / Sink

Flow origin/destination points.

```python
Source(position=(x, y), weight=1.0, label="A")
Sink(position=(x, y), weight=1.0, label="B")
```

### CostFieldBuilder

Factory for common cost field patterns.

**Methods:**

- `uniform(weight)`: Constant cost (pure distance)
- `gaussian_hotspot(center, sigma, amplitude, weight)`: Avoid hotspots
- `linear_gradient(direction, weight)`: Elevation/slope
- `radial(center, power, weight)`: Distance from point
- `combine(fields)`: Weighted sum of multiple fields

### ObstacleBuilder

Factory for common obstacle geometries.

**Methods:**

- `rectangle(xmin, xmax, ymin, ymax, type)`: Rectangular obstacle
- `circle(center, radius, type)`: Circular obstacle
- `polygon(vertices, type)`: Polygonal obstacle
- `union(obstacles)`: Combine multiple obstacles

### Path / Network

Result objects from solving.

```python
Path:
  - points: List of (x, y) coordinates
  - cost: Total path cost
  - source: Source label
  - sink: Sink label
  - length(): Euclidean path length

Network:
  - paths: List of Path objects
  - total_cost: Sum of all path costs
```

## Examples

### Example 1: Basic Routing

```python
from triality.spatial_flow import SpatialFlowEngine

engine = SpatialFlowEngine()
engine.set_domain((0, 1), (0, 1))
engine.add_source((0.1, 0.5), label="A")
engine.add_sink((0.9, 0.5), label="B")
engine.set_resolution(50)

network = engine.solve()
print(f"Path length: {network.paths[0].length():.3f}")
```

### Example 2: Multiple Destinations

```python
engine = SpatialFlowEngine()
engine.set_domain((0, 1), (0, 1))
engine.add_source((0.1, 0.5), weight=3.0, label="Source")
engine.add_sink((0.9, 0.3), weight=1.0, label="D1")
engine.add_sink((0.9, 0.5), weight=1.0, label="D2")
engine.add_sink((0.9, 0.7), weight=1.0, label="D3")

network = engine.solve()
# Returns 3 paths: Source→D1, Source→D2, Source→D3
```

### Example 3: Custom Cost Field

```python
from triality.spatial_flow import SpatialFlowEngine, CostFieldBuilder

# Define problem
engine = SpatialFlowEngine()
engine.set_domain((0, 1), (0, 1))
engine.add_source((0.1, 0.5))
engine.add_sink((0.9, 0.5))

# Add heat avoidance
heat = CostFieldBuilder.gaussian_hotspot(
    center=(0.5, 0.5),
    sigma=0.1,
    amplitude=10.0
)
engine.set_cost_field(heat)

network = engine.solve()
# Path will curve around the hot spot
```

### Example 4: Complex Obstacles

```python
from triality.spatial_flow import SpatialFlowEngine, ObstacleBuilder, ObstacleType

engine = SpatialFlowEngine()
engine.set_domain((0, 1), (0, 1))
engine.add_source((0.1, 0.5))
engine.add_sink((0.9, 0.5))

# Hard obstacle (forbidden)
hard = ObstacleBuilder.rectangle(0.4, 0.6, 0.3, 0.5,
                                obstacle_type=ObstacleType.HARD)
engine.add_obstacle(hard)

# Soft obstacle (high cost but traversable)
soft = ObstacleBuilder.circle((0.5, 0.7), 0.1,
                             obstacle_type=ObstacleType.SOFT)
soft.cost_multiplier = 100.0
engine.add_obstacle(soft)

network = engine.solve()
# Path will avoid hard obstacle, preferentially avoid soft obstacle
```

## Testing

Run comprehensive tests:

```bash
python triality/spatial_flow/test_spatial_flow.py
```

Tests cover:
- Basic routing
- Obstacle avoidance
- Custom cost fields
- Multiple sinks
- Template APIs
- Input validation

## Performance

- **Resolution**: Higher = more accurate, but slower
  - 50: Fast, good for prototyping
  - 100: Balanced (default)
  - 200: High quality, slower

- **Grid size**: O(n²) for 2D problems
- **Solver time**: O(n²) to O(n²·⁵) depending on method
- **Typical**: 0.1-1s for 100×100 grid

## Future Extensions

Planned templates:
- `pipe_routing`: Fluid pipeline networks
- `hvac_routing`: HVAC duct layout
- `warehouse_routing`: Logistics paths
- `robot_navigation`: Dynamic path planning

Planned features:
- 3D routing support
- Time-varying costs (dynamic obstacles)
- Multi-objective optimization (Pareto fronts)
- Network redundancy (backup paths)

## Mathematical Background

The engine solves the **Laplace equation**:

```
∇²φ = 0
```

Where `φ` is the potential field. This is the same equation that governs:
- Electrostatic potential
- Temperature in steady-state heat conduction
- Fluid flow in porous media

The gradient `∇φ` points in the direction of steepest increase. Paths follow
the negative gradient `-∇φ` (steepest descent) from sinks to sources.

For problems with cost fields, the engine solves the **Eikonal equation** or
weighted Laplace equation to incorporate spatial costs.

## References

- Triality PDE Solver: `triality/solvers/`
- Production verification: `triality/verification/`
- Core IR specification: `triality/IR_SPEC.md`

## License

Part of the Triality library.
