# Triality Examples: Real-World Applications

This directory contains **production-ready examples** demonstrating Triality's Spatial Flow Engine applied to real industrial problems. These are not toys - they solve actual problems with real constraints, real physics, and real ROI calculations.

## Quick Start

```bash
cd triality/examples

# Run any example
python datacenter_power_routing.py
python manufacturing_material_flow.py
python hospital_evacuation_planning.py
python underground_utility_routing.py
python warehouse_robot_navigation.py
```

## Examples by Industry

### 1. Data Center Infrastructure
**File:** `datacenter_power_routing.py`

**Problem:** Route power cables from UPS to server racks while avoiding:
- Hot aisles (thermal degradation)
- Transformer EMI (signal integrity)
- Coolant pipes (clearance requirements)

**Real-world constraints:**
- Multi-objective cost (distance + heat + EMI)
- Hard obstacles (structural columns)
- Soft obstacles (traversable but costly)
- 6 server racks @ 3kW each = 18kW total load

**Output:**
- Optimal cable routes
- Total cable length required
- Installation cost breakdown
- Safety compliance checklist

**Key metrics:**
```
Total cable: ~45-60m
Cost premium: 15-25% (avoidance penalty)
Installation cost: ~$2,300-3,000
```

---

### 2. Manufacturing & Logistics
**File:** `manufacturing_material_flow.py`

**Problem:** Route material flow from warehouse to assembly stations in automotive plant

**Real-world constraints:**
- Worker safety zones (high foot traffic)
- Overhead crane paths (4m clearance)
- Congestion bottlenecks (throughput)
- Machinery obstacles (welding, press, paint)

**Physics modeled:**
- Congestion-dependent flow cost
- Bottleneck penalties
- Distance optimization (lean manufacturing)

**Output:**
- Material flow paths for 6 stations
- Throughput analysis (parts/hour)
- Labor cost calculations
- Lean manufacturing metrics

**Key metrics:**
```
Total path length: ~280-320m
Fleet requirement: 2-3 forklifts
Daily capacity: 4,800 parts
Efficiency: 85-92%
ROI: Labor savings quantified
```

---

### 3. Healthcare & Emergency Planning
**File:** `hospital_evacuation_planning.py`

**Problem:** Plan fire evacuation routes from patient wards to fire exits

**Real-world constraints:**
- Smoke accumulation zones (respiratory safety)
- Congestion at doorways (anti-trampling)
- Mobility constraints (wheelchairs, beds)
- Exit capacity limits

**Life safety codes:**
- NFPA 101 Life Safety Code
- Evacuation time < 4 minutes
- Exit load balancing

**Output:**
- Evacuation routes for 48 patients
- Time estimates by mobility type
- Staff requirements (2:1 for beds, 1:1 for wheelchairs)
- Bottleneck analysis

**Key metrics:**
```
Total patients: 48 (6 wards × 8 beds)
Avg evacuation time: 2.5-3.5 minutes
Staff required: 16-20 (for bed/wheelchair assist)
Exit utilization: Balanced across 4 exits
```

---

### 4. Urban Infrastructure
**File:** `underground_utility_routing.py`

**Problem:** Route gas distribution pipes to buildings in urban subsurface

**Real-world constraints:**
- Existing utilities (water, electrical, sewer)
- Bedrock zones (expensive excavation)
- Water table (pumping + corrosion)
- Protected areas (archaeological sites, tree roots)

**Regulatory compliance:**
- NEC 300.5 (electrical separation)
- NFPA 54 (gas main depth)
- Municipal codes (tree protection)

**Output:**
- Gas pipe routes to 7 buildings
- Total excavation length
- Project cost breakdown
- Construction timeline

**Key metrics:**
```
Pipe length: ~180-220m
Project cost: $75,000-95,000
Construction time: 12-15 weeks
Utility crossings: 4-6
Service life: 50+ years
```

---

### 5. Warehouse Automation
**File:** `warehouse_robot_navigation.py`

**Problem:** Plan navigation routes for fleet of warehouse robots

**Real-world constraints:**
- Static obstacles (shelves, columns)
- Dynamic obstacles (forklifts, humans)
- Energy consumption (battery life)
- Congestion at junctions
- Human safety zones

**Fleet optimization:**
- Multi-robot coordination
- Collision avoidance
- Battery management
- Throughput maximization

**Output:**
- Robot routes from charging to picking locations
- Energy consumption per trip
- Fleet throughput (picks/hour)
- ROI calculation

**Key metrics:**
```
Fleet size: 8 robots
Picks/hour: 360-450
Battery: 8-10 trips per charge
Daily capacity: 5,760-7,200 picks
Payback period: 1.5-2.5 years
```

---

## Common Features Across All Examples

### 1. Multi-Objective Optimization
All examples combine multiple cost factors:
- **Distance** (minimize travel)
- **Safety** (avoid hazards)
- **Efficiency** (reduce congestion, energy)
- **Compliance** (meet regulations)

### 2. Real Physics
Not simplified models:
- Heat dissipation (Gaussian fields)
- EMI propagation (inverse square law approximation)
- Congestion dynamics (flow-dependent costs)
- Energy consumption (distance + terrain)

### 3. Hard + Soft Constraints
- **Hard obstacles:** Cannot traverse (walls, machines, protected areas)
- **Soft obstacles:** Can traverse with penalty (heat zones, congestion, rough terrain)

### 4. Production Outputs
Every example provides:
- ✅ Quantified results (meters, dollars, hours)
- ✅ Safety compliance checklists
- ✅ Cost breakdowns
- ✅ ROI calculations
- ✅ Implementation recommendations

---

## How These Examples Differ from Typical Routing Demos

| Typical Demo | These Examples |
|--------------|----------------|
| Toy problem (5x5 grid) | Real scale (40m-100m domains) |
| Single objective (distance) | Multi-objective (3-5 factors) |
| Binary obstacles | Physics-based cost fields |
| No units | Real units (meters, watts, dollars) |
| "It works!" | Full cost/time/ROI analysis |
| Academic validation | Industry compliance codes |

---

## Technical Details

### Resolution
All examples use **high resolution (150-250 grid points)** for production accuracy:
- Data center: 150 (precision wiring)
- Manufacturing: 200 (complex layout)
- Hospital: 180 (safety-critical)
- Underground: 250 (regulatory compliance)
- Warehouse: 220 (robot precision)

### Cost Field Composition
Examples demonstrate sophisticated cost field construction:
```python
cost_fields = {
    'base_distance': CostFieldBuilder.uniform(weight=1.0),
    'hazard_1': CostFieldBuilder.gaussian_hotspot(center, sigma, amplitude, weight),
    'hazard_2': CostFieldBuilder.gaussian_hotspot(...),
    'terrain': CostFieldBuilder.radial(...),
}
combined = CostFieldBuilder.combine(cost_fields)
```

### Obstacle Types
- **Hard:** `ObstacleType.HARD` (impassable)
- **Soft:** `ObstacleType.SOFT` with `cost_multiplier` (penalty-based)

### Output Analysis
All examples compute:
1. **Route metrics:** Length, cost, time
2. **Resource requirements:** Materials, labor, equipment
3. **Financial analysis:** Installation cost, ROI, payback period
4. **Compliance:** Safety codes, regulations, best practices

---

## Use Cases by Industry

### Facilities & Construction
- Data center cable routing
- Underground utility planning
- HVAC duct layout
- Building evacuation routes

### Manufacturing & Logistics
- Material flow optimization
- Warehouse robot navigation
- Factory layout planning
- AGV path planning

### Healthcare & Public Safety
- Hospital evacuation planning
- Emergency egress design
- Patient transport routing
- Hazmat corridor planning

### Urban Infrastructure
- Gas/water main routing
- Electrical distribution
- Fiber optic networks
- Subway/tunnel planning

---

## Running the Examples

### Basic Usage
```bash
python datacenter_power_routing.py
```

### Expected Runtime
- Simple examples (data center): ~2-5 seconds
- Complex examples (underground): ~5-10 seconds
- All examples complete in < 15 seconds on standard hardware

### Dependencies
- NumPy ≥ 1.20
- SciPy ≥ 1.7
- Triality (installed via `pip install -e .`)

---

## Extending These Examples

### Add New Cost Factors
```python
# Add gradient penalty (slope/elevation)
slope_cost = CostFieldBuilder.linear_gradient(
    direction=(1.0, 0.5),  # Uphill direction
    weight=0.8
)
```

### Modify Physics
```python
# Change heat dissipation model
heat_field = CostFieldBuilder.gaussian_hotspot(
    sigma=2.0,  # Change spread distance
    amplitude=10.0,  # Change intensity
    weight=1.5  # Change importance
)
```

### Add Constraints
```python
# Add new obstacle
new_obstacle = ObstacleBuilder.polygon(
    vertices=[(x1, y1), (x2, y2), (x3, y3)],
    obstacle_type=ObstacleType.SOFT
)
new_obstacle.cost_multiplier = 75.0
engine.add_obstacle(new_obstacle)
```

---

## What Makes These Examples "Deep"

1. **Real constraints** - Not simplified models
2. **Real units** - Meters, watts, dollars (not arbitrary)
3. **Real codes** - Actual safety regulations cited
4. **Real ROI** - Installation costs and payback periods
5. **Real complexity** - Multiple objectives, physics-based costs
6. **Real scale** - Production domain sizes (40m-100m)

These are not academic exercises. They solve **real industrial problems** that companies pay consultants $50k-500k to optimize.

---

## Next Steps

Want to apply Triality to your specific use case?

1. **Start with closest example** (by industry)
2. **Modify domain dimensions** to match your facility
3. **Adjust obstacles** to match your layout
4. **Tune cost weights** for your priorities
5. **Add domain-specific constraints**

For custom applications, see `triality/spatial_flow/README.md` for the low-level API.

---

## License
MIT (same as Triality)

## Questions?
See main Triality documentation: `../README.md`
