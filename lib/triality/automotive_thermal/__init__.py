"""
Layer 11: Automotive Thermal & Heat Transfer

Production-ready thermal analysis for automotive power electronics,
battery packs, and electrical systems.

Key Capabilities:
==================
- Transient heat conduction (time-dependent thermal response)
- Joule heating from electrical current
- Multi-source heat aggregation
- Hotspot detection and ranking
- Thermal margin computation
- Cooling effectiveness evaluation

Physics Models:
===============
Transient Heat Equation:
∂T/∂t = α∇²T + Q/(ρc)

where:
- α = k/(ρc): thermal diffusivity
- Q: volumetric heat generation [W/m³]

Joule Heating:
Q_joule = ρ_e · J² = ρ_e · (I/A)²

where:
- ρ_e: electrical resistivity [Ω·m]
- J: current density [A/m²]
- I: current [A]
- A: cross-sectional area [m²]

Convective Cooling:
q = h · A · (T - T_ambient)

where:
- h: heat transfer coefficient [W/(m²·K)]
- A: surface area [m²]

Typical Usage:
==============
```python
from triality.automotive_thermal import (
    TransientHeatSolver1D, COPPER, HeatSource,
    calculate_busbar_temperature_rise
)

# Busbar steady-state analysis
result = calculate_busbar_temperature_rise(
    current=500.0,  # A
    length=0.3,     # m
    width=0.05,     # m
    thickness=0.005, # m
    material=COPPER
)
print(f"Temperature: {result['temperature_C']:.1f}°C")
print(f"Margin: {result['thermal_margin_K']:.1f} K")

# Transient analysis with pulsed power
solver = TransientHeatSolver1D(
    length=0.1,  # m
    n_points=50,
    material=COPPER
)

# Heat source (e.g., IGBT switching loss)
source = HeatSource(
    position=(0.05, 0, 0),  # Center
    power=100.0,  # W
    volume=1e-6   # m³
)

result = solver.solve_transient(
    t_end=60.0,   # 1 minute
    dt=0.1,       # s
    heat_sources=[source],
    h_conv=100.0  # Forced air cooling
)

print(f"Max temperature: {result.max_temperature:.1f} K")
print(f"Hotspots: {len(result.hotspots)}")
for hs in result.hotspots:
    print(f"  {hs.position}: {hs.temperature:.1f} K ({hs.risk_level})")
```

Automotive Applications:
========================
1. **Busbar Thermal Analysis**
   - Current carrying capacity
   - Temperature rise under load
   - Thermal derating

2. **Power Module Cooling**
   - IGBT/MOSFET junction temperature
   - Heatsink effectiveness
   - Transient thermal impedance

3. **Cable Ampacity**
   - Continuous current rating
   - Short-circuit heating
   - Bundle derating

4. **Battery Pack Thermal**
   - Cell heating from I²R losses
   - Pack-level hot spots
   - Cooling system design

Safety Margins:
===============
- Copper busbar: < 200°C (473 K)
- Aluminum cable: < 180°C (453 K)
- Silicon junction: < 175°C (448 K)
- Thermal interface: < 150°C (423 K)

Typical thermal margins: 20-50 K for safe operation

Accuracy:
=========
- 1D transient: ±5-10% (validated against FEA)
- Busbar steady-state: ±10-15%
- Assumes uniform material properties
- Neglects radiation (typically < 5% for T < 200°C)

Limitations:
============
- 1D solver for simple geometries
- Temperature-independent properties
- Simplified convection model
- No radiation heat transfer
- No phase change (melting)

For complex 3D geometries, couple with FEA or use as
first-order estimate for design iterations.
"""

from triality.automotive_thermal.transient_heat import (
    TransientHeatSolver1D,
    ThermalMaterial,
    HeatSource,
    Hotspot,
    ThermalResult,
    CoolingMode,
    COPPER,
    ALUMINUM,
    SILICON,
    THERMAL_PAD,
    COOLANT,
    calculate_busbar_temperature_rise,
)

from triality.automotive_thermal.solver import (
    AutomotiveThermalSolver,
    AutomotiveThermalResult,
    ComponentNode,
    ThermalContact,
)

__all__ = [
    'TransientHeatSolver1D',
    'ThermalMaterial',
    'HeatSource',
    'Hotspot',
    'ThermalResult',
    'CoolingMode',
    'COPPER',
    'ALUMINUM',
    'SILICON',
    'THERMAL_PAD',
    'COOLANT',
    'calculate_busbar_temperature_rise',
    'AutomotiveThermalSolver',
    'AutomotiveThermalResult',
    'ComponentNode',
    'ThermalContact',
]
