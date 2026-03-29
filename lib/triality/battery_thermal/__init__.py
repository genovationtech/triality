"""
Layer 13: Battery Pack Thermal Analysis for Automotive

Production-ready thermal modeling for EV battery packs focusing on
safety, cooling design, and thermal propagation.

**IMPORTANT**: This is NON-ELECTROCHEMICAL thermal physics
- Models heat generation and propagation
- Does NOT model lithium-ion chemistry details
- Suitable for pack-level thermal design and safety

Key Capabilities:
==================
- Cell-level heat generation (I²R + calendar)
- Cell-to-cell thermal propagation
- Thermal runaway trigger and propagation
- Cooling system effectiveness evaluation
- Temperature uniformity analysis
- Safety spacing logic

Physics Models:
===============
Cell Energy Balance:
m·c·dT/dt = Q_gen - Q_cool - Q_neighbors

Heat Generation:
Q_gen = I²R + Q_calendar + Q_runaway(if triggered)

Cooling:
Q_cool = h·A·(T - T_coolant)

Cell-to-Cell:
Q_ij = k_eff·A_contact·(T_i - T_j) / distance

Thermal Runaway:
- Trigger: T > 130°C (NMC) or 150°C (LFP)
- Heat release: Q_max = 50-100 kW per automotive cell
- Time constant: τ ~15 seconds

Typical Usage:
==============
```python
from triality.battery_thermal import (
    BatteryPackThermalModel, PackConfiguration,
    CoolingType, evaluate_cooling_adequacy
)

# Configure pack
config = PackConfiguration(
    n_cells=96,  # Typical module
    cell_spacing=0.022,  # m (22 mm)
    cooling_type=CoolingType.LIQUID_INDIRECT,
    h_convection=100.0,  # Cold plate cooling
    T_coolant=298.0  # 25°C
)

# Create model
pack = BatteryPackThermalModel(config)

# Current profile (2C discharge for 10 minutes)
current_profile = np.ones(6000) * 100.0  # 100 A (2C for 50 Ah cell)

# Solve transient
result = pack.solve_transient(
    current_profile=current_profile,
    t_end=600.0,  # 10 minutes
    dt=0.1
)

print(f"Max temperature: {result.max_temperature - 273.15:.1f}°C")
print(f"Temperature delta: {result.max_temp_delta:.1f} K")
print(f"Cells in runaway: {result.cells_in_runaway}")
print(f"Cooling adequate: {result.cooling_adequate}")

# Evaluate cooling system
evaluation = evaluate_cooling_adequacy(
    pack_model=pack,
    current_A=100.0,
    duration_s=600.0
)

print(f"Meets temperature limit: {evaluation['meets_temp_limit']}")
print(f"Meets uniformity: {evaluation['meets_uniformity']}")
```

Automotive Battery Pack Design:
================================
**Typical Configurations**:
- Module: 8-24 cells
- Pack: 200-400 cells
- Voltage: 400-800 V
- Capacity: 50-100 kWh

**Cooling Requirements**:
- Air cooling: h = 20-50 W/(m²·K)
- Liquid indirect: h = 100-200 W/(m²·K)
- Immersion: h = 500-1000 W/(m²·K)

**Temperature Limits**:
- Operating: 15-45°C (optimal)
- Maximum: 55-60°C (continuous)
- Critical: 80°C (degradation accelerates)
- Runaway: 130°C (NMC), 150°C (LFP)

**Uniformity Requirements**:
- ΔT across pack: < 5 K (preferred)
- ΔT across module: < 3 K (preferred)

Thermal Runaway Propagation:
=============================
**Critical Safety Analysis**:
- Single cell enters runaway at T > 130°C
- Releases 50-100 kW heat over 15-30 seconds
- Adjacent cells heat up from thermal propagation
- Chain reaction if spacing insufficient

**Mitigation Strategies**:
- Cell spacing: > 10 mm minimum
- Thermal barriers: Intumescent materials
- Active venting: Pressure relief valves
- Cooling augmentation: Emergency cooling boost

**Propagation Time**:
- Without barriers: 1-5 minutes to adjacent cells
- With barriers: 10-30 minutes (allows evacuation)

Safety Analysis Features:
==========================
1. **Hotspot Detection**: Identifies cells exceeding limits
2. **Propagation Modeling**: Predicts chain reaction
3. **Cooling Adequacy**: Validates cooling system design
4. **Temperature Uniformity**: Ensures even aging
5. **Thermal Runaway Events**: Tracks trigger and spread

Validation Against Standards:
==============================
- SAE J2464: EV Battery Safety
- UN 38.3: Transport Testing
- UL 2580: Battery Safety for EVs
- ISO 6469: EV Safety Requirements

(Note: This tool provides physics-based analysis,
not certification-grade compliance testing)

Non-Capabilities (Intentional):
================================
✗ Detailed electrochemistry (lithium plating, SEI)
✗ Capacity fade prediction
✗ State-of-health estimation
✗ Battery management system (BMS) logic
✗ Cell balancing

These require electrochemical models beyond thermal physics.

Accuracy:
=========
- Temperature prediction: ±2-5 K (compared to testing)
- Runaway timing: ±10-20% (high uncertainty in trigger)
- Cooling effectiveness: ±10-15%

Assumptions:
- Lumped thermal mass per cell
- Simplified cell-to-cell coupling
- No electrochemical detail
- Uniform cell aging

For detailed cell-level electrochemical-thermal coupling,
use specialized battery modeling tools (COMSOL Battery, GT-AutoLion).
"""

from triality.battery_thermal.pack_thermal import (
    BatteryPackThermalModel,
    BatteryCell,
    PackConfiguration,
    ThermalRunawayEvent,
    PackThermalResult,
    CellChemistry,
    CoolingType,
    evaluate_cooling_adequacy,
)

from triality.battery_thermal.solver import (
    BatteryThermalSolver,
    BatteryThermalSolverResult,
    DriveCycleSegment,
)

__all__ = [
    'BatteryPackThermalModel',
    'BatteryCell',
    'PackConfiguration',
    'ThermalRunawayEvent',
    'PackThermalResult',
    'CellChemistry',
    'CoolingType',
    'evaluate_cooling_adequacy',
    'BatteryThermalSolver',
    'BatteryThermalSolverResult',
    'DriveCycleSegment',
]
