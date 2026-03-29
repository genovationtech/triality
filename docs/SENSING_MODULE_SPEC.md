# Triality Sensing Systems Module - Complete Specification

**Module**: `triality.sensing`
**Status**: 📚 Framework Implementation (Early-Stage Design Tool)
**Version**: 0.3.0

---

## Overview

The Triality Sensing Systems module provides **system-level trade-off analysis** and **early feasibility checking** for radar, lidar, sonar, and general sensing applications.

**Key Differentiation from Commercial Tools:**
- ❌ **ANSYS/CST**: Solve for fields, require full geometry
- ❌ **MATLAB**: Require you to know what you're doing
- ✅ **Triality**: "Will this even work?" + "What are the trade-offs?"

**Philosophy**: Kill bad ideas fast, illuminate good trade-offs, provide agent-friendly reasoning.

---

## The 8 Core Capability Buckets

### 1️⃣ Wave & Signal Fundamentals

**Purpose**: "What kind of wave am I sending?"

**Module**: `triality.sensing.signals`

**Capabilities:**
```python
from triality.sensing.signals import (
    SignalType,
    PulseCharacteristics,
    FrequencyBandwidth,
    TimeFrequencyTrade off
)

# Define signal
pulse = PulseCharacteristics(
    duration=1e-6,  # 1 microsecond
    bandwidth=10e6,  # 10 MHz
    pulse_type='chirp'
)

# Fundamental trade-offs
tradeoff = TimeFrequencyTradeoff.analyze(pulse)
print(tradeoff.resolution_capability)  # "1.5 cm range resolution"
print(tradeoff.detectability)          # "Moderate (1 μs pulse)"

# Layman output
print(pulse.explain())
# "Short pulse → better resolution, needs more bandwidth"
```

**Key Functions:**
- `PulseCharacteristics.from_requirements(resolution, range, velocity)`
- `SignalType.recommend(application, constraints)`
- `TimeFrequencyTradeoff.explain(signal)`

**Outputs (Agent-Friendly):**
- "Shorter pulse = better resolution but higher bandwidth"
- "Chirp allows long pulse with high resolution"
- "Burst detection trades off range for Doppler measurement"

---

### 2️⃣ Medium-Aware Propagation Physics (CRITICAL)

**Purpose**: "What happens to the wave while traveling?"

**Module**: `triality.sensing.propagation`

**Capabilities:**

**EM Propagation (Radar/Lidar):**
```python
from triality.sensing.propagation import EMPropagation, AtmosphericConditions

# Define environment
conditions = AtmosphericConditions(
    frequency=94e9,  # 94 GHz (W-band radar)
    range_m=1000,
    rain_rate_mm_hr=10,
    fog_visibility_m=500,
    temperature_c=15,
    humidity_percent=80
)

prop = EMPropagation(conditions)
loss_db = prop.total_loss()
print(f"Total propagation loss: {loss_db:.1f} dB")
print(f"Free-space: {prop.free_space_loss_db():.1f} dB")
print(f"Rain attenuation: {prop.rain_attenuation_db():.1f} dB")
print(f"Fog attenuation: {prop.fog_attenuation_db():.1f} dB")

# Layman explanation
print(prop.explain())
# "At 94 GHz in rain, signal attenuates 0.8 dB/km. At 1 km: 0.8 dB loss."
# "Rain dominates over fog at this frequency."
# "Consider lower frequency for all-weather operation."
```

**Acoustic Propagation (Sonar/Ultrasound):**
```python
from triality.sensing.propagation import AcousticPropagation

# Underwater sonar
conditions = AcousticPropagation.underwater(
    frequency=100e3,  # 100 kHz
    range_m=500,
    temperature_c=10,
    salinity_ppt=35,
    depth_m=50
)

loss_db = conditions.total_loss()
print(f"Absorption loss: {conditions.absorption_loss_db():.1f} dB")
print(f"Spreading loss: {conditions.spreading_loss_db():.1f} dB")

# Critical output
print(conditions.frequency_recommendation())
# "100 kHz absorbs quickly in water (0.03 dB/m)."
# "For 500m range, expect 15 dB absorption loss."
# "Consider 10-50 kHz for longer range."
```

**Key Classes:**
- `EMPropagation`: Radar/lidar propagation
- `AcousticPropagation`: Sonar/ultrasound propagation
- `AtmosphericConditions`: Weather effects on EM
- `UnderwaterConditions`: Ocean/freshwater effects on acoustic

**Outputs:**
- Loss breakdown (free-space, absorption, weather)
- Frequency recommendations
- "This frequency dies quickly in this medium"
- Range-dependent attenuation curves

---

### 3️⃣ Interaction With Targets (Reflection/Scattering)

**Purpose**: "What happens when the wave hits something?"

**Module**: `triality.sensing.targets`

**Capabilities:**

**Radar Cross Section (Conceptual):**
```python
from triality.sensing.targets import TargetSignature, TargetType

# Define target
drone = TargetSignature(
    target_type=TargetType.SMALL_DRONE,
    size_m=0.5,
    material='plastic_composite',
    frequency=10e9  # X-band
)

rcs_dbsm = drone.effective_rcs()
print(f"Effective RCS: {rcs_dbsm:.1f} dBsm")
print(f"Detectability: {drone.detectability_rating()}")  # "Poor - small size + non-metallic"

# Compare frequencies
rcs_comparison = drone.compare_frequencies([3e9, 10e9, 35e9, 94e9])
print(rcs_comparison.recommend())
# "Lower frequencies (3 GHz) give better detection for small drones"
# "Higher frequencies (94 GHz) see detail but weak return"
```

**Acoustic Reflectivity:**
```python
from triality.sensing.targets import AcousticTarget

submarine = AcousticTarget(
    target_type='large_vessel',
    length_m=100,
    material='steel',
    frequency=10e3  # 10 kHz
)

target_strength_db = submarine.target_strength()
print(f"Target Strength: {target_strength_db:.1f} dB")
print(submarine.explain())
# "Large steel target at 10 kHz: strong return (+30 dB)"
# "Good detection probability at moderate range"
```

**Key Functions:**
- `TargetSignature.estimate_rcs(size, shape, material, frequency)`
- `AcousticTarget.target_strength(size, material, frequency)`
- `SurfaceReflection.angle_dependent_return(incidence_angle, roughness)`

**Outputs:**
- Order-of-magnitude RCS/target strength
- Frequency sensitivity
- Material effects
- "Small drones reflect poorly at this frequency"

---

### 4️⃣ Noise, Clutter & Interference Reasoning

**Purpose**: "What hides my signal?"

**Module**: `triality.sensing.noise`

**Capabilities:**

```python
from triality.sensing.noise import (
    NoiseEnvironment,
    ClutterModel,
    SNRCalculator
)

# Define noise environment
env = NoiseEnvironment(
    frequency=10e9,
    bandwidth=10e6,
    temperature_k=290,
    clutter_type='urban',
    receiver_noise_figure_db=3.0
)

thermal_noise_dbm = env.thermal_noise_power()
clutter_power_dbm = env.clutter_power(range_m=1000)
interference_dbm = env.interference_estimate()

print(f"Thermal noise: {thermal_noise_dbm:.1f} dBm")
print(f"Clutter power: {clutter_power_dbm:.1f} dBm")
print(f"Dominant noise source: {env.dominant_source()}")

# SNR calculation
snr = SNRCalculator(
    transmit_power_w=1000,
    range_m=1000,
    rcs_dbsm=-10,  # Small target
    propagation=prop,
    noise_env=env
)

snr_db = snr.calculate()
print(f"SNR: {snr_db:.1f} dB")

if snr_db < 13:
    print("⚠️ SNR too low for reliable detection")
    print(snr.improve_suggestions())
    # "Increase power by 10 dB OR reduce range to 500m OR use larger antenna"
```

**Clutter Models:**
```python
# Ground clutter
ground_clutter = ClutterModel.ground_clutter(
    frequency=10e9,
    grazing_angle_deg=10,
    terrain='rough',
    polarization='VV'
)

# Sea clutter
sea_clutter = ClutterModel.sea_clutter(
    frequency=10e9,
    sea_state=3,
    grazing_angle_deg=5
)

# Urban clutter
urban_clutter = ClutterModel.urban_clutter(
    frequency=94e9,
    density='high_rise'
)
```

**Key Outputs:**
- "Your signal is 12 dB weaker than noise — detection unlikely"
- "Clutter dominates over thermal noise at this range"
- "Urban environment scatters signal — expect false alarms"

---

### 5️⃣ Detection & Resolution Limits

**Purpose**: "Even if physics allows it, can I see it?"

**Module**: `triality.sensing.detection`

**Capabilities:**

```python
from triality.sensing.detection import (
    RangeResolution,
    VelocityResolution,
    DetectionProbability,
    TrackingCapability
)

# Range resolution
range_res = RangeResolution.from_bandwidth(bandwidth=10e6)
print(f"Range resolution: {range_res.meters():.2f} m")
# "15 m range resolution — can't separate two cars"

# Velocity resolution
vel_res = VelocityResolution.from_coherent_integration(
    frequency=10e9,
    integration_time=0.1  # 100 ms
)
print(f"Velocity resolution: {vel_res.m_per_s():.2f} m/s")
# "0.15 m/s — can distinguish walking from running"

# Detection probability
det_prob = DetectionProbability.calculate(
    snr_db=13,
    num_pulses=10,
    false_alarm_rate=1e-6
)
print(f"Detection probability: {det_prob:.1%}")
# "90% detection probability with 10 pulses"

# Trade-off analysis
tradeoff = det_prob.explain_tradeoffs()
print(tradeoff)
# "Higher false alarm rate → better detection"
# "More pulses → better detection but slower update rate"
```

**Resolution Trade-offs:**
```python
from triality.sensing.detection import ResolutionTradeoff

# Analyze resolution capability
res = ResolutionTradeoff(
    bandwidth=10e6,
    integration_time=0.1,
    array_size_m=0.5
)

print(res.range_resolution())      # "15 m"
print(res.velocity_resolution())   # "0.15 m/s"
print(res.angular_resolution())    # "1.2 degrees"

# Can you see two close targets?
separation_m = 20
print(res.can_resolve_range(separation_m))
# "Yes - separation (20m) > resolution (15m)"
```

**Key Outputs:**
- "You can detect it, but can't distinguish two close targets"
- "Need 100 ms integration to measure velocity"
- "10 dB SNR → 50% detection probability"

---

### 6️⃣ System-Level Trade-off Engine (YOUR SUPERPOWER)

**Purpose**: Triality's killer feature - automated trade-off reasoning

**Module**: `triality.sensing.tradeoffs`

**Capabilities:**

```python
from triality.sensing.tradeoffs import (
    SensorTradeoffEngine,
    DesignConstraints,
    PerformanceGoals
)

# Define requirements
goals = PerformanceGoals(
    max_range_m=1000,
    range_resolution_m=1.0,
    detection_probability=0.9,
    false_alarm_rate=1e-6,
    update_rate_hz=10
)

constraints = DesignConstraints(
    max_power_w=100,
    max_antenna_size_m=0.5,
    frequency_band='X-band',  # 8-12 GHz
    environment='urban'
)

# Run trade-off analysis
engine = SensorTradeoffEngine(goals, constraints)
result = engine.analyze()

print(result.feasibility)
# "⚠️ MARGINAL - Requires optimization"

print(result.bottlenecks)
# "1. Power insufficient for 1000m range with 0.9 Pd"
# "2. 1m resolution requires 150 MHz bandwidth (may exceed frequency allocation)"
# "3. Urban clutter will degrade performance by ~5 dB"

print(result.recommended_changes)
# "Option A: Reduce range to 700m (keeps all other specs)"
# "Option B: Increase power to 250W (exceeds constraint by 2.5×)"
# "Option C: Accept 0.7 detection probability (from 0.9)"
# "Option D: Use coherent integration (slower update: 5 Hz)"

# Automated trade-space exploration
tradespace = engine.explore_tradespace(
    variables=['power', 'antenna_size', 'bandwidth', 'integration_time'],
    objective='minimize_cost'
)

print(tradespace.pareto_frontier())
# Shows multiple viable design points with trade-offs
```

**Multi-Dimensional Trade-offs:**
```python
from triality.sensing.tradeoffs import MultiObjectiveTrade off

# Classic radar range equation trade-offs
tradeoff = MultiObjectiveTradeoff.radar_range_equation()

# What happens if I change X?
sensitivity = tradeoff.sensitivity_analysis(
    parameter='transmit_power',
    change_factor=2.0  # Double power
)
print(sensitivity)
# "Doubling power → range increases by 19% (fourth-root dependence)"
# "Cost increases 2×, thermal management complexity increases 3×"

# Bandwidth vs Power vs Range
surface = tradeoff.generate_3d_tradeoff_surface(
    x='bandwidth',
    y='power',
    z='max_range'
)
surface.plot()  # Interactive 3D trade-space
```

**Agent-Friendly Recommendations:**
```python
# Ask for design recommendation
recommendation = engine.recommend_design()

print(recommendation.summary)
# "For your requirements, recommend:"
# "  - Frequency: 10 GHz (X-band)"
# "  - Power: 150W (1.5× constraint - justify with performance gain)"
# "  - Bandwidth: 120 MHz (1.2m resolution - close to goal)"
# "  - Antenna: 0.5m (at constraint limit)"
# "  - Integration: 50ms (20 Hz update rate)"
# ""
# "Expected Performance:"
# "  - Max range: 950m (95% of goal)"
# "  - Detection prob: 0.88 (98% of goal)"
# "  - Resolution: 1.2m (20% worse than goal)"
# ""
# "Trade-off Justification:"
# "  - Slight power exceedance yields 90% range improvement"
# "  - Urban clutter limits further optimization"
```

**This is what ANSYS CAN'T do.**

---

### 7️⃣ Multi-Sensor & Architecture Reasoning

**Purpose**: "Should I even use radar here?"

**Module**: `triality.sensing.architecture`

**Capabilities:**

```python
from triality.sensing.architecture import (
    SensorComparison,
    MultiSensorFusion,
    SensorSelection
)

# Compare modalities
comparison = SensorComparison.compare(
    application='small_drone_detection',
    range_m=500,
    environment='urban',
    weather='rain'
)

print(comparison.results)
# "Radar: Good range, poor resolution, works in rain"
# "Lidar: Excellent resolution, fails in rain/fog"
# "Camera: Good detail, limited range, fails at night"
# "Acoustic: Poor range, good for classification"

print(comparison.recommendation)
# "Fusion recommended: Radar (primary) + Camera (classification)"

# Multi-sensor fusion reasoning
fusion = MultiSensorFusion(
    sensors=['radar', 'lidar', 'camera'],
    environment='all-weather',
    priority='reliability'
)

print(fusion.strategy)
# "Use radar as primary (all-weather)"
# "Add lidar when visibility > 500m (precision)"
# "Use camera for classification when light available"
# ""
# "Fusion benefit: 25% better detection, 10× better classification"

# Active vs Passive trade-off
active_passive = SensorSelection.active_vs_passive_tradeoff(
    requirement='stealth_operation',
    range_m=1000
)
print(active_passive)
# "Active (radar): Detectable by ESM, good range performance"
# "Passive (IR/RF): Covert, limited range, depends on target emissions"
# "Recommendation: Use passive with intermittent active confirmation"
```

**Key Outputs:**
- "Radar for range, lidar for precision, camera for classification"
- "Lidar fails in rain — need radar backup"
- "Active reveals your position — consider passive"

---

### 8️⃣ Early Kill-Switch Logic (CRITICAL)

**Purpose**: "This idea won't work. Here's why."

**Module**: `triality.sensing.feasibility`

**Capabilities:**

```python
from triality.sensing.feasibility import FeasibilityCheck, PhysicsViolation

# Define proposed system
system = {
    'type': 'radar',
    'frequency': 94e9,  # 94 GHz
    'power': 10,  # 10W
    'antenna_size': 0.1,  # 10 cm
    'range_goal': 5000,  # 5 km
    'target_rcs': -20,  # Very small target (dBsm)
    'environment': 'heavy_rain'
}

# Run feasibility check
check = FeasibilityCheck(system)
result = check.evaluate()

if result.feasible:
    print("✅ System is feasible")
else:
    print("❌ SYSTEM NOT FEASIBLE")
    print("\nKill-Switch Reasons:")
    for reason in result.violations:
        print(f"  • {reason}")

    # Example output:
    # "❌ SYSTEM NOT FEASIBLE"
    # ""
    # "Kill-Switch Reasons:"
    # "  • Physics violation: 94 GHz attenuates 15 dB/km in heavy rain"
    # "  • At 5 km range: 75 dB rain loss alone"
    # "  • Required SNR impossible with 10W power and small antenna"
    # "  • Noise floor: -90 dBm, Return signal: -120 dBm (30 dB deficit)"
    # "  • Recommendation: Use 10 GHz (rain loss 0.5 dB/km) OR reduce range to 500m"

# Detailed violation analysis
for violation in result.violations:
    print(f"\n{violation.type}: {violation.severity}")
    print(violation.explanation)
    print(violation.fix_options)

# Example violation:
# "PhysicsViolation: CRITICAL"
# "Rain attenuation at 94 GHz is 15 dB/km. At 5 km, total loss is 75 dB."
# "This exceeds available link budget by 30 dB."
# ""
# "Fix Options:"
# "  1. Reduce frequency to 10 GHz (rain loss: 0.5 dB/km) - RECOMMENDED"
# "  2. Reduce range to 1 km (rain loss: 15 dB - marginal)"
# "  3. Increase power to 1000W (10× increase - impractical)"
```

**Kill-Switch Triggers:**

```python
# Automatic kill-switch detection
kill_switches = [
    PhysicsViolation.PROPAGATION_IMPOSSIBLE,
    PhysicsViolation.NOISE_DOMINATES,
    PhysicsViolation.RESOLUTION_IMPOSSIBLE,
    PhysicsViolation.POWER_INSANE,
    PhysicsViolation.SIZE_IMPRACTICAL,
    PhysicsViolation.ENVIRONMENT_DESTRUCTIVE
]

for trigger in result.kill_switches:
    print(f"🛑 {trigger.name}")
    print(f"   {trigger.explanation}")
    print(f"   Impact: {trigger.impact}")
    print(f"   Confidence: {trigger.confidence}\n")

# Example:
# "🛑 PROPAGATION_IMPOSSIBLE"
# "   Signal attenuates below noise floor before reaching target"
# "   Impact: Zero detection probability regardless of other parameters"
# "   Confidence: 100% (fundamental physics)"
```

**Automated Feasibility Report:**

```python
# Generate report for stakeholders
report = check.generate_report(audience='non_technical')

print(report)
# "FEASIBILITY ASSESSMENT: NOT VIABLE"
# ""
# "Your proposed 94 GHz radar will NOT work for the following reasons:"
# ""
# "1. RAIN KILLS THE SIGNAL (Critical)"
# "   - Radio waves at 94 GHz are absorbed heavily by rain"
# "   - At 5 km distance in heavy rain, signal is reduced by 75 dB"
# "   - This is like trying to hear a whisper in a loud stadium"
# "   - Fix: Use lower frequency (10 GHz) where rain absorption is 30× less"
# ""
# "2. POWER TOO LOW (Critical)"
# "   - Even in clear weather, 10W is insufficient for 5 km range"
# "   - Would need 200W minimum for reliable detection"
# "   - Fix: Increase power OR reduce range to 1.5 km"
# ""
# "3. TARGET TOO SMALL (Major)"
# "   - Small target reflects poorly at all frequencies"
# "   - Combined with rain and range, detection probability < 1%"
# "   - Fix: Accept closer range (1 km) or larger antenna (0.5m)"
# ""
# "RECOMMENDATION:"
# "Redesign with 10 GHz frequency, 100W power, and 2 km range goal."
# "This is physically achievable and practical."
```

**This saves companies millions by killing bad ideas BEFORE building hardware.**

---

## Complete API Example: End-to-End Analysis

```python
from triality.sensing import SensingSystemAnalysis

# Define proposed system
system = SensingSystemAnalysis(
    modality='radar',
    frequency=10e9,  # X-band
    power_w=100,
    antenna_diameter_m=0.5,
    pulse_width_s=1e-6,
    bandwidth_hz=50e6
)

# Define mission
mission = system.set_mission(
    target_type='small_drone',
    target_rcs_dbsm=-10,
    range_m=1000,
    environment='urban',
    weather='light_rain'
)

# Run complete analysis
result = system.analyze_full()

# 1. Feasibility check
print("=== FEASIBILITY ===")
print(result.feasibility.summary)
if not result.feasibility.viable:
    print("Kill-switches triggered:")
    for ks in result.feasibility.kill_switches:
        print(f"  • {ks}")
    exit()

# 2. Link budget
print("\n=== LINK BUDGET ===")
print(f"Transmit power: {result.link_budget.tx_power_dbm:.1f} dBm")
print(f"Free-space loss: {result.link_budget.free_space_loss_db:.1f} dB")
print(f"Rain loss: {result.link_budget.rain_loss_db:.1f} dB")
print(f"RCS: {result.link_budget.rcs_dbsm:.1f} dBsm")
print(f"Received power: {result.link_budget.rx_power_dbm:.1f} dBm")
print(f"Noise floor: {result.link_budget.noise_floor_dbm:.1f} dBm")
print(f"SNR: {result.link_budget.snr_db:.1f} dB")

# 3. Detection performance
print("\n=== DETECTION PERFORMANCE ===")
print(f"Detection probability: {result.detection.prob_detect:.1%}")
print(f"False alarm rate: {result.detection.false_alarm_rate:.1e}")
print(f"Range resolution: {result.detection.range_resolution_m:.2f} m")
print(f"Velocity resolution: {result.detection.velocity_resolution_mps:.2f} m/s")

# 4. Trade-off analysis
print("\n=== TRADE-OFFS ===")
print(result.tradeoffs.summary)
# "Current design achieves 85% of performance goals."
# ""
# "To improve performance:"
# "  - Increase power to 150W → +3 dB SNR → 95% detection probability"
# "  - Increase bandwidth to 100 MHz → 1.5m range resolution (50% better)"
# "  - Use coherent integration (100ms) → +10 dB processing gain"
# ""
# "Cost-benefit:"
# "  - Power increase: +$5k, +20% size → RECOMMENDED"
# "  - Bandwidth increase: Regulatory approval required → DIFFICULT"
# "  - Coherent integration: -50% update rate → ACCEPTABLE"

# 5. Alternative sensor reasoning
print("\n=== SENSOR ALTERNATIVES ===")
alternatives = result.compare_modalities(['radar', 'lidar', 'passive_rf'])
print(alternatives.recommendation)
# "Radar is appropriate for this mission."
# "Lidar offers better resolution but fails in rain."
# "Consider lidar for precision in clear weather."

# 6. Final recommendation
print("\n=== RECOMMENDATION ===")
print(result.recommendation.summary)
print(f"Confidence: {result.recommendation.confidence:.0%}")
```

---

## Module Organization

```
triality/sensing/
├── __init__.py                 # Top-level API
├── signals.py                  # Bucket 1: Signals & waveforms
├── propagation.py              # Bucket 2: Medium propagation
│   ├── em_propagation.py       # EM-specific (radar/lidar)
│   ├── acoustic_propagation.py # Acoustic (sonar/ultrasound)
│   └── atmospheric_models.py   # Weather effects
├── targets.py                  # Bucket 3: Target interaction
│   ├── rcs_models.py           # Radar cross section
│   ├── acoustic_target.py      # Acoustic reflectivity
│   └── material_properties.py  # Material-dependent scattering
├── noise.py                    # Bucket 4: Noise & clutter
│   ├── thermal_noise.py
│   ├── clutter_models.py
│   └── interference.py
├── detection.py                # Bucket 5: Detection & resolution
│   ├── range_resolution.py
│   ├── velocity_resolution.py
│   ├── angular_resolution.py
│   └── detection_probability.py
├── tradeoffs.py                # Bucket 6: Trade-off engine ⭐
│   ├── link_budget.py
│   ├── radar_range_equation.py
│   ├── sonar_equation.py
│   ├── pareto_frontier.py
│   └── sensitivity_analysis.py
├── architecture.py             # Bucket 7: Multi-sensor reasoning
│   ├── sensor_comparison.py
│   ├── fusion_strategy.py
│   └── active_passive.py
├── feasibility.py              # Bucket 8: Kill-switch logic ⭐
│   ├── physics_violations.py
│   ├── constraint_checking.py
│   └── recommendations.py
├── radar.py                    # Radar-specific high-level API
├── lidar.py                    # Lidar-specific high-level API
├── sonar.py                    # Sonar-specific high-level API
├── passive_sensing.py          # Passive sensors (IR, RF, etc.)
└── examples/
    ├── drone_detection.py
    ├── automotive_radar.py
    ├── underwater_sonar.py
    └── medical_ultrasound.py
```

---

## Agent-Friendly Interface

**Key Design Principle**: Triality sensing module is designed to be called by autonomous agents.

```python
from triality.sensing import AgentInterface

# Agent asks: "Can I detect a small drone at 1 km with 10W radar?"
query = AgentInterface.query(
    question="Can I detect a small drone at 1 km with 10W X-band radar in urban environment?",
    constraints={'budget': 10000, 'size': 'handheld'}
)

response = query.answer()
print(response.yes_no)  # "No - insufficient power"
print(response.explanation)
# "A 10W X-band radar cannot reliably detect a small drone at 1 km in an urban environment."
# ""
# "Reasons:"
# "  1. Small drone RCS (~-10 dBsm) gives weak return"
# "  2. Urban clutter adds ~5 dB noise"
# "  3. Calculated SNR: 8 dB (need 13 dB for 90% detection)"
# ""
# "To make this work:"
# "  - Increase power to 50W (5× increase) OR"
# "  - Reduce range to 500m OR"
# "  - Use larger antenna (0.5m vs 0.2m) OR"
# "  - Accept lower detection probability (70% instead of 90%)"

print(response.alternatives)
# ["Use 50W power (+$2k)", "Reduce range to 500m", "Use lidar (better but rain-limited)"]

print(response.confidence)  # 0.95 (high confidence in physics-based answer)
```

**Natural Language Interface:**
```python
# Agent can ask questions in natural language
agent = AgentInterface()

q1 = agent.ask("What frequency is best for detecting drones in rain?")
# "Lower frequencies (3-10 GHz) penetrate rain better than higher frequencies (35-94 GHz)."
# "Rain attenuation at 10 GHz: 0.5 dB/km vs 94 GHz: 15 dB/km."
# "Recommendation: Use X-band (8-12 GHz) for all-weather drone detection."

q2 = agent.ask("Why can't I use lidar underwater?")
# "Light attenuates rapidly in water (meters) vs sound (kilometers)."
# "Water absorption coefficient for optical: ~0.1-1 /m → range < 100m."
# "For underwater sensing beyond 100m, use sonar (acoustic) not lidar (optical)."

q3 = agent.ask("Trade off bandwidth vs power for better range resolution")
# "Bandwidth improves range resolution (Δr = c/2B) but costs spectrum."
# "Power improves SNR and detection range but costs size/cost/thermal."
# ""
# "If goal is resolution: Increase bandwidth (resolution scales linearly with B)."
# "If goal is detection: Increase power (range scales as P^(1/4))."
# ""
# "Bandwidth is usually the better lever for resolution."
```

---

## Status & Roadmap

**Current Status**: 📚 Framework Specification (v0.3.0)

**Implementation Priority:**

**Phase 1: Core Sensing Physics** (High Priority)
- ✅ Bucket 2: Propagation models (EM + acoustic)
- ✅ Bucket 3: Target interaction basics
- ✅ Bucket 4: Noise & clutter
- ✅ Bucket 5: Detection & resolution

**Phase 2: Trade-off Engine** (CRITICAL - Triality's Differentiator)
- ⭐ Bucket 6: System-level trade-offs
- ⭐ Bucket 8: Kill-switch logic
- Integration with Layers 1-2 for sensor placement optimization

**Phase 3: Advanced Features**
- Bucket 7: Multi-sensor fusion reasoning
- Bucket 1: Advanced waveform design
- Agent-friendly natural language interface

**Phase 4: Domain-Specific Packages**
- Automotive radar (77 GHz, FMCW)
- Drone detection
- Underwater sonar
- Medical ultrasound
- Through-wall radar

---

## Why This Module Matters

**Commercial tools (ANSYS, CST, MATLAB) force you to:**
1. Know the answer already
2. Build full geometric models
3. Wait hours for results
4. Interpret results yourself

**Triality Sensing module:**
1. ✅ Tells you if it will work BEFORE you build
2. ✅ Provides trade-off reasoning automatically
3. ✅ Answers in seconds, not hours
4. ✅ Gives actionable recommendations

**Use Case:**
- Engineer proposes 94 GHz radar for 5 km drone detection in rain
- ANSYS: "Build model, run simulation, discover it fails 3 weeks later"
- **Triality: "Won't work - rain kills signal at 94 GHz. Use 10 GHz. Here's why."** ⚡

**This is the "kill bad ideas fast" philosophy that saves companies millions.**

---

## Documentation References

Update documentation to include:
- **README.md**: Add sensing systems to Layer 4 or as separate capability
- **PHYSICS_MANIFESTO.md**: Add sensing to extended modules
- **TRIALITY_REFERENCE.md**: Full sensing module reference
- **BUSINESS.md**: Add sensing as key differentiator vs ANSYS/Mathworks

---

**© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.**
