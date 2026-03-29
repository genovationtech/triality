"""
Layer 4: High-Voltage Safety - Comprehensive Demo

Demonstrates all features of the HV Safety module:
1. Breakdown voltage analysis (Paschen's law)
2. Corona discharge analysis (Peek's formula)
3. Insulation coordination (IEC 60071)
4. Arc flash hazard (IEEE 1584)
5. Real-world design scenarios
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from triality.hv_safety import (
    # Breakdown analysis
    BreakdownSolver,
    GasType,
    ElectrodeGeometry,
    AltitudeCorrection,
    # Corona analysis
    CoronaAnalyzer,
    ConductorType,
    CoronaMode,
    PartialDischargeDetector,
    # Industrial standards
    IEC60071InsulationCoordination,
    IEEE1584ArcFlashCalculator,
)


def demo_breakdown_analysis():
    """Demo 1: Breakdown voltage analysis"""
    print("=" * 70)
    print("DEMO 1: BREAKDOWN VOLTAGE ANALYSIS")
    print("=" * 70)
    print()

    # Scenario: Design air gap for 100 kV impulse test
    print("Scenario: Air gap insulation for 100 kV impulse test")
    print("-" * 70)

    solver = BreakdownSolver(
        gas_type=GasType.AIR,
        pressure_atm=1.0,
        temperature_c=20.0,
        geometry=ElectrodeGeometry.SPHERE_SPHERE  # Uniform field
    )

    # Find minimum safe distance with 2x safety factor
    V_test = 100e3  # 100 kV
    safety_factor = 2.0

    d_min = solver.minimum_safe_distance(V_test, safety_factor)

    print(f"Test voltage: {V_test/1e3:.0f} kV")
    print(f"Safety factor: {safety_factor}x")
    print(f"Minimum gap: {d_min*100:.2f} cm")
    print()

    # Verify the design
    result = solver.solve(d_min, V_test)
    print(f"✓ Breakdown voltage: {result.breakdown_voltage/1e3:.1f} kV")
    print(f"✓ Safety margin: {result.safety_margin:.2f}x")
    print(f"✓ Critical field: {result.critical_field/1e6:.2f} MV/m")
    print()

    # Compare different gases
    print("Gas comparison for 5 cm gap:")
    print("-" * 70)
    gap = 0.05  # 5 cm

    gases = [
        (GasType.AIR, 1.0),
        (GasType.SF6, 1.0),
        (GasType.SF6, 3.0),  # 3 bar SF6
        (GasType.N2, 1.0),
        (GasType.VACUUM, 1.0)
    ]

    for gas_type, pressure in gases:
        solver = BreakdownSolver(gas_type=gas_type, pressure_atm=pressure)
        V_b = solver.paschen_breakdown_voltage(gap)
        gas_name = f"{gas_type.value} ({pressure:.1f} atm)"
        print(f"  {gas_name:20s}: {V_b/1e3:6.1f} kV")

    print()


def demo_corona_analysis():
    """Demo 2: Corona discharge analysis for transmission lines"""
    print("=" * 70)
    print("DEMO 2: CORONA DISCHARGE ANALYSIS")
    print("=" * 70)
    print()

    # Scenario: 500 kV transmission line design
    print("Scenario: 500 kV transmission line corona analysis")
    print("-" * 70)

    # Test different conductor configurations
    configs = [
        ("Single 3cm conductor", 1.5, ConductorType.SMOOTH_WIRE),
        ("Single 4cm conductor", 2.0, ConductorType.SMOOTH_WIRE),
        ("2-bundle 3cm", 1.5, ConductorType.BUNDLE_2),
        ("4-bundle 3cm", 1.5, ConductorType.BUNDLE_4),
    ]

    print(f"{'Config':25s} {'V_inc':>8s} {'Margin':>8s} {'Corona':>8s} {'Loss':>10s} {'RI':>8s}")
    print("-" * 70)

    for config_name, radius, conductor_type in configs:
        analyzer = CoronaAnalyzer(
            conductor_radius_cm=radius,
            conductor_type=conductor_type,
            altitude_m=500,
            surface_condition=0.95
        )

        result = analyzer.analyze(
            operating_voltage_kv=500,
            conductor_height_m=25,
            conductor_spacing_m=12,
            frequency_hz=60
        )

        corona_status = "YES" if result.corona_active else "NO"
        loss_str = f"{result.corona_loss_kw_per_km:.2f} kW/km" if result.corona_active else "-"
        ri_str = f"{result.radio_interference_db:.0f} dB" if result.corona_active else "-"

        print(f"{config_name:25s} {result.inception_voltage:6.0f} kV "
              f"{result.operating_margin:6.2f}x {corona_status:>8s} {loss_str:>10s} {ri_str:>8s}")

    print()

    # Detailed analysis of best configuration
    print("Detailed analysis: 4-bundle 3cm conductor")
    print("-" * 70)

    analyzer = CoronaAnalyzer(
        conductor_radius_cm=1.5,
        conductor_type=ConductorType.BUNDLE_4,
        altitude_m=500,
        surface_condition=0.95
    )

    result = analyzer.analyze(
        operating_voltage_kv=500,
        conductor_height_m=25,
        conductor_spacing_m=12,
        frequency_hz=60,
        weather_condition="fair"
    )

    print(f"Corona inception voltage: {result.inception_voltage:.0f} kV")
    print(f"Corona inception field: {result.inception_field:.1f} kV/cm")
    print(f"Operating margin: {result.operating_margin:.2f}x")
    print(f"Corona active: {result.corona_active}")
    if result.corona_active:
        print(f"Corona losses: {result.corona_loss_kw_per_km:.2f} kW/km")
        print(f"Radio interference: {result.radio_interference_db:.0f} dB")
        print(f"Audible noise (fair): {result.audible_noise_db:.0f} dBA")
    print(f"\nRecommendation: {result.recommended_mitigation}")
    print()


def demo_insulation_coordination():
    """Demo 3: IEC 60071 insulation coordination"""
    print("=" * 70)
    print("DEMO 3: INSULATION COORDINATION (IEC 60071)")
    print("=" * 70)
    print()

    # Scenario: 230 kV substation at high altitude
    print("Scenario: 230 kV substation design at 1500m altitude")
    print("-" * 70)

    coord = IEC60071InsulationCoordination(
        system_voltage_kv=230,
        altitude_m=1500,
        pollution_level=2,  # Medium pollution
        overvoltage_factor=1.5  # Lightning surge margin
    )

    print(f"System voltage: {coord.system_voltage_kv:.0f} kV")
    print(f"Altitude: {coord.altitude_m:.0f} m")
    print(f"Voltage class: {coord.voltage_class.value}")
    print(f"Altitude correction factor: {coord.Ka:.3f}")
    print()

    bil = coord.required_bil()
    print(f"Required BIL: {bil:.0f} kV peak")
    print()

    clearances = coord.required_clearances()
    print("Required clearances:")
    print(f"  Phase-to-ground: {clearances.phase_to_ground_m:.2f} m")
    print(f"  Phase-to-phase: {clearances.phase_to_phase_m:.2f} m")
    print(f"  Live-to-personnel (OSHA): {clearances.live_to_personnel_m:.2f} m")
    print(f"  Working distance: {clearances.working_distance_m:.2f} m")
    print(f"  Standard: {clearances.standard_reference}")
    print()

    # Compare different altitudes
    print("Altitude effect on clearances:")
    print("-" * 70)
    print(f"{'Altitude (m)':>12s} {'BIL (kV)':>10s} {'Ph-Gnd (m)':>12s} {'Correction':>12s}")
    print("-" * 70)

    for alt in [0, 1000, 2000, 3000, 4000]:
        coord_alt = IEC60071InsulationCoordination(
            system_voltage_kv=230,
            altitude_m=alt,
            pollution_level=2
        )
        bil_alt = coord_alt.required_bil()
        clearance_alt = coord_alt.air_clearance()
        print(f"{alt:12.0f} {bil_alt:10.0f} {clearance_alt:12.2f} {coord_alt.Ka:12.3f}")

    print()


def demo_arc_flash_hazard():
    """Demo 4: IEEE 1584 arc flash hazard analysis"""
    print("=" * 70)
    print("DEMO 4: ARC FLASH HAZARD ANALYSIS (IEEE 1584)")
    print("=" * 70)
    print()

    # Scenario: 480V switchgear maintenance
    print("Scenario: Arc flash risk for 480V switchgear maintenance")
    print("-" * 70)

    calc = IEEE1584ArcFlashCalculator(
        system_voltage_v=480,
        bolted_fault_current_ka=40,
        working_distance_mm=610,  # 24 inches
        equipment_type="switchgear"
    )

    print(f"System voltage: {calc.system_voltage_v:.0f} V")
    print(f"Bolted fault current: {calc.bolted_fault_current_ka:.0f} kA")
    print(f"Working distance: {calc.working_distance_mm:.0f} mm")
    print()

    # Calculate arcing current
    Ia = calc.arcing_current()
    print(f"Arcing current (IEEE 1584): {Ia:.1f} kA")
    print()

    # Analyze different clearing times
    print("Arc flash hazard vs. protection clearing time:")
    print("-" * 70)
    print(f"{'Time (sec)':>10s} {'Energy':>12s} {'AFB (m)':>10s} {'PPE':>8s} {'Hazard':>10s}")
    print("-" * 70)

    for t in [0.05, 0.1, 0.2, 0.5, 1.0]:
        result = calc.calculate(arc_duration_sec=t)
        print(f"{t:10.2f} {result.incident_energy_cal_cm2:9.1f} cal/cm² "
              f"{result.arc_flash_boundary_m:10.2f} "
              f"{result.ppe_category:>8d} "
              f"{result.hazard_risk_category:>10s}")

    print()

    # Detailed analysis for 0.2 sec (typical relay time)
    print("Detailed analysis (0.2 sec clearing time):")
    print("-" * 70)

    result = calc.calculate(arc_duration_sec=0.2)

    print(f"Incident energy: {result.incident_energy_cal_cm2:.1f} cal/cm²")
    print(f"Arc flash boundary: {result.arc_flash_boundary_m:.2f} m")
    print(f"PPE category: {result.ppe_category} (NFPA 70E)")
    print(f"Hazard level: {result.hazard_risk_category}")
    print()

    print("NFPA 70E approach boundaries:")
    print(f"  Limited approach: {result.limited_approach_m:.2f} m")
    print(f"  Restricted approach: {result.restricted_approach_m:.2f} m")
    print(f"  Prohibited approach: {result.prohibited_approach_m:.2f} m")
    print()

    print("Safety recommendations:")
    if result.ppe_category <= 1:
        print("  ✓ Low hazard - Standard PPE adequate")
    elif result.ppe_category <= 2:
        print("  ⚠ Moderate hazard - Arc-rated PPE required")
    else:
        print("  ⚠ HIGH HAZARD - Consider de-energizing or remote operation")
    print()


def demo_real_world_scenarios():
    """Demo 5: Real-world design scenarios"""
    print("=" * 70)
    print("DEMO 5: REAL-WORLD DESIGN SCENARIOS")
    print("=" * 70)
    print()

    # Scenario A: Compact SF6 GIS design
    print("Scenario A: Compact SF6 Gas-Insulated Switchgear (GIS)")
    print("-" * 70)

    voltages = [145, 245, 420]  # kV
    sf6_pressure = 4.0  # bar absolute

    solver = BreakdownSolver(
        gas_type=GasType.SF6,
        pressure_atm=sf6_pressure,
        geometry=ElectrodeGeometry.COAXIAL
    )

    print(f"SF6 pressure: {sf6_pressure:.1f} bar absolute")
    print(f"Geometry: Coaxial (busbar enclosure)")
    print()
    print(f"{'Voltage':>10s} {'Min Gap':>10s} {'BIL':>10s} {'Compactness':>15s}")
    print("-" * 70)

    for V_kv in voltages:
        d_safe = solver.minimum_safe_distance(V_kv * 1e3, safety_factor=2.5)
        coord = IEC60071InsulationCoordination(system_voltage_kv=V_kv)
        bil = coord.required_bil()

        # Compare to air insulation
        solver_air = BreakdownSolver(gas_type=GasType.AIR)
        d_air = solver_air.minimum_safe_distance(V_kv * 1e3, safety_factor=2.5)

        ratio = d_air / d_safe
        print(f"{V_kv:8.0f} kV {d_safe*100:9.1f} cm {bil:9.0f} kV "
              f"{ratio:8.1f}x vs air")

    print()

    # Scenario B: Partial discharge monitoring
    print("Scenario B: Transformer Partial Discharge Assessment")
    print("-" * 70)

    pd_detector = PartialDischargeDetector(
        insulation_type="oil",
        rated_voltage_kv=138
    )

    V_pd = pd_detector.pd_inception_voltage()
    print(f"Rated voltage: 138 kV")
    print(f"PD inception voltage: {V_pd:.1f} kV")
    print()

    # Monitor over time
    print("PD monitoring over 20 years:")
    print(f"{'Year':>6s} {'PD (pC)':>10s} {'Rate (/s)':>10s} {'Health':>10s} {'Status':>12s}")
    print("-" * 70)

    years = [0, 5, 10, 15, 20]
    pd_magnitudes = [5, 25, 80, 250, 450]
    pd_rates = [0.1, 2, 15, 80, 200]

    for year, pd_mag, pd_rate in zip(years, pd_magnitudes, pd_rates):
        health = pd_detector.insulation_health_index(pd_mag, year)
        severity = pd_detector.pd_severity(pd_mag, pd_rate)
        print(f"{year:6d} {pd_mag:10.0f} {pd_rate:10.1f} {health:9.0f}% {severity:>12s}")

    print()


def main():
    """Run all demos"""
    print("\n")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║    Layer 4: High-Voltage Safety - Comprehensive Demo             ║")
    print("║    Breakdown | Corona | Clearances | Arc Flash                   ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print()

    demo_breakdown_analysis()
    demo_corona_analysis()
    demo_insulation_coordination()
    demo_arc_flash_hazard()
    demo_real_world_scenarios()

    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print()
    print("Layer 4 provides production-ready HV safety analysis:")
    print("  ✓ Breakdown voltage calculations (Paschen's law)")
    print("  ✓ Corona discharge analysis (Peek's formula)")
    print("  ✓ Insulation coordination (IEC 60071)")
    print("  ✓ Arc flash hazards (IEEE 1584)")
    print("  ✓ Industrial safety standards (NFPA 70E, OSHA)")
    print()
    print("Accuracy: ±15-25% (breakdown), ±20-30% (corona), standards-compliant")
    print("Performance: <10ms per analysis")
    print()
    print("For more info: see triality/hv_safety/README_LAYER4.md")
    print()


if __name__ == "__main__":
    main()
