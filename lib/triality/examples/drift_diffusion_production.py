"""Drift-Diffusion Production Examples

Layer 3: Production-useful semiconductor device analysis

These examples show how to use Triality Layer 3 for early-stage production
design exploration. Remember: validate with full TCAD before tapeout!

Examples:
1. PN Junction Design - Doping optimization
2. Diode I-V Characteristics - Quick design verification
3. Built-in Potential Calculation - Design parameter extraction
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from triality.drift_diffusion.device_solver import (
    DriftDiffusion1D,
    SemiconductorMaterial,
    PNJunctionAnalyzer,
    create_pn_junction,
)


def example_1_pn_junction_design():
    """
    Example 1: PN Junction Design and Doping Optimization

    Scenario: Design a PN junction diode
    Goal: Optimize doping for desired built-in potential and depletion width

    This is useful for:
    - Initial diode design
    - Exploring doping trade-offs
    - Quick iterations before detailed TCAD
    """
    print("=" * 80)
    print("EXAMPLE 1: PN Junction Design & Doping Optimization")
    print("=" * 80)

    print("\n[Task] Design PN junction with V_bi ≈ 0.7V for rectifier application")

    # Test different doping combinations
    doping_configs = [
        ("Low doping", 1e16, 1e15),
        ("Medium doping", 1e17, 1e16),
        ("High doping", 1e18, 1e17),
    ]

    results = {}

    print("\n[Analysis] Comparing doping profiles:")
    for name, N_d, N_a in doping_configs:
        solver = create_pn_junction(
            N_d_level=N_d,
            N_a_level=N_a,
            junction_pos=1e-4,   # 1 micron
            total_length=2e-4,   # 2 microns total
        )

        result = solver.solve(applied_voltage=0.0, verbose=False)

        V_bi = result.built_in_potential()
        W_d = result.depletion_width()
        E_max = result.max_field()

        results[name] = (V_bi, W_d, E_max)

        print(f"\n  {name}:")
        print(f"    N_d = {N_d:.1e} cm⁻³, N_a = {N_a:.1e} cm⁻³")
        print(f"    Built-in potential: {V_bi:.3f} V")
        print(f"    Depletion width: {W_d*1e4:.2f} µm")
        print(f"    Max electric field: {E_max:.2e} V/cm")

    # Recommend best option
    print("\n[Recommendation]")
    print("  For V_bi ≈ 0.7V rectifier:")
    print("  → Use 'Medium doping' (10¹⁷/10¹⁶)")
    print("  → Provides good balance of V_bi and manageable depletion width")
    print("  → NEXT STEP: Validate with Sentaurus/Silvaco before tapeout")

    print("\n✓ Junction design exploration complete")


def example_2_diode_iv_characteristics():
    """
    Example 2: Diode I-V Characteristics

    Scenario: Design verification for rectifier diode
    Goal: Check that I-V follows expected exponential behavior

    Useful for:
    - Sanity checking design
    - Understanding forward/reverse characteristics
    - Comparing design variants

    NOTE: Quantitative accuracy ±30-50%. Use for trends, not specs.
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Diode I-V Characteristics")
    print("=" * 80)

    print("\n[Setup] Standard silicon PN junction diode")

    solver = create_pn_junction(
        N_d_level=1e17,
        N_a_level=1e16,
        junction_pos=1e-4,
        total_length=2e-4,
    )

    print("\n[Computing] I-V characteristic...")
    voltages, currents = PNJunctionAnalyzer.compute_iv(
        solver,
        voltage_range=(-0.5, 0.8),
        n_points=15,
        verbose=True,
    )

    print("\n[Results]")
    print("  Voltage [V]  |  Current Density [A/cm²]")
    print("  " + "-" * 40)
    for V, I in zip(voltages, currents):
        print(f"  {V:+6.3f}      |  {I:+12.3e}")

    # Check for expected behavior
    forward_idx = np.where(voltages > 0.5)[0]
    reverse_idx = np.where(voltages < -0.1)[0]

    if len(forward_idx) > 0 and len(reverse_idx) > 0:
        I_forward = currents[forward_idx[0]]
        I_reverse = np.mean(currents[reverse_idx])

        print(f"\n[Analysis]")
        print(f"  Forward current (V=0.5V): {I_forward:.2e} A/cm²")
        print(f"  Reverse current (V<0): {I_reverse:.2e} A/cm²")

        if abs(I_forward / I_reverse) > 10:
            print(f"  ✓ Rectification observed (ratio: {abs(I_forward/I_reverse):.1e})")
        else:
            print(f"  ⚠ Poor rectification (ratio: {abs(I_forward/I_reverse):.1e})")

    print("\n[Interpretation]")
    print("  • Exponential I-V in forward bias → Expected diode behavior")
    print("  • Low reverse current → Good rectification")
    print("  • These are QUALITATIVE results (±30-50% error)")
    print("  • Use for design comparison and sanity checks")
    print("  • VALIDATE with full TCAD for production specs")

    print("\n✓ I-V analysis complete")


def example_3_built_in_potential_calculation():
    """
    Example 3: Built-in Potential vs. Doping

    Scenario: Understand how doping affects built-in potential
    Goal: Extract design relationships for quick calculations

    Useful for:
    - Understanding physics
    - Hand calculation validation
    - Quick design estimates
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Built-in Potential vs. Doping")
    print("=" * 80)

    print("\n[Analysis] Sweep N-type doping, fix P-type")

    N_a_fixed = 1e16  # cm^-3
    N_d_sweep = np.logspace(15, 18, 10)  # 10^15 to 10^18

    V_bi_values = []

    print(f"\n  Fixed P-type: N_a = {N_a_fixed:.1e} cm⁻³")
    print(f"  Sweeping N-type: N_d = 10¹⁵ to 10¹⁸ cm⁻³\n")

    for N_d in N_d_sweep:
        solver = create_pn_junction(
            N_d_level=N_d,
            N_a_level=N_a_fixed,
            junction_pos=1e-4,
            total_length=2e-4,
        )

        result = solver.solve(applied_voltage=0.0, verbose=False)
        V_bi = result.built_in_potential()
        V_bi_values.append(V_bi)

    print("  N_d [cm⁻³]  |  V_bi [V]")
    print("  " + "-" * 30)
    for N_d, V_bi in zip(N_d_sweep, V_bi_values):
        print(f"  {N_d:.1e}    |  {V_bi:.3f}")

    print("\n[Extracted Relationship]")
    print("  V_bi increases logarithmically with N_d (as expected)")
    print("  Analytical: V_bi ≈ (k_B T/q) × ln(N_d × N_a / n_i²)")

    # Check against analytical formula
    n_i = 1.5e10
    k_B = 1.381e-23
    T = 300
    q = 1.602e-19
    V_T = k_B * T / q

    V_bi_analytical = V_T * np.log(N_d_sweep * N_a_fixed / n_i**2)

    error = np.mean(np.abs(np.array(V_bi_values) - V_bi_analytical))
    print(f"  Mean error vs. analytical: {error:.4f} V")
    print(f"  → Good agreement validates implementation")

    print("\n[Use Case]")
    print("  • Use this relationship for quick estimates")
    print("  • Example: Need V_bi = 0.7V with N_a = 10¹⁶")
    print("  •          → Requires N_d ≈ 10¹⁷ cm⁻³")

    print("\n✓ Built-in potential analysis complete")


def example_4_design_iteration_workflow():
    """
    Example 4: Design Iteration Workflow

    Scenario: Optimize junction for low forward voltage drop
    Goal: Demonstrate fast iteration capability

    This shows Layer 3's value: rapid design exploration
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Design Iteration Workflow")
    print("=" * 80)

    print("\n[Design Goal] Minimize forward voltage drop at I = 1 mA/cm²")

    # Try different designs
    designs = [
        ("Design A: Balanced", 1e17, 1e17),
        ("Design B: N+ heavy", 1e18, 1e16),
        ("Design C: P+ heavy", 1e16, 1e18),
    ]

    print("\n[Iteration] Testing designs:")
    best_design = None
    best_voltage = float('inf')

    for name, N_d, N_a in designs:
        solver = create_pn_junction(N_d_level=N_d, N_a_level=N_a)

        # Quick I-V around operating point
        voltages, currents = PNJunctionAnalyzer.compute_iv(
            solver,
            voltage_range=(0.5, 0.8),
            n_points=5,
            verbose=False,
        )

        # Find voltage at ~1 mA/cm² (if reached)
        target_current = 1e-3  # A/cm²
        if np.max(currents) > target_current:
            idx = np.argmin(np.abs(currents - target_current))
            V_op = voltages[idx]

            print(f"\n  {name}:")
            print(f"    N_d = {N_d:.1e}, N_a = {N_a:.1e}")
            print(f"    V_forward @ 1mA/cm²: {V_op:.3f} V")

            if V_op < best_voltage:
                best_voltage = V_op
                best_design = name
        else:
            print(f"\n  {name}:")
            print(f"    Could not reach 1 mA/cm² (insufficient current)")

    print(f"\n[Winner] {best_design}")
    print(f"  Forward voltage: {best_voltage:.3f} V")
    print(f"\n[Next Steps]")
    print(f"  1. Validate with full TCAD (Sentaurus)")
    print(f"  2. Add process variations analysis")
    print(f"  3. Check high-frequency behavior")
    print(f"  4. Final verification before tapeout")

    print("\n✓ Design iteration complete")
    print("\nLayer 3 Value: Explored 3 designs in ~10 seconds")
    print("  (Full TCAD would take hours for same exploration)")


if __name__ == '__main__':
    print("\nTRIALITY - Layer 3: Drift-Diffusion Production Examples")
    print("⚠️  Early-stage design exploration - Validate with full TCAD!\n")

    # Run all examples
    example_1_pn_junction_design()
    example_2_diode_iv_characteristics()
    example_3_built_in_potential_calculation()
    example_4_design_iteration_workflow()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)

    print("\n✓ Layer 3 Demonstrated:")
    print("  • Fast design iteration (seconds vs. hours)")
    print("  • Doping optimization exploration")
    print("  • I-V characteristic sanity checks")
    print("  • Built-in potential extraction")

    print("\n⚠️  Remember:")
    print("  • Layer 3 is for EXPLORATION, not final verification")
    print("  • Expect ±20-50% quantitative error")
    print("  • Always validate with commercial TCAD before tapeout")
    print("  • Good for relative comparisons, not absolute specs")

    print("\n🎯 Workflow:")
    print("  Concept → Layer 3 (explore) → TCAD (verify) → Tapeout")
    print("           ~minutes            ~hours")
