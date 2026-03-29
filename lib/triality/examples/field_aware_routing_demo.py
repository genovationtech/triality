"""Field-Aware Routing Demonstrations

Layer 2: Physics-informed intelligent routing

Demonstrates the killer differentiator: using electromagnetic analysis
to create smarter, safer routes that traditional tools cannot achieve.

Examples:
1. EMI-aware PCB routing (avoid high-field zones)
2. High-current busbar routing (thermal management)
3. Ground return optimization
4. Multi-trace crosstalk mitigation
"""

import numpy as np

from triality.electrostatics import (
    ElectrostaticSolver,
    ConductiveSolver,
    BoundaryCondition,
    BoundaryType,
    Material,
)

from triality.field_aware_routing import (
    ElectricFieldCostBuilder,
    CurrentDensityCostBuilder,
    EMICostBuilder,
    ThermalRiskCostBuilder,
    ClearanceCostBuilder,
    PhysicsAwareRouter,
    OptimizationObjective,
    MultiRouteOptimizer,
    ReturnPathAnalyzer,
)


def example_1_emi_aware_pcb_routing():
    """
    Example 1: EMI-Aware PCB Routing

    Scenario: Route sensitive signal trace on PCB with nearby high-voltage traces
    Goal: Minimize EMI coupling while maintaining reasonable path length
    """
    print("=" * 80)
    print("EXAMPLE 1: EMI-Aware PCB Routing")
    print("=" * 80)

    # Step 1: Analyze electromagnetic environment
    print("\n[Step 1] Analyzing electromagnetic environment...")

    solver = ElectrostaticSolver()
    solver.set_domain((0, 0.05), (0, 0.03))  # 5cm × 3cm PCB
    solver.set_resolution(60)

    # High-voltage trace at top (5V differential)
    solver.add_boundary(BoundaryCondition(
        type=BoundaryType.DIRICHLET,
        value=5.0,
        region=lambda x, y: 0.01 < x < 0.04 and 0.025 < y < 0.028
    ))

    # Ground plane at bottom
    solver.add_boundary(BoundaryCondition(
        type=BoundaryType.GROUNDED,
        region=lambda x, y: y < 0.002
    ))

    result = solver.solve(verbose=False)
    print("  ✓ Electric field analysis complete")

    # Step 2: Build physics-aware cost fields
    print("\n[Step 2] Building physics-aware cost fields...")

    # EMI risk from field gradients
    emi_cost = EMICostBuilder.from_result(
        result,
        scaling='exponential',
        base_cost=1.0
    )
    print(f"  ✓ EMI cost field created: {emi_cost.description}")

    # Clearance from high-field zones
    field_cost = ElectricFieldCostBuilder.from_result(
        result,
        scaling='quadratic',
        base_cost=1.0
    )
    print(f"  ✓ Field clearance cost created: {field_cost.description}")

    # Step 3: Route with physics awareness
    print("\n[Step 3] Computing physics-aware routes...")

    router = PhysicsAwareRouter()
    router.set_domain((0, 0.05), (0, 0.03))
    router.set_resolution(60)

    router.add_physics_cost('EMI', emi_cost, weight=2.0)
    router.add_physics_cost('Field', field_cost, weight=1.0)

    # Route A: Without physics (baseline)
    print("\n  Route A: Baseline (no physics awareness)")
    router_baseline = PhysicsAwareRouter()
    router_baseline.set_domain((0, 0.05), (0, 0.03))
    router_baseline.set_resolution(60)

    route_baseline = router_baseline.route(
        start=(0.005, 0.015),
        end=(0.045, 0.015),
        objective=OptimizationObjective.MIN_LENGTH,
        verbose=False
    )
    print(f"    Length: {route_baseline.length*1000:.1f} mm")
    print(f"    Cost: {route_baseline.cost:.2f}")

    # Route B: With EMI awareness
    print("\n  Route B: EMI-aware")
    route_emi_aware = router.route(
        start=(0.005, 0.015),
        end=(0.045, 0.015),
        objective=OptimizationObjective.MIN_EMI,
        verbose=False
    )
    print(f"    Length: {route_emi_aware.length*1000:.1f} mm")
    print(f"    Cost: {route_emi_aware.cost:.2f}")
    print(f"    EMI score: {route_emi_aware.emi_score:.3f} (lower is better)")

    # Compare
    print("\n  --- Comparison ---")
    length_increase = (route_emi_aware.length / route_baseline.length - 1) * 100
    emi_improvement = ((1 - route_emi_aware.emi_score / (route_baseline.emi_score or 1)) * 100)

    print(f"  Length increase: {length_increase:+.1f}%")
    if route_baseline.emi_score and route_emi_aware.emi_score:
        print(f"  EMI reduction: {emi_improvement:+.1f}%")

    print("\n✓ EMI-aware routing complete")
    print("  Benefit: Reduced EMI coupling with minimal length penalty")


def example_2_high_current_thermal_routing():
    """
    Example 2: High-Current Trace Thermal Management

    Scenario: Route power trace avoiding hotspot zones
    Goal: Minimize thermal risk while delivering current
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: High-Current Thermal-Aware Routing")
    print("=" * 80)

    # Step 1: Analyze thermal environment
    print("\n[Step 1] Analyzing thermal/current density environment...")

    solver = ConductiveSolver()
    solver.set_domain((0, 0.1), (0, 0.05))  # 10cm × 5cm board
    solver.set_resolution(80)

    # Copper pour with narrow section (bottleneck)
    copper = Material('Copper', conductivity=5.96e7)

    # Create geometry with thermal hotspot
    def copper_region(x, y):
        # Wide sections
        if y < 0.02 or y > 0.03:
            return 0.01 < x < 0.09
        # Narrow section (hotspot)
        else:
            return 0.04 < x < 0.06

    solver.add_material(copper, region=copper_region)

    # Voltage differential
    solver.add_boundary(BoundaryCondition(
        type=BoundaryType.DIRICHLET,
        value=12.0,
        region=lambda x, y: x < 0.002
    ))
    solver.add_boundary(BoundaryCondition(
        type=BoundaryType.GROUNDED,
        region=lambda x, y: x > 0.098
    ))

    result = solver.solve(verbose=False)
    print("  ✓ Current density analysis complete")

    # Find hotspots
    hotspots = result.find_hotspots(max_power_density=1e7)
    print(f"  ✓ Identified {len(hotspots)} thermal hotspots")

    # Step 2: Build thermal-aware cost fields
    print("\n[Step 2] Building thermal-aware cost fields...")

    thermal_cost = ThermalRiskCostBuilder.from_result(
        result,
        threshold=1e6,
        scaling='exponential',
        base_cost=1.0
    )
    print(f"  ✓ Thermal cost field created: {thermal_cost.description}")

    current_cost = CurrentDensityCostBuilder.from_result(
        result,
        scaling='quadratic',
        base_cost=1.0
    )
    print(f"  ✓ Current density cost created: {current_cost.description}")

    # Step 3: Route with thermal awareness
    print("\n[Step 3] Computing thermal-aware route...")

    router = PhysicsAwareRouter()
    router.set_domain((0, 0.1), (0, 0.05))
    router.set_resolution(80)

    router.add_physics_cost('Thermal', thermal_cost, weight=3.0)
    router.add_physics_cost('Current', current_cost, weight=2.0)

    route = router.route(
        start=(0.01, 0.025),
        end=(0.09, 0.025),
        objective=OptimizationObjective.MIN_THERMAL,
        verbose=False
    )

    print(f"    Route length: {route.length*1000:.1f} mm")
    print(f"    Thermal score: {route.thermal_score:.3f} (lower is better)")

    print("\n✓ Thermal-aware routing complete")
    print("  Benefit: Power trace avoids hotspot zones, reducing reliability risk")


def example_3_ground_return_optimization():
    """
    Example 3: Ground Return Path Optimization

    Scenario: Optimize signal routing for best return path quality
    Goal: Minimize ground impedance and voltage drop
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Ground Return Path Optimization")
    print("=" * 80)

    # Step 1: Analyze ground plane
    print("\n[Step 1] Analyzing ground plane impedance...")

    solver = ConductiveSolver()
    solver.set_domain((0, 0.08), (0, 0.06))  # 8cm × 6cm
    solver.set_resolution(60)

    # Ground plane with split/gap (creates higher impedance path)
    copper = Material('Copper', conductivity=5.96e7)

    def ground_plane(x, y):
        # Split at x=0.04 (creates two sections)
        if x < 0.038 or x > 0.042:
            return True
        # Small bridge at top
        elif y > 0.055:
            return True
        return False

    solver.add_material(copper, region=ground_plane)

    # Current injection
    solver.add_boundary(BoundaryCondition(
        type=BoundaryType.DIRICHLET,
        value=5.0,
        region=lambda x, y: np.sqrt((x - 0.02)**2 + (y - 0.03)**2) < 0.003
    ))
    solver.add_boundary(BoundaryCondition(
        type=BoundaryType.GROUNDED,
        region=lambda x, y: np.sqrt((x - 0.06)**2 + (y - 0.03)**2) < 0.003
    ))

    result = solver.solve(verbose=False)
    print("  ✓ Ground impedance analysis complete")

    # Step 2: Evaluate return paths
    print("\n[Step 2] Evaluating return path quality...")

    # Path A: Straight across gap
    path_a = [(0.02, 0.03), (0.06, 0.03)]
    metrics_a = ReturnPathAnalyzer.evaluate_return_path(path_a, result, current=1.0)

    # Path B: Around gap (through bridge)
    path_b = [(0.02, 0.03), (0.02, 0.057), (0.06, 0.057), (0.06, 0.03)]
    metrics_b = ReturnPathAnalyzer.evaluate_return_path(path_b, result, current=1.0)

    print("\n  Path A (straight across gap):")
    print(f"    Voltage drop: {metrics_a['voltage_drop']:.3f} V")
    print(f"    Return impedance: {metrics_a['return_impedance']:.3f} Ω")
    print(f"    Quality score: {metrics_a['quality_score']:.3f}")

    print("\n  Path B (around gap via bridge):")
    print(f"    Voltage drop: {metrics_b['voltage_drop']:.3f} V")
    print(f"    Return impedance: {metrics_b['return_impedance']:.3f} Ω")
    print(f"    Quality score: {metrics_b['quality_score']:.3f}")

    # Compare
    if metrics_b['quality_score'] > metrics_a['quality_score']:
        print("\n  ✓ Path B has better return path quality")
        improvement = (metrics_b['quality_score'] / metrics_a['quality_score'] - 1) * 100
        print(f"    Improvement: {improvement:+.1f}%")
    else:
        print("\n  ✓ Path A has better return path quality")

    print("\n✓ Ground return optimization complete")
    print("  Benefit: Lower ground impedance improves signal integrity")


def example_4_multi_trace_crosstalk_mitigation():
    """
    Example 4: Multi-Trace Crosstalk Mitigation

    Scenario: Route multiple signal traces with crosstalk awareness
    Goal: Maintain adequate separation while minimizing total length
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Multi-Trace Crosstalk Mitigation")
    print("=" * 80)

    # Step 1: Set up electromagnetic environment
    print("\n[Step 1] Setting up routing environment...")

    # Simple domain without pre-existing fields
    # Crosstalk will be managed through separation constraints

    # Step 2: Route multiple traces with awareness
    print("\n[Step 2] Routing multiple traces with crosstalk awareness...")

    optimizer = MultiRouteOptimizer()
    optimizer.router.set_domain((0, 0.05), (0, 0.03))
    optimizer.router.set_resolution(60)

    # Define 3 parallel signal routes
    route_pairs = [
        ((0.005, 0.010), (0.045, 0.010)),  # Signal 1
        ((0.005, 0.015), (0.045, 0.015)),  # Signal 2
        ((0.005, 0.020), (0.045, 0.020)),  # Signal 3
    ]

    min_separation = 0.003  # 3mm minimum

    routes = optimizer.route_multiple(
        route_pairs,
        min_separation=min_separation,
        objective=OptimizationObjective.MIN_CROSSTALK,
        verbose=False
    )

    print(f"\n  Routed {len(routes)} traces:")
    for i, route in enumerate(routes):
        print(f"    Trace {i+1}: {route.length*1000:.1f} mm, clearance score: {route.clearance_score:.3f}")

    # Verify separations
    print("\n  Verifying separations...")
    from triality.field_aware_routing.coupling_analysis import CrosstalkAnalyzer

    for i in range(len(routes)):
        for j in range(i+1, len(routes)):
            metrics = CrosstalkAnalyzer.evaluate_crosstalk(
                routes[i].path,
                routes[j].path,
                min_separation=min_separation
            )
            print(f"    Trace {i+1} ↔ Trace {j+1}: min sep = {metrics['min_separation']*1000:.2f} mm, " +
                  f"crosstalk risk = {metrics['crosstalk_risk']:.3f}")

    print("\n✓ Multi-trace routing complete")
    print("  Benefit: Adequate separation reduces crosstalk, improves signal quality")


if __name__ == '__main__':
    print("\nTRIALITY - Field-Aware Routing Module")
    print("Layer 2: Physics-Informed Intelligent Routing\n")

    # Run all examples
    example_1_emi_aware_pcb_routing()
    example_2_high_current_thermal_routing()
    example_3_ground_return_optimization()
    example_4_multi_trace_crosstalk_mitigation()

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nLayer 2 Capabilities Demonstrated:")
    print("  ✓ EMI-aware routing (avoid high-field zones)")
    print("  ✓ Thermal-aware routing (avoid hotspots)")
    print("  ✓ Ground return optimization (minimize impedance)")
    print("  ✓ Multi-trace crosstalk mitigation")
    print("\nThis is the missing middle layer between hand calculations and")
    print("full-wave EM simulation - fast, physics-informed, actionable!")
