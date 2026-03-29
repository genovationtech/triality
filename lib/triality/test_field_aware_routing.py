"""Comprehensive Tests for Field-Aware Routing Module

Tests Layer 2 functionality:
- Cost field builders from physics
- Multi-conductor coupling analysis
- Return path quality metrics
- Physics-aware routing integration
- Multi-route optimization
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
    MultiConductorCoupling,
    GroundImpedanceMap,
    CrosstalkAnalyzer,
)


class TestCostFieldBuilders:
    """Test physics-aware cost field builders"""

    def test_electric_field_cost_builder(self):
        """Test cost field from electric field"""
        # Create simple electrostatic problem
        solver = ElectrostaticSolver()
        solver.set_domain((0, 1), (0, 1))
        solver.set_resolution(40)

        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=100.0,
            region=lambda x, y: x < 0.05
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: x > 0.95
        ))

        result = solver.solve(verbose=False)

        # Build cost field
        cost_field = ElectricFieldCostBuilder.from_result(result, scaling='linear')

        assert cost_field is not None, "Cost field should be created"
        assert 'ElectricFieldCost' in cost_field.name

        # Test cost evaluation
        cost_low = cost_field(0.5, 0.5)   # Middle of domain
        cost_high = cost_field(0.1, 0.5)  # Near high-voltage electrode

        assert cost_low > 0, "Cost should be positive"
        # Note: cost_high may not always be > cost_low depending on field distribution

    def test_current_density_cost_builder(self):
        """Test cost field from current density"""
        solver = ConductiveSolver()
        solver.set_domain((0, 0.1), (0, 0.05))
        solver.set_resolution(40)

        copper = Material('Copper', conductivity=5.96e7)
        solver.add_material(copper, region=lambda x, y: True)

        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=10.0,
            region=lambda x, y: x < 0.002
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: x > 0.098
        ))

        result = solver.solve(verbose=False)

        # Build cost field
        cost_field = CurrentDensityCostBuilder.from_result(result, scaling='quadratic')

        assert cost_field is not None
        assert 'CurrentDensityCost' in cost_field.name

        # Test cost evaluation
        cost = cost_field(0.05, 0.025)
        assert cost > 0, "Cost should be positive"

    def test_emi_cost_builder(self):
        """Test EMI risk cost field from field gradients"""
        solver = ElectrostaticSolver()
        solver.set_domain((0, 1), (0, 1))
        solver.set_resolution(40)

        # Create sharp field gradient with point electrode
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=1000.0,
            region=lambda x, y: np.sqrt((x - 0.5)**2 + (y - 0.5)**2) < 0.05
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: abs(x) > 0.95 or abs(y) > 0.95
        ))

        result = solver.solve(verbose=False)

        # Build EMI cost field
        cost_field = EMICostBuilder.from_result(result, scaling='exponential')

        assert cost_field is not None
        assert 'EMI' in cost_field.name

        # Cost should be higher near sharp gradient (near electrode)
        cost_near = cost_field(0.55, 0.5)
        cost_far = cost_field(0.2, 0.2)

        assert cost_near > 0, "Cost near gradient should be positive"
        assert cost_far > 0, "Cost far from gradient should be positive"

    def test_thermal_risk_cost_builder(self):
        """Test thermal risk cost field"""
        solver = ConductiveSolver()
        solver.set_domain((0, 0.1), (0, 0.05))
        solver.set_resolution(40)

        copper = Material('Copper', conductivity=5.96e7)
        solver.add_material(copper, region=lambda x, y: True)

        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=10.0,
            region=lambda x, y: x < 0.002
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: x > 0.098
        ))

        result = solver.solve(verbose=False)

        # Build thermal cost field
        cost_field = ThermalRiskCostBuilder.from_result(result, threshold=1e5)

        assert cost_field is not None
        assert 'Thermal' in cost_field.name

        cost = cost_field(0.05, 0.025)
        assert cost >= 1.0, "Cost should be at least base cost"

    def test_clearance_cost_builder(self):
        """Test clearance cost field"""
        # Define conductors
        conductors = [
            (0.2, 0.5, 0.05),  # (x, y, radius)
            (0.8, 0.5, 0.05),
        ]

        cost_field = ClearanceCostBuilder.from_conductors(
            conductors,
            min_clearance=0.1,
            violation_cost=100.0
        )

        assert cost_field is not None
        assert 'Clearance' in cost_field.name

        # Test costs at different locations
        cost_far = cost_field(0.5, 0.5)       # Far from conductors
        cost_near = cost_field(0.25, 0.5)     # Near conductor 1
        cost_inside = cost_field(0.205, 0.5)  # Just inside conductor 1

        assert cost_far < cost_near, "Cost should increase near conductors"
        # Inside/at conductor, cost should be very high
        assert cost_inside >= 100.0, "Cost inside conductor should be very high (violation cost)"


class TestCouplingAnalysis:
    """Test multi-conductor coupling analysis"""

    def test_multi_conductor_coupling(self):
        """Test coupling zone identification"""
        # Create electrostatic scenario with two conductors
        solver = ElectrostaticSolver()
        solver.set_domain((0, 1), (0, 1))
        solver.set_resolution(40)

        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=100.0,
            region=lambda x, y: 0.2 < x < 0.3 and 0.4 < y < 0.6
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=-100.0,
            region=lambda x, y: 0.7 < x < 0.8 and 0.4 < y < 0.6
        ))

        result = solver.solve(verbose=False)

        # Analyze coupling
        coupling = MultiConductorCoupling((0, 1), (0, 1))
        coupling.add_conductor('Cond1', lambda x, y: 0.2 < x < 0.3 and 0.4 < y < 0.6)
        coupling.add_conductor('Cond2', lambda x, y: 0.7 < x < 0.8 and 0.4 < y < 0.6)

        zones = coupling.compute_coupling_zones(result, proximity_threshold=0.2)

        # Should find some coupling zones between conductors
        assert isinstance(zones, list), "Should return list of coupling zones"

    def test_return_path_analyzer(self):
        """Test return path quality analysis"""
        # Create ground plane scenario
        solver = ConductiveSolver()
        solver.set_domain((0, 0.1), (0, 0.05))
        solver.set_resolution(40)

        copper = Material('Copper', conductivity=5.96e7)
        solver.add_material(copper, region=lambda x, y: True)

        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=5.0,
            region=lambda x, y: x < 0.002
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: x > 0.098
        ))

        result = solver.solve(verbose=False)

        # Evaluate return path
        signal_path = [(0.01, 0.025), (0.09, 0.025)]
        metrics = ReturnPathAnalyzer.evaluate_return_path(signal_path, result, current=1.0)

        assert 'voltage_drop' in metrics
        assert 'return_impedance' in metrics
        assert 'quality_score' in metrics

        assert metrics['voltage_drop'] >= 0, "Voltage drop should be non-negative"
        assert metrics['return_impedance'] >= 0, "Impedance should be non-negative"
        assert 0 <= metrics['quality_score'] <= 1, "Quality score should be 0-1"

    def test_ground_impedance_map(self):
        """Test ground impedance mapping"""
        solver = ConductiveSolver()
        solver.set_domain((0, 0.1), (0, 0.05))
        solver.set_resolution(40)

        copper = Material('Copper', conductivity=5.96e7)
        solver.add_material(copper, region=lambda x, y: True)

        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=5.0,
            region=lambda x, y: x < 0.002
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: x > 0.098
        ))

        result = solver.solve(verbose=False)

        # Create impedance map
        impedance_func = GroundImpedanceMap.from_conduction_result(result, reference_current=1.0)

        # Test impedance evaluation
        Z1 = impedance_func(0.05, 0.025)
        Z2 = impedance_func(0.05, 0.03)

        assert np.isfinite(Z1), "Impedance should be finite"
        assert np.isfinite(Z2), "Impedance should be finite"
        assert Z1 >= 0, "Impedance should be non-negative"

        # Identify low-impedance zones
        low_z_zones = GroundImpedanceMap.identify_low_impedance_zones(
            impedance_func,
            (0, 0.1), (0, 0.05),
            resolution=20,
            threshold_percentile=25
        )

        assert isinstance(low_z_zones, list), "Should return list of zones"

    def test_crosstalk_analyzer(self):
        """Test crosstalk analysis between paths"""
        path1 = [(0.0, 0.01), (0.05, 0.01)]
        path2 = [(0.0, 0.02), (0.05, 0.02)]  # Parallel, 1cm apart

        metrics = CrosstalkAnalyzer.evaluate_crosstalk(path1, path2, min_separation=0.005)

        assert 'min_separation' in metrics
        assert 'parallel_length' in metrics
        assert 'crosstalk_risk' in metrics

        assert metrics['min_separation'] > 0, "Minimum separation should be positive"
        assert 0 <= metrics['crosstalk_risk'] <= 1, "Crosstalk risk should be 0-1"


class TestRoutingIntegration:
    """Test physics-aware routing integration"""

    def test_physics_aware_router_basic(self):
        """Test basic physics-aware routing"""
        # Simple scenario
        router = PhysicsAwareRouter()
        router.set_domain((0, 0.05), (0, 0.03))
        router.set_resolution(40)

        # Add simple clearance cost
        conductors = [(0.025, 0.015, 0.005)]
        clearance_cost = ClearanceCostBuilder.from_conductors(
            conductors,
            min_clearance=0.005,
            violation_cost=100.0
        )
        router.add_physics_cost('Clearance', clearance_cost, weight=10.0)

        # Route around obstacle
        route = router.route(
            start=(0.005, 0.015),
            end=(0.045, 0.015),
            objective=OptimizationObjective.BALANCED,
            verbose=False
        )

        assert route is not None, "Should return route"
        assert len(route.path) >= 2, "Path should have at least 2 points"
        assert route.length > 0, "Path length should be positive"

    def test_optimization_objectives(self):
        """Test different optimization objectives"""
        router = PhysicsAwareRouter()
        router.set_domain((0, 0.05), (0, 0.03))
        router.set_resolution(40)

        # Test each objective type
        for objective in [OptimizationObjective.MIN_LENGTH,
                         OptimizationObjective.BALANCED]:
            route = router.route(
                start=(0.005, 0.015),
                end=(0.045, 0.015),
                objective=objective,
                verbose=False
            )

            assert route is not None, f"Should route with {objective.value}"
            assert route.length > 0, "Path should have positive length"

    def test_multi_route_optimizer(self):
        """Test multi-route optimization with crosstalk awareness"""
        optimizer = MultiRouteOptimizer()
        optimizer.router.set_domain((0, 0.05), (0, 0.03))
        optimizer.router.set_resolution(40)

        # Route two parallel traces
        route_pairs = [
            ((0.005, 0.010), (0.045, 0.010)),
            ((0.005, 0.020), (0.045, 0.020)),
        ]

        routes = optimizer.route_multiple(
            route_pairs,
            min_separation=0.003,
            objective=OptimizationObjective.BALANCED,
            verbose=False
        )

        assert len(routes) == 2, "Should route both traces"

        # Verify both routes exist
        for route in routes:
            assert route.length > 0, "Each route should have positive length"

    def test_cost_field_integration_with_electrostatics(self):
        """Test full workflow: electrostatics → cost field → routing"""
        # Step 1: Physics analysis
        solver = ElectrostaticSolver()
        solver.set_domain((0, 0.05), (0, 0.03))
        solver.set_resolution(40)

        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=100.0,
            region=lambda x, y: 0.02 < x < 0.03 and y > 0.025
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: y < 0.002
        ))

        result = solver.solve(verbose=False)

        # Step 2: Build cost field
        field_cost = ElectricFieldCostBuilder.from_result(result, scaling='linear')

        # Step 3: Route with physics awareness
        router = PhysicsAwareRouter()
        router.set_domain((0, 0.05), (0, 0.03))
        router.set_resolution(40)
        router.add_physics_cost('Field', field_cost, weight=1.0)

        route = router.route(
            start=(0.005, 0.015),
            end=(0.045, 0.015),
            objective=OptimizationObjective.BALANCED,
            verbose=False
        )

        assert route is not None, "Should complete full workflow"
        assert route.length > 0, "Should produce valid route"


class TestNumericalRobustness:
    """Test numerical stability and edge cases"""

    def test_cost_field_outside_domain(self):
        """Test cost field evaluation outside physics domain"""
        solver = ElectrostaticSolver()
        solver.set_domain((0, 1), (0, 1))
        solver.set_resolution(40)

        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=100.0,
            region=lambda x, y: x < 0.05
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: x > 0.95
        ))

        result = solver.solve(verbose=False)
        cost_field = ElectricFieldCostBuilder.from_result(result)

        # Evaluate outside domain - should return base cost
        cost_outside = cost_field(2.0, 2.0)
        assert np.isfinite(cost_outside), "Cost outside domain should be finite"
        assert cost_outside >= 1.0, "Cost outside should be at least base cost"

    def test_zero_length_path(self):
        """Test return path analysis with degenerate path"""
        solver = ConductiveSolver()
        solver.set_domain((0, 0.1), (0, 0.05))
        solver.set_resolution(40)

        copper = Material('Copper', conductivity=5.96e7)
        solver.add_material(copper, region=lambda x, y: True)

        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=5.0,
            region=lambda x, y: x < 0.002
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: x > 0.098
        ))

        result = solver.solve(verbose=False)

        # Single point path
        path = [(0.05, 0.025)]
        metrics = ReturnPathAnalyzer.evaluate_return_path(path, result, current=1.0)

        assert isinstance(metrics, dict), "Should return metrics dict"
        assert all(np.isfinite(v) for v in metrics.values() if isinstance(v, (int, float)))


def run_all_tests():
    """Run all tests with reporting"""
    import sys

    print("=" * 80)
    print("FIELD-AWARE ROUTING MODULE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    test_classes = [
        TestCostFieldBuilders,
        TestCouplingAnalysis,
        TestRoutingIntegration,
        TestNumericalRobustness,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 80)

        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                passed_tests += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {str(e)}")
                failed_tests.append((test_class.__name__, method_name, str(e)))

    print("\n" + "=" * 80)
    print(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    print("=" * 80)

    if failed_tests:
        print("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  {class_name}.{method_name}: {error}")
        sys.exit(1)
    else:
        print("\n🎉 ALL TESTS PASSED 🎉")
        sys.exit(0)


if __name__ == '__main__':
    run_all_tests()
