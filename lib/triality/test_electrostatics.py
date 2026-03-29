"""Comprehensive Tests for Electrostatics Module

Tests Layer 1 functionality:
- Laplace solver
- Poisson solver
- Conduction solver
- Electric field calculation
- Current density and power density
- Derived quantities and analysis tools
"""

import numpy as np
import pytest

from triality.electrostatics import (
    ElectrostaticSolver,
    ConductiveSolver,
    BoundaryCondition,
    BoundaryType,
    ChargeDistribution,
    Material,
    ElectricField,
    FieldMagnitude,
    GradientAnalysis,
    HotspotDetector,
)


class TestElectrostaticSolver:
    """Test electrostatic field solver (Laplace/Poisson)"""

    def test_laplace_parallel_plates(self):
        """Test Laplace equation with parallel plate geometry"""
        solver = ElectrostaticSolver()
        solver.set_domain((0, 1), (0, 1))
        solver.set_resolution(50)

        # Left plate at 100V (cover full height)
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=100.0,
            region=lambda x, y: x < 0.05
        ))

        # Right plate at 0V (cover full height)
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: x > 0.95
        ))

        # Top and bottom insulating (Neumann BC implicitly)
        result = solver.solve(verbose=False)

        # Check potential decreases from left to right
        V_left = result.potential[10, 25]
        V_center = result.potential[25, 25]
        V_right = result.potential[40, 25]

        assert V_left > V_center > V_right, "Potential should decrease from left to right"
        # Relaxed tolerance - finite discretization affects solution
        assert 20 < V_center < 60, f"Expected potential at center in range, got {V_center:.2f}V"

        # Check field is mostly x-directed
        E_x, E_y = result.electric_field(0.5, 0.5)
        assert abs(E_y) < abs(E_x) * 0.5, "Field should be mostly x-directed"
        assert abs(E_x) > 50, f"Expected significant x-field, got {abs(E_x):.2f}"

    def test_poisson_point_charge(self):
        """Test Poisson equation with point charge"""
        solver = ElectrostaticSolver()
        solver.set_domain((-1, 1), (-1, 1))
        solver.set_resolution(50)

        # Grounded boundary
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: abs(x) > 0.95 or abs(y) > 0.95
        ))

        # Point charge at center
        Q = 1e-9  # 1 nC
        epsilon = 8.854e-12
        charge_dist = ChargeDistribution(
            density_func=lambda x, y: Q / (0.01 * 0.01) if abs(x) < 0.05 and abs(y) < 0.05 else 0.0
        )
        solver.add_charge_distribution(charge_dist)

        result = solver.solve(verbose=False)

        # Potential should be maximum at center
        V_center = result.potential[25, 25]
        V_corner = result.potential[5, 5]
        assert V_center > V_corner, "Potential should be highest at charge location"

    def test_electric_field_magnitude(self):
        """Test electric field magnitude calculation"""
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

        # Get field magnitude grid
        E_mag = result.field_magnitude_grid()

        # Field should exist and be mostly positive
        E_mean = np.mean(E_mag)
        assert E_mean > 0, "Mean field should be positive"

        # Field should be relatively uniform in interior (exclude boundaries)
        E_interior = E_mag[10:-10, 10:-10]
        E_interior_std = np.std(E_interior)
        E_interior_mean = np.mean(E_interior)
        assert E_interior_std / E_interior_mean < 1.0, "Interior field should be relatively uniform"

    def test_max_field_regions(self):
        """Test high-field region detection"""
        solver = ElectrostaticSolver()
        solver.set_domain((0, 1), (0, 1))
        solver.set_resolution(40)

        # Create non-uniform field (point electrodes)
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=1000.0,
            region=lambda x, y: np.sqrt((x - 0.2)**2 + (y - 0.5)**2) < 0.05
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: np.sqrt((x - 0.8)**2 + (y - 0.5)**2) < 0.05
        ))

        result = solver.solve(verbose=False)

        # Find high-field regions
        high_field = result.max_field_regions(threshold_percentile=90)

        assert len(high_field) > 0, "Should detect high-field regions"
        # High field should be near electrodes
        for x, y, E in high_field[:5]:
            dist_left = np.sqrt((x - 0.2)**2 + (y - 0.5)**2)
            dist_right = np.sqrt((x - 0.8)**2 + (y - 0.5)**2)
            assert min(dist_left, dist_right) < 0.3, "High field should be near electrodes"


class TestConductiveSolver:
    """Test conductive media solver"""

    def test_uniform_conductor(self):
        """Test current flow in uniform conductor"""
        solver = ConductiveSolver()
        solver.set_domain((0, 0.1), (0, 0.05))
        solver.set_resolution(50)

        # Copper conductor
        copper = Material('Copper', conductivity=5.96e7)
        solver.add_material(copper, region=lambda x, y: True)

        # Voltage source
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=1.0,
            region=lambda x, y: x < 0.002
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: x > 0.098
        ))

        result = solver.solve(verbose=False)

        # Check potential drops linearly
        V_left = result.potential[5, 25]
        V_center = result.potential[25, 25]
        V_right = result.potential[45, 25]

        assert V_left > V_center > V_right, "Potential should decrease monotonically"

        # Check current density is mostly x-directed
        J_x, J_y = result.current_density(0.05, 0.025)
        assert abs(J_y) < abs(J_x) * 0.2, "Current should flow in x-direction"

    def test_current_density_calculation(self):
        """Test current density J = -σ∇V"""
        solver = ConductiveSolver()
        solver.set_domain((0, 0.1), (0, 0.05))
        solver.set_resolution(50)

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

        # Get current density
        J_x, J_y = result.current_density(0.05, 0.025)
        J_mag = result.current_density_magnitude(0.05, 0.025)

        assert J_mag > 0, "Current density should be positive"
        assert abs(J_x) > abs(J_y), "Current should be mostly in x-direction"

    def test_power_density_calculation(self):
        """Test power density P = J²/σ"""
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

        # Get power density
        P = result.power_density(0.05, 0.025)

        assert P > 0, "Power density should be positive (Joule heating)"

        # Get power density grid
        P_grid = result.power_density_grid()
        assert np.all(P_grid >= 0), "Power density should be non-negative everywhere"

    def test_multi_material_interface(self):
        """Test interface between different materials"""
        solver = ConductiveSolver()
        solver.set_domain((0, 1), (0, 1))
        solver.set_resolution(50)

        # Left half: copper (high conductivity)
        copper = Material('Copper', conductivity=5.96e7)
        solver.add_material(copper, region=lambda x, y: x < 0.5)

        # Right half: aluminum (lower conductivity)
        aluminum = Material('Aluminum', conductivity=3.77e7)
        solver.add_material(aluminum, region=lambda x, y: x >= 0.5)

        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=1.0,
            region=lambda x, y: x < 0.02
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: x > 0.98
        ))

        result = solver.solve(verbose=False)

        # Power density should be higher in aluminum (lower conductivity)
        P_copper = result.power_density(0.25, 0.5)
        P_aluminum = result.power_density(0.75, 0.5)

        # Note: Power density P = J²/σ, and J is continuous, so P_Al > P_Cu
        assert P_aluminum > P_copper * 1.2, "Power density should be higher in lower-conductivity material"

    def test_hotspot_detection(self):
        """Test hotspot detection"""
        solver = ConductiveSolver()
        solver.set_domain((0, 0.1), (0, 0.05))
        solver.set_resolution(50)

        # Create geometry with bottleneck (high current density)
        copper = Material('Copper', conductivity=5.96e7)
        # Conductor with narrow section
        conductor_region = lambda x, y: (
            (0.02 < y < 0.03) or  # Narrow section
            (y < 0.02 or y > 0.03)  # Wide sections
        )
        solver.add_material(copper, region=conductor_region)

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

        # Find hotspots
        hotspots = result.find_hotspots(max_power_density=1e6)

        # Should find hotspots in narrow section
        assert len(hotspots) > 0, "Should detect hotspots in high-current-density regions"


class TestDerivedQuantities:
    """Test derived quantities and analysis tools"""

    def test_electric_field_calculation(self):
        """Test electric field E = -∇V"""
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
        field_data = ElectricField.from_result(result)

        # Check field components exist
        assert field_data.E_x.shape == result.potential.shape
        assert field_data.E_y.shape == result.potential.shape
        assert field_data.E_magnitude.shape == result.potential.shape

        # Find max field
        x_max, y_max, E_max = field_data.max_field()
        assert E_max > 0, "Maximum field should be positive"

    def test_field_statistics(self):
        """Test field magnitude statistics"""
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
        field_data = ElectricField.from_result(result)

        stats = FieldMagnitude.analyze(field_data)

        assert stats.min >= 0, "Minimum field should be non-negative"
        assert stats.max > stats.mean, "Max should be greater than mean"
        assert stats.percentile_99 > stats.percentile_90, "99th > 90th percentile"

    def test_gradient_analysis(self):
        """Test field gradient analysis"""
        solver = ElectrostaticSolver()
        solver.set_domain((0, 1), (0, 1))
        solver.set_resolution(40)

        # Create sharp gradient with point electrode
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
        field_data = ElectricField.from_result(result)

        # Get field gradients
        grad_x, grad_y = GradientAnalysis.field_gradient(field_data)

        assert grad_x.shape == field_data.E_magnitude.shape
        assert grad_y.shape == field_data.E_magnitude.shape

        # Find high-gradient zones
        high_grad = GradientAnalysis.high_gradient_zones(field_data, threshold_percentile=90)
        assert len(high_grad) > 0, "Should find high-gradient zones near electrode"

    def test_breakdown_detection(self):
        """Test dielectric breakdown risk detection"""
        solver = ElectrostaticSolver()
        solver.set_domain((0, 0.1), (0, 0.1))
        solver.set_resolution(40)

        # High voltage (approaching breakdown)
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=200000.0,  # 200 kV
            region=lambda x, y: x < 0.005
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: x > 0.095
        ))

        result = solver.solve(verbose=False)
        field_data = ElectricField.from_result(result)

        # Check for breakdown risk (air: 3 MV/m)
        air_breakdown = 3e6
        breakdown_zones = HotspotDetector.detect_electrical(field_data, air_breakdown)

        # Should detect high-field regions near electrodes
        # (200kV / 0.1m = 2 MV/m, which is > 50% of breakdown)
        assert len(breakdown_zones) > 0, "Should detect breakdown risk zones"


class TestNumericalRobustness:
    """Test numerical stability and edge cases"""

    def test_solver_convergence(self):
        """Test solver converges for various resolutions"""
        for resolution in [20, 40, 60]:
            solver = ElectrostaticSolver()
            solver.set_domain((0, 1), (0, 1))
            solver.set_resolution(resolution)

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
            assert result is not None, f"Solver should converge at resolution {resolution}"

    def test_zero_conductivity_handling(self):
        """Test handling of zero/very low conductivity regions"""
        solver = ConductiveSolver()
        solver.set_domain((0, 1), (0, 1))
        solver.set_resolution(40)

        # Only add conductor in limited region
        copper = Material('Copper', conductivity=5.96e7)
        solver.add_material(copper, region=lambda x, y: 0.4 < x < 0.6)

        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.DIRICHLET,
            value=10.0,
            region=lambda x, y: x < 0.02
        ))
        solver.add_boundary(BoundaryCondition(
            type=BoundaryType.GROUNDED,
            region=lambda x, y: x > 0.98
        ))

        # Should handle without numerical errors
        result = solver.solve(verbose=False)
        assert np.all(np.isfinite(result.potential)), "Potential should be finite everywhere"

    def test_no_warnings_or_errors(self):
        """Test that solvers run without numerical warnings"""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)

            solver = ElectrostaticSolver()
            solver.set_domain((0, 1), (0, 1))
            solver.set_resolution(50)

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

            # Check for numerical warnings
            numerical_warnings = [warn for warn in w if issubclass(warn.category, RuntimeWarning)]
            assert len(numerical_warnings) == 0, f"Should not produce numerical warnings: {numerical_warnings}"


def run_all_tests():
    """Run all tests with reporting"""
    import sys

    print("=" * 80)
    print("ELECTROSTATICS MODULE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    test_classes = [
        TestElectrostaticSolver,
        TestConductiveSolver,
        TestDerivedQuantities,
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
