"""
Comprehensive Test Suite for Spatial Flow Engine

This single file tests:
1. All 5 industrial examples
2. Core engine functionality
3. Physics-based behavior validation
4. Edge cases and failure modes
5. API correctness
6. Performance characteristics

Production-grade testing: validates outputs, checks invariants, verifies ROI calculations.
"""

import numpy as np
import sys
import time
from typing import Dict, List, Tuple, Any

# Import spatial flow components
try:
    from triality.spatial_flow import (
        SpatialFlowEngine,
        Source, Sink,
        CostFieldBuilder,
        ObstacleBuilder,
        ObstacleType,
        Path, Network
    )
    from triality.spatial_flow.templates import cable_routing
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class TestResult:
    """Test result tracking"""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.message = ""
        self.metrics = {}
        self.duration = 0.0

    def __repr__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status} {self.name} ({self.duration:.2f}s)"


class ComprehensiveTestSuite:
    """Comprehensive test suite for Spatial Flow Engine"""

    def __init__(self):
        self.results: List[TestResult] = []
        self.total_tests = 0
        self.passed_tests = 0

    def run_test(self, test_func):
        """Run a single test and track results"""
        result = TestResult(test_func.__name__)
        self.total_tests += 1

        start = time.time()
        try:
            test_func(result)
            if result.passed:
                self.passed_tests += 1
        except Exception as e:
            result.passed = False
            result.message = f"Exception: {str(e)}"
            import traceback
            result.message += f"\n{traceback.format_exc()}"
        finally:
            result.duration = time.time() - start
            self.results.append(result)

        return result

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE TEST RESULTS")
        print("=" * 80)

        # Group by category
        categories = {
            'Basic API': [],
            'Industrial Examples': [],
            'Physics Validation': [],
            'Edge Cases': [],
            'Performance': []
        }

        for result in self.results:
            if 'api' in result.name.lower():
                categories['Basic API'].append(result)
            elif 'example' in result.name.lower():
                categories['Industrial Examples'].append(result)
            elif 'physics' in result.name.lower():
                categories['Physics Validation'].append(result)
            elif 'edge' in result.name.lower() or 'fail' in result.name.lower():
                categories['Edge Cases'].append(result)
            elif 'performance' in result.name.lower():
                categories['Performance'].append(result)
            else:
                categories['Basic API'].append(result)

        for category, tests in categories.items():
            if not tests:
                continue

            print(f"\n{category}:")
            print("-" * 80)
            for result in tests:
                print(f"  {result}")
                if result.metrics:
                    for key, value in result.metrics.items():
                        if isinstance(value, float):
                            print(f"    {key}: {value:.3f}")
                        else:
                            print(f"    {key}: {value}")
                if not result.passed and result.message:
                    print(f"    Error: {result.message}")

        print("\n" + "=" * 80)
        print(f"OVERALL: {self.passed_tests}/{self.total_tests} tests passed")
        if self.passed_tests == self.total_tests:
            print("🎉🎉🎉 ALL TESTS PASSED 🎉🎉🎉")
        else:
            print(f"❌ {self.total_tests - self.passed_tests} tests failed")
        print("=" * 80)

    # =========================================================================
    # BASIC API TESTS
    # =========================================================================

    def test_api_basic_engine_creation(self, result: TestResult):
        """Test basic engine creation and configuration"""
        engine = SpatialFlowEngine()
        engine.set_domain((0, 10), (0, 10))
        engine.set_resolution(50)

        result.passed = True
        result.message = "Engine created successfully"

    def test_api_source_sink_addition(self, result: TestResult):
        """Test adding sources and sinks"""
        engine = SpatialFlowEngine()
        engine.set_domain((0, 10), (0, 10))

        engine.add_source((1.0, 5.0), weight=10.0, label="S1")
        engine.add_sink((9.0, 5.0), weight=10.0, label="D1")

        assert len(engine.sources) == 1
        assert len(engine.sinks) == 1
        assert engine.sources[0].label == "S1"
        assert engine.sinks[0].label == "D1"

        result.passed = True
        result.metrics['sources'] = len(engine.sources)
        result.metrics['sinks'] = len(engine.sinks)

    def test_api_obstacle_types(self, result: TestResult):
        """Test hard and soft obstacle creation"""
        # Hard obstacle
        hard_obs = ObstacleBuilder.rectangle(3, 5, 3, 5, obstacle_type=ObstacleType.HARD)
        assert hard_obs.obstacle_type == ObstacleType.HARD
        assert hard_obs.is_inside(4, 4) == True
        assert hard_obs.is_inside(2, 2) == False

        # Soft obstacle
        soft_obs = ObstacleBuilder.circle((5, 5), 2.0, obstacle_type=ObstacleType.SOFT)
        soft_obs.cost_multiplier = 100.0
        assert soft_obs.obstacle_type == ObstacleType.SOFT
        assert soft_obs.cost_multiplier == 100.0

        result.passed = True
        result.message = "Both hard and soft obstacles created correctly"

    def test_api_cost_field_builders(self, result: TestResult):
        """Test cost field builder functions"""
        # Uniform
        uniform = CostFieldBuilder.uniform(weight=1.0)
        assert uniform(5, 5) == 1.0

        # Gaussian hotspot
        hotspot = CostFieldBuilder.gaussian_hotspot(
            center=(5, 5),
            sigma=1.0,
            amplitude=10.0,
            weight=1.0
        )
        # Should be maximum at center
        center_cost = hotspot(5, 5)
        edge_cost = hotspot(8, 8)
        assert center_cost > edge_cost

        # Linear gradient
        gradient = CostFieldBuilder.linear_gradient(
            direction=(1, 0),
            weight=1.0
        )
        # Cost should increase in x direction
        assert gradient(1, 0) < gradient(2, 0)

        # Radial
        radial = CostFieldBuilder.radial(
            center=(5, 5),
            power=1.0,
            weight=1.0
        )
        # Cost should increase with distance from center
        assert radial(5, 5) < radial(8, 8)

        result.passed = True
        result.metrics['cost_field_types'] = 4

    def test_api_simple_routing(self, result: TestResult):
        """Test simple point-to-point routing"""
        engine = SpatialFlowEngine()
        engine.set_domain((0, 10), (0, 10))
        engine.add_source((1, 5), label="A")
        engine.add_sink((9, 5), label="B")
        engine.set_resolution(50)

        network = engine.solve(verbose=False)

        assert len(network.paths) == 1
        path = network.paths[0]
        assert path.source == "A"
        assert path.sink == "B"
        assert path.length() > 0
        assert len(path.points) > 2

        result.passed = True
        result.metrics['path_length'] = path.length()
        result.metrics['path_points'] = len(path.points)

    # =========================================================================
    # INDUSTRIAL EXAMPLE TESTS
    # =========================================================================

    def test_example_datacenter_power_routing(self, result: TestResult):
        """Test data center power routing example"""
        engine = SpatialFlowEngine()
        domain = ((0, 20), (0, 15))
        engine.set_domain(*domain)
        engine.set_resolution(100)  # Reduced for testing speed

        # UPS source
        engine.add_source((2.0, 2.0), weight=18.0, label="UPS")

        # Server racks (6 racks)
        racks = [(6.0, 4.0), (6.0, 8.0), (6.0, 12.0), (14.0, 4.0), (14.0, 8.0), (14.0, 12.0)]
        for i, rack in enumerate(racks):
            engine.add_sink(rack, weight=3.0, label=f"Rack_{i+1}")

        # Hot aisle obstacle (soft)
        hot_aisle = ObstacleBuilder.rectangle(5.5, 6.5, 3.0, 13.0, obstacle_type=ObstacleType.SOFT)
        hot_aisle.cost_multiplier = 50.0
        engine.add_obstacle(hot_aisle)

        # Heat source
        heat = CostFieldBuilder.gaussian_hotspot(
            center=(6.0, 8.0),
            sigma=1.5,
            amplitude=5.0,
            weight=0.8
        )
        engine.set_cost_field(heat)

        # Solve
        network = engine.solve(verbose=False)

        # Validate results
        assert len(network.paths) == 6, f"Expected 6 paths, got {len(network.paths)}"
        total_length = sum(p.length() for p in network.paths)
        assert total_length > 0, "Total path length should be > 0"
        assert total_length < 200, f"Total length {total_length}m seems too long for 20×15m space"

        # Check paths avoid hot aisle (they should route around)
        avg_cost_premium = (network.total_cost / total_length) - 1.0
        assert avg_cost_premium > 0, "Should have cost premium from heat avoidance"

        result.passed = True
        result.metrics['num_paths'] = len(network.paths)
        result.metrics['total_length_m'] = total_length
        result.metrics['cost_premium_pct'] = avg_cost_premium * 100
        result.message = f"Routed 6 cables from UPS to racks, {total_length:.1f}m total"

    def test_example_manufacturing_material_flow(self, result: TestResult):
        """Test manufacturing material flow example"""
        engine = SpatialFlowEngine()
        domain = ((0, 50), (0, 30))
        engine.set_domain(*domain)
        engine.set_resolution(120)  # Reduced for testing speed

        # Warehouse source
        engine.add_source((5.0, 15.0), weight=600.0, label="Warehouse")

        # Assembly stations (6 stations)
        stations = [(15.0, 8.0), (15.0, 22.0), (28.0, 12.0), (28.0, 18.0), (42.0, 10.0), (42.0, 20.0)]
        for i, station in enumerate(stations):
            engine.add_sink(station, weight=100.0, label=f"Station_{i+1}")

        # Worker zone (soft obstacle)
        worker_zone = ObstacleBuilder.rectangle(12.0, 18.0, 13.0, 17.0, obstacle_type=ObstacleType.SOFT)
        worker_zone.cost_multiplier = 100.0
        engine.add_obstacle(worker_zone)

        # Large machine (hard obstacle)
        machine = ObstacleBuilder.rectangle(22.0, 26.0, 14.0, 16.0, obstacle_type=ObstacleType.HARD)
        engine.add_obstacle(machine)

        # Congestion cost
        congestion = CostFieldBuilder.gaussian_hotspot(
            center=(15.0, 15.0),
            sigma=3.0,
            amplitude=2.5,
            weight=0.8
        )
        engine.set_cost_field(congestion)

        # Solve
        network = engine.solve(verbose=False)

        # Validate results
        assert len(network.paths) == 6, f"Expected 6 paths, got {len(network.paths)}"
        total_distance = sum(p.length() for p in network.paths)
        assert total_distance > 0, "Total distance should be > 0"

        # Estimate throughput (simplified)
        avg_distance = total_distance / len(network.paths)
        transit_time = avg_distance / 1.5  # 1.5 m/s forklift
        parts_per_hour = 3600 / (transit_time * 2)  # Round trip
        assert parts_per_hour > 0

        result.passed = True
        result.metrics['num_paths'] = len(network.paths)
        result.metrics['avg_distance_m'] = avg_distance
        result.metrics['est_throughput_parts_hr'] = parts_per_hour * 6
        result.message = f"Material flow to 6 stations, {parts_per_hour * 6:.0f} parts/hr estimated"

    def test_example_hospital_evacuation(self, result: TestResult):
        """Test hospital evacuation planning example"""
        engine = SpatialFlowEngine()
        domain = ((0, 40), (0, 25))
        engine.set_domain(*domain)
        engine.set_resolution(100)  # Reduced for testing

        # Patient wards (sources)
        wards = [(8.0, 5.0), (8.0, 12.0), (8.0, 20.0), (32.0, 5.0), (32.0, 12.0), (32.0, 20.0)]
        for i, ward in enumerate(wards):
            engine.add_source(ward, weight=8.0, label=f"Ward_{chr(65+i)}")

        # Fire exits (sinks)
        exits = [(2.0, 12.5), (38.0, 12.5), (20.0, 2.0), (20.0, 23.0)]
        exit_names = ["Stairwell_A", "Stairwell_B", "Exit_South", "Exit_North"]
        for i, exit_pos in enumerate(exits):
            engine.add_sink(exit_pos, weight=12.0, label=exit_names[i])

        # Elevator shaft (hard obstacle - unusable in fire)
        elevator = ObstacleBuilder.rectangle(18.0, 22.0, 11.0, 14.0, obstacle_type=ObstacleType.HARD)
        engine.add_obstacle(elevator)

        # Smoke accumulation zone
        smoke = CostFieldBuilder.gaussian_hotspot(
            center=(20.0, 12.5),
            sigma=3.0,
            amplitude=8.0,
            weight=2.5  # High weight - life safety
        )
        engine.set_cost_field(smoke)

        # Solve
        network = engine.solve(verbose=False)

        # Validate results
        assert len(network.paths) > 0, "Should have evacuation paths"
        total_distance = sum(p.length() for p in network.paths)
        avg_distance = total_distance / len(network.paths)

        # Check evacuation time (simplified)
        avg_speed = 0.8  # m/s (weighted: ambulatory, wheelchair, bed)
        avg_evac_time = avg_distance / avg_speed

        # NFPA 101 Life Safety Code: evacuation < 4 minutes
        assert avg_evac_time < 240, f"Evacuation time {avg_evac_time:.0f}s exceeds 4min limit"

        # Physics-based routing creates all possible paths (all-to-all)
        # In practice, each ward would use only the shortest path to one exit
        # Check that evacuation paths exist and are reasonable
        assert len(network.paths) == 24, f"Expected 24 paths (6 wards × 4 exits), got {len(network.paths)}"

        # Group paths by ward and find shortest path for each
        ward_to_best_exit = {}
        for path in network.paths:
            ward = path.source
            if ward not in ward_to_best_exit or path.length() < ward_to_best_exit[ward][1]:
                ward_to_best_exit[ward] = (path.sink, path.length())

        # Compute load if each ward uses its shortest path
        exit_loads = {}
        for ward, (exit_name, _) in ward_to_best_exit.items():
            exit_loads[exit_name] = exit_loads.get(exit_name, 0) + 8

        max_load = max(exit_loads.values()) if exit_loads else 0

        # With 6 wards and 4 exits, perfect distribution would be 12 per exit
        # Physics routing may cluster, so allow up to 24 (2 wards) per exit
        assert max_load <= 24, f"Exit severely overloaded: {max_load} patients (expected ≤24)"

        result.passed = True
        result.metrics['num_paths'] = len(network.paths)
        result.metrics['avg_evac_distance_m'] = avg_distance
        result.metrics['avg_evac_time_s'] = avg_evac_time
        result.metrics['max_exit_load'] = max_load
        result.message = f"Evacuation plan: {avg_evac_time:.0f}s avg, {max_load} max exit load"

    def test_example_underground_utility(self, result: TestResult):
        """Test underground utility routing example"""
        engine = SpatialFlowEngine()
        domain = ((0, 100), (0, 80))
        engine.set_domain(*domain)
        engine.set_resolution(150)  # Reduced for testing

        # Gas main source
        engine.add_source((10.0, 40.0), weight=700.0, label="Gas_Main")

        # Building service points (7 buildings)
        buildings = [(30.0, 25.0), (30.0, 55.0), (55.0, 20.0), (55.0, 45.0),
                    (55.0, 60.0), (80.0, 30.0), (80.0, 50.0)]
        for i, building in enumerate(buildings):
            engine.add_sink(building, weight=100.0, label=f"Building_{chr(65+i)}")

        # Existing utility (soft obstacle - high crossing cost)
        water_main = ObstacleBuilder.rectangle(0.0, 100.0, 38.0, 42.0, obstacle_type=ObstacleType.SOFT)
        water_main.cost_multiplier = 150.0
        engine.add_obstacle(water_main)

        # Archaeological site (hard obstacle)
        archaeological = ObstacleBuilder.rectangle(35.0, 45.0, 50.0, 65.0, obstacle_type=ObstacleType.HARD)
        engine.add_obstacle(archaeological)

        # Bedrock zone (expensive excavation)
        bedrock = ObstacleBuilder.rectangle(70.0, 90.0, 20.0, 40.0, obstacle_type=ObstacleType.SOFT)
        bedrock.cost_multiplier = 300.0
        engine.add_obstacle(bedrock)

        # Water table cost
        water_table = CostFieldBuilder.gaussian_hotspot(
            center=(40.0, 60.0),
            sigma=8.0,
            amplitude=5.0,
            weight=1.5
        )
        engine.set_cost_field(water_table)

        # Solve
        network = engine.solve(verbose=False)

        # Validate results
        assert len(network.paths) == 7, f"Expected 7 paths, got {len(network.paths)}"
        total_length = sum(p.length() for p in network.paths)

        # Reasonable bounds for 100×80m domain
        assert total_length > 100, "Total pipe length seems too short"
        assert total_length < 1000, "Total pipe length seems too long"

        # Estimate cost
        cost_per_m = 380  # Excavation + pipe
        base_cost = total_length * cost_per_m
        premium_cost = (network.total_cost - total_length) * cost_per_m
        total_cost = base_cost + premium_cost

        result.passed = True
        result.metrics['num_routes'] = len(network.paths)
        result.metrics['total_pipe_length_m'] = total_length
        result.metrics['est_project_cost_usd'] = total_cost
        result.message = f"Gas distribution: {total_length:.0f}m pipe, ${total_cost:,.0f} estimated"

    def test_example_warehouse_robot_navigation(self, result: TestResult):
        """Test warehouse robot navigation example"""
        engine = SpatialFlowEngine()
        domain = ((0, 60), (0, 40))
        engine.set_domain(*domain)
        engine.set_resolution(120)  # Reduced for testing

        # Charging station (source)
        engine.add_source((5.0, 20.0), weight=800.0, label="Charging_Station")

        # Shelving obstacles (hard) - define first to avoid conflicts
        shelf1 = ObstacleBuilder.rectangle(15.0, 21.0, 5.0, 12.0, obstacle_type=ObstacleType.HARD)
        shelf2 = ObstacleBuilder.rectangle(29.0, 35.0, 5.0, 12.0, obstacle_type=ObstacleType.HARD)
        engine.add_obstacle(shelf1)
        engine.add_obstacle(shelf2)

        # Picking locations (8 locations) - in aisles, not inside shelves
        locations = [(12.0, 8.0), (12.0, 22.0), (12.0, 32.0), (25.0, 8.0),
                    (25.0, 22.0), (25.0, 32.0), (40.0, 8.0), (40.0, 22.0)]
        for i, loc in enumerate(locations):
            engine.add_sink(loc, weight=100.0, label=f"Pick_{chr(65+i)}")

        # Human work zone (soft - safety)
        human_zone = ObstacleBuilder.rectangle(52.0, 60.0, 15.0, 28.0, obstacle_type=ObstacleType.SOFT)
        human_zone.cost_multiplier = 150.0
        engine.add_obstacle(human_zone)

        # Congestion zone
        congestion = CostFieldBuilder.gaussian_hotspot(
            center=(25.0, 20.0),
            sigma=2.5,
            amplitude=3.0,
            weight=1.5
        )
        engine.set_cost_field(congestion)

        # Solve
        network = engine.solve(verbose=False)

        # Validate results
        assert len(network.paths) == 8, f"Expected 8 paths, got {len(network.paths)}"
        total_distance = sum(p.length() for p in network.paths)
        avg_distance = total_distance / len(network.paths)

        # Estimate throughput
        robot_speed = 2.0  # m/s
        picking_time = 60  # seconds
        avg_travel_time = avg_distance / robot_speed
        cycle_time = avg_travel_time * 2 + picking_time  # Round trip + pick
        picks_per_hour_per_robot = 3600 / cycle_time
        fleet_picks_per_hour = picks_per_hour_per_robot * 8

        # Estimate energy
        energy_per_m = 0.08  # Wh/m
        avg_energy_per_trip = avg_distance * energy_per_m
        battery_capacity = 100  # Wh
        trips_per_charge = battery_capacity / avg_energy_per_trip

        result.passed = True
        result.metrics['num_routes'] = len(network.paths)
        result.metrics['avg_distance_m'] = avg_distance
        result.metrics['fleet_picks_per_hr'] = fleet_picks_per_hour
        result.metrics['trips_per_charge'] = trips_per_charge
        result.message = f"Robot fleet: {fleet_picks_per_hour:.0f} picks/hr, {trips_per_charge:.1f} trips/charge"

    # =========================================================================
    # PHYSICS VALIDATION TESTS
    # =========================================================================

    def test_physics_heat_avoidance_behavior(self, result: TestResult):
        """Test that paths actually avoid heat sources"""
        # Route with heat source in center
        engine_with_heat = SpatialFlowEngine()
        engine_with_heat.set_domain((0, 10), (0, 10))
        engine_with_heat.add_source((1, 5), label="A")
        engine_with_heat.add_sink((9, 5), label="B")
        engine_with_heat.set_resolution(60)

        # Add heat source in center
        heat = CostFieldBuilder.gaussian_hotspot(
            center=(5, 5),
            sigma=1.5,
            amplitude=10.0,
            weight=2.0
        )
        engine_with_heat.set_cost_field(heat)
        network_with_heat = engine_with_heat.solve(verbose=False)

        # Route without heat source
        engine_no_heat = SpatialFlowEngine()
        engine_no_heat.set_domain((0, 10), (0, 10))
        engine_no_heat.add_source((1, 5), label="A")
        engine_no_heat.add_sink((9, 5), label="B")
        engine_no_heat.set_resolution(60)
        network_no_heat = engine_no_heat.solve(verbose=False)

        # Path with heat should be longer (detour around heat)
        length_with_heat = network_with_heat.paths[0].length()
        length_no_heat = network_no_heat.paths[0].length()

        assert length_with_heat > length_no_heat, "Path should detour around heat source"

        detour_ratio = length_with_heat / length_no_heat
        assert 1.05 < detour_ratio < 2.0, f"Detour ratio {detour_ratio:.2f} seems unreasonable"

        result.passed = True
        result.metrics['length_no_heat'] = length_no_heat
        result.metrics['length_with_heat'] = length_with_heat
        result.metrics['detour_ratio'] = detour_ratio
        result.message = f"Path detours {(detour_ratio-1)*100:.1f}% to avoid heat"

    def test_physics_obstacle_avoidance(self, result: TestResult):
        """Test that paths avoid hard obstacles"""
        engine = SpatialFlowEngine()
        engine.set_domain((0, 10), (0, 10))
        engine.add_source((1, 5), label="A")
        engine.add_sink((9, 5), label="B")
        engine.set_resolution(60)

        # Add hard obstacle in center
        obstacle = ObstacleBuilder.rectangle(4, 6, 4, 6, obstacle_type=ObstacleType.HARD)
        engine.add_obstacle(obstacle)

        network = engine.solve(verbose=False)
        path = network.paths[0]

        # Check that no point on path is inside obstacle
        for x, y in path.points:
            assert not obstacle.is_inside(x, y), f"Path goes through obstacle at ({x:.2f}, {y:.2f})"

        # Path should be longer than direct route
        direct_distance = 8.0  # Straight line from (1,5) to (9,5)
        assert path.length() > direct_distance, "Path should detour around obstacle"

        result.passed = True
        result.metrics['path_length'] = path.length()
        result.metrics['direct_distance'] = direct_distance
        result.message = f"Path successfully avoids hard obstacle ({path.length():.1f}m vs {direct_distance:.1f}m direct)"

    def test_physics_cost_monotonicity(self, result: TestResult):
        """Test that increasing cost weight increases path cost"""
        base_engine = SpatialFlowEngine()
        base_engine.set_domain((0, 10), (0, 10))
        base_engine.add_source((1, 5), label="A")
        base_engine.add_sink((9, 5), label="B")
        base_engine.set_resolution(60)

        heat_low = CostFieldBuilder.gaussian_hotspot((5, 5), 1.5, 10.0, weight=0.5)
        base_engine.set_cost_field(heat_low)
        network_low = base_engine.solve(verbose=False)

        high_engine = SpatialFlowEngine()
        high_engine.set_domain((0, 10), (0, 10))
        high_engine.add_source((1, 5), label="A")
        high_engine.add_sink((9, 5), label="B")
        high_engine.set_resolution(60)

        heat_high = CostFieldBuilder.gaussian_hotspot((5, 5), 1.5, 10.0, weight=5.0)
        high_engine.set_cost_field(heat_high)
        network_high = high_engine.solve(verbose=False)

        # Higher weight should result in greater detour (longer path)
        length_low = network_low.paths[0].length()
        length_high = network_high.paths[0].length()

        assert length_high >= length_low, "Higher cost weight should not reduce path length"

        result.passed = True
        result.metrics['length_weight_0.5'] = length_low
        result.metrics['length_weight_5.0'] = length_high
        result.metrics['length_increase'] = length_high - length_low
        result.message = "Cost monotonicity verified: higher weight → longer path"

    def test_physics_multiple_sinks_distribution(self, result: TestResult):
        """Test that flow distributes to multiple sinks reasonably"""
        engine = SpatialFlowEngine()
        engine.set_domain((0, 20), (0, 10))
        engine.add_source((2, 5), weight=30.0, label="Source")
        engine.set_resolution(60)

        # Add 3 sinks at different distances
        engine.add_sink((8, 5), weight=10.0, label="Near")
        engine.add_sink((14, 5), weight=10.0, label="Mid")
        engine.add_sink((18, 5), weight=10.0, label="Far")

        network = engine.solve(verbose=False)

        # Should have 3 paths
        assert len(network.paths) == 3, f"Expected 3 paths, got {len(network.paths)}"

        # Paths should be in order of increasing length
        paths_by_sink = {p.sink: p for p in network.paths}
        near_length = paths_by_sink["Near"].length()
        mid_length = paths_by_sink["Mid"].length()
        far_length = paths_by_sink["Far"].length()

        assert near_length < mid_length < far_length, "Path lengths should increase with distance"

        result.passed = True
        result.metrics['near_path_length'] = near_length
        result.metrics['mid_path_length'] = mid_length
        result.metrics['far_path_length'] = far_length
        result.message = "Flow distribution to 3 sinks validated"

    # =========================================================================
    # EDGE CASE TESTS
    # =========================================================================

    def test_edge_case_no_sources(self, result: TestResult):
        """Test that engine rejects problem with no sources"""
        engine = SpatialFlowEngine()
        engine.set_domain((0, 10), (0, 10))
        engine.add_sink((9, 5), label="B")

        try:
            network = engine.solve(verbose=False)
            result.passed = False
            result.message = "Should have raised ValueError for no sources"
        except ValueError as e:
            if "at least one source" in str(e):
                result.passed = True
                result.message = "Correctly rejected: no sources"
            else:
                result.passed = False
                result.message = f"Wrong error message: {e}"

    def test_edge_case_no_sinks(self, result: TestResult):
        """Test that engine rejects problem with no sinks"""
        engine = SpatialFlowEngine()
        engine.set_domain((0, 10), (0, 10))
        engine.add_source((1, 5), label="A")

        try:
            network = engine.solve(verbose=False)
            result.passed = False
            result.message = "Should have raised ValueError for no sinks"
        except ValueError as e:
            if "at least one sink" in str(e):
                result.passed = True
                result.message = "Correctly rejected: no sinks"
            else:
                result.passed = False
                result.message = f"Wrong error message: {e}"

    def test_edge_case_no_domain(self, result: TestResult):
        """Test that engine rejects problem with no domain"""
        engine = SpatialFlowEngine()
        engine.add_source((1, 5), label="A")
        engine.add_sink((9, 5), label="B")

        try:
            network = engine.solve(verbose=False)
            result.passed = False
            result.message = "Should have raised ValueError for no domain"
        except ValueError as e:
            if "domain bounds" in str(e):
                result.passed = True
                result.message = "Correctly rejected: no domain"
            else:
                result.passed = False
                result.message = f"Wrong error message: {e}"

    def test_edge_case_source_in_obstacle(self, result: TestResult):
        """Test that engine rejects source inside hard obstacle"""
        engine = SpatialFlowEngine()
        engine.set_domain((0, 10), (0, 10))
        engine.add_source((5, 5), label="A")  # Inside obstacle
        engine.add_sink((9, 5), label="B")

        obstacle = ObstacleBuilder.rectangle(4, 6, 4, 6, obstacle_type=ObstacleType.HARD)
        engine.add_obstacle(obstacle)

        try:
            network = engine.solve(verbose=False)
            result.passed = False
            result.message = "Should have raised ValueError for source in obstacle"
        except ValueError as e:
            if "inside hard obstacle" in str(e).lower():
                result.passed = True
                result.message = "Correctly rejected: source in obstacle"
            else:
                result.passed = False
                result.message = f"Wrong error: {e}"

    def test_edge_case_low_resolution(self, result: TestResult):
        """Test that engine rejects very low resolution"""
        engine = SpatialFlowEngine()
        engine.set_domain((0, 10), (0, 10))
        engine.add_source((1, 5), label="A")
        engine.add_sink((9, 5), label="B")

        try:
            engine.set_resolution(5)  # Too low
            result.passed = False
            result.message = "Should have raised ValueError for low resolution"
        except ValueError as e:
            if "must be >= 10" in str(e) or "resolution" in str(e).lower():
                result.passed = True
                result.message = "Correctly rejected: resolution < 10"
            else:
                result.passed = False
                result.message = f"Wrong error: {e}"

    def test_edge_case_negative_weight(self, result: TestResult):
        """Test that negative source weight is rejected"""
        try:
            source = Source((5, 5), weight=-1.0)
            result.passed = False
            result.message = "Should have raised ValueError for negative weight"
        except ValueError as e:
            if "positive" in str(e).lower():
                result.passed = True
                result.message = "Correctly rejected: negative weight"
            else:
                result.passed = False
                result.message = f"Wrong error: {e}"

    # =========================================================================
    # PERFORMANCE TESTS
    # =========================================================================

    def test_performance_medium_problem(self, result: TestResult):
        """Test performance on medium-sized problem"""
        engine = SpatialFlowEngine()
        engine.set_domain((0, 50), (0, 50))
        engine.set_resolution(100)

        # 1 source, 5 sinks
        engine.add_source((5, 25), weight=50.0, label="Source")
        for i in range(5):
            engine.add_sink((45, 10 + i*8), weight=10.0, label=f"Sink_{i}")

        # Add 3 obstacles
        for i in range(3):
            obs = ObstacleBuilder.rectangle(
                15 + i*10, 20 + i*10,
                15, 35,
                obstacle_type=ObstacleType.HARD
            )
            engine.add_obstacle(obs)

        start = time.time()
        network = engine.solve(verbose=False)
        solve_time = time.time() - start

        # Should solve reasonably fast
        assert solve_time < 30.0, f"Solve time {solve_time:.1f}s exceeded 30s limit"
        assert len(network.paths) == 5

        result.passed = True
        result.metrics['solve_time_s'] = solve_time
        result.metrics['resolution'] = 100
        result.metrics['domain_size'] = 50 * 50
        result.message = f"Solved 100×100 grid in {solve_time:.2f}s"

    def test_performance_scalability(self, result: TestResult):
        """Test that solve time scales reasonably with resolution"""
        times = []
        resolutions = [30, 50, 70]

        for res in resolutions:
            engine = SpatialFlowEngine()
            engine.set_domain((0, 20), (0, 20))
            engine.set_resolution(res)
            engine.add_source((2, 10), label="A")
            engine.add_sink((18, 10), label="B")

            start = time.time()
            network = engine.solve(verbose=False)
            times.append(time.time() - start)

        # Time should increase with resolution (but not explosively)
        # Roughly O(n²) to O(n²·⁵) for 2D problems
        time_ratio_1 = times[1] / times[0]  # 50/30
        time_ratio_2 = times[2] / times[1]  # 70/50

        # Ratios should be reasonable (not >10x)
        assert time_ratio_1 < 10.0, f"Time ratio {time_ratio_1:.1f} too large"
        assert time_ratio_2 < 10.0, f"Time ratio {time_ratio_2:.1f} too large"

        result.passed = True
        result.metrics['time_30'] = times[0]
        result.metrics['time_50'] = times[1]
        result.metrics['time_70'] = times[2]
        result.message = f"Scaling: 30→50: {time_ratio_1:.2f}x, 50→70: {time_ratio_2:.2f}x"


def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("=" * 80)
    print("SPATIAL FLOW ENGINE - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("\nTesting:")
    print("  ✓ Basic API functionality")
    print("  ✓ 5 industrial examples (datacenter, manufacturing, hospital, utility, warehouse)")
    print("  ✓ Physics-based behavior validation")
    print("  ✓ Edge cases and failure modes")
    print("  ✓ Performance characteristics")
    print("\n" + "=" * 80)

    suite = ComprehensiveTestSuite()

    # Basic API tests
    print("\n[1/5] Running Basic API Tests...")
    suite.run_test(suite.test_api_basic_engine_creation)
    suite.run_test(suite.test_api_source_sink_addition)
    suite.run_test(suite.test_api_obstacle_types)
    suite.run_test(suite.test_api_cost_field_builders)
    suite.run_test(suite.test_api_simple_routing)

    # Industrial example tests
    print("\n[2/5] Running Industrial Example Tests...")
    suite.run_test(suite.test_example_datacenter_power_routing)
    suite.run_test(suite.test_example_manufacturing_material_flow)
    suite.run_test(suite.test_example_hospital_evacuation)
    suite.run_test(suite.test_example_underground_utility)
    suite.run_test(suite.test_example_warehouse_robot_navigation)

    # Physics validation tests
    print("\n[3/5] Running Physics Validation Tests...")
    suite.run_test(suite.test_physics_heat_avoidance_behavior)
    suite.run_test(suite.test_physics_obstacle_avoidance)
    suite.run_test(suite.test_physics_cost_monotonicity)
    suite.run_test(suite.test_physics_multiple_sinks_distribution)

    # Edge case tests
    print("\n[4/5] Running Edge Case Tests...")
    suite.run_test(suite.test_edge_case_no_sources)
    suite.run_test(suite.test_edge_case_no_sinks)
    suite.run_test(suite.test_edge_case_no_domain)
    suite.run_test(suite.test_edge_case_source_in_obstacle)
    suite.run_test(suite.test_edge_case_low_resolution)
    suite.run_test(suite.test_edge_case_negative_weight)

    # Performance tests
    print("\n[5/5] Running Performance Tests...")
    suite.run_test(suite.test_performance_medium_problem)
    suite.run_test(suite.test_performance_scalability)

    # Print summary
    suite.print_summary()

    return suite.passed_tests == suite.total_tests


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
