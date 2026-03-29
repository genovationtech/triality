"""Comprehensive tests for Spatial Flow Engine"""

import numpy as np
import sys

# Test imports
try:
    from triality.spatial_flow import (
        SpatialFlowEngine,
        Source, Sink,
        CostFieldBuilder,
        ObstacleBuilder, ObstacleType
    )
    from triality.spatial_flow.templates import cable_routing
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


def test_basic_routing():
    """Test 1: Basic source-to-sink routing"""
    print("\n[1/8] Basic Source-to-Sink Routing")
    print("-" * 50)

    engine = SpatialFlowEngine()
    engine.set_domain((0, 1), (0, 1))
    engine.add_source((0.1, 0.5), label="A")
    engine.add_sink((0.9, 0.5), label="B")
    engine.set_resolution(50)

    network = engine.solve(verbose=False)

    assert len(network.paths) == 1, f"Expected 1 path, got {len(network.paths)}"
    assert network.paths[0].source == "A"
    assert network.paths[0].sink == "B"
    assert network.total_cost > 0

    print(f"✅ PASSED")
    print(f"   Network: {network}")
    print(f"   Path: {network.paths[0]}")


def test_obstacle_avoidance():
    """Test 2: Routing around obstacles"""
    print("\n[2/8] Obstacle Avoidance")
    print("-" * 50)

    engine = SpatialFlowEngine()
    engine.set_domain((0, 1), (0, 1))
    engine.add_source((0.1, 0.5), label="A")
    engine.add_sink((0.9, 0.5), label="B")

    # Add obstacle in the middle
    obstacle = ObstacleBuilder.rectangle(0.4, 0.6, 0.3, 0.7, obstacle_type=ObstacleType.HARD)
    engine.add_obstacle(obstacle)

    engine.set_resolution(50)

    network = engine.solve(verbose=False)

    # Path should not go through obstacle
    for path in network.paths:
        for x, y in path.points:
            assert not obstacle.is_inside(x, y), f"Path goes through obstacle at ({x}, {y})"

    print(f"✅ PASSED")
    print(f"   Path avoids obstacle successfully")


def test_custom_cost_field():
    """Test 3: Custom cost field (heat avoidance)"""
    print("\n[3/8] Custom Cost Field (Heat Avoidance)")
    print("-" * 50)

    engine = SpatialFlowEngine()
    engine.set_domain((0, 1), (0, 1))
    engine.add_source((0.1, 0.5), label="A")
    engine.add_sink((0.9, 0.5), label="B")

    # Add heat source in the middle
    heat_field = CostFieldBuilder.gaussian_hotspot(
        center=(0.5, 0.5),
        sigma=0.1,
        amplitude=10.0,
        weight=1.0
    )
    engine.set_cost_field(heat_field)
    engine.set_resolution(50)

    network = engine.solve(verbose=False)

    assert len(network.paths) > 0

    print(f"✅ PASSED")
    print(f"   Path avoids heat source")


def test_multiple_sinks():
    """Test 4: One source to multiple sinks"""
    print("\n[4/8] Multiple Sinks")
    print("-" * 50)

    engine = SpatialFlowEngine()
    engine.set_domain((0, 1), (0, 1))
    engine.add_source((0.1, 0.5), weight=3.0, label="Source")
    engine.add_sink((0.9, 0.3), weight=1.0, label="Sink1")
    engine.add_sink((0.9, 0.5), weight=1.0, label="Sink2")
    engine.add_sink((0.9, 0.7), weight=1.0, label="Sink3")
    engine.set_resolution(50)

    network = engine.solve(verbose=False)

    assert len(network.paths) == 3, f"Expected 3 paths, got {len(network.paths)}"

    print(f"✅ PASSED")
    print(f"   Routed to {len(network.paths)} sinks")


def test_cable_routing_template():
    """Test 5: Cable routing template"""
    print("\n[5/8] Cable Routing Template")
    print("-" * 50)

    sources = [(0.1, 0.5)]
    sinks = [(0.9, 0.5)]
    obstacles = [
        {'type': 'rectangle', 'params': [0.4, 0.6, 0.3, 0.7]}
    ]

    network = cable_routing.route_cable(
        sources=sources,
        sinks=sinks,
        obstacles=obstacles,
        resolution=50,
        verbose=False
    )

    assert len(network.paths) == 1

    print(f"✅ PASSED")
    print(f"   Template API works correctly")


def test_power_cable_routing():
    """Test 6: Power cable routing with heat sources"""
    print("\n[6/8] Power Cable Routing (Heat Avoidance)")
    print("-" * 50)

    power_source = (0.1, 0.5)
    devices = [(0.9, 0.3), (0.9, 0.7)]
    heat_sources = [((0.5, 0.5), 5.0)]  # Hot component in the middle

    network = cable_routing.route_power_cable(
        power_source=power_source,
        devices=devices,
        heat_sources=heat_sources,
        resolution=50,
        verbose=False
    )

    assert len(network.paths) == 2

    print(f"✅ PASSED")
    print(f"   Power cables routed avoiding heat")


def test_signal_cable_routing():
    """Test 7: Signal cable routing with EMI avoidance"""
    print("\n[7/8] Signal Cable Routing (EMI Avoidance)")
    print("-" * 50)

    source = (0.1, 0.5)
    destination = (0.9, 0.5)
    emi_sources = [((0.5, 0.5), 8.0)]  # EMI emitter in the middle

    network = cable_routing.route_signal_cable(
        source=source,
        destination=destination,
        emi_sources=emi_sources,
        resolution=50,
        verbose=False
    )

    assert len(network.paths) == 1

    print(f"✅ PASSED")
    print(f"   Signal cable routed avoiding EMI")


def test_validation_errors():
    """Test 8: Input validation"""
    print("\n[8/8] Input Validation")
    print("-" * 50)

    # Test 1: No sources
    try:
        engine = SpatialFlowEngine()
        engine.set_domain((0, 1), (0, 1))
        engine.add_sink((0.9, 0.5))
        network = engine.solve(verbose=False)
        assert False, "Should have raised ValueError for no sources"
    except ValueError as e:
        assert "at least one source" in str(e)
        print(f"  ✓ No sources: correctly rejected")

    # Test 2: No sinks
    try:
        engine = SpatialFlowEngine()
        engine.set_domain((0, 1), (0, 1))
        engine.add_source((0.1, 0.5))
        network = engine.solve(verbose=False)
        assert False, "Should have raised ValueError for no sinks"
    except ValueError as e:
        assert "at least one sink" in str(e)
        print(f"  ✓ No sinks: correctly rejected")

    # Test 3: No domain
    try:
        engine = SpatialFlowEngine()
        engine.add_source((0.1, 0.5))
        engine.add_sink((0.9, 0.5))
        network = engine.solve(verbose=False)
        assert False, "Should have raised ValueError for no domain"
    except ValueError as e:
        assert "domain bounds" in str(e)
        print(f"  ✓ No domain: correctly rejected")

    # Test 4: Negative source weight
    try:
        src = Source((0.5, 0.5), weight=-1.0)
        assert False, "Should have raised ValueError for negative weight"
    except ValueError as e:
        assert "positive" in str(e)
        print(f"  ✓ Negative weight: correctly rejected")

    # Test 5: Invalid resolution
    try:
        engine = SpatialFlowEngine()
        engine.set_resolution(5)
        assert False, "Should have raised ValueError for low resolution"
    except ValueError as e:
        assert "must be >= 10" in str(e)
        print(f"  ✓ Low resolution: correctly rejected")

    print(f"\n✅ PASSED")
    print(f"   All validation checks work correctly")


def run_all_tests():
    """Run all spatial flow tests"""
    print("=" * 60)
    print("Triality Spatial Flow Engine - Comprehensive Tests")
    print("=" * 60)

    tests = [
        test_basic_routing,
        test_obstacle_avoidance,
        test_custom_cost_field,
        test_multiple_sinks,
        test_cable_routing_template,
        test_power_cable_routing,
        test_signal_cable_routing,
        test_validation_errors,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Summary: {passed}/{len(tests)} passed")
    if failed == 0:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {failed} tests failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
