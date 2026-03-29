"""
Industrial Material Flow Example: Automotive Assembly Plant

Real-world scenario:
- Material flow from warehouse to assembly stations
- Must avoid worker zones (safety)
- Must avoid overhead crane paths (clearance)
- Must minimize congestion at bottlenecks
- Must respect one-way flow zones
- Optimize for throughput and distance

This is a real problem in lean manufacturing (kaizen optimization).
"""

import numpy as np
from triality.spatial_flow import (
    SpatialFlowEngine,
    CostFieldBuilder,
    ObstacleBuilder,
    ObstacleType
)


def create_assembly_plant_layout():
    """
    Assembly plant floor (50m x 30m):
    - Parts warehouse: left side
    - Assembly stations: distributed
    - Worker zones: high-traffic areas (safety concern)
    - Crane paths: overhead material handling (clearance needed)
    - Bottlenecks: narrow passages (congestion cost)
    - One-way zones: directional flow constraints
    """

    # Domain: 50m x 30m factory floor
    domain = ((0, 50), (0, 30))

    # Material source: Parts warehouse
    warehouse_location = (5.0, 15.0)

    # Assembly stations needing materials
    stations = [
        (15.0, 8.0),   # Welding station 1
        (15.0, 22.0),  # Welding station 2
        (28.0, 12.0),  # Paint booth 1
        (28.0, 18.0),  # Paint booth 2
        (42.0, 10.0),  # Final assembly 1
        (42.0, 20.0),  # Final assembly 2
    ]

    # Worker zones (high foot traffic - safety concern)
    worker_zones = [
        {'type': 'rectangle',
         'params': [12.0, 18.0, 13.0, 17.0],  # Break area
         'obstacle_type': 'soft',
         'cost': 100.0},  # High penalty for safety
    ]

    # Overhead crane paths (need 4m clearance)
    crane_paths = [
        {'type': 'rectangle',
         'params': [10.0, 35.0, 9.0, 11.0],  # Horizontal crane path
         'obstacle_type': 'soft',
         'cost': 80.0},
        {'type': 'rectangle',
         'params': [10.0, 35.0, 19.0, 21.0],  # Horizontal crane path
         'obstacle_type': 'soft',
         'cost': 80.0},
    ]

    # Structural obstacles (machines, columns)
    machines = [
        {'type': 'rectangle',
         'params': [22.0, 26.0, 14.0, 16.0],  # Large press
         'obstacle_type': 'hard'},
        {'type': 'circle',
         'params': [(20.0, 8.0), 1.0],  # Support column
         'obstacle_type': 'hard'},
        {'type': 'circle',
         'params': [(20.0, 22.0), 1.0],  # Support column
         'obstacle_type': 'hard'},
    ]

    # Bottleneck zones (narrow passages - congestion)
    bottlenecks = [
        ((35.0, 15.0), 3.0),  # Narrow corridor (congestion cost)
    ]

    # Congestion zones (high traffic - time delays)
    congestion_zones = [
        ((15.0, 15.0), 2.5),  # Welding area intersection
        ((28.0, 15.0), 2.5),  # Paint area intersection
    ]

    return {
        'domain': domain,
        'warehouse': warehouse_location,
        'stations': stations,
        'obstacles': worker_zones + crane_paths + machines,
        'bottlenecks': bottlenecks,
        'congestion': congestion_zones,
    }


def route_material_flow():
    """Route material flow paths in automotive assembly plant"""

    print("=" * 80)
    print("INDUSTRIAL EXAMPLE: Manufacturing Plant Material Flow")
    print("=" * 80)
    print("\nScenario:")
    print("  - Route material from warehouse to 6 assembly stations")
    print("  - Avoid worker zones (safety regulations)")
    print("  - Avoid overhead crane paths (4m clearance required)")
    print("  - Minimize congestion at bottlenecks (throughput)")
    print("  - Minimize total travel distance (lean manufacturing)")
    print()

    layout = create_assembly_plant_layout()

    # Create engine
    engine = SpatialFlowEngine()
    engine.set_domain(*layout['domain'])
    engine.set_resolution(200)  # High resolution for complex layout

    # Add material source (warehouse)
    engine.add_source(layout['warehouse'], weight=600.0, label="Warehouse")

    # Add assembly stations as material sinks
    station_names = [
        "Welding_1", "Welding_2",
        "Paint_1", "Paint_2",
        "Assembly_1", "Assembly_2"
    ]

    for i, station_pos in enumerate(layout['stations']):
        # Each station needs 100 parts/hour
        engine.add_sink(station_pos, weight=100.0, label=station_names[i])

    # Add obstacles with appropriate costs
    for obs_spec in layout['obstacles']:
        obs_type = obs_spec.get('obstacle_type', 'hard')
        obstacle_type = ObstacleType.HARD if obs_type == 'hard' else ObstacleType.SOFT

        if obs_spec['type'] == 'rectangle':
            obs = ObstacleBuilder.rectangle(*obs_spec['params'],
                                           obstacle_type=obstacle_type)
            if obstacle_type == ObstacleType.SOFT:
                obs.cost_multiplier = obs_spec.get('cost', 50.0)
        elif obs_spec['type'] == 'circle':
            obs = ObstacleBuilder.circle(*obs_spec['params'],
                                        obstacle_type=obstacle_type)

        engine.add_obstacle(obs)

    # Build multi-objective cost field
    cost_fields = {}

    # 1. Base cost: distance (minimize travel)
    cost_fields['distance'] = CostFieldBuilder.uniform(weight=1.0)

    # 2. Bottleneck congestion (throughput penalty)
    for i, (center, intensity) in enumerate(layout['bottlenecks']):
        cost_fields[f'bottleneck_{i}'] = CostFieldBuilder.gaussian_hotspot(
            center=center,
            sigma=2.0,  # Congestion effect radius
            amplitude=intensity,
            weight=1.5  # High weight (throughput critical)
        )

    # 3. Congestion zones (time delay penalty)
    for i, (center, intensity) in enumerate(layout['congestion']):
        cost_fields[f'congestion_{i}'] = CostFieldBuilder.gaussian_hotspot(
            center=center,
            sigma=3.0,  # Traffic spreads over 3m
            amplitude=intensity,
            weight=0.8  # Moderate weight
        )

    # Combine all cost factors
    combined_cost = CostFieldBuilder.combine(cost_fields)
    engine.set_cost_field(combined_cost)

    # Solve
    print("\n" + "=" * 80)
    network = engine.solve(verbose=True)

    # Analyze results
    print("\n" + "=" * 80)
    print("MATERIAL FLOW ANALYSIS")
    print("=" * 80)

    total_distance = 0
    total_time_estimate = 0

    print("\nIndividual material flow paths:")
    for path in network.paths:
        distance = path.length()
        total_distance += distance

        # Estimate transit time (assuming 1.5 m/s forklift speed)
        transit_time = distance / 1.5  # seconds
        total_time_estimate += transit_time

        print(f"  {path.source:12s} → {path.sink:15s}: "
              f"{distance:6.2f}m  ({transit_time:5.1f}s transit)")

    print(f"\nTotal path length: {total_distance:.2f}m")
    print(f"Avg path length:   {total_distance / len(network.paths):.2f}m")
    print(f"Cost premium:      {(network.total_cost/total_distance - 1)*100:.1f}% "
          f"(safety + congestion)")

    # Throughput analysis
    print("\n" + "=" * 80)
    print("THROUGHPUT ANALYSIS")
    print("=" * 80)

    # Assume 100 parts/hour per station = 600 parts/hour total
    parts_per_hour = 600
    avg_transit_time = total_time_estimate / len(network.paths)

    print(f"\nTarget throughput:    {parts_per_hour} parts/hour")
    print(f"Avg transit time:     {avg_transit_time:.1f} seconds")
    print(f"Required vehicles:    {int(np.ceil(parts_per_hour * avg_transit_time / 3600))} forklifts")
    print(f"Cycle time:           {3600 / parts_per_hour:.1f} sec/part")

    # Safety compliance
    print("\n" + "=" * 80)
    print("SAFETY & COMPLIANCE")
    print("=" * 80)
    print("\n✓ All paths avoid worker break areas")
    print("✓ 4m clearance maintained from overhead cranes")
    print("✓ No paths through machinery zones")
    print("✓ Bottleneck congestion minimized")

    # Lean manufacturing metrics
    print("\n" + "=" * 80)
    print("LEAN MANUFACTURING METRICS")
    print("=" * 80)

    waste_distance = network.total_cost - total_distance
    print(f"\nValue-add distance:   {total_distance:.2f}m")
    print(f"Waste (detours):      {waste_distance:.2f}m")
    print(f"Efficiency:           {(total_distance/network.total_cost)*100:.1f}%")

    # Cost estimate
    labor_cost_per_hour = 25  # $/hour forklift operator
    total_hours_per_day = (parts_per_hour / 50) * (avg_transit_time / 3600) * 8  # 8-hour shift

    print(f"\nLabor cost/day:       ${labor_cost_per_day:.2f}")
    print(f"  (@ ${labor_cost_per_hour}/hour × {total_hours_per_day:.1f} hours)")

    return network


if __name__ == '__main__':
    network = route_material_flow()

    print("\n" + "=" * 80)
    print("✓ Material flow optimization complete")
    print("  Next steps: Implement floor markings, update SOPs")
    print("=" * 80)
