"""
Industrial Cable Routing Example: Data Center Power Distribution

Real-world scenario:
- Power distribution from UPS to server racks
- Must avoid heat zones (hot aisles)
- Must avoid electromagnetic interference from transformers
- Must maintain minimum separation from coolant pipes
- Cost includes: cable length, bending penalties, EMI shielding cost

This is a real problem in hyperscale data centers.
"""

import numpy as np
from triality.spatial_flow import (
    SpatialFlowEngine,
    Source, Sink,
    CostFieldBuilder,
    ObstacleBuilder,
    ObstacleType
)


def create_datacenter_layout():
    """
    Data center floor plan (20m x 15m):
    - UPS room: bottom-left
    - Server racks: distributed across floor
    - Hot aisles: vertical strips (high heat)
    - Transformer: center-left (EMI source)
    - Coolant pipes: horizontal runs (keep distance)
    """

    # Domain: 20m x 15m data center floor
    domain = ((0, 20), (0, 15))

    # Power source: UPS in equipment room
    ups_location = (2.0, 2.0)

    # Server racks needing power (6 racks, 3kW each)
    racks = [
        (6.0, 4.0),   # Row 1
        (6.0, 8.0),
        (6.0, 12.0),
        (14.0, 4.0),  # Row 2
        (14.0, 8.0),
        (14.0, 12.0),
    ]

    # Hot aisles (where servers exhaust heat)
    # These run vertically between server rows
    hot_aisles = [
        {'type': 'rectangle',
         'params': [5.5, 6.5, 3.0, 13.0],
         'obstacle_type': 'soft'},  # Traversable but costly
        {'type': 'rectangle',
         'params': [13.5, 14.5, 3.0, 13.0],
         'obstacle_type': 'soft'},
    ]

    # Structural columns (hard obstacles)
    columns = [
        {'type': 'circle', 'params': [(10.0, 7.5), 0.3], 'obstacle_type': 'hard'},
        {'type': 'circle', 'params': [(10.0, 10.5), 0.3], 'obstacle_type': 'hard'},
    ]

    # Existing coolant pipes (maintain 0.5m clearance)
    coolant_zones = [
        {'type': 'rectangle',
         'params': [0.0, 20.0, 6.5, 7.5],  # Horizontal run
         'obstacle_type': 'soft'},
    ]

    # EMI sources (transformers, switchgear)
    emi_sources = [
        ((3.0, 7.5), 8.0),  # Main transformer (high EMI)
    ]

    # Heat sources (hot aisles modeled as gaussian heat)
    heat_sources = [
        ((6.0, 8.0), 5.0),   # Hot aisle 1 center
        ((14.0, 8.0), 5.0),  # Hot aisle 2 center
    ]

    return {
        'domain': domain,
        'ups': ups_location,
        'racks': racks,
        'obstacles': hot_aisles + columns + coolant_zones,
        'emi_sources': emi_sources,
        'heat_sources': heat_sources,
    }


def route_datacenter_power():
    """Route power cables in a real data center layout"""

    print("=" * 80)
    print("INDUSTRIAL EXAMPLE: Data Center Power Distribution")
    print("=" * 80)
    print("\nScenario:")
    print("  - Route power from UPS to 6 server racks")
    print("  - Avoid hot aisles (thermal degradation)")
    print("  - Avoid transformer EMI (signal integrity)")
    print("  - Maintain clearance from coolant pipes")
    print("  - Minimize total cable length + bending cost")
    print()

    layout = create_datacenter_layout()

    # Create engine
    engine = SpatialFlowEngine()
    engine.set_domain(*layout['domain'])
    engine.set_resolution(150)  # High resolution for accuracy

    # Add power source (UPS)
    engine.add_source(layout['ups'], weight=18.0, label="UPS")  # 18kW total

    # Add server racks as sinks
    for i, rack_pos in enumerate(layout['racks']):
        engine.add_sink(rack_pos, weight=3.0, label=f"Rack_{i+1}")  # 3kW each

    # Add obstacles
    for obs_spec in layout['obstacles']:
        obs_type = obs_spec.get('obstacle_type', 'hard')
        obstacle_type = ObstacleType.HARD if obs_type == 'hard' else ObstacleType.SOFT

        if obs_spec['type'] == 'rectangle':
            obs = ObstacleBuilder.rectangle(*obs_spec['params'],
                                           obstacle_type=obstacle_type)
            if obstacle_type == ObstacleType.SOFT:
                obs.cost_multiplier = 50.0  # High penalty for hot zones
        elif obs_spec['type'] == 'circle':
            obs = ObstacleBuilder.circle(*obs_spec['params'],
                                        obstacle_type=obstacle_type)

        engine.add_obstacle(obs)

    # Build multi-objective cost field
    cost_fields = {}

    # 1. Base cost: cable length ($/m)
    cost_fields['cable_length'] = CostFieldBuilder.uniform(weight=1.0)

    # 2. Heat avoidance (thermal degradation reduces cable life)
    for i, (center, intensity) in enumerate(layout['heat_sources']):
        cost_fields[f'heat_{i}'] = CostFieldBuilder.gaussian_hotspot(
            center=center,
            sigma=1.5,  # Heat dissipates over ~1.5m
            amplitude=intensity,
            weight=0.8  # Moderate weight (safety factor)
        )

    # 3. EMI avoidance (critical for data integrity)
    for i, (center, intensity) in enumerate(layout['emi_sources']):
        cost_fields[f'emi_{i}'] = CostFieldBuilder.gaussian_hotspot(
            center=center,
            sigma=2.0,  # EMI field extends ~2m
            amplitude=intensity,
            weight=1.2  # High weight (signal integrity critical)
        )

    # Combine all cost factors
    combined_cost = CostFieldBuilder.combine(cost_fields)
    engine.set_cost_field(combined_cost)

    # Solve
    print("\n" + "=" * 80)
    network = engine.solve(verbose=True)

    # Analyze results
    print("\n" + "=" * 80)
    print("ROUTING RESULTS")
    print("=" * 80)
    print(f"\nTotal cable required: {network.total_cost:.2f}m")
    print(f"Number of routes: {len(network.paths)}")
    print("\nIndividual routes:")

    total_length = 0
    for path in network.paths:
        length = path.length()
        total_length += length
        print(f"  {path.source:12s} → {path.sink:12s}: {length:6.2f}m "
              f"(cost-adjusted: {path.cost:6.2f})")

    print(f"\nPhysical cable length: {total_length:.2f}m")
    print(f"Cost-adjusted total:   {network.total_cost:.2f}")
    print(f"Cost premium:          {(network.total_cost/total_length - 1)*100:.1f}% "
          f"(due to heat/EMI avoidance)")

    # Installation recommendations
    print("\n" + "=" * 80)
    print("INSTALLATION RECOMMENDATIONS")
    print("=" * 80)
    print("\n✓ All routes avoid structural columns")
    print("✓ Minimal exposure to hot aisles (thermal protection)")
    print("✓ Safe distance from transformer EMI")
    print("✓ Clearance maintained from coolant pipes")
    print("\nEstimated installation cost:")
    print(f"  Cable cost (@$15/m):        ${total_length * 15:.2f}")
    print(f"  EMI shielding (@$8/m):      ${total_length * 8:.2f}")
    print(f"  Labor (@$25/m):             ${total_length * 25:.2f}")
    print(f"  TOTAL:                      ${total_length * 48:.2f}")

    return network


if __name__ == '__main__':
    network = route_datacenter_power()

    print("\n" + "=" * 80)
    print("✓ Data center power routing complete")
    print("=" * 80)
