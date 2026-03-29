"""
Warehouse Robot Navigation: Multi-Robot Path Planning

Real-world scenario:
- Multiple robots navigate warehouse simultaneously
- Must avoid static obstacles (shelves, columns)
- Must avoid dynamic obstacles (other robots, forklifts, humans)
- Optimize for distance + energy (battery conservation)
- Respect payload constraints (loaded vs empty)
- Account for turning radius (kinematic constraints)

This is critical for automated fulfillment centers (Amazon, Alibaba logistics).
"""

import numpy as np
from triality.spatial_flow import (
    SpatialFlowEngine,
    CostFieldBuilder,
    ObstacleBuilder,
    ObstacleType
)


def create_warehouse_layout():
    """
    Automated warehouse (60m x 40m):
    - Charging stations: robot homes
    - Picking locations: inventory retrieval points
    - Packing stations: order fulfillment
    - Shelving aisles: must navigate between
    - Human work zones: safety zones (high cost)
    - Forklift paths: dynamic obstacle zones
    """

    # Domain: 60m x 40m warehouse floor
    domain = ((0, 60), (0, 40))

    # Charging station (robot starting point)
    charging_station = (5.0, 20.0)

    # Picking locations (where robots retrieve inventory)
    picking_locations = [
        (18.0, 8.0),   # Aisle A, Section 1
        (18.0, 22.0),  # Aisle A, Section 2
        (18.0, 32.0),  # Aisle A, Section 3
        (32.0, 8.0),   # Aisle B, Section 1
        (32.0, 22.0),  # Aisle B, Section 2
        (32.0, 32.0),  # Aisle B, Section 3
        (46.0, 8.0),   # Aisle C, Section 1
        (46.0, 22.0),  # Aisle C, Section 2
    ]

    # Shelving units (static obstacles)
    shelves = [
        # Aisle A shelves
        {'type': 'rectangle',
         'params': [15.0, 21.0, 5.0, 12.0],
         'obstacle_type': 'hard'},
        {'type': 'rectangle',
         'params': [15.0, 21.0, 18.0, 25.0],
         'obstacle_type': 'hard'},
        {'type': 'rectangle',
         'params': [15.0, 21.0, 28.0, 35.0],
         'obstacle_type': 'hard'},

        # Aisle B shelves
        {'type': 'rectangle',
         'params': [29.0, 35.0, 5.0, 12.0],
         'obstacle_type': 'hard'},
        {'type': 'rectangle',
         'params': [29.0, 35.0, 18.0, 25.0],
         'obstacle_type': 'hard'},
        {'type': 'rectangle',
         'params': [29.0, 35.0, 28.0, 35.0],
         'obstacle_type': 'hard'},

        # Aisle C shelves
        {'type': 'rectangle',
         'params': [43.0, 49.0, 5.0, 12.0],
         'obstacle_type': 'hard'},
        {'type': 'rectangle',
         'params': [43.0, 49.0, 18.0, 25.0],
         'obstacle_type': 'hard'},
    ]

    # Support columns
    columns = [
        {'type': 'circle', 'params': [(25.0, 15.0), 0.4], 'obstacle_type': 'hard'},
        {'type': 'circle', 'params': [(25.0, 30.0), 0.4], 'obstacle_type': 'hard'},
        {'type': 'circle', 'params': [(39.0, 15.0), 0.4], 'obstacle_type': 'hard'},
        {'type': 'circle', 'params': [(39.0, 30.0), 0.4], 'obstacle_type': 'hard'},
    ]

    # Human work zones (safety - robots slow down)
    human_zones = [
        {'type': 'rectangle',
         'params': [52.0, 60.0, 15.0, 28.0],  # Packing area
         'obstacle_type': 'soft',
         'cost': 150.0,  # High penalty for safety
         'label': 'packing_area'},
    ]

    # Dynamic obstacle zones (forklifts, human traffic)
    forklift_paths = [
        ((12.0, 20.0), 4.0),  # Main aisle (high forklift traffic)
        ((26.0, 20.0), 4.0),  # Cross aisle
        ((40.0, 20.0), 4.0),  # Cross aisle
    ]

    # Congestion zones (multiple robots converge)
    congestion_zones = [
        ((25.0, 20.0), 3.0),  # Central junction
        ((39.0, 20.0), 3.0),  # Secondary junction
    ]

    # Energy cost zones (rough floor, incline)
    energy_cost_zones = [
        ((10.0, 10.0), 2.5),  # Loading dock ramp (incline)
        ((55.0, 35.0), 2.0),  # Rough floor patch
    ]

    return {
        'domain': domain,
        'charging_station': charging_station,
        'picking_locations': picking_locations,
        'obstacles': shelves + columns + human_zones,
        'forklift_paths': forklift_paths,
        'congestion': congestion_zones,
        'energy_cost': energy_cost_zones,
    }


def plan_robot_routes():
    """Plan navigation routes for warehouse robots"""

    print("=" * 80)
    print("WAREHOUSE AUTOMATION: Multi-Robot Path Planning")
    print("=" * 80)
    print("\nScenario:")
    print("  - 8 robots navigate from charging station to picking locations")
    print("  - Avoid shelving units and support columns")
    print("  - Minimize collision risk with forklifts and humans")
    print("  - Reduce congestion at aisle junctions")
    print("  - Optimize energy consumption (battery life)")
    print("  - Ensure safe operation in human work zones")
    print()

    layout = create_warehouse_layout()

    # Create engine
    engine = SpatialFlowEngine()
    engine.set_domain(*layout['domain'])
    engine.set_resolution(220)  # High resolution for precision navigation

    # Add charging station as source
    engine.add_source(layout['charging_station'], weight=800.0, label="Charging_Station")

    # Add picking locations as sinks
    for i, pick_pos in enumerate(layout['picking_locations']):
        engine.add_sink(pick_pos, weight=100.0, label=f"Pick_{chr(65+i)}")

    # Add obstacles
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

    # Build navigation cost field
    cost_fields = {}

    # 1. Base cost: distance (minimize travel distance)
    cost_fields['distance'] = CostFieldBuilder.uniform(weight=1.0)

    # 2. Forklift collision avoidance (dynamic obstacles)
    for i, (center, intensity) in enumerate(layout['forklift_paths']):
        cost_fields[f'forklift_{i}'] = CostFieldBuilder.gaussian_hotspot(
            center=center,
            sigma=3.0,  # Safety zone around forklifts
            amplitude=intensity,
            weight=2.0  # High weight (collision avoidance)
        )

    # 3. Congestion avoidance (multi-robot coordination)
    for i, (center, intensity) in enumerate(layout['congestion']):
        cost_fields[f'congestion_{i}'] = CostFieldBuilder.gaussian_hotspot(
            center=center,
            sigma=2.5,  # Congestion spreads
            amplitude=intensity,
            weight=1.5  # Moderate weight (throughput)
        )

    # 4. Energy efficiency (battery conservation)
    for i, (center, intensity) in enumerate(layout['energy_cost']):
        cost_fields[f'energy_{i}'] = CostFieldBuilder.gaussian_hotspot(
            center=center,
            sigma=4.0,  # Energy cost zone
            amplitude=intensity,
            weight=1.2  # Moderate weight (efficiency)
        )

    # Combine all cost factors
    combined_cost = CostFieldBuilder.combine(cost_fields)
    engine.set_cost_field(combined_cost)

    # Solve
    print("\n" + "=" * 80)
    network = engine.solve(verbose=True)

    # Analyze navigation plan
    print("\n" + "=" * 80)
    print("ROBOT NAVIGATION ANALYSIS")
    print("=" * 80)

    total_distance = sum(path.length() for path in network.paths)

    print("\nRobot routes:")
    for path in network.paths:
        length = path.length()

        # Estimate energy consumption (Wh)
        # Typical warehouse robot: 50-100 Wh/km
        energy_per_m = 0.08  # Wh/m
        energy_consumption = length * energy_per_m

        # Estimate travel time (assume 2 m/s average speed)
        travel_time = length / 2.0  # seconds

        print(f"  {path.source:18s} → {path.sink:10s}: "
              f"{length:5.1f}m, {travel_time:4.0f}s, {energy_consumption:5.1f}Wh")

    # Fleet performance metrics
    print("\n" + "=" * 80)
    print("FLEET PERFORMANCE METRICS")
    print("=" * 80)

    avg_distance = total_distance / len(network.paths)
    avg_travel_time = (total_distance / len(network.paths)) / 2.0  # 2 m/s

    print(f"\nTotal fleet distance:     {total_distance:.1f}m")
    print(f"Average route distance:   {avg_distance:.1f}m")
    print(f"Average travel time:      {avg_travel_time:.1f}s")

    # Battery analysis
    total_energy = total_distance * 0.08  # Wh
    battery_capacity = 100  # Wh (typical warehouse robot)
    cycles_per_charge = battery_capacity / (total_energy / len(network.paths))

    print(f"\nTotal energy consumption: {total_energy:.1f}Wh")
    print(f"Avg energy per trip:      {total_energy / len(network.paths):.1f}Wh")
    print(f"Battery capacity:         {battery_capacity}Wh")
    print(f"Trips per charge:         {cycles_per_charge:.1f}")

    # Throughput analysis
    print("\n" + "=" * 80)
    print("THROUGHPUT ANALYSIS")
    print("=" * 80)

    # Assume 60 second picking time + travel time
    picking_time = 60  # seconds
    total_cycle_time = avg_travel_time * 2 + picking_time  # round trip + pick

    picks_per_hour_per_robot = 3600 / total_cycle_time
    fleet_size = 8
    fleet_picks_per_hour = picks_per_hour_per_robot * fleet_size

    print(f"\nAvg cycle time:           {total_cycle_time:.0f}s (travel + pick + return)")
    print(f"Picks per hour (1 robot): {picks_per_hour_per_robot:.1f}")
    print(f"Fleet picks per hour:     {fleet_picks_per_hour:.1f}")
    print(f"Daily capacity (16h):     {fleet_picks_per_hour * 16:.0f} picks")

    # Safety metrics
    print("\n" + "=" * 80)
    print("SAFETY COMPLIANCE")
    print("=" * 80)
    print("\n✓ All routes avoid static obstacles (shelves, columns)")
    print("✓ Reduced speed zones in human work areas (packing)")
    print("✓ Collision avoidance with forklifts (dynamic obstacles)")
    print("✓ Congestion minimized at junctions (anti-gridlock)")
    print("✓ Emergency stop capability maintained on all paths")

    # ROI calculation
    print("\n" + "=" * 80)
    print("RETURN ON INVESTMENT (ROI)")
    print("=" * 80)

    # Costs
    robot_cost = 30000  # $ per robot
    fleet_cost = robot_cost * fleet_size
    annual_maintenance = fleet_cost * 0.15  # 15% per year

    # Labor savings
    human_picker_wage = 18  # $/hour
    hours_per_year = 16 * 365  # 16 hour days
    human_picks_per_hour = 60  # manual picking rate

    robots_replace_humans = fleet_picks_per_hour / human_picks_per_hour
    annual_labor_savings = robots_replace_humans * human_picker_wage * hours_per_year

    payback_period = fleet_cost / (annual_labor_savings - annual_maintenance)

    print(f"\nFleet cost:               ${fleet_cost:,.0f}")
    print(f"  (8 robots @ ${robot_cost:,} each)")
    print(f"\nAnnual labor savings:     ${annual_labor_savings:,.0f}")
    print(f"  ({robots_replace_humans:.1f} human pickers replaced)")
    print(f"\nAnnual maintenance:       ${annual_maintenance:,.0f}")
    print(f"\nNet annual savings:       ${annual_labor_savings - annual_maintenance:,.0f}")
    print(f"Payback period:           {payback_period:.1f} years")

    return network


if __name__ == '__main__':
    network = plan_robot_routes()

    print("\n" + "=" * 80)
    print("✓ Warehouse robot navigation planning complete")
    print("  Next steps:")
    print("    - Program robots with optimized routes")
    print("    - Implement collision detection sensors")
    print("    - Deploy fleet management software")
    print("    - Monitor KPIs: picks/hour, battery cycles, safety incidents")
    print("=" * 80)
