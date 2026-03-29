"""Geospatial Network Solver

Solves facility-location and coverage-optimisation problems on a
geospatial network using the existing travel-time, isochrone,
population, and feasibility models.

Core problem:
    Given a set of candidate warehouse locations, select the subset
    of size *p* that maximises population coverage within a travel-time
    constraint (p-median / maximal-covering location problem).

The solver uses a greedy set-cover heuristic followed by optional
pair-wise swap improvement (Teitz & Bart style), all driven by the
Haversine travel-time physics already in the module.

Typical workflow:
    1. Build a FacilityLocationConfig (candidates, demand points, budget).
    2. Instantiate GeospatialSolver(config).
    3. Call solver.solve() -> GeospatialSolverResult.
    4. Inspect coverage, assignments, and feasibility in the result.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from .travel_time import (
    haversine_distance,
    calculate_travel_time,
    estimate_travel_time_matrix,
    RoadType,
    TravelTimeResult,
)
from .population import (
    INDIA_POPULATION_CENTERS,
    calculate_population_coverage,
    PopulationCoverageResult,
)
from .isochrones import calculate_isochrone, IsochroneResult
from .feasibility import (
    GeospatialFeasibilityChecker,
    GeospatialFeasibilityResult,
    FeasibilityStatus,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class FacilityLocationConfig:
    """Configuration for the facility-location solver.

    Attributes:
        candidate_locations:  List of (lat, lon) candidate warehouse sites.
        demand_points:        List of (lat, lon, population_millions, name)
                              demand centres.  If ``None``, the built-in
                              India population centres are used.
        max_facilities:       Maximum number of facilities to open (budget).
        time_limit_hours:     Maximum acceptable delivery time [h].
        target_coverage:      Required population-coverage fraction (0-1).
        road_type:            Road classification for speed look-up.
        circuity_factor:      Road-distance / straight-line ratio.
        swap_iterations:      Number of Teitz-Bart swap-improvement passes.
        country:              Country code (used for built-in population data).
    """
    candidate_locations: List[Tuple[float, float]] = field(default_factory=list)
    demand_points: Optional[List[Tuple[float, float, float, str]]] = None
    max_facilities: int = 5
    time_limit_hours: float = 24.0
    target_coverage: float = 0.95
    road_type: RoadType = RoadType.STATE_HIGHWAY
    circuity_factor: float = 1.3
    swap_iterations: int = 50
    country: str = "india"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class GeospatialSolverResult:
    """Result of the geospatial facility-location solver.

    Attributes:
        selected_locations:       Chosen facility (lat, lon) coordinates.
        num_facilities:           Number of facilities opened.
        coverage_fraction:        Fraction of demand population covered.
        covered_population_millions: Total covered population [M].
        total_population_millions:   Total demand population [M].
        uncovered_regions:        Names of demand centres not served.
        assignment:               Dict mapping demand-point name to
                                  (facility_index, travel_time_hours).
        travel_time_matrix:       Full candidate-to-demand travel-time
                                  matrix [hours], shape (n_candidates, n_demand).
        feasibility:              GeospatialFeasibilityResult from the
                                  feasibility checker run on the solution.
        objective_value:          Optimisation objective (total weighted
                                  travel time for served demand).
        iterations:               Number of swap-improvement iterations
                                  executed.
        config:                   The solver configuration.
    """
    selected_locations: List[Tuple[float, float]]
    num_facilities: int
    coverage_fraction: float
    covered_population_millions: float
    total_population_millions: float
    uncovered_regions: List[str]
    assignment: Dict[str, Tuple[int, float]]
    travel_time_matrix: np.ndarray
    feasibility: GeospatialFeasibilityResult
    objective_value: float
    iterations: int
    config: FacilityLocationConfig


@dataclass
class NetworkFlowResult:
    """Result container for the Level 3 network flow optimisation solver.

    Contains the optimised facility allocation with capacity constraints,
    Voronoi tessellation, shortest-path routing on a weighted graph,
    demand propagation via advection-diffusion, and service level contours.

    Attributes:
        selected_locations:         Chosen facility (lat, lon) coordinates.
        num_facilities:             Number of facilities opened.
        coverage_fraction:          Fraction of demand population covered.
        covered_population_millions: Total covered population [M].
        total_population_millions:  Total demand population [M].
        uncovered_regions:          Names of demand centres not served.
        facility_allocations:       Dict mapping facility index to list of
                                    (demand_index, flow_amount) tuples.
        facility_loads:             Array of total flow assigned per facility.
        facility_capacities:        Array of capacity per facility [M].
        voronoi_labels:             2-D grid of nearest-facility indices.
        voronoi_distances:          2-D grid of distance to nearest facility.
        demand_density:             2-D steady-state demand density from
                                    advection-diffusion propagation.
        demand_density_initial:     2-D initial demand density before propagation.
        shortest_path_matrix:       (n_facilities, n_demand) matrix of
                                    shortest-path distances on the graph [km].
        graph_adjacency:            Sparse adjacency matrix of the routing graph.
        graph_node_coords:          (N_nodes, 2) array of graph node (lat, lon).
        service_level_contours:     Dict mapping travel-time thresholds [h]
                                    to 2-D boolean grids (True = reachable).
        flow_objective:             Total weighted flow cost (minimised).
        diffusion_converged:        Whether the advection-diffusion solver converged.
        diffusion_iterations:       Iterations used for advection-diffusion solve.
        grid_lat:                   1-D latitude array for the 2-D grids.
        grid_lon:                   1-D longitude array for the 2-D grids.
        config:                     The FacilityLocationConfig used.
    """
    selected_locations: List[Tuple[float, float]]
    num_facilities: int
    coverage_fraction: float
    covered_population_millions: float
    total_population_millions: float
    uncovered_regions: List[str]
    facility_allocations: Dict[str, List[Tuple[int, float]]]
    facility_loads: np.ndarray
    facility_capacities: np.ndarray
    voronoi_labels: np.ndarray
    voronoi_distances: np.ndarray
    demand_density: np.ndarray
    demand_density_initial: np.ndarray
    shortest_path_matrix: np.ndarray
    graph_adjacency: np.ndarray
    graph_node_coords: np.ndarray
    service_level_contours: Dict[float, np.ndarray]
    flow_objective: float
    diffusion_converged: bool
    diffusion_iterations: int
    grid_lat: np.ndarray
    grid_lon: np.ndarray
    config: FacilityLocationConfig


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class GeospatialSolver:
    """Greedy + swap-improvement solver for the maximal-covering
    facility-location problem on a geospatial network.

    The algorithm proceeds in two phases:

    1. **Greedy construction** -- iteratively select the candidate that
       covers the most uncovered population until the facility budget
       is exhausted or coverage target is met.

    2. **Swap improvement** (Teitz-Bart) -- for each opened facility,
       try replacing it with every unopened candidate; accept the swap
       if total weighted travel time decreases.

    Parameters
    ----------
    config : FacilityLocationConfig
        Problem specification.
    """

    fidelity_tier = FidelityTier.HEURISTIC
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(self, config: FacilityLocationConfig):
        self.config = config

        # Demand points
        if config.demand_points is not None:
            self.demand = config.demand_points
        else:
            self.demand = list(INDIA_POPULATION_CENTERS)

        self.n_demand = len(self.demand)
        self.n_candidates = len(config.candidate_locations)

        # Pre-compute travel-time matrix  (candidates x demand)
        self.tt_matrix = self._build_travel_time_matrix()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def solve(self) -> GeospatialSolverResult:
        """Run the facility-location solver.

        Returns
        -------
        GeospatialSolverResult
        """
        cfg = self.config
        populations = np.array([d[2] for d in self.demand])  # millions
        names = [d[3] for d in self.demand]
        total_pop = float(populations.sum())

        # Binary reachability matrix  (candidates x demand)
        reachable = self.tt_matrix <= cfg.time_limit_hours  # bool

        # Phase 1 -- greedy construction
        selected_indices = self._greedy_select(reachable, populations)

        # Phase 2 -- swap improvement
        selected_indices = self._swap_improve(
            selected_indices, populations, cfg.swap_iterations
        )

        # Build solution details
        selected_locs = [cfg.candidate_locations[i] for i in selected_indices]

        # Assignment: each demand point -> nearest selected facility
        assignment: Dict[str, Tuple[int, float]] = {}
        covered_pop = 0.0
        uncovered: List[str] = []

        for j in range(self.n_demand):
            best_fac = -1
            best_tt = np.inf
            for rank, fi in enumerate(selected_indices):
                if self.tt_matrix[fi, j] < best_tt:
                    best_tt = self.tt_matrix[fi, j]
                    best_fac = rank
            if best_tt <= cfg.time_limit_hours:
                assignment[names[j]] = (best_fac, float(best_tt))
                covered_pop += populations[j]
            else:
                uncovered.append(names[j])

        coverage_frac = covered_pop / total_pop if total_pop > 0 else 0.0

        # Objective: population-weighted total travel time (served only)
        obj = 0.0
        for j in range(self.n_demand):
            min_tt = min(self.tt_matrix[fi, j] for fi in selected_indices)
            if min_tt <= cfg.time_limit_hours:
                obj += populations[j] * min_tt

        # Run feasibility checker on the solution
        feasibility = GeospatialFeasibilityChecker.check_24h_coverage(
            warehouse_locations=selected_locs,
            target_coverage=cfg.target_coverage,
            time_limit_hours=cfg.time_limit_hours,
            country=cfg.country,
            road_type=cfg.road_type,
        )

        return GeospatialSolverResult(
            selected_locations=selected_locs,
            num_facilities=len(selected_indices),
            coverage_fraction=coverage_frac,
            covered_population_millions=float(covered_pop),
            total_population_millions=float(total_pop),
            uncovered_regions=uncovered,
            assignment=assignment,
            travel_time_matrix=self.tt_matrix,
            feasibility=feasibility,
            objective_value=float(obj),
            iterations=cfg.swap_iterations,
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_travel_time_matrix(self) -> np.ndarray:
        """Compute travel-time from every candidate to every demand point.

        Returns shape (n_candidates, n_demand) in hours.
        """
        cfg = self.config
        tt = np.empty((self.n_candidates, self.n_demand))
        for i, (clat, clon) in enumerate(cfg.candidate_locations):
            for j, (dlat, dlon, _pop, _name) in enumerate(self.demand):
                dist_km = haversine_distance(clat, clon, dlat, dlon)
                road_km = dist_km * cfg.circuity_factor
                speed = cfg.road_type.value  # km/h
                tt[i, j] = road_km / speed
        return tt

    def _greedy_select(self, reachable: np.ndarray,
                       populations: np.ndarray) -> List[int]:
        """Greedy set-cover: pick candidate covering most uncovered pop."""
        cfg = self.config
        selected: List[int] = []
        covered = np.zeros(self.n_demand, dtype=bool)
        remaining = set(range(self.n_candidates))

        for _ in range(cfg.max_facilities):
            best_gain = -1.0
            best_cand = -1
            for c in remaining:
                # New demand covered if we add c
                new_covered = reachable[c] & ~covered
                gain = float(populations[new_covered].sum())
                if gain > best_gain:
                    best_gain = gain
                    best_cand = c
            if best_cand < 0 or best_gain <= 0:
                break
            selected.append(best_cand)
            remaining.discard(best_cand)
            covered |= reachable[best_cand]

            # Early exit if target met
            if populations[covered].sum() / populations.sum() >= cfg.target_coverage:
                break

        return selected

    def export_state(self, result: GeospatialSolverResult) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="geospatial")
        state.set_field("travel_time", result.travel_time_matrix, "s")
        state.metadata["coverage_fraction"] = result.coverage_fraction
        state.metadata["covered_population_millions"] = result.covered_population_millions
        state.metadata["total_population_millions"] = result.total_population_millions
        state.metadata["num_facilities"] = result.num_facilities
        state.metadata["objective_value"] = result.objective_value
        return state

    def _swap_improve(self, selected: List[int],
                      populations: np.ndarray,
                      max_iters: int) -> List[int]:
        """Teitz-Bart swap improvement on weighted travel time."""
        cfg = self.config
        selected = list(selected)
        unopened = [c for c in range(self.n_candidates) if c not in selected]

        def objective(sel: List[int]) -> float:
            """Population-weighted total travel time (served demand only)."""
            total = 0.0
            for j in range(self.n_demand):
                min_tt = min(self.tt_matrix[fi, j] for fi in sel)
                if min_tt <= cfg.time_limit_hours:
                    total += populations[j] * min_tt
            return total

        current_obj = objective(selected)

        for _ in range(max_iters):
            improved = False
            for si, s in enumerate(selected):
                for ui, u in enumerate(unopened):
                    trial = selected.copy()
                    trial[si] = u
                    trial_obj = objective(trial)
                    if trial_obj < current_obj:
                        # Accept swap
                        unopened[ui] = s
                        selected = trial
                        current_obj = trial_obj
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break

        return selected

    # ------------------------------------------------------------------
    # Level 3: Network flow optimisation
    # ------------------------------------------------------------------
    def solve_network_flow(
        self,
        grid_resolution: int = 64,
        facility_capacity_millions: Optional[List[float]] = None,
        diffusion_coefficient: float = 0.5,
        advection_velocity: Tuple[float, float] = (0.0, 0.0),
        diffusion_iterations: int = 2000,
        diffusion_tolerance: float = 1e-5,
        service_time_thresholds: Optional[List[float]] = None,
        demand_spread_sigma: float = 1.5,
        graph_connectivity: int = 8,
    ) -> NetworkFlowResult:
        """Network flow optimisation with spatial demand propagation,
        Voronoi tessellation, Dijkstra shortest paths, capacity-constrained
        allocation, and service level contour mapping.

        Solves the following coupled problems on a 2-D spatial grid:

        1. **Advection-diffusion equation** for demand propagation:
           dC/dt = D * Laplacian(C) - v . grad(C)
           Solved to steady state via explicit FDM.  Initial demand is
           placed as Gaussian blobs at each demand point.  The diffused
           field represents effective demand density accounting for
           population mobility and access patterns.

        2. **Voronoi tessellation** from selected facility locations:
           Each grid cell is assigned to its nearest facility using
           Haversine distance, producing service regions.

        3. **Dijkstra shortest path** on a weighted spatial graph:
           Nodes are placed at candidate + demand locations.  Edge
           weights are Haversine distances multiplied by a terrain
           factor derived from the demand density (higher demand =
           better infrastructure = lower cost).

        4. **Capacity-constrained facility allocation**:
           Demand points are assigned to facilities respecting capacity
           limits.  Overflow demand is redirected to next-nearest
           facility via a greedy augmenting path algorithm.

        5. **Service level contour mapping**:
           For each travel-time threshold, a boolean reachability grid
           is computed showing which areas can be served within that time.

        Parameters
        ----------
        grid_resolution : int
            Number of grid cells in each spatial dimension.
        facility_capacity_millions : list of float, optional
            Capacity of each candidate facility [millions of people].
            Defaults to total_pop / max_facilities * 1.5 per facility.
        diffusion_coefficient : float
            Diffusion coefficient D for demand propagation [deg^2/step].
        advection_velocity : tuple of float
            (v_lat, v_lon) advection velocity for demand drift.
        diffusion_iterations : int
            Maximum iterations for the advection-diffusion solver.
        diffusion_tolerance : float
            Convergence tolerance for the diffusion solver.
        service_time_thresholds : list of float, optional
            Travel-time thresholds [hours] for service contours.
            Defaults to [6, 12, 24, 48].
        demand_spread_sigma : float
            Gaussian spread [degrees] for initial demand placement.
        graph_connectivity : int
            Graph connectivity: 4 (cardinal) or 8 (incl. diagonal).

        Returns
        -------
        NetworkFlowResult
            Full solution with allocations, Voronoi regions, demand
            fields, shortest paths, and service contours.
        """
        cfg = self.config
        populations = np.array([d[2] for d in self.demand])
        names = [d[3] for d in self.demand]
        total_pop = float(populations.sum())

        if service_time_thresholds is None:
            service_time_thresholds = [6.0, 12.0, 24.0, 48.0]

        # =============================================================
        # Step 0: Determine spatial extent and build grid
        # =============================================================
        all_lats = ([d[0] for d in self.demand]
                    + [c[0] for c in cfg.candidate_locations])
        all_lons = ([d[1] for d in self.demand]
                    + [c[1] for c in cfg.candidate_locations])
        lat_min = min(all_lats) - 2.0
        lat_max = max(all_lats) + 2.0
        lon_min = min(all_lons) - 2.0
        lon_max = max(all_lons) + 2.0

        n_grid = grid_resolution
        grid_lat = np.linspace(lat_min, lat_max, n_grid)
        grid_lon = np.linspace(lon_min, lon_max, n_grid)
        dlat = grid_lat[1] - grid_lat[0]
        dlon = grid_lon[1] - grid_lon[0]

        # =============================================================
        # Step 1: Build initial demand density on 2-D grid
        #         Gaussian blobs centred at each demand point
        # =============================================================
        demand_density_init = np.zeros((n_grid, n_grid))
        for d_idx, (dlat_c, dlon_c, dpop, _) in enumerate(self.demand):
            for i in range(n_grid):
                for j in range(n_grid):
                    dist_sq = ((grid_lat[i] - dlat_c) ** 2
                               + (grid_lon[j] - dlon_c) ** 2)
                    demand_density_init[i, j] += dpop * np.exp(
                        -dist_sq / (2.0 * demand_spread_sigma ** 2)
                    )

        # Normalise so total integral approximates total population
        grid_integral = np.sum(demand_density_init) * dlat * dlon
        if grid_integral > 0:
            demand_density_init *= total_pop / grid_integral

        # =============================================================
        # Step 2: Solve advection-diffusion for demand propagation
        #         dC/dt = D * Laplacian(C) - v . grad(C)
        #         Explicit FDM with upwind scheme for advection
        # =============================================================
        C = demand_density_init.copy()
        D = diffusion_coefficient
        v_lat, v_lon = advection_velocity

        # Stability condition for explicit scheme
        dt_diff = 0.25 * min(dlat, dlon) ** 2 / (D + 1e-15)
        dt_adv = 0.5 * min(dlat, dlon) / (
            abs(v_lat) + abs(v_lon) + 1e-15
        )
        dt = 0.8 * min(dt_diff, dt_adv, 1.0)

        rx = D * dt / (dlat ** 2)
        ry = D * dt / (dlon ** 2)
        ax = v_lat * dt / (2.0 * dlat)
        ay = v_lon * dt / (2.0 * dlon)

        diff_converged = False
        diff_iters_used = 0

        for iteration in range(diffusion_iterations):
            C_old = C.copy()

            # Diffusion: central difference Laplacian
            laplacian = np.zeros_like(C)
            laplacian[1:-1, 1:-1] = (
                rx * (C_old[2:, 1:-1] + C_old[:-2, 1:-1]
                      - 2.0 * C_old[1:-1, 1:-1])
                + ry * (C_old[1:-1, 2:] + C_old[1:-1, :-2]
                        - 2.0 * C_old[1:-1, 1:-1])
            )

            # Advection: upwind differencing
            advection = np.zeros_like(C)
            if v_lat >= 0:
                advection[1:-1, :] -= ax * (
                    C_old[1:-1, :] - C_old[:-2, :]
                )
            else:
                advection[1:-1, :] -= ax * (
                    C_old[2:, :] - C_old[1:-1, :]
                )
            if v_lon >= 0:
                advection[:, 1:-1] -= ay * (
                    C_old[:, 1:-1] - C_old[:, :-2]
                )
            else:
                advection[:, 1:-1] -= ay * (
                    C_old[:, 2:] - C_old[:, 1:-1]
                )

            C = C_old + laplacian + advection

            # Zero-flux (Neumann) boundary conditions
            C[0, :] = C[1, :]
            C[-1, :] = C[-2, :]
            C[:, 0] = C[:, 1]
            C[:, -1] = C[:, -2]

            # Ensure non-negativity
            C = np.maximum(C, 0.0)

            residual = np.max(np.abs(C - C_old))
            diff_iters_used = iteration + 1
            if residual < diffusion_tolerance:
                diff_converged = True
                break

        demand_density = C

        # =============================================================
        # Step 3: Run base greedy+swap to select facilities
        # =============================================================
        reachable = self.tt_matrix <= cfg.time_limit_hours
        selected_indices = self._greedy_select(reachable, populations)
        selected_indices = self._swap_improve(
            selected_indices, populations, cfg.swap_iterations
        )
        selected_locs = [cfg.candidate_locations[i] for i in selected_indices]
        n_selected = len(selected_indices)

        # =============================================================
        # Step 4: Voronoi tessellation on the 2-D grid
        # =============================================================
        voronoi_labels = np.full((n_grid, n_grid), -1, dtype=int)
        voronoi_distances = np.full((n_grid, n_grid), np.inf)

        for f_rank, f_idx in enumerate(selected_indices):
            f_lat, f_lon = cfg.candidate_locations[f_idx]
            for i in range(n_grid):
                for j in range(n_grid):
                    d = haversine_distance(
                        grid_lat[i], grid_lon[j], f_lat, f_lon
                    )
                    if d < voronoi_distances[i, j]:
                        voronoi_distances[i, j] = d
                        voronoi_labels[i, j] = f_rank

        # =============================================================
        # Step 5: Build weighted spatial graph and Dijkstra shortest paths
        # =============================================================
        # Graph nodes: all candidate + demand point locations
        node_coords = []
        node_type = []  # 'candidate' or 'demand'
        for c_lat, c_lon in cfg.candidate_locations:
            node_coords.append((c_lat, c_lon))
            node_type.append("candidate")
        for d_lat, d_lon, _, _ in self.demand:
            node_coords.append((d_lat, d_lon))
            node_type.append("demand")
        n_nodes = len(node_coords)
        node_coords_arr = np.array(node_coords)

        # Build adjacency: connect each node to its K nearest neighbors
        k_neighbors = min(graph_connectivity, n_nodes - 1)
        adj = np.full((n_nodes, n_nodes), np.inf)
        np.fill_diagonal(adj, 0.0)

        # Compute all pairwise Haversine distances
        dist_matrix = np.zeros((n_nodes, n_nodes))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                d = haversine_distance(
                    node_coords[i][0], node_coords[i][1],
                    node_coords[j][0], node_coords[j][1],
                )
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Connect K nearest neighbors with weighted edges
        for i in range(n_nodes):
            dists = dist_matrix[i].copy()
            dists[i] = np.inf
            nearest = np.argsort(dists)[:k_neighbors]
            for j in nearest:
                # Weight: distance * circuity, reduced where demand is high
                raw_dist = dist_matrix[i, j] * cfg.circuity_factor
                # Infrastructure bonus: interpolate demand density at midpoint
                mid_lat = 0.5 * (node_coords[i][0] + node_coords[j][0])
                mid_lon = 0.5 * (node_coords[i][1] + node_coords[j][1])
                gi = int(np.clip(
                    (mid_lat - lat_min) / dlat, 0, n_grid - 1
                ))
                gj = int(np.clip(
                    (mid_lon - lon_min) / dlon, 0, n_grid - 1
                ))
                density_factor = 1.0 / (
                    1.0 + 0.01 * demand_density[gi, gj]
                )
                weight = raw_dist * density_factor
                adj[i, j] = min(adj[i, j], weight)
                adj[j, i] = min(adj[j, i], weight)

        # Dijkstra shortest paths from each selected facility to all nodes
        def _dijkstra(source: int) -> np.ndarray:
            """Single-source Dijkstra on adjacency matrix."""
            dist = np.full(n_nodes, np.inf)
            dist[source] = 0.0
            visited = np.zeros(n_nodes, dtype=bool)
            for _ in range(n_nodes):
                # Pick unvisited node with smallest distance
                temp_dist = dist.copy()
                temp_dist[visited] = np.inf
                u = int(np.argmin(temp_dist))
                if dist[u] == np.inf:
                    break
                visited[u] = True
                # Relax neighbors
                for v in range(n_nodes):
                    if not visited[v] and adj[u, v] < np.inf:
                        alt = dist[u] + adj[u, v]
                        if alt < dist[v]:
                            dist[v] = alt
            return dist

        # Shortest path matrix: facilities -> demand points
        n_cand = self.n_candidates
        sp_matrix = np.full((n_selected, self.n_demand), np.inf)
        for f_rank, f_idx in enumerate(selected_indices):
            # Node index for this facility candidate
            sp_dists = _dijkstra(f_idx)
            for d_idx in range(self.n_demand):
                d_node = n_cand + d_idx
                sp_matrix[f_rank, d_idx] = sp_dists[d_node]

        # Convert graph distances to travel times
        speed = cfg.road_type.value  # km/h
        sp_time_matrix = sp_matrix / speed  # hours

        # =============================================================
        # Step 6: Capacity-constrained facility allocation
        # =============================================================
        if facility_capacity_millions is not None:
            capacities = np.array([
                facility_capacity_millions[selected_indices[r]]
                if selected_indices[r] < len(facility_capacity_millions)
                else total_pop / n_selected * 1.5
                for r in range(n_selected)
            ])
        else:
            capacities = np.full(
                n_selected, total_pop / max(n_selected, 1) * 1.5
            )

        # Greedy allocation: assign each demand to nearest facility
        # with remaining capacity
        facility_loads = np.zeros(n_selected)
        facility_alloc: Dict[str, List[Tuple[int, float]]] = {
            str(r): [] for r in range(n_selected)
        }
        assignment: Dict[str, Tuple[int, float]] = {}
        covered_pop = 0.0
        uncovered: List[str] = []

        # Sort demand points by population (descending) for priority
        demand_order = np.argsort(-populations)

        for d_idx in demand_order:
            d_name = names[d_idx]
            d_pop = populations[d_idx]

            # Rank facilities by shortest-path travel time
            fac_order = np.argsort(sp_time_matrix[:, d_idx])

            assigned = False
            for f_rank in fac_order:
                tt = sp_time_matrix[f_rank, d_idx]
                if tt > cfg.time_limit_hours:
                    continue
                remaining_cap = capacities[f_rank] - facility_loads[f_rank]
                if remaining_cap <= 0:
                    continue
                # Assign (possibly partial if over capacity)
                alloc_amount = min(d_pop, remaining_cap)
                facility_loads[f_rank] += alloc_amount
                facility_alloc[str(f_rank)].append(
                    (int(d_idx), float(alloc_amount))
                )

                if not assigned:
                    assignment[d_name] = (int(f_rank), float(tt))
                    assigned = True

                d_pop -= alloc_amount
                covered_pop += alloc_amount

                if d_pop <= 1e-6:
                    break

            if not assigned:
                uncovered.append(d_name)
            elif d_pop > 1e-6:
                # Partial coverage: remaining demand is unserved
                pass

        coverage_frac = covered_pop / total_pop if total_pop > 0 else 0.0

        # Flow objective: total weighted cost
        flow_obj = 0.0
        for f_key, allocs in facility_alloc.items():
            f_rank = int(f_key)
            for (d_idx, amount) in allocs:
                flow_obj += amount * sp_time_matrix[f_rank, d_idx]

        # =============================================================
        # Step 7: Service level contour mapping on 2-D grid
        # =============================================================
        service_contours: Dict[float, np.ndarray] = {}

        for threshold in service_time_thresholds:
            reachable_grid = np.zeros((n_grid, n_grid), dtype=bool)
            for f_rank, f_idx in enumerate(selected_indices):
                f_lat, f_lon = cfg.candidate_locations[f_idx]
                # Maximum reachable distance at this threshold
                max_dist_km = threshold * speed / cfg.circuity_factor
                for i in range(n_grid):
                    for j in range(n_grid):
                        d_km = haversine_distance(
                            grid_lat[i], grid_lon[j], f_lat, f_lon
                        )
                        if d_km <= max_dist_km:
                            reachable_grid[i, j] = True
            service_contours[threshold] = reachable_grid

        return NetworkFlowResult(
            selected_locations=selected_locs,
            num_facilities=n_selected,
            coverage_fraction=coverage_frac,
            covered_population_millions=float(covered_pop),
            total_population_millions=float(total_pop),
            uncovered_regions=uncovered,
            facility_allocations=facility_alloc,
            facility_loads=facility_loads,
            facility_capacities=capacities,
            voronoi_labels=voronoi_labels,
            voronoi_distances=voronoi_distances,
            demand_density=demand_density,
            demand_density_initial=demand_density_init,
            shortest_path_matrix=sp_matrix,
            graph_adjacency=adj,
            graph_node_coords=node_coords_arr,
            service_level_contours=service_contours,
            flow_objective=float(flow_obj),
            diffusion_converged=diff_converged,
            diffusion_iterations=diff_iters_used,
            grid_lat=grid_lat,
            grid_lon=grid_lon,
            config=cfg,
        )


# ---------------------------------------------------------------------------
# Level 3 2-D: Terrain analysis solver
# ---------------------------------------------------------------------------

@dataclass
class Geospatial2DResult:
    """Result container for the 2-D geospatial terrain analysis solver.

    Attributes
    ----------
    elevation : np.ndarray
        Elevation field (ny, nx) [m].
    slope : np.ndarray
        Slope magnitude field (ny, nx) [radians].
    aspect : np.ndarray
        Aspect (downhill direction) field (ny, nx) [radians from north, clockwise].
    hillshade : np.ndarray
        Hillshade illumination field (ny, nx) [0-1].
    viewshed : np.ndarray
        Line-of-sight visibility mask from observer (ny, nx) [bool].
    curvature : np.ndarray
        Surface curvature (Laplacian of elevation) (ny, nx) [1/m].
    flow_accumulation : np.ndarray
        D8 flow accumulation (ny, nx) [cell count].
    x : np.ndarray
        x-coordinate array (nx,) [m].
    y : np.ndarray
        y-coordinate array (ny,) [m].
    observer_pos : Tuple[float, float, float]
        Observer position (x, y, height_above_ground) [m].
    mean_slope_deg : float
        Domain-averaged slope [degrees].
    max_elevation : float
        Maximum elevation [m].
    min_elevation : float
        Minimum elevation [m].
    visible_fraction : float
        Fraction of domain visible from observer.
    """
    elevation: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    slope: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    aspect: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    hillshade: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    viewshed: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    curvature: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    flow_accumulation: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    y: np.ndarray = field(default_factory=lambda: np.zeros(0))
    observer_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mean_slope_deg: float = 0.0
    max_elevation: float = 0.0
    min_elevation: float = 0.0
    visible_fraction: float = 0.0


class Geospatial2DSolver:
    """2-D terrain analysis solver.

    Generates a synthetic terrain elevation field and computes slope,
    aspect, hillshade, line-of-sight viewshed, curvature, and D8 flow
    accumulation on a regular Cartesian grid.

    The terrain is generated as a superposition of sinusoidal modes plus
    Perlin-like noise to produce realistic topography.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    Lx, Ly : float
        Domain size [m].
    base_elevation : float
        Mean terrain elevation [m].
    roughness : float
        Terrain roughness amplitude [m].
    n_peaks : int
        Number of dominant terrain peaks/ridges.
    observer_x, observer_y : float
        Observer position [m].
    observer_height : float
        Observer height above local ground [m].
    sun_azimuth_deg : float
        Sun azimuth for hillshade [degrees from north, clockwise].
    sun_altitude_deg : float
        Sun altitude for hillshade [degrees above horizon].
    seed : int or None
        Random seed for terrain generation.
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        nx: int = 100,
        ny: int = 100,
        Lx: float = 10000.0,
        Ly: float = 10000.0,
        base_elevation: float = 500.0,
        roughness: float = 200.0,
        n_peaks: int = 5,
        observer_x: float = 5000.0,
        observer_y: float = 5000.0,
        observer_height: float = 2.0,
        sun_azimuth_deg: float = 315.0,
        sun_altitude_deg: float = 45.0,
        seed: Optional[int] = None,
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.base_elevation = base_elevation
        self.roughness = roughness
        self.n_peaks = n_peaks
        self.observer_x = observer_x
        self.observer_y = observer_y
        self.observer_height = observer_height
        self.sun_azimuth_deg = sun_azimuth_deg
        self.sun_altitude_deg = sun_altitude_deg
        self.rng = np.random.default_rng(seed)

    def solve(self) -> Geospatial2DResult:
        """Generate terrain and compute all derived fields.

        Returns
        -------
        Geospatial2DResult
        """
        nx, ny = self.nx, self.ny
        dx = self.Lx / (nx - 1)
        dy = self.Ly / (ny - 1)
        x = np.linspace(0, self.Lx, nx)
        y = np.linspace(0, self.Ly, ny)
        X, Y = np.meshgrid(x, y)  # (ny, nx)

        # --- Generate synthetic terrain ---
        elev = np.full((ny, nx), self.base_elevation)
        for _ in range(self.n_peaks):
            cx = self.rng.uniform(0, self.Lx)
            cy = self.rng.uniform(0, self.Ly)
            amp = self.rng.uniform(0.3, 1.0) * self.roughness
            sx = self.rng.uniform(self.Lx * 0.05, self.Lx * 0.3)
            sy = self.rng.uniform(self.Ly * 0.05, self.Ly * 0.3)
            elev += amp * np.exp(-((X - cx) ** 2 / (2 * sx ** 2)
                                   + (Y - cy) ** 2 / (2 * sy ** 2)))
        # Add multi-scale noise
        for freq in [2, 4, 8, 16]:
            amp = self.roughness / (freq * 2.0)
            phase_x = self.rng.uniform(0, 2 * np.pi)
            phase_y = self.rng.uniform(0, 2 * np.pi)
            elev += amp * np.sin(2 * np.pi * freq * X / self.Lx + phase_x) * \
                    np.cos(2 * np.pi * freq * Y / self.Ly + phase_y)

        # --- Slope and aspect (central differences) ---
        dzdx = np.zeros((ny, nx))
        dzdy = np.zeros((ny, nx))
        dzdx[:, 1:-1] = (elev[:, 2:] - elev[:, :-2]) / (2 * dx)
        dzdx[:, 0] = (elev[:, 1] - elev[:, 0]) / dx
        dzdx[:, -1] = (elev[:, -1] - elev[:, -2]) / dx
        dzdy[1:-1, :] = (elev[2:, :] - elev[:-2, :]) / (2 * dy)
        dzdy[0, :] = (elev[1, :] - elev[0, :]) / dy
        dzdy[-1, :] = (elev[-1, :] - elev[-2, :]) / dy

        slope = np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2))
        aspect = np.arctan2(-dzdx, dzdy)  # north=0, clockwise positive
        aspect = np.mod(aspect, 2 * np.pi)

        # --- Hillshade ---
        sun_az = np.radians(self.sun_azimuth_deg)
        sun_alt = np.radians(self.sun_altitude_deg)
        hillshade = (np.cos(sun_alt) * np.sin(slope)
                     * np.cos(aspect - sun_az)
                     + np.sin(sun_alt) * np.cos(slope))
        hillshade = np.clip(hillshade, 0.0, 1.0)

        # --- Curvature (Laplacian of elevation) ---
        curv = np.zeros((ny, nx))
        curv[1:-1, 1:-1] = (
            (elev[1:-1, 2:] + elev[1:-1, :-2] - 2 * elev[1:-1, 1:-1]) / dx ** 2
            + (elev[2:, 1:-1] + elev[:-2, 1:-1] - 2 * elev[1:-1, 1:-1]) / dy ** 2
        )

        # --- Viewshed (line-of-sight from observer) ---
        obs_i = int(np.clip(self.observer_x / dx, 0, nx - 1))
        obs_j = int(np.clip(self.observer_y / dy, 0, ny - 1))
        obs_z = elev[obs_j, obs_i] + self.observer_height
        viewshed = np.zeros((ny, nx), dtype=bool)
        viewshed[obs_j, obs_i] = True

        for j in range(ny):
            for i in range(nx):
                if i == obs_i and j == obs_j:
                    continue
                dist_x = (i - obs_i) * dx
                dist_y = (j - obs_j) * dy
                dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
                if dist < 1e-6:
                    viewshed[j, i] = True
                    continue
                target_z = elev[j, i]
                angle_to_target = (target_z - obs_z) / dist

                # Check intermediate cells along the ray
                n_samples = max(abs(i - obs_i), abs(j - obs_j))
                visible = True
                for s in range(1, n_samples):
                    frac = s / n_samples
                    si = int(obs_i + frac * (i - obs_i))
                    sj = int(obs_j + frac * (j - obs_j))
                    si = max(0, min(si, nx - 1))
                    sj = max(0, min(sj, ny - 1))
                    s_dist = frac * dist
                    if s_dist < 1e-6:
                        continue
                    angle_to_cell = (elev[sj, si] - obs_z) / s_dist
                    if angle_to_cell > angle_to_target:
                        visible = False
                        break
                viewshed[j, i] = visible

        # --- D8 flow accumulation ---
        flow_acc = np.ones((ny, nx), dtype=float)
        # Sort cells by descending elevation
        flat_idx = np.argsort(elev.ravel())[::-1]
        d8_di = [-1, -1, 0, 1, 1, 1, 0, -1]
        d8_dj = [0, 1, 1, 1, 0, -1, -1, -1]
        for idx in flat_idx:
            j, i = divmod(idx, nx)
            # Find steepest downhill neighbour
            max_drop = 0.0
            best_ni, best_nj = -1, -1
            for d in range(8):
                ni = i + d8_di[d]
                nj = j + d8_dj[d]
                if 0 <= ni < nx and 0 <= nj < ny:
                    dd = np.sqrt((d8_di[d] * dx) ** 2 + (d8_dj[d] * dy) ** 2)
                    drop = (elev[j, i] - elev[nj, ni]) / dd
                    if drop > max_drop:
                        max_drop = drop
                        best_ni, best_nj = ni, nj
            if best_ni >= 0:
                flow_acc[best_nj, best_ni] += flow_acc[j, i]

        mean_slope_deg = float(np.degrees(np.mean(slope)))
        visible_fraction = float(np.sum(viewshed)) / (nx * ny)

        return Geospatial2DResult(
            elevation=elev,
            slope=slope,
            aspect=aspect,
            hillshade=hillshade,
            viewshed=viewshed,
            curvature=curv,
            flow_accumulation=flow_acc,
            x=x,
            y=y,
            observer_pos=(self.observer_x, self.observer_y, self.observer_height),
            mean_slope_deg=mean_slope_deg,
            max_elevation=float(np.max(elev)),
            min_elevation=float(np.min(elev)),
            visible_fraction=visible_fraction,
        )

    def export_state(self) -> PhysicsState:
        """Run solver and export as PhysicsState."""
        result = self.solve()
        state = PhysicsState(solver_name="geospatial_2d")
        state.set_field("elevation", result.elevation, "m")
        state.set_field("slope", result.slope, "rad")
        state.set_field("aspect", result.aspect, "rad")
        state.set_field("hillshade", result.hillshade, "1")
        state.metadata["mean_slope_deg"] = result.mean_slope_deg
        state.metadata["max_elevation"] = result.max_elevation
        state.metadata["min_elevation"] = result.min_elevation
        state.metadata["visible_fraction"] = result.visible_fraction
        return state
