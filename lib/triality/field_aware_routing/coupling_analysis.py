"""Multi-Conductor Coupling Analysis

Quasi-static approximation of multi-conductor interactions:
- Mutual coupling zones
- Return path quality
- Ground impedance regions
- Crosstalk risk assessment

No wave equations. No S-parameters. Stay quasi-static for speed.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass

from triality.electrostatics.conduction import ConductionResult


@dataclass
class CouplingZone:
    """Region with significant electromagnetic coupling"""
    center: Tuple[float, float]
    radius: float
    coupling_strength: float  # 0-1 scale
    conductors: List[str]     # Names of coupled conductors
    coupling_type: str        # 'capacitive', 'inductive', or 'mixed'


class MultiConductorCoupling:
    """
    Analyze coupling between multiple conductors using quasi-static approximation.

    Assumes low frequencies where wave propagation can be ignored.
    Uses field overlap and proximity metrics instead of full-wave analysis.
    """

    def __init__(self, domain_x: Tuple[float, float], domain_y: Tuple[float, float]):
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.conductors: List[Tuple[str, Callable]] = []  # (name, region_func)

    def add_conductor(self, name: str, region: Callable[[float, float], bool]):
        """
        Add conductor to coupling analysis.

        Args:
            name: Conductor identifier
            region: Function (x, y) -> bool defining conductor location
        """
        self.conductors.append((name, region))

    def compute_coupling_zones(self,
                              electrostatic_result,
                              proximity_threshold: float = 0.01,
                              field_threshold: float = None) -> List[CouplingZone]:
        """
        Identify zones with significant inter-conductor coupling.

        Args:
            electrostatic_result: ElectrostaticResult from Layer 1
            proximity_threshold: Distance threshold for proximity coupling [m]
            field_threshold: Field strength threshold for capacitive coupling [V/m]

        Returns:
            List of CouplingZone objects
        """
        if field_threshold is None:
            E_mag = electrostatic_result.field_magnitude_grid()
            field_threshold = np.percentile(E_mag, 75)

        coupling_zones = []

        # Sample grid points
        grid_x = electrostatic_result.grid_x
        grid_y = electrostatic_result.grid_y
        dx = grid_x[1] - grid_x[0]
        dy = grid_y[1] - grid_y[0]

        # For each conductor pair, find coupling regions
        for i, (name1, region1) in enumerate(self.conductors):
            for j, (name2, region2) in enumerate(self.conductors[i+1:], start=i+1):

                # Find regions where both conductors are nearby
                for ix in range(len(grid_x)):
                    for iy in range(len(grid_y)):
                        x, y = grid_x[ix], grid_y[iy]

                        # Check proximity to both conductors
                        dist1 = self._distance_to_conductor(x, y, region1, dx)
                        dist2 = self._distance_to_conductor(x, y, region2, dx)

                        if dist1 < proximity_threshold and dist2 < proximity_threshold:
                            # Both conductors nearby - potential coupling zone
                            E_mag = electrostatic_result.field_magnitude(x, y)

                            if E_mag > field_threshold:
                                # Significant field - create coupling zone
                                strength = min(1.0, E_mag / (field_threshold * 2))

                                coupling_zones.append(CouplingZone(
                                    center=(x, y),
                                    radius=min(dist1, dist2),
                                    coupling_strength=strength,
                                    conductors=[name1, name2],
                                    coupling_type='capacitive'
                                ))

        # Merge nearby coupling zones
        merged_zones = self._merge_coupling_zones(coupling_zones, merge_distance=proximity_threshold * 2)

        return merged_zones

    def _distance_to_conductor(self, x: float, y: float, region: Callable, resolution: float) -> float:
        """Estimate distance from point to conductor region"""
        # Check if inside conductor
        if region(x, y):
            return 0.0

        # Sample around point to estimate distance
        for r in np.linspace(resolution, resolution * 10, 10):
            for angle in np.linspace(0, 2*np.pi, 16):
                px = x + r * np.cos(angle)
                py = y + r * np.sin(angle)
                if region(px, py):
                    return r

        return float('inf')

    def _merge_coupling_zones(self, zones: List[CouplingZone], merge_distance: float) -> List[CouplingZone]:
        """Merge nearby coupling zones"""
        if len(zones) <= 1:
            return zones

        merged = []
        used = set()

        for i, zone1 in enumerate(zones):
            if i in used:
                continue

            # Find all zones within merge distance
            cluster = [zone1]
            for j, zone2 in enumerate(zones[i+1:], start=i+1):
                if j in used:
                    continue

                dist = np.sqrt((zone1.center[0] - zone2.center[0])**2 +
                             (zone1.center[1] - zone2.center[1])**2)

                if dist < merge_distance:
                    cluster.append(zone2)
                    used.add(j)

            # Create merged zone
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                # Average properties
                center_x = np.mean([z.center[0] for z in cluster])
                center_y = np.mean([z.center[1] for z in cluster])
                max_radius = max([z.radius for z in cluster])
                avg_strength = np.mean([z.coupling_strength for z in cluster])

                # Union of conductors
                all_conductors = []
                for z in cluster:
                    all_conductors.extend(z.conductors)
                unique_conductors = list(set(all_conductors))

                merged.append(CouplingZone(
                    center=(center_x, center_y),
                    radius=max_radius,
                    coupling_strength=avg_strength,
                    conductors=unique_conductors,
                    coupling_type='capacitive'
                ))

        return merged


class ReturnPathAnalyzer:
    """
    Analyze current return path quality.

    Good return paths:
    - Low impedance
    - Short length
    - Close to signal path
    """

    @staticmethod
    def evaluate_return_path(signal_path: List[Tuple[float, float]],
                            ground_result: ConductionResult,
                            current: float = 1.0) -> Dict:
        """
        Evaluate quality of current return path.

        Args:
            signal_path: List of (x, y) points defining signal route
            ground_result: ConductionResult for ground plane/return
            current: Signal current magnitude [A]

        Returns:
            Dictionary with return path metrics
        """
        # Sample points along signal path
        path_voltages = []
        path_currents = []

        for x, y in signal_path:
            # Get ground potential at this location
            i = np.searchsorted(ground_result.grid_x, x) - 1
            j = np.searchsorted(ground_result.grid_y, y) - 1
            i = np.clip(i, 0, len(ground_result.grid_x) - 2)
            j = np.clip(j, 0, len(ground_result.grid_y) - 2)

            V = ground_result.potential[i, j]
            J_x, J_y = ground_result.current_density(x, y)
            J_mag = np.sqrt(J_x**2 + J_y**2)

            path_voltages.append(V)
            path_currents.append(J_mag)

        # Calculate metrics
        voltage_drop = max(path_voltages) - min(path_voltages)
        avg_current_density = np.mean(path_currents)
        max_current_density = max(path_currents)

        # Estimate return impedance
        if current > 1e-10:
            return_impedance = voltage_drop / current
        else:
            return_impedance = float('inf')

        # Path length
        path_length = 0.0
        for i in range(len(signal_path) - 1):
            dx = signal_path[i+1][0] - signal_path[i][0]
            dy = signal_path[i+1][1] - signal_path[i][1]
            path_length += np.sqrt(dx**2 + dy**2)

        return {
            'voltage_drop': voltage_drop,
            'return_impedance': return_impedance,
            'avg_current_density': avg_current_density,
            'max_current_density': max_current_density,
            'path_length': path_length,
            'quality_score': ReturnPathAnalyzer._compute_quality_score(
                voltage_drop, return_impedance, path_length
            )
        }

    @staticmethod
    def _compute_quality_score(voltage_drop: float,
                              impedance: float,
                              path_length: float) -> float:
        """
        Compute return path quality score (0-1, higher is better).

        Good return path has:
        - Low voltage drop
        - Low impedance
        - Short path
        """
        # Normalize each metric (assuming typical ranges)
        v_score = max(0, 1 - voltage_drop / 1.0)  # 1V is "bad"
        z_score = max(0, 1 - impedance / 1.0)      # 1Ω is "bad"
        l_score = max(0, 1 - path_length / 1.0)    # 1m is "bad"

        # Weighted average
        quality = 0.4 * v_score + 0.4 * z_score + 0.2 * l_score

        return quality


class GroundImpedanceMap:
    """
    Create ground impedance map for routing optimization.

    Low impedance zones → Preferred routing regions
    High impedance zones → Avoid or add return paths
    """

    @staticmethod
    def from_conduction_result(conduction_result: ConductionResult,
                              reference_current: float = 1.0) -> Callable[[float, float], float]:
        """
        Create ground impedance map from conduction analysis.

        Args:
            conduction_result: ConductionResult for ground system
            reference_current: Current for impedance calculation [A]

        Returns:
            Function (x, y) -> impedance [Ω]
        """
        # Compute voltage gradient magnitude as proxy for impedance
        n = len(conduction_result.grid_x)
        dx = conduction_result.grid_x[1] - conduction_result.grid_x[0]
        dy = conduction_result.grid_y[1] - conduction_result.grid_y[0]

        # Voltage gradients
        dV_dx = np.zeros((n, n))
        dV_dy = np.zeros((n, n))

        for i in range(1, n-1):
            for j in range(1, n-1):
                dV_dx[i, j] = (conduction_result.potential[i+1, j] - conduction_result.potential[i-1, j]) / (2 * dx)
                dV_dy[i, j] = (conduction_result.potential[i, j+1] - conduction_result.potential[i, j-1]) / (2 * dy)

        grad_V_mag = np.sqrt(dV_dx**2 + dV_dy**2)

        # Normalize to impedance (V/m per A → Ω)
        if reference_current > 0:
            impedance_map = grad_V_mag / reference_current
        else:
            impedance_map = grad_V_mag

        def impedance_func(x: float, y: float) -> float:
            """Interpolate impedance at point"""
            grid_x = conduction_result.grid_x
            grid_y = conduction_result.grid_y

            if x < grid_x[0] or x > grid_x[-1] or y < grid_y[0] or y > grid_y[-1]:
                return float('inf')  # Outside domain

            i = np.searchsorted(grid_x, x) - 1
            j = np.searchsorted(grid_y, y) - 1
            i = np.clip(i, 0, n - 2)
            j = np.clip(j, 0, n - 2)

            return impedance_map[i, j]

        return impedance_func

    @staticmethod
    def identify_low_impedance_zones(impedance_func: Callable[[float, float], float],
                                    domain_x: Tuple[float, float],
                                    domain_y: Tuple[float, float],
                                    resolution: int = 50,
                                    threshold_percentile: float = 25) -> List[Tuple[float, float, float]]:
        """
        Identify low-impedance zones (preferred routing areas).

        Args:
            impedance_func: Impedance function from from_conduction_result
            domain_x: X domain bounds
            domain_y: Y domain bounds
            resolution: Sampling resolution
            threshold_percentile: Percentile for "low" impedance

        Returns:
            List of (x, y, impedance) for low-impedance points
        """
        x_vals = np.linspace(domain_x[0], domain_x[1], resolution)
        y_vals = np.linspace(domain_y[0], domain_y[1], resolution)

        impedances = []
        points = []

        for x in x_vals:
            for y in y_vals:
                Z = impedance_func(x, y)
                if np.isfinite(Z):
                    impedances.append(Z)
                    points.append((x, y, Z))

        # Find threshold
        if impedances:
            threshold = np.percentile(impedances, threshold_percentile)
            low_z_zones = [(x, y, Z) for x, y, Z in points if Z <= threshold]
            return low_z_zones
        else:
            return []


class CrosstalkAnalyzer:
    """
    Analyze crosstalk risk between signal paths.

    Uses proximity and field coupling as quasi-static proxy.
    """

    @staticmethod
    def evaluate_crosstalk(path1: List[Tuple[float, float]],
                          path2: List[Tuple[float, float]],
                          min_separation: float = 0.001) -> Dict:
        """
        Evaluate crosstalk risk between two paths.

        Args:
            path1: First signal path
            path2: Second signal path
            min_separation: Minimum safe separation [m]

        Returns:
            Dictionary with crosstalk metrics
        """
        # Find minimum separation
        min_dist = float('inf')
        closest_pair = None

        for p1 in path1:
            for p2 in path2:
                dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (p1, p2)

        # Calculate parallel length (segments within 2x min_separation)
        parallel_length = 0.0
        for i in range(len(path1) - 1):
            seg1_start = path1[i]
            seg1_end = path1[i + 1]
            seg1_mid = ((seg1_start[0] + seg1_end[0]) / 2,
                       (seg1_start[1] + seg1_end[1]) / 2)

            for j in range(len(path2) - 1):
                seg2_start = path2[j]
                seg2_end = path2[j + 1]
                seg2_mid = ((seg2_start[0] + seg2_end[0]) / 2,
                           (seg2_start[1] + seg2_end[1]) / 2)

                # Distance between segment midpoints
                mid_dist = np.sqrt((seg1_mid[0] - seg2_mid[0])**2 +
                                 (seg1_mid[1] - seg2_mid[1])**2)

                if mid_dist < 2 * min_separation:
                    # Segments are close - add to parallel length
                    seg_len = np.sqrt((seg1_end[0] - seg1_start[0])**2 +
                                    (seg1_end[1] - seg1_start[1])**2)
                    parallel_length += seg_len

        # Crosstalk risk score (0-1, higher is worse)
        separation_risk = max(0, 1 - min_dist / min_separation)
        parallel_risk = min(1, parallel_length / 0.1)  # 0.1m parallel is "bad"

        crosstalk_risk = 0.6 * separation_risk + 0.4 * parallel_risk

        return {
            'min_separation': min_dist,
            'parallel_length': parallel_length,
            'separation_risk': separation_risk,
            'parallel_risk': parallel_risk,
            'crosstalk_risk': crosstalk_risk,
            'closest_points': closest_pair,
        }
