"""Derived Quantities and Analysis Tools

Utilities for analyzing electrostatic and conduction results:
- Electric field analysis
- Field magnitude and gradients
- Hotspot detection
- High-field zone identification
- EMI risk assessment
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

from .field_solver import ElectrostaticResult
from .conduction import ConductionResult


@dataclass
class ElectricField:
    """Electric field E = -∇V analysis"""

    @staticmethod
    def from_result(result: ElectrostaticResult) -> 'ElectricFieldData':
        """
        Compute electric field components on entire grid.

        Returns:
            ElectricFieldData with E_x, E_y, |E|
        """
        n = len(result.grid_x)
        dx = result.grid_x[1] - result.grid_x[0]
        dy = result.grid_y[1] - result.grid_y[0]

        E_x = np.zeros((n, n))
        E_y = np.zeros((n, n))

        # Compute gradients
        for i in range(n):
            for j in range(n):
                if i > 0 and i < n - 1:
                    dV_dx = (result.potential[i + 1, j] - result.potential[i - 1, j]) / (2 * dx)
                elif i == 0:
                    dV_dx = (result.potential[i + 1, j] - result.potential[i, j]) / dx
                else:
                    dV_dx = (result.potential[i, j] - result.potential[i - 1, j]) / dx

                if j > 0 and j < n - 1:
                    dV_dy = (result.potential[i, j + 1] - result.potential[i, j - 1]) / (2 * dy)
                elif j == 0:
                    dV_dy = (result.potential[i, j + 1] - result.potential[i, j]) / dy
                else:
                    dV_dy = (result.potential[i, j] - result.potential[i, j - 1]) / dy

                # E = -∇V
                E_x[i, j] = -dV_dx
                E_y[i, j] = -dV_dy

        E_mag = np.sqrt(E_x**2 + E_y**2)

        return ElectricFieldData(
            E_x=E_x,
            E_y=E_y,
            E_magnitude=E_mag,
            grid_x=result.grid_x,
            grid_y=result.grid_y,
        )


@dataclass
class ElectricFieldData:
    """Electric field data on grid"""
    E_x: np.ndarray          # E_x[i, j] [V/m]
    E_y: np.ndarray          # E_y[i, j] [V/m]
    E_magnitude: np.ndarray  # |E|[i, j] [V/m]
    grid_x: np.ndarray       # x coordinates
    grid_y: np.ndarray       # y coordinates

    def max_field(self) -> Tuple[float, float, float]:
        """
        Find maximum field location.

        Returns:
            (x, y, |E|_max)
        """
        i_max, j_max = np.unravel_index(np.argmax(self.E_magnitude), self.E_magnitude.shape)
        x = self.grid_x[i_max]
        y = self.grid_y[j_max]
        return (x, y, self.E_magnitude[i_max, j_max])

    def field_lines(self, start_points: List[Tuple[float, float]], max_steps=1000) -> List[List[Tuple[float, float]]]:
        """
        Trace electric field lines from start points.

        Args:
            start_points: List of (x, y) starting positions
            max_steps: Maximum integration steps

        Returns:
            List of field line paths
        """
        dx = self.grid_x[1] - self.grid_x[0]
        step_size = 0.1 * min(dx, self.grid_y[1] - self.grid_y[0])

        field_lines = []

        for x0, y0 in start_points:
            line = [(x0, y0)]
            x, y = x0, y0

            for _ in range(max_steps):
                # Interpolate E field at current position
                i = np.searchsorted(self.grid_x, x) - 1
                j = np.searchsorted(self.grid_y, y) - 1

                if i < 0 or i >= len(self.grid_x) - 1 or j < 0 or j >= len(self.grid_y) - 1:
                    break

                E_x_val = self.E_x[i, j]
                E_y_val = self.E_y[i, j]
                E_mag = np.sqrt(E_x_val**2 + E_y_val**2)

                if E_mag < 1e-10:
                    break

                # Move in field direction
                x += step_size * E_x_val / E_mag
                y += step_size * E_y_val / E_mag

                line.append((x, y))

            field_lines.append(line)

        return field_lines


class FieldMagnitude:
    """Field magnitude analysis and statistics"""

    @staticmethod
    def analyze(field_data: ElectricFieldData) -> 'FieldStatistics':
        """
        Compute field statistics.

        Returns:
            FieldStatistics with min, max, mean, percentiles
        """
        E_mag = field_data.E_magnitude

        return FieldStatistics(
            min=np.min(E_mag),
            max=np.max(E_mag),
            mean=np.mean(E_mag),
            std=np.std(E_mag),
            percentile_50=np.percentile(E_mag, 50),
            percentile_90=np.percentile(E_mag, 90),
            percentile_95=np.percentile(E_mag, 95),
            percentile_99=np.percentile(E_mag, 99),
        )


@dataclass
class FieldStatistics:
    """Field magnitude statistics"""
    min: float
    max: float
    mean: float
    std: float
    percentile_50: float
    percentile_90: float
    percentile_95: float
    percentile_99: float

    def __repr__(self):
        return (
            f"FieldStats(min={self.min:.2e}, max={self.max:.2e}, "
            f"mean={self.mean:.2e}, p99={self.percentile_99:.2e})"
        )


class GradientAnalysis:
    """Gradient analysis for EMI risk and field crowding"""

    @staticmethod
    def field_gradient(field_data: ElectricFieldData) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spatial gradient of field magnitude: ∇|E|

        Rapid gradients indicate:
        - Field crowding zones
        - Sharp corners / edges
        - Potential EMI sources

        Returns:
            (d|E|/dx, d|E|/dy) gradient components
        """
        dx = field_data.grid_x[1] - field_data.grid_x[0]
        dy = field_data.grid_y[1] - field_data.grid_y[0]

        E_mag = field_data.E_magnitude
        n = len(field_data.grid_x)

        grad_x = np.zeros_like(E_mag)
        grad_y = np.zeros_like(E_mag)

        # Central differences
        grad_x[1:-1, :] = (E_mag[2:, :] - E_mag[:-2, :]) / (2 * dx)
        grad_y[:, 1:-1] = (E_mag[:, 2:] - E_mag[:, :-2]) / (2 * dy)

        # Boundaries (one-sided)
        grad_x[0, :] = (E_mag[1, :] - E_mag[0, :]) / dx
        grad_x[-1, :] = (E_mag[-1, :] - E_mag[-2, :]) / dx
        grad_y[:, 0] = (E_mag[:, 1] - E_mag[:, 0]) / dy
        grad_y[:, -1] = (E_mag[:, -1] - E_mag[:, -2]) / dy

        return grad_x, grad_y

    @staticmethod
    def high_gradient_zones(field_data: ElectricFieldData,
                           threshold_percentile: float = 95) -> List[Tuple[float, float, float]]:
        """
        Find zones with high field gradients (EMI risk zones).

        Args:
            field_data: Electric field data
            threshold_percentile: Percentile for "high" gradient (default 95%)

        Returns:
            List of (x, y, |∇|E||) for high-gradient points
        """
        grad_x, grad_y = GradientAnalysis.field_gradient(field_data)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        threshold = np.percentile(grad_mag, threshold_percentile)

        high_grad_zones = []
        for i in range(len(field_data.grid_x)):
            for j in range(len(field_data.grid_y)):
                if grad_mag[i, j] >= threshold:
                    x = field_data.grid_x[i]
                    y = field_data.grid_y[j]
                    high_grad_zones.append((x, y, grad_mag[i, j]))

        return high_grad_zones


class HotspotDetector:
    """Hotspot detection for thermal and electrical hazards"""

    @staticmethod
    def detect_electrical(field_data: ElectricFieldData,
                         dielectric_strength: float) -> List[Tuple[float, float, float, float]]:
        """
        Detect regions at risk of dielectric breakdown.

        Args:
            field_data: Electric field data
            dielectric_strength: Material breakdown field [V/m]
                Examples:
                - Air: 3e6 V/m
                - FR4: 20e6 V/m
                - Polyimide: 280e6 V/m

        Returns:
            List of (x, y, |E|, safety_factor) where safety_factor = E/E_breakdown
        """
        E_mag = field_data.E_magnitude
        breakdown_risk = []

        for i in range(len(field_data.grid_x)):
            for j in range(len(field_data.grid_y)):
                if E_mag[i, j] > 0.5 * dielectric_strength:  # 50% threshold
                    x = field_data.grid_x[i]
                    y = field_data.grid_y[j]
                    safety_factor = E_mag[i, j] / dielectric_strength
                    breakdown_risk.append((x, y, E_mag[i, j], safety_factor))

        return breakdown_risk

    @staticmethod
    def detect_thermal(conduction_result: ConductionResult,
                      max_power_density: float) -> List[Tuple[float, float, float, Optional[str]]]:
        """
        Detect thermal hotspots in conductive media.

        Args:
            conduction_result: Conduction solve result
            max_power_density: Threshold power density [W/m³]

        Returns:
            List of (x, y, P, material) for hotspot locations
        """
        return conduction_result.find_hotspots(max_power_density)

    @staticmethod
    def clearance_check(field_data: ElectricFieldData,
                       conductor_regions: List[Callable[[float, float], bool]],
                       min_clearance: float,
                       field_threshold: float) -> List[Tuple[float, float, float, int]]:
        """
        Check clearance requirements between conductors.

        Identifies points where high field exists near multiple conductors,
        indicating insufficient spacing.

        Args:
            field_data: Electric field data
            conductor_regions: List of region functions defining conductors
            min_clearance: Minimum required clearance [m]
            field_threshold: Field threshold for concern [V/m]

        Returns:
            List of (x, y, |E|, num_nearby_conductors) for clearance violations
        """
        violations = []

        for i in range(len(field_data.grid_x)):
            for j in range(len(field_data.grid_y)):
                x = field_data.grid_x[i]
                y = field_data.grid_y[j]
                E_mag = field_data.E_magnitude[i, j]

                if E_mag < field_threshold:
                    continue

                # Count nearby conductors
                nearby_count = 0
                for region in conductor_regions:
                    # Check if any conductor is within min_clearance
                    # Simplified: check grid points within clearance radius
                    for di in range(-2, 3):
                        for dj in range(-2, 3):
                            ii = i + di
                            jj = j + dj
                            if 0 <= ii < len(field_data.grid_x) and 0 <= jj < len(field_data.grid_y):
                                x_check = field_data.grid_x[ii]
                                y_check = field_data.grid_y[jj]
                                dist = np.sqrt((x - x_check)**2 + (y - y_check)**2)
                                if dist < min_clearance and region(x_check, y_check):
                                    nearby_count += 1
                                    break

                if nearby_count >= 2:
                    violations.append((x, y, E_mag, nearby_count))

        return violations
