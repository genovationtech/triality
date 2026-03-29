"""
Aerodynamic load integration from surface pressure distributions.

Integrates pressure distributions to compute:
- Total forces (lift, drag, side force)
- Moments (pitch, yaw, roll)
- Center of pressure
- Load distributions for structural analysis
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class PressurePanel:
    """Surface panel with pressure distribution"""
    centroid: np.ndarray  # Panel centroid position [x, y, z] [m]
    area: float  # Panel area [m²]
    normal: np.ndarray  # Unit normal vector (outward)
    pressure: float  # Surface pressure [Pa]


@dataclass
class AeroLoads:
    """Integrated aerodynamic loads"""
    force: np.ndarray  # Total force vector [N]
    moment: np.ndarray  # Total moment vector about origin [N·m]
    center_of_pressure: np.ndarray  # Center of pressure [m]

    # Force components in body frame
    @property
    def axial_force(self) -> float:
        """Axial force (along x-axis)"""
        return self.force[0]

    @property
    def side_force(self) -> float:
        """Side force (along y-axis)"""
        return self.force[1]

    @property
    def normal_force(self) -> float:
        """Normal force (along z-axis)"""
        return self.force[2]

    # Moment components
    @property
    def rolling_moment(self) -> float:
        """Rolling moment (about x-axis)"""
        return self.moment[0]

    @property
    def pitching_moment(self) -> float:
        """Pitching moment (about y-axis)"""
        return self.moment[1]

    @property
    def yawing_moment(self) -> float:
        """Yawing moment (about z-axis)"""
        return self.moment[2]


class LoadIntegration:
    """
    Integrate surface pressures to forces and moments.
    """

    @staticmethod
    def integrate_panels(panels: List[PressurePanel],
                        p_ref: float = 101325.0) -> AeroLoads:
        """
        Integrate pressure over all panels.

        Force: F = Σ (p - p_ref) · n · A
        Moment: M = Σ r × F_panel

        Args:
            panels: List of pressure panels
            p_ref: Reference pressure (freestream) [Pa]

        Returns:
            Integrated loads
        """
        total_force = np.zeros(3)
        total_moment = np.zeros(3)

        # Weighted centroid for center of pressure
        force_magnitude_sum = 0.0
        cop_numerator = np.zeros(3)

        for panel in panels:
            # Pressure force on panel
            dp = panel.pressure - p_ref
            dF = dp * panel.area * panel.normal

            total_force += dF

            # Moment about origin
            dM = np.cross(panel.centroid, dF)
            total_moment += dM

            # Accumulate for center of pressure
            dF_mag = np.linalg.norm(dF)
            force_magnitude_sum += dF_mag
            cop_numerator += dF_mag * panel.centroid

        # Center of pressure
        if force_magnitude_sum > 1e-10:
            cop = cop_numerator / force_magnitude_sum
        else:
            cop = np.zeros(3)

        return AeroLoads(
            force=total_force,
            moment=total_moment,
            center_of_pressure=cop
        )

    @staticmethod
    def force_coefficients(loads: AeroLoads, q_inf: float, S_ref: float,
                          alpha: float = 0.0, beta: float = 0.0) -> dict:
        """
        Compute dimensionless force and moment coefficients.

        Args:
            loads: Integrated loads
            q_inf: Dynamic pressure [Pa]
            S_ref: Reference area [m²]
            alpha: Angle of attack [rad]
            beta: Sideslip angle [rad]

        Returns:
            Dictionary of coefficients
        """
        # Force coefficients
        CA = loads.axial_force / (q_inf * S_ref)
        CY = loads.side_force / (q_inf * S_ref)
        CN = loads.normal_force / (q_inf * S_ref)

        # Transform to wind axes (lift, drag, side)
        # For small angles approximation:
        CL = CN * np.cos(alpha) - CA * np.sin(alpha)
        CD = CA * np.cos(alpha) + CN * np.sin(alpha)

        return {
            'CA': CA,  # Axial force coefficient
            'CN': CN,  # Normal force coefficient
            'CY': CY,  # Side force coefficient
            'CL': CL,  # Lift coefficient
            'CD': CD,  # Drag coefficient
        }

    @staticmethod
    def moment_coefficients(loads: AeroLoads, q_inf: float, S_ref: float,
                           L_ref: float) -> dict:
        """
        Compute dimensionless moment coefficients.

        Args:
            loads: Integrated loads
            q_inf: Dynamic pressure [Pa]
            S_ref: Reference area [m²]
            L_ref: Reference length [m]

        Returns:
            Dictionary of coefficients
        """
        Cl = loads.rolling_moment / (q_inf * S_ref * L_ref)
        Cm = loads.pitching_moment / (q_inf * S_ref * L_ref)
        Cn = loads.yawing_moment / (q_inf * S_ref * L_ref)

        return {
            'Cl': Cl,  # Rolling moment coefficient
            'Cm': Cm,  # Pitching moment coefficient
            'Cn': Cn,  # Yawing moment coefficient
        }


class StructuralLoadDistribution:
    """
    Convert aero loads to structural load distributions for stress analysis.
    """

    @staticmethod
    def panel_pressure_to_nodal_force(panel: PressurePanel, nodes: np.ndarray) -> np.ndarray:
        """
        Distribute panel pressure to corner nodes.

        Assumes uniform pressure over panel, distributes equally to nodes.

        Args:
            panel: Pressure panel
            nodes: Node positions (N×3 array)

        Returns:
            Forces at each node (N×3 array)
        """
        n_nodes = len(nodes)

        # Total force on panel
        F_total = (panel.pressure * panel.area) * panel.normal

        # Distribute equally to nodes
        F_nodal = np.tile(F_total / n_nodes, (n_nodes, 1))

        return F_nodal

    @staticmethod
    def shear_and_moment_distribution(x_array: np.ndarray, load_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute shear and moment distributions from distributed load.

        For beam with distributed load q(x):
            V(x) = ∫ q(x) dx  (shear)
            M(x) = ∫ V(x) dx  (moment)

        Args:
            x_array: Position along beam [m]
            load_array: Distributed load [N/m]

        Returns:
            (shear_array, moment_array)
        """
        # Integrate load to get shear
        shear_array = np.zeros_like(load_array)
        for i in range(1, len(x_array)):
            dx = x_array[i] - x_array[i-1]
            shear_array[i] = shear_array[i-1] + load_array[i] * dx

        # Integrate shear to get moment
        moment_array = np.zeros_like(shear_array)
        for i in range(1, len(x_array)):
            dx = x_array[i] - x_array[i-1]
            moment_array[i] = moment_array[i-1] + shear_array[i] * dx

        return shear_array, moment_array


class PressureCoefficientAnalysis:
    """
    Analyze pressure coefficient distributions.
    """

    @staticmethod
    def Cp_from_pressure(p: float, p_inf: float, q_inf: float) -> float:
        """
        Compute pressure coefficient.

        Cp = (p - p_∞) / q_∞

        Args:
            p: Local pressure [Pa]
            p_inf: Freestream pressure [Pa]
            q_inf: Dynamic pressure [Pa]

        Returns:
            Pressure coefficient
        """
        return (p - p_inf) / q_inf

    @staticmethod
    def critical_pressure_coefficient(M_inf: float, gamma: float = 1.4) -> float:
        """
        Critical pressure coefficient (sonic condition at surface).

        Cp* = (2/(γ·M²)) · [((2/(γ+1))·(1 + (γ-1)/2·M²))^(γ/(γ-1)) - 1]

        Args:
            M_inf: Freestream Mach number
            gamma: Specific heat ratio

        Returns:
            Critical Cp
        """
        M2 = M_inf**2

        bracket_term = (2 / (gamma + 1)) * (1 + (gamma - 1)/2 * M2)
        power_term = bracket_term ** (gamma / (gamma - 1))

        Cp_crit = (2 / (gamma * M2)) * (power_term - 1)

        return Cp_crit

    @staticmethod
    def suction_peak_magnitude(Cp_array: np.ndarray) -> float:
        """
        Find magnitude of suction peak (most negative Cp).

        Args:
            Cp_array: Pressure coefficient distribution

        Returns:
            Magnitude of suction peak (positive value)
        """
        return abs(np.min(Cp_array))


class AeroelasticLoads:
    """
    Simplified aeroelastic load coupling (no full FSI).

    Accounts for structural deformation effects on loads.
    """

    @staticmethod
    def effective_angle_of_attack(alpha_rigid: float, twist: float, deflection_slope: float) -> float:
        """
        Effective angle of attack accounting for structural deformation.

        α_eff = α_rigid + twist + dw/dx

        Args:
            alpha_rigid: Rigid-body angle of attack [rad]
            twist: Structural twist [rad]
            deflection_slope: Slope of bending deflection [rad]

        Returns:
            Effective angle of attack [rad]
        """
        return alpha_rigid + twist + deflection_slope

    @staticmethod
    def divergence_dynamic_pressure(K_theta: float, dCm_dalpha: float, c: float) -> float:
        """
        Estimate divergence dynamic pressure (simplified).

        q_D = K_θ / (c · dCm/dα)

        where K_θ is torsional stiffness.

        Args:
            K_theta: Torsional stiffness [N·m/rad]
            dCm_dalpha: Pitching moment curve slope [1/rad]
            c: Chord length [m]

        Returns:
            Divergence dynamic pressure [Pa]
        """
        if abs(dCm_dalpha) < 1e-10:
            return float('inf')  # No divergence

        q_D = K_theta / (c * abs(dCm_dalpha))

        return q_D
