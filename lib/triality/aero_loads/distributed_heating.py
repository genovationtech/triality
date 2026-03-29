"""
Distributed aerodynamic heating for hypersonic vehicles.

Engineering correlations for surface heating distribution on:
- Flat plates (laminar and turbulent)
- Wedges
- Cones
- Blunt bodies
- Leading edges

Goes beyond stagnation point heating to provide full surface heating maps.
"""

import numpy as np
from typing import Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class HeatingConditions:
    """Flow conditions for heating calculations"""
    velocity: float  # m/s
    density: float  # kg/m³
    temperature: float  # K
    pressure: float  # Pa

    @property
    def enthalpy(self) -> float:
        """Total enthalpy ≈ ½V²"""
        return 0.5 * self.velocity**2


class DistributedHeating:
    """
    Engineering correlations for distributed surface heating.
    """

    @staticmethod
    def flat_plate_laminar(x: float, conditions: HeatingConditions,
                          T_wall: float = 300.0, Pr: float = 0.71) -> float:
        """
        Laminar flat plate heating distribution.

        Reference temperature method:
            q(x) = 0.332 · √(ρ_e·μ_e·u_e³/x) · Pr^(-2/3) · (h_aw - h_w)

        Simplified:
            q(x) ∝ x^(-1/2)

        Args:
            x: Distance from leading edge [m]
            conditions: Freestream conditions
            T_wall: Wall temperature [K]
            Pr: Prandtl number

        Returns:
            Heat flux [W/m²]
        """
        if x <= 0:
            return 0.0

        # Reference temperature (approximation)
        T_ref = 0.5 * (conditions.temperature + T_wall) + 0.22 * (conditions.temperature - T_wall)

        # Viscosity at reference temperature (Sutherland's law approximation)
        mu_ref = 1.458e-6 * T_ref**1.5 / (T_ref + 110.4)

        # Reynolds number
        Re_x = conditions.density * conditions.velocity * x / mu_ref

        # Local heat transfer coefficient correlation
        # St = 0.332 / √(Re_x) · Pr^(-2/3)  (Stanton number)
        St = 0.332 / np.sqrt(Re_x) * Pr**(-2/3)

        # Heat flux
        c_p = 1005  # Specific heat [J/(kg·K)]
        h_aw = c_p * (conditions.temperature + 0.5 * conditions.velocity**2 / c_p)  # Adiabatic wall enthalpy
        h_w = c_p * T_wall

        q = conditions.density * conditions.velocity * St * (h_aw - h_w)

        return q

    @staticmethod
    def flat_plate_turbulent(x: float, conditions: HeatingConditions,
                            T_wall: float = 300.0, Pr: float = 0.71,
                            x_transition: float = 0.0) -> float:
        """
        Turbulent flat plate heating distribution.

        Correlation:
            St = 0.0296 / Re_x^0.2 · Pr^(-2/3)

        Args:
            x: Distance from leading edge [m]
            conditions: Freestream conditions
            T_wall: Wall temperature [K]
            Pr: Prandtl number
            x_transition: Transition location [m]

        Returns:
            Heat flux [W/m²]
        """
        if x <= x_transition:
            return DistributedHeating.flat_plate_laminar(x, conditions, T_wall, Pr)

        T_ref = 0.5 * (conditions.temperature + T_wall)
        mu_ref = 1.458e-6 * T_ref**1.5 / (T_ref + 110.4)

        Re_x = conditions.density * conditions.velocity * x / mu_ref

        # Turbulent Stanton number
        St = 0.0296 / Re_x**0.2 * Pr**(-2/3)

        c_p = 1005
        h_aw = c_p * (conditions.temperature + 0.5 * conditions.velocity**2 / c_p)
        h_w = c_p * T_wall

        q = conditions.density * conditions.velocity * St * (h_aw - h_w)

        return q

    @staticmethod
    def wedge_heating(x: float, theta_wedge: float, conditions: HeatingConditions,
                     T_wall: float = 300.0) -> float:
        """
        Heating on wedge surface accounting for oblique shock.

        Args:
            x: Distance along surface [m]
            theta_wedge: Wedge half-angle [rad]
            conditions: Freestream conditions
            T_wall: Wall temperature [K]

        Returns:
            Heat flux [W/m²]
        """
        # Oblique shock relations (simplified)
        # Pressure increases, temperature increases
        # Use post-shock conditions for flat plate correlation

        # Shock angle approximation (weak shock)
        M_inf = conditions.velocity / np.sqrt(1.4 * 287 * conditions.temperature)
        beta = np.arcsin(1 / M_inf)  # Approximation for weak shock

        # Post-shock conditions (simplified)
        p2_p1 = 1 + 2 * 1.4 / (1.4 + 1) * (M_inf**2 * np.sin(beta)**2 - 1)
        T2_T1 = p2_p1  # Approximation

        post_shock = HeatingConditions(
            velocity=conditions.velocity * 0.8,  # Reduced velocity
            density=conditions.density * 1.5,  # Increased density
            temperature=conditions.temperature * T2_T1,
            pressure=conditions.pressure * p2_p1
        )

        # Use flat plate correlation with post-shock conditions
        return DistributedHeating.flat_plate_laminar(x, post_shock, T_wall)

    @staticmethod
    def cone_heating(x: float, theta_c: float, conditions: HeatingConditions,
                    T_wall: float = 300.0) -> float:
        """
        Heating on cone surface.

        Uses empirical correlation:
            q_cone = q_flat_plate · √3

        Args:
            x: Distance from apex [m]
            theta_c: Cone half-angle [rad]
            conditions: Freestream conditions
            T_wall: Wall temperature [K]

        Returns:
            Heat flux [W/m²]
        """
        # Cone heating is approximately √3 times flat plate
        q_flat = DistributedHeating.flat_plate_laminar(x, conditions, T_wall)

        return q_flat * np.sqrt(3.0)

    @staticmethod
    def leading_edge_heating(s: float, R_n: float, conditions: HeatingConditions,
                            T_wall: float = 300.0) -> float:
        """
        Heating distribution along cylindrical leading edge.

        Near stagnation point:
            q(s) = q_stag · f(s/R_n)

        where f(s/R_n) is empirical distribution function.

        Args:
            s: Arc length from stagnation [m]
            R_n: Nose radius [m]
            conditions: Freestream conditions
            T_wall: Wall temperature [K]

        Returns:
            Heat flux [W/m²]
        """
        # Stagnation point heating (Fay-Riddell)
        C = 1.83e-4  # SI units
        rho = conditions.density
        V = conditions.velocity

        q_stag = C * np.sqrt(rho / R_n) * V**3

        # Distribution function (empirical)
        # Decreases away from stagnation point
        s_over_R = s / R_n

        if s_over_R < 0.5:
            # Near stagnation: gradual decrease
            f = np.exp(-0.5 * s_over_R**2)
        else:
            # Far from stagnation: transition to flat plate
            f = 0.6 * np.exp(-s_over_R)

        return q_stag * f


class HeatLoadIntegration:
    """
    Integrate distributed heating over surfaces for total heat load.
    """

    @staticmethod
    def integrate_1d(x_array: np.ndarray, q_array: np.ndarray, width: float = 1.0) -> float:
        """
        Integrate heat flux over 1D surface.

        Q = ∫ q(x) · w · dx

        Args:
            x_array: Position array [m]
            q_array: Heat flux array [W/m²]
            width: Surface width [m]

        Returns:
            Total heat load [W]
        """
        Q_total = np.trapezoid(q_array, x_array) * width

        return Q_total

    @staticmethod
    def integrate_axisymmetric(x_array: np.ndarray, q_array: np.ndarray,
                              r_array: np.ndarray) -> float:
        """
        Integrate heat flux over axisymmetric body.

        Q = ∫ q(x) · 2π·r(x) · dx

        Args:
            x_array: Axial position [m]
            q_array: Heat flux [W/m²]
            r_array: Local radius [m]

        Returns:
            Total heat load [W]
        """
        # Differential area: dA = 2π·r·√(1 + (dr/dx)²)·dx
        # Simplified: dA ≈ 2π·r·dx

        integrand = q_array * 2 * np.pi * r_array

        Q_total = np.trapezoid(integrand, x_array)

        return Q_total

    @staticmethod
    def peak_location(x_array: np.ndarray, q_array: np.ndarray) -> Tuple[float, float]:
        """
        Find peak heating location and magnitude.

        Args:
            x_array: Position array
            q_array: Heat flux array

        Returns:
            (x_peak, q_peak)
        """
        idx_peak = np.argmax(q_array)

        return x_array[idx_peak], q_array[idx_peak]


class SurfaceHeatingMap:
    """
    Generate 2D surface heating map for vehicle.
    """

    def __init__(self, conditions: HeatingConditions):
        self.conditions = conditions
        self.heating_map = {}

    def add_panel(self, panel_id: str, x_positions: np.ndarray,
                 heating_func: Callable[[np.ndarray], np.ndarray]):
        """
        Add panel heating distribution to map.

        Args:
            panel_id: Panel identifier
            x_positions: Position array
            heating_func: Function that computes q(x)
        """
        q_values = heating_func(x_positions)

        self.heating_map[panel_id] = {
            'x': x_positions,
            'q': q_values,
            'q_max': np.max(q_values),
            'q_avg': np.mean(q_values)
        }

    def get_peak_heating(self) -> Tuple[str, float, float]:
        """
        Find panel and location with peak heating.

        Returns:
            (panel_id, x_peak, q_peak)
        """
        max_q = 0.0
        peak_panel = None
        peak_x = 0.0

        for panel_id, data in self.heating_map.items():
            if data['q_max'] > max_q:
                max_q = data['q_max']
                peak_panel = panel_id
                idx = np.argmax(data['q'])
                peak_x = data['x'][idx]

        return peak_panel, peak_x, max_q

    def total_heat_load(self) -> float:
        """
        Sum total heat load over all panels.

        Returns:
            Total heat load [W]
        """
        Q_total = 0.0

        for panel_id, data in self.heating_map.items():
            Q_panel = np.trapezoid(data['q'], data['x'])
            Q_total += Q_panel

        return Q_total
