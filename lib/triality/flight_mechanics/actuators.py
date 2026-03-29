"""
Actuator models for spacecraft and aircraft control.

Includes:
- Reaction Control System (RCS) thrusters
- Reaction wheels (momentum exchange devices)
- Control surfaces (aerodynamic)
- Actuator dynamics (first-order lag, saturation, dead zones)
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


class ActuatorType(Enum):
    """Actuator type enumeration"""
    RCS_THRUSTER = 'rcs_thruster'
    REACTION_WHEEL = 'reaction_wheel'
    CONTROL_SURFACE = 'control_surface'
    MAGNETORQUER = 'magnetorquer'


@dataclass
class RCSThruster:
    """
    Reaction Control System thruster model.

    Simple bang-bang thruster with minimum impulse bit.
    """
    position: np.ndarray  # Position in body frame [m]
    direction: np.ndarray  # Thrust direction unit vector in body frame
    max_thrust: float  # Maximum thrust [N]
    min_impulse_bit: float = 0.001  # Minimum impulse bit [N·s]
    specific_impulse: float = 220.0  # Specific impulse [s]

    # State
    is_firing: bool = False
    cumulative_impulse: float = 0.0  # Total impulse delivered [N·s]

    def __post_init__(self):
        """Normalize direction vector"""
        self.direction = self.direction / np.linalg.norm(self.direction)

    def compute_force_torque(self, command: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute force and torque from thruster.

        Args:
            command: Command value [-1, 1] (sign determines direction)
            dt: Time step [s]

        Returns:
            (force_body, torque_body)
        """
        # Bang-bang control: threshold at 0.1
        if abs(command) > 0.1:
            thrust_magnitude = self.max_thrust * np.sign(command)
            self.is_firing = True
        else:
            thrust_magnitude = 0.0
            self.is_firing = False

        # Force in body frame
        force = thrust_magnitude * self.direction

        # Torque: M = r × F
        torque = np.cross(self.position, force)

        # Track impulse
        self.cumulative_impulse += abs(thrust_magnitude) * dt

        return force, torque

    def fuel_consumption(self, dt: float) -> float:
        """
        Compute fuel consumption rate.

        ṁ = F / (I_sp · g₀)

        Args:
            dt: Time step [s]

        Returns:
            Fuel mass consumed [kg]
        """
        if not self.is_firing:
            return 0.0

        g0 = 9.80665  # Standard gravity [m/s²]
        thrust = self.max_thrust if self.is_firing else 0.0
        mdot = thrust / (self.specific_impulse * g0)

        return mdot * dt


@dataclass
class ReactionWheel:
    """
    Reaction wheel (momentum exchange device).

    Stores angular momentum and exchanges it with spacecraft.
    """
    axis: np.ndarray  # Spin axis in body frame (unit vector)
    max_torque: float  # Maximum torque [N·m]
    max_momentum: float  # Maximum storable momentum [N·m·s]
    inertia: float  # Wheel inertia [kg·m²]
    time_constant: float = 0.1  # First-order lag time constant [s]

    # State
    momentum: float = 0.0  # Current stored momentum [N·m·s]
    omega_wheel: float = 0.0  # Wheel spin rate [rad/s]

    def __post_init__(self):
        """Normalize axis"""
        self.axis = self.axis / np.linalg.norm(self.axis)

    def compute_torque(self, command_torque: float, dt: float) -> np.ndarray:
        """
        Compute torque on spacecraft from reaction wheel.

        Args:
            command_torque: Commanded torque [N·m] (positive = increase wheel speed)
            dt: Time step [s]

        Returns:
            Torque on spacecraft body [N·m] (opposite of wheel acceleration)
        """
        # Saturate command
        command_torque = np.clip(command_torque, -self.max_torque, self.max_torque)

        # First-order lag dynamics: T_actual = T_cmd / (τ·s + 1)
        # Approximated as: dT/dt = (T_cmd - T_actual) / τ
        # For simplicity, use direct response with saturation
        actual_torque = command_torque

        # Update wheel momentum: h_dot = T
        self.momentum += actual_torque * dt

        # Saturate momentum
        if abs(self.momentum) > self.max_momentum:
            self.momentum = np.sign(self.momentum) * self.max_momentum
            actual_torque = 0.0  # Can't accelerate further (saturated)

        # Wheel spin rate: h = I·ω
        self.omega_wheel = self.momentum / self.inertia

        # Torque on spacecraft = -torque on wheel (reaction torque)
        torque_body = -actual_torque * self.axis

        return torque_body

    def desaturate(self, external_torque: np.ndarray, dt: float) -> float:
        """
        Desaturate wheel using external torque (e.g., from magnetic torquers).

        Args:
            external_torque: External torque vector [N·m]
            dt: Time step [s]

        Returns:
            Change in momentum [N·m·s]
        """
        # Component of external torque along wheel axis
        T_desat = np.dot(external_torque, self.axis)

        # Reduce wheel momentum
        delta_h = -T_desat * dt
        self.momentum += delta_h

        # Clamp to limits
        self.momentum = np.clip(self.momentum, -self.max_momentum, self.max_momentum)

        return delta_h


@dataclass
class ControlSurface:
    """
    Aerodynamic control surface model.

    Simplified model for elevons, rudders, ailerons, etc.
    """
    name: str
    max_deflection: float  # Maximum deflection [rad]
    effectiveness: float  # Moment per radian [N·m/rad]
    axis: np.ndarray  # Moment axis in body frame (unit vector)
    time_constant: float = 0.05  # Actuator lag [s]

    # State
    deflection: float = 0.0  # Current deflection [rad]
    deflection_rate: float = 0.0  # Deflection rate [rad/s]

    def __post_init__(self):
        """Normalize axis"""
        self.axis = self.axis / np.linalg.norm(self.axis)

    def compute_moment(self, command: float, dynamic_pressure: float, dt: float) -> np.ndarray:
        """
        Compute aerodynamic moment from control surface.

        M = effectiveness · δ · q_dyn

        Args:
            command: Commanded deflection [-1, 1] (normalized)
            dynamic_pressure: Dynamic pressure q = 0.5·ρ·V² [Pa]
            dt: Time step [s]

        Returns:
            Moment vector in body frame [N·m]
        """
        # Commanded deflection
        delta_cmd = command * self.max_deflection

        # First-order lag: dδ/dt = (δ_cmd - δ) / τ
        delta_dot = (delta_cmd - self.deflection) / self.time_constant
        self.deflection += delta_dot * dt
        self.deflection_rate = delta_dot

        # Saturate deflection
        self.deflection = np.clip(self.deflection, -self.max_deflection, self.max_deflection)

        # Aerodynamic moment
        moment_magnitude = self.effectiveness * self.deflection * dynamic_pressure

        moment_body = moment_magnitude * self.axis

        return moment_body


class ActuatorArray:
    """
    Collection of actuators with combined control allocation.
    """

    def __init__(self):
        self.rcs_thrusters: List[RCSThruster] = []
        self.reaction_wheels: List[ReactionWheel] = []
        self.control_surfaces: List[ControlSurface] = []

    def add_rcs_thruster(self, thruster: RCSThruster):
        """Add RCS thruster to array"""
        self.rcs_thrusters.append(thruster)

    def add_reaction_wheel(self, wheel: ReactionWheel):
        """Add reaction wheel to array"""
        self.reaction_wheels.append(wheel)

    def add_control_surface(self, surface: ControlSurface):
        """Add control surface to array"""
        self.control_surfaces.append(surface)

    def compute_total_torque(self, rcs_commands: Optional[np.ndarray] = None,
                            rw_commands: Optional[np.ndarray] = None,
                            cs_commands: Optional[np.ndarray] = None,
                            dynamic_pressure: float = 0.0,
                            dt: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute total force and torque from all actuators.

        Args:
            rcs_commands: Commands for RCS thrusters
            rw_commands: Commands for reaction wheels
            cs_commands: Commands for control surfaces
            dynamic_pressure: For control surfaces [Pa]
            dt: Time step [s]

        Returns:
            (total_force, total_torque) in body frame
        """
        total_force = np.zeros(3)
        total_torque = np.zeros(3)

        # RCS thrusters
        if rcs_commands is not None and len(self.rcs_thrusters) > 0:
            for i, thruster in enumerate(self.rcs_thrusters):
                if i < len(rcs_commands):
                    force, torque = thruster.compute_force_torque(rcs_commands[i], dt)
                    total_force += force
                    total_torque += torque

        # Reaction wheels
        if rw_commands is not None and len(self.reaction_wheels) > 0:
            for i, wheel in enumerate(self.reaction_wheels):
                if i < len(rw_commands):
                    torque = wheel.compute_torque(rw_commands[i], dt)
                    total_torque += torque

        # Control surfaces
        if cs_commands is not None and len(self.control_surfaces) > 0:
            for i, surface in enumerate(self.control_surfaces):
                if i < len(cs_commands):
                    moment = surface.compute_moment(cs_commands[i], dynamic_pressure, dt)
                    total_torque += moment

        return total_force, total_torque

    def get_reaction_wheel_momentum(self) -> np.ndarray:
        """Get total reaction wheel momentum vector"""
        h_total = np.zeros(3)
        for wheel in self.reaction_wheels:
            h_total += wheel.momentum * wheel.axis
        return h_total

    def total_fuel_consumed(self, dt: float) -> float:
        """Get total fuel consumed by all RCS thrusters"""
        total_fuel = 0.0
        for thruster in self.rcs_thrusters:
            total_fuel += thruster.fuel_consumption(dt)
        return total_fuel
