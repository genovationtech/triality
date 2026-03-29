"""
6-DoF rigid body dynamics for spacecraft and aircraft.

Implements:
- Translational dynamics (Newton's 2nd law)
- Rotational dynamics (Euler's equations)
- Quaternion attitude kinematics
- Direction Cosine Matrix (DCM) representation
- Body-frame and inertial-frame transformations
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class AttitudeRepresentation(Enum):
    """Attitude representation type"""
    QUATERNION = 'quaternion'
    DCM = 'dcm'
    EULER_321 = 'euler_321'  # Yaw-Pitch-Roll


@dataclass
class RigidBodyState:
    """
    Complete 6-DoF rigid body state.

    Position and velocity in inertial frame.
    Attitude as quaternion (scalar-last convention: [qx, qy, qz, qw])
    Angular velocity in body frame.
    """
    position: np.ndarray  # [x, y, z] in inertial frame [m]
    velocity: np.ndarray  # [vx, vy, vz] in inertial frame [m/s]
    quaternion: np.ndarray  # [qx, qy, qz, qw] (unit quaternion)
    angular_velocity: np.ndarray  # [wx, wy, wz] in body frame [rad/s]

    def __post_init__(self):
        """Normalize quaternion"""
        self.quaternion = self.quaternion / np.linalg.norm(self.quaternion)


class QuaternionKinematics:
    """
    Quaternion attitude kinematics and operations.

    Convention: Scalar-last [qx, qy, qz, qw]
    """

    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        """Normalize quaternion to unit magnitude"""
        return q / (np.linalg.norm(q) + 1e-16)

    @staticmethod
    def conjugate(q: np.ndarray) -> np.ndarray:
        """Quaternion conjugate: q* = [-qx, -qy, -qz, qw]"""
        return np.array([-q[0], -q[1], -q[2], q[3]])

    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Quaternion multiplication: q1 ⊗ q2

        Represents sequential rotations: first q2, then q1
        """
        qx1, qy1, qz1, qw1 = q1
        qx2, qy2, qz2, qw2 = q2

        qw = qw1*qw2 - qx1*qx2 - qy1*qy2 - qz1*qz2
        qx = qw1*qx2 + qx1*qw2 + qy1*qz2 - qz1*qy2
        qy = qw1*qy2 - qx1*qz2 + qy1*qw2 + qz1*qx2
        qz = qw1*qz2 + qx1*qy2 - qy1*qx2 + qz1*qw2

        return np.array([qx, qy, qz, qw])

    @staticmethod
    def rotate_vector(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate vector v by quaternion q.

        v_rotated = q ⊗ [0, v] ⊗ q*
        """
        # Convert vector to pure quaternion
        v_quat = np.array([v[0], v[1], v[2], 0.0])

        # Compute q ⊗ v ⊗ q*
        q_conj = QuaternionKinematics.conjugate(q)
        temp = QuaternionKinematics.multiply(q, v_quat)
        result = QuaternionKinematics.multiply(temp, q_conj)

        return result[:3]

    @staticmethod
    def to_dcm(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to Direction Cosine Matrix (DCM).

        DCM rotates from body frame to inertial frame.
        """
        qx, qy, qz, qw = q

        # DCM elements
        R = np.zeros((3, 3))

        R[0, 0] = 1 - 2*(qy**2 + qz**2)
        R[0, 1] = 2*(qx*qy - qz*qw)
        R[0, 2] = 2*(qx*qz + qy*qw)

        R[1, 0] = 2*(qx*qy + qz*qw)
        R[1, 1] = 1 - 2*(qx**2 + qz**2)
        R[1, 2] = 2*(qy*qz - qx*qw)

        R[2, 0] = 2*(qx*qz - qy*qw)
        R[2, 1] = 2*(qy*qz + qx*qw)
        R[2, 2] = 1 - 2*(qx**2 + qy**2)

        return R

    @staticmethod
    def from_dcm(R: np.ndarray) -> np.ndarray:
        """Convert DCM to quaternion"""
        trace = np.trace(R)

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2, 1] - R[1, 2]) * s
            qy = (R[0, 2] - R[2, 0]) * s
            qz = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        return QuaternionKinematics.normalize(np.array([qx, qy, qz, qw]))

    @staticmethod
    def to_euler_321(q: np.ndarray) -> np.ndarray:
        """
        Convert quaternion to Euler angles (3-2-1 sequence: yaw-pitch-roll).

        Returns:
            [roll, pitch, yaw] in radians
        """
        qx, qy, qz, qw = q

        # Roll (φ)
        roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))

        # Pitch (θ)
        sin_pitch = 2*(qw*qy - qz*qx)
        sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
        pitch = np.arcsin(sin_pitch)

        # Yaw (ψ)
        yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))

        return np.array([roll, pitch, yaw])

    @staticmethod
    def from_euler_321(euler: np.ndarray) -> np.ndarray:
        """
        Convert Euler angles (3-2-1) to quaternion.

        Args:
            euler: [roll, pitch, yaw] in radians
        """
        roll, pitch, yaw = euler

        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return QuaternionKinematics.normalize(np.array([qx, qy, qz, qw]))

    @staticmethod
    def derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Quaternion kinematic differential equation.

        q̇ = 0.5 * Ω(ω) * q

        where Ω is the skew-symmetric matrix form of ω.

        Args:
            q: Current quaternion [qx, qy, qz, qw]
            omega: Angular velocity in body frame [wx, wy, wz] [rad/s]

        Returns:
            q̇: Quaternion derivative
        """
        wx, wy, wz = omega
        qx, qy, qz, qw = q

        # Ω matrix (skew-symmetric)
        Omega = np.array([
            [0, wz, -wy, wx],
            [-wz, 0, wx, wy],
            [wy, -wx, 0, wz],
            [-wx, -wy, -wz, 0]
        ])

        q_dot = 0.5 * Omega @ q

        return q_dot


class RigidBodyDynamics:
    """
    6-DoF rigid body dynamics solver.

    Translational: m·dv/dt = F
    Rotational: I·dω/dt + ω × (I·ω) = M (Euler's equations)
    """

    def __init__(self, mass: float, inertia: np.ndarray):
        """
        Initialize rigid body.

        Args:
            mass: Body mass [kg]
            inertia: 3×3 inertia tensor in body frame [kg·m²]
        """
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = np.linalg.inv(inertia)

    def translational_acceleration(self, force: np.ndarray) -> np.ndarray:
        """
        Compute translational acceleration from applied force.

        a = F / m

        Args:
            force: Force vector in inertial frame [N]

        Returns:
            Acceleration in inertial frame [m/s²]
        """
        return force / self.mass

    def rotational_acceleration(self, torque: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Compute rotational acceleration from applied torque.

        α = I⁻¹ · (M - ω × I·ω)

        Args:
            torque: Torque vector in body frame [N·m]
            omega: Angular velocity in body frame [rad/s]

        Returns:
            Angular acceleration in body frame [rad/s²]
        """
        # Gyroscopic term: ω × (I·ω)
        I_omega = self.inertia @ omega
        gyro_term = np.cross(omega, I_omega)

        # α = I⁻¹ · (M - gyro_term)
        alpha = self.inertia_inv @ (torque - gyro_term)

        return alpha

    def dynamics(self, state: RigidBodyState, force_inertial: np.ndarray,
                torque_body: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute state derivatives.

        Args:
            state: Current state
            force_inertial: Force in inertial frame [N]
            torque_body: Torque in body frame [N·m]

        Returns:
            (r_dot, v_dot, q_dot, omega_dot)
        """
        # Position derivative
        r_dot = state.velocity

        # Velocity derivative (acceleration)
        v_dot = self.translational_acceleration(force_inertial)

        # Quaternion derivative
        q_dot = QuaternionKinematics.derivative(state.quaternion, state.angular_velocity)

        # Angular velocity derivative (angular acceleration)
        omega_dot = self.rotational_acceleration(torque_body, state.angular_velocity)

        return r_dot, v_dot, q_dot, omega_dot

    def integrate_step(self, state: RigidBodyState, force_inertial: np.ndarray,
                      torque_body: np.ndarray, dt: float,
                      method: str = 'rk4') -> RigidBodyState:
        """
        Integrate dynamics one time step.

        Args:
            state: Current state
            force_inertial: Force in inertial frame [N]
            torque_body: Torque in body frame [N·m]
            dt: Time step [s]
            method: Integration method ('euler', 'rk4')

        Returns:
            New state after time step
        """
        if method == 'euler':
            r_dot, v_dot, q_dot, omega_dot = self.dynamics(state, force_inertial, torque_body)

            new_position = state.position + r_dot * dt
            new_velocity = state.velocity + v_dot * dt
            new_quaternion = QuaternionKinematics.normalize(state.quaternion + q_dot * dt)
            new_omega = state.angular_velocity + omega_dot * dt

        elif method == 'rk4':
            # RK4 integration
            k1_r, k1_v, k1_q, k1_omega = self.dynamics(state, force_inertial, torque_body)

            state2 = RigidBodyState(
                position=state.position + 0.5*dt*k1_r,
                velocity=state.velocity + 0.5*dt*k1_v,
                quaternion=QuaternionKinematics.normalize(state.quaternion + 0.5*dt*k1_q),
                angular_velocity=state.angular_velocity + 0.5*dt*k1_omega
            )
            k2_r, k2_v, k2_q, k2_omega = self.dynamics(state2, force_inertial, torque_body)

            state3 = RigidBodyState(
                position=state.position + 0.5*dt*k2_r,
                velocity=state.velocity + 0.5*dt*k2_v,
                quaternion=QuaternionKinematics.normalize(state.quaternion + 0.5*dt*k2_q),
                angular_velocity=state.angular_velocity + 0.5*dt*k2_omega
            )
            k3_r, k3_v, k3_q, k3_omega = self.dynamics(state3, force_inertial, torque_body)

            state4 = RigidBodyState(
                position=state.position + dt*k3_r,
                velocity=state.velocity + dt*k3_v,
                quaternion=QuaternionKinematics.normalize(state.quaternion + dt*k3_q),
                angular_velocity=state.angular_velocity + dt*k3_omega
            )
            k4_r, k4_v, k4_q, k4_omega = self.dynamics(state4, force_inertial, torque_body)

            new_position = state.position + (dt/6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
            new_velocity = state.velocity + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
            new_quaternion = QuaternionKinematics.normalize(
                state.quaternion + (dt/6) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
            )
            new_omega = state.angular_velocity + (dt/6) * (k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)

        else:
            raise ValueError(f"Unknown integration method: {method}")

        return RigidBodyState(
            position=new_position,
            velocity=new_velocity,
            quaternion=new_quaternion,
            angular_velocity=new_omega
        )
