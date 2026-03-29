"""
Control laws for spacecraft and aircraft attitude control.

Includes:
- PID controller with anti-windup
- LQR (Linear Quadratic Regulator)
- Gain scheduling
- Fault-tolerant control modes
"""

import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class ControlMode(Enum):
    """Control mode enumeration"""
    NOMINAL = 'nominal'
    SAFE = 'safe'
    DETUMBLE = 'detumble'
    POINTING = 'pointing'
    RATE_DAMPING = 'rate_damping'


@dataclass
class PIDController:
    """
    PID controller for attitude control.

    u(t) = K_p·e + K_i·∫e·dt + K_d·ė
    """
    Kp: np.ndarray  # Proportional gain (3×3 or 3×1)
    Ki: np.ndarray  # Integral gain (3×3 or 3×1)
    Kd: np.ndarray  # Derivative gain (3×3 or 3×1)

    # Anti-windup limits
    integral_limit: float = 10.0  # Maximum integral term magnitude

    # State
    integral: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_error: Optional[np.ndarray] = None

    def reset(self):
        """Reset integral and derivative states"""
        self.integral = np.zeros(3)
        self.last_error = None

    def compute(self, error: np.ndarray, dt: float,
               error_derivative: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute control output.

        Args:
            error: Error signal (e.g., attitude error)
            dt: Time step [s]
            error_derivative: Derivative of error (if available)

        Returns:
            Control output
        """
        # Proportional term
        P = self.Kp * error if self.Kp.shape == (3,) else self.Kp @ error

        # Integral term with anti-windup
        self.integral += error * dt

        # Clamp integral
        integral_magnitude = np.linalg.norm(self.integral)
        if integral_magnitude > self.integral_limit:
            self.integral = self.integral / integral_magnitude * self.integral_limit

        I = self.Ki * self.integral if self.Ki.shape == (3,) else self.Ki @ self.integral

        # Derivative term
        if error_derivative is not None:
            error_dot = error_derivative
        elif self.last_error is not None:
            error_dot = (error - self.last_error) / dt
        else:
            error_dot = np.zeros(3)

        self.last_error = error.copy()

        D = self.Kd * error_dot if self.Kd.shape == (3,) else self.Kd @ error_dot

        # Control output
        u = P + I + D

        return u


@dataclass
class LQRController:
    """
    Linear Quadratic Regulator for attitude control.

    Minimizes cost: J = ∫(x'Qx + u'Ru)dt

    Assumes linear dynamics: ẋ = Ax + Bu
    Control law: u = -Kx where K is LQR gain
    """
    A: np.ndarray  # State matrix
    B: np.ndarray  # Input matrix
    Q: np.ndarray  # State cost matrix
    R: np.ndarray  # Control cost matrix

    K: Optional[np.ndarray] = None  # LQR gain (computed)

    def __post_init__(self):
        """Compute LQR gain on initialization"""
        if self.K is None:
            self.compute_gain()

    def compute_gain(self):
        """
        Compute LQR gain by solving Algebraic Riccati Equation.

        A'P + PA - PBR⁻¹B'P + Q = 0
        K = R⁻¹B'P
        """
        try:
            from scipy.linalg import solve_continuous_are

            P = solve_continuous_are(self.A, self.B, self.Q, self.R)
            self.K = np.linalg.inv(self.R) @ self.B.T @ P

        except ImportError:
            # Fallback: use simple gain (not optimal)
            print("Warning: scipy not available, using simplified gain")
            self.K = np.eye(self.B.shape[1], self.A.shape[0])

    def compute(self, state: np.ndarray) -> np.ndarray:
        """
        Compute control output.

        u = -K·x

        Args:
            state: State vector

        Returns:
            Control output
        """
        if self.K is None:
            raise ValueError("LQR gain not computed")

        u = -self.K @ state

        return u


class AttitudeController:
    """
    Attitude control system with multiple modes.
    """

    def __init__(self, mode: ControlMode = ControlMode.NOMINAL):
        self.mode = mode

        # Controllers for different modes
        self.pid_controller: Optional[PIDController] = None
        self.lqr_controller: Optional[LQRController] = None

        # Gain scheduling
        self.gain_schedule: Optional[Callable[[float], np.ndarray]] = None

    def set_pid_controller(self, pid: PIDController):
        """Set PID controller"""
        self.pid_controller = pid

    def set_lqr_controller(self, lqr: LQRController):
        """Set LQR controller"""
        self.lqr_controller = lqr

    def set_gain_schedule(self, schedule_func: Callable[[float], np.ndarray]):
        """
        Set gain scheduling function.

        Args:
            schedule_func: Function that maps parameter to gain matrix
        """
        self.gain_schedule = schedule_func

    def attitude_error_quaternion(self, q_current: np.ndarray,
                                  q_desired: np.ndarray) -> np.ndarray:
        """
        Compute attitude error quaternion.

        q_error = q_desired ⊗ q_current*

        Returns vector part of error quaternion (small angle approximation).

        Args:
            q_current: Current attitude quaternion
            q_desired: Desired attitude quaternion

        Returns:
            Error vector [rad] (3×1)
        """
        from .rigid_body_dynamics import QuaternionKinematics

        q_current_conj = QuaternionKinematics.conjugate(q_current)
        q_error = QuaternionKinematics.multiply(q_desired, q_current_conj)

        # Extract vector part (small angle: θ ≈ 2·q_vec)
        error_vector = 2.0 * q_error[:3]

        return error_vector

    def compute_control(self, q_current: np.ndarray, omega_current: np.ndarray,
                       q_desired: np.ndarray, omega_desired: np.ndarray,
                       dt: float, schedule_param: Optional[float] = None) -> np.ndarray:
        """
        Compute control torque based on current mode.

        Args:
            q_current: Current quaternion
            omega_current: Current angular velocity [rad/s]
            q_desired: Desired quaternion
            omega_desired: Desired angular velocity [rad/s]
            dt: Time step [s]
            schedule_param: Parameter for gain scheduling

        Returns:
            Control torque [N·m]
        """
        if self.mode == ControlMode.DETUMBLE:
            # Simple rate damping
            return self._detumble_control(omega_current)

        elif self.mode == ControlMode.RATE_DAMPING:
            # Damp to zero rate
            return self._rate_damping_control(omega_current)

        elif self.mode == ControlMode.POINTING:
            # Attitude tracking with PID or LQR
            if self.pid_controller is not None:
                return self._pid_pointing_control(q_current, omega_current,
                                                  q_desired, omega_desired, dt)
            elif self.lqr_controller is not None:
                return self._lqr_pointing_control(q_current, omega_current,
                                                  q_desired, omega_desired)
            else:
                raise ValueError("No controller configured for POINTING mode")

        elif self.mode == ControlMode.SAFE:
            # Safe mode: sun pointing + rate damping
            return self._safe_mode_control(omega_current)

        else:
            # Nominal mode
            if self.pid_controller is not None:
                return self._pid_pointing_control(q_current, omega_current,
                                                  q_desired, omega_desired, dt)
            else:
                return np.zeros(3)

    def _detumble_control(self, omega: np.ndarray) -> np.ndarray:
        """Detumble control: aggressive rate damping"""
        K_detumble = 100.0  # High gain for fast detumbling
        torque = -K_detumble * omega
        return torque

    def _rate_damping_control(self, omega: np.ndarray) -> np.ndarray:
        """Rate damping control"""
        K_damp = 10.0
        torque = -K_damp * omega
        return torque

    def _safe_mode_control(self, omega: np.ndarray) -> np.ndarray:
        """Safe mode: sun pointing (simplified as rate damping)"""
        K_safe = 5.0
        torque = -K_safe * omega
        return torque

    def _pid_pointing_control(self, q_current: np.ndarray, omega_current: np.ndarray,
                             q_desired: np.ndarray, omega_desired: np.ndarray,
                             dt: float) -> np.ndarray:
        """PID-based attitude tracking"""
        # Attitude error
        att_error = self.attitude_error_quaternion(q_current, q_desired)

        # Rate error
        rate_error = omega_desired - omega_current

        # Combined error state (for derivative term)
        error = att_error

        torque = self.pid_controller.compute(error, dt, error_derivative=rate_error)

        return torque

    def _lqr_pointing_control(self, q_current: np.ndarray, omega_current: np.ndarray,
                             q_desired: np.ndarray, omega_desired: np.ndarray) -> np.ndarray:
        """LQR-based attitude tracking"""
        # Attitude error
        att_error = self.attitude_error_quaternion(q_current, q_desired)

        # Rate error
        rate_error = omega_current - omega_desired

        # State vector: [attitude_error, rate_error]
        state = np.concatenate([att_error, rate_error])

        torque = self.lqr_controller.compute(state)

        return torque

    def switch_mode(self, new_mode: ControlMode):
        """
        Switch control mode.

        Args:
            new_mode: New control mode
        """
        print(f"Switching control mode: {self.mode.value} -> {new_mode.value}")
        self.mode = new_mode

        # Reset controllers on mode switch
        if self.pid_controller is not None:
            self.pid_controller.reset()


class FaultTolerantController:
    """
    Fault-tolerant control with actuator failure handling.
    """

    def __init__(self, nominal_controller: AttitudeController):
        self.nominal_controller = nominal_controller

        # Fault detection
        self.actuator_health: dict = {}  # {actuator_id: health_status}

    def detect_actuator_fault(self, actuator_id: str, response: np.ndarray,
                             expected_response: np.ndarray, threshold: float = 0.1) -> bool:
        """
        Detect actuator fault by comparing response to expected.

        Args:
            actuator_id: Actuator identifier
            response: Actual actuator response
            expected_response: Expected response
            threshold: Fault detection threshold

        Returns:
            True if fault detected
        """
        error = np.linalg.norm(response - expected_response)

        if error > threshold:
            self.actuator_health[actuator_id] = 'failed'
            return True
        else:
            self.actuator_health[actuator_id] = 'nominal'
            return False

    def reconfigure_control(self, failed_actuators: list):
        """
        Reconfigure control allocation for failed actuators.

        Args:
            failed_actuators: List of failed actuator IDs
        """
        if len(failed_actuators) > 0:
            # Switch to safe mode if multiple failures
            if len(failed_actuators) >= 2:
                self.nominal_controller.switch_mode(ControlMode.SAFE)
            else:
                # Try to maintain POINTING mode with degraded performance
                pass
