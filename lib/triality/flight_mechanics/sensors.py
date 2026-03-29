"""
Sensor models for spacecraft and aircraft navigation.

Includes:
- Inertial Measurement Unit (IMU) with gyro bias and noise
- GNSS receiver with dropout and multipath
- Star tracker with noise and field-of-view limits
- Realistic sensor error models
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SensorStatus(Enum):
    """Sensor operational status"""
    NOMINAL = 'nominal'
    DEGRADED = 'degraded'
    FAILED = 'failed'


@dataclass
class IMU:
    """
    Inertial Measurement Unit model.

    Measures angular velocity and linear acceleration with bias and noise.
    """
    # Gyroscope parameters
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Bias [rad/s]
    gyro_noise_density: float = 1e-4  # Noise density [rad/s/√Hz]
    gyro_bias_stability: float = 1e-5  # Bias stability [rad/s]
    gyro_random_walk: float = 1e-6  # Random walk [rad/s²/√Hz]

    # Accelerometer parameters
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Bias [m/s²]
    accel_noise_density: float = 1e-3  # Noise density [m/s²/√Hz]
    accel_bias_stability: float = 1e-4  # Bias stability [m/s²]

    # Sample rate
    sample_rate: float = 100.0  # Hz

    # Status
    status: SensorStatus = SensorStatus.NOMINAL

    def measure_angular_velocity(self, true_omega: np.ndarray, dt: float) -> np.ndarray:
        """
        Measure angular velocity with bias and noise.

        ω_meas = ω_true + bias + noise

        Args:
            true_omega: True angular velocity [rad/s]
            dt: Time step [s]

        Returns:
            Measured angular velocity [rad/s]
        """
        if self.status == SensorStatus.FAILED:
            return np.zeros(3)

        # Update bias (random walk)
        bias_drift = self.gyro_random_walk * np.random.randn(3) * np.sqrt(dt)
        self.gyro_bias += bias_drift

        # Add noise
        noise = self.gyro_noise_density * np.random.randn(3) / np.sqrt(dt)

        omega_meas = true_omega + self.gyro_bias + noise

        if self.status == SensorStatus.DEGRADED:
            omega_meas += 0.1 * np.random.randn(3)  # Additional degraded mode noise

        return omega_meas

    def measure_acceleration(self, true_accel: np.ndarray, dt: float) -> np.ndarray:
        """
        Measure linear acceleration with bias and noise.

        a_meas = a_true + bias + noise

        Args:
            true_accel: True acceleration [m/s²]
            dt: Time step [s]

        Returns:
            Measured acceleration [m/s²]
        """
        if self.status == SensorStatus.FAILED:
            return np.zeros(3)

        # Add noise
        noise = self.accel_noise_density * np.random.randn(3) / np.sqrt(dt)

        accel_meas = true_accel + self.accel_bias + noise

        if self.status == SensorStatus.DEGRADED:
            accel_meas += 0.5 * np.random.randn(3)

        return accel_meas

    def reset_bias(self):
        """Reset gyro and accelerometer biases (e.g., after calibration)"""
        self.gyro_bias = np.zeros(3)
        self.accel_bias = np.zeros(3)


@dataclass
class GNSSReceiver:
    """
    GNSS (GPS/Galileo/GLONASS) receiver model.

    Provides position and velocity with errors and dropout.
    """
    # Accuracy parameters
    position_accuracy: float = 5.0  # 1-σ position accuracy [m]
    velocity_accuracy: float = 0.1  # 1-σ velocity accuracy [m/s]

    # Dropout model
    dropout_probability: float = 0.01  # Probability of signal loss per update
    signal_available: bool = True

    # Multipath and atmospheric delay
    multipath_error_std: float = 2.0  # [m]
    atmospheric_delay_std: float = 1.0  # [m]

    # Update rate
    sample_rate: float = 1.0  # Hz

    # Status
    status: SensorStatus = SensorStatus.NOMINAL

    def measure_position_velocity(self, true_position: np.ndarray,
                                  true_velocity: np.ndarray,
                                  dt: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Measure position and velocity with GNSS.

        Args:
            true_position: True position [m]
            true_velocity: True velocity [m/s]
            dt: Time step [s]

        Returns:
            (position_meas, velocity_meas) or (None, None) if dropout
        """
        if self.status == SensorStatus.FAILED:
            return None, None

        # Check for dropout
        if np.random.rand() < self.dropout_probability * dt * self.sample_rate:
            self.signal_available = False
        else:
            self.signal_available = True

        if not self.signal_available:
            # Random recovery
            if np.random.rand() < 0.1:  # 10% chance of recovery per update
                self.signal_available = True

        if not self.signal_available:
            return None, None

        # Position measurement with errors
        pos_noise = self.position_accuracy * np.random.randn(3)
        multipath_error = self.multipath_error_std * np.random.randn(3)
        atmos_delay = self.atmospheric_delay_std * np.random.randn(3)

        position_meas = true_position + pos_noise + multipath_error + atmos_delay

        # Velocity measurement
        vel_noise = self.velocity_accuracy * np.random.randn(3)
        velocity_meas = true_velocity + vel_noise

        if self.status == SensorStatus.DEGRADED:
            # Degraded mode: 10x worse accuracy
            position_meas += 10 * self.position_accuracy * np.random.randn(3)
            velocity_meas += 10 * self.velocity_accuracy * np.random.randn(3)

        return position_meas, velocity_meas


@dataclass
class StarTracker:
    """
    Star tracker attitude sensor.

    Provides high-accuracy attitude quaternion with noise and FOV limits.
    """
    # Accuracy
    cross_boresight_accuracy: float = 10.0  # arcsec (cross-boresight)
    boresight_accuracy: float = 30.0  # arcsec (about boresight)

    # Field of view
    fov_half_angle: float = np.deg2rad(15)  # Half-angle FOV [rad]

    # Update rate
    sample_rate: float = 10.0  # Hz

    # Minimum stars required
    min_stars: int = 3

    # Status
    status: SensorStatus = SensorStatus.NOMINAL
    stars_in_fov: int = 10  # Simulated star count

    def measure_attitude(self, true_quaternion: np.ndarray,
                        sun_vector_body: Optional[np.ndarray] = None,
                        dt: float = 0.1) -> Optional[np.ndarray]:
        """
        Measure attitude quaternion with noise.

        Args:
            true_quaternion: True attitude quaternion [qx, qy, qz, qw]
            sun_vector_body: Sun direction in body frame (for FOV check)
            dt: Time step [s]

        Returns:
            Measured quaternion or None if insufficient stars
        """
        if self.status == SensorStatus.FAILED:
            return None

        # Check if sun is in FOV (blinds sensor)
        if sun_vector_body is not None:
            # Assume boresight is +Z axis
            boresight = np.array([0, 0, 1])
            angle_to_sun = np.arccos(np.clip(np.dot(sun_vector_body, boresight), -1, 1))

            if angle_to_sun < self.fov_half_angle:
                # Sun in FOV - no stars visible
                return None

        # Simulate star count (random)
        self.stars_in_fov = np.random.poisson(10)

        if self.stars_in_fov < self.min_stars:
            return None

        # Convert accuracy from arcsec to radians
        cross_acc_rad = np.deg2rad(self.cross_boresight_accuracy / 3600)
        bore_acc_rad = np.deg2rad(self.boresight_accuracy / 3600)

        # Generate attitude error as small rotation
        # Error about each axis
        error_angles = np.array([
            cross_acc_rad * np.random.randn(),
            cross_acc_rad * np.random.randn(),
            bore_acc_rad * np.random.randn()
        ])

        # Convert error angles to error quaternion
        error_magnitude = np.linalg.norm(error_angles)
        if error_magnitude > 0:
            error_axis = error_angles / error_magnitude
            error_quat = np.array([
                error_axis[0] * np.sin(error_magnitude/2),
                error_axis[1] * np.sin(error_magnitude/2),
                error_axis[2] * np.sin(error_magnitude/2),
                np.cos(error_magnitude/2)
            ])
        else:
            error_quat = np.array([0, 0, 0, 1])

        # Apply error to true quaternion: q_meas = q_error ⊗ q_true
        from .rigid_body_dynamics import QuaternionKinematics
        measured_quat = QuaternionKinematics.multiply(error_quat, true_quaternion)

        if self.status == SensorStatus.DEGRADED:
            # Additional error in degraded mode
            large_error = 10 * cross_acc_rad * np.random.randn(3)
            large_error_mag = np.linalg.norm(large_error)
            if large_error_mag > 0:
                deg_error_axis = large_error / large_error_mag
                deg_error_quat = np.array([
                    deg_error_axis[0] * np.sin(large_error_mag/2),
                    deg_error_axis[1] * np.sin(large_error_mag/2),
                    deg_error_axis[2] * np.sin(large_error_mag/2),
                    np.cos(large_error_mag/2)
                ])
                measured_quat = QuaternionKinematics.multiply(deg_error_quat, measured_quat)

        return QuaternionKinematics.normalize(measured_quat)


class SensorSuite:
    """
    Collection of sensors for integrated navigation.
    """

    def __init__(self):
        self.imu: Optional[IMU] = None
        self.gnss: Optional[GNSSReceiver] = None
        self.star_tracker: Optional[StarTracker] = None

    def add_imu(self, imu: IMU):
        """Add IMU to sensor suite"""
        self.imu = imu

    def add_gnss(self, gnss: GNSSReceiver):
        """Add GNSS receiver"""
        self.gnss = gnss

    def add_star_tracker(self, star_tracker: StarTracker):
        """Add star tracker"""
        self.star_tracker = star_tracker

    def get_measurements(self, true_state, dt: float) -> dict:
        """
        Get all available sensor measurements.

        Args:
            true_state: RigidBodyState with true values
            dt: Time step [s]

        Returns:
            Dictionary of measurements
        """
        measurements = {}

        if self.imu is not None:
            # Compute true acceleration in body frame
            # For simplicity, assume zero acceleration (would need force input)
            true_accel_body = np.zeros(3)

            measurements['omega'] = self.imu.measure_angular_velocity(
                true_state.angular_velocity, dt
            )
            measurements['accel'] = self.imu.measure_acceleration(true_accel_body, dt)

        if self.gnss is not None:
            pos, vel = self.gnss.measure_position_velocity(
                true_state.position, true_state.velocity, dt
            )
            measurements['gnss_position'] = pos
            measurements['gnss_velocity'] = vel

        if self.star_tracker is not None:
            quat = self.star_tracker.measure_attitude(true_state.quaternion, dt=dt)
            measurements['star_tracker_quaternion'] = quat

        return measurements
