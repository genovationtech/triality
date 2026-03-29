"""
Layer 21: Flight Mechanics & GNC (Guidance, Navigation, and Control)

6-DoF rigid body dynamics, attitude control, sensors, and actuators for
spacecraft and aircraft.

Physics Basis:
--------------
Translational Dynamics:
    F = m·a                           (Newton's 2nd law)
    ṙ = v                            (position derivative)
    v̇ = F/m                          (velocity derivative)

Rotational Dynamics (Euler's Equations):
    I·ω̇ + ω × (I·ω) = M              (angular momentum balance)

    where I = inertia tensor [kg·m²]
          ω = angular velocity [rad/s]
          M = applied torque [N·m]

Quaternion Kinematics:
    q̇ = ½·Ω(ω)·q                     (quaternion derivative)

    where q = [qx, qy, qz, qw] (unit quaternion)
          Ω(ω) = skew-symmetric matrix of ω

Attitude Representations:
    - Quaternion: q (4 parameters, no singularities)
    - DCM: R (9 parameters, orthogonal)
    - Euler angles: [φ, θ, ψ] (3 parameters, gimbal lock)

Control Laws:
    PID: u = Kp·e + Ki·∫e·dt + Kd·ė
    LQR: u = -K·x  (K from Riccati equation)

Actuator Models:
    RCS Thruster: F = F_max·sign(cmd)
    Reaction Wheel: M = -I_wheel·α
    Control Surface: M = C_m·δ·q_dyn

Sensor Models:
    IMU: ω_meas = ω_true + bias + noise
    GNSS: r_meas = r_true + N(0, σ²)
    Star Tracker: q_meas = q_error ⊗ q_true

Features:
---------
1. 6-DoF rigid body dynamics (RK4 integration)
2. Quaternion attitude kinematics (singularity-free)
3. RCS thrusters with min impulse bit
4. Reaction wheels with saturation and desaturation
5. Control surfaces with first-order lag
6. IMU with gyro bias drift and noise
7. GNSS with dropout and multipath
8. Star tracker with FOV and sun exclusion
9. PID controller with anti-windup
10. LQR optimal control
11. Fault-tolerant control modes
12. Gain scheduling

Applications:
------------
- Spacecraft attitude control
- Satellite detumbling and pointing
- Aircraft flight control
- Launch vehicle guidance
- Re-entry vehicle control
- Orbital maneuvering

Typical Use Cases:
-----------------
- Three-axis stabilized spacecraft
- Momentum-bias spacecraft (reaction wheels)
- Thrust-vector-controlled rockets
- Aircraft autopilot systems
- Fault-tolerant control with actuator failures
"""

from .rigid_body_dynamics import (
    RigidBodyState,
    RigidBodyDynamics,
    QuaternionKinematics,
    AttitudeRepresentation
)

from .actuators import (
    RCSThruster,
    ReactionWheel,
    ControlSurface,
    ActuatorType,
    ActuatorArray
)

from .sensors import (
    IMU,
    GNSSReceiver,
    StarTracker,
    SensorStatus,
    SensorSuite
)

from .control_laws import (
    PIDController,
    LQRController,
    AttitudeController,
    FaultTolerantController,
    ControlMode
)

from .solver import (
    FlightMechanicsSolver,
    FlightMechanicsResult,
)

__all__ = [
    # Rigid body dynamics
    'RigidBodyState',
    'RigidBodyDynamics',
    'QuaternionKinematics',
    'AttitudeRepresentation',

    # Actuators
    'RCSThruster',
    'ReactionWheel',
    'ControlSurface',
    'ActuatorType',
    'ActuatorArray',

    # Sensors
    'IMU',
    'GNSSReceiver',
    'StarTracker',
    'SensorStatus',
    'SensorSuite',

    # Control laws
    'PIDController',
    'LQRController',
    'AttitudeController',
    'FaultTolerantController',
    'ControlMode',

    # Numerical solver
    'FlightMechanicsSolver',
    'FlightMechanicsResult',
]
