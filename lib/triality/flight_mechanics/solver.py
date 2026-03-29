"""
Numerical solver for 6-DOF flight mechanics simulation.

Provides FlightMechanicsSolver that integrates the full 6-DOF rigid body
equations of motion coupled with sensor models, actuator dynamics, and
attitude control laws from the flight_mechanics module.

Governing equations:

    Translational:  m * dv/dt = F_total  (in inertial frame)
    Rotational:     I * domega/dt + omega x (I * omega) = M_total  (Euler's eqns)
    Kinematics:     dq/dt = 0.5 * Omega(omega) * q  (quaternion propagation)

The solver uses RK4 integration (via RigidBodyDynamics.integrate_step) and
supports:
  - External force/torque callbacks
  - Gravity with optional J2 oblateness
  - Aerodynamic forces (simple model)
  - Sensor noise injection (IMU, GNSS, star tracker)
  - Actuator response (RCS, reaction wheels, control surfaces)
  - Closed-loop attitude control (PID or LQR)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Tuple

from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from .rigid_body_dynamics import (
    RigidBodyState,
    RigidBodyDynamics,
    QuaternionKinematics,
)
from .actuators import (
    ActuatorArray,
    RCSThruster,
    ReactionWheel,
    ControlSurface,
)
from .sensors import (
    IMU,
    GNSSReceiver,
    StarTracker,
    SensorSuite,
)
from .control_laws import (
    PIDController,
    LQRController,
    AttitudeController,
    ControlMode,
    FaultTolerantController,
)


@dataclass
class FlightMechanicsResult:
    """Result container for the 6-DOF flight mechanics solver.

    Attributes
    ----------
    time : np.ndarray
        Time array [s].
    position : np.ndarray
        Position history (N, 3) in inertial frame [m].
    velocity : np.ndarray
        Velocity history (N, 3) [m/s].
    quaternion : np.ndarray
        Quaternion history (N, 4) [qx, qy, qz, qw].
    angular_velocity : np.ndarray
        Angular velocity history (N, 3) in body frame [rad/s].
    euler_angles : np.ndarray
        Euler angles history (N, 3) [roll, pitch, yaw] [rad].
    control_torque : np.ndarray
        Applied control torque history (N, 3) [N*m].
    attitude_error : np.ndarray
        Attitude error history (N, 3) [rad].
    speed : np.ndarray
        Speed history [m/s].
    altitude : np.ndarray
        Altitude (z-component of position) history [m].
    fuel_consumed : float
        Total fuel consumed by RCS [kg].
    wheel_momentum : np.ndarray
        Final reaction wheel momentum vector [N*m*s].
    settling_time : Optional[float]
        Time to reach steady-state pointing (within 1 deg), if achieved [s].
    max_angular_rate : float
        Peak angular rate during simulation [rad/s].
    """

    time: np.ndarray = field(default_factory=lambda: np.array([]))
    position: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    velocity: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    quaternion: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    euler_angles: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    control_torque: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    attitude_error: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    speed: np.ndarray = field(default_factory=lambda: np.array([]))
    altitude: np.ndarray = field(default_factory=lambda: np.array([]))
    fuel_consumed: float = 0.0
    wheel_momentum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    settling_time: Optional[float] = None
    max_angular_rate: float = 0.0


class FlightMechanicsSolver:
    """6-DOF flight mechanics numerical solver.

    Integrates the coupled translational and rotational dynamics for a
    rigid body with sensors, actuators, and control.

    Parameters
    ----------
    mass : float
        Vehicle mass [kg].
    inertia : np.ndarray
        3x3 inertia tensor in body frame [kg*m^2].
    initial_state : RigidBodyState
        Initial position, velocity, attitude, and angular velocity.
    dt : float
        Integration time step [s].
    gravity : bool
        Enable gravity force (along -Z in inertial frame).
    mu : float
        Gravitational parameter [m^3/s^2] (for orbital problems).
        If 0, uses flat-Earth g = 9.80665 m/s^2.

    Examples
    --------
    >>> inertia = np.diag([100.0, 120.0, 80.0])
    >>> state0 = RigidBodyState(
    ...     position=np.zeros(3),
    ...     velocity=np.zeros(3),
    ...     quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
    ...     angular_velocity=np.array([0.01, -0.02, 0.005]),
    ... )
    >>> solver = FlightMechanicsSolver(mass=500.0, inertia=inertia,
    ...                                 initial_state=state0)
    >>> result = solver.solve(t_final=60.0)
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        mass: float,
        inertia: np.ndarray,
        initial_state: Optional[RigidBodyState] = None,
        dt: float = 0.01,
        gravity: bool = True,
        mu: float = 0.0,
    ):
        self.mass = mass
        self.inertia = np.asarray(inertia, dtype=float)
        self.dt = dt
        self.gravity = gravity
        self.mu = mu

        if initial_state is None:
            initial_state = RigidBodyState(
                position=np.zeros(3),
                velocity=np.zeros(3),
                quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
                angular_velocity=np.zeros(3),
            )
        self.initial_state = initial_state

        # Dynamics engine
        self.dynamics = RigidBodyDynamics(mass, self.inertia)

        # Optional subsystems
        self.actuators: Optional[ActuatorArray] = None
        self.sensors: Optional[SensorSuite] = None
        self.controller: Optional[AttitudeController] = None
        self.target_quaternion: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
        self.target_omega: np.ndarray = np.zeros(3)

        # External force/torque callback: f(t, state) -> (force_inertial, torque_body)
        self.external_forces: Optional[
            Callable[[float, RigidBodyState], Tuple[np.ndarray, np.ndarray]]
        ] = None

    def set_actuators(self, actuators: ActuatorArray):
        """Attach an actuator array to the solver."""
        self.actuators = actuators

    def set_sensors(self, sensors: SensorSuite):
        """Attach a sensor suite to the solver."""
        self.sensors = sensors

    def set_controller(
        self,
        controller: AttitudeController,
        target_quaternion: Optional[np.ndarray] = None,
        target_omega: Optional[np.ndarray] = None,
    ):
        """Attach an attitude controller and set the reference state."""
        self.controller = controller
        if target_quaternion is not None:
            self.target_quaternion = target_quaternion / np.linalg.norm(target_quaternion)
        if target_omega is not None:
            self.target_omega = target_omega

    def _gravity_force(self, state: RigidBodyState) -> np.ndarray:
        """Compute gravity force in the inertial frame."""
        if not self.gravity:
            return np.zeros(3)

        if self.mu > 0:
            # Inverse-square gravity for orbital mechanics
            r = state.position
            r_mag = np.linalg.norm(r)
            if r_mag < 1.0:
                return np.zeros(3)
            return -self.mu * self.mass / r_mag**3 * r
        else:
            # Flat-Earth gravity
            return np.array([0.0, 0.0, -self.mass * 9.80665])

    def solve(
        self,
        t_final: float,
        q_desired: Optional[np.ndarray] = None,
        omega_desired: Optional[np.ndarray] = None,
        progress_callback=None,
    ) -> FlightMechanicsResult:
        """Integrate the 6-DOF equations of motion.

        Parameters
        ----------
        t_final : float
            Simulation end time [s].
        q_desired : np.ndarray or None
            Desired quaternion (overrides constructor target if given).
        omega_desired : np.ndarray or None
            Desired angular velocity [rad/s].

        Returns
        -------
        FlightMechanicsResult
            Time histories of all state variables and derived quantities.
        """
        if q_desired is not None:
            self.target_quaternion = q_desired / np.linalg.norm(q_desired)
        if omega_desired is not None:
            self.target_omega = omega_desired

        dt = self.dt
        n_steps = int(np.ceil(t_final / dt))
        n_out = n_steps + 1

        # Preallocate output arrays
        t_arr = np.zeros(n_out)
        pos_arr = np.zeros((n_out, 3))
        vel_arr = np.zeros((n_out, 3))
        quat_arr = np.zeros((n_out, 4))
        omega_arr = np.zeros((n_out, 3))
        euler_arr = np.zeros((n_out, 3))
        ctrl_arr = np.zeros((n_out, 3))
        err_arr = np.zeros((n_out, 3))

        # Initial conditions
        state = RigidBodyState(
            position=self.initial_state.position.copy(),
            velocity=self.initial_state.velocity.copy(),
            quaternion=self.initial_state.quaternion.copy(),
            angular_velocity=self.initial_state.angular_velocity.copy(),
        )

        pos_arr[0] = state.position
        vel_arr[0] = state.velocity
        quat_arr[0] = state.quaternion
        omega_arr[0] = state.angular_velocity
        euler_arr[0] = QuaternionKinematics.to_euler_321(state.quaternion)

        total_fuel = 0.0
        settling_time = None
        max_omega = np.linalg.norm(state.angular_velocity)

        _prog_interval = max(n_steps // 50, 1)
        for k in range(n_steps):
            t = k * dt
            actual_dt = min(dt, t_final - t)
            if progress_callback and k % _prog_interval == 0:
                progress_callback(k, n_steps)

            # --- Compute forces and torques ---
            force_inertial = self._gravity_force(state)
            torque_body = np.zeros(3)

            # External forces
            if self.external_forces is not None:
                f_ext, tau_ext = self.external_forces(t, state)
                force_inertial += f_ext
                torque_body += tau_ext

            # Control torque
            ctrl_torque = np.zeros(3)
            if self.controller is not None:
                ctrl_torque = self.controller.compute_control(
                    state.quaternion,
                    state.angular_velocity,
                    self.target_quaternion,
                    self.target_omega,
                    actual_dt,
                )

            # Actuator allocation
            if self.actuators is not None:
                # Distribute control torque to reaction wheels
                n_rw = len(self.actuators.reaction_wheels)
                if n_rw > 0:
                    rw_cmds = np.zeros(n_rw)
                    for i, wheel in enumerate(self.actuators.reaction_wheels):
                        rw_cmds[i] = -np.dot(ctrl_torque, wheel.axis)
                    _, act_torque = self.actuators.compute_total_torque(
                        rw_commands=rw_cmds, dt=actual_dt,
                    )
                    torque_body += act_torque
                else:
                    torque_body += ctrl_torque
                total_fuel += self.actuators.total_fuel_consumed(actual_dt)
            else:
                torque_body += ctrl_torque

            ctrl_arr[k] = ctrl_torque

            # Attitude error
            if self.controller is not None:
                err_arr[k] = self.controller.attitude_error_quaternion(
                    state.quaternion, self.target_quaternion,
                )
            else:
                err_arr[k] = np.zeros(3)

            # --- Integrate one step ---
            state = self.dynamics.integrate_step(
                state, force_inertial, torque_body, actual_dt, method='rk4',
            )

            # Record
            idx = k + 1
            t_arr[idx] = t + actual_dt
            pos_arr[idx] = state.position
            vel_arr[idx] = state.velocity
            quat_arr[idx] = state.quaternion
            omega_arr[idx] = state.angular_velocity
            euler_arr[idx] = QuaternionKinematics.to_euler_321(state.quaternion)

            omega_mag = np.linalg.norm(state.angular_velocity)
            if omega_mag > max_omega:
                max_omega = omega_mag

            # Settling time detection (attitude error < 1 deg ~ 0.0175 rad)
            if settling_time is None and self.controller is not None:
                err_mag = np.linalg.norm(err_arr[k])
                if err_mag < np.radians(1.0):
                    settling_time = t + actual_dt

        # Fill last control/error entries
        ctrl_arr[-1] = ctrl_arr[-2] if n_steps > 0 else np.zeros(3)
        err_arr[-1] = err_arr[-2] if n_steps > 0 else np.zeros(3)

        speed_arr = np.linalg.norm(vel_arr, axis=1)
        alt_arr = pos_arr[:, 2]

        wheel_mom = np.zeros(3)
        if self.actuators is not None:
            wheel_mom = self.actuators.get_reaction_wheel_momentum()

        return FlightMechanicsResult(
            time=t_arr,
            position=pos_arr,
            velocity=vel_arr,
            quaternion=quat_arr,
            angular_velocity=omega_arr,
            euler_angles=euler_arr,
            control_torque=ctrl_arr,
            attitude_error=err_arr,
            speed=speed_arr,
            altitude=alt_arr,
            fuel_consumed=total_fuel,
            wheel_momentum=wheel_mom,
            settling_time=settling_time,
            max_angular_rate=max_omega,
        )

    def export_state(self, result: FlightMechanicsResult) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="flight_mechanics")
        state.set_field("velocity", result.velocity, "m/s")
        state.set_field("displacement", result.position, "m")
        state.metadata["fuel_consumed"] = result.fuel_consumed
        state.metadata["max_angular_rate"] = result.max_angular_rate
        state.metadata["settling_time"] = result.settling_time
        return state


# ========================================================================
# Level 3 2-D: Flight envelope / performance map solver
# ========================================================================

@dataclass
class FlightMechanics2DResult:
    """Result container for 2-D flight envelope performance map.

    Attributes
    ----------
    mach : np.ndarray
        Mach number grid values (n_mach,).
    altitude : np.ndarray
        Altitude grid values (n_alt,) [m].
    turn_rate : np.ndarray
        Sustained turn rate map (n_alt, n_mach) [deg/s].
    climb_rate : np.ndarray
        Maximum climb rate map (n_alt, n_mach) [m/s].
    load_factor : np.ndarray
        Maximum load factor map (n_alt, n_mach) [g].
    specific_excess_power : np.ndarray
        Specific excess power Ps map (n_alt, n_mach) [m/s].
    dynamic_pressure : np.ndarray
        Dynamic pressure map (n_alt, n_mach) [Pa].
    flight_envelope : np.ndarray
        Boolean flight-envelope mask (n_alt, n_mach).
    max_turn_rate : float
        Peak sustained turn rate [deg/s].
    max_climb_rate : float
        Peak climb rate [m/s].
    max_Ps : float
        Peak specific excess power [m/s].
    """
    mach: np.ndarray = field(default_factory=lambda: np.zeros(0))
    altitude: np.ndarray = field(default_factory=lambda: np.zeros(0))
    turn_rate: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    climb_rate: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    load_factor: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    specific_excess_power: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    dynamic_pressure: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    flight_envelope: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=bool))
    max_turn_rate: float = 0.0
    max_climb_rate: float = 0.0
    max_Ps: float = 0.0


class FlightMechanics2DSolver:
    """2-D flight envelope and performance map solver.

    Computes performance metrics (sustained turn rate, climb rate,
    specific excess power, load factor) over a Mach-altitude grid
    using simplified point-performance equations.

    Physics:
        Ps = V * (T - D) / W        (specific excess power)
        n_max = min(T/W + 0.5*rho*V^2*CL_max*S/W, n_structural)
        turn_rate = g*sqrt(n^2 - 1) / V
        climb_rate = Ps

    Parameters
    ----------
    nx, ny : int
        Grid resolution (Mach x altitude).
    mach_range : tuple
        (Mach_min, Mach_max).
    alt_range : tuple
        (alt_min, alt_max) [m].
    weight : float
        Aircraft weight [N].
    wing_area : float
        Wing reference area [m^2].
    thrust_sl : float
        Sea-level static thrust [N].
    cd0 : float
        Zero-lift drag coefficient.
    k_drag : float
        Induced drag factor (CD = CD0 + k*CL^2).
    cl_max : float
        Maximum lift coefficient.
    n_structural : float
        Structural load factor limit [g].
    """

    fidelity_tier = FidelityTier.REDUCED_ORDER
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        nx: int = 60,
        ny: int = 50,
        mach_range: Tuple[float, float] = (0.2, 2.0),
        alt_range: Tuple[float, float] = (0.0, 20000.0),
        weight: float = 100000.0,
        wing_area: float = 40.0,
        thrust_sl: float = 80000.0,
        cd0: float = 0.020,
        k_drag: float = 0.05,
        cl_max: float = 1.5,
        n_structural: float = 9.0,
    ):
        self.nx = nx
        self.ny = ny
        self.mach_range = mach_range
        self.alt_range = alt_range
        self.weight = weight
        self.wing_area = wing_area
        self.thrust_sl = thrust_sl
        self.cd0 = cd0
        self.k_drag = k_drag
        self.cl_max = cl_max
        self.n_structural = n_structural

    @staticmethod
    def _atmosphere(alt: float):
        """Simple ISA atmosphere: returns (rho, T, a) at altitude [m]."""
        if alt < 11000.0:
            T = 288.15 - 0.0065 * alt
            p = 101325.0 * (T / 288.15) ** 5.2561
        else:
            T = 216.65
            p = 22632.1 * np.exp(-0.000157688 * (alt - 11000.0))
        rho = p / (287.05 * T)
        a = np.sqrt(1.4 * 287.05 * T)
        return rho, T, a

    def solve(self) -> FlightMechanics2DResult:
        """Compute flight performance over Mach-altitude grid.

        Returns
        -------
        FlightMechanics2DResult
        """
        nx, ny = self.nx, self.ny
        mach_arr = np.linspace(self.mach_range[0], self.mach_range[1], nx)
        alt_arr = np.linspace(self.alt_range[0], self.alt_range[1], ny)

        turn_rate = np.zeros((ny, nx))
        climb_rate = np.zeros((ny, nx))
        load_factor = np.zeros((ny, nx))
        Ps_map = np.zeros((ny, nx))
        q_map = np.zeros((ny, nx))
        envelope = np.zeros((ny, nx), dtype=bool)

        W = self.weight
        S = self.wing_area
        g = 9.80665

        for j in range(ny):
            rho, T_atm, a_sound = self._atmosphere(alt_arr[j])
            # Thrust lapse with altitude (simple model)
            sigma = rho / 1.225
            for i in range(nx):
                M = mach_arr[i]
                V = M * a_sound
                q = 0.5 * rho * V ** 2
                q_map[j, i] = q

                # Thrust model: lapse with density ratio and Mach
                T_avail = self.thrust_sl * sigma * (1.0 - 0.2 * M)
                T_avail = max(T_avail, 0.0)

                # Drag at 1g level flight
                CL_1g = W / (q * S) if q * S > 0 else 10.0
                CD = self.cd0 + self.k_drag * CL_1g ** 2
                D = q * S * CD

                # Specific excess power
                Ps = V * (T_avail - D) / W if W > 0 else 0.0
                Ps_map[j, i] = Ps

                # Maximum load factor
                CL_struct = self.n_structural * W / (q * S) if q * S > 0 else 0.0
                CL_eff = min(self.cl_max, CL_struct)
                n_aero = q * S * CL_eff / W if W > 0 else 0.0
                n_max = min(n_aero, self.n_structural)
                load_factor[j, i] = n_max

                # Sustained turn (T = D at n_turn)
                # q*S*(CD0 + k*(n*W/(q*S))^2) = T_avail
                # Solve for n: k*(n*W)^2/(q*S) + CD0*q*S = T_avail
                a_coeff = self.k_drag * W ** 2 / (q * S) if q * S > 0 else 1e30
                b_coeff = self.cd0 * q * S
                n_sust_sq = (T_avail - b_coeff) / (a_coeff + 1e-30)
                n_sust = np.sqrt(max(n_sust_sq, 0.0))
                n_sust = min(n_sust, n_max)

                # Turn rate: omega = g*sqrt(n^2-1)/V
                if n_sust > 1.0 and V > 1.0:
                    omega_turn = g * np.sqrt(n_sust ** 2 - 1.0) / V
                    turn_rate[j, i] = np.degrees(omega_turn)
                else:
                    turn_rate[j, i] = 0.0

                # Climb rate ~ Ps
                climb_rate[j, i] = max(Ps, 0.0)

                # Flight envelope: Ps > 0, CL_1g < CL_max, q > q_min
                if Ps > 0.0 and CL_1g < self.cl_max and q > 500.0:
                    envelope[j, i] = True

        return FlightMechanics2DResult(
            mach=mach_arr,
            altitude=alt_arr,
            turn_rate=turn_rate,
            climb_rate=climb_rate,
            load_factor=load_factor,
            specific_excess_power=Ps_map,
            dynamic_pressure=q_map,
            flight_envelope=envelope,
            max_turn_rate=float(np.max(turn_rate)),
            max_climb_rate=float(np.max(climb_rate)),
            max_Ps=float(np.max(Ps_map)),
        )

    def export_state(self) -> PhysicsState:
        """Run solver and export as PhysicsState."""
        result = self.solve()
        state = PhysicsState(solver_name="flight_mechanics_2d")
        state.set_field("turn_rate", result.turn_rate, "deg/s")
        state.set_field("climb_rate", result.climb_rate, "m/s")
        state.set_field("specific_excess_power", result.specific_excess_power, "m/s")
        state.metadata["max_turn_rate"] = result.max_turn_rate
        state.metadata["max_climb_rate"] = result.max_climb_rate
        state.metadata["max_Ps"] = result.max_Ps
        return state
