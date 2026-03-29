"""Adaptive Timestep Control for Transient Simulations.

Implements PID-based adaptive timestepping that automatically adjusts dt
based on local truncation error estimates. This is what separates
"runs on textbook problems" from "handles real-world transients."

Key idea: If the estimated error is below tolerance, grow dt.
If above, shrink dt and reject the step. PID smooths the adaptation.

Usage:
    controller = PIDTimestepController(tol=1e-6)
    dt = controller.dt

    while t < t_end:
        y_new, y_embed = take_step(y, dt)  # embedded pair gives error estimate
        err = controller.estimate_error(y_new, y_embed)
        accepted, dt = controller.adapt(err)
        if accepted:
            y = y_new
            t += dt_used
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class TimestepHistory:
    """Records timestep adaptation decisions for diagnostics."""
    times: List[float] = field(default_factory=list)
    dts: List[float] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    accepted: List[bool] = field(default_factory=list)
    rejections: int = 0
    total_steps: int = 0

    @property
    def acceptance_rate(self) -> float:
        return (self.total_steps - self.rejections) / max(1, self.total_steps)


class PIDTimestepController:
    """PID Controller for adaptive timestep selection.

    Uses the PI3.4 controller of Soderlind (2003) for smooth dt adaptation:

        dt_new = dt * (err_n/tol)^(-k_I) * (err_{n-1}/tol)^(k_P) * (err_{n-2}/tol)^(-k_D)

    Default coefficients give the H211b controller (robust, smooth).

    Features:
    - Smooth dt transitions (no oscillation)
    - Rejected step handling with safety factor
    - Min/max dt bounds to prevent runaway
    - Diagnostic history for post-mortem analysis
    """

    def __init__(self, dt_init: float = 1e-3, tol: float = 1e-6,
                 dt_min: float = 1e-12, dt_max: float = 1.0,
                 safety: float = 0.9, order: int = 2,
                 max_growth: float = 2.0, max_shrink: float = 0.1,
                 k_I: float = 0.25, k_P: float = 0.14, k_D: float = 0.0):
        """
        Args:
            dt_init: Initial timestep.
            tol: Error tolerance (absolute + relative mixed).
            dt_min: Minimum allowed timestep.
            dt_max: Maximum allowed timestep.
            safety: Safety factor (< 1.0 to be conservative).
            order: Order of the time integration method.
            max_growth: Maximum dt growth factor per step.
            max_shrink: Minimum dt shrink factor per step.
            k_I: Integral gain (main control).
            k_P: Proportional gain (smoothing).
            k_D: Derivative gain (stability, usually 0).
        """
        self.dt = dt_init
        self.tol = tol
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.safety = safety
        self.order = order
        self.max_growth = max_growth
        self.max_shrink = max_shrink
        self.k_I = k_I
        self.k_P = k_P
        self.k_D = k_D

        # Error history for PID
        self._err_prev = [1.0, 1.0, 1.0]  # [n, n-1, n-2]
        self._dt_prev = dt_init
        self.history = TimestepHistory()

    def estimate_error(self, y_high: np.ndarray, y_low: np.ndarray,
                       atol: float = None, rtol: float = 1e-6) -> float:
        """Estimate local truncation error from embedded pair.

        Uses mixed absolute/relative error norm:
            err_i = |y_high_i - y_low_i| / (atol + rtol * |y_high_i|)
            err = RMS(err_i)

        Args:
            y_high: Higher-order solution.
            y_low: Lower-order solution (or embedded estimate).
            atol: Absolute tolerance (defaults to self.tol).
            rtol: Relative tolerance.

        Returns:
            Normalized error estimate (< 1.0 means step is acceptable).
        """
        if atol is None:
            atol = self.tol
        diff = np.abs(y_high - y_low)
        scale = atol + rtol * np.abs(y_high)
        err = np.sqrt(np.mean((diff / scale) ** 2))
        return err

    def adapt(self, err: float) -> Tuple[bool, float]:
        """Adapt timestep based on error estimate.

        Args:
            err: Normalized error from estimate_error().

        Returns:
            (accepted, new_dt): Whether step is accepted and the new dt to use.
        """
        self.history.total_steps += 1

        if err <= 0.0:
            err = 1e-10  # Avoid log(0)

        # PID controller
        p = self.order + 1  # Error is O(h^{p})
        err_ratio = err / 1.0  # Normalized to tolerance (already done in estimate)

        # H211b controller: dt_new/dt = (e_{n}/e_{n-1})^k_P * (1/e_n)^k_I
        factor = (self._err_prev[1] / max(err, 1e-10)) ** self.k_P * \
                 (1.0 / max(err, 1e-10)) ** self.k_I * \
                 (self._err_prev[2] / max(self._err_prev[1], 1e-10)) ** self.k_D

        # Apply safety factor
        factor = self.safety * factor

        # Clamp growth/shrink
        factor = max(self.max_shrink, min(self.max_growth, factor))

        dt_new = self.dt * factor
        dt_new = max(self.dt_min, min(self.dt_max, dt_new))

        accepted = err <= 1.0

        if accepted:
            # Record and advance
            self.history.times.append(self.history.total_steps)
            self.history.dts.append(self.dt)
            self.history.errors.append(err)
            self.history.accepted.append(True)

            # Shift error history
            self._err_prev = [err, self._err_prev[0], self._err_prev[1]]
            self._dt_prev = self.dt
            self.dt = dt_new
        else:
            # Reject: shrink dt more aggressively
            self.history.rejections += 1
            self.history.accepted.append(False)
            self.history.errors.append(err)
            self.dt = max(self.dt_min, self.dt * max(self.max_shrink, self.safety / max(err, 1e-10) ** (1.0/p)))

        return accepted, self.dt

    def reset(self, dt_init: float = None):
        """Reset controller state (e.g., after a discontinuity)."""
        if dt_init is not None:
            self.dt = dt_init
        self._err_prev = [1.0, 1.0, 1.0]
        self.history = TimestepHistory()


class CFLController:
    """CFL-based timestep controller for explicit methods.

    Computes dt from the CFL condition:
        dt = CFL * min(dx) / max(|wave_speed|)

    For convection-dominated problems where stability, not accuracy,
    limits the timestep.
    """

    def __init__(self, cfl_target: float = 0.5, dt_min: float = 1e-15,
                 dt_max: float = 1.0, safety: float = 0.9):
        self.cfl_target = cfl_target
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.safety = safety
        self.dt = dt_max

    def compute_dt(self, dx: np.ndarray, wave_speed: np.ndarray) -> float:
        """Compute stable timestep from CFL condition.

        Args:
            dx: Cell sizes (array).
            wave_speed: Local wave speeds (array, same size as dx).

        Returns:
            CFL-limited timestep.
        """
        max_speed = np.max(np.abs(wave_speed))
        if max_speed < 1e-15:
            self.dt = self.dt_max
            return self.dt

        min_dx = np.min(dx)
        self.dt = self.safety * self.cfl_target * min_dx / max_speed
        self.dt = max(self.dt_min, min(self.dt_max, self.dt))
        return self.dt


class DualController:
    """Combined PID + CFL controller.

    Uses the minimum of PID-adapted dt (accuracy) and CFL dt (stability).
    This is what production codes actually use for transient PDEs.
    """

    def __init__(self, pid: PIDTimestepController, cfl: CFLController):
        self.pid = pid
        self.cfl = cfl

    def adapt(self, err: float, dx: np.ndarray, wave_speed: np.ndarray) -> Tuple[bool, float]:
        """Adapt timestep using both accuracy and stability constraints.

        Returns:
            (accepted, new_dt)
        """
        accepted, dt_pid = self.pid.adapt(err)
        dt_cfl = self.cfl.compute_dt(dx, wave_speed)
        dt_final = min(dt_pid, dt_cfl)
        self.pid.dt = dt_final
        return accepted, dt_final
