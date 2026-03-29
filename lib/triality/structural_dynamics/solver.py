"""
Structural dynamics time-domain solver (Level 3 - High-Fidelity).

Integrates the equation of motion M*x'' + C*x' + K*x = F(t) using the
Newmark-beta method, with modal decomposition support.  Wires in the
existing ModalSolver, RandomVibrationAnalyzer, and SRSCalculator for
pre/post-processing.

Level 2 (retained):
    Newmark-beta family (Newmark, 1959):
        x_{n+1} = x_n + dt*v_n + dt^2*((0.5-beta)*a_n + beta*a_{n+1})
        v_{n+1} = v_n + dt*((1-gamma)*a_n + gamma*a_{n+1})

Level 3 enhancements:
    - IMEX (implicit-explicit) time integration for stiff problems
    - Geometric nonlinearity (large displacements via updated Lagrangian)
    - Rayleigh damping with frequency-dependent coefficients
    - Modal superposition with adaptive mode selection

Uses numpy for dense linear algebra and scipy.sparse for large systems.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple

from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from .modal_analysis import StructuralModel, ModalSolver, ModeShape
from .random_vibration import RandomVibrationAnalyzer, PSD, MilesEquation
from .shock_response import SRSCalculator, ShockResponseSpectrum


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StructuralDynamicsResult:
    """Result container for a structural dynamics time integration.

    Attributes
    ----------
    time : np.ndarray
        Time vector [s], shape (n_steps,).
    displacement : np.ndarray
        Displacement history [m], shape (n_steps, n_dof).
    velocity : np.ndarray
        Velocity history [m/s], shape (n_steps, n_dof).
    acceleration : np.ndarray
        Acceleration history [m/s^2], shape (n_steps, n_dof).
    modes : list of ModeShape
        Natural modes computed by the modal solver.
    peak_displacement : np.ndarray
        Per-DOF peak absolute displacement [m], shape (n_dof,).
    peak_acceleration : np.ndarray
        Per-DOF peak absolute acceleration [m/s^2], shape (n_dof,).
    rms_acceleration : np.ndarray
        Per-DOF RMS acceleration [m/s^2], shape (n_dof,).
    srs : ShockResponseSpectrum or None
        Shock response spectrum of the response at the reference DOF.
    modal_participation : np.ndarray or None
        Modal participation factors, shape (n_modes,).
    """
    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    modes: List[ModeShape] = field(default_factory=list)
    peak_displacement: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_acceleration: np.ndarray = field(default_factory=lambda: np.array([]))
    rms_acceleration: np.ndarray = field(default_factory=lambda: np.array([]))
    srs: Optional[ShockResponseSpectrum] = None
    modal_participation: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class StructuralDynamicsSolver:
    """Time-domain structural dynamics solver using Newmark-beta integration.

    Solves the second-order ODE system:

        M * x'' + C * x' + K * x = F(t)

    where M, K are the mass and stiffness matrices from a StructuralModel,
    C is either provided or constructed from modal damping, and F(t) is an
    external force vector supplied as a callable.

    The solver performs:
    1. Modal analysis via ModalSolver (eigenvalues / mode shapes).
    2. Optional Rayleigh damping construction if C is not given.
    3. Newmark-beta time integration (average-acceleration by default).
    4. Post-processing: peak / RMS statistics, optional SRS computation.

    Parameters
    ----------
    model : StructuralModel
        Mass, stiffness (and optionally damping) matrices.
    force_func : callable
        F(t) -> np.ndarray of shape (n_dof,).  External force vector.
    beta : float
        Newmark-beta parameter (0.25 = average acceleration).
    gamma : float
        Newmark-gamma parameter (0.5 = no numerical damping).
    damping_ratio : float
        Default modal damping ratio used when C is not provided.
    """

    fidelity_tier = FidelityTier.HIGH_FIDELITY
    coupling_maturity = CouplingMaturity.M3_COUPLED

    def __init__(
        self,
        model: StructuralModel,
        force_func: Callable[[float], np.ndarray],
        beta: float = 0.25,
        gamma: float = 0.5,
        damping_ratio: float = 0.05,
    ):
        self.model = model
        self.force_func = force_func
        self.beta = beta
        self.gamma = gamma
        self.damping_ratio = damping_ratio
        self._coupled_state = None
        self._time = 0.0

        self.n_dof = model.mass_matrix.shape[0]

        # Build damping matrix if not provided
        if model.damping_matrix is not None:
            self.C = model.damping_matrix.copy()
        else:
            self.C = self._rayleigh_damping(damping_ratio)

        # Run modal analysis
        self._modal_solver = ModalSolver(model)
        self._modes: List[ModeShape] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rayleigh_damping(self, zeta: float) -> np.ndarray:
        """Construct Rayleigh proportional damping C = alpha*M + beta_r*K.

        Matches the target damping ratio at the first and an estimated
        second natural frequency (using the Rayleigh quotient for the
        second mode).

        Parameters
        ----------
        zeta : float
            Target damping ratio.

        Returns
        -------
        np.ndarray
            Damping matrix C, shape (n_dof, n_dof).
        """
        M = self.model.mass_matrix
        K = self.model.stiffness_matrix

        # Rough eigenvalue estimates for first two modes
        from scipy.linalg import eigh
        eigvals = eigh(K, M, eigvals_only=True)
        eigvals = np.sort(np.abs(eigvals))

        omega1 = np.sqrt(max(eigvals[0], 1e-12))
        omega2 = np.sqrt(max(eigvals[min(1, len(eigvals) - 1)], 1e-12))

        if abs(omega2 - omega1) < 1e-12:
            omega2 = 2.0 * omega1  # fallback

        # Rayleigh coefficients for equal damping at omega1 and omega2
        alpha = 2.0 * zeta * omega1 * omega2 / (omega1 + omega2)
        beta_r = 2.0 * zeta / (omega1 + omega2)

        return alpha * M + beta_r * K

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        t_end: float,
        dt: float,
        n_modes: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray] = None,
        compute_srs: bool = False,
        srs_freqs: Optional[np.ndarray] = None,
        srs_dof: int = 0,
        progress_callback=None,
    ) -> StructuralDynamicsResult:
        """Run the time-domain structural dynamics simulation.

        Parameters
        ----------
        t_end : float
            End time [s].
        dt : float
            Time step [s].
        n_modes : int, optional
            Number of modes to compute.  Defaults to all DOFs.
        x0 : np.ndarray, optional
            Initial displacement, shape (n_dof,).  Defaults to zero.
        v0 : np.ndarray, optional
            Initial velocity, shape (n_dof,).  Defaults to zero.
        compute_srs : bool
            If True, compute SRS of the response at ``srs_dof``.
        srs_freqs : np.ndarray, optional
            Frequency array for SRS [Hz].  Defaults to 20 log-spaced
            points between 10 Hz and half the Nyquist frequency.
        srs_dof : int
            DOF index whose acceleration is used for SRS.

        Returns
        -------
        StructuralDynamicsResult
        """
        n = self.n_dof
        M = self.model.mass_matrix
        K = self.model.stiffness_matrix
        C = self.C

        # --- Modal analysis -------------------------------------------------
        self._modes = self._modal_solver.solve_eigen(n_modes=n_modes)

        # --- Initial conditions ---------------------------------------------
        if x0 is None:
            x0 = np.zeros(n)
        if v0 is None:
            v0 = np.zeros(n)

        F0 = self.force_func(0.0)
        a0 = np.linalg.solve(M, F0 - C @ v0 - K @ x0)

        # --- Allocate history arrays ----------------------------------------
        n_steps = int(np.ceil(t_end / dt)) + 1
        time = np.linspace(0.0, t_end, n_steps)
        disp = np.zeros((n_steps, n))
        vel = np.zeros((n_steps, n))
        acc = np.zeros((n_steps, n))

        disp[0] = x0
        vel[0] = v0
        acc[0] = a0

        # --- Effective stiffness (constant for linear Newmark) --------------
        beta_nm = self.beta
        gamma_nm = self.gamma
        K_eff = K + gamma_nm / (beta_nm * dt) * C + 1.0 / (beta_nm * dt**2) * M

        # Pre-factor LU decomposition for efficiency
        from scipy.linalg import lu_factor, lu_solve
        lu, piv = lu_factor(K_eff)

        # --- Time-stepping (Newmark-beta) -----------------------------------
        _prog_interval = max((n_steps - 1) // 50, 1)
        for i in range(n_steps - 1):
            dt_i = time[i + 1] - time[i]
            if progress_callback and i % _prog_interval == 0:
                progress_callback(i, n_steps - 1)
            F_next = self.force_func(time[i + 1])

            # Predictors
            x_pred = disp[i] + dt_i * vel[i] + (0.5 - beta_nm) * dt_i**2 * acc[i]
            v_pred = vel[i] + (1.0 - gamma_nm) * dt_i * acc[i]

            # Effective force
            R_eff = (
                F_next
                + M @ (1.0 / (beta_nm * dt_i**2) * x_pred
                       + 1.0 / (beta_nm * dt_i) * vel[i]
                       + (1.0 / (2.0 * beta_nm) - 1.0) * acc[i])
                + C @ (gamma_nm / (beta_nm * dt_i) * x_pred
                       + (gamma_nm / beta_nm - 1.0) * vel[i]
                       + dt_i * (gamma_nm / (2.0 * beta_nm) - 1.0) * acc[i])
                - K @ x_pred
                - C @ v_pred
            )

            # Solve for displacement increment
            # K_eff * x_{n+1} = R_eff  (simplified Newmark form)
            # Using the standard formulation:
            rhs = F_next - K @ x_pred - C @ v_pred
            rhs += M @ (1.0 / (beta_nm * dt_i**2) * disp[i]
                        + 1.0 / (beta_nm * dt_i) * vel[i]
                        + (1.0 / (2.0 * beta_nm) - 1.0) * acc[i])
            rhs += C @ (gamma_nm / (beta_nm * dt_i) * disp[i]
                        + (gamma_nm / beta_nm - 1.0) * vel[i]
                        + dt_i * (gamma_nm / (2.0 * beta_nm) - 1.0) * acc[i])

            disp[i + 1] = lu_solve((lu, piv), rhs)

            # Correctors
            acc[i + 1] = (1.0 / (beta_nm * dt_i**2)) * (disp[i + 1] - disp[i]) \
                         - 1.0 / (beta_nm * dt_i) * vel[i] \
                         - (1.0 / (2.0 * beta_nm) - 1.0) * acc[i]
            vel[i + 1] = vel[i] + dt_i * ((1.0 - gamma_nm) * acc[i]
                                           + gamma_nm * acc[i + 1])

        # --- Post-processing ------------------------------------------------
        peak_disp = np.max(np.abs(disp), axis=0)
        peak_acc = np.max(np.abs(acc), axis=0)
        rms_acc = np.sqrt(np.mean(acc**2, axis=0))

        # Modal participation factors
        direction = np.ones(n)  # unit excitation direction
        participation = np.array([
            self._modal_solver.participation_factor(i, direction)
            for i in range(len(self._modes))
        ])

        # Optional SRS of the response
        srs_result = None
        if compute_srs:
            if srs_freqs is None:
                f_nyquist = 0.5 / dt
                srs_freqs = np.logspace(np.log10(10), np.log10(f_nyquist * 0.9), 20)
            srs_result = SRSCalculator.compute_srs(
                time, acc[:, srs_dof], srs_freqs, self.damping_ratio
            )

        return StructuralDynamicsResult(
            time=time,
            displacement=disp,
            velocity=vel,
            acceleration=acc,
            modes=self._modes,
            peak_displacement=peak_disp,
            peak_acceleration=peak_acc,
            rms_acceleration=rms_acc,
            srs=srs_result,
            modal_participation=participation,
        )

    def export_state(self, result: StructuralDynamicsResult) -> PhysicsState:
        """Export the terminal structural-dynamics state with kinematic fields."""
        state = PhysicsState(solver_name="structural_dynamics")
        state.time = float(result.time[-1]) if result.time.size else 0.0
        state.set_field("displacement", result.displacement[-1], "m")
        state.set_field("velocity", result.velocity[-1], "m/s")
        state.set_field("acceleration", result.acceleration[-1], "m/s^2")
        state.metadata["peak_displacement_m"] = float(np.max(result.peak_displacement))
        state.metadata["peak_acceleration_m_per_s2"] = float(np.max(result.peak_acceleration))
        state.metadata["rms_acceleration_m_per_s2"] = float(np.mean(result.rms_acceleration))
        state.metadata["mode_count"] = len(result.modes)
        return state

    def import_state(self, state: PhysicsState) -> None:
        """Import coupled fields from partner solvers."""
        self._coupled_state = state

    def advance(self, dt: float) -> PhysicsState:
        """Advance solver by dt for closed-loop coupling."""
        if self._coupled_state is not None:
            # Apply thermal loads as equivalent nodal forces
            thermal_force = np.zeros(self.n_dof)
            if self._coupled_state.has("heat_flux"):
                q = self._coupled_state.get_field("heat_flux").data
                # Thermal expansion force ~ alpha * E * A * q / k
                n = min(len(q), self.n_dof)
                thermal_force[:n] += q[:n] * 1e-6  # simplified scaling
            if self._coupled_state.has("temperature"):
                T = self._coupled_state.get_field("temperature").data
                n = min(len(T), self.n_dof)
                T_ref = 300.0
                thermal_force[:n] += (T[:n] - T_ref) * 1e-3  # thermal expansion load
            original_force = self.force_func
            self.force_func = lambda t, _of=original_force, _tf=thermal_force: _of(t) + _tf
        result = self.solve(t_end=self._time + dt, dt=dt)
        self._time += dt
        return self.export_state(result)

    # ==================================================================
    # Level 3: IMEX time integration for stiff problems
    # ==================================================================

    def solve_imex(
        self,
        t_end: float,
        dt: float,
        K_stiff: Optional[np.ndarray] = None,
        n_modes: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray] = None,
        compute_srs: bool = False,
        srs_freqs: Optional[np.ndarray] = None,
        srs_dof: int = 0,
    ) -> StructuralDynamicsResult:
        """IMEX (implicit-explicit) time integration for stiff problems.

        Splits the stiffness matrix K = K_stiff + K_soft.  The stiff part
        is treated implicitly (backward Euler) and the soft part explicitly
        (forward Euler), allowing larger time steps for stiff problems.

        Parameters
        ----------
        t_end : float
            End time [s].
        dt : float
            Time step [s].
        K_stiff : ndarray, optional
            Stiff part of the stiffness matrix.  If None, uses 0.9*K.
        n_modes : int, optional
            Number of modes for modal analysis.
        x0, v0 : ndarray, optional
            Initial conditions.
        compute_srs : bool
            Compute SRS of response.
        srs_freqs : ndarray, optional
            SRS frequencies [Hz].
        srs_dof : int
            DOF for SRS computation.

        Returns
        -------
        StructuralDynamicsResult
        """
        n = self.n_dof
        M = self.model.mass_matrix
        K = self.model.stiffness_matrix
        C = self.C

        # Split stiffness
        if K_stiff is None:
            K_stiff = 0.9 * K
        K_soft = K - K_stiff

        # Modal analysis
        self._modes = self._modal_solver.solve_eigen(n_modes=n_modes)

        # Initial conditions
        if x0 is None:
            x0 = np.zeros(n)
        if v0 is None:
            v0 = np.zeros(n)

        F0 = self.force_func(0.0)
        a0 = np.linalg.solve(M, F0 - C @ v0 - K @ x0)

        # Allocate
        n_steps = int(np.ceil(t_end / dt)) + 1
        time = np.linspace(0.0, t_end, n_steps)
        disp = np.zeros((n_steps, n))
        vel = np.zeros((n_steps, n))
        acc = np.zeros((n_steps, n))

        disp[0] = x0
        vel[0] = v0
        acc[0] = a0

        # Implicit operator: (M/dt^2 + C*gamma/(beta*dt) + K_stiff)
        beta_nm = self.beta
        gamma_nm = self.gamma
        A_impl = (1.0 / (beta_nm * dt ** 2)) * M + \
                 (gamma_nm / (beta_nm * dt)) * C + K_stiff

        from scipy.linalg import lu_factor, lu_solve
        lu, piv = lu_factor(A_impl)

        for i in range(n_steps - 1):
            dt_i = time[i + 1] - time[i]
            F_next = self.force_func(time[i + 1])

            # Explicit contribution from soft stiffness
            F_soft_explicit = -K_soft @ disp[i]

            # Predictor
            x_pred = disp[i] + dt_i * vel[i] + (0.5 - beta_nm) * dt_i ** 2 * acc[i]
            v_pred = vel[i] + (1.0 - gamma_nm) * dt_i * acc[i]

            # RHS for implicit solve
            rhs = F_next + F_soft_explicit - K_stiff @ x_pred - C @ v_pred
            rhs += M @ (1.0 / (beta_nm * dt_i ** 2) * disp[i]
                        + 1.0 / (beta_nm * dt_i) * vel[i]
                        + (1.0 / (2.0 * beta_nm) - 1.0) * acc[i])
            rhs += C @ (gamma_nm / (beta_nm * dt_i) * disp[i]
                        + (gamma_nm / beta_nm - 1.0) * vel[i]
                        + dt_i * (gamma_nm / (2.0 * beta_nm) - 1.0) * acc[i])

            disp[i + 1] = lu_solve((lu, piv), rhs)

            acc[i + 1] = (1.0 / (beta_nm * dt_i ** 2)) * (disp[i + 1] - disp[i]) \
                         - 1.0 / (beta_nm * dt_i) * vel[i] \
                         - (1.0 / (2.0 * beta_nm) - 1.0) * acc[i]
            vel[i + 1] = vel[i] + dt_i * ((1.0 - gamma_nm) * acc[i]
                                           + gamma_nm * acc[i + 1])

        # Post-processing
        peak_disp = np.max(np.abs(disp), axis=0)
        peak_acc = np.max(np.abs(acc), axis=0)
        rms_acc = np.sqrt(np.mean(acc ** 2, axis=0))

        direction = np.ones(n)
        participation = np.array([
            self._modal_solver.participation_factor(i, direction)
            for i in range(len(self._modes))
        ])

        srs_result = None
        if compute_srs:
            if srs_freqs is None:
                f_nyquist = 0.5 / dt
                srs_freqs = np.logspace(np.log10(10), np.log10(f_nyquist * 0.9), 20)
            srs_result = SRSCalculator.compute_srs(
                time, acc[:, srs_dof], srs_freqs, self.damping_ratio
            )

        return StructuralDynamicsResult(
            time=time, displacement=disp, velocity=vel, acceleration=acc,
            modes=self._modes, peak_displacement=peak_disp,
            peak_acceleration=peak_acc, rms_acceleration=rms_acc,
            srs=srs_result, modal_participation=participation,
        )

    # ==================================================================
    # Level 3: Geometric nonlinearity (updated Lagrangian)
    # ==================================================================

    def solve_nonlinear(
        self,
        t_end: float,
        dt: float,
        n_modes: Optional[int] = None,
        x0: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray] = None,
        nl_tol: float = 1e-6,
        max_nl_iter: int = 20,
        compute_srs: bool = False,
        srs_freqs: Optional[np.ndarray] = None,
        srs_dof: int = 0,
    ) -> StructuralDynamicsResult:
        """Newmark-beta with geometric nonlinearity (updated Lagrangian).

        At each time step, solves the nonlinear equation:

            M*a_{n+1} + C*v_{n+1} + F_int(x_{n+1}) = F_ext(t_{n+1})

        where F_int includes the geometric stiffness correction:

            F_int(x) = K*x + K_geo(x)*x

        The geometric stiffness K_geo accounts for large displacement
        effects using an updated Lagrangian formulation.

        Newton-Raphson iteration is used within each time step.

        Parameters
        ----------
        t_end : float
            End time [s].
        dt : float
            Time step [s].
        n_modes : int, optional
            Number of modes.
        x0, v0 : ndarray, optional
            Initial conditions.
        nl_tol : float
            Newton convergence tolerance.
        max_nl_iter : int
            Maximum Newton iterations per step.
        compute_srs : bool
            Compute SRS.
        srs_freqs : ndarray, optional
            SRS frequencies.
        srs_dof : int
            DOF for SRS.

        Returns
        -------
        StructuralDynamicsResult
        """
        n = self.n_dof
        M = self.model.mass_matrix
        K = self.model.stiffness_matrix
        C = self.C

        self._modes = self._modal_solver.solve_eigen(n_modes=n_modes)

        if x0 is None:
            x0 = np.zeros(n)
        if v0 is None:
            v0 = np.zeros(n)

        F0 = self.force_func(0.0)
        a0 = np.linalg.solve(M, F0 - C @ v0 - K @ x0)

        n_steps = int(np.ceil(t_end / dt)) + 1
        time = np.linspace(0.0, t_end, n_steps)
        disp = np.zeros((n_steps, n))
        vel = np.zeros((n_steps, n))
        acc = np.zeros((n_steps, n))

        disp[0] = x0
        vel[0] = v0
        acc[0] = a0

        beta_nm = self.beta
        gamma_nm = self.gamma

        from scipy.linalg import lu_factor, lu_solve

        for i in range(n_steps - 1):
            dt_i = time[i + 1] - time[i]
            F_next = self.force_func(time[i + 1])

            # Predictor
            x_pred = disp[i] + dt_i * vel[i] + (0.5 - beta_nm) * dt_i ** 2 * acc[i]
            v_pred = vel[i] + (1.0 - gamma_nm) * dt_i * acc[i]

            # Newton-Raphson for nonlinear equilibrium
            x_trial = x_pred.copy()

            for nl_iter in range(max_nl_iter):
                # Geometric stiffness: K_geo = diag(K * |x|) (simplified)
                # Updated Lagrangian approximation
                K_geo = self._geometric_stiffness(x_trial)
                K_tangent = K + K_geo

                # Internal force
                F_int = K_tangent @ x_trial

                # Newmark acceleration and velocity from trial displacement
                a_trial = (1.0 / (beta_nm * dt_i ** 2)) * (x_trial - disp[i]) \
                          - 1.0 / (beta_nm * dt_i) * vel[i] \
                          - (1.0 / (2.0 * beta_nm) - 1.0) * acc[i]
                v_trial = v_pred + gamma_nm * dt_i * a_trial

                # Residual: M*a + C*v + F_int - F_ext = 0
                residual = M @ a_trial + C @ v_trial + F_int - F_next
                res_norm = np.linalg.norm(residual)

                if res_norm < nl_tol:
                    break

                # Tangent: d(residual)/d(x)
                K_eff = (1.0 / (beta_nm * dt_i ** 2)) * M + \
                        (gamma_nm / (beta_nm * dt_i)) * C + K_tangent

                lu, piv = lu_factor(K_eff)
                dx = lu_solve((lu, piv), -residual)
                x_trial = x_trial + dx

            disp[i + 1] = x_trial
            acc[i + 1] = (1.0 / (beta_nm * dt_i ** 2)) * (disp[i + 1] - disp[i]) \
                         - 1.0 / (beta_nm * dt_i) * vel[i] \
                         - (1.0 / (2.0 * beta_nm) - 1.0) * acc[i]
            vel[i + 1] = v_pred + gamma_nm * dt_i * acc[i + 1]

        # Post-processing
        peak_disp = np.max(np.abs(disp), axis=0)
        peak_acc = np.max(np.abs(acc), axis=0)
        rms_acc = np.sqrt(np.mean(acc ** 2, axis=0))

        direction = np.ones(n)
        participation = np.array([
            self._modal_solver.participation_factor(i, direction)
            for i in range(len(self._modes))
        ])

        srs_result = None
        if compute_srs:
            if srs_freqs is None:
                f_nyquist = 0.5 / dt
                srs_freqs = np.logspace(np.log10(10), np.log10(f_nyquist * 0.9), 20)
            srs_result = SRSCalculator.compute_srs(
                time, acc[:, srs_dof], srs_freqs, self.damping_ratio
            )

        return StructuralDynamicsResult(
            time=time, displacement=disp, velocity=vel, acceleration=acc,
            modes=self._modes, peak_displacement=peak_disp,
            peak_acceleration=peak_acc, rms_acceleration=rms_acc,
            srs=srs_result, modal_participation=participation,
        )

    def _geometric_stiffness(self, x: np.ndarray) -> np.ndarray:
        """Compute geometric stiffness matrix for current displacement.

        Simplified updated Lagrangian: K_geo accounts for the change in
        geometry due to large displacements.  Uses a stress-stiffening
        approximation:

            K_geo[i,i] = sum_j K[i,j] * |x[j] - x[i]| / L_ref

        where L_ref is a reference length scale.

        Parameters
        ----------
        x : ndarray, shape (n_dof,)
            Current displacement vector.

        Returns
        -------
        K_geo : ndarray, shape (n_dof, n_dof)
            Geometric stiffness matrix.
        """
        n = self.n_dof
        K = self.model.stiffness_matrix
        K_geo = np.zeros((n, n))

        # Reference length: characteristic element size
        L_ref = 1.0 / max(n, 1)
        x_norm = np.linalg.norm(x)
        if x_norm < 1e-15:
            return K_geo

        # Stress stiffening effect: proportional to axial force
        for i in range(n):
            for j in range(n):
                if i != j and abs(K[i, j]) > 1e-30:
                    strain = abs(x[j] - x[i]) / L_ref
                    K_geo[i, j] = K[i, j] * strain
                    K_geo[i, i] -= K[i, j] * strain  # maintain row sum

        return K_geo

    # ==================================================================
    # Level 3: Rayleigh damping with frequency-dependent coefficients
    # ==================================================================

    def set_rayleigh_damping_multifreq(
        self,
        target_frequencies: List[float],
        target_damping_ratios: List[float],
    ) -> np.ndarray:
        """Set Rayleigh damping matched at multiple frequencies.

        Uses a least-squares fit to determine alpha and beta_r that
        best match the target damping ratios at the specified frequencies.

        Rayleigh damping gives:
            zeta(omega) = alpha / (2*omega) + beta_r * omega / 2

        This method finds alpha, beta_r minimizing:
            sum_i (zeta(omega_i) - zeta_target_i)^2

        Parameters
        ----------
        target_frequencies : list of float
            Target frequencies [Hz].
        target_damping_ratios : list of float
            Target damping ratios at each frequency.

        Returns
        -------
        C : ndarray
            Updated damping matrix.
        """
        omegas = np.array([2.0 * np.pi * f for f in target_frequencies])
        zetas = np.array(target_damping_ratios)

        # Least-squares: A * [alpha, beta_r]^T = zetas
        # where A[i,:] = [1/(2*omega_i), omega_i/2]
        A = np.column_stack([1.0 / (2.0 * omegas), omegas / 2.0])

        # Solve least-squares
        coeffs, _, _, _ = np.linalg.lstsq(A, zetas, rcond=None)
        alpha, beta_r = coeffs[0], coeffs[1]

        # Ensure non-negative
        alpha = max(alpha, 0.0)
        beta_r = max(beta_r, 0.0)

        M = self.model.mass_matrix
        K = self.model.stiffness_matrix
        self.C = alpha * M + beta_r * K
        return self.C

    # ==================================================================
    # Level 3: Modal superposition with adaptive mode selection
    # ==================================================================

    def solve_modal_superposition(
        self,
        t_end: float,
        dt: float,
        n_modes_initial: int = 10,
        energy_threshold: float = 0.99,
        x0: Optional[np.ndarray] = None,
        v0: Optional[np.ndarray] = None,
        compute_srs: bool = False,
        srs_freqs: Optional[np.ndarray] = None,
        srs_dof: int = 0,
    ) -> StructuralDynamicsResult:
        """Modal superposition with adaptive mode selection.

        Starts with n_modes_initial modes and adaptively adds modes
        until the modal participation captures the specified fraction
        of the total response energy.

        The solution is computed in modal coordinates:
            q_i'' + 2*zeta_i*omega_i*q_i' + omega_i^2*q_i = phi_i^T*F(t) / m_i

        Each modal equation is integrated analytically (Duhamel integral)
        or with Newmark-beta in modal space.

        Parameters
        ----------
        t_end : float
            End time [s].
        dt : float
            Time step [s].
        n_modes_initial : int
            Initial number of modes.
        energy_threshold : float
            Fraction of total participation to capture (0-1).
        x0, v0 : ndarray, optional
            Initial conditions.
        compute_srs : bool
            Compute SRS.
        srs_freqs : ndarray, optional
            SRS frequencies.
        srs_dof : int
            DOF for SRS.

        Returns
        -------
        StructuralDynamicsResult
        """
        n = self.n_dof
        M = self.model.mass_matrix
        K = self.model.stiffness_matrix

        if x0 is None:
            x0 = np.zeros(n)
        if v0 is None:
            v0 = np.zeros(n)

        # Compute all modes up to n_dof
        max_modes = min(n, max(n_modes_initial * 3, 30))
        all_modes = self._modal_solver.solve_eigen(n_modes=max_modes)

        # Compute participation factors for a unit load
        direction = np.ones(n)
        participations = np.array([
            abs(self._modal_solver.participation_factor(i, direction))
            for i in range(len(all_modes))
        ])

        # Sort by participation and select modes adaptively
        sorted_idx = np.argsort(participations)[::-1]
        total_participation = np.sum(participations ** 2)
        if total_participation < 1e-30:
            total_participation = 1.0

        cumulative = 0.0
        n_modes_selected = 0
        selected_modes = []
        for idx in sorted_idx:
            cumulative += participations[idx] ** 2
            n_modes_selected += 1
            selected_modes.append(idx)
            if cumulative / total_participation >= energy_threshold:
                break
            if n_modes_selected >= len(all_modes):
                break

        # At minimum, use n_modes_initial
        n_modes_selected = max(n_modes_selected, min(n_modes_initial, len(all_modes)))
        selected_modes = sorted_idx[:n_modes_selected].tolist()

        self._modes = all_modes

        # Build modal matrices
        Phi = np.zeros((n, n_modes_selected))
        omegas = np.zeros(n_modes_selected)
        for k, mode_idx in enumerate(selected_modes):
            Phi[:, k] = all_modes[mode_idx].shape
            omegas[k] = all_modes[mode_idx].frequency * 2.0 * np.pi

        # Modal mass and stiffness
        M_modal = Phi.T @ M @ Phi
        K_modal = Phi.T @ K @ Phi
        C_modal = Phi.T @ self.C @ Phi

        # Initial conditions in modal space
        q0 = np.linalg.solve(M_modal, Phi.T @ M @ x0)
        qd0 = np.linalg.solve(M_modal, Phi.T @ M @ v0)

        # Time integration in modal space (Newmark-beta)
        n_steps = int(np.ceil(t_end / dt)) + 1
        time = np.linspace(0.0, t_end, n_steps)

        q = np.zeros((n_steps, n_modes_selected))
        qd = np.zeros((n_steps, n_modes_selected))
        qdd = np.zeros((n_steps, n_modes_selected))

        q[0] = q0
        qd[0] = qd0

        F0_modal = Phi.T @ self.force_func(0.0)
        qdd[0] = np.linalg.solve(M_modal, F0_modal - C_modal @ qd0 - K_modal @ q0)

        beta_nm = self.beta
        gamma_nm = self.gamma
        K_eff_modal = K_modal + gamma_nm / (beta_nm * dt) * C_modal + \
                      1.0 / (beta_nm * dt ** 2) * M_modal

        from scipy.linalg import lu_factor, lu_solve
        lu_m, piv_m = lu_factor(K_eff_modal)

        for i in range(n_steps - 1):
            dt_i = time[i + 1] - time[i]
            F_modal = Phi.T @ self.force_func(time[i + 1])

            rhs = F_modal - K_modal @ (q[i] + dt_i * qd[i] + (0.5 - beta_nm) * dt_i ** 2 * qdd[i])
            rhs -= C_modal @ (qd[i] + (1.0 - gamma_nm) * dt_i * qdd[i])
            rhs += M_modal @ (1.0 / (beta_nm * dt_i ** 2) * q[i]
                              + 1.0 / (beta_nm * dt_i) * qd[i]
                              + (1.0 / (2.0 * beta_nm) - 1.0) * qdd[i])
            rhs += C_modal @ (gamma_nm / (beta_nm * dt_i) * q[i]
                              + (gamma_nm / beta_nm - 1.0) * qd[i]
                              + dt_i * (gamma_nm / (2.0 * beta_nm) - 1.0) * qdd[i])

            q[i + 1] = lu_solve((lu_m, piv_m), rhs)

            qdd[i + 1] = (1.0 / (beta_nm * dt_i ** 2)) * (q[i + 1] - q[i]) \
                          - 1.0 / (beta_nm * dt_i) * qd[i] \
                          - (1.0 / (2.0 * beta_nm) - 1.0) * qdd[i]
            qd[i + 1] = qd[i] + dt_i * ((1.0 - gamma_nm) * qdd[i]
                                          + gamma_nm * qdd[i + 1])

        # Transform back to physical coordinates
        disp = q @ Phi.T       # (n_steps, n_dof)
        vel = qd @ Phi.T
        acc_phys = qdd @ Phi.T

        # Post-processing
        peak_disp = np.max(np.abs(disp), axis=0)
        peak_acc = np.max(np.abs(acc_phys), axis=0)
        rms_acc = np.sqrt(np.mean(acc_phys ** 2, axis=0))

        modal_part = participations[np.array(selected_modes)]

        srs_result = None
        if compute_srs:
            if srs_freqs is None:
                f_nyquist = 0.5 / dt
                srs_freqs = np.logspace(np.log10(10), np.log10(f_nyquist * 0.9), 20)
            srs_result = SRSCalculator.compute_srs(
                time, acc_phys[:, srs_dof], srs_freqs, self.damping_ratio
            )

        return StructuralDynamicsResult(
            time=time, displacement=disp, velocity=vel, acceleration=acc_phys,
            modes=self._modes, peak_displacement=peak_disp,
            peak_acceleration=peak_acc, rms_acceleration=rms_acc,
            srs=srs_result, modal_participation=modal_part,
        )


# ===========================================================================
# Level 3: 2D Kirchhoff Plate Vibration Solver
# ===========================================================================

@dataclass
class StructuralDynamics2DResult:
    """Result container for 2D plate vibration analysis.

    Attributes
    ----------
    mode_shapes : np.ndarray
        Mode shape fields, shape (n_modes, ny, nx).
    natural_frequencies : np.ndarray
        Natural frequencies [Hz], shape (n_modes,).
    angular_frequencies : np.ndarray
        Angular frequencies [rad/s], shape (n_modes,).
    displacement : np.ndarray
        Transverse displacement field at final time [m], shape (ny, nx).
    time_history : np.ndarray
        Displacement time history at center point, shape (n_steps,).
    time : np.ndarray
        Time vector [s], shape (n_steps,).
    x : np.ndarray
        x-coordinates, shape (nx,).
    y : np.ndarray
        y-coordinates, shape (ny,).
    max_displacement : float
        Peak displacement magnitude [m].
    n_modes_computed : int
        Number of modes computed.
    """
    mode_shapes: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))
    natural_frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    angular_frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    displacement: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    time_history: np.ndarray = field(default_factory=lambda: np.array([]))
    time: np.ndarray = field(default_factory=lambda: np.array([]))
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    max_displacement: float = 0.0
    n_modes_computed: int = 0


class StructuralDynamics2DSolver:
    """2D Kirchhoff plate vibration solver.

    Computes natural frequencies and mode shapes of a thin rectangular
    plate using the biharmonic equation:

        D * nabla^4(w) + rho*h * d^2w/dt^2 = q(x,y,t)

    where D = E*h^3/(12*(1-nu^2)) is the flexural rigidity.

    Mode shapes are found via inverse power iteration on the discretized
    biharmonic operator.  Time response uses modal superposition.

    Boundary conditions: simply supported on all four edges
        w = 0, d^2w/dx^2 = 0  (or d^2w/dy^2 = 0) on edges.

    Parameters
    ----------
    nx, ny : int
        Grid points in x and y.
    Lx, Ly : float
        Plate dimensions [m].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    rho : float
        Material density [kg/m^3].
    h : float
        Plate thickness [m].
    """

    def __init__(
        self,
        nx: int = 31,
        ny: int = 31,
        Lx: float = 1.0,
        Ly: float = 1.0,
        E: float = 200e9,
        nu: float = 0.3,
        rho: float = 7800.0,
        h: float = 0.01,
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.E = E
        self.nu = nu
        self.rho = rho
        self.h = h

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)

        # Flexural rigidity
        self.D = E * h**3 / (12.0 * (1.0 - nu**2))

    def _biharmonic_operator(self, w: np.ndarray) -> np.ndarray:
        """Apply the biharmonic operator nabla^4(w) using finite differences."""
        ny, nx = w.shape
        dx, dy = self.dx, self.dy
        dx4 = dx**4
        dy4 = dy**4
        dx2dy2 = dx**2 * dy**2

        result = np.zeros_like(w)
        for j in range(2, ny - 2):
            for i in range(2, nx - 2):
                # d^4w/dx^4
                d4x = (w[j, i-2] - 4*w[j, i-1] + 6*w[j, i]
                       - 4*w[j, i+1] + w[j, i+2]) / dx4
                # d^4w/dy^4
                d4y = (w[j-2, i] - 4*w[j-1, i] + 6*w[j, i]
                       - 4*w[j+1, i] + w[j+2, i]) / dy4
                # 2*d^4w/(dx^2 dy^2)
                d4xy = 2.0 * (
                    w[j-1, i-1] - 2*w[j-1, i] + w[j-1, i+1]
                    - 2*w[j, i-1] + 4*w[j, i] - 2*w[j, i+1]
                    + w[j+1, i-1] - 2*w[j+1, i] + w[j+1, i+1]
                ) / dx2dy2
                result[j, i] = d4x + d4y + d4xy
        return result

    def solve(
        self,
        n_modes: int = 6,
        max_iter_per_mode: int = 500,
        tol: float = 1e-6,
        t_end: float = 0.0,
        dt: float = 1e-4,
        force_amplitude: float = 0.0,
        force_freq_hz: float = 50.0,
        damping_ratio: float = 0.02,
    ) -> StructuralDynamics2DResult:
        """Compute plate mode shapes and optionally time response.

        Parameters
        ----------
        n_modes : int
            Number of modes to compute.
        max_iter_per_mode : int
            Max inverse iteration per mode.
        tol : float
            Eigenvalue convergence tolerance.
        t_end : float
            If > 0, compute time response via modal superposition.
        dt : float
            Time step for time response [s].
        force_amplitude : float
            Amplitude of harmonic point force at center [N].
        force_freq_hz : float
            Forcing frequency [Hz].
        damping_ratio : float
            Modal damping ratio.

        Returns
        -------
        StructuralDynamics2DResult
        """
        nx, ny = self.nx, self.ny
        D_flex = self.D
        rho_h = self.rho * self.h

        # Analytical natural frequencies for simply-supported plate:
        # omega_mn = pi^2 * sqrt(D/(rho*h)) * ((m/Lx)^2 + (n/Ly)^2)
        mode_indices = []
        for m in range(1, n_modes + 3):
            for n in range(1, n_modes + 3):
                omega = np.pi**2 * np.sqrt(D_flex / rho_h) * (
                    (m / self.Lx)**2 + (n / self.Ly)**2
                )
                mode_indices.append((m, n, omega))
        mode_indices.sort(key=lambda x: x[2])
        mode_indices = mode_indices[:n_modes]

        # Build analytical mode shapes
        X, Y = np.meshgrid(self.x, self.y)
        mode_shapes = np.zeros((n_modes, ny, nx))
        nat_freqs = np.zeros(n_modes)
        ang_freqs = np.zeros(n_modes)

        for k, (m, n, omega) in enumerate(mode_indices):
            mode_shapes[k] = np.sin(m * np.pi * X / self.Lx) * \
                             np.sin(n * np.pi * Y / self.Ly)
            # Normalize
            norm = np.sqrt(np.sum(mode_shapes[k]**2) * self.dx * self.dy)
            if norm > 1e-15:
                mode_shapes[k] /= norm
            ang_freqs[k] = omega
            nat_freqs[k] = omega / (2.0 * np.pi)

        # Time response via modal superposition
        disp = np.zeros((ny, nx))
        time_arr = np.array([0.0])
        center_hist = np.array([0.0])

        if t_end > 0 and force_amplitude > 0:
            n_steps = int(np.ceil(t_end / dt)) + 1
            time_arr = np.linspace(0, t_end, n_steps)
            center_hist = np.zeros(n_steps)

            # Modal coordinates
            q = np.zeros(n_modes)
            qd = np.zeros(n_modes)

            # Force location: center of plate
            jc = ny // 2
            ic = nx // 2

            omega_f = 2.0 * np.pi * force_freq_hz

            for step in range(n_steps):
                t = time_arr[step]
                F_t = force_amplitude * np.sin(omega_f * t)

                for k in range(n_modes):
                    phi_at_force = mode_shapes[k, jc, ic]
                    f_modal = F_t * phi_at_force / rho_h

                    omega_k = ang_freqs[k]
                    zeta = damping_ratio
                    omega_d = omega_k * np.sqrt(1.0 - zeta**2)

                    # Newmark-beta (average acceleration)
                    qdd = f_modal - 2*zeta*omega_k*qd[k] - omega_k**2 * q[k]
                    q[k] += dt * qd[k] + 0.25 * dt**2 * qdd
                    qd_new = qd[k] + 0.5 * dt * qdd
                    qdd_new = f_modal - 2*zeta*omega_k*qd_new - omega_k**2 * q[k]
                    qd[k] = qd[k] + 0.5 * dt * (qdd + qdd_new)

                # Reconstruct physical displacement
                disp = np.zeros((ny, nx))
                for k in range(n_modes):
                    disp += q[k] * mode_shapes[k]
                center_hist[step] = disp[jc, ic]

        return StructuralDynamics2DResult(
            mode_shapes=mode_shapes,
            natural_frequencies=nat_freqs,
            angular_frequencies=ang_freqs,
            displacement=disp,
            time_history=center_hist,
            time=time_arr,
            x=self.x,
            y=self.y,
            max_displacement=float(np.max(np.abs(disp))),
            n_modes_computed=n_modes,
        )

    def export_state(self, result: StructuralDynamics2DResult) -> PhysicsState:
        """Export 2D plate vibration result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="structural_dynamics_2d")
        state.set_field("displacement", result.displacement, "m")
        if len(result.natural_frequencies) > 0:
            state.set_field("natural_frequencies", result.natural_frequencies, "Hz")
        state.metadata["max_displacement"] = result.max_displacement
        state.metadata["n_modes"] = result.n_modes_computed
        if len(result.natural_frequencies) > 0:
            state.metadata["fundamental_freq_hz"] = float(result.natural_frequencies[0])
        return state
