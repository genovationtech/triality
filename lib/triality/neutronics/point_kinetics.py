"""
Layer 5: Reactor Neutronics - Enhanced Point Kinetics Solver

Production-ready point kinetics engine with proper six-group precursor
tracking, stiffness-aware integration, and spatial power shape coupling.

This module provides an ENHANCED point kinetics solver that improves on
the simpler version in triality.safety.point_kinetics by offering:
    - Full six-group precursor tracking via PrecursorField
    - Stiffness-aware time integration that switches between implicit
      and exponential methods depending on proximity to prompt critical
    - Spatial power distribution coupling for volumetric heat generation
    - Reactor period and inhour equation solvers

Physics Equations:
    Point kinetics with six delayed neutron groups:

        dP/dt = [(rho - beta_eff) / Lambda] * P + sum_i(lambda_i * C_i)
        dC_i/dt = (beta_i / Lambda) * P - lambda_i * C_i

    Where:
        P: Reactor power [W]
        rho: Total reactivity [delta-k/k]
        beta_eff: Total delayed neutron fraction [-]
        Lambda: Prompt neutron generation time [s]
        C_i: Precursor concentration for group i
        lambda_i: Decay constant for group i [1/s]
        beta_i: Delayed neutron fraction for group i [-]

    Inhour equation (asymptotic period relation):
        rho = Lambda / T + sum_i [ beta_i / (1 + lambda_i * T) ]

    where T is the stable reactor period.

Stiffness Handling:
    The point kinetics equations become extremely stiff near prompt
    critical (rho -> beta_eff) because the prompt neutron time scale
    (Lambda ~ 1e-5 s) is ~1000x faster than the delayed precursor
    time scales (~0.01-3 s).

    - |rho| < beta_eff (delayed regime): implicit Euler is stable and
      efficient; the delayed neutrons control the time scale.
    - |rho| >= beta_eff (prompt supercritical): an exponential integrator
      resolves the fast prompt time scale exactly, avoiding the need
      for microsecond-level time steps.

    Ref: Ott & Neuhold, "Introductory Nuclear Reactor Dynamics" (1985)
         Kinard & Allen, J. Nucl. Eng. Des. 228 (2004) 293-305

Applications:
    - Reactor startup and shutdown transients
    - Control rod worth analysis
    - Reactivity insertion accident (RIA) simulation
    - Xenon oscillation studies
    - Coupled thermal-hydraulic / neutronics transients

Accuracy Expectations:
    - Implicit method: first-order accurate, unconditionally stable
    - Exponential method: exact for prompt term, first-order for delayed
    - Period calculation: +-1% for |rho| < 0.5 * beta_eff
    - Suitable for engineering design and safety scoping

Dependencies:
    numpy
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from triality.neutronics.precursors import PrecursorField

try:
    from triality.neutronics.diffusion_solver import (
        MultiGroupDiffusion1D,
        CrossSectionSet,
        MaterialType,
        NeutronicsResult,
    )
except ImportError:
    MultiGroupDiffusion1D = None
    CrossSectionSet = None
    MaterialType = None
    NeutronicsResult = None

try:
    from triality.reactivity_feedback.feedback_models import (
        DopplerFeedback,
        ModeratorTemperatureFeedback,
        VoidFeedback,
        ReactivityFeedbackAggregator,
        ReactivityComponents,
        FeedbackState,
    )
except ImportError:
    DopplerFeedback = None
    ModeratorTemperatureFeedback = None
    VoidFeedback = None
    ReactivityFeedbackAggregator = None
    ReactivityComponents = None
    FeedbackState = None

try:
    from triality.burnup.depletion import DepletionSolver
except ImportError:
    DepletionSolver = None

try:
    from triality.burnup.isotopes import FuelComposition, Isotope
except ImportError:
    FuelComposition = None
    Isotope = None


@dataclass
class PointKineticsState:
    """Snapshot of the reactor point kinetics state at a given time

    Attributes:
        time: Current simulation time [s]
        power: Reactor power [W]
        precursor_concentrations: Delayed neutron precursor C_i array
        rho: Current total reactivity [delta-k/k]
        generation_time: Prompt neutron generation time Lambda [s]
    """
    time: float                             # [s]
    power: float                            # [W]
    precursor_concentrations: np.ndarray    # [arbitrary]
    rho: float                              # [delta-k/k]
    generation_time: float                  # [s]

    @property
    def power_MW(self) -> float:
        """Reactor power in MW"""
        return self.power * 1e-6

    @property
    def rho_pcm(self) -> float:
        """Reactivity in per cent mille (pcm)"""
        return self.rho * 1e5

    def __repr__(self) -> str:
        return (
            f"PointKineticsState(t={self.time:.6f} s, "
            f"P={self.power:.4e} W, "
            f"rho={self.rho:.6e})"
        )


class PointKineticsEngine:
    """
    Enhanced Point Kinetics Solver with Stiffness-Aware Integration

    Advances the coupled power / precursor system using a method that
    adapts to the stiffness regime:

        - Delayed regime (|rho| < beta_eff): implicit Euler, stable with
          time steps up to ~0.1 s.
        - Prompt supercritical regime (|rho| >= beta_eff): exponential
          integrator that resolves the prompt jump exactly.

    Example:
        >>> from triality.neutronics.precursors import PrecursorField
        >>> precursors = PrecursorField(n_groups=6, fuel_type='U235')
        >>> engine = PointKineticsEngine(
        ...     precursors=precursors,
        ...     generation_time=2e-5,
        ...     initial_power=3000e6
        ... )
        >>> engine.step(dt=0.01, rho_total=0.001)
        >>> print(f"Power = {engine.power:.4e} W")
        >>> print(f"Period = {engine.period():.2f} s")
    """

    def __init__(
        self,
        precursors: PrecursorField,
        generation_time: float = 1e-5,
        initial_power: float = 1.0,
    ):
        """Initialize point kinetics engine

        Args:
            precursors: PrecursorField with delayed neutron group data
            generation_time: Prompt neutron generation time Lambda [s]
            initial_power: Initial reactor power [W]
        """
        self._precursors = precursors
        self._generation_time = generation_time
        self._power = float(initial_power)
        self._time = 0.0
        self._rho = 0.0

        # Previous power for period estimation
        self._power_prev = self._power
        self._dt_prev = 1.0  # dummy initial value

        # Initialise precursors to steady state at initial power
        self._precursors.initialize_steady_state(
            power=self._power,
            generation_time=self._generation_time,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def power(self) -> float:
        """Current reactor power [W]"""
        return self._power

    @property
    def time(self) -> float:
        """Current simulation time [s]"""
        return self._time

    @property
    def generation_time(self) -> float:
        """Prompt neutron generation time Lambda [s]"""
        return self._generation_time

    @generation_time.setter
    def generation_time(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(
                f"Generation time must be positive, got {value}"
            )
        self._generation_time = value

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def step(self, dt: float, rho_total: float) -> None:
        """Advance reactor state by one time step

        Automatically selects the integration method based on the
        reactivity magnitude relative to beta_eff.

        Args:
            dt: Time step size [s]
            rho_total: Total reactivity for this step [delta-k/k]
        """
        self._rho = rho_total
        self._power_prev = self._power
        self._dt_prev = dt

        beta_eff = self._precursors.beta_eff

        if abs(rho_total) < beta_eff:
            self._implicit_step(dt, rho_total)
        else:
            self._exponential_step(dt, rho_total)

        # Ensure power stays non-negative
        self._power = max(self._power, 0.0)

        self._time += dt

    def _implicit_step(self, dt: float, rho: float) -> None:
        """Standard implicit Euler for the delayed-critical regime

        Power equation (implicit):
            P^{n+1} = [P^n + dt * S_d^n]
                      / [1 - dt * (rho - beta) / Lambda]

        Then advance precursors with the new power using the
        PrecursorField implicit step.

        This method is first-order accurate and unconditionally stable
        for |rho| < beta_eff.

        Args:
            dt: Time step [s]
            rho: Total reactivity [delta-k/k]
        """
        beta_eff = self._precursors.beta_eff
        Lambda = self._generation_time

        # Delayed neutron source from current precursor concentrations
        S_d = self._precursors.delayed_neutron_source()

        # Prompt neutron coefficient
        alpha_prompt = (rho - beta_eff) / Lambda

        # Implicit Euler for power
        denominator = 1.0 - dt * alpha_prompt
        if abs(denominator) < 1e-30:
            # Degenerate case -- fall back to explicit
            self._power = self._power + dt * (alpha_prompt * self._power + S_d)
        else:
            self._power = (self._power + dt * S_d) / denominator

        # Advance precursors with new power
        self._precursors.step(
            dt=dt,
            power=self._power,
            generation_time=Lambda,
            method='implicit',
        )

    def _exponential_step(self, dt: float, rho: float) -> None:
        """Exponential integrator for the prompt supercritical regime

        Near or above prompt critical (|rho| >= beta_eff), the prompt
        neutron time scale Lambda / (rho - beta) becomes very short.
        The exponential integrator resolves this exactly:

            P(t+dt) = P(t) * exp(alpha_prompt * dt)
                      + (S_d / alpha_prompt)
                        * (exp(alpha_prompt * dt) - 1)

        where alpha_prompt = (rho - beta) / Lambda.

        The delayed source S_d is treated as constant over the step
        (valid because precursor time scales are much longer than
        the prompt jump).

        Args:
            dt: Time step [s]
            rho: Total reactivity [delta-k/k]
        """
        beta_eff = self._precursors.beta_eff
        Lambda = self._generation_time

        alpha_prompt = (rho - beta_eff) / Lambda
        S_d = self._precursors.delayed_neutron_source()

        if abs(alpha_prompt) < 1e-30:
            # alpha ~ 0, revert to linear
            self._power = self._power + dt * S_d
        else:
            exp_factor = np.exp(alpha_prompt * dt)

            # Limit exponential growth to prevent overflow in extreme cases
            exp_factor = min(exp_factor, 1e30)

            self._power = (
                self._power * exp_factor
                + (S_d / alpha_prompt) * (exp_factor - 1.0)
            )

        # Advance precursors (analytic method better for large steps)
        self._precursors.step(
            dt=dt,
            power=self._power,
            generation_time=Lambda,
            method='analytic',
        )

    # ------------------------------------------------------------------
    # Analysis methods
    # ------------------------------------------------------------------

    def period(self) -> float:
        """Compute reactor period T = P / (dP/dt)

        The reactor period is the e-folding time of the power:
            P(t) ~ P_0 * exp(t / T)

        Positive period means increasing power; negative means decreasing.
        Returns np.inf for constant power.

        Returns:
            Reactor period [s]
        """
        if self._dt_prev <= 0.0 or self._power_prev <= 0.0:
            return np.inf

        dP_dt = (self._power - self._power_prev) / self._dt_prev

        if abs(dP_dt) < 1e-30:
            return np.inf

        return self._power / dP_dt

    def is_prompt_critical(self) -> bool:
        """Check whether the reactor is prompt critical

        The reactor is prompt critical when rho >= beta_eff.
        This is an extremely dangerous condition in which the
        power rises on the prompt neutron time scale (microseconds).

        Returns:
            True if rho >= beta_eff
        """
        return self._rho >= self._precursors.beta_eff

    def inhour(self, rho: float) -> float:
        """Compute stable reactor period from the inhour equation

        The inhour equation relates a constant reactivity insertion
        to the asymptotic stable period T:

            rho = Lambda / T + sum_i [ beta_i / (1 + lambda_i * T) ]

        This is an implicit equation in T.  We solve it iteratively
        using Newton's method.

        For |rho| < beta_eff the period is controlled by delayed
        neutrons (T ~ seconds).  For rho > beta_eff the period
        is controlled by prompt neutrons (T ~ microseconds).

        Args:
            rho: Reactivity [delta-k/k]

        Returns:
            Stable period T [s].  Positive for supercritical,
            negative for subcritical, np.inf for critical.
        """
        if abs(rho) < 1e-10:
            return np.inf

        Lambda = self._generation_time
        beta_i = self._precursors.beta_groups
        lambda_i = self._precursors.lambda_groups
        beta_eff = self._precursors.beta_eff

        # Initial guess
        if rho > beta_eff:
            # Prompt supercritical -- period is very short
            T = Lambda / (rho - beta_eff)
        elif rho > 0:
            # Delayed supercritical -- period controlled by slowest group
            T = 1.0 / (rho / beta_eff * lambda_i[0])
            T = max(T, 0.01)
        else:
            # Subcritical -- negative period
            T = -1.0 / (abs(rho) / beta_eff * lambda_i[0])
            T = min(T, -0.01)

        # Newton iteration to solve f(T) = 0 where
        # f(T) = Lambda/T + sum(beta_i / (1 + lambda_i*T)) - rho
        for _ in range(50):
            if abs(T) < 1e-30:
                break

            f = Lambda / T - rho
            df = -Lambda / (T * T)

            for i in range(len(beta_i)):
                denom = 1.0 + lambda_i[i] * T
                f += beta_i[i] / denom
                df -= beta_i[i] * lambda_i[i] / (denom * denom)

            if abs(df) < 1e-30:
                break

            dT = -f / df

            # Damped Newton step to avoid overshooting
            if abs(dT) > 0.5 * abs(T):
                dT = np.sign(dT) * 0.5 * abs(T)

            T += dT

            if abs(f) < 1e-12:
                break

        return T

    def reactivity_dollars(self) -> float:
        """Current reactivity expressed in dollars

        $ = rho / beta_eff

        One dollar of reactivity corresponds to prompt critical.

        Returns:
            Reactivity in dollars [$]
        """
        beta_eff = self._precursors.beta_eff
        if beta_eff < 1e-30:
            return 0.0
        return self._rho / beta_eff

    def delayed_neutron_source(self) -> float:
        """Current total delayed neutron source rate

        Returns:
            S_d = sum(lambda_i * C_i) [same units as power]
        """
        return self._precursors.delayed_neutron_source()

    def get_state(self) -> PointKineticsState:
        """Return a snapshot of the current reactor state

        Returns:
            PointKineticsState dataclass with all current values
        """
        return PointKineticsState(
            time=self._time,
            power=self._power,
            precursor_concentrations=self._precursors.get_concentrations(),
            rho=self._rho,
            generation_time=self._generation_time,
        )

    def __repr__(self) -> str:
        return (
            f"PointKineticsEngine(P={self._power:.4e} W, "
            f"rho={self._rho:.6e}, "
            f"Lambda={self._generation_time:.2e} s, "
            f"t={self._time:.6f} s)"
        )


class SpatialKineticsAdapter:
    """
    Bridge Between Point Kinetics and Spatial Power Distribution

    Point kinetics tracks the total reactor power P(t) but does not
    resolve spatial variation.  This adapter maintains an axial power
    shape profile P(z)/P_avg and converts the scalar total power into
    a volumetric heat generation rate q'''(z) for thermal-hydraulic
    coupling.

    The power shape can be initialised from:
        - A diffusion solution (flux_profile from MultiGroupDiffusion1D)
        - A user-specified cosine or flat profile
        - A measured or Monte Carlo power distribution

    The adapter also computes peaking factors and hot-channel power
    for safety analysis.

    Example:
        >>> adapter = SpatialKineticsAdapter(n_axial=50)
        >>> adapter.set_power_shape(np.cos(np.linspace(-np.pi/2, np.pi/2, 50)))
        >>> q_triple_prime = adapter.get_volumetric_heat(
        ...     total_power=3000e6, fuel_volume=20.0
        ... )
        >>> print(f"Peak heat rate = {np.max(q_triple_prime):.4e} W/m^3")
        >>> print(f"Axial peaking = {adapter.axial_peaking_factor():.3f}")
    """

    def __init__(
        self,
        n_axial: int,
        power_shape: Optional[np.ndarray] = None,
    ):
        """Initialize spatial kinetics adapter

        Args:
            n_axial: Number of axial nodes
            power_shape: Initial normalised axial power shape P(z)/P_avg.
                         If None, a cosine shape is used.
        """
        self.n_axial = n_axial

        if power_shape is not None:
            self.set_power_shape(power_shape)
        else:
            # Default cosine shape (fundamental mode)
            z_norm = np.linspace(-np.pi / 2, np.pi / 2, n_axial)
            cosine_shape = np.cos(z_norm)
            self.set_power_shape(cosine_shape)

    # ------------------------------------------------------------------
    # Shape management
    # ------------------------------------------------------------------

    def set_power_shape(self, shape: np.ndarray) -> None:
        """Set normalised axial power distribution P(z) / P_avg

        The shape is normalised so that its mean equals 1.0.
        This ensures that sum(shape) * dz = total length, and
        total_power = integral(q''' dV) is preserved.

        Args:
            shape: Array of length n_axial with relative power values.
                   Need not be pre-normalised; will be normalised
                   internally.

        Raises:
            ValueError: If shape has wrong length or is all zeros
        """
        shape = np.asarray(shape, dtype=float)

        if shape.shape != (self.n_axial,):
            raise ValueError(
                f"Shape must have length {self.n_axial}, "
                f"got {shape.shape}"
            )

        # Ensure non-negative
        shape = np.maximum(shape, 0.0)

        mean_val = np.mean(shape)
        if mean_val < 1e-30:
            raise ValueError("Power shape is all zeros or negative")

        # Normalise so mean = 1.0
        self._shape = shape / mean_val

    def get_power_shape(self) -> np.ndarray:
        """Return the current normalised power shape

        Returns:
            Array of shape (n_axial,) with mean = 1.0
        """
        return self._shape.copy()

    def update_shape_from_diffusion(
        self,
        flux_profile: np.ndarray,
    ) -> None:
        """Update the power shape from a diffusion solution flux profile

        The power density is proportional to the fission rate, which
        is proportional to the neutron flux in the fuel region.
        This method takes a 1D flux profile (e.g., thermal flux from
        MultiGroupDiffusion1D) and converts it to a normalised power
        shape.

        If the flux profile has a different number of points than
        n_axial, it is linearly interpolated.

        Args:
            flux_profile: 1D array of neutron flux values [n/(cm^2 s)]
        """
        flux_profile = np.asarray(flux_profile, dtype=float)
        flux_profile = np.maximum(flux_profile, 0.0)

        # Interpolate to our axial mesh if needed
        if len(flux_profile) != self.n_axial:
            x_old = np.linspace(0, 1, len(flux_profile))
            x_new = np.linspace(0, 1, self.n_axial)
            flux_profile = np.interp(x_new, x_old, flux_profile)

        self.set_power_shape(flux_profile)

    # ------------------------------------------------------------------
    # Heat generation
    # ------------------------------------------------------------------

    def get_volumetric_heat(
        self,
        total_power: float,
        fuel_volume: float,
    ) -> np.ndarray:
        """Convert total reactor power to volumetric heat rate q'''(z)

        The average volumetric heat rate is:
            q'''_avg = P_total / V_fuel

        The local value at each axial node is:
            q'''(z) = q'''_avg * shape(z)

        Args:
            total_power: Total reactor power [W]
            fuel_volume: Total fuel volume [m^3]

        Returns:
            Array of volumetric heat generation rates [W/m^3]
        """
        if fuel_volume <= 0.0:
            raise ValueError(
                f"Fuel volume must be positive, got {fuel_volume}"
            )

        q_avg = total_power / fuel_volume
        return q_avg * self._shape

    # ------------------------------------------------------------------
    # Peaking factors
    # ------------------------------------------------------------------

    def axial_peaking_factor(self) -> float:
        """Axial power peaking factor F_z

        F_z = max(shape) / mean(shape) = max(shape)

        Since the shape is normalised to mean = 1.0, the peaking
        factor is simply the maximum value.

        Returns:
            Axial peaking factor F_z [-]
        """
        return float(np.max(self._shape))

    def hot_channel_power(
        self,
        total_power: float,
        radial_peaking: float,
    ) -> float:
        """Power in the hot (limiting) fuel channel

        The hot channel is the fuel pin with the highest linear
        heat rate, determined by the product of axial and radial
        peaking factors applied to the core-average channel power.

        For a core with N_channels fuel pins:
            P_hot = P_total / N_channels * F_r * F_z

        Since we do not track individual channels, this method
        returns the product of total_power with the combined
        peaking factor, normalised per unit:

            P_hot = total_power * F_r * F_z / n_axial

        This represents the power of one axial segment of the
        hot channel.

        Args:
            total_power: Total reactor power [W]
            radial_peaking: Radial power peaking factor F_r [-]

        Returns:
            Hot channel power per axial segment [W]
        """
        F_z = self.axial_peaking_factor()
        return total_power * radial_peaking * F_z / self.n_axial

    def axial_power_distribution(
        self,
        total_power: float,
    ) -> np.ndarray:
        """Axial distribution of total power across nodes

        P(z_j) = total_power * shape(z_j) / sum(shape)

        The sum of the returned array equals total_power.

        Args:
            total_power: Total reactor power [W]

        Returns:
            Array of power per axial node [W]
        """
        return total_power * self._shape / np.sum(self._shape)

    def enthalpy_rise_factor(self) -> float:
        """Enthalpy rise hot-channel factor F_delta_H

        This is the ratio of the enthalpy rise in the hot channel
        to the core-average enthalpy rise, equal to the axial
        integral of the power shape divided by its mean:

            F_dH = mean(shape) * n / sum_over_shape = 1.0

        For our normalised shape (mean=1), F_dH equals the ratio
        of the maximum integrated power to the average:

            F_dH = max(cumsum(shape)) / mean(cumsum(shape))

        This accounts for the fact that the enthalpy rise depends
        on the integral of power from inlet to the point of interest.

        Returns:
            Enthalpy rise factor F_delta_H [-]
        """
        cumulative = np.cumsum(self._shape)
        mean_cumulative = np.mean(cumulative)
        if mean_cumulative < 1e-30:
            return 1.0
        return float(np.max(cumulative) / mean_cumulative)

    def __repr__(self) -> str:
        return (
            f"SpatialKineticsAdapter(n_axial={self.n_axial}, "
            f"F_z={self.axial_peaking_factor():.3f})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    'PointKineticsState',
    'PointKineticsEngine',
    'SpatialKineticsAdapter',
]
