"""
Layer 5: Reactor Neutronics - Delayed Neutron Precursor Tracking

Production-ready delayed neutron precursor models for reactor kinetics analysis.

Physics Equations:
    Precursor concentration evolution for group i:

        dC_i/dt = (beta_i / Lambda) * P(t) - lambda_i * C_i(t)

    Where:
        C_i: Precursor concentration for group i [arbitrary/normalised]
        beta_i: Delayed neutron fraction for group i [-]
        Lambda: Prompt neutron generation time [s]
        P(t): Reactor power (or neutron population) [W]
        lambda_i: Decay constant for group i [1/s]

    Steady-state solution:
        C_i = beta_i * P_0 / (lambda_i * Lambda)

    Delayed neutron source:
        S_d = sum_i(lambda_i * C_i)

    The delayed neutron fraction beta_eff = sum(beta_i) is typically
    ~650 pcm for U-235 thermal fission.  This small fraction controls
    the boundary between delayed-critical and prompt-critical regimes.

Photoneutron Sources:
    In heavy-water (D2O) and beryllium moderated reactors, energetic
    gamma rays can eject neutrons from deuterium or Be-9 nuclei:

        gamma + D  -> n + p   (threshold 2.226 MeV)
        gamma + Be -> n + 2He (threshold 1.666 MeV)

    These photoneutrons create additional delayed groups with very long
    half-lives (seconds to minutes), complicating shutdown behaviour.

    Ref: Glasstone & Sesonske, Sec. 5.3; Lamarsh, Sec. 7.4

Applications:
    - Point and spatial kinetics coupling
    - Reactor period estimation
    - Shutdown margin analysis with photoneutrons
    - Transient safety analysis

Accuracy Expectations:
    - Beta values: standard 6-group data, +-5% vs evaluated data
    - Precursor tracking: exact analytic or implicit Euler integration
    - Photoneutron groups: representative data, +-20%

Dependencies:
    numpy
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict

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


# ---------------------------------------------------------------------------
# Standard delayed neutron group data
# ---------------------------------------------------------------------------

# Six-group delayed neutron data for common fuel types
# Ref: Keepin, Wimett & Zeigler (1957); Brady & England (1989)
_DELAYED_NEUTRON_DATA: Dict[str, Dict[str, np.ndarray]] = {
    'U235': {
        'beta':    np.array([0.000215, 0.001424, 0.001274,
                             0.002568, 0.000748, 0.000273]),
        'lambda_': np.array([0.0124, 0.0305, 0.111,
                             0.301, 1.14, 3.01]),
    },
    'U238': {
        'beta':    np.array([0.000052, 0.000546, 0.000723,
                             0.001329, 0.000468, 0.000182]),
        'lambda_': np.array([0.0132, 0.0321, 0.139,
                             0.358, 1.41, 4.02]),
    },
    'Pu239': {
        'beta':    np.array([0.000072, 0.000626, 0.000444,
                             0.000685, 0.000181, 0.000092]),
        'lambda_': np.array([0.0129, 0.0311, 0.134,
                             0.331, 1.26, 3.21]),
    },
}


# ---------------------------------------------------------------------------
# Photoneutron group data
# ---------------------------------------------------------------------------

# Photoneutron delayed groups for D2O and Be moderators
# These groups have much longer half-lives than fission-product precursors.
# Ref: Glasstone & Sesonske, Table 5.3; Garland (2012)
_PHOTONEUTRON_DATA: Dict[str, Dict[str, np.ndarray]] = {
    'D2O': {
        # Effective yields (fraction of total photoneutron source)
        'yield_fraction': np.array([0.32, 0.25, 0.18, 0.13, 0.08, 0.04]),
        # Decay constants [1/s] -- much slower than fission precursors
        'lambda_': np.array([0.0005, 0.0018, 0.0062, 0.024, 0.089, 0.32]),
        # Gamma energy threshold [MeV]
        'threshold_MeV': 2.226,
    },
    'Be': {
        'yield_fraction': np.array([0.35, 0.28, 0.17, 0.11, 0.06, 0.03]),
        'lambda_': np.array([0.0004, 0.0015, 0.0055, 0.020, 0.075, 0.28]),
        'threshold_MeV': 1.666,
    },
}


@dataclass
class DelayedNeutronGroup:
    """Single delayed neutron precursor group

    Each group represents a collection of fission-product precursors
    with similar half-lives lumped into a single effective group.

    Attributes:
        group_index: Group number (1-6 by convention)
        beta_i: Delayed neutron fraction for this group [-]
        lambda_i: Decay constant [1/s]
        energy_spectrum: Fractional energy spectrum weight for this group [-]
    """
    group_index: int
    beta_i: float           # Delayed neutron fraction [-]
    lambda_i: float         # Decay constant [1/s]
    energy_spectrum: float = 1.0  # Spectrum weight (default: all in thermal)

    @property
    def half_life(self) -> float:
        """Precursor half-life [s]"""
        return np.log(2.0) / self.lambda_i

    @property
    def mean_life(self) -> float:
        """Precursor mean lifetime [s]"""
        return 1.0 / self.lambda_i

    def __repr__(self) -> str:
        return (f"DelayedNeutronGroup(i={self.group_index}, "
                f"beta_i={self.beta_i:.6f}, "
                f"lambda_i={self.lambda_i:.4f} 1/s, "
                f"t_half={self.half_life:.2f} s)")


class PrecursorField:
    """
    Delayed Neutron Precursor Concentration Tracker

    Maintains and evolves the six-group precursor concentrations C_i(t)
    that feed the delayed neutron source in reactor kinetics.

    The precursor equations are:
        dC_i/dt = (beta_i / Lambda) * P - lambda_i * C_i

    Two integration methods are provided:
        - 'implicit': First-order implicit Euler (unconditionally stable)
        - 'analytic': Exact analytic solution assuming P constant over dt

    Example:
        >>> precursors = PrecursorField(n_groups=6, fuel_type='U235')
        >>> precursors.initialize_steady_state(power=3000e6, generation_time=2e-5)
        >>> precursors.step(dt=0.01, power=3000e6, generation_time=2e-5)
        >>> print(f"Delayed source = {precursors.delayed_neutron_source():.4e}")
    """

    def __init__(
        self,
        n_groups: int = 6,
        fuel_type: str = 'U235',
    ):
        """Initialize precursor field with standard group data

        Args:
            n_groups: Number of delayed neutron groups (default 6)
            fuel_type: Fuel isotope key ('U235', 'U238', 'Pu239')
        """
        self.n_groups = n_groups
        self.fuel_type = fuel_type

        # Load group data
        if fuel_type not in _DELAYED_NEUTRON_DATA:
            raise ValueError(
                f"Unknown fuel type '{fuel_type}'. "
                f"Available: {list(_DELAYED_NEUTRON_DATA.keys())}"
            )

        data = _DELAYED_NEUTRON_DATA[fuel_type]
        beta_raw = data['beta']
        lambda_raw = data['lambda_']

        if n_groups != len(beta_raw):
            raise ValueError(
                f"Requested {n_groups} groups but {fuel_type} data "
                f"has {len(beta_raw)} groups"
            )

        # Build group objects
        self.groups: List[DelayedNeutronGroup] = []
        for i in range(n_groups):
            group = DelayedNeutronGroup(
                group_index=i + 1,
                beta_i=beta_raw[i],
                lambda_i=lambda_raw[i],
            )
            self.groups.append(group)

        # Arrays for fast computation
        self._beta = beta_raw.copy()
        self._lambda = lambda_raw.copy()

        # Precursor concentrations (initialised to zero)
        self._concentrations = np.zeros(n_groups)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def beta_eff(self) -> float:
        """Total effective delayed neutron fraction [-]

        This is the sum of all group beta_i values.  For U-235 thermal
        fission the standard value is approximately 0.0065 (650 pcm).
        """
        return float(np.sum(self._beta))

    @property
    def beta_groups(self) -> np.ndarray:
        """Individual group delayed neutron fractions [-]"""
        return self._beta.copy()

    @property
    def lambda_groups(self) -> np.ndarray:
        """Individual group decay constants [1/s]"""
        return self._lambda.copy()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_steady_state(
        self,
        power: float,
        generation_time: float,
    ) -> None:
        """Set precursor concentrations for steady-state equilibrium

        At equilibrium with constant power P and generation time Lambda:
            C_i = beta_i * P / (lambda_i * Lambda)

        This is the correct initial condition for transient calculations
        starting from a steady operating state.

        Args:
            power: Steady-state power (or neutron population) [W]
            generation_time: Prompt neutron generation time Lambda [s]
        """
        if generation_time <= 0.0:
            raise ValueError(
                f"Generation time must be positive, got {generation_time}"
            )

        self._concentrations = (
            self._beta * power / (self._lambda * generation_time)
        )

    # ------------------------------------------------------------------
    # Time stepping
    # ------------------------------------------------------------------

    def step(
        self,
        dt: float,
        power: float,
        generation_time: float,
        method: str = 'implicit',
    ) -> None:
        """Advance precursor concentrations by one time step

        Solves:
            dC_i/dt = (beta_i / Lambda) * P - lambda_i * C_i

        Args:
            dt: Time step size [s]
            power: Current reactor power [W]
            generation_time: Prompt neutron generation time Lambda [s]
            method: Integration method ('implicit' or 'analytic')

        Raises:
            ValueError: If method is not recognised
        """
        if method == 'implicit':
            self._step_implicit(dt, power, generation_time)
        elif method == 'analytic':
            self._step_analytic(dt, power, generation_time)
        else:
            raise ValueError(
                f"Unknown integration method '{method}'. "
                f"Use 'implicit' or 'analytic'."
            )

    def _step_implicit(
        self,
        dt: float,
        power: float,
        generation_time: float,
    ) -> None:
        """Implicit Euler step (unconditionally stable)

        C_i^{n+1} = (C_i^n + dt * beta_i * P / Lambda)
                     / (1 + dt * lambda_i)

        This is first-order accurate but unconditionally stable,
        making it suitable for stiff systems with large lambda_i * dt.
        """
        source = dt * self._beta * power / generation_time
        self._concentrations = (
            (self._concentrations + source) / (1.0 + dt * self._lambda)
        )

    def _step_analytic(
        self,
        dt: float,
        power: float,
        generation_time: float,
    ) -> None:
        """Analytic solution assuming constant power over dt

        C_i(t + dt) = C_i(t) * exp(-lambda_i * dt)
                      + (beta_i * P) / (lambda_i * Lambda)
                        * (1 - exp(-lambda_i * dt))

        This is exact when power is constant over the step, and
        second-order accurate for slowly varying power.
        """
        exp_factor = np.exp(-self._lambda * dt)
        equilibrium = self._beta * power / (self._lambda * generation_time)
        self._concentrations = (
            self._concentrations * exp_factor
            + equilibrium * (1.0 - exp_factor)
        )

    # ------------------------------------------------------------------
    # Source terms and queries
    # ------------------------------------------------------------------

    def delayed_neutron_source(self) -> float:
        """Total delayed neutron source rate

        S_d = sum_i(lambda_i * C_i)

        This term appears as the delayed source in the point kinetics
        equation: dn/dt = [(rho - beta)/Lambda]*n + S_d

        Returns:
            Delayed neutron source [same units as power]
        """
        return float(np.sum(self._lambda * self._concentrations))

    def get_concentrations(self) -> np.ndarray:
        """Return array of precursor concentrations C_i

        Returns:
            Array of shape (n_groups,) with current concentrations
        """
        return self._concentrations.copy()

    def set_concentrations(self, concentrations: np.ndarray) -> None:
        """Set precursor concentrations directly

        Args:
            concentrations: Array of shape (n_groups,)
        """
        concentrations = np.asarray(concentrations, dtype=float)
        if concentrations.shape != (self.n_groups,):
            raise ValueError(
                f"Expected shape ({self.n_groups},), "
                f"got {concentrations.shape}"
            )
        self._concentrations = concentrations.copy()

    def reactivity_worth(self) -> float:
        """Equivalent reactivity stored in precursors

        Computes the reactivity equivalent of the current precursor
        inventory relative to the steady-state value.  A positive
        value means the precursor inventory is above what steady
        state at the current delayed source rate would require.

        Returns:
            Reactivity worth [delta-k/k]
        """
        source = self.delayed_neutron_source()
        if source < 1e-30:
            return 0.0

        # At steady state, S_d = beta * P / Lambda
        # The precursor reactivity worth is proportional to the
        # ratio of actual delayed source to what it would be in
        # equilibrium.  For a rough estimate, sum(beta_i * C_i / C_i_eq).
        # Here we return the weighted fractional deviation.
        total_beta = self.beta_eff
        weighted_sum = float(np.sum(self._beta * self._lambda * self._concentrations))
        if weighted_sum < 1e-30:
            return 0.0

        # This is an approximation: rho_precursors ~ beta * (S_d / S_d_eq - 1)
        # Without knowing the reference power, return the total beta-weighted
        # fractional source contribution.
        return total_beta * source / (weighted_sum / total_beta + 1e-30)

    def group_sources(self) -> np.ndarray:
        """Individual group delayed neutron source rates

        Returns:
            Array of lambda_i * C_i for each group
        """
        return self._lambda * self._concentrations

    def __repr__(self) -> str:
        return (
            f"PrecursorField(n_groups={self.n_groups}, "
            f"fuel_type='{self.fuel_type}', "
            f"beta_eff={self.beta_eff:.6f}, "
            f"S_d={self.delayed_neutron_source():.4e})"
        )


class PhotoneutronSource:
    """
    Photoneutron Delayed Source for D2O and Beryllium Moderated Reactors

    In heavy-water and beryllium moderated reactors, high-energy gamma
    rays from fission products can eject neutrons from the moderator
    nuclei via (gamma, n) reactions.  These photoneutrons act as
    additional delayed neutron groups with decay constants determined
    by the parent fission-product gamma emitters.

    The photoneutron precursor equations are analogous to fission
    precursors:
        dP_i/dt = f_i * S_gamma - lambda_i * P_i

    where S_gamma is the total gamma source (proportional to fission
    rate), f_i is the yield fraction for group i, and lambda_i is the
    effective decay constant.

    These additional delayed groups make heavy-water reactors easier
    to control (larger effective beta) but complicate shutdown
    transients due to the very long photoneutron half-lives.

    Example:
        >>> pn = PhotoneutronSource(moderator_type='D2O')
        >>> pn.initialize_steady_state(gamma_source=1e15)
        >>> pn.step(dt=1.0, gamma_source=1e15)
        >>> print(f"Photoneutron source = {pn.source_rate():.4e}")
    """

    def __init__(self, moderator_type: str = 'D2O'):
        """Initialize photoneutron source with moderator data

        Args:
            moderator_type: Moderator material ('D2O' or 'Be')
        """
        if moderator_type not in _PHOTONEUTRON_DATA:
            raise ValueError(
                f"Unknown moderator type '{moderator_type}'. "
                f"Available: {list(_PHOTONEUTRON_DATA.keys())}"
            )

        self.moderator_type = moderator_type
        data = _PHOTONEUTRON_DATA[moderator_type]

        self._yield_fraction = data['yield_fraction'].copy()
        self._lambda = data['lambda_'].copy()
        self.threshold_MeV = data['threshold_MeV']
        self.n_groups = len(self._yield_fraction)

        # Photoneutron precursor concentrations
        self._concentrations = np.zeros(self.n_groups)

    def initialize_steady_state(self, gamma_source: float) -> None:
        """Set photoneutron precursors for steady-state equilibrium

        At equilibrium:
            P_i = f_i * S_gamma / lambda_i

        Args:
            gamma_source: Total gamma source rate [gammas/s or proportional]
        """
        self._concentrations = (
            self._yield_fraction * gamma_source / self._lambda
        )

    def step(self, dt: float, gamma_source: float) -> None:
        """Advance photoneutron precursors one time step

        Uses analytic solution assuming constant gamma source over dt:
            P_i(t+dt) = P_i(t) * exp(-lambda_i * dt)
                        + (f_i * S_gamma / lambda_i)
                          * (1 - exp(-lambda_i * dt))

        Args:
            dt: Time step [s]
            gamma_source: Current gamma source rate
        """
        exp_factor = np.exp(-self._lambda * dt)
        equilibrium = self._yield_fraction * gamma_source / self._lambda
        self._concentrations = (
            self._concentrations * exp_factor
            + equilibrium * (1.0 - exp_factor)
        )

    def source_rate(self) -> float:
        """Total photoneutron source rate

        S_pn = sum_i(lambda_i * P_i)

        Returns:
            Photoneutron source rate
        """
        return float(np.sum(self._lambda * self._concentrations))

    def get_concentrations(self) -> np.ndarray:
        """Return array of photoneutron precursor concentrations

        Returns:
            Array of shape (n_groups,)
        """
        return self._concentrations.copy()

    def effective_beta_addition(self, fission_rate: float) -> float:
        """Effective additional delayed neutron fraction from photoneutrons

        The photoneutron source acts like additional delayed neutrons,
        effectively increasing beta_eff.  This is particularly important
        for CANDU reactors (D2O moderated) where the photoneutron
        contribution can add ~0.7 mk to the effective delayed fraction.

        Args:
            fission_rate: Current fission neutron production rate

        Returns:
            Effective additional beta from photoneutrons [-]
        """
        if fission_rate < 1e-30:
            return 0.0
        return self.source_rate() / fission_rate

    def __repr__(self) -> str:
        return (
            f"PhotoneutronSource(moderator='{self.moderator_type}', "
            f"n_groups={self.n_groups}, "
            f"S_pn={self.source_rate():.4e})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    'DelayedNeutronGroup',
    'PrecursorField',
    'PhotoneutronSource',
]
