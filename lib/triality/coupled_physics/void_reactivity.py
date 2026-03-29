"""
Spatially-Resolved Void and Density Reactivity Feedback

Extends the basic reactivity feedback models (Layer 8) with spatially-resolved
(axially-dependent) treatment of void fraction, moderator density, and fuel
temperature (Doppler) feedback.  This is essential for accurate analysis of:

- BWR void reactivity dynamics (spatially varying void fraction)
- LOCA analysis (voiding propagation along the channel)
- Power shape distortion from non-uniform feedback
- Adjoint-weighted reactivity worth calculations

Physics Models
==============

Void Reactivity (VoidReactivityMap):
    rho_void = sum_z [ alpha_void(z) * (alpha(z) - alpha_ref(z)) * dz/L ]

    The void reactivity is computed by integrating the product of the local
    void coefficient and the local void change over the core height.  The
    void coefficient is weighted by the neutron importance (adjoint flux)
    to properly account for the greater reactivity effect of void changes
    near the centre of the core.

Moderator Density Feedback (ModeratorDensityFeedback):
    rho_mod = alpha_density * (rho_w(T,P) - rho_w(T_ref, P_ref))

    Uses a simplified IAPWS-IF97 water density correlation to compute
    the moderator density change from temperature and pressure variations.

Doppler Feedback (FuelTemperatureFeedback):
    rho_D = alpha_D * sum_z [ (sqrt(T_f(z)) - sqrt(T_f_ref(z))) * dz/L ]

    Spatially-resolved Doppler feedback using the square-root temperature
    dependence of the resonance integral.  Includes burnup correction for
    the change in Doppler coefficient with fuel depletion.

ReactivityCoupling:
    Aggregates all spatially-resolved feedback mechanisms into a single
    total reactivity.  Provides breakdown and safety checking (e.g.
    positive void coefficient detection).

Dependencies
------------
numpy

References
----------
- Duderstadt & Hamilton, "Nuclear Reactor Analysis", Ch. 6
- Todreas & Kazimi, "Nuclear Systems", Vol. I, Ch. 3
- Glasstone & Sesonske, "Nuclear Reactor Engineering", Ch. 5
- IAPWS-IF97: International Association for the Properties of Water and Steam
- ANS-19.6.1: Reload Startup Physics Tests for Pressurized Water Reactors
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable

try:
    from triality.reactivity_feedback.feedback_models import (
        ReactivityFeedbackAggregator,
        ReactivityComponents,
        FeedbackState,
        VoidFeedback,
    )
except ImportError:
    ReactivityFeedbackAggregator = None
    ReactivityComponents = None
    FeedbackState = None
    VoidFeedback = None

try:
    from triality.coupled_physics.neutronics_thermal_coupled import (
        CoupledNeutronicsThermal,
        CoupledResult,
        FeedbackMode,
    )
except ImportError:
    CoupledNeutronicsThermal = None
    CoupledResult = None
    FeedbackMode = None


# ---------------------------------------------------------------------------
# Void Reactivity Map
# ---------------------------------------------------------------------------

class VoidReactivityMap:
    """Maps a local void fraction field to reactivity.

    Computes the total void reactivity by integrating the product of
    the spatially-varying void coefficient and the local void fraction
    change over the core height, optionally weighted by the adjoint
    neutron flux (importance weighting).

    Parameters
    ----------
    n_axial : int
        Number of axial nodes.
    alpha_void_nominal : float
        Nominal void coefficient [dk/k per unit void fraction].
        Typical LWR: -1500e-5 to -3000e-5 (i.e. -1500 to -3000 pcm
        per unit void).  Default: -1500e-5.
    """

    def __init__(
        self,
        n_axial: int,
        alpha_void_nominal: float = -1500e-5,
    ):
        self.n_axial = n_axial
        self.alpha_void_nominal = alpha_void_nominal

        # Default: uniform void coefficient profile
        self._z = np.linspace(0.0, 1.0, n_axial)
        self._alpha_void_z = np.full(n_axial, alpha_void_nominal)

        # Default importance weighting: cosine (fundamental mode)
        self._importance = np.cos(np.pi * (self._z - 0.5))
        self._importance /= np.mean(self._importance)

    def set_void_coefficient_profile(
        self,
        z: np.ndarray,
        alpha_void_z: np.ndarray,
    ) -> None:
        """Set a spatially varying void coefficient profile.

        Parameters
        ----------
        z : ndarray
            Axial positions (normalised to [0, 1]).
        alpha_void_z : ndarray
            Void coefficient at each axial position [dk/k per unit void].
        """
        if len(z) != len(alpha_void_z):
            raise ValueError("z and alpha_void_z must have the same length")

        # Interpolate to internal grid
        self._alpha_void_z = np.interp(self._z, z, alpha_void_z)

    def compute_rho_void(
        self,
        alpha_z: np.ndarray,
        alpha_ref_z: Optional[np.ndarray] = None,
    ) -> float:
        """Compute total void reactivity from axial void distribution.

        rho_void = sum_z [ alpha_void(z) * (alpha(z) - alpha_ref(z))
                           * w(z) * dz / L ]

        Parameters
        ----------
        alpha_z : ndarray
            Void fraction at each axial node [0, 1].
        alpha_ref_z : ndarray or None
            Reference void fraction profile.  If None, zero void is assumed.

        Returns
        -------
        rho_void : float
            Total void reactivity [dk/k].
        """
        if len(alpha_z) != self.n_axial:
            # Interpolate to internal grid
            z_in = np.linspace(0.0, 1.0, len(alpha_z))
            alpha_z = np.interp(self._z, z_in, alpha_z)

        if alpha_ref_z is None:
            alpha_ref_z = np.zeros(self.n_axial)
        elif len(alpha_ref_z) != self.n_axial:
            z_in = np.linspace(0.0, 1.0, len(alpha_ref_z))
            alpha_ref_z = np.interp(self._z, z_in, alpha_ref_z)

        dz = 1.0 / self.n_axial
        d_alpha = alpha_z - alpha_ref_z

        # Importance-weighted integration
        rho = np.sum(self._alpha_void_z * d_alpha * self._importance * dz)

        return rho

    def importance_weighting(self, flux_shape: np.ndarray) -> None:
        """Set importance weighting from neutron flux (adjoint approximation).

        In first-order perturbation theory, the reactivity worth of a
        local change is weighted by the product of the forward and adjoint
        flux.  For a one-group approximation, this is proportional to
        phi^2 (since the adjoint equals the forward flux for a critical
        reactor in one group).

        Parameters
        ----------
        flux_shape : ndarray
            Axial flux shape (normalised or unnormalised).
        """
        if len(flux_shape) != self.n_axial:
            z_in = np.linspace(0.0, 1.0, len(flux_shape))
            flux_shape = np.interp(self._z, z_in, flux_shape)

        # Weight ~ phi * phi_adjoint ~ phi^2 (one-group)
        w = flux_shape ** 2
        mean_w = np.mean(w)
        if mean_w > 0:
            self._importance = w / mean_w
        else:
            self._importance = np.ones(self.n_axial)

    def local_contribution(self, z: Optional[np.ndarray] = None) -> np.ndarray:
        """Return void reactivity contribution at each axial node.

        This is the integrand: alpha_void(z) * importance(z) * dz / L.

        Parameters
        ----------
        z : ndarray or None
            Axial positions.  If None, uses internal grid.

        Returns
        -------
        contributions : ndarray
            Void reactivity contribution per node [dk/k per unit void
            at that node].
        """
        dz = 1.0 / self.n_axial
        return self._alpha_void_z * self._importance * dz


# ---------------------------------------------------------------------------
# Moderator Density Feedback
# ---------------------------------------------------------------------------

class ModeratorDensityFeedback:
    """Moderator density reactivity feedback with water properties.

    A more detailed model than simple temperature-coefficient feedback,
    using actual water density computed from temperature and pressure
    (simplified IAPWS-IF97 correlation).  This captures the non-linear
    density change near saturation that the linear MTC model misses.

    Model:
        rho_mod = alpha_density * (rho_w(T, P) - rho_w(T_ref, P_ref))

    Parameters
    ----------
    alpha_density : float
        Density coefficient [dk/k per (kg/m^3)].
        Typical: 30e-5 (i.e. 30 pcm per kg/m^3).  Default: 30e-5.
    """

    def __init__(self, alpha_density: float = 30e-5):
        self.alpha_density = alpha_density

    def water_density(self, T: float, P: float) -> float:
        """Compute water density from temperature and pressure.

        Simplified IAPWS-IF97 correlation for subcooled and low-quality
        two-phase water.  Valid for T < 620 K, P < 20 MPa.

        Parameters
        ----------
        T : float
            Temperature [K].
        P : float
            Pressure [MPa].

        Returns
        -------
        rho : float
            Water density [kg/m^3].
        """
        T_C = T - 273.15

        # Simplified subcooled water density correlation
        # Based on IAPWS-IF97 polynomial fit for liquid water
        rho = (
            1000.0
            - 0.0736 * T_C
            - 0.00355 * T_C ** 2
            + 1.5e-6 * T_C ** 3
        )

        # Pressure correction (compressibility)
        # d(rho)/dP ~ 0.5 kg/m^3 per MPa for liquid water
        rho += 0.5 * (P - 0.1)

        # Saturation temperature at pressure (simplified)
        T_sat = 373.15 + 25.0 * np.log(P / 0.1013) if P > 0.1013 else 373.15

        # Near saturation, density drops rapidly
        if T > T_sat * 0.95:
            # Smooth transition toward steam density
            x_quality = np.clip((T - T_sat * 0.95) / (T_sat * 0.1), 0.0, 1.0)
            rho_steam = P * 1e6 / (461.5 * T)  # Ideal gas approximation
            rho = rho * (1.0 - x_quality) + rho_steam * x_quality

        return max(rho, 1.0)

    def compute_rho_mod(
        self,
        T_mod_z: np.ndarray,
        P_z: np.ndarray,
        T_ref: float,
        P_ref: float,
    ) -> float:
        """Compute axially averaged moderator density feedback.

        Parameters
        ----------
        T_mod_z : ndarray
            Moderator temperature at each axial node [K].
        P_z : ndarray
            Pressure at each axial node [MPa].
        T_ref : float
            Reference moderator temperature [K].
        P_ref : float
            Reference pressure [MPa].

        Returns
        -------
        rho_mod : float
            Moderator density reactivity feedback [dk/k].
        """
        n_z = len(T_mod_z)
        rho_ref = self.water_density(T_ref, P_ref)

        rho_total = 0.0
        for i in range(n_z):
            rho_local = self.water_density(T_mod_z[i], P_z[i])
            drho = rho_local - rho_ref
            rho_total += self.alpha_density * drho

        # Average over axial nodes
        rho_total /= n_z

        return rho_total

    def boiling_penalty(
        self,
        void_fraction: float,
        T_mod: float,
    ) -> float:
        """Additional reactivity effect from subcooled or bulk boiling.

        When voiding occurs, there is an additional reactivity penalty
        beyond the density change, because steam has essentially zero
        moderation capability.

        Parameters
        ----------
        void_fraction : float
            Local void fraction [0, 1].
        T_mod : float
            Moderator temperature [K].

        Returns
        -------
        rho_penalty : float
            Additional negative reactivity from boiling [dk/k].
        """
        if void_fraction <= 0:
            return 0.0

        # Additional penalty: approximately -150 pcm per 10% void
        alpha_boiling = -150e-5  # dk/k per unit void
        return alpha_boiling * np.clip(void_fraction, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Fuel Temperature (Doppler) Feedback - Spatially Resolved
# ---------------------------------------------------------------------------

class FuelTemperatureFeedback:
    """Spatially-resolved Doppler (fuel temperature) reactivity feedback.

    Computes the Doppler reactivity using the square-root temperature
    model integrated axially over the core, with optional burnup
    correction.

    Model:
        rho_D = alpha_D * sum_z [ (sqrt(T_f(z)) - sqrt(T_f_ref(z))) * dz/L ]

    Parameters
    ----------
    alpha_doppler_nominal : float
        Nominal Doppler coefficient [dk/k per sqrt(K)].
        Typical PWR: -2.5e-5 to -5.0e-5.  Default: -2.5e-5.
    """

    def __init__(self, alpha_doppler_nominal: float = -2.5e-5):
        self.alpha_doppler_nominal = alpha_doppler_nominal

    def effective_fuel_temp(self, T_fuel_radial: np.ndarray) -> float:
        """Compute volume-averaged effective fuel temperature for Doppler.

        The effective Doppler temperature is a weighted average of the
        radial fuel temperature profile.  The weighting accounts for the
        fact that the fuel centre contributes less to the Doppler effect
        per unit volume than the fuel surface (due to self-shielding).

        Empirical weighting:
            T_eff = 0.3 * T_surface + 0.7 * T_center  (simplified)

        More accurately:
            T_eff = (1/V) * integral[ T(r) * w(r) * 2*pi*r dr ]

        Parameters
        ----------
        T_fuel_radial : ndarray
            Radial fuel temperature profile [K], from centre to surface.

        Returns
        -------
        T_eff : float
            Effective fuel temperature for Doppler calculation [K].
        """
        n = len(T_fuel_radial)
        if n == 1:
            return T_fuel_radial[0]

        # Radial positions normalised to [0, 1]
        r = np.linspace(0.0, 1.0, n)

        # Weighting: linear in r (volume element ~ r * dr for cylinder)
        # with additional emphasis on the outer region (self-shielding)
        w = r + 0.3  # More weight near surface
        w /= np.sum(w)

        return np.sum(T_fuel_radial * w)

    def compute_rho_doppler(
        self,
        T_fuel_z: np.ndarray,
        T_fuel_ref_z: np.ndarray,
    ) -> float:
        """Compute axially resolved Doppler feedback.

        rho_D = alpha_D * sum_z [ (sqrt(T_f(z)) - sqrt(T_f_ref(z))) * dz/L ]

        Parameters
        ----------
        T_fuel_z : ndarray
            Fuel temperature at each axial node [K].
        T_fuel_ref_z : ndarray
            Reference fuel temperature at each axial node [K].

        Returns
        -------
        rho_D : float
            Doppler reactivity feedback [dk/k].
        """
        n_z = len(T_fuel_z)

        # Ensure positive temperatures
        T_f = np.maximum(T_fuel_z, 300.0)
        T_ref = np.maximum(T_fuel_ref_z, 300.0)

        dz = 1.0 / n_z
        delta_sqrt_T = np.sqrt(T_f) - np.sqrt(T_ref)

        rho = self.alpha_doppler_nominal * np.sum(delta_sqrt_T) * dz

        return rho

    def burnup_correction(self, burnup_MWd_kgU: float) -> float:
        """Burnup correction factor for the Doppler coefficient.

        As fuel burns up, the fissile content changes (U-235 depleted,
        Pu-239 produced) and the resonance structure evolves.  The
        Doppler coefficient generally becomes less negative with burnup
        because:
        1. Less U-238 resonance absorption relative to fissile material
        2. Pu-239 resonances are different from U-238

        Parameters
        ----------
        burnup_MWd_kgU : float
            Fuel burnup [MWd/kgU].

        Returns
        -------
        correction : float
            Multiplicative correction factor for the Doppler coefficient.
            1.0 at zero burnup, decreasing with burnup.
        """
        # Empirical fit: coefficient magnitude decreases ~20% over
        # 40 MWd/kgU burnup
        if burnup_MWd_kgU <= 0:
            return 1.0

        # Linear decrease: 0.5% per MWd/kgU
        return max(0.5, 1.0 - 0.005 * burnup_MWd_kgU)


# ---------------------------------------------------------------------------
# Reactivity Coupling (aggregator)
# ---------------------------------------------------------------------------

class ReactivityCoupling:
    """Aggregates all spatially-resolved feedback mechanisms.

    Combines void, moderator density, Doppler, control rod, and boron
    reactivity contributions into a single total reactivity value.
    Provides breakdown reporting and safety checks.

    Parameters
    ----------
    n_axial : int
        Number of axial nodes for spatial resolution.
    """

    def __init__(self, n_axial: int):
        self.n_axial = n_axial

        # Individual feedback models (set via setters)
        self._void_map: Optional[VoidReactivityMap] = None
        self._moderator: Optional[ModeratorDensityFeedback] = None
        self._doppler: Optional[FuelTemperatureFeedback] = None
        self._rod_worth_fn: Optional[Callable[[float], float]] = None
        self._boron_fb_fn: Optional[Callable[[float], float]] = None

    def set_void_map(self, void_map: VoidReactivityMap) -> None:
        """Set void reactivity map.

        Parameters
        ----------
        void_map : VoidReactivityMap
            Void feedback model.
        """
        self._void_map = void_map

    def set_moderator(self, mod_fb: ModeratorDensityFeedback) -> None:
        """Set moderator density feedback model.

        Parameters
        ----------
        mod_fb : ModeratorDensityFeedback
            Moderator density feedback model.
        """
        self._moderator = mod_fb

    def set_doppler(self, doppler_fb: FuelTemperatureFeedback) -> None:
        """Set Doppler (fuel temperature) feedback model.

        Parameters
        ----------
        doppler_fb : FuelTemperatureFeedback
            Doppler feedback model.
        """
        self._doppler = doppler_fb

    def set_control_rods(self, rod_worth_fn: Callable[[float], float]) -> None:
        """Set control rod worth function.

        Parameters
        ----------
        rod_worth_fn : callable
            Function mapping normalised rod position [0, 1] to reactivity
            [dk/k].  position = 0 -> fully inserted, position = 1 -> fully
            withdrawn.
        """
        self._rod_worth_fn = rod_worth_fn

    def set_boron(self, boron_fb_fn: Callable[[float], float]) -> None:
        """Set boron reactivity feedback function.

        Parameters
        ----------
        boron_fb_fn : callable
            Function mapping boron concentration [ppm] to reactivity [dk/k].
        """
        self._boron_fb_fn = boron_fb_fn

    def compute_total(self, state_dict: dict) -> float:
        """Compute total reactivity from all feedback mechanisms.

        Parameters
        ----------
        state_dict : dict
            Dictionary of state variables.  Expected keys:
            - 'T_fuel_z' : ndarray, fuel temperature profile [K]
            - 'T_fuel_ref_z' : ndarray, reference fuel temperature [K]
            - 'T_mod_z' : ndarray, moderator temperature profile [K]
            - 'T_mod_ref' : float, reference moderator temperature [K]
            - 'alpha_void_z' : ndarray, void fraction profile [0, 1]
            - 'alpha_void_ref_z' : ndarray or None, reference void profile
            - 'P_z' : ndarray, pressure profile [MPa]
            - 'P_ref' : float, reference pressure [MPa]
            - 'rod_position' : float, normalised rod position [0, 1]
            - 'boron_ppm' : float, boron concentration [ppm]

        Returns
        -------
        rho_total : float
            Total reactivity [dk/k].
        """
        components = self.breakdown(state_dict)
        return sum(components.values())

    def breakdown(self, state_dict: dict) -> dict:
        """Return individual reactivity contributions.

        Parameters
        ----------
        state_dict : dict
            State variables (see ``compute_total`` for expected keys).

        Returns
        -------
        contributions : dict
            Dictionary with keys: 'doppler', 'moderator', 'void',
            'control_rods', 'boron'.  Values are reactivity in dk/k.
        """
        rho_doppler = 0.0
        rho_moderator = 0.0
        rho_void = 0.0
        rho_rods = 0.0
        rho_boron = 0.0

        # Doppler feedback
        if self._doppler is not None:
            T_fuel_z = state_dict.get('T_fuel_z', None)
            T_fuel_ref_z = state_dict.get('T_fuel_ref_z', None)
            if T_fuel_z is not None and T_fuel_ref_z is not None:
                rho_doppler = self._doppler.compute_rho_doppler(
                    np.asarray(T_fuel_z), np.asarray(T_fuel_ref_z),
                )

        # Moderator density feedback
        if self._moderator is not None:
            T_mod_z = state_dict.get('T_mod_z', None)
            P_z = state_dict.get('P_z', None)
            T_mod_ref = state_dict.get('T_mod_ref', 580.0)
            P_ref = state_dict.get('P_ref', 15.5)
            if T_mod_z is not None and P_z is not None:
                rho_moderator = self._moderator.compute_rho_mod(
                    np.asarray(T_mod_z), np.asarray(P_z), T_mod_ref, P_ref,
                )

        # Void feedback
        if self._void_map is not None:
            alpha_void_z = state_dict.get('alpha_void_z', None)
            alpha_void_ref_z = state_dict.get('alpha_void_ref_z', None)
            if alpha_void_z is not None:
                rho_void = self._void_map.compute_rho_void(
                    np.asarray(alpha_void_z),
                    np.asarray(alpha_void_ref_z) if alpha_void_ref_z is not None else None,
                )

        # Control rod reactivity
        if self._rod_worth_fn is not None:
            rod_position = state_dict.get('rod_position', 1.0)
            rho_rods = self._rod_worth_fn(rod_position)

        # Boron reactivity
        if self._boron_fb_fn is not None:
            boron_ppm = state_dict.get('boron_ppm', 500.0)
            rho_boron = self._boron_fb_fn(boron_ppm)

        return {
            'doppler': rho_doppler,
            'moderator': rho_moderator,
            'void': rho_void,
            'control_rods': rho_rods,
            'boron': rho_boron,
        }

    def is_positive_void_coefficient(self) -> bool:
        """Safety check: verify void coefficient is negative.

        In light water reactors, the void coefficient must be negative
        for inherent safety.  A positive void coefficient means that
        steam formation increases reactivity, leading to a positive
        feedback loop.

        Returns
        -------
        positive : bool
            True if any axial node has a positive void coefficient
            (UNSAFE condition for LWRs).
        """
        if self._void_map is None:
            return False

        return np.any(self._void_map._alpha_void_z > 0)

    def power_coefficient(
        self,
        state_dict: dict,
        delta_P: float = 0.01,
    ) -> float:
        """Compute total power coefficient (Doppler + moderator + void).

        Estimates the power coefficient by perturbing the state variables
        proportionally to a small power change and computing the resulting
        reactivity change.

        The power coefficient should be negative for stable reactor
        operation.

        Parameters
        ----------
        state_dict : dict
            Current state (see ``compute_total`` for keys).
        delta_P : float
            Fractional power perturbation (e.g. 0.01 = 1%).

        Returns
        -------
        alpha_power : float
            Power coefficient [dk/k per fraction power change].
        """
        # Compute reference reactivity
        rho_ref = self.compute_total(state_dict)

        # Perturbed state: increase fuel and moderator temperatures
        # proportionally to power increase
        perturbed = dict(state_dict)

        T_fuel_z = state_dict.get('T_fuel_z', None)
        if T_fuel_z is not None:
            T_fuel_z = np.asarray(T_fuel_z, dtype=float)
            T_fuel_ref = state_dict.get('T_fuel_ref_z', T_fuel_z)
            T_fuel_ref = np.asarray(T_fuel_ref, dtype=float)
            # Fuel temperature rise proportional to power rise
            dT_fuel = (T_fuel_z - T_fuel_ref) * delta_P + 10.0 * delta_P
            perturbed['T_fuel_z'] = T_fuel_z + dT_fuel

        T_mod_z = state_dict.get('T_mod_z', None)
        if T_mod_z is not None:
            T_mod_z = np.asarray(T_mod_z, dtype=float)
            # Moderator temperature rise (smaller than fuel)
            dT_mod = 5.0 * delta_P
            perturbed['T_mod_z'] = T_mod_z + dT_mod

        alpha_void_z = state_dict.get('alpha_void_z', None)
        if alpha_void_z is not None:
            alpha_void_z = np.asarray(alpha_void_z, dtype=float)
            # Void fraction increase with power
            d_void = 0.02 * delta_P
            perturbed['alpha_void_z'] = np.clip(alpha_void_z + d_void, 0.0, 1.0)

        # Perturbed reactivity
        rho_pert = self.compute_total(perturbed)

        # Power coefficient
        if delta_P > 0:
            return (rho_pert - rho_ref) / delta_P
        return 0.0
