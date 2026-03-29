"""
Burnup-Dependent Reactivity Feedback Coefficients

Parametric models for the principal reactivity feedback coefficients as
functions of burnup, fuel temperature, moderator conditions, and boron
concentration.  These provide the "state-dependent" coefficients that the
existing feedback_models.py uses with constant values.

Typical uses:
- Cycle depletion analysis: how do Doppler/MTC/void coefficients change?
- Safety analysis: worst-case coefficient selection for transients
- Agent decision-making: "at what burnup does MTC go positive?"

Physics Models
==============

Doppler Coefficient vs Burnup:
    alpha_D(BU) = alpha_D_0 * (1 + c_Pu * BU)

    As fuel depletes, Pu-240 builds up.  Its resonance structure differs
    from U-238, modifying the Doppler coefficient.  At high burnup the
    Doppler coefficient becomes less negative (smaller magnitude) by
    ~15-25% relative to fresh fuel.

    Ref: Driscoll et al., "The Linear Reactivity Model", Ch. 4

Void Coefficient vs State:
    alpha_void(BU, boron, enrichment) = alpha_void_0 * f(BU) * g(boron)

    The void coefficient depends on the spectral hardness:
    - Higher burnup -> more Pu -> slightly harder spectrum -> less negative
    - Higher boron -> less negative (boron competes with moderation)
    - Higher enrichment -> less negative

    Ref: Todreas & Kazimi Vol I Ch 3; ANS-19.6.1

Moderator Temperature Coefficient (MTC) vs State:
    MTC(BU, boron, T) = MTC_base + delta_MTC_boron + delta_MTC_BU

    At BOL with high boron, MTC can be slightly positive.  It becomes
    more negative with:
    - Increasing burnup (boron dilution)
    - Increasing temperature (density effect dominates)
    - Decreasing boron concentration

    Ref: Stacey, "Nuclear Reactor Physics", Sec 5.5

Dependencies
------------
numpy
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class BurnupState:
    """Current depletion state of the fuel.

    Attributes
    ----------
    burnup_MWd_kgU : float
        Assembly-average burnup [MWd/kgU]. Typical range: 0-65.
    enrichment_w_pct : float
        U-235 enrichment [weight %]. Typical: 2.0-5.0.
    boron_ppm : float
        Soluble boron concentration [ppm]. Range: 0-2000.
    T_fuel_K : float
        Average fuel temperature [K].
    T_moderator_K : float
        Average moderator temperature [K].
    void_fraction : float
        Average void fraction [0-1].
    cycle_exposure : float
        Fraction of current cycle completed [0-1].
    """
    burnup_MWd_kgU: float = 0.0
    enrichment_w_pct: float = 4.5
    boron_ppm: float = 1000.0
    T_fuel_K: float = 900.0
    T_moderator_K: float = 580.0
    void_fraction: float = 0.0
    cycle_exposure: float = 0.0


class DopplerCoefficientModel:
    """Burnup-dependent Doppler (fuel temperature) coefficient.

    The Doppler coefficient magnitude decreases with burnup due to
    Pu-240 buildup, which has broader resonances than U-238.

    Model:
        alpha_D(BU, T) = alpha_D_0 * (1 + c_Pu * BU / BU_ref)
                         * (T_ref / T)^n_temp

    where:
        alpha_D_0 : base Doppler coefficient at fresh fuel conditions
        c_Pu : Pu buildup correction (positive, makes alpha less negative)
        n_temp : temperature scaling exponent (typically 0.5 for sqrt model)

    Parameters
    ----------
    alpha_D_0 : float
        Doppler coefficient at BOL, T_ref [dk/k per sqrt(K)].
        Default: -3.5e-5.
    c_Pu : float
        Plutonium buildup correction factor [per MWd/kgU].
        Typical: +0.003 to +0.006.
        Default: +0.004.
    BU_ref : float
        Reference burnup for normalization [MWd/kgU].
        Default: 40.0.
    T_ref : float
        Reference fuel temperature [K]. Default: 900.0.
    n_temp : float
        Temperature scaling exponent. Default: 0.5.
    """

    def __init__(self, alpha_D_0: float = -3.5e-5, c_Pu: float = 0.004,
                 BU_ref: float = 40.0, T_ref: float = 900.0,
                 n_temp: float = 0.5):
        self.alpha_D_0 = alpha_D_0
        self.c_Pu = c_Pu
        self.BU_ref = BU_ref
        self.T_ref = T_ref
        self.n_temp = n_temp

    def coefficient(self, state: BurnupState) -> float:
        """Compute Doppler coefficient at current state.

        Parameters
        ----------
        state : BurnupState

        Returns
        -------
        alpha_D : float
            Doppler coefficient [dk/k per sqrt(K)].
            Negative (inherently safe).
        """
        BU = state.burnup_MWd_kgU
        T = max(state.T_fuel_K, 300.0)

        # Burnup correction: Pu buildup makes coefficient less negative
        burnup_factor = 1.0 + self.c_Pu * BU / self.BU_ref

        # Temperature scaling
        temp_factor = (self.T_ref / T) ** self.n_temp

        alpha = self.alpha_D_0 * burnup_factor * temp_factor
        return alpha

    def reactivity(self, state: BurnupState, T_fuel_ref: float = 900.0) -> float:
        """Compute Doppler reactivity contribution.

        Parameters
        ----------
        state : BurnupState
        T_fuel_ref : float
            Reference temperature [K].

        Returns
        -------
        rho_D : float
            Doppler reactivity [dk/k].
        """
        alpha = self.coefficient(state)
        T = max(state.T_fuel_K, 300.0)
        T_ref = max(T_fuel_ref, 300.0)
        return alpha * (np.sqrt(T) - np.sqrt(T_ref))

    def sensitivity_table(self, burnup_range: np.ndarray = None,
                          T_range: np.ndarray = None) -> Dict:
        """Generate a sensitivity table for the Doppler coefficient.

        Parameters
        ----------
        burnup_range : array, optional
            Burnup values [MWd/kgU].
        T_range : array, optional
            Temperature values [K].

        Returns
        -------
        dict with keys 'burnup', 'temperature', 'alpha' (2D array).
        """
        if burnup_range is None:
            burnup_range = np.linspace(0, 60, 13)
        if T_range is None:
            T_range = np.linspace(400, 2500, 22)

        alpha_table = np.zeros((len(burnup_range), len(T_range)))
        for i, bu in enumerate(burnup_range):
            for j, T in enumerate(T_range):
                st = BurnupState(burnup_MWd_kgU=bu, T_fuel_K=T)
                alpha_table[i, j] = self.coefficient(st)

        return {
            'burnup_MWd_kgU': burnup_range,
            'temperature_K': T_range,
            'alpha_D': alpha_table,
        }


class VoidCoefficientModel:
    """State-dependent void reactivity coefficient.

    The void coefficient depends on:
    - Burnup (Pu buildup changes spectrum)
    - Boron concentration (boron competes with water for moderation role)
    - Enrichment (higher enrichment -> less negative)
    - Void fraction itself (non-linear at high void)

    Model:
        alpha_void(state) = alpha_v_0
                            * (1 + c_BU * BU / BU_ref)
                            * (1 + c_boron * boron / boron_ref)
                            * (1 + c_void * alpha^2)

    All corrections make the coefficient less negative (positive c values).

    Parameters
    ----------
    alpha_v_0 : float
        Base void coefficient at BOL, zero boron [dk/k per void fraction].
        Default: -0.015 (typical PWR).
    c_BU : float
        Burnup correction. Default: +0.003.
    BU_ref : float
        Reference burnup. Default: 40.0 MWd/kgU.
    c_boron : float
        Boron correction. Default: +0.0005.
    boron_ref : float
        Reference boron. Default: 1000 ppm.
    c_void : float
        Non-linear void correction. Default: +0.5.
    """

    def __init__(self, alpha_v_0: float = -0.015, c_BU: float = 0.003,
                 BU_ref: float = 40.0, c_boron: float = 0.0005,
                 boron_ref: float = 1000.0, c_void: float = 0.5):
        self.alpha_v_0 = alpha_v_0
        self.c_BU = c_BU
        self.BU_ref = BU_ref
        self.c_boron = c_boron
        self.boron_ref = boron_ref
        self.c_void = c_void

    def coefficient(self, state: BurnupState) -> float:
        """Compute void coefficient at current state.

        Parameters
        ----------
        state : BurnupState

        Returns
        -------
        alpha_void : float
            Void coefficient [dk/k per unit void fraction].
            Negative for LWR (inherent safety).
        """
        BU = state.burnup_MWd_kgU
        boron = state.boron_ppm
        alpha = state.void_fraction

        burnup_factor = 1.0 + self.c_BU * BU / self.BU_ref
        boron_factor = 1.0 + self.c_boron * boron / self.boron_ref
        void_nonlinear = 1.0 + self.c_void * alpha**2

        return self.alpha_v_0 * burnup_factor * boron_factor * void_nonlinear

    def reactivity(self, state: BurnupState, void_ref: float = 0.0) -> float:
        """Compute void reactivity contribution.

        Parameters
        ----------
        state : BurnupState
        void_ref : float
            Reference void fraction.

        Returns
        -------
        rho_void : float [dk/k].
        """
        alpha_v = self.coefficient(state)
        return alpha_v * (state.void_fraction - void_ref)

    def is_positive(self, state: BurnupState) -> bool:
        """Check if void coefficient is positive (safety concern).

        Returns True if alpha_void > 0, which violates LWR safety
        design criteria.
        """
        return self.coefficient(state) > 0

    def sensitivity_table(self, burnup_range: np.ndarray = None,
                          boron_range: np.ndarray = None) -> Dict:
        """Generate sensitivity table."""
        if burnup_range is None:
            burnup_range = np.linspace(0, 60, 13)
        if boron_range is None:
            boron_range = np.linspace(0, 2000, 11)

        alpha_table = np.zeros((len(burnup_range), len(boron_range)))
        for i, bu in enumerate(burnup_range):
            for j, b in enumerate(boron_range):
                st = BurnupState(burnup_MWd_kgU=bu, boron_ppm=b)
                alpha_table[i, j] = self.coefficient(st)

        return {
            'burnup_MWd_kgU': burnup_range,
            'boron_ppm': boron_range,
            'alpha_void': alpha_table,
        }


class ModeratorTemperatureCoefficientModel:
    """State-dependent moderator temperature coefficient (MTC).

    The MTC is the most complex feedback coefficient because it depends
    on multiple competing effects:
    - Water density change (moderating power)
    - Boron worth change (boron is dissolved in water)
    - Spectral shift (harder spectrum at higher temperature)

    At BOL with high boron, MTC can be slightly positive because:
    - Boron removal effect (higher T -> lower density -> less boron
      in core -> positive reactivity) outweighs
    - Moderation loss (higher T -> lower density -> less moderation
      -> negative reactivity)

    Model:
        MTC = MTC_base(BU)
              + delta_boron(boron)
              + delta_temp(T)

    where:
        MTC_base = MTC_0 * (1 - c_BU * BU / BU_ref)
        delta_boron = c_boron_MTC * (boron - boron_ref) / boron_scale
        delta_temp = c_temp * (T - T_ref) / T_scale

    Parameters
    ----------
    MTC_0 : float
        Base MTC at BOL, critical boron, nominal temperature
        [dk/k per K]. Default: -2.0e-4.
    c_BU : float
        Burnup improvement factor (MTC becomes more negative).
        Default: 0.5.
    BU_ref : float
        Reference burnup [MWd/kgU]. Default: 40.
    c_boron_MTC : float
        Boron effect on MTC [dk/k per K per ppm_normalized].
        Positive: more boron makes MTC less negative.
        Default: 1.5e-4.
    boron_ref : float
        Reference boron [ppm]. Default: 1000.
    boron_scale : float
        Boron normalization scale [ppm]. Default: 1000.
    c_temp : float
        Temperature non-linearity [dk/k per K per K_normalized].
        Default: -5e-5.
    T_ref : float
        Reference moderator temperature [K]. Default: 580.
    T_scale : float
        Temperature normalization scale [K]. Default: 100.
    """

    def __init__(self, MTC_0: float = -2.0e-4, c_BU: float = 0.5,
                 BU_ref: float = 40.0, c_boron_MTC: float = 1.5e-4,
                 boron_ref: float = 1000.0, boron_scale: float = 1000.0,
                 c_temp: float = -5e-5, T_ref: float = 580.0,
                 T_scale: float = 100.0):
        self.MTC_0 = MTC_0
        self.c_BU = c_BU
        self.BU_ref = BU_ref
        self.c_boron_MTC = c_boron_MTC
        self.boron_ref = boron_ref
        self.boron_scale = boron_scale
        self.c_temp = c_temp
        self.T_ref = T_ref
        self.T_scale = T_scale

    def coefficient(self, state: BurnupState) -> float:
        """Compute MTC at current state.

        Parameters
        ----------
        state : BurnupState

        Returns
        -------
        MTC : float
            Moderator temperature coefficient [dk/k per K].
            Should be negative at power for LWR safety.
        """
        BU = state.burnup_MWd_kgU
        boron = state.boron_ppm
        T = state.T_moderator_K

        # Base MTC improves (more negative) with burnup as boron dilutes
        MTC_base = self.MTC_0 * (1.0 - self.c_BU * BU / self.BU_ref)

        # Boron effect: higher boron -> less negative MTC
        delta_boron = self.c_boron_MTC * (boron - self.boron_ref) / self.boron_scale

        # Temperature non-linearity
        delta_temp = self.c_temp * (T - self.T_ref) / self.T_scale

        return MTC_base + delta_boron + delta_temp

    def reactivity(self, state: BurnupState, T_mod_ref: float = 580.0) -> float:
        """Compute moderator temperature reactivity.

        Parameters
        ----------
        state : BurnupState
        T_mod_ref : float
            Reference moderator temperature [K].

        Returns
        -------
        rho_mod : float [dk/k].
        """
        mtc = self.coefficient(state)
        return mtc * (state.T_moderator_K - T_mod_ref)

    def is_positive(self, state: BurnupState) -> bool:
        """Check if MTC is positive (regulatory concern).

        Per technical specifications, MTC must be negative at full
        power.  A slightly positive MTC at hot zero power with high
        boron may be acceptable.
        """
        return self.coefficient(state) > 0

    def critical_boron_for_zero_MTC(self, burnup: float = 0.0,
                                     T_mod: float = 580.0) -> float:
        """Find the boron concentration where MTC = 0.

        This is the maximum allowable boron for the given conditions.

        Parameters
        ----------
        burnup : float
            Burnup [MWd/kgU].
        T_mod : float
            Moderator temperature [K].

        Returns
        -------
        boron_critical : float
            Boron concentration [ppm] where MTC = 0.
        """
        # MTC_base + c_boron * (B - B_ref) / B_scale + c_temp * (T - T_ref) / T_scale = 0
        MTC_base = self.MTC_0 * (1.0 - self.c_BU * burnup / self.BU_ref)
        delta_temp = self.c_temp * (T_mod - self.T_ref) / self.T_scale

        if abs(self.c_boron_MTC) < 1e-20:
            return float('inf')

        B_critical = self.boron_ref - (MTC_base + delta_temp) * self.boron_scale / self.c_boron_MTC
        return max(B_critical, 0.0)

    def sensitivity_table(self, burnup_range: np.ndarray = None,
                          boron_range: np.ndarray = None) -> Dict:
        """Generate MTC sensitivity table."""
        if burnup_range is None:
            burnup_range = np.linspace(0, 60, 13)
        if boron_range is None:
            boron_range = np.linspace(0, 2000, 11)

        mtc_table = np.zeros((len(burnup_range), len(boron_range)))
        for i, bu in enumerate(burnup_range):
            for j, b in enumerate(boron_range):
                st = BurnupState(burnup_MWd_kgU=bu, boron_ppm=b)
                mtc_table[i, j] = self.coefficient(st)

        return {
            'burnup_MWd_kgU': burnup_range,
            'boron_ppm': boron_range,
            'MTC': mtc_table,
        }


class FeedbackCoefficientManager:
    """Aggregates all burnup-dependent coefficient models.

    Provides a single interface for the transient simulator to query
    state-dependent coefficients and compute total reactivity.

    Parameters
    ----------
    doppler : DopplerCoefficientModel, optional
    void : VoidCoefficientModel, optional
    mtc : ModeratorTemperatureCoefficientModel, optional
    """

    def __init__(self, doppler: Optional[DopplerCoefficientModel] = None,
                 void: Optional[VoidCoefficientModel] = None,
                 mtc: Optional[ModeratorTemperatureCoefficientModel] = None):
        self.doppler = doppler or DopplerCoefficientModel()
        self.void = void or VoidCoefficientModel()
        self.mtc = mtc or ModeratorTemperatureCoefficientModel()

    def coefficients(self, state: BurnupState) -> Dict[str, float]:
        """Get all feedback coefficients at current state.

        Returns
        -------
        dict
            Keys: 'alpha_doppler', 'alpha_void', 'MTC'.
            Values in standard units (dk/k per sqrt(K), per void, per K).
        """
        return {
            'alpha_doppler': self.doppler.coefficient(state),
            'alpha_void': self.void.coefficient(state),
            'MTC': self.mtc.coefficient(state),
        }

    def reactivity_breakdown(self, state: BurnupState,
                              T_fuel_ref: float = 900.0,
                              T_mod_ref: float = 580.0,
                              void_ref: float = 0.0) -> Dict[str, float]:
        """Compute all reactivity components.

        Returns
        -------
        dict
            Keys: 'rho_doppler', 'rho_void', 'rho_moderator', 'rho_total'.
        """
        rho_D = self.doppler.reactivity(state, T_fuel_ref)
        rho_v = self.void.reactivity(state, void_ref)
        rho_m = self.mtc.reactivity(state, T_mod_ref)

        return {
            'rho_doppler': rho_D,
            'rho_void': rho_v,
            'rho_moderator': rho_m,
            'rho_total': rho_D + rho_v + rho_m,
        }

    def safety_check(self, state: BurnupState) -> Dict[str, bool]:
        """Check all coefficient signs for LWR safety compliance.

        Returns
        -------
        dict
            Keys: 'doppler_negative', 'void_negative', 'MTC_negative',
                  'all_safe'.
        """
        alpha_D = self.doppler.coefficient(state)
        alpha_v = self.void.coefficient(state)
        mtc = self.mtc.coefficient(state)

        return {
            'doppler_negative': alpha_D < 0,
            'void_negative': alpha_v < 0,
            'MTC_negative': mtc < 0,
            'all_safe': alpha_D < 0 and alpha_v < 0 and mtc < 0,
            'alpha_doppler': alpha_D,
            'alpha_void': alpha_v,
            'MTC': mtc,
        }
