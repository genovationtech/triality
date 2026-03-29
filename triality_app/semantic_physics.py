"""
Semantic Physics Layer (SPL) for Triality
==========================================
Converts correct numerical results into correct physical meaning.

Pipeline position:
    Solver -> Observables -> Physics Truth Layer -> **SPL** -> Programmatic Insights -> LLM

The SPL does five things:
    1. Feature extraction   -- monotonicity, curvature, regime classification
    2. Semantic correction  -- rejects naive interpretations (e.g. first-nonzero != knee)
    3. Concept filtering    -- blocks physically invalid conclusions (e.g. "optimum" in monotonic I-V)
    4. Feature abstraction  -- high-level physics features from raw data
    5. Decision-ready output -- structured dict the decision engine can act on directly
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("triality.semantic_physics")


# ═══════════════════════════════════════════════════════════════════════════════
#  Physical Constants
# ═══════════════════════════════════════════════════════════════════════════════
K_BOLTZMANN_EV = 8.617e-5      # eV/K
K_BOLTZMANN_J = 1.381e-23      # J/K
Q_ELECTRON = 1.602e-19         # C
EPS_0 = 8.854e-12             # F/m
EPS_0_CGS = 8.854e-14         # F/cm
NI_SI_300K = 1.5e10            # cm^-3 intrinsic carrier Si at 300K
EPS_SI = 11.7                  # relative permittivity Si
ME_ELECTRON = 9.109e-31        # kg
FARADAY = 96485.0              # C/mol
R_GAS = 8.314                  # J/(mol K)
STEFAN_BOLTZMANN = 5.670e-8    # W/(m^2 K^4)
SPEED_OF_LIGHT = 3.0e8         # m/s


# ═══════════════════════════════════════════════════════════════════════════════
#  Core data structures
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class SemanticFeatures:
    """Extracted numerical features from raw data."""
    monotonic: Optional[bool] = None
    monotonic_direction: Optional[str] = None   # "increasing" | "decreasing"
    exponential: Optional[bool] = None
    linear: Optional[bool] = None
    has_plateau: Optional[bool] = None
    plateau_value: Optional[float] = None
    has_inflection: Optional[bool] = None
    inflection_param: Optional[float] = None
    dynamic_range_decades: Optional[float] = None
    sign_uniform: Optional[bool] = None
    dominant_sign: Optional[str] = None         # "positive" | "negative"
    curvature: Optional[str] = None             # "convex" | "concave" | "mixed"
    derivative_trend: Optional[str] = None      # "accelerating" | "decelerating" | "variable"
    data_points: int = 0
    param_range: Optional[Tuple[float, float]] = None
    value_range: Optional[Tuple[float, float]] = None
    # Extended features
    has_hysteresis: Optional[bool] = None
    noise_floor: Optional[float] = None
    peak_count: int = 0
    peak_params: List[float] = field(default_factory=list)
    zero_crossings: int = 0


@dataclass
class SemanticUnderstanding:
    """Decision-ready semantic output for one analysis run."""
    regime: str = "unknown"
    behavior: str = "unknown"
    key_estimates: Dict[str, Any] = field(default_factory=dict)
    monotonic: Optional[bool] = None
    optimum_exists: Optional[bool] = None
    optimum_value: Optional[float] = None
    physics_valid: bool = True
    rejected_concepts: List[str] = field(default_factory=list)
    corrections: Dict[str, str] = field(default_factory=dict)
    confidence: str = "high"
    warnings: List[str] = field(default_factory=list)
    domain: str = "generic"
    summary: str = ""
    # Extended
    physics_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    governing_equations: List[str] = field(default_factory=list)
    dimensional_checks: Dict[str, str] = field(default_factory=dict)

    def to_context(self) -> str:
        """Generate context block for the decision engine / LLM."""
        lines = ["## Semantic Physics Layer -- Interpretation Guide\n"]

        lines.append(f"**Domain:** {self.domain}")
        lines.append(f"**Regime:** {self.regime}")
        lines.append(f"**Behavior:** {self.behavior}")
        lines.append(f"**Monotonic:** {self.monotonic}")
        lines.append(f"**Optimum exists:** {self.optimum_exists}")
        lines.append(f"**Physics valid:** {self.physics_valid}")
        lines.append(f"**Interpretation confidence:** {self.confidence}")
        lines.append("")

        if self.key_estimates:
            lines.append("**Key estimates (use these):**")
            for k, v in self.key_estimates.items():
                lines.append(f"  - {k}: {v}")
            lines.append("")

        if self.physics_bounds:
            lines.append("**Physical bounds (values outside these are unphysical):**")
            for k, (lo, hi) in self.physics_bounds.items():
                lo_s = f"{lo:.4g}" if isinstance(lo, float) else str(lo)
                hi_s = f"{hi:.4g}" if isinstance(hi, float) else str(hi)
                lines.append(f"  - {k}: [{lo_s}, {hi_s}]")
            lines.append("")

        if self.governing_equations:
            lines.append("**Governing equations (ground truth):**")
            for eq in self.governing_equations:
                lines.append(f"  - {eq}")
            lines.append("")

        if self.rejected_concepts:
            lines.append("**REJECTED CONCEPTS (do NOT use these in your answer):**")
            for c in self.rejected_concepts:
                lines.append(f"  - {c}")
            lines.append("")

        if self.corrections:
            lines.append("**Semantic corrections:**")
            for k, v in self.corrections.items():
                lines.append(f"  - {k}: {v}")
            lines.append("")

        if self.dimensional_checks:
            lines.append("**Dimensional/unit checks:**")
            for k, v in self.dimensional_checks.items():
                lines.append(f"  - {k}: {v}")
            lines.append("")

        if self.warnings:
            lines.append("**Warnings:**")
            for w in self.warnings:
                lines.append(f"  - {w}")
            lines.append("")

        if self.summary:
            lines.append(f"**Summary:** {self.summary}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 1: Feature Extraction (domain-agnostic)
# ═══════════════════════════════════════════════════════════════════════════════
def extract_features(
    params: List[float],
    values: List[float],
) -> SemanticFeatures:
    """Extract numerical features from a parameter-value curve."""
    f = SemanticFeatures()
    n = len(values)
    f.data_points = n

    if n < 2 or len(params) != n:
        return f

    f.param_range = (min(params), max(params))
    f.value_range = (min(values), max(values))

    # --- Sign analysis ---
    positive = [v for v in values if v > 0]
    negative = [v for v in values if v < 0]
    f.sign_uniform = len(positive) == 0 or len(negative) == 0
    f.dominant_sign = "positive" if len(positive) >= len(negative) else "negative"

    # --- Zero crossings ---
    for i in range(n - 1):
        if values[i] * values[i + 1] < 0:
            f.zero_crossings += 1

    # --- Monotonicity ---
    abs_vals = [abs(v) for v in values]
    diffs = [values[i + 1] - values[i] for i in range(n - 1)]
    abs_diffs = [abs_vals[i + 1] - abs_vals[i] for i in range(n - 1)]

    all_inc = all(d > -1e-30 for d in diffs)
    all_dec = all(d < 1e-30 for d in diffs)
    abs_all_inc = all(d > -1e-30 for d in abs_diffs)
    abs_all_dec = all(d < 1e-30 for d in abs_diffs)

    f.monotonic = all_inc or all_dec or abs_all_inc or abs_all_dec
    if all_inc or abs_all_inc:
        f.monotonic_direction = "increasing"
    elif all_dec or abs_all_dec:
        f.monotonic_direction = "decreasing"

    # --- Dynamic range ---
    nonzero = [abs(v) for v in values if abs(v) > 1e-30]
    if len(nonzero) >= 2:
        f.dynamic_range_decades = math.log10(max(nonzero) / min(nonzero))

    # --- Noise floor estimation ---
    if n >= 5:
        sorted_abs = sorted(abs(v) for v in values)
        f.noise_floor = sorted_abs[max(0, len(sorted_abs) // 10)]

    # --- Exponential detection ---
    if n >= 4 and abs_all_inc:
        pos_diffs = [d for d in abs_diffs if d > 1e-30]
        if len(pos_diffs) >= 3:
            ratios = [pos_diffs[i + 1] / pos_diffs[i] for i in range(len(pos_diffs) - 1)
                      if pos_diffs[i] > 1e-30]
            if ratios and all(r > 1.3 for r in ratios):
                f.exponential = True
                f.linear = False
            else:
                f.exponential = False
    if f.exponential is None:
        f.exponential = False

    # --- Linear detection ---
    if not f.exponential and n >= 3 and f.monotonic:
        avg_diff = sum(abs(d) for d in diffs) / len(diffs) if diffs else 0
        if avg_diff > 1e-30:
            relative_var = sum((abs(d) - avg_diff) ** 2 for d in diffs) / len(diffs)
            cv = math.sqrt(relative_var) / avg_diff if avg_diff > 0 else 0
            f.linear = cv < 0.3
        else:
            f.linear = True

    # --- Plateau detection ---
    if n >= 3:
        last_vals = values[-max(3, n // 3):]
        avg_last = sum(last_vals) / len(last_vals)
        if avg_last != 0 and all(abs(v - avg_last) / abs(avg_last) < 0.05 for v in last_vals):
            f.has_plateau = True
            f.plateau_value = avg_last
        else:
            f.has_plateau = False

    # --- Peak detection ---
    if n >= 3:
        for i in range(1, n - 1):
            if values[i] > values[i - 1] and values[i] > values[i + 1]:
                f.peak_count += 1
                f.peak_params.append(params[i])
            elif values[i] < values[i - 1] and values[i] < values[i + 1]:
                f.peak_count += 1  # valley is also a turning point
                f.peak_params.append(params[i])

    # --- Inflection / knee detection ---
    if n >= 4:
        second_diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        max_curv_idx = max(range(len(second_diffs)), key=lambda j: abs(second_diffs[j]))
        total_change = abs(values[-1] - values[0])
        if total_change > 0 and abs(second_diffs[max_curv_idx]) > total_change * 0.1:
            f.has_inflection = True
            f.inflection_param = params[max_curv_idx + 1]
        else:
            f.has_inflection = False

    # --- Curvature classification ---
    if n >= 4:
        second_diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
        pos_curv = sum(1 for d in second_diffs if d > 1e-30)
        neg_curv = sum(1 for d in second_diffs if d < -1e-30)
        total = pos_curv + neg_curv
        if total > 0:
            if pos_curv / total > 0.7:
                f.curvature = "convex"
            elif neg_curv / total > 0.7:
                f.curvature = "concave"
            else:
                f.curvature = "mixed"

    # --- Derivative trend ---
    if n >= 3:
        abs_diffs_mag = [abs(d) for d in diffs]
        diff_of_diffs = [abs_diffs_mag[i + 1] - abs_diffs_mag[i] for i in range(len(abs_diffs_mag) - 1)]
        if all(d > -1e-30 for d in diff_of_diffs):
            f.derivative_trend = "accelerating"
        elif all(d < 1e-30 for d in diff_of_diffs):
            f.derivative_trend = "decelerating"
        else:
            f.derivative_trend = "variable"

    return f


def detect_exponential_onset(
    params: List[float],
    values: List[float],
    min_decades: float = 2.0,
) -> Optional[float]:
    """Find the parameter value where exponential growth begins."""
    if len(params) < 4 or len(values) < 4:
        return None

    abs_vals = [abs(v) if abs(v) > 1e-30 else 1e-30 for v in values]
    log_vals = [math.log10(v) for v in abs_vals]

    slopes = []
    for i in range(len(params) - 1):
        dp = params[i + 1] - params[i]
        if abs(dp) < 1e-15:
            slopes.append(0)
        else:
            slopes.append((log_vals[i + 1] - log_vals[i]) / dp)

    threshold_slope = 3.0
    for i, s in enumerate(slopes):
        if s > threshold_slope:
            return params[i]

    if len(slopes) >= 3:
        dslopes = [slopes[i + 1] - slopes[i] for i in range(len(slopes) - 1)]
        max_idx = max(range(len(dslopes)), key=lambda j: dslopes[j])
        if dslopes[max_idx] > 0.5:
            return params[max_idx + 1]

    return None


def _get_obs_value(observables: Dict, name: str) -> Optional[float]:
    """Extract a numeric value from observables dict by name."""
    v = observables.get(name)
    if isinstance(v, dict):
        return v.get("value") if isinstance(v.get("value"), (int, float)) else None
    if isinstance(v, (int, float)):
        return v
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Base Domain Rule Set
# ═══════════════════════════════════════════════════════════════════════════════
class DomainRuleSet:
    """Base class for domain-specific semantic interpretation rules."""

    domain: str = "generic"
    modules: List[str] = []

    @classmethod
    def classify_regime(cls, features: SemanticFeatures, observables: Dict, config: Dict) -> str:
        return "unknown"

    @classmethod
    def classify_behavior(cls, features: SemanticFeatures) -> str:
        if features.exponential:
            return "exponential"
        if features.linear:
            return "linear"
        if features.has_plateau:
            return "saturating"
        if features.monotonic:
            return "monotonic"
        if features.peak_count >= 2:
            return "oscillatory"
        if features.peak_count == 1:
            return "peaked"
        return "complex"

    @classmethod
    def correct_interpretation(
        cls,
        features: SemanticFeatures,
        observables: Dict,
        config: Dict,
        params: List[float],
        values: List[float],
    ) -> SemanticUnderstanding:
        u = SemanticUnderstanding()
        u.domain = cls.domain
        u.regime = cls.classify_regime(features, observables, config)
        u.behavior = cls.classify_behavior(features)
        u.monotonic = features.monotonic

        if features.monotonic:
            u.optimum_exists = False
            u.rejected_concepts.append(
                "optimum -- curve is monotonic, no interior extremum exists"
            )
        else:
            u.optimum_exists = features.peak_count > 0

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  SEMICONDUCTOR / PN Junction / Drift-Diffusion
# ═══════════════════════════════════════════════════════════════════════════════
class SemiconductorRules(DomainRuleSet):
    domain = "semiconductor"
    modules = ["drift_diffusion", "coupled_electrical_thermal"]

    # ---- Material constants ----
    TYPICAL_KNEE_SI = (0.55, 0.75)
    TYPICAL_VBI_SI = (0.55, 0.85)
    MIN_IV_DECADES = 3.0
    # Breakdown voltage bounds for Si
    BREAKDOWN_RANGE_SI = (5.0, 1000.0)         # V
    IDEALITY_FACTOR_RANGE = (1.0, 2.0)
    SAT_CURRENT_RANGE = (1e-15, 1e-8)          # A/cm^2
    TEMP_COEFF_KNEE = (-2.5e-3, -1.5e-3)       # V/K for Si diode
    SERIES_RESISTANCE_TYPICAL = (0.01, 100.0)   # Ohm

    @classmethod
    def classify_regime(cls, features, observables, config):
        v_app = None
        if config:
            v_app = config.get("solver", {}).get("applied_voltage")

        if v_app is not None:
            if v_app > 0.8:
                return "high_forward_injection"
            elif v_app > 0:
                return "forward_bias"
            elif v_app > -5:
                return "reverse_bias"
            else:
                return "near_breakdown"

        if features.exponential and features.monotonic_direction == "increasing":
            if features.dynamic_range_decades and features.dynamic_range_decades > 8:
                return "forward_bias"
            return "forward_conduction"

        if features.has_plateau and features.monotonic_direction == "decreasing":
            return "reverse_saturation"

        return "unknown"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        # ---- Governing equations ----
        u.governing_equations = [
            "Shockley diode: I = I_0 * (exp(qV/nkT) - 1)",
            "Built-in potential: Vbi = (kT/q) * ln(Nd*Na/ni^2)",
            "Depletion width: W = sqrt(2*eps*Vbi/q * (Na+Nd)/(Na*Nd))",
            "Peak E-field: E_max = q*Nd*W_n/eps = q*Na*W_p/eps",
            "Junction capacitance: C_j = eps*A/W (voltage-dependent via W)",
            "Diffusion current: J_diff = q*Dp*pn0/Lp + q*Dn*np0/Ln",
        ]

        # ---- Physics bounds ----
        u.physics_bounds["ideality_factor_n"] = cls.IDEALITY_FACTOR_RANGE
        u.physics_bounds["saturation_current_I0_A_cm2"] = cls.SAT_CURRENT_RANGE
        u.physics_bounds["knee_voltage_Si_V"] = cls.TYPICAL_KNEE_SI
        u.physics_bounds["built_in_potential_Si_V"] = cls.TYPICAL_VBI_SI
        u.physics_bounds["breakdown_voltage_Si_V"] = cls.BREAKDOWN_RANGE_SI

        # ---- Forward bias: monotonic exponential, no optimum ----
        if u.regime in ("forward_bias", "forward_conduction", "high_forward_injection"):
            u.optimum_exists = False
            u.rejected_concepts = [c for c in u.rejected_concepts if "optimum" not in c]
            u.rejected_concepts.append(
                "optimum -- forward-bias I-V of a PN junction is monotonic exponential (Shockley equation); "
                "no optimum exists in current vs. voltage"
            )

        # ---- High-injection effects ----
        if u.regime == "high_forward_injection":
            u.corrections["high_injection"] = (
                "At high forward bias (V >> Vbi), high-injection effects dominate: "
                "ideality factor approaches n=2 (from n=1 at low injection), "
                "series resistance causes I-V roll-off (deviation from exponential), "
                "and ohmic drop V_Rs = I*R_s becomes significant. "
                "The apparent 'saturation' at high current is R_s-limited, NOT a physical optimum."
            )
            u.rejected_concepts.append(
                "current_saturation_at_high_bias -- this is series resistance roll-off, not saturation"
            )

        # ---- Built-in potential is intrinsic ----
        u.corrections["built_in_potential"] = (
            "Built-in potential (Vbi) is a FIXED intrinsic property determined solely by doping: "
            "Vbi = (kT/q)*ln(Nd*Na/ni^2). For Si at 300K with Nd=Na=1e16: Vbi ~ 0.72V. "
            "It does NOT vary with applied voltage. It has a weak temperature dependence: "
            "dVbi/dT ~ -2 mV/K. Do NOT report Vbi as changing across a voltage sweep."
        )

        # ---- Knee voltage correction ----
        onset = detect_exponential_onset(params, values)
        if onset is not None:
            if onset < cls.TYPICAL_KNEE_SI[0]:
                u.corrections["knee_voltage"] = (
                    f"Detected onset at {onset:.3f}V is below the physical knee for silicon "
                    f"({cls.TYPICAL_KNEE_SI[0]}-{cls.TYPICAL_KNEE_SI[1]}V). This reflects "
                    f"leakage current or insufficient sweep resolution, not the true turn-on. "
                    f"The true knee is where J exceeds ~1 A/cm^2, which occurs at "
                    f"V ~ Vbi - (nkT/q)*ln(J_0_typical/1) ~ 0.6-0.7V for Si. "
                    f"USE {cls.TYPICAL_KNEE_SI[0]}-{cls.TYPICAL_KNEE_SI[1]}V."
                )
                u.key_estimates["knee_voltage_range"] = cls.TYPICAL_KNEE_SI
                u.warnings.append(f"Knee at {onset:.3f}V corrected to {cls.TYPICAL_KNEE_SI}")
            else:
                u.key_estimates["knee_voltage"] = round(onset, 3)
        else:
            u.key_estimates["knee_voltage_range"] = cls.TYPICAL_KNEE_SI
            u.warnings.append("Knee not detectable from data -- using Vbi physics estimate.")

        # ---- Breakdown regime ----
        if u.regime == "near_breakdown":
            u.corrections["breakdown"] = (
                "In reverse bias beyond breakdown voltage, current increases sharply due to "
                "avalanche multiplication (impact ionization) or Zener tunneling (for V_BR < 5V). "
                "Avalanche: M = 1/(1-(V/V_BR)^n) where n=3-6. "
                "Zener: tunneling probability ~ exp(-const/E). "
                "Breakdown voltage V_BR ~ eps*E_crit^2/(2*q*N_lighter) scales inversely with doping."
            )

        # ---- Dynamic range check ----
        if features.dynamic_range_decades is not None:
            u.key_estimates["iv_dynamic_range_decades"] = round(features.dynamic_range_decades, 1)
            if features.dynamic_range_decades < cls.MIN_IV_DECADES:
                u.warnings.append(
                    f"I-V curve spans only {features.dynamic_range_decades:.1f} decades "
                    f"(expected >{cls.MIN_IV_DECADES} for a proper forward-bias characteristic). "
                    f"This suggests: (a) voltage range too narrow, (b) too few sweep points, "
                    f"or (c) solver resolution insufficient. Increase sweep range to 0-0.8V "
                    f"with >= 20 points."
                )
                u.confidence = "medium"

        # ---- Current sign semantics ----
        u.corrections["current_sign"] = (
            "Negative current values in forward bias are a SIGN CONVENTION artifact: "
            "conventional current flows from p to n (positive terminal), but the solver "
            "may report electron current (opposite sign). Use |J| for magnitude. "
            "The physical direction is set by bias polarity, not the numerical sign."
        )

        # ---- Temperature dependence ----
        u.corrections["temperature_effects"] = (
            "Temperature affects the PN junction through three mechanisms: "
            "(1) ni^2 ~ exp(-Eg/kT) increases exponentially => leakage current doubles every ~10K, "
            "(2) Vbi decreases ~2mV/K as ni increases, "
            "(3) mobility decreases as T^(-1.5) due to phonon scattering. "
            "Net effect: forward voltage at fixed current DECREASES ~2mV/K (negative tempco)."
        )

        # ---- Capacitance ----
        u.corrections["junction_capacitance"] = (
            "Junction capacitance C_j = A*sqrt(q*eps*N_eff/(2*(Vbi-V))) varies as (Vbi-V)^(-1/2) "
            "for an abrupt junction. It diverges as V -> Vbi. In reverse bias, C_j decreases "
            "as V becomes more negative. 1/C^2 vs V should be LINEAR for uniform doping "
            "(Mott-Schottky plot). Non-linearity indicates graded doping."
        )

        # ---- Dimensional checks ----
        u.dimensional_checks["current_density"] = "J must be in A/cm^2 or A/m^2; check factor of 1e4"
        u.dimensional_checks["voltage"] = "V must be in Volts; typical range -100V to +1V for Si diode"
        u.dimensional_checks["doping"] = "N_d, N_a in cm^-3; typical range 1e14 to 1e20"

        u.summary = (
            f"PN junction in {u.regime} regime. I-V is {u.behavior}. "
            f"No optimum exists in forward I-V. Vbi is fixed by doping (not variable). "
            f"Knee ~ {u.key_estimates.get('knee_voltage', u.key_estimates.get('knee_voltage_range', 'unknown'))}V. "
            f"Series resistance causes high-injection roll-off, not saturation."
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  FLUID DYNAMICS / Navier-Stokes / CFD
# ═══════════════════════════════════════════════════════════════════════════════
class FluidDynamicsRules(DomainRuleSet):
    domain = "fluid_dynamics"
    modules = [
        "navier_stokes", "cfd_turbulence", "multiphase_vof",
        "thermal_hydraulics", "conjugate_heat_transfer",
    ]

    @classmethod
    def _compute_reynolds(cls, config):
        if not config:
            return None
        solver = config.get("solver", {})
        nu = solver.get("nu")
        U = solver.get("U_lid") or solver.get("U_inlet") or solver.get("velocity")
        L = solver.get("Lx", 1.0)
        if nu and U and nu > 0:
            return abs(U) * L / nu
        return None

    @classmethod
    def classify_regime(cls, features, observables, config):
        Re = cls._compute_reynolds(config)
        if Re is not None:
            if Re < 1:
                return "stokes_flow"
            elif Re < 100:
                return "laminar_low_re"
            elif Re < 2000:
                return "laminar"
            elif Re < 4000:
                return "transitional"
            elif Re < 1e6:
                return "turbulent"
            else:
                return "high_re_turbulent"
        return "unknown"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        # ---- Governing equations ----
        u.governing_equations = [
            "Reynolds number: Re = U*L/nu (ratio of inertial to viscous forces)",
            "Navier-Stokes: rho*(du/dt + u.grad(u)) = -grad(p) + mu*lap(u) + f",
            "Continuity (incompressible): div(u) = 0",
            "Pressure Poisson: lap(p) = -rho*div(u.grad(u))",
            "Skin friction (laminar flat plate): C_f = 0.664/sqrt(Re_x)",
            "Skin friction (turbulent flat plate): C_f ~ 0.027/Re_x^(1/7)",
            "Pressure coefficient: C_p = (p - p_inf)/(0.5*rho*U^2), C_p <= 1 at stagnation",
        ]

        # ---- Compute Re and inject ----
        Re = cls._compute_reynolds(config)
        if Re is not None:
            u.key_estimates["reynolds_number"] = round(Re, 1)

            # Nusselt correlation bounds
            if Re < 2000:
                # Laminar pipe: Nu ~ 3.66 (const wall T) or 4.36 (const heat flux)
                u.physics_bounds["nusselt_number_laminar"] = (3.0, 10.0)
            else:
                # Dittus-Boelter: Nu = 0.023 Re^0.8 Pr^0.4
                u.physics_bounds["nusselt_number_turbulent"] = (10.0, 1e4)

            # Skin friction bounds
            if Re < 2000:
                cf_est = 0.664 / math.sqrt(max(Re, 1))
                u.key_estimates["skin_friction_estimate"] = round(cf_est, 6)
            else:
                cf_est = 0.027 / (Re ** (1.0 / 7.0))
                u.key_estimates["skin_friction_estimate"] = round(cf_est, 6)

        # ---- Physics bounds ----
        u.physics_bounds["pressure_coefficient_Cp"] = (-10.0, 1.0)  # Cp <= 1 at stagnation
        u.physics_bounds["kinematic_viscosity_m2s"] = (1e-7, 10.0)
        u.physics_bounds["density_kg_m3"] = (0.01, 20000.0)         # gas to liquid metal

        # ---- Viscosity must be positive ----
        if config:
            nu = config.get("solver", {}).get("nu")
            if nu is not None and nu <= 0:
                u.physics_valid = False
                u.rejected_concepts.append("negative_viscosity -- kinematic viscosity must be positive (second law)")
                u.confidence = "low"
                u.warnings.append("CRITICAL: nu <= 0 is unphysical. Results are invalid.")

        # ---- Velocity optimum caveats ----
        if features.monotonic:
            u.corrections["velocity_optimum"] = (
                "Velocity metrics (max velocity, vorticity, kinetic energy) are typically monotonic "
                "with Re. An 'optimum' at a sweep boundary is NOT a true optimum -- the metric is "
                "still improving. Use a composite objective (e.g., mixing quality = std(C)/mean(C), "
                "pressure drop ratio dp/Q, or thermal effectiveness) for meaningful optimization."
            )

        # ---- Recirculation semantics ----
        u.corrections["recirculation"] = (
            "Flow recirculation (negative velocity regions) is a NORMAL physical phenomenon. "
            "In lid-driven cavities: primary vortex forms at all Re > 0; secondary corner vortices "
            "appear at Re > ~100; tertiary vortices at Re > ~1000. "
            "In channel flows: recirculation behind steps/obstacles is expected. "
            "Map recirculation zones to PHYSICAL LOCATION, don't flag as error."
        )

        # ---- Stagnation pressure ----
        u.corrections["pressure_interpretation"] = (
            "Pressure in incompressible flow: total pressure p_total = p_static + 0.5*rho*U^2. "
            "The maximum static pressure occurs at stagnation points (C_p = 1). "
            "Pressure DROP across the domain drives the flow (dp/dx < 0 in flow direction). "
            "Negative pressure is gauge pressure, not a physical impossibility."
        )

        # ---- Non-dimensional groups beyond Re ----
        u.corrections["dimensionless_groups"] = (
            "Beyond Re, check these for regime classification: "
            "Grashof Gr = g*beta*dT*L^3/nu^2 (buoyancy vs. viscosity; Gr > 10^9 = turbulent natural convection), "
            "Froude Fr = U/sqrt(g*L) (inertia vs. gravity; Fr < 1 = subcritical free-surface flow), "
            "Strouhal St = f*L/U (vortex shedding frequency; St ~ 0.2 for cylinder wake), "
            "Knudsen Kn = lambda/L (molecular vs. continuum; Kn > 0.01 = slip flow regime)."
        )

        # ---- Grid resolution ----
        if config:
            Nx = config.get("solver", {}).get("Nx", 0)
            Ny = config.get("solver", {}).get("Ny", 0)
            if Nx and Ny:
                u.key_estimates["grid_resolution"] = f"{Nx}x{Ny}"
                if Nx * Ny < 2500:
                    u.warnings.append(
                        f"Grid {Nx}x{Ny} is coarse ({Nx*Ny} cells). "
                        f"For Re={Re:.0f if Re else '?'}: need ~{max(int(math.sqrt(Re or 100)*2), 50)}^2 "
                        f"cells for resolved solution. Quantitative values may shift 10-30% with refinement."
                    )
                if Re and Re > 1000 and Nx * Ny < 10000:
                    u.warnings.append(
                        f"At Re={Re:.0f}, grid should resolve boundary layer: "
                        f"delta ~ L/sqrt(Re) = {1.0/math.sqrt(Re):.4f}*L. "
                        f"Current grid spacing {1.0/max(Nx,1):.4f}*L may be too coarse."
                    )

        # ---- CFL stability ----
        u.corrections["numerical_stability"] = (
            "Check CFL condition: CFL = U*dt/dx < 1 for explicit schemes. "
            "Unphysical pressure oscillations or velocity spikes often indicate CFL violation. "
            "For implicit schemes, CFL > 1 is allowed but accuracy degrades. "
            "Checkerboard pressure patterns indicate need for pressure stabilization."
        )

        # ---- Dimensional checks ----
        u.dimensional_checks["velocity"] = "m/s; compare against U_inlet or U_lid for consistency"
        u.dimensional_checks["pressure"] = "Pa (or Pa gauge); dp across domain should scale as rho*U^2"
        u.dimensional_checks["viscosity"] = "nu in m^2/s (kinematic) or mu in Pa*s (dynamic); mu = rho*nu"

        u.summary = (
            f"Flow regime: {u.regime}"
            + (f" (Re={Re:.0f})" if Re else "")
            + f". Behavior: {u.behavior}. "
            + ("Velocity metrics are monotonic with Re -- no interior optimum. " if features.monotonic else "")
            + "Recirculation is normal physics, not an artifact."
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  THERMAL / COMBUSTION / CHEMICAL KINETICS
# ═══════════════════════════════════════════════════════════════════════════════
class ThermalCombustionRules(DomainRuleSet):
    domain = "thermal_combustion"
    modules = [
        "combustion_chemistry", "combustor_simulation", "thermal_runaway_kinetics",
        "pack_thermal", "reacting_flows", "turbulence_combustion",
        "pollutant_formation", "soot_model", "chemical_kinetics",
        "battery_abuse_simulation", "battery_thermal",
    ]

    # ---- Material property bounds ----
    ACTIVATION_ENERGY_RANGE = (20e3, 400e3)    # J/mol (organics, batteries, explosives)
    PRE_EXPONENTIAL_RANGE = (1e5, 1e20)        # 1/s
    SPECIFIC_HEAT_RANGE = (0.5e3, 5.0e3)       # J/(kg K) for liquids/solids
    ADIABATIC_FLAME_T = {
        "hydrocarbon_air": (1800, 2800),        # K
        "hydrogen_air": (2300, 2500),            # K
        "battery_thermal_runaway": (400, 1200),  # K (cell level)
    }
    HEAT_OF_REACTION_RANGE = (50e3, 50e6)       # J/kg

    @classmethod
    def classify_regime(cls, features, observables, config):
        if features.exponential and features.derivative_trend == "accelerating":
            return "thermal_runaway"
        if features.exponential and features.derivative_trend != "accelerating":
            return "ignition_induction"
        if features.has_inflection:
            return "ignition_transition"
        if features.has_plateau:
            if features.plateau_value and features.plateau_value > 1000:
                return "steady_flame"
            return "thermal_equilibrium"
        if features.monotonic:
            if features.monotonic_direction == "increasing":
                return "transient_heating"
            return "transient_cooling"
        if features.peak_count > 0:
            return "cool_flame_oscillation"
        return "transient"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        # ---- Governing equations ----
        u.governing_equations = [
            "Arrhenius rate: k(T) = A * exp(-Ea/(R*T))",
            "Energy balance: rho*cp*dT/dt = Q_rxn*r(T) - h*A_s*(T-T_amb) (+ conduction, radiation)",
            "Species: dC_i/dt = -nu_i * k(T) * prod(C_j^n_j)  (mass action kinetics)",
            "Semenov criterion (runaway): Q_gen(T_c) = Q_loss(T_c) AND dQ_gen/dT = dQ_loss/dT",
            "Frank-Kamenetskii: delta_cr = (rho*Q*A*Ea*r0^2)/(lambda*R*T_a^2) * exp(-Ea/(R*T_a))",
            "Flame speed: S_L = sqrt(alpha * A * exp(-Ea/(R*T_f))) where alpha = thermal diffusivity",
        ]

        # ---- Physics bounds ----
        u.physics_bounds["temperature_K"] = (0.0, 1e5)  # absolute zero to plasma
        u.physics_bounds["activation_energy_J_mol"] = cls.ACTIVATION_ENERGY_RANGE
        u.physics_bounds["pre_exponential_1_s"] = cls.PRE_EXPONENTIAL_RANGE
        u.physics_bounds["specific_heat_J_kgK"] = cls.SPECIFIC_HEAT_RANGE
        u.physics_bounds["heat_of_reaction_J_kg"] = cls.HEAT_OF_REACTION_RANGE

        # ---- Arrhenius kinetics are inherently exponential ----
        if features.exponential:
            u.corrections["exponential_growth"] = (
                "Exponential temperature rise is the SIGNATURE of Arrhenius thermal runaway: "
                "k(T) = A*exp(-Ea/RT) creates positive feedback (higher T => faster reaction => more heat). "
                "This is physically expected, not numerical error. The transition from induction "
                "to runaway is governed by the Semenov or Frank-Kamenetskii criterion: "
                "when heat generation rate exceeds heat loss rate at a critical temperature."
            )

        # ---- Ignition delay is a threshold phenomenon ----
        if features.has_inflection:
            u.key_estimates["ignition_onset_parameter"] = features.inflection_param
            u.corrections["ignition_delay"] = (
                f"Inflection at parameter={features.inflection_param:.4g} marks ignition onset. "
                f"Before: induction phase (slow chemistry, T rises gradually, exothermic reactions "
                f"barely activated). After: rapid exothermic runaway (rate doubles every ~10K). "
                f"This is a THRESHOLD, not an optimum. The ignition delay time tau_ign scales as "
                f"tau ~ (1/A)*exp(Ea/RT_0) -- exponentially sensitive to initial temperature."
            )

        # ---- Cool flame detection ----
        if features.peak_count > 0 and u.regime == "cool_flame_oscillation":
            u.corrections["cool_flames"] = (
                "Oscillatory temperature behavior indicates cool-flame regime: "
                "low-temperature oxidation (LTO) releases enough heat to raise T, "
                "but the negative temperature coefficient (NTC) region suppresses the high-T pathway. "
                "This causes periodic ignition/extinction. Cool flames occur at 500-800K "
                "for hydrocarbons. They are precursors to hot ignition."
            )
            u.key_estimates["oscillation_peaks"] = features.peak_params

        # ---- Deflagration vs. detonation ----
        u.corrections["deflagration_vs_detonation"] = (
            "Deflagration: flame propagates subsonically via thermal diffusion (S_L ~ 0.1-10 m/s). "
            "Detonation: supersonic combustion wave coupled to shock (D ~ 1500-3000 m/s). "
            "DDT (deflagration-to-detonation transition) requires confinement, obstacles, or "
            "critical flame acceleration. Do NOT conflate flame speed with detonation velocity."
        )

        # ---- Temperature cannot be negative (Kelvin) ----
        if features.value_range and features.value_range[0] < 0:
            u.warnings.append(
                "CRITICAL: Negative temperature detected. Temperature in Kelvin is always >= 0. "
                "Check: (a) Celsius used instead of Kelvin, (b) numerical instability, "
                "(c) sign error in heat source term."
            )
            u.physics_valid = False
            u.confidence = "low"

        # ---- Thermal runaway has no steady-state ----
        if u.regime == "thermal_runaway":
            u.optimum_exists = False
            u.rejected_concepts = [c for c in u.rejected_concepts if "optimum" not in c]
            u.rejected_concepts.extend([
                "optimum -- thermal runaway is UNBOUNDED exponential; no equilibrium without intervention",
                "steady_state -- runaway by definition means heat generation exceeds all loss mechanisms",
            ])
            u.corrections["runaway_mitigation"] = (
                "Thermal runaway mitigation requires: (1) reducing heat generation rate (cooling, "
                "dilution, inhibition), (2) increasing heat loss (forced convection, radiation), "
                "or (3) removing fuel. There is no 'optimal' runaway -- it must be prevented entirely."
            )

        # ---- Adiabatic flame temperature bounds ----
        u.corrections["adiabatic_flame_temperature"] = (
            "Adiabatic flame temperature T_ad is the MAXIMUM possible temperature for a given "
            "fuel-oxidizer mixture. It is computed from enthalpy balance: "
            "sum(n_i*h_i)_products = sum(n_i*h_i)_reactants at T_ad. "
            "Typical values: CH4/air=2230K, H2/air=2400K, C3H8/air=2268K. "
            "If computed T exceeds T_ad, check energy balance or heat source terms."
        )

        # ---- Extinction threshold ----
        u.corrections["extinction"] = (
            "Flame extinction occurs when strain rate exceeds critical value or when "
            "heat loss exceeds heat generation. The Damkohler number Da = tau_flow/tau_chem "
            "determines this: Da < 1 means chemistry is too slow to sustain combustion. "
            "Blow-off and flashback are opposing limits of flame stability."
        )

        # ---- Dimensional checks ----
        u.dimensional_checks["temperature"] = "Must be in Kelvin (K); check if Celsius was used"
        u.dimensional_checks["heat_release_rate"] = "W or W/m^3; must match volume/surface convention"
        u.dimensional_checks["activation_energy"] = "J/mol or kJ/mol -- factor of 1000 error is common"
        u.dimensional_checks["species_concentration"] = "mol/m^3 or mass fraction (0-1)"

        u.summary = (
            f"Thermal regime: {u.regime}. Behavior: {u.behavior}. "
            + (f"Ignition onset near {features.inflection_param:.4g}. " if features.has_inflection else "")
            + ("Runaway detected -- no steady-state optimum, only prevention. " if u.regime == "thermal_runaway" else "")
            + ("Cool-flame oscillations detected -- NTC regime precursor to hot ignition. " if u.regime == "cool_flame_oscillation" else "")
            + "Arrhenius kinetics make all thermal thresholds exponentially sensitive to temperature."
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  SENSING / RADAR / DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
class SensingRadarRules(DomainRuleSet):
    domain = "sensing_radar"
    modules = [
        "sensing", "radar_detection", "radar_absorbing_materials",
        "counter_stealth", "counter_uas", "passive_detection",
        "sensor_fusion", "tracking", "radar_waveforms",
        "electronic_countermeasures", "rf_jamming",
    ]

    @classmethod
    def classify_regime(cls, features, observables, config):
        pd = _get_obs_value(observables, "detection_probability")
        snr = _get_obs_value(observables, "snr") or _get_obs_value(observables, "signal_to_noise")

        if pd is not None:
            if pd > 0.9:
                return "reliable_detection"
            elif pd > 0.5:
                return "marginal_detection"
            elif pd > 0.1:
                return "low_probability_detection"
            else:
                return "noise_limited"

        if snr is not None:
            if snr > 20:
                return "high_snr"
            elif snr > 13:
                return "detection_threshold"
            elif snr > 0:
                return "marginal_snr"
            else:
                return "below_threshold"

        if features.has_plateau and features.plateau_value and features.plateau_value > 0.9:
            return "reliable_detection"
        if features.monotonic and features.monotonic_direction == "decreasing":
            return "range_limited"
        return "operational"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        # ---- Governing equations ----
        u.governing_equations = [
            "Radar range equation: R_max = [Pt*G^2*lambda^2*sigma / ((4*pi)^3 * k*T0*Fn*B*SNR_min*L)]^(1/4)",
            "SNR (single pulse): SNR = Pt*G^2*lambda^2*sigma / ((4*pi)^3 * R^4 * k*T0*B*Fn*L)",
            "Integration gain: SNR_N = N * SNR_1 (coherent) or sqrt(N)*SNR_1 (non-coherent)",
            "Pd = 0.5*erfc(erfc_inv(2*Pfa) - sqrt(SNR)) (Swerling 0, Gaussian approximation)",
            "CFAR threshold: T = alpha * (1/N * sum(reference_cells)) where alpha = N*(Pfa^(-1/N) - 1)",
            "Doppler: f_d = 2*v*f_c/c (radial velocity to Doppler frequency)",
            "Range resolution: delta_R = c/(2*B) where B = bandwidth",
            "Velocity resolution: delta_v = lambda/(2*T_dwell)",
        ]

        # ---- Physics bounds ----
        u.physics_bounds["detection_probability"] = (0.0, 1.0)
        u.physics_bounds["false_alarm_probability"] = (1e-12, 1.0)
        u.physics_bounds["rcs_m2"] = (1e-5, 1e4)          # insect to aircraft carrier
        u.physics_bounds["antenna_gain_dBi"] = (0.0, 60.0)
        u.physics_bounds["noise_figure_dB"] = (0.5, 15.0)
        u.physics_bounds["snr_detection_threshold_dB"] = (10.0, 18.0)  # typical CFAR range

        # ---- R^4 scaling law ----
        u.corrections["range_scaling"] = (
            "Detection range follows R^4 law from the radar equation. "
            "Doubling range requires 16x power or 4x antenna area (each G doubles => R doubles). "
            "Do NOT extrapolate linearly. To extend range by factor k: "
            "need k^4 more power, OR k^2 more antenna area, OR k^2 larger RCS. "
            "At R = R_max, SNR = SNR_min; detection probability is at the threshold."
        )

        # ---- Detection probability semantics ----
        u.corrections["detection_probability"] = (
            "Pd depends on BOTH signal and noise statistics. Swerling models: "
            "Case 0: non-fluctuating (deterministic RCS), "
            "Case 1: slow fluctuation (RCS constant within CPI, varies between), "
            "Case 3: fast fluctuation (RCS varies pulse-to-pulse). "
            "Pd increases with SNR but SATURATES near 1.0 asymptotically. "
            "The relationship is sigmoidal, NOT linear."
        )

        # ---- SNR threshold semantics ----
        u.corrections["snr_threshold"] = (
            "SNR > 0 dB does NOT guarantee detection. Detection requires SNR above the "
            "CFAR threshold, which depends on Pfa and the number of reference cells: "
            "Pfa=1e-6: SNR_min ~ 13.2 dB (Swerling 0), "
            "Pfa=1e-8: SNR_min ~ 15.1 dB, "
            "Pfa=1e-4: SNR_min ~ 11.0 dB. "
            "With N-pulse integration: effective SNR = N*SNR_1 (coherent), sqrt(N)*SNR_1 (non-coherent)."
        )

        # ---- Clutter vs. noise ----
        u.corrections["clutter_discrimination"] = (
            "In clutter-limited environments, performance is governed by signal-to-clutter ratio (SCR), "
            "NOT SNR. SCR = sigma_target / (sigma_clutter * resolution_cell_area). "
            "Improving transmit power does NOT help in clutter (noise floor is irrelevant). "
            "Solutions: better resolution (narrower beam, wider bandwidth), Doppler processing, "
            "or STAP (space-time adaptive processing)."
        )

        # ---- False alarm rate ----
        u.corrections["false_alarm_rate"] = (
            "False alarm NUMBER (not rate): FAR = Pfa * N_cells. "
            "If N_cells = 10^6 and Pfa = 10^-6, expect ~1 false alarm per scan. "
            "Reducing Pfa by 10x raises the detection threshold ~1 dB, reducing Pd. "
            "There is a fundamental Pd-Pfa trade-off (ROC curve); you cannot improve both."
        )

        # ---- Pd bounds check ----
        if features.value_range:
            vmin, vmax = features.value_range
            if vmax > 1.0 or vmin < 0.0:
                u.warnings.append(
                    "Detection probability outside [0, 1] is UNPHYSICAL. "
                    "Check: probability vs. percentage (divide by 100?), "
                    "or numerical overflow in erfc computation."
                )
                u.physics_valid = False

        # ---- No "optimum" RCS ----
        if features.monotonic:
            u.rejected_concepts.append(
                "optimal_rcs -- RCS reduction is monotonically beneficial for stealth; "
                "minimum observable RCS is the goal, not an optimum trade-off"
            )

        # ---- Dimensional checks ----
        u.dimensional_checks["range"] = "metres or km; check factor of 1000"
        u.dimensional_checks["rcs"] = "m^2 or dBsm (10*log10(sigma_m2)); common error: linear vs. dB"
        u.dimensional_checks["power"] = "Watts or dBW; Pt in kW to MW range for search radars"
        u.dimensional_checks["frequency"] = "Hz or GHz; check factor of 1e9"

        u.summary = (
            f"Sensing regime: {u.regime}. Behavior: {u.behavior}. "
            + "R^4 law governs range -- no linear extrapolation. "
            + "Pd is sigmoidal with SNR, saturates near 1.0. "
            + ("Detection saturated. " if features.has_plateau and u.regime == "reliable_detection" else "")
            + "Clutter-limited scenarios need SCR analysis, not SNR."
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  NUCLEAR / NEUTRONICS / REACTOR
# ═══════════════════════════════════════════════════════════════════════════════
class NuclearRules(DomainRuleSet):
    domain = "nuclear"
    modules = [
        "neutronics", "monte_carlo_neutron", "burnup",
        "radiation_transport", "reactivity_feedback",
        "reactor_transient", "reactor_transients",
        "radiation_environment", "shielding",
    ]

    # ---- Nuclear constants ----
    DELAYED_NEUTRON_FRACTION_THERMAL = 0.0065   # beta_eff for thermal reactors (U-235)
    DELAYED_NEUTRON_FRACTION_FAST = 0.0035      # beta_eff for fast reactors (Pu-239)
    PROMPT_NEUTRON_LIFETIME_THERMAL = 1e-4      # seconds
    PROMPT_NEUTRON_LIFETIME_FAST = 1e-7         # seconds

    @classmethod
    def classify_regime(cls, features, observables, config):
        k_eff = _get_obs_value(observables, "k_effective") or _get_obs_value(observables, "k_eff")
        reactivity = _get_obs_value(observables, "reactivity")

        if k_eff is not None:
            rho = (k_eff - 1.0) / k_eff  # reactivity
            beta = cls.DELAYED_NEUTRON_FRACTION_THERMAL
            if rho > beta:
                return "prompt_supercritical"
            elif rho > 0.001:
                return "delayed_supercritical"
            elif abs(rho) <= 0.001:
                return "critical"
            elif rho > -beta:
                return "subcritical"
            else:
                return "deeply_subcritical"

        if reactivity is not None:
            beta = cls.DELAYED_NEUTRON_FRACTION_THERMAL
            if reactivity > beta:
                return "prompt_supercritical"
            elif reactivity > 0:
                return "delayed_supercritical"
            elif abs(reactivity) < 0.001:
                return "critical"
            else:
                return "subcritical"

        if features.exponential and features.monotonic_direction == "increasing":
            if features.derivative_trend == "accelerating":
                return "prompt_supercritical"
            return "delayed_supercritical"
        if features.has_plateau:
            return "critical"
        return "unknown"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        # ---- Governing equations ----
        u.governing_equations = [
            "k_effective: k_eff = (neutrons in generation n+1) / (neutrons in generation n)",
            "Reactivity: rho = (k-1)/k; units: dk/k, pcm (1 pcm = 1e-5), dollars ($1 = beta_eff)",
            "Point kinetics: dn/dt = (rho-beta)/Lambda * n + sum(lambda_i * C_i)",
            "Precursor: dC_i/dt = beta_i/Lambda * n - lambda_i * C_i",
            "Period: T = Lambda/(rho-beta) for rho < beta (delayed supercritical)",
            "Prompt jump: n_f/n_i = beta/(beta-rho) (instantaneous power change at rho < beta)",
            "Inhour equation: rho = l*omega + sum(beta_i*omega/(omega+lambda_i))",
            "Burnup: BU(t) = integral(P(t')*dt') / M_fuel [MWd/tHM]",
        ]

        # ---- Physics bounds ----
        u.physics_bounds["k_effective"] = (0.0, 2.0)           # 0=pure absorber, >1.5 very unusual
        u.physics_bounds["reactivity_dollars"] = (-30.0, 1.5)  # beyond $1.5 = prompt supercritical
        u.physics_bounds["neutron_flux_n_cm2_s"] = (0.0, 1e16)
        u.physics_bounds["beta_effective"] = (0.002, 0.008)
        u.physics_bounds["prompt_neutron_lifetime_s"] = (1e-8, 1e-2)
        u.physics_bounds["control_rod_worth_pcm"] = (100, 10000)
        u.physics_bounds["fuel_temperature_K"] = (300, 3200)    # up to UO2 melting ~3120K

        # ---- Criticality is a threshold ----
        u.corrections["criticality"] = (
            "k_eff = 1.0 is the CRITICAL threshold: neutron population is self-sustaining. "
            "k > 1 (supercritical): power rises on a period T = Lambda/(rho-beta). "
            "k < 1 (subcritical): power decays exponentially. "
            "Safety margin is the SHUTDOWN MARGIN: distance from k=1 when all rods inserted, "
            "most reactive rod stuck out. Typical requirement: SDM > 1% dk/k (1000 pcm)."
        )

        # ---- Prompt criticality ----
        if u.regime == "prompt_supercritical":
            u.warnings.append(
                "PROMPT SUPERCRITICAL (rho > beta_eff): power rises on prompt neutron lifetime "
                f"(~{cls.PROMPT_NEUTRON_LIFETIME_THERMAL:.0e} s for thermal). This is a SAFETY-CRITICAL "
                "condition. Power doubles in microseconds, faster than any control system. "
                "This regime must NEVER occur in normal operation."
            )
            u.confidence = "low"
            u.corrections["prompt_critical_response"] = (
                "In prompt supercritical regime, the reactor period is T = Lambda/rho "
                f"(~{cls.PROMPT_NEUTRON_LIFETIME_THERMAL:.0e} s). Only intrinsic feedback "
                "(Doppler, thermal expansion) can terminate the excursion. "
                "If analyzing this regime, check: (1) is the reactivity insertion rate realistic? "
                "(2) are feedback coefficients included? (3) what terminates the excursion?"
            )

        # ---- Reactivity feedback ----
        u.corrections["reactivity_feedback"] = (
            "Reactor stability depends on feedback coefficients: "
            "Doppler coefficient (fuel temperature): alpha_D typically -2 to -5 pcm/K (NEGATIVE = stable). "
            "Moderator temperature coefficient: alpha_M can be + or - depending on boron concentration. "
            "Void coefficient: positive in BWRs under some conditions (safety concern). "
            "All feedback coefficients must have correct SIGN and MAGNITUDE for stability analysis."
        )

        # ---- Xenon dynamics ----
        u.corrections["xenon_poisoning"] = (
            "Xe-135 is the strongest neutron absorber (sigma_a = 2.6e6 barns). "
            "Xe buildup after shutdown peaks at ~11 hours (iodine pit). "
            "Xe oscillations (spatial instability) occur in large cores when local flux "
            "changes cause local Xe to overshoot/undershoot equilibrium. "
            "Period of Xe oscillation: ~24-30 hours. Must be controlled by partial rod insertion."
        )

        # ---- Burnup is monotonic ----
        if features.monotonic and features.monotonic_direction == "increasing":
            u.rejected_concepts = [c for c in u.rejected_concepts if "optimum" not in c]
            u.rejected_concepts.append(
                "burnup_optimum -- burnup (MWd/tHM) is CUMULATIVE energy extraction, "
                "monotonically increasing by definition. 'Optimal burnup' is an economic/fuel-cycle "
                "decision, not a physics optimum."
            )

        # ---- Flux must be non-negative ----
        if features.value_range and features.value_range[0] < 0:
            u.warnings.append(
                "Negative neutron flux is UNPHYSICAL (flux = n*v >= 0). "
                "Check: (a) diffusion solver overshoot near boundaries, "
                "(b) incorrect source term, (c) insufficient spatial resolution."
            )
            u.physics_valid = False

        # ---- Eigenvalue convergence ----
        u.corrections["eigenvalue_convergence"] = (
            "k_eff must converge to < 10 pcm (0.01% dk/k) for safety decisions. "
            "Monte Carlo: statistical uncertainty must be < 0.001 (100 pcm). "
            "Diffusion: iteration error should be < 1e-6 on fission source. "
            "Report the uncertainty/convergence alongside k_eff, always."
        )

        # ---- Dimensional checks ----
        u.dimensional_checks["reactivity"] = "dk/k (dimensionless), pcm (1e-5), or dollars ($1 = beta_eff)"
        u.dimensional_checks["flux"] = "n/(cm^2 s); thermal flux: 1e12-1e14 in power reactors"
        u.dimensional_checks["burnup"] = "MWd/tHM (megawatt-days per tonne heavy metal) or GWd/tU"

        u.summary = (
            f"Nuclear regime: {u.regime}. Behavior: {u.behavior}. "
            + "k_eff=1 is a threshold, not an optimum. "
            + ("PROMPT SUPERCRITICAL -- safety-critical condition. " if u.regime == "prompt_supercritical" else "")
            + "Safety margin = shutdown margin (SDM). "
            + "Burnup is cumulative, not optimizable. "
            + "Xe-135 dynamics dominate post-shutdown behavior."
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  STRUCTURAL / FRACTURE / AEROELASTICITY
# ═══════════════════════════════════════════════════════════════════════════════
class StructuralRules(DomainRuleSet):
    domain = "structural"
    modules = [
        "structural_analysis", "aeroelasticity", "fracture_mechanics",
        "material_erosion", "thermo_mechanical", "tps_ablation",
        "injury_biomechanics",
    ]

    # ---- Material property bounds ----
    YIELD_STRESS = {
        "mild_steel": (200e6, 400e6),       # Pa
        "stainless_steel": (200e6, 800e6),
        "aluminum_alloy": (50e6, 600e6),
        "titanium_alloy": (800e6, 1200e6),
        "cfrp": (500e6, 2000e6),
    }
    FRACTURE_TOUGHNESS = {
        "aluminum": (15, 45),                # MPa*sqrt(m)
        "steel": (30, 150),
        "titanium": (30, 80),
        "ceramic": (1, 10),
    }
    PARIS_LAW_EXPONENT = (2.0, 5.0)          # m in da/dN = C*(dK)^m
    FATIGUE_RATIO = (0.3, 0.6)               # sigma_fatigue / sigma_ult for steel

    @classmethod
    def classify_regime(cls, features, observables, config):
        # Check for safety margin
        margin = _get_obs_value(observables, "structural_margin") or _get_obs_value(observables, "safety_factor")

        if margin is not None:
            if margin > 2.0:
                return "safe_elastic"
            elif margin > 1.0:
                return "low_margin_elastic"
            elif margin > 0:
                return "near_yield"
            else:
                return "failure"

        if features.linear:
            return "elastic"
        if features.has_inflection:
            return "yield_transition"
        if features.exponential and features.monotonic_direction == "increasing":
            return "failure_onset"
        if features.peak_count > 0:
            return "buckling_or_instability"
        return "deformation"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        # ---- Governing equations ----
        u.governing_equations = [
            "Hooke's law (elastic): sigma = E * epsilon (stress-strain proportional)",
            "Von Mises yield: sigma_vm = sqrt(0.5*((s1-s2)^2+(s2-s3)^2+(s3-s1)^2)) <= sigma_y",
            "Safety factor: SF = sigma_y / sigma_applied (must be > 1 for safety)",
            "Stress intensity: K_I = sigma * sqrt(pi*a) * Y(a/W) (mode I crack)",
            "Fracture criterion: K_I >= K_Ic => unstable crack propagation",
            "Paris law (fatigue): da/dN = C * (delta_K)^m",
            "Euler buckling: P_cr = pi^2 * E * I / L_eff^2",
            "Flutter speed: V_f = sqrt(K_eff / (rho * S * q_dyn_sensitivity))",
        ]

        # ---- Physics bounds ----
        u.physics_bounds["safety_factor"] = (0.0, 20.0)
        u.physics_bounds["elastic_modulus_GPa"] = (0.5, 500.0)   # rubber to diamond
        u.physics_bounds["poisson_ratio"] = (0.0, 0.5)           # thermodynamic limit
        u.physics_bounds["paris_law_exponent_m"] = cls.PARIS_LAW_EXPONENT
        u.physics_bounds["strain_to_failure"] = (0.001, 1.0)     # 0.1% to 100%

        # ---- Yield point is a threshold ----
        if features.has_inflection:
            u.key_estimates["yield_onset_parameter"] = features.inflection_param
            u.corrections["yield_point"] = (
                f"Inflection at parameter={features.inflection_param:.4g} indicates yield onset. "
                f"Below: elastic (reversible, Hooke's law). Above: plastic (permanent deformation). "
                f"The yield point is a SAFETY THRESHOLD: "
                f"Factor of Safety = sigma_yield / sigma_applied. "
                f"Typical required FoS: aerospace 1.5, structural 2.0, pressure vessels 3.0-4.0. "
                f"Yield stress is temperature-dependent: decreases ~50% from RT to 0.5*T_melt."
            )

        # ---- Fracture toughness ----
        u.corrections["fracture_threshold"] = (
            "Fracture toughness K_Ic is a MATERIAL PROPERTY (temperature and rate dependent). "
            "The stress intensity factor K_I depends on GEOMETRY and LOADING: "
            "K_I = sigma * sqrt(pi*a) * Y(a/W). "
            "Failure criterion: K_I >= K_Ic => catastrophic crack propagation. "
            "Safety = K_Ic / K_I_applied. "
            "IMPORTANT: K_Ic is a MINIMUM value (plane strain); K_c (plane stress) is higher. "
            "Thickness must satisfy B > 2.5*(K_Ic/sigma_y)^2 for valid K_Ic measurement."
        )

        # ---- Fatigue is NOT monotonic failure ----
        u.corrections["fatigue"] = (
            "Fatigue failure occurs at stresses BELOW yield. S-N curve (Wohler curve) regimes: "
            "1. Low-cycle fatigue (N < 10^4): plastic strain dominated, Coffin-Manson law. "
            "2. High-cycle fatigue (10^4 < N < 10^7): elastic strain dominated, Basquin law. "
            "3. Endurance limit (N > 10^7): for ferrous metals, sigma_e ~ 0.4-0.6 * sigma_ult. "
            "   Non-ferrous metals (Al, Cu) have NO endurance limit -- always fatigue eventually. "
            "Paris law: da/dN = C*(dK)^m where m=2-4 (metals), C is material-specific."
        )

        # ---- Subcritical crack growth ----
        u.corrections["subcritical_crack_growth"] = (
            "Cracks can grow BELOW K_Ic through: "
            "1. Fatigue (cyclic loading): Paris law regime with dK_th < dK < K_Ic. "
            "2. Stress corrosion cracking (SCC): K_ISCC < K < K_Ic in corrosive environment. "
            "3. Creep crack growth: at high T, C* integral governs crack extension. "
            "Threshold dK_th is the MINIMUM stress intensity range for fatigue crack growth "
            "(typically 2-10 MPa*sqrt(m) for metals)."
        )

        # ---- Buckling ----
        if features.peak_count > 0:
            u.corrections["buckling"] = (
                "Load peak followed by decrease indicates BUCKLING instability. "
                "Euler critical load: P_cr = pi^2*E*I/L_eff^2. "
                "Post-buckling behavior depends on geometry: "
                "columns collapse (unstable), plates can carry load beyond buckling (stable post-buckling). "
                "Imperfection sensitivity: real P_cr is 50-80% of Euler prediction due to initial curvature."
            )

        # ---- Deflection monotonicity ----
        if features.monotonic and features.monotonic_direction == "increasing":
            u.rejected_concepts = [c for c in u.rejected_concepts if "optimum" not in c]
            u.rejected_concepts.append(
                "deflection_optimum -- deflection increases monotonically with load in elastic regime "
                "(u = P*L^3/(3*E*I) for cantilever). No minimum deflection at non-zero load."
            )

        # ---- Aeroelastic flutter ----
        u.corrections["flutter"] = (
            "Flutter speed V_f is a STABILITY BOUNDARY, not an operating point. "
            "Below V_f: stable oscillations (damped). Above V_f: divergent oscillations (catastrophic). "
            "Flutter margin = (V_f - V_operating) / V_f. Typically require > 15% margin. "
            "Flutter is coupling between structural modes and aerodynamic forces; "
            "it depends on: mass ratio, frequency ratio, reduced velocity, and structural damping."
        )

        # ---- Dimensional checks ----
        u.dimensional_checks["stress"] = "Pa (or MPa, GPa); common error: MPa vs GPa factor of 1000"
        u.dimensional_checks["strain"] = "Dimensionless (or %); elastic strains < 0.2% for metals"
        u.dimensional_checks["K_I"] = "Pa*sqrt(m) = MPa*sqrt(m); check m vs mm in sqrt(a)"
        u.dimensional_checks["deflection"] = "metres; compare against L/500 (serviceability) or L/200 (ultimate)"

        u.summary = (
            f"Structural regime: {u.regime}. Behavior: {u.behavior}. "
            + ("Yield onset is a safety threshold, not an optimum. " if features.has_inflection else "")
            + "K_I < K_Ic for crack stability. "
            + "Fatigue occurs below yield -- S-N curve governs life. "
            + ("Buckling instability detected. " if features.peak_count > 0 else "")
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  QUANTUM / OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
class QuantumOptimizationRules(DomainRuleSet):
    domain = "quantum_optimization"
    modules = ["quantum_optimization", "quantum_nanoscale"]

    @classmethod
    def classify_regime(cls, features, observables, config):
        # Check solver type from config
        if config:
            solver_type = config.get("solver", {}).get("type", "")
            if "qubo" in str(solver_type).lower() or "ising" in str(solver_type).lower():
                return "combinatorial_optimization"
            if "milp" in str(solver_type).lower():
                return "mixed_integer_optimization"
        return "optimization"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        # ---- Governing equations ----
        u.governing_equations = [
            "QUBO: minimize x^T Q x, x in {0,1}^n",
            "Ising: minimize sum(J_ij s_i s_j) + sum(h_i s_i), s_i in {-1,+1}",
            "QUBO <-> Ising: s = 2x - 1; equivalent formulations",
            "MILP: minimize c^T x, subject to A x <= b, x_i in Z for integer vars",
            "Approximation ratio: alpha = f(x_found) / f(x_optimal)",
        ]

        # ---- Optima DO exist ----
        u.optimum_exists = True
        u.rejected_concepts = [c for c in u.rejected_concepts if "optimum" not in c]

        # ---- Energy landscape ----
        u.corrections["energy_landscape"] = (
            "QUBO/Ising energy landscapes are DISCRETE and typically have many local minima. "
            "The number of local minima grows exponentially with problem size. "
            "A found minimum is NOT guaranteed global unless: "
            "(1) verified by exhaustive enumeration (feasible only for n < ~25), "
            "(2) optimality gap from LP/SDP relaxation is zero, or "
            "(3) the problem has known structure (e.g., convex QUBO). "
            "For practical problems: report the best found solution AND the optimality gap."
        )

        # ---- Quantum vs. classical ----
        u.corrections["quantum_advantage"] = (
            "Quantum annealing and QAOA do NOT guarantee quantum advantage for all instances. "
            "Classical solvers (simulated annealing, Gurobi MILP, branch-and-bound) often "
            "outperform quantum on small-to-medium problems. Quantum advantage appears for: "
            "(1) specific problem structures with quantum tunneling benefits, "
            "(2) large-scale instances where classical heuristics get trapped."
        )

        # ---- Scaling ----
        u.corrections["scaling"] = (
            "NP-hard problems (most QUBO instances): no polynomial-time exact solver exists "
            "(unless P=NP). Exponential worst-case runtime is inherent, not a solver limitation. "
            "Focus on: solution quality at fixed runtime, not runtime to exact optimum."
        )

        # ---- Physics bounds ----
        u.physics_bounds["binary_variables"] = (0, 1)
        u.physics_bounds["ising_spin"] = (-1, 1)

        u.summary = (
            f"Quantum/combinatorial optimization. Regime: {u.regime}. "
            "Optima exist but global optimum is not guaranteed (NP-hard). "
            "Report optimality gap and best-found solution."
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  ELECTROMAGNETICS
# ═══════════════════════════════════════════════════════════════════════════════
class ElectromagneticsRules(DomainRuleSet):
    domain = "electromagnetics"
    modules = [
        "electrostatics", "emi_emc", "electromagnetic_shielding",
        "rcs_shaping", "directed_energy", "ir_signature",
        "acoustic_signature", "thermal_signature_coupling",
        "signature_simulation", "propagation",
    ]

    @classmethod
    def classify_regime(cls, features, observables, config):
        # Check for shielding effectiveness
        se = _get_obs_value(observables, "shielding_effectiveness")
        if se is not None:
            if se > 60:
                return "excellent_shielding"
            elif se > 30:
                return "good_shielding"
            elif se > 10:
                return "marginal_shielding"
            else:
                return "poor_shielding"

        if features.has_plateau:
            return "saturated"
        if features.exponential and features.monotonic_direction == "decreasing":
            return "shielding_effective"
        if features.peak_count > 0:
            return "resonant"
        return "operational"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        # ---- Governing equations ----
        u.governing_equations = [
            "Maxwell's equations: curl(E) = -dB/dt, curl(H) = J + dD/dt, div(D) = rho, div(B) = 0",
            "Coulomb's law: F = q1*q2/(4*pi*eps*r^2)",
            "Poisson's equation (electrostatics): lap(V) = -rho/eps",
            "Skin depth: delta = sqrt(2*rho/(omega*mu)) [penetration depth in conductor]",
            "Shielding effectiveness: SE(dB) = 20*log10(E_inc/E_trans) = A + R + B",
            "  A = absorption loss, R = reflection loss, B = multiple reflection correction",
            "RCS: sigma = lim(4*pi*R^2 * |E_s|^2/|E_i|^2) as R->inf",
            "Friis equation: P_r/P_t = G_t*G_r*(lambda/(4*pi*R))^2",
            "IR radiance: L = eps*sigma*T^4/(pi) [Planck law, broadband approximation]",
        ]

        # ---- Physics bounds ----
        u.physics_bounds["shielding_effectiveness_dB"] = (0.0, 200.0)
        u.physics_bounds["relative_permittivity"] = (1.0, 1e5)     # vacuum to BaTiO3
        u.physics_bounds["relative_permeability"] = (0.0, 1e6)     # diamagnetic to mu-metal
        u.physics_bounds["conductivity_S_m"] = (1e-15, 1e8)        # insulator to superconductor
        u.physics_bounds["rcs_dBsm"] = (-60.0, 50.0)              # stealth to large ship

        # ---- dB is logarithmic ----
        u.corrections["shielding_db"] = (
            "Shielding effectiveness in dB is LOGARITHMIC: "
            "20 dB = 10x field reduction, 40 dB = 100x, 60 dB = 1000x. "
            "Do NOT interpret dB linearly. A 3 dB improvement halves the transmitted power. "
            "SE depends on frequency: absorption increases with sqrt(freq*sigma*mu), "
            "reflection is higher at low frequencies for conductive materials."
        )

        # ---- Skin depth ----
        u.corrections["skin_depth"] = (
            "Skin depth delta = sqrt(2/(omega*mu*sigma)) determines how deep EM fields "
            "penetrate a conductor. At 1 skin depth, amplitude drops to 1/e (37%). "
            "For copper at 1 GHz: delta ~ 2 um. At 60 Hz: delta ~ 8.5 mm. "
            "Shielding requires wall thickness >> delta (typ. 3-5 skin depths for 40+ dB SE)."
        )

        # ---- RCS is frequency and angle dependent ----
        u.corrections["rcs_frequency"] = (
            "RCS varies DRAMATICALLY with frequency and aspect angle. "
            "Rayleigh region (2*pi*a/lambda << 1): sigma ~ f^4 (small objects). "
            "Resonance region (2*pi*a/lambda ~ 1): sigma oscillates. "
            "Optical region (2*pi*a/lambda >> 1): sigma ~ physical projected area. "
            "A single-frequency, single-angle RCS does NOT characterize the full signature."
        )

        # ---- IR signature ----
        u.corrections["ir_signature"] = (
            "IR signature follows Stefan-Boltzmann law: radiance ~ eps*T^4. "
            "10% temperature increase => ~46% radiance increase (T^4 sensitivity). "
            "Atmospheric windows for IR detection: 3-5 um (MWIR), 8-12 um (LWIR). "
            "Background clutter and atmospheric attenuation dominate detection range."
        )

        # ---- Resonance effects ----
        if features.peak_count > 0:
            u.corrections["resonance"] = (
                "Peaks in frequency sweep indicate electromagnetic RESONANCE. "
                "Cavity resonance: f_mnp = c/(2*pi) * sqrt((m*pi/a)^2 + (n*pi/b)^2 + (p*pi/d)^2). "
                "Antenna resonance: occurs when electrical length = lambda/4 or lambda/2. "
                "At resonance, shielding effectiveness may DROP significantly (aperture coupling)."
            )
            u.key_estimates["resonance_parameters"] = features.peak_params

        # ---- Dimensional checks ----
        u.dimensional_checks["electric_field"] = "V/m; typical breakdown in air: 3 MV/m"
        u.dimensional_checks["magnetic_field"] = "A/m (H) or Tesla (B); Earth's field: ~50 uT"
        u.dimensional_checks["frequency"] = "Hz; check GHz vs MHz factor of 1000"
        u.dimensional_checks["wavelength"] = "metres; lambda = c/f; check consistency"

        u.summary = (
            f"EM regime: {u.regime}. Behavior: {u.behavior}. "
            + "dB is logarithmic -- 20 dB = 10x reduction. "
            + "RCS is frequency/angle dependent. "
            + "IR signature scales as T^4 (highly sensitive to temperature). "
            + ("Resonance detected -- SE may degrade at specific frequencies. " if features.peak_count > 0 else "")
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  PROPULSION / AEROSPACE / HYPERSONIC
# ═══════════════════════════════════════════════════════════════════════════════
class PropulsionAerospaceRules(DomainRuleSet):
    domain = "propulsion_aerospace"
    modules = [
        "propulsion", "uav_propulsion", "uav_navigation",
        "plasma_thruster_simulation", "hypersonic_vehicle_simulation",
        "reentry_simulation", "aerothermodynamics", "aero_loads",
        "ballistic_trajectory", "missile_guidance", "missile_defense",
        "uav_aerodynamics",
    ]

    # ---- Isp bounds by propulsion type ----
    ISP_BOUNDS = {
        "solid_rocket": (200, 290),          # s
        "liquid_bipropellant": (280, 460),   # s
        "ion_thruster": (1500, 10000),       # s
        "hall_thruster": (1200, 3000),       # s
        "cold_gas": (30, 80),               # s
    }

    @classmethod
    def classify_regime(cls, features, observables, config):
        mach = _get_obs_value(observables, "mach_number") or _get_obs_value(observables, "mach")
        alt = _get_obs_value(observables, "altitude") or _get_obs_value(observables, "altitude_km")

        if mach is not None:
            if mach < 0.3:
                return "incompressible"
            elif mach < 0.8:
                return "subsonic"
            elif mach < 1.2:
                return "transonic"
            elif mach < 5.0:
                return "supersonic"
            else:
                return "hypersonic"

        if alt is not None:
            # km
            alt_km = alt if alt > 200 else alt  # assume km if < 200
            if alt_km > 100:
                return "exoatmospheric"
            elif alt_km > 80:
                return "rarefied_transition"

        if features.exponential and features.monotonic_direction == "increasing":
            return "heating_dominated"
        return "unknown"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        # ---- Governing equations ----
        u.governing_equations = [
            "Rocket equation: delta_v = Isp * g0 * ln(m0/mf)",
            "Thrust: F = m_dot * Ve + (Pe - Pa) * Ae",
            "Specific impulse: Isp = Ve / g0 = F / (m_dot * g0)",
            "Stagnation enthalpy: h0 = h + V^2/2; T0/T = 1 + (gamma-1)/2 * M^2",
            "Normal shock: M2^2 = ((gamma-1)*M1^2 + 2) / (2*gamma*M1^2 - (gamma-1))",
            "Oblique shock: tan(theta) = 2*cot(beta) * (M1^2*sin^2(beta)-1) / (M1^2*(gamma+cos(2*beta))+2)",
            "Aero heating (Fay-Riddell): q_s ~ C * sqrt(rho_inf/r_n) * V^3 (stagnation point)",
            "Drag: D = 0.5 * rho * V^2 * S * Cd; Cd varies strongly with Mach near M=1",
            "Lift: L = 0.5 * rho * V^2 * S * Cl; stall occurs at Cl_max",
        ]

        # ---- Physics bounds ----
        for ptype, (lo, hi) in cls.ISP_BOUNDS.items():
            u.physics_bounds[f"Isp_{ptype}_s"] = (lo, hi)
        u.physics_bounds["mach_number"] = (0.0, 30.0)           # reentry max ~25
        u.physics_bounds["dynamic_pressure_Pa"] = (0.0, 1e6)    # max ~ 100 kPa in atmosphere
        u.physics_bounds["drag_coefficient"] = (0.01, 2.0)      # streamlined to bluff body
        u.physics_bounds["lift_coefficient"] = (-1.5, 3.0)      # inverted to high-lift
        u.physics_bounds["aoa_degrees"] = (-15.0, 40.0)         # pre-stall range

        # ---- Transonic drag rise ----
        u.corrections["transonic_drag"] = (
            "Drag coefficient has a SHARP rise near Mach 1 (transonic drag rise): "
            "Cd can increase 2-4x between Mach 0.8 and 1.2 due to shock wave formation. "
            "Do NOT assume smooth or linear drag vs. Mach. "
            "The critical Mach number M_cr is where local flow first reaches M=1. "
            "Wave drag: Cd_wave ~ (M^2-1)^(-1/2) in supersonic (Prandtl-Glauert). "
            "Area rule (Whitcomb): minimum wave drag requires smooth cross-section area distribution."
        )

        # ---- Isp bounds ----
        u.corrections["specific_impulse"] = (
            "Isp is bounded by propellant thermochemistry: "
            "Chemical rockets: 200-460 s (limited by combustion temperature and molecular weight). "
            "Electric propulsion: 1000-10000 s (limited by power supply, not chemistry). "
            "Nuclear thermal: 800-1000 s (H2 propellant through reactor core). "
            "Values outside these ranges indicate unit error (check g0 = 9.81 m/s^2) "
            "or invalid assumptions."
        )

        # ---- Hypersonic heating ----
        if u.regime == "hypersonic":
            u.corrections["hypersonic_heating"] = (
                "Aerodynamic heating scales as V^3 at stagnation point (Fay-Riddell). "
                "Stagnation temperature: T0 = T_inf * (1 + 0.2*M^2) for air (gamma=1.4). "
                "At Mach 5: T0 ~ 1800K. At Mach 10: T0 ~ 6000K. At Mach 20: T0 ~ 24000K. "
                "Real-gas effects (dissociation, ionization) become important above M ~ 7: "
                "O2 dissociates at ~2500K, N2 at ~4000K, ionization above ~7000K. "
                "Catalytic vs. non-catalytic wall matters: fully catalytic wall gets 2x the heating."
            )
            u.corrections["tps_requirements"] = (
                "Thermal Protection System (TPS) material limits: "
                "PICA: max ~1600K surface temp, ablation rate ~0.05 mm/s at 100 W/cm^2. "
                "Carbon-carbon: max ~1800K (oxidizing), ~2800K (inert). "
                "UHTC (ZrB2/HfB2): max ~2500K. "
                "Bondline temperature (adhesive to structure) must stay below ~250-400K (material-dependent). "
                "If bondline T exceeds limit => structural failure."
            )

        # ---- Stall ----
        u.corrections["stall"] = (
            "Stall occurs when angle of attack (AoA) exceeds the critical angle (~12-18 deg for "
            "conventional airfoils). Cl drops sharply, Cd increases dramatically. "
            "Stall speed: V_stall = sqrt(2*W/(rho*S*Cl_max)). "
            "Post-stall behavior is highly nonlinear and unsteady."
        )

        # ---- Dimensional checks ----
        u.dimensional_checks["velocity"] = "m/s; Mach 1 at sea level = 340 m/s, at 30km = 303 m/s"
        u.dimensional_checks["altitude"] = "metres or km; check factor of 1000"
        u.dimensional_checks["heat_flux"] = "W/m^2 or W/cm^2; check factor of 1e4"
        u.dimensional_checks["thrust"] = "Newtons (N) or kN; check g0 factor in Isp conversion"

        u.summary = (
            f"Aerospace regime: {u.regime}. Behavior: {u.behavior}. "
            + ("Transonic drag rise near M=1 is nonlinear. " if u.regime == "transonic" else "")
            + ("Hypersonic heating ~ V^3; real-gas effects above M~7. " if u.regime == "hypersonic" else "")
            + "Isp has propellant-type-specific upper bounds. "
            + "Stall is an AoA threshold, not an optimization target."
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  ELECTROCHEMISTRY
# ═══════════════════════════════════════════════════════════════════════════════
class ElectrochemistryRules(DomainRuleSet):
    domain = "electrochemistry"
    modules = [
        "electrochemistry", "gas_generation", "power_processing",
    ]

    @classmethod
    def classify_regime(cls, features, observables, config):
        if features.linear:
            return "ohmic"
        if features.has_plateau:
            return "mass_transport_limited"
        if features.exponential:
            return "activation_controlled"
        if features.has_inflection:
            return "mixed_kinetics"
        if features.peak_count > 0:
            return "voltammetric_peaks"
        return "mixed_kinetics"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        # ---- Governing equations ----
        u.governing_equations = [
            "Butler-Volmer: i = i0 * [exp(alpha_a*F*eta/(R*T)) - exp(-alpha_c*F*eta/(R*T))]",
            "Tafel (high overpotential): eta = a + b*log(i), b = 2.303*R*T/(alpha*n*F)",
            "Nernst equation: E = E0 + (R*T)/(n*F) * ln(a_ox/a_red)",
            "Limiting current: i_L = n*F*D*c_bulk/delta (diffusion layer thickness delta)",
            "Randles-Sevcik (CV peak): i_p = 0.4463*n*F*A*c*(n*F*D*v/(R*T))^0.5",
            "Warburg impedance: Z_W = sigma*omega^(-0.5)*(1-j) (semi-infinite diffusion)",
            "Sand equation (transition time): tau = pi*D*(n*F*c)^2 / (4*i^2)",
            "Faraday's law: m = (M*I*t)/(n*F) (mass deposited/dissolved)",
        ]

        # ---- Physics bounds ----
        u.physics_bounds["transfer_coefficient_alpha"] = (0.0, 1.0)
        u.physics_bounds["exchange_current_density_A_cm2"] = (1e-12, 1e-1)
        u.physics_bounds["standard_reduction_potential_V_SHE"] = (-3.0, 3.0)
        u.physics_bounds["diffusion_coefficient_cm2_s"] = (1e-9, 1e-4)
        u.physics_bounds["double_layer_capacitance_uF_cm2"] = (5.0, 150.0)
        u.physics_bounds["tafel_slope_mV_decade"] = (30.0, 120.0)  # 30 = alpha=2, 120 = alpha=0.5

        # ---- Tafel behavior ----
        if features.exponential:
            u.corrections["tafel_behavior"] = (
                "Exponential I-V follows Tafel kinetics (high-overpotential limit of Butler-Volmer). "
                "Tafel slope b = 2.303*R*T/(alpha*n*F): "
                "At 25C: b = 59.2/(alpha*n) mV/decade. "
                "For alpha=0.5, n=1: b = 118 mV/decade (common for outer-sphere ET). "
                "For alpha=0.5, n=2: b = 59 mV/decade. "
                "Slope change indicates mechanism change (rate-determining step shift)."
            )

        # ---- Overpotential semantics ----
        u.corrections["overpotential"] = (
            "Overpotential eta = E_applied - E_equilibrium is always an irreversible LOSS: "
            "eta > 0 for anodic (oxidation), eta < 0 for cathodic (reduction). "
            "Components: eta_activation (Butler-Volmer) + eta_ohmic (IR drop) + eta_concentration (mass transport). "
            "At high currents, mass transport limits: i -> i_L and eta_conc -> -infinity."
        )

        # ---- Limiting current ----
        if features.has_plateau:
            u.key_estimates["limiting_current_plateau"] = features.plateau_value
            u.corrections["limiting_current"] = (
                f"Current plateau at ~{features.plateau_value:.4g} indicates MASS TRANSPORT limitation. "
                f"The limiting current i_L = n*F*D*c_bulk/delta depends on: "
                f"diffusion coefficient D (~10^-5 cm^2/s for aqueous ions), "
                f"bulk concentration c_bulk, and diffusion layer thickness delta (~0.01-0.05 cm still, "
                f"~10^-3 cm with stirring). "
                f"Increasing current beyond i_L is IMPOSSIBLE without changing mass transport "
                f"(stirring, flow, thinner delta)."
            )
            u.rejected_concepts.append(
                "exceeding_limiting_current -- current cannot exceed i_L without changing mass transport"
            )

        # ---- Cyclic voltammetry peaks ----
        if features.peak_count > 0:
            u.corrections["cv_peaks"] = (
                "Peaks in I-V indicate voltammetric (CV) behavior: "
                "Forward peak: surface concentration depleted, current limited by diffusion. "
                "Peak separation dE_p: reversible = 59/n mV (Nernstian), "
                "quasi-reversible = 59/n to 200/n mV, irreversible > 200/n mV. "
                "Peak current: i_p ~ v^0.5 for diffusion-controlled (Randles-Sevcik), "
                "i_p ~ v for adsorption-controlled."
            )
            u.key_estimates["cv_peak_parameters"] = features.peak_params

        # ---- Open circuit potential ----
        u.corrections["open_circuit_potential"] = (
            "OCP (open circuit potential) is the equilibrium (Nernst) potential: "
            "E_OCP = E0 + (R*T/nF)*ln(a_ox/a_red). At OCP, net current = 0 "
            "(anodic and cathodic currents cancel). OCP is a starting point for analysis, "
            "not an operating point."
        )

        # ---- Dimensional checks ----
        u.dimensional_checks["current_density"] = "A/cm^2 or mA/cm^2; check factor of 1000"
        u.dimensional_checks["potential"] = "V vs. SHE (or other reference); state the reference electrode"
        u.dimensional_checks["scan_rate"] = "mV/s or V/s; peak current scales as sqrt(v)"
        u.dimensional_checks["concentration"] = "mol/L (M) or mol/cm^3; check factor of 1000"

        u.summary = (
            f"Electrochemical regime: {u.regime}. Behavior: {u.behavior}. "
            + ("Tafel kinetics: slope reveals transfer coefficient and mechanism. " if features.exponential else "")
            + ("Mass transport limited at i_L -- cannot exceed without changing hydrodynamics. " if features.has_plateau else "")
            + ("CV peaks detected -- peak separation reveals reaction reversibility. " if features.peak_count > 0 else "")
            + "Overpotential is always a loss."
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  PLASMA PHYSICS
# ═══════════════════════════════════════════════════════════════════════════════
class PlasmaRules(DomainRuleSet):
    domain = "plasma"
    modules = [
        "plasma_fluid", "particle_in_cell", "sheath_model",
        "nonequilibrium",
    ]

    @classmethod
    def classify_regime(cls, features, observables, config):
        Te = _get_obs_value(observables, "electron_temperature") or _get_obs_value(observables, "T_e")
        ne = _get_obs_value(observables, "electron_density") or _get_obs_value(observables, "n_e")

        if Te is not None and Te > 1e6:
            return "high_temperature_plasma"
        if Te is not None and Te < 1e4:
            return "low_temperature_plasma"

        if features.has_plateau:
            return "steady_state_plasma"
        if features.exponential and features.monotonic_direction == "increasing":
            return "ionization_growth"
        if features.peak_count > 0:
            return "instability_oscillation"
        return "transient_plasma"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        # ---- Governing equations ----
        u.governing_equations = [
            "Debye length: lambda_D = sqrt(eps_0 * k_B * T_e / (n_e * e^2))",
            "Plasma frequency: omega_pe = sqrt(n_e * e^2 / (eps_0 * m_e))",
            "Cyclotron frequency: omega_ce = e*B/m_e (electron), omega_ci = Z*e*B/m_i (ion)",
            "Bohm criterion: v_i >= sqrt(k_B*T_e/m_i) at sheath edge",
            "Child-Langmuir law: J = (4/9)*eps_0*sqrt(2*e/m_i)*(V^(3/2)/d^2) [sheath current]",
            "Spitzer resistivity: eta = 5.2e-5 * Z * ln(Lambda) / T_e^(3/2) [Ohm*m, T_e in eV]",
            "Ambipolar diffusion: D_a = D_i * (1 + T_e/T_i) [enhanced ion diffusion]",
            "Ionization rate: S_iz = n_e * n_n * <sigma*v>_iz (Arrhenius-like with T_e)",
            "Quasi-neutrality: |n_e - Z*n_i| / n_e << 1 (in bulk plasma, not sheaths)",
        ]

        # ---- Physics bounds ----
        u.physics_bounds["electron_temperature_eV"] = (0.01, 1e5)    # cold plasma to fusion
        u.physics_bounds["electron_density_cm3"] = (1e6, 1e25)       # ionosphere to fusion core
        u.physics_bounds["debye_length_m"] = (1e-6, 1e2)             # dense plasma to space
        u.physics_bounds["plasma_frequency_Hz"] = (1e6, 1e15)
        u.physics_bounds["ionization_fraction"] = (1e-8, 1.0)
        u.physics_bounds["coulomb_logarithm"] = (5, 30)

        # ---- Debye shielding ----
        u.corrections["debye_shielding"] = (
            "Debye length lambda_D sets the FUNDAMENTAL spatial scale: "
            "fields are shielded beyond lambda_D in the bulk plasma. "
            "Grid resolution MUST be finer than lambda_D for accurate sheath resolution. "
            "Typical values: glow discharge ~0.1 mm, fusion ~0.01 mm, space plasma ~10 m. "
            "If grid spacing >> lambda_D, sheath physics is NOT resolved -- results are qualitative only."
        )

        # ---- Plasma frequency ----
        u.corrections["plasma_frequency"] = (
            "omega_pe sets the FASTEST timescale. Timestep must satisfy: "
            "dt < 0.1 / omega_pe for explicit PIC, dt < 0.2 / omega_pe for fluid. "
            "omega_pe = 56.4 * sqrt(n_e [cm^-3]) rad/s. "
            "For n_e = 1e12 cm^-3: omega_pe ~ 56 GHz. "
            "EM waves below omega_pe are REFLECTED (skin effect); above are transmitted."
        )

        # ---- Collisionality ----
        u.corrections["collisionality"] = (
            "Plasma regime depends on collision frequency nu_c vs. other rates: "
            "Collisional (nu_c >> omega_pe): fluid description valid, Ohm's law applies. "
            "Collisionless (nu_c << omega_pe): kinetic description needed (PIC or Vlasov). "
            "Magnetized (omega_ce >> nu_c): particles gyrate; transport is anisotropic. "
            "Unmagnetized (omega_ce << nu_c): magnetic field negligible for transport. "
            "The Coulomb collision frequency: nu_ei ~ n_e * Z^2 * e^4 * ln(Lambda) / (m_e^0.5 * (k*Te)^1.5)"
        )

        # ---- Sheath physics ----
        u.corrections["sheath"] = (
            "Sheath forms at ALL surfaces in contact with plasma. "
            "Bohm criterion: ions must enter sheath at >= ion acoustic speed cs = sqrt(kT_e/m_i). "
            "Sheath potential drop: ~3-5 * kT_e/e for floating surface. "
            "Child-Langmuir law limits current collection. "
            "Sheath thickness ~ few Debye lengths (up to 100s in high-voltage sheaths). "
            "DO NOT assume quasi-neutrality inside the sheath."
        )

        # ---- Quasi-neutrality check ----
        ne = _get_obs_value(observables, "electron_density") or _get_obs_value(observables, "n_e")
        ni = _get_obs_value(observables, "ion_density") or _get_obs_value(observables, "n_i")
        if ne is not None and ni is not None and ne > 0:
            ratio = abs(ne - ni) / ne
            u.key_estimates["quasi_neutrality_error"] = round(ratio, 6)
            if ratio > 0.01:
                u.warnings.append(
                    f"Quasi-neutrality violated: |n_e - n_i|/n_e = {ratio:.3g} (should be << 0.01 in bulk). "
                    f"This is expected in sheaths but indicates error in bulk plasma."
                )

        # ---- Non-equilibrium ----
        u.corrections["non_equilibrium"] = (
            "In non-equilibrium (non-thermal) plasmas: T_e >> T_i >> T_gas. "
            "Typical: T_e = 1-10 eV, T_i = 0.05-0.5 eV, T_gas = 300-1000 K. "
            "Electron temperature determines ionization and dissociation rates. "
            "Ion temperature determines sputtering and surface modification. "
            "Gas temperature determines neutral chemistry and thermal effects. "
            "Do NOT assume T_e = T_i unless explicitly verified (LTE condition)."
        )

        # ---- Instabilities ----
        if features.peak_count > 0:
            u.corrections["instabilities"] = (
                "Oscillatory behavior indicates plasma INSTABILITY: "
                "Drift waves: omega ~ k_y * v_diamagnetic (in magnetized plasma). "
                "Two-stream instability: when beam velocity > thermal velocity. "
                "Ionization instability: positive feedback between ionization and heating. "
                "Breathing mode (in Hall thrusters): ~10-30 kHz oscillation of ionization front."
            )
            u.key_estimates["oscillation_parameters"] = features.peak_params

        # ---- Dimensional checks ----
        u.dimensional_checks["temperature"] = "eV or K; 1 eV = 11604 K; check conversion"
        u.dimensional_checks["density"] = "cm^-3 or m^-3; factor of 1e6 between them"
        u.dimensional_checks["potential"] = "Volts; sheath drop ~ few T_e/e"
        u.dimensional_checks["magnetic_field"] = "Tesla or Gauss; 1 T = 10000 G"

        u.summary = (
            f"Plasma regime: {u.regime}. Behavior: {u.behavior}. "
            + "Debye length = minimum grid scale. Plasma frequency = maximum timestep. "
            + "Sheath forms at all surfaces (Bohm criterion). "
            + "Non-equilibrium: T_e >> T_i is normal, not an error. "
            + ("Instability oscillations detected. " if features.peak_count > 0 else "")
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  SAFETY / HAZARD MODULES
# ═══════════════════════════════════════════════════════════════════════════════
class SafetyRules(DomainRuleSet):
    domain = "safety"
    modules = [
        "safety", "safety_logic", "hv_safety",
        "warhead_lethality", "collision_avoidance",
        "swarm_dynamics", "sensors_actuators",
    ]

    @classmethod
    def classify_regime(cls, features, observables, config):
        margin = _get_obs_value(observables, "safety_margin") or _get_obs_value(observables, "margin")

        if margin is not None:
            if margin > 0.5:
                return "safe_with_margin"
            elif margin > 0:
                return "marginal_safety"
            else:
                return "unsafe"

        if features.monotonic and features.monotonic_direction == "increasing":
            return "escalating_hazard"
        if features.has_plateau:
            return "bounded_risk"
        if features.exponential:
            return "runaway_hazard"
        return "variable_risk"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        # ---- Governing equations ----
        u.governing_equations = [
            "Risk = Probability * Consequence (probability x severity matrix)",
            "Safety margin = (Capacity - Demand) / Capacity",
            "Reliability: R(t) = exp(-lambda*t) (exponential failure distribution)",
            "MTBF = 1/lambda (mean time between failures)",
            "Fault tree: P(top) = 1 - product(1-P(basic_event_i)) for OR gate",
            "HV safety: I_lethal > 50 mA (ventricular fibrillation threshold at 50/60 Hz)",
            "Blast scaling: Z = R/W^(1/3) (Hopkinson-Cranz, scaled distance)",
        ]

        # ---- Physics bounds ----
        u.physics_bounds["probability"] = (0.0, 1.0)
        u.physics_bounds["safety_margin"] = (-1.0, 10.0)
        u.physics_bounds["reliability"] = (0.0, 1.0)
        u.physics_bounds["lethal_current_mA_AC"] = (30.0, 100.0)   # 50 mA typical threshold

        # ---- Safety is pass/fail ----
        u.corrections["safety_threshold"] = (
            "Safety metrics have PASS/FAIL thresholds, not optima. "
            "The goal is SUFFICIENT MARGIN below the hazard threshold: "
            "margin = (threshold - value) / threshold * 100%. "
            "Typical engineering margins: 20% for well-characterized systems, "
            "50% for uncertain/novel systems, 100% for safety-critical with human exposure. "
            "A 'better' safety metric is one with MORE margin, not a minimum."
        )

        # ---- Runaway hazard ----
        if u.regime == "runaway_hazard":
            u.corrections["runaway_escalation"] = (
                "Exponential hazard escalation indicates positive feedback: "
                "each increment of the hazard variable accelerates the next. "
                "Examples: thermal runaway, cascading failures, chain reactions. "
                "Mitigation MUST break the feedback loop, not just slow it. "
                "There is no stable equilibrium in runaway -- only prevention or containment."
            )
            u.rejected_concepts.append(
                "hazard_optimum -- runaway hazard has no equilibrium; it must be prevented entirely"
            )

        # ---- HV safety specifics ----
        u.corrections["hv_safety"] = (
            "Electric shock hazard depends on CURRENT through body, not voltage alone: "
            "I = V / R_body where R_body = 500-5000 Ohm (dry skin high, wet skin low). "
            "Thresholds (IEC 60479): perception ~1 mA, let-go ~10 mA, "
            "ventricular fibrillation ~50 mA (50/60 Hz, > 1 cardiac cycle). "
            "DC is less dangerous at same current than AC (higher threshold). "
            "Touch voltage limits: 50V AC / 120V DC in normal conditions (IEC 60364)."
        )

        # ---- Collision avoidance ----
        u.corrections["collision_metrics"] = (
            "Time-to-collision (TTC) and miss distance are the primary metrics. "
            "TTC < 0 means collision has already occurred or is inevitable. "
            "Minimum separation distance depends on: closing speed, reaction time, "
            "maneuver capability, and uncertainty in state estimation. "
            "Safety requires: actual_separation > required_separation at all times."
        )

        # ---- Dimensional checks ----
        u.dimensional_checks["probability"] = "Dimensionless [0,1]; NOT percentage unless stated"
        u.dimensional_checks["current"] = "Amperes; mA for body current thresholds"
        u.dimensional_checks["distance"] = "metres; check km vs m for collision avoidance"

        u.summary = (
            f"Safety regime: {u.regime}. Behavior: {u.behavior}. "
            + "Safety metrics are pass/fail thresholds with required margins. "
            + ("Escalating hazard -- feedback loop must be broken. " if u.regime in ("escalating_hazard", "runaway_hazard") else "")
            + "Report: margin = (threshold - value)/threshold as percentage."
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE CHANGE / BOILING / SURFACE TENSION
# ═══════════════════════════════════════════════════════════════════════════════
class PhaseChangeRules(DomainRuleSet):
    domain = "phase_change"
    modules = [
        "phase_change", "nucleate_boiling", "surface_tension",
        "contact_line", "microlayer",
    ]

    @classmethod
    def classify_regime(cls, features, observables, config):
        if features.has_inflection and features.peak_count > 0:
            return "boiling_crisis"
        if features.has_inflection:
            return "boiling_transition"
        if features.has_plateau:
            return "film_boiling"
        if features.exponential:
            return "nucleate_boiling"
        if features.linear:
            return "natural_convection"
        return "subcooled"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        # ---- Governing equations ----
        u.governing_equations = [
            "Clausius-Clapeyron: dP/dT = h_fg / (T * Delta_v) [phase boundary slope]",
            "Nucleation rate: J = J0 * exp(-16*pi*gamma^3 / (3*kT*(Delta_G_v)^2))",
            "CHF (Zuber): q_CHF = 0.131 * h_fg * rho_v * [sigma*g*(rho_l-rho_v)/rho_v^2]^(1/4)",
            "Rohsenow (nucleate): q = mu_l*h_fg * [g*(rho_l-rho_v)/sigma]^0.5 * [cp*dT/(C_sf*h_fg*Pr^s)]^3",
            "Film boiling (Bromley): h = 0.62 * [k_v^3*rho_v*(rho_l-rho_v)*g*h_fg'/(mu_v*D*dT)]^0.25",
            "Young-Laplace: dP = gamma * (1/R1 + 1/R2) [capillary pressure]",
            "Contact angle (Young): cos(theta) = (gamma_sv - gamma_sl) / gamma_lv",
            "Stefan problem: X(t) = 2*lambda*sqrt(alpha*t) [solidification front position]",
        ]

        # ---- Physics bounds ----
        u.physics_bounds["superheat_K"] = (0.0, 500.0)
        u.physics_bounds["heat_transfer_coefficient_W_m2K"] = (1.0, 1e6)  # natural conv to boiling
        u.physics_bounds["surface_tension_N_m"] = (1e-4, 1.0)            # liquid metals to organics
        u.physics_bounds["contact_angle_degrees"] = (0.0, 180.0)
        u.physics_bounds["latent_heat_J_kg"] = (1e4, 1e7)

        # ---- Boiling curve regimes ----
        u.corrections["boiling_curve"] = (
            "The boiling curve (heat flux vs. superheat) has FOUR distinct regimes: "
            "1. Natural convection (dT_sat < 5K): q ~ dT^1.33 (Rayleigh correlation). "
            "2. Nucleate boiling (5K < dT_sat < 30K): q ~ dT^3 (steep increase, very efficient). "
            "3. Critical Heat Flux (CHF) / boiling crisis: q_max (Zuber correlation). "
            "4. Film boiling (dT_sat > 100K): vapor blanket, h DROPS (Leidenfrost effect). "
            "CHF is a SAFETY LIMIT: exceeding it causes rapid temperature excursion "
            "(dryout in nuclear reactors, burnout in electronics cooling). "
            "The transition boiling regime (between CHF and film boiling) is UNSTABLE."
        )

        if features.has_inflection:
            u.key_estimates["transition_parameter"] = features.inflection_param
            u.corrections["boiling_transition"] = (
                f"Inflection at {features.inflection_param:.4g} likely indicates CHF or onset of "
                f"transition boiling. Below this: efficient nucleate boiling. Above: vapor blanketing "
                f"causes dramatic decrease in heat transfer. This is a CRITICAL SAFETY THRESHOLD "
                f"in nuclear reactor thermalhydraulics and electronics cooling."
            )

        # ---- CHF ----
        u.corrections["critical_heat_flux"] = (
            "CHF (critical heat flux) is the MAXIMUM heat flux sustainable by nucleate boiling. "
            "Zuber's pool boiling CHF: q_CHF = 0.131*h_fg*rho_v*[sigma*g*(rho_l-rho_v)/rho_v^2]^0.25. "
            "For water at 1 atm: q_CHF ~ 1.0-1.3 MW/m^2. "
            "CHF depends on: pressure, geometry, mass flux (flow boiling), subcooling, surface. "
            "DNB (departure from nucleate boiling) ratio: DNBR = q_CHF/q_actual. "
            "Nuclear safety limit: DNBR > 1.3 (with margin for uncertainty)."
        )

        # ---- Contact angle ----
        u.corrections["contact_angle"] = (
            "Contact angle theta determines wettability: "
            "theta < 90 deg: hydrophilic (liquid spreads). "
            "theta > 90 deg: hydrophobic (liquid beads up). "
            "theta ~ 0 deg: complete wetting. theta ~ 180 deg: superhydrophobic. "
            "Contact angle is affected by: surface roughness (Wenzel/Cassie-Baxter), "
            "temperature, contamination, and dynamic effects (advancing > receding)."
        )

        # ---- Stefan problem ----
        u.corrections["solidification_melting"] = (
            "Phase change front moves as sqrt(time) (Stefan problem): X ~ 2*lambda*sqrt(alpha*t). "
            "The Stefan number St = cp*dT/h_fg controls the relative rates of sensible and latent heat. "
            "St << 1: latent heat dominates, slow front. St >> 1: sensible heat dominates, fast front. "
            "Mushy zone exists for alloys (liquidus to solidus temperature range)."
        )

        # ---- Dimensional checks ----
        u.dimensional_checks["heat_flux"] = "W/m^2; typical CHF ~ 1 MW/m^2 for water at 1 atm"
        u.dimensional_checks["temperature"] = "Kelvin or Celsius; superheat = T_surface - T_sat"
        u.dimensional_checks["surface_tension"] = "N/m; water at 20C ~ 0.073 N/m"

        u.summary = (
            f"Phase change regime: {u.regime}. Behavior: {u.behavior}. "
            + "Boiling curve has 4 regimes; CHF is the critical safety limit. "
            + ("Boiling transition/CHF detected -- safety threshold. " if features.has_inflection else "")
            + "Contact angle determines wetting; solidification front ~ sqrt(t)."
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  DIAGNOSTICS / GROUND TRUTH / UNCERTAINTY (catch-all for remaining modules)
# ═══════════════════════════════════════════════════════════════════════════════
class DiagnosticsUncertaintyRules(DomainRuleSet):
    domain = "diagnostics_uncertainty"
    modules = [
        "diagnostics", "ground_truth", "uncertainty",
    ]

    @classmethod
    def classify_regime(cls, features, observables, config):
        if features.has_plateau:
            return "converged"
        if features.monotonic and features.monotonic_direction == "decreasing":
            return "convergent"
        if features.monotonic and features.monotonic_direction == "increasing":
            return "divergent"
        return "diagnostic"

    @classmethod
    def correct_interpretation(cls, features, observables, config, params, values):
        u = super().correct_interpretation(features, observables, config, params, values)

        u.governing_equations = [
            "Monte Carlo standard error: SE = sigma / sqrt(N)",
            "Sobol sensitivity: S_i = V(E(Y|X_i)) / V(Y) [first-order]",
            "Convergence: error(h) = C * h^p where p = order of accuracy",
            "GCI (grid convergence index): GCI = Fs * |f2-f1| / (r^p - 1)",
            "Richardson extrapolation: f_exact ~ f1 + (f1-f2)/(r^p - 1)",
        ]

        u.corrections["uncertainty_quantification"] = (
            "UQ results give CONFIDENCE INTERVALS, not point values. "
            "Report: mean +/- 2*sigma (95% CI) or 5th-95th percentile. "
            "Coefficient of variation CV = sigma/mean: "
            "CV < 5%: low uncertainty (trustworthy). "
            "CV 5-20%: moderate (report with caution). "
            "CV > 20%: high (conclusions are weak). "
            "Sensitivity indices (Sobol) rank which inputs drive output variance."
        )

        u.corrections["grid_convergence"] = (
            "Grid convergence must be verified before trusting quantitative results: "
            "Run at 3 grid levels (coarse, medium, fine) and apply GCI. "
            "GCI < 5% indicates adequate resolution. "
            "Richardson extrapolation gives the grid-independent estimate. "
            "Results on a single grid are QUALITATIVE only."
        )

        u.corrections["statistical_convergence"] = (
            "Monte Carlo results: standard error SE = sigma/sqrt(N). "
            "For 1% relative error: need N > (100*CV)^2 samples. "
            "For Sobol indices: need N*(2k+2) total evaluations (k = number of parameters). "
            "Convergence of mean != convergence of tails (use bootstrap CI for percentiles)."
        )

        u.summary = (
            f"Diagnostic/UQ regime: {u.regime}. "
            "Report confidence intervals, not point values. "
            "CV < 5% = reliable, CV > 20% = uncertain. "
            "Grid convergence required for quantitative trust."
        )

        return u


# ═══════════════════════════════════════════════════════════════════════════════
#  Rule Registry
# ═══════════════════════════════════════════════════════════════════════════════
ALL_RULE_SETS: List[type] = [
    SemiconductorRules,
    FluidDynamicsRules,
    ThermalCombustionRules,
    SensingRadarRules,
    NuclearRules,
    StructuralRules,
    QuantumOptimizationRules,
    ElectromagneticsRules,
    PropulsionAerospaceRules,
    ElectrochemistryRules,
    PlasmaRules,
    SafetyRules,
    PhaseChangeRules,
    DiagnosticsUncertaintyRules,
]

# Build module -> rule set lookup
_MODULE_TO_RULES: Dict[str, type] = {}
for _rs in ALL_RULE_SETS:
    for _mod in _rs.modules:
        _MODULE_TO_RULES[_mod] = _rs


def get_rule_set(module_name: str) -> type:
    """Get the domain rule set for a module, falling back to generic."""
    return _MODULE_TO_RULES.get(module_name, DomainRuleSet)


# ═══════════════════════════════════════════════════════════════════════════════
#  Step 5: Main SPL Entry Point
# ═══════════════════════════════════════════════════════════════════════════════
def _extract_sweep_data(results: List[Dict]) -> Tuple[List[float], List[float], str]:
    """Extract parameter-value pairs from sweep results."""
    for r in results:
        if r.get("_tool_name") != "sweep_parameter":
            continue
        sweep_results = r.get("results", [])
        if not sweep_results:
            continue

        params = []
        values = []
        obs_name = ""

        for sr in sweep_results:
            if not sr.get("success"):
                continue
            pv = sr.get("param_value")
            if pv is None:
                continue

            obs = sr.get("observables", {})
            best_val = None
            if isinstance(obs, dict):
                for oname, odata in obs.items():
                    if isinstance(odata, dict):
                        v = odata.get("value")
                        if isinstance(v, (int, float)):
                            best_val = v
                            obs_name = oname
                            break

            if best_val is not None:
                params.append(float(pv))
                values.append(float(best_val))

        if params:
            return params, values, obs_name

    return [], [], ""


def _extract_observables_dict(results: List[Dict]) -> Dict:
    """Extract a flat dict of observable name -> {value, unit, ...} from results."""
    obs_dict = {}
    for r in results:
        observables = r.get("observables", {})
        if isinstance(observables, list):
            for o in observables:
                obs_dict[o.get("name", "")] = o
        elif isinstance(observables, dict):
            obs_dict.update(observables)

        if r.get("_tool_name") == "sweep_parameter":
            for sr in r.get("results", []):
                sr_obs = sr.get("observables", {})
                if isinstance(sr_obs, dict):
                    obs_dict.update(sr_obs)
    return obs_dict


def interpret_results(
    results: List[Dict[str, Any]],
    module_name: Optional[str] = None,
    config: Optional[Dict] = None,
) -> SemanticUnderstanding:
    """Main SPL entry point: interpret solver results into semantic understanding."""
    if not module_name:
        for r in results:
            mn = r.get("module_name")
            if mn:
                module_name = mn
                break
    if not module_name:
        module_name = "unknown"

    config = config or {}

    RuleSet = get_rule_set(module_name)

    params, values, obs_name = _extract_sweep_data(results)
    observables = _extract_observables_dict(results)

    if not params and observables:
        features = SemanticFeatures(data_points=1)
    else:
        features = extract_features(params, values)

    understanding = RuleSet.correct_interpretation(
        features, observables, config, params, values
    )

    logger.info(
        "SPL [%s/%s]: regime=%s, behavior=%s, optimum=%s, valid=%s, rejected=%d, bounds=%d, eqs=%d",
        module_name, RuleSet.domain, understanding.regime, understanding.behavior,
        understanding.optimum_exists, understanding.physics_valid,
        len(understanding.rejected_concepts), len(understanding.physics_bounds),
        len(understanding.governing_equations),
    )

    return understanding


def full_semantic_analysis(
    results: List[Dict[str, Any]],
    module_name: Optional[str] = None,
    config: Optional[Dict] = None,
) -> Tuple[SemanticUnderstanding, str]:
    """Run full SPL analysis and return (understanding, context_string)."""
    understanding = interpret_results(results, module_name, config)
    context = understanding.to_context()
    return understanding, context
