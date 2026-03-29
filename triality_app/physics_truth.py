"""
Physics Truth Layer (PTL) for Triality
=======================================
Validates analysis results against physics constraints before they reach the LLM.
Detects contradictions, recomputes known quantities from first principles when
solver values are suspect, and blocks high-confidence answers when violations exist.

This layer sits between the solver results and the LLM summarization:
    Solver → Observables → **PTL** → LLM

Responsibilities:
    1. Observable validation against physical bounds
    2. Recomputation of known quantities from first principles
    3. Contradiction detection between observables
    4. Confidence downgrade when violations are found
    5. Correction annotations injected into LLM context
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("triality.physics_truth")


# ---------------------------------------------------------------------------
#  Physics Constants
# ---------------------------------------------------------------------------
K_BOLTZMANN_EV = 8.617e-5   # eV/K
Q_ELECTRON = 1.602e-19      # C
EPS_0 = 8.854e-14           # F/cm
NI_SI_300K = 1.5e10          # intrinsic carrier concentration for Si at 300K (cm^-3)
EPS_SI = 11.7                # relative permittivity of Si


# ---------------------------------------------------------------------------
#  Data Classes
# ---------------------------------------------------------------------------
@dataclass
class PhysicsViolation:
    """A detected physics violation in analysis results."""
    severity: str          # "critical", "warning", "info"
    observable: str        # which observable is wrong
    reported_value: Any    # what the solver said
    expected_range: Optional[Tuple[float, float]] = None
    corrected_value: Optional[float] = None
    explanation: str = ""
    recomputed: bool = False


@dataclass
class TruthReport:
    """Complete truth validation report for a set of results."""
    violations: List[PhysicsViolation] = field(default_factory=list)
    corrections: Dict[str, float] = field(default_factory=dict)
    confidence_modifier: float = 1.0   # 1.0 = no change, 0.5 = halved, 0.0 = blocked
    blocked: bool = False
    summary: str = ""

    @property
    def has_critical(self) -> bool:
        return any(v.severity == "critical" for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        return any(v.severity == "warning" for v in self.violations)

    def to_llm_context(self) -> str:
        """Generate context string to inject into LLM prompts."""
        if not self.violations and not self.corrections:
            return ""

        lines = ["## Physics Validation Results\n"]

        if self.corrections:
            lines.append("**CORRECTED VALUES (use these instead of solver output):**")
            for name, val in self.corrections.items():
                lines.append(f"  - {name} = {val:.6g} (recomputed from first principles)")
            lines.append("")

        for v in self.violations:
            icon = "CRITICAL" if v.severity == "critical" else "WARNING" if v.severity == "warning" else "NOTE"
            lines.append(f"**{icon}: {v.observable}**")
            lines.append(f"  Solver reported: {v.reported_value}")
            if v.expected_range:
                lines.append(f"  Expected range: [{v.expected_range[0]:.4g}, {v.expected_range[1]:.4g}]")
            if v.corrected_value is not None:
                lines.append(f"  Corrected value: {v.corrected_value:.6g}")
            lines.append(f"  {v.explanation}")
            lines.append("")

        if self.blocked:
            lines.append("**ANSWER BLOCKED: Critical physics violations prevent a reliable conclusion. "
                         "Report the violations and do NOT assign high confidence.**\n")
        elif self.has_critical:
            lines.append("**CONFIDENCE DOWNGRADED: Use corrected values above. Do NOT report solver values "
                         "for corrected quantities. State that values were recomputed from first principles.**\n")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Module-Specific Validators
# ---------------------------------------------------------------------------
def validate_drift_diffusion(
    results: List[Dict[str, Any]],
    config: Optional[Dict] = None,
) -> TruthReport:
    """Validate drift-diffusion (PN junction) results against physics."""
    report = TruthReport()

    # Extract doping and temperature from config
    N_d = None
    N_a = None
    temperature = 300.0
    if config:
        doping = config.get("doping", {})
        N_d = doping.get("N_d_level")
        N_a = doping.get("N_a_level")
        solver = config.get("solver", {})
        temperature = solver.get("temperature", 300.0)

    # ---- Recompute built-in potential from first principles ----
    if N_d is not None and N_a is not None and N_d > 0 and N_a > 0:
        kT = K_BOLTZMANN_EV * temperature  # eV
        ni = NI_SI_300K  # intrinsic concentration
        # Vbi = (kT/q) * ln(N_d * N_a / ni^2) — but kT is already in eV so Vbi is in V
        Vbi_computed = kT * math.log(N_d * N_a / (ni ** 2))

        report.corrections["built_in_potential"] = round(Vbi_computed, 4)

        # Check solver's reported built-in potential
        for r in results:
            obs = r.get("observables", {})
            if isinstance(obs, dict):
                vbi_obs = obs.get("built_in_potential", {})
                if isinstance(vbi_obs, dict):
                    solver_vbi = vbi_obs.get("value")
                elif isinstance(obs, list):
                    for o in obs:
                        if o.get("name") == "built_in_potential":
                            solver_vbi = o.get("value")
                            break
                else:
                    solver_vbi = None

                if solver_vbi is not None and isinstance(solver_vbi, (int, float)):
                    # Check if solver value matches physics
                    if abs(solver_vbi - Vbi_computed) > 0.1:
                        report.violations.append(PhysicsViolation(
                            severity="critical",
                            observable="built_in_potential",
                            reported_value=solver_vbi,
                            expected_range=(0.55, 0.85),
                            corrected_value=Vbi_computed,
                            explanation=(
                                f"Solver reported Vbi={solver_vbi:.4g} V but first-principles calculation "
                                f"gives Vbi = (kT/q) * ln(Nd*Na/ni^2) = {Vbi_computed:.4f} V. "
                                f"The solver value is a numerical artifact. USE THE CORRECTED VALUE."
                            ),
                            recomputed=True,
                        ))

        # ---- Recompute depletion width at zero bias ----
        eps = EPS_SI * EPS_0  # F/cm
        Vbi_for_W = Vbi_computed
        # W = sqrt(2 * eps * Vbi * (1/Na + 1/Nd) / q)
        # q in CGS-Gaussian... actually let's use SI-CGS consistent:
        # W = sqrt(2 * eps_Si * eps_0 * Vbi / q * (Na + Nd) / (Na * Nd))
        W_computed = math.sqrt(2 * eps * Vbi_for_W / Q_ELECTRON * (N_a + N_d) / (N_a * N_d))
        report.corrections["depletion_width_zero_bias"] = W_computed

        # ---- Peak electric field at zero bias ----
        E_max = Q_ELECTRON * N_d * N_a / (eps * (N_d + N_a)) * W_computed
        report.corrections["peak_electric_field_zero_bias"] = E_max

    # ---- Validate sweep data for physics consistency ----
    for r in results:
        if r.get("_tool_name") != "sweep_parameter":
            continue
        sweep_results = r.get("results", [])
        param_path = r.get("param_path", "")

        if "applied_voltage" in param_path:
            # I-V curve validation: current should increase with forward voltage
            voltages = []
            currents = []
            for sr in sweep_results:
                if not sr.get("success"):
                    continue
                v = sr.get("param_value")
                obs = sr.get("observables", {})
                j = None
                for oname in ("terminal_current_density", "current_density"):
                    od = obs.get(oname, {})
                    if isinstance(od, dict) and od.get("value") is not None:
                        j = od["value"]
                        break
                if v is not None and j is not None:
                    voltages.append(float(v))
                    currents.append(abs(float(j)))

            if len(voltages) >= 3 and len(currents) >= 3:
                # Check for exponential behavior (expected for forward bias)
                # Current should increase by orders of magnitude
                j_min = min(c for c in currents if c > 0) if any(c > 0 for c in currents) else 1e-30
                j_max = max(currents)
                if j_max > 0 and j_min > 0:
                    decades = math.log10(j_max / j_min) if j_min > 0 else 0
                    if decades < 2 and max(voltages) >= 0.5:
                        report.violations.append(PhysicsViolation(
                            severity="warning",
                            observable="I-V_curve_dynamic_range",
                            reported_value=f"{decades:.1f} decades",
                            expected_range=(3, 15),
                            explanation=(
                                f"Forward bias I-V curve spans only {decades:.1f} decades. "
                                f"A typical Si diode at 300K should show 5-10+ decades between 0V and 0.7V. "
                                f"This may indicate insufficient sweep resolution or numerical issues."
                            ),
                        ))

                # Detect I-V knee voltage
                if len(voltages) >= 3 and j_max > 1.0:  # at least 1 A/cm^2 max
                    # Find where current first exceeds 1% of max
                    threshold_j = j_max * 0.01
                    for i, (v, j) in enumerate(zip(voltages, currents)):
                        if j > threshold_j:
                            report.corrections["knee_voltage_estimated"] = round(v, 3)
                            break

    # ---- Confidence modifier ----
    n_critical = sum(1 for v in report.violations if v.severity == "critical")
    n_warning = sum(1 for v in report.violations if v.severity == "warning")
    if n_critical >= 2:
        report.confidence_modifier = 0.3
        report.blocked = True
    elif n_critical == 1:
        report.confidence_modifier = 0.6
    elif n_warning >= 2:
        report.confidence_modifier = 0.7
    elif n_warning == 1:
        report.confidence_modifier = 0.85

    # Build summary
    if report.corrections:
        parts = [f"{k}={v:.4g}" for k, v in report.corrections.items()]
        report.summary = f"Recomputed from first principles: {', '.join(parts)}"

    return report


def validate_navier_stokes(
    results: List[Dict[str, Any]],
    config: Optional[Dict] = None,
) -> TruthReport:
    """Validate Navier-Stokes results against physics."""
    report = TruthReport()

    if not config:
        return report

    solver = config.get("solver", {})
    nu = solver.get("nu")
    U_lid = solver.get("U_lid")
    Lx = solver.get("Lx", 1.0)

    if nu and U_lid and nu > 0:
        Re = U_lid * Lx / nu
        report.corrections["reynolds_number"] = round(Re, 1)

        # Check if Re > 2000 (turbulent) but using laminar solver
        if Re > 2000:
            report.violations.append(PhysicsViolation(
                severity="warning",
                observable="reynolds_number",
                reported_value=Re,
                expected_range=(0, 2000),
                explanation=(
                    f"Re={Re:.0f} exceeds the laminar-turbulent transition (~2000). "
                    f"The laminar solver may not accurately capture turbulent flow features. "
                    f"Results should be interpreted with caution."
                ),
            ))

    # Check for negative viscosity in sweep
    for r in results:
        if r.get("_tool_name") == "sweep_parameter" and "nu" in r.get("param_path", ""):
            for sr in r.get("results", []):
                pv = sr.get("param_value")
                if isinstance(pv, (int, float)) and pv <= 0:
                    report.violations.append(PhysicsViolation(
                        severity="critical",
                        observable="kinematic_viscosity",
                        reported_value=pv,
                        expected_range=(1e-7, 10),
                        explanation="Kinematic viscosity must be positive. Negative viscosity is unphysical.",
                    ))
                    report.confidence_modifier = min(report.confidence_modifier, 0.3)
                    break

    return report


# ---------------------------------------------------------------------------
#  Main Validation Entry Point
# ---------------------------------------------------------------------------
VALIDATORS = {
    "drift_diffusion": validate_drift_diffusion,
    "navier_stokes": validate_navier_stokes,
}


def validate_results(
    results: List[Dict[str, Any]],
    module_name: Optional[str] = None,
    config: Optional[Dict] = None,
) -> TruthReport:
    """Run physics validation on analysis results.

    Args:
        results: List of tool execution results
        module_name: Physics module used (auto-detected if not provided)
        config: Solver configuration

    Returns:
        TruthReport with violations, corrections, and confidence modifier
    """
    # Auto-detect module from results
    if not module_name:
        for r in results:
            mn = r.get("module_name")
            if mn:
                module_name = mn
                break

    if not module_name:
        return TruthReport()

    # Auto-detect config from results
    if not config:
        for r in results:
            if r.get("_tool_name") in ("run_module", "sweep_parameter", "optimize_parameter"):
                # Try to find config in the args (it's not always stored in results)
                pass

    validator = VALIDATORS.get(module_name)
    if not validator:
        return TruthReport()

    report = validator(results, config)

    if report.violations:
        logger.info("Physics Truth Layer: %d violations (%d critical) for %s",
                     len(report.violations),
                     sum(1 for v in report.violations if v.severity == "critical"),
                     module_name)

    return report
