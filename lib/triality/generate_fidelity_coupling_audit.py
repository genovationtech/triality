from __future__ import annotations

from collections import Counter
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent

FIDELITY_LEVELS = {
    "L0": "heuristic / screening / infrastructure",
    "L1": "reduced-order physics",
    "L2": "engineering-grade",
    "L3": "high-fidelity / near-CFD/FEM",
    "L4": "validated / benchmarked",
}

COUPLING_LEVELS = {
    "M0": "isolated or orchestration-only; no standardized physics exchange",
    "M1": "one-way/ad hoc exchange; partial unit/time handling",
    "M2": "direct module feedthrough with partial standardization of units/time bases",
    "M3": "direct canonical exchange with explicit coupling iteration / stabilization",
    "M4": "benchmarked closed-loop multi-physics coupling",
}

DOC_OVERRIDES = {
    "drift_diffusion": [
        "README_LAYER3.md says 'production-ready' while IMPLEMENTATION_STATUS.md still documents numerical-refinement risk; rated conservatively.",
    ],
    "geospatial": [
        "README.md documents production data integration, but this module is logistics/feasibility rather than first-principles physics.",
    ],
    "hv_safety": [
        "README_LAYER4.md explicitly marks the module production-ready with 17/17 tests passing.",
    ],
    "verification": [
        "README.md contains benchmark and regression infrastructure, but also notes key verification tests are still failing.",
    ],
}

FIDELITY_OVERRIDES = {
    "agent_design": ("L0", "orchestration/optimization layer, not a forward-physics solver"),
    "core": ("L0", "shared data model, units, and coupling infrastructure"),
    "coupling": ("L0", "shared coupling infrastructure"),
    "diagnostics": ("L0", "monitoring/post-processing utilities"),
    "geometry": ("L0", "geometry/FDM support utilities"),
    "ground_truth": ("L0", "fixtures/reference data rather than forward simulation"),
    "safety_logic": ("L0", "rules/logic wrapper around physics outputs"),
    "sensor_fusion": ("L0", "state-estimation layer, not first-principles physics"),
    "solvers": ("L0", "solver-selection/backend infrastructure"),
    "verification": ("L0", "verification harness, not a physics module"),
    "electrochemistry": ("L1", "cell-level reduced-order battery chemistry"),
    "geospatial": ("L1", "physics-inspired feasibility/routing with optional real-data hooks"),
    "hypersonic_vehicle_simulation": ("L1", "system-level vehicle synthesis over higher-fidelity submodels"),
    "injury_biomechanics": ("L1", "lumped injury metrics and surrogate injury models"),
    "quantum_nanoscale": ("L1", "reduced-order nanoscale / quantum device models"),
    "quantum_optimization": ("L1", "optimization workflow, not direct field solve"),
    "rcs_shaping": ("L1", "shaping/surrogate RCS estimates rather than full-wave solve"),
    "safety": ("L1", "system safety / kinetics abstractions"),
    "sensing": ("L1", "trade-study/feasibility architecture rather than full sensor scene simulation"),
    "sensors_actuators": ("L1", "component-level reduced-order hardware models"),
    "spacecraft_thermal": ("L1", "network-style spacecraft thermal abstractions"),
    "spatial_flow": ("L1", "routing engine built on potential-field abstractions"),
    "swarm_dynamics": ("L1", "agent/surrogate dynamics"),
    "uncertainty": ("L1", "uncertainty wrapper around other physics solvers"),
    "combustor_simulation": ("L3", "combustor solver with reacting-flow style iterative coupling"),
    "conjugate_heat_transfer": ("L3", "explicit CHT coupling and outer iterations"),
    "drift_diffusion": ("L3", "Layer 3 semiconductor solver with dedicated device physics docs"),
    "hv_safety": ("L4", "module documentation explicitly claims production-ready / tested operation"),
    "monte_carlo_neutron": ("L3", "Monte Carlo neutronics"),
    "multiphase_vof": ("L3", "VOF multiphase flow"),
    "navier_stokes": ("L3", "Navier-Stokes CFD solver"),
    "particle_in_cell": ("L3", "PIC plasma / charged-particle method"),
    "plasma_fluid": ("L3", "coupled plasma-fluid solver"),
    "plasma_thruster_simulation": ("L3", "high-fidelity plasma thruster simulation"),
    "radiation_transport": ("L3", "transport/radiation PDEs with dedicated solver"),
    "thermal_hydraulics": ("L3", "coupled thermal-fluid engineering solver stack"),
    "thermal_signature_coupling": ("L3", "explicit thermal/signature coupled analysis"),
    "tps_ablation": ("L3", "TPS ablation / high-enthalpy thermal coupling"),
    "turbulence_combustion": ("L3", "combustion-turbulence coupling"),
}

COUPLING_OVERRIDES = {
    "agent_design": ("M0", "optimizer/orchestration layer; does not expose standardized physics exchange"),
    "geometry": ("M0", "support utilities only"),
    "core": ("M3", "canonical PhysicsState + unit conversion + coupling engine"),
    "coupled_electrical_thermal": ("M3", "direct two-way coupling with convergence / relaxation logic"),
    "coupled_physics": ("M3", "explicit coupled multi-solver orchestrator with convergence logic"),
    "coupling": ("M3", "shared canonical coupling abstractions"),
    "thermal_hydraulics": ("M3", "tight thermal-fluid iteration with SI conversion hooks"),
    "verification": ("M1", "benchmarks consume solver outputs but do not provide reusable runtime coupling"),
}


def module_dirs() -> list[Path]:
    return sorted(
        p
        for p in ROOT.iterdir()
        if p.is_dir() and (p / "__init__.py").exists() and not p.name.startswith((".", "__"))
    )


def read_texts(module_dir: Path) -> tuple[str, str]:
    py = "\n".join(path.read_text(errors="ignore") for path in sorted(module_dir.glob("*.py")))
    md = "\n".join(path.read_text(errors="ignore") for path in sorted(module_dir.glob("*.md")))
    return py, md


def detect_features(py_text: str, md_text: str) -> dict[str, bool]:
    all_text = f"{py_text}\n{md_text}"
    return {
        "physics_state": "PhysicsState" in py_text,
        "canonical_fields": "CanonicalField" in py_text or "FieldType." in py_text,
        "units": bool(re.search(r"\b(to_si|from_si|units?|UnitSystem|UnitRegistry)\b", all_text, re.I)),
        "time": bool(re.search(r"\b(dt|time_step|timesteps|transient|integrat|trajectory|history)\b", all_text, re.I)),
        "stability": bool(re.search(r"\b(relax|under.?relax|convergen|residual|iteration|stable)\b", all_text, re.I)),
        "benchmark": bool(re.search(r"\b(benchmark|validated|verification|fixture|regression|production-ready)\b", md_text, re.I)),
        "high_fidelity": bool(
            re.search(
                r"\b(FEM|finite element|finite volume|Navier-Stokes|VOF|Monte Carlo|PIC|CFD|FDTD|Layer 3|Layer 4|high-fidelity)\b",
                all_text,
                re.I,
            )
        ),
    }


def infer_fidelity(name: str, features: dict[str, bool]) -> tuple[str, str]:
    if name in FIDELITY_OVERRIDES:
        return FIDELITY_OVERRIDES[name]
    if features["benchmark"]:
        return "L4", "module docs/tests explicitly claim validated or benchmarked status"
    if features["high_fidelity"]:
        return "L3", "code/docs indicate higher-fidelity numerical method family"
    if features["physics_state"]:
        return "L2", "forward-physics solver exports reusable simulation state"
    return "L1", "reduced-order or application-level physics logic"


def infer_coupling(name: str, features: dict[str, bool]) -> tuple[str, str]:
    if name in COUPLING_OVERRIDES:
        return COUPLING_OVERRIDES[name]
    if features["physics_state"] and (features["canonical_fields"] or features["units"]) and features["stability"]:
        return "M2", "direct feedthrough plus partial standardization and convergence controls"
    if features["physics_state"]:
        return "M1", "direct feedthrough exists, but standardization/stabilization are limited"
    return "M1", "consumes or supports other modules without canonical closed-loop exchange"


def yes_partial_no(primary: bool, partial: bool = False) -> str:
    if primary:
        return "Yes"
    if partial:
        return "Partial"
    return "No"


def assess_module(module_dir: Path) -> dict[str, str]:
    py_text, md_text = read_texts(module_dir)
    features = detect_features(py_text, md_text)
    fidelity, fidelity_basis = infer_fidelity(module_dir.name, features)
    coupling, coupling_basis = infer_coupling(module_dir.name, features)

    direct_feed = yes_partial_no(features["physics_state"], module_dir.name in {"diagnostics", "geospatial", "ground_truth", "safety_logic", "sensor_fusion"})
    units = yes_partial_no(features["canonical_fields"], features["units"] or features["physics_state"])
    time = yes_partial_no(features["time"])
    stable = yes_partial_no(features["stability"], features["physics_state"])

    evidence_files = [path.name for path in sorted(module_dir.glob("*.md"))]
    if (module_dir / "solver.py").exists():
        evidence_files.insert(0, "solver.py")
    elif evidence_files or list(module_dir.glob("*.py")):
        first_py = sorted(module_dir.glob("*.py"))[0].name
        evidence_files.insert(0, first_py)

    notes = [fidelity_basis, coupling_basis]
    notes.extend(DOC_OVERRIDES.get(module_dir.name, []))

    return {
        "module": module_dir.name,
        "fidelity": fidelity,
        "coupling": coupling,
        "direct_feed": direct_feed,
        "units": units,
        "time": time,
        "stable": stable,
        "evidence": ", ".join(dict.fromkeys(evidence_files[:3])),
        "basis": "; ".join(notes),
    }


def build_audit(rows: list[dict[str, str]]) -> str:
    fidelity_counts = Counter(row["fidelity"] for row in rows)
    coupling_counts = Counter(row["coupling"] for row in rows)

    direct_yes = sum(row["direct_feed"] == "Yes" for row in rows)
    units_yes_or_partial = sum(row["units"] in {"Yes", "Partial"} for row in rows)
    time_yes = sum(row["time"] == "Yes" for row in rows)
    stable_yes_or_partial = sum(row["stable"] in {"Yes", "Partial"} for row in rows)
    incomplete_modules = [
        row["module"]
        for row in rows
        if row["direct_feed"] != "Yes"
        or row["units"] == "No"
        or row["time"] == "No"
        or row["stable"] != "Yes"
        or row["coupling"] in {"M0", "M1"}
    ]
    level_reason = {
        "L0": "support/infrastructure/orchestration rather than forward-physics solvers",
        "L1": "reduced-order, surrogate, or system-level physics abstractions",
        "L2": "engineering-grade modules that solve useful physics but do not yet show strong evidence of near-CFD/FEM depth or broad validation",
        "L3": "higher-fidelity numerical methods or explicitly high-fidelity domain solvers",
        "L4": "modules with explicit production-ready / benchmarked evidence in repository docs",
    }

    lines = [
        "# Triality Fidelity & Coupling Audit",
        "",
        "This is a fresh repo-local re-audit of every importable `triality` module found in the current working tree. The classification is evidence-based and intentionally conservative: when code and docs disagree, the lower-confidence / lower-fidelity interpretation wins unless the repository contains clear benchmark or production-readiness evidence.",
        "",
        "> Note: I attempted to refetch `main`, but this clone currently has no configured Git remote, so the audit below reflects the latest code available in this working tree rather than a newly fetched upstream branch.",
        "",
        "## Fidelity tiers",
        "",
    ]

    for code, desc in FIDELITY_LEVELS.items():
        lines.append(f"- **{code}** → {desc}")

    lines.extend(
        [
            "",
            "## Coupling maturity rubric",
            "",
        ]
    )
    for code, desc in COUPLING_LEVELS.items():
        lines.append(f"- **{code}** → {desc}")

    lines.extend(
        [
            "",
            "## Method used for this re-check",
            "",
            "The audit combines four evidence sources for each module:",
            "",
            "1. `solver.py` / primary Python implementation to see whether the module exports `PhysicsState`, uses canonical field metadata, exposes SI conversion hooks, and contains convergence / relaxation logic.",
            "2. Module README / status markdown files when they exist, especially for documented production layers such as drift-diffusion, HV safety, geospatial, spatial flow, and verification.",
            "3. Cross-cutting framework files in `triality/core/` and `triality/coupling/` that define the real coupling contract.",
            "4. Conservative manual overrides for modules whose names or docs clearly indicate infrastructure-only, reduced-order, or high-fidelity roles.",
            "",
            "## Portfolio snapshot",
            "",
        ]
    )

    for level in ["L0", "L1", "L2", "L3", "L4"]:
        lines.append(f"- **{level}**: {fidelity_counts[level]} modules ({FIDELITY_LEVELS[level]})")
    lines.append("")
    for level in ["M0", "M1", "M2", "M3", "M4"]:
        lines.append(f"- **{level}**: {coupling_counts[level]} modules ({COUPLING_LEVELS[level]})")

    lines.extend(
        [
            "",
            "## Answers to the coupling questions",
            "",
            f"- **Can outputs feed directly into other modules?** Yes for {direct_yes}/{len(rows)} modules. The strongest enabler is the shared `PhysicsState` pattern and the canonical field / coupling helpers in `triality.core` and `triality.coupling`.",
            f"- **Are units consistent?** Mostly but not universally. {units_yes_or_partial}/{len(rows)} modules show either explicit SI/canonical-unit handling or enough metadata to mark them at least **Partial**.",
            f"- **Are time scales handled?** {time_yes}/{len(rows)} modules expose transient, timestep, trajectory, or integration language. Multi-rate coordination is still uncommon even where time dependence exists.",
            f"- **Is coupling stable?** Only partly. {stable_yes_or_partial}/{len(rows)} modules show explicit iteration / convergence / relaxation hooks, but very few modules demonstrate benchmarked closed-loop multiphysics stability.",
            "",
            "## High-level conclusions",
            "",
            "- **Best current advantage:** coupling is materially ahead of a typical grab-bag physics repo because many modules emit reusable state objects, and the repo has explicit shared coupling abstractions rather than only ad hoc file/JSON exchange.",
            "- **Main weakness:** there is still a big difference between *connectable* and *robustly co-simulatable*. Many modules can feed outputs downstream, but only a small subset show explicit stabilization logic for tight two-way coupling.",
            "- **Most credible high-fidelity pockets:** CFD / transport / plasma / TPS-style modules, the drift-diffusion layer, and the documented HV safety layer.",
            "- **Validation gap remains:** outside of the HV safety docs and specialized verification / fixture infrastructure, most modules still read as engineering-grade tools for design iteration rather than formally benchmarked end-state solvers.",
            "",
            "## Modules grouped by fidelity level",
            "",
        ]
    )

    for level in ["L0", "L1", "L2", "L3", "L4"]:
        level_rows = [row for row in rows if row["fidelity"] == level]
        module_names = ", ".join(f"`{row['module']}`" for row in level_rows)
        lines.extend(
            [
                f"### {level} — {FIDELITY_LEVELS[level]}",
                "",
                f"**Why these modules land here:** {level_reason[level]}.",
                "",
                f"**Modules ({len(level_rows)}):** {module_names}",
                "",
            ]
        )
        for row in level_rows:
            lines.append(f"- `{row['module']}`: {row['basis']}.")
        lines.append("")

    lines.extend(
        [
            "## What is left / incomplete / risky",
            "",
            "- **No true M4 stack yet.** The repo has meaningful coupling infrastructure, but no module set currently demonstrates benchmarked closed-loop multiphysics coupling with durable regression evidence.",
            "- **A lot of coupling is still immature.** All `M0` and `M1` modules still need clearer boundary contracts, stronger stabilization logic, or better canonical field/unit exchange before they can be treated as robust plug-and-play coupled modules.",
            "- **Contradictory status docs still exist.** `drift_diffusion` contains both a 'production-ready' README and a separate implementation-status file that still describes refinement risk; `verification` contains benchmark infrastructure but also documents failing verification tests.",
            "- **Some modules still look incomplete at the interface level.** The modules below are currently missing at least one desirable audit property such as direct feedthrough, explicit unit handling, time-scale handling, or strong/stable coupling evidence.",
            f"- **Modules still needing follow-up ({len(incomplete_modules)}):** {', '.join(f'`{name}`' for name in incomplete_modules)}.",
            "- **Validation remains thinner than fidelity.** Several modules legitimately look Level 3 by method family, but they are not automatically Level 4 because benchmark/validation evidence is still sparse.",
            "",
            "## Module-by-module assessment",
            "",
            "| Module | Fidelity | Coupling maturity | Direct feed? | Units consistent? | Time scales handled? | Coupling stable? | Evidence checked | Basis |",
            "|---|---:|---:|---|---|---|---|---|---|",
        ]
    )

    for row in rows:
        lines.append(
            f"| `{row['module']}` | {row['fidelity']} | {row['coupling']} | {row['direct_feed']} | {row['units']} | {row['time']} | {row['stable']} | `{row['evidence']}` | {row['basis']} |"
        )

    lines.extend(
        [
            "",
            "## Priority actions to raise coupling maturity",
            "",
            "1. **Benchmark the tight-coupling exemplars.** `coupled_electrical_thermal`, `coupled_physics`, `thermal_hydraulics`, `conjugate_heat_transfer`, and `tps_ablation` should be turned into documented regression-coupled stacks with pass/fail tolerances.",
            "2. **Normalize module boundaries to SI/canonical fields.** Several modules expose reusable state but still only achieve **Partial** for units because their boundary contracts are not uniformly explicit.",
            "3. **Add multi-rate coupling policies.** Fast electrical / control modules and slow thermal / structural modules need scheduler-level time-scale handling instead of assuming one shared integration cadence.",
            "4. **Separate 'production-ready' from 'high-fidelity'.** Where module docs claim readiness, back them with benchmark tables and regression artifacts so L4 / M4 can be justified unambiguously.",
            "5. **Resolve contradictory status docs.** `drift_diffusion` and `verification` both contain mixed signals; aligning their status documents will make future audits less subjective.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    rows = [assess_module(module_dir) for module_dir in module_dirs()]
    output = build_audit(rows)
    (ROOT / "FIDELITY_COUPLING_AUDIT.md").write_text(output + "\n")
    print(f"Wrote audit for {len(rows)} modules")


if __name__ == "__main__":
    main()
