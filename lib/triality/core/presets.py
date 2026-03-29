"""
Multi-Physics Coupling Presets

Factory functions that configure the CouplingEngine for standard
multi-physics scenarios.  Each preset returns a ready-to-run
CouplingEngine with solvers, links, and sensible defaults.

Available presets
-----------------
- reactor_core:          Neutronics ↔ Thermal-Hydraulics
- conjugate_heat:        Navier-Stokes ↔ Conjugate Heat Transfer
- fluid_structure:       Navier-Stokes ↔ Structural Analysis
- thermo_mechanical:     Thermal (any) ↔ Structural Analysis
- aero_thermal:          Aero Loads ↔ TPS Ablation
- battery_pack:          Electrochemistry ↔ Battery Thermal
- combustion_radiation:  Reacting Flows ↔ Radiation Transport
- plasma_em:             Plasma Fluid ↔ Electrostatics/EM

Usage
-----
>>> from triality.core.presets import reactor_core
>>> engine = reactor_core(neutronics_solver, th_solver)
>>> result = engine.run(t_end=10.0)
"""

from __future__ import annotations

from typing import Any, Optional

from triality.core.coupling import (
    CouplingEngine,
    CouplingLink,
    CouplingStrategy,
    RelaxationMethod,
)
from triality.core.adapters import GenericAdapter, SteadyAdapter


def reactor_core(
    neutronics_solver: Any,
    th_solver: Any,
    *,
    strategy: CouplingStrategy = CouplingStrategy.GAUSS_SEIDEL,
    dt_sync: float = 0.01,
    max_coupling_iter: int = 30,
    convergence_tol: float = 1e-4,
    relaxation_omega: float = 0.5,
    neutronics_max_dt: float = 0.01,
    th_max_dt: float = 0.01,
) -> CouplingEngine:
    """Configure Neutronics ↔ Thermal-Hydraulics coupling.

    Coupling loop:
        Neutronics → power_density → TH
        TH → temperature → Neutronics (Doppler + moderator feedback)

    Parameters
    ----------
    neutronics_solver : NeutronicsSolver or Neutronics2DSolver
        Must have import_state / advance.
    th_solver : ThermalHydraulicsSolver
        Must have import_state / advance.
    """
    engine = CouplingEngine(
        strategy=strategy,
        relaxation_omega=relaxation_omega,
        relaxation_method=RelaxationMethod.AITKEN,
        convergence_tol=convergence_tol,
        max_coupling_iter=max_coupling_iter,
        dt_sync=dt_sync,
    )

    engine.add_solver(GenericAdapter(
        "neutronics", neutronics_solver,
        max_dt=neutronics_max_dt, priority=0,
    ))
    engine.add_solver(GenericAdapter(
        "thermal_hydraulics", th_solver,
        max_dt=th_max_dt, priority=1,
    ))

    # Neutronics → TH: power density drives heating
    engine.add_link(CouplingLink(
        source_solver="neutronics",
        target_solver="thermal_hydraulics",
        source_field="power_density",
    ))
    # TH → Neutronics: temperature drives reactivity feedback
    engine.add_link(CouplingLink(
        source_solver="thermal_hydraulics",
        target_solver="neutronics",
        source_field="temperature",
    ))

    return engine


def conjugate_heat(
    cfd_solver: Any,
    cht_solver: Any,
    *,
    strategy: CouplingStrategy = CouplingStrategy.GAUSS_SEIDEL,
    dt_sync: float = 0.001,
    max_coupling_iter: int = 20,
    convergence_tol: float = 1e-4,
    cfd_max_dt: float = 0.001,
    cht_max_dt: float = 0.01,
) -> CouplingEngine:
    """Configure Navier-Stokes ↔ Conjugate Heat Transfer coupling.

    Coupling loop:
        CFD → velocity, pressure → CHT
        CHT → wall_temperature, heat_flux → CFD
    """
    engine = CouplingEngine(
        strategy=strategy,
        relaxation_omega=0.7,
        relaxation_method=RelaxationMethod.AITKEN,
        convergence_tol=convergence_tol,
        max_coupling_iter=max_coupling_iter,
        dt_sync=dt_sync,
    )

    engine.add_solver(GenericAdapter(
        "cfd", cfd_solver, max_dt=cfd_max_dt, priority=0,
    ))
    engine.add_solver(GenericAdapter(
        "cht", cht_solver, max_dt=cht_max_dt, priority=1,
    ))

    engine.add_link(CouplingLink("cfd", "cht", "velocity_x"))
    engine.add_link(CouplingLink("cfd", "cht", "velocity_y"))
    engine.add_link(CouplingLink("cfd", "cht", "pressure"))
    engine.add_link(CouplingLink("cht", "cfd", "wall_temperature"))
    engine.add_link(CouplingLink("cht", "cfd", "heat_flux"))

    return engine


def fluid_structure(
    cfd_solver: Any,
    structural_solver: Any,
    *,
    strategy: CouplingStrategy = CouplingStrategy.GAUSS_SEIDEL,
    dt_sync: float = 0.001,
    max_coupling_iter: int = 30,
    convergence_tol: float = 1e-5,
    cfd_max_dt: float = 0.001,
    structural_max_dt: float = 0.01,
) -> CouplingEngine:
    """Configure Navier-Stokes ↔ Structural (FSI) coupling.

    Coupling loop:
        CFD → pressure, wall_shear_stress → Structural
        Structural → displacement → CFD (mesh deformation)
    """
    engine = CouplingEngine(
        strategy=strategy,
        relaxation_omega=0.5,
        relaxation_method=RelaxationMethod.AITKEN,
        convergence_tol=convergence_tol,
        max_coupling_iter=max_coupling_iter,
        dt_sync=dt_sync,
    )

    engine.add_solver(GenericAdapter(
        "cfd", cfd_solver, max_dt=cfd_max_dt, priority=0,
    ))
    engine.add_solver(GenericAdapter(
        "structural", structural_solver,
        max_dt=structural_max_dt, priority=1,
    ))

    engine.add_link(CouplingLink("cfd", "structural", "pressure"))
    engine.add_link(CouplingLink("cfd", "structural", "wall_shear_stress"))
    engine.add_link(CouplingLink("structural", "cfd", "displacement"))

    return engine


def thermo_mechanical(
    thermal_solver: Any,
    structural_solver: Any,
    *,
    strategy: CouplingStrategy = CouplingStrategy.GAUSS_SEIDEL,
    dt_sync: float = 0.1,
    max_coupling_iter: int = 20,
    convergence_tol: float = 1e-4,
    thermal_max_dt: float = 0.1,
    structural_max_dt: float = 1.0,
) -> CouplingEngine:
    """Configure Thermal ↔ Structural (thermo-mechanical) coupling.

    Coupling loop:
        Thermal → temperature → Structural (thermal strain)
        Structural → displacement, stress_von_mises → Thermal (deformation heating)
    """
    engine = CouplingEngine(
        strategy=strategy,
        relaxation_omega=0.7,
        relaxation_method=RelaxationMethod.CONSTANT,
        convergence_tol=convergence_tol,
        max_coupling_iter=max_coupling_iter,
        dt_sync=dt_sync,
    )

    engine.add_solver(GenericAdapter(
        "thermal", thermal_solver, max_dt=thermal_max_dt, priority=0,
    ))
    engine.add_solver(GenericAdapter(
        "structural", structural_solver,
        max_dt=structural_max_dt, priority=1,
    ))

    engine.add_link(CouplingLink("thermal", "structural", "temperature"))
    engine.add_link(CouplingLink(
        "structural", "thermal", "stress_von_mises",
        target_field="heat_source",
    ))

    return engine


def aero_thermal(
    aero_solver: Any,
    ablation_solver: Any,
    *,
    strategy: CouplingStrategy = CouplingStrategy.GAUSS_SEIDEL,
    dt_sync: float = 0.01,
    max_coupling_iter: int = 15,
    convergence_tol: float = 1e-3,
    aero_max_dt: float = 1.0,
    ablation_max_dt: float = 0.01,
    sub_cycle_ablation: bool = True,
) -> CouplingEngine:
    """Configure Aero Loads ↔ TPS Ablation coupling.

    Coupling loop:
        Aero → heat_flux, pressure → Ablation
        Ablation → wall_temperature, remaining_thickness → Aero (shape change)
    """
    engine = CouplingEngine(
        strategy=strategy,
        relaxation_omega=0.6,
        relaxation_method=RelaxationMethod.AITKEN,
        convergence_tol=convergence_tol,
        max_coupling_iter=max_coupling_iter,
        dt_sync=dt_sync,
    )

    engine.add_solver(GenericAdapter(
        "aero", aero_solver, max_dt=aero_max_dt, priority=0,
    ))
    engine.add_solver(
        GenericAdapter(
            "ablation", ablation_solver,
            max_dt=ablation_max_dt, priority=1,
        ),
        sub_cycle=sub_cycle_ablation,
    )

    engine.add_link(CouplingLink("aero", "ablation", "heat_flux"))
    engine.add_link(CouplingLink("aero", "ablation", "pressure"))
    engine.add_link(CouplingLink("ablation", "aero", "wall_temperature"))

    return engine


def battery_pack(
    echem_solver: Any,
    thermal_solver: Any,
    *,
    strategy: CouplingStrategy = CouplingStrategy.GAUSS_SEIDEL,
    dt_sync: float = 1.0,
    max_coupling_iter: int = 15,
    convergence_tol: float = 1e-3,
    echem_max_dt: float = 1.0,
    thermal_max_dt: float = 1.0,
) -> CouplingEngine:
    """Configure Electrochemistry ↔ Battery Thermal coupling.

    Coupling loop:
        Electrochemistry → heat_source (Joule + entropic) → Thermal
        Thermal → temperature → Electrochemistry (Arrhenius kinetics)
    """
    engine = CouplingEngine(
        strategy=strategy,
        relaxation_omega=0.8,
        relaxation_method=RelaxationMethod.CONSTANT,
        convergence_tol=convergence_tol,
        max_coupling_iter=max_coupling_iter,
        dt_sync=dt_sync,
    )

    engine.add_solver(GenericAdapter(
        "electrochemistry", echem_solver,
        max_dt=echem_max_dt, priority=0,
    ))
    engine.add_solver(GenericAdapter(
        "battery_thermal", thermal_solver,
        max_dt=thermal_max_dt, priority=1,
    ))

    engine.add_link(CouplingLink("electrochemistry", "battery_thermal", "heat_source"))
    engine.add_link(CouplingLink("battery_thermal", "electrochemistry", "temperature"))

    return engine


def combustion_radiation(
    reacting_solver: Any,
    radiation_solver: Any,
    *,
    strategy: CouplingStrategy = CouplingStrategy.GAUSS_SEIDEL,
    dt_sync: float = 1e-4,
    max_coupling_iter: int = 10,
    convergence_tol: float = 1e-3,
    reacting_max_dt: float = 1e-4,
    radiation_max_dt: float = 1.0,
) -> CouplingEngine:
    """Configure Reacting Flows ↔ Radiation Transport coupling.

    Coupling loop:
        Reacting Flows → temperature, absorption_coefficient → Radiation
        Radiation → heat_source (div q_rad) → Reacting Flows
    """
    engine = CouplingEngine(
        strategy=strategy,
        relaxation_omega=0.5,
        relaxation_method=RelaxationMethod.AITKEN,
        convergence_tol=convergence_tol,
        max_coupling_iter=max_coupling_iter,
        dt_sync=dt_sync,
    )

    engine.add_solver(GenericAdapter(
        "reacting_flows", reacting_solver,
        max_dt=reacting_max_dt, priority=0,
    ))
    engine.add_solver(GenericAdapter(
        "radiation", radiation_solver,
        max_dt=radiation_max_dt, priority=1,
    ))

    engine.add_link(CouplingLink("reacting_flows", "radiation", "temperature"))
    engine.add_link(CouplingLink(
        "reacting_flows", "radiation",
        "species_mass_fraction", target_field="absorption_coefficient",
    ))
    engine.add_link(CouplingLink("radiation", "reacting_flows", "heat_source"))

    return engine


def plasma_em(
    plasma_solver: Any,
    em_solver: Any,
    *,
    strategy: CouplingStrategy = CouplingStrategy.GAUSS_SEIDEL,
    dt_sync: float = 1e-8,
    max_coupling_iter: int = 20,
    convergence_tol: float = 1e-4,
    plasma_max_dt: float = 1e-8,
    em_max_dt: float = 1.0,
    sub_cycle_plasma: bool = True,
) -> CouplingEngine:
    """Configure Plasma Fluid ↔ EM Solver coupling.

    Coupling loop:
        Plasma → charge_density, current_density → EM
        EM → electric_field, electric_potential → Plasma
    """
    engine = CouplingEngine(
        strategy=strategy,
        relaxation_omega=0.5,
        relaxation_method=RelaxationMethod.AITKEN,
        convergence_tol=convergence_tol,
        max_coupling_iter=max_coupling_iter,
        dt_sync=dt_sync,
    )

    engine.add_solver(
        GenericAdapter(
            "plasma", plasma_solver,
            max_dt=plasma_max_dt, priority=0,
        ),
        sub_cycle=sub_cycle_plasma,
    )
    engine.add_solver(GenericAdapter(
        "em", em_solver, max_dt=em_max_dt, priority=1,
    ))

    engine.add_link(CouplingLink("plasma", "em", "charge_density"))
    engine.add_link(CouplingLink("plasma", "em", "current_density"))
    engine.add_link(CouplingLink("em", "plasma", "electric_field"))
    engine.add_link(CouplingLink("em", "plasma", "electric_potential"))

    return engine
