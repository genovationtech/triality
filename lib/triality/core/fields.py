"""
Physics Field Standardization Protocol

Defines a canonical vocabulary for physical fields that all TRIALITY solvers
must use at their external interfaces. Internal naming can differ, but
the coupling boundary exposes standardized names.

Canonical field vocabulary:
    temperature         [K]         Bulk / node temperature
    pressure            [Pa]        Static pressure
    density             [kg/m^3]    Mass density
    velocity            [m/s]       Velocity vector or component
    velocity_x          [m/s]       x-component of velocity
    velocity_y          [m/s]       y-component of velocity
    velocity_z          [m/s]       z-component of velocity
    energy              [J]         Total energy
    specific_energy     [J/kg]      Specific energy
    enthalpy            [J/kg]      Specific enthalpy
    species_mass_fraction [-]       Mass fraction of chemical species
    displacement        [m]         Structural displacement
    displacement_x      [m]         x-displacement
    displacement_y      [m]         y-displacement
    displacement_z      [m]         z-displacement
    stress              [Pa]        Stress tensor component
    stress_von_mises    [Pa]        von Mises equivalent stress
    strain              [-]         Strain tensor component
    heat_flux           [W/m^2]     Surface or volume heat flux
    heat_flux_x         [W/m^2]     x-component of heat flux
    heat_flux_y         [W/m^2]     y-component of heat flux
    heat_source         [W/m^3]     Volumetric heat source
    linear_heat_rate    [W/m]       Linear heat rate (SI)
    thermal_conductivity [W/(m*K)] Thermal conductivity
    electron_temperature [K]        Electron temperature (plasma)
    ion_temperature     [K]         Ion temperature (plasma)
    number_density      [1/m^3]     Number density
    electric_potential  [V]         Electric potential
    electric_field      [V/m]       Electric field magnitude
    electric_field_x    [V/m]       x-component electric field
    electric_field_y    [V/m]       y-component electric field
    current_density_x   [A/m^2]     x-component current density
    current_density_y   [A/m^2]     y-component current density
    acceleration        [m/s^2]     Acceleration magnitude or component
    detection_probability [1]       Probability of detection
    neutron_flux        [n/(m^2*s)] Scalar neutron flux (SI)
    power_density       [W/m^3]     Volumetric power density (SI)
    reaction_rate       [1/s]       Chemical or nuclear reaction rate
    turbulent_kinetic_energy [m^2/s^2] TKE
    turbulent_dissipation [m^2/s^3] Turbulent dissipation rate
    turbulent_viscosity [Pa*s]      Eddy viscosity
    void_fraction       [-]         Void fraction (two-phase)
    quality             [-]         Thermodynamic quality
    mach_number         [-]         Mach number
    wall_shear_stress   [Pa]        Wall shear stress
    nusselt_number      [-]         Nusselt number
    lift_coefficient    [-]         CL
    drag_coefficient    [-]         CD
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, Union, Callable, Protocol, runtime_checkable
from enum import Enum, IntEnum

from triality.core.units import UnitSpec, UnitMetadata, SI_UNITS, convert


# ---------------------------------------------------------------------------
# Fidelity tier classification
# ---------------------------------------------------------------------------

class FidelityTier(IntEnum):
    """Fidelity tier for physics solver classification.

    Levels
    ------
    HEURISTIC (0)
        Screening-level: algebraic correlations, look-up tables, rule-based
        logic, no PDE solving. Fast but low physical fidelity.
    REDUCED_ORDER (1)
        Reduced-order physics: 0-D lumped models, 1-D network/beam models,
        simple ODE systems, closed-form engineering formulas.
    ENGINEERING (2)
        Engineering-grade: 1-D or 2-D PDE discretization with basic
        numerical schemes (explicit Euler, SOR, Thomas tridiagonal). Includes
        2-D algebraic or simplified physics on spatial grids.
    HIGH_FIDELITY (3)
        High-fidelity / near-CFD/FEM: proper 2-D or 3-D PDE solving with
        higher-order schemes (MUSCL, split-step, Newmark-beta, implicit
        Euler/CN), validated numerical schemes, physically consistent BCs.
    VALIDATED (4)
        Validated / benchmarked: includes embedded convergence tests,
        comparison to analytical solutions or experimental data, formal error
        estimation and grid-independence studies.
    """
    HEURISTIC     = 0
    REDUCED_ORDER = 1
    ENGINEERING   = 2
    HIGH_FIDELITY = 3
    VALIDATED     = 4

    def label(self) -> str:
        labels = {0: "L0-Heuristic", 1: "L1-ReducedOrder",
                  2: "L2-Engineering", 3: "L3-HighFidelity", 4: "L4-Validated"}
        return labels[self.value]


class CouplingMaturity(IntEnum):
    """Coupling maturity level for multi-physics integration.

    Levels
    ------
    M0_STANDALONE (0)
        No coupling interface. Solver runs in isolation.
    M1_CONNECTABLE (1)
        Has export_state() — can expose fields via PhysicsState, but cannot
        receive fields from other solvers.
    M2_INTEROPERABLE (2)
        Has export_state() + import_state() — can both send and receive
        canonical fields. Supports one-way and loose coupling.
    M3_COUPLED (3)
        Has export_state() + import_state() + advance(dt) — supports
        closed-loop partitioned iteration via the CouplingEngine.
        Time-scale coordination, sub-cycling, and convergence monitoring
        are available at this level.
    """
    M0_STANDALONE    = 0
    M1_CONNECTABLE   = 1
    M2_INTEROPERABLE = 2
    M3_COUPLED       = 3

    def label(self) -> str:
        labels = {
            0: "M0-Standalone",
            1: "M1-Connectable",
            2: "M2-Interoperable",
            3: "M3-Coupled",
        }
        return labels[self.value]


# ---------------------------------------------------------------------------
# Canonical field definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldDefinition:
    """Definition of a canonical physical field.

    Attributes
    ----------
    canonical_name : str
        The standard name used across all couplings.
    si_unit : str
        SI unit key from the unit registry.
    description : str
        Human-readable description.
    is_intensive : bool
        True for intensive properties (T, P, rho), False for extensive (F, E).
    """
    canonical_name: str
    si_unit: str
    description: str
    is_intensive: bool = True


# Master vocabulary
CANONICAL_FIELDS: Dict[str, FieldDefinition] = {}

def _register(name: str, unit: str, desc: str, intensive: bool = True):
    CANONICAL_FIELDS[name] = FieldDefinition(name, unit, desc, intensive)

# Thermal
_register("temperature", "K", "Bulk temperature")
_register("electron_temperature", "K", "Electron temperature (plasma)")
_register("ion_temperature", "K", "Ion temperature (plasma)")
_register("wall_temperature", "K", "Wall/surface temperature")

# Fluid mechanics
_register("pressure", "Pa", "Static pressure")
_register("density", "kg/m^3", "Mass density")
_register("velocity", "m/s", "Velocity magnitude")
_register("velocity_x", "m/s", "x-velocity component")
_register("velocity_y", "m/s", "y-velocity component")
_register("velocity_z", "m/s", "z-velocity component")
_register("acceleration", "m/s^2", "Acceleration magnitude")
_register("mach_number", "1", "Mach number")
_register("mass_flux", "kg/(m^2*s)", "Mass flux")
_register("number_density", "1/m^3", "Number density")

# Electromagnetics / sensing
_register("electric_potential", "V", "Electric potential")
_register("electric_field", "V/m", "Electric field magnitude")
_register("electric_field_x", "V/m", "x-component electric field")
_register("electric_field_y", "V/m", "y-component electric field")
_register("current_density_x", "A/m^2", "x-component current density")
_register("current_density_y", "A/m^2", "y-component current density")
_register("detection_probability", "1", "Probability of detection")

# Energy
_register("energy", "J", "Total energy", intensive=False)
_register("specific_energy", "J/(kg*K)", "Specific energy")
_register("enthalpy", "J/(kg*K)", "Specific enthalpy")

# Heat transfer
_register("heat_flux", "W/m^2", "Heat flux")
_register("heat_flux_x", "W/m^2", "x-component of heat flux")
_register("heat_flux_y", "W/m^2", "y-component of heat flux")
_register("heat_source", "W/m^3", "Volumetric heat source")
_register("linear_heat_rate", "W/m", "Linear heat rate (SI)")
_register("thermal_conductivity", "W/(m*K)", "Thermal conductivity")
_register("nusselt_number", "1", "Nusselt number")

# Structural
_register("displacement", "m", "Displacement magnitude")
_register("displacement_x", "m", "x-displacement")
_register("displacement_y", "m", "y-displacement")
_register("displacement_z", "m", "z-displacement")
_register("stress", "Pa", "Stress tensor component")
_register("stress_von_mises", "Pa", "von Mises equivalent stress")
_register("strain", "1", "Strain tensor component")

# Turbulence
_register("turbulent_kinetic_energy", "m^2/s^2", "Turbulent kinetic energy")
_register("turbulent_dissipation", "m^2/s^3", "Turbulent dissipation rate")
_register("turbulent_viscosity", "Pa*s", "Eddy viscosity")
_register("wall_shear_stress", "Pa", "Wall shear stress")

# Species / chemistry
_register("species_mass_fraction", "1", "Species mass fraction")
_register("reaction_rate", "Hz", "Reaction rate")

# Two-phase
_register("void_fraction", "1", "Void fraction")
_register("quality", "1", "Thermodynamic quality")

# Nuclear
_register("neutron_flux", "n/(m^2*s)", "Scalar neutron flux (SI)")
_register("power_density", "W/m^3", "Volumetric power density (SI)")

# Aerodynamics
_register("lift_coefficient", "1", "Lift coefficient CL")
_register("drag_coefficient", "1", "Drag coefficient CD")
_register("pressure_coefficient", "1", "Pressure coefficient Cp")

# Radiation
_register("radiative_heat_flux", "W/m^2", "Radiative heat flux")
_register("incident_radiation", "W/m^2", "Incident radiation G")
_register("absorption_coefficient", "1/m", "Absorption coefficient")

# Electromagnetics / plasma
_register("electric_potential", "V", "Electrostatic / plasma potential")
_register("electric_field", "V/m", "Electric field magnitude")
_register("electric_field_x", "V/m", "x-component of electric field")
_register("electric_field_y", "V/m", "y-component of electric field")
_register("electric_field_z", "V/m", "z-component of electric field")
_register("current_density", "A/m^2", "Current density magnitude")
_register("current_density_x", "A/m^2", "x-component of current density")
_register("current_density_y", "A/m^2", "y-component of current density")
_register("charge_density", "C/m^3", "Volumetric charge density")
_register("electron_density", "1/m^3", "Electron number density")
_register("ion_density", "1/m^3", "Ion number density")
_register("number_density", "1/m^3", "Generic number density")
_register("number_density_2d", "1/m^2", "Areal number density (2-D continuum)")

# Irradiance / laser / directed energy
_register("irradiance", "W/m^2", "Optical or laser irradiance on target")
_register("dwell_time", "s", "Beam dwell time on target")

# Radar / EO signatures
_register("radar_cross_section", "m^2", "Radar cross-section (linear)")
_register("radar_cross_section_dbsm", "dBsm", "Radar cross-section (log, dBsm)")
_register("signal_to_noise_ratio", "1", "Signal-to-noise ratio (linear)")
_register("jamming_ratio", "1", "Jamming-to-signal ratio (linear)")
_register("detection_probability", "1", "Probability of detection (0-1)")
_register("kill_probability", "1", "Kill / intercept probability (0-1)")
_register("threat_level", "1", "Dimensionless threat-level metric (0-1)")
_register("coverage_fraction", "1", "Spatial coverage fraction (0-1)")

# Radiation / space environment
_register("particle_flux", "1/(m^2*s)", "Particle flux (any species)")
_register("dose_rate", "Gy/s", "Radiation dose rate (SI)")
_register("fluence", "1/m^2", "Time-integrated particle fluence")

# Structural / ablation / contact
_register("remaining_thickness", "m", "Remaining wall or TPS thickness")
_register("ablation_rate", "m/s", "Surface recession (ablation) rate")
_register("surface_mass_flux", "kg/(m^2*s)", "Surface mass flux (e.g. ablation)")

# Kinematics / tracking
_register("acceleration", "m/s^2", "Translational acceleration magnitude")
_register("acceleration_x", "m/s^2", "x-component of acceleration")
_register("acceleration_y", "m/s^2", "y-component of acceleration")
_register("acceleration_z", "m/s^2", "z-component of acceleration")
_register("track_error", "m", "Tracking position error")
_register("angular_rate", "rad/s", "Angular / rotation rate")

# Geospatial / terrain
_register("elevation", "m", "Terrain elevation above datum")
_register("slope", "rad", "Terrain slope angle")
_register("aspect", "rad", "Terrain aspect angle")
_register("hillshade", "1", "Hillshade index (dimensionless)")
_register("travel_time", "s", "Travel / transit time")

# Nuclear (additions)
_register("proton_flux", "1/(m^2*s)", "Proton flux (space radiation)")
_register("electron_flux", "1/(m^2*s)", "Electron flux (space radiation)")


# ---------------------------------------------------------------------------
# Alias registry for mapping local names to canonical names
# ---------------------------------------------------------------------------

# Default aliases (local_name -> canonical_name)
DEFAULT_ALIASES: Dict[str, str] = {
    # Temperature variants
    "T": "temperature",
    "T_fuel": "temperature",
    "T_coolant": "temperature",
    "T_surface": "temperature",
    "T_clad_outer": "wall_temperature",
    "T_fuel_centerline": "temperature",
    "surface_temp_K": "wall_temperature",
    "temperatures": "temperature",
    "temperature_field": "temperature",
    "electron_temperature_eV": "electron_temperature",
    "T_wall": "wall_temperature",
    "potential": "electric_potential",
    "E": "electric_field",
    "E_x": "electric_field_x",
    "E_y": "electric_field_y",
    "J_x": "current_density_x",
    "J_y": "current_density_y",
    "charge_density": "number_density",
    "combined_pd": "detection_probability",
    # Pressure variants
    "p": "pressure",
    "P": "pressure",
    "pressure_Pa": "pressure",
    # Velocity variants
    "u": "velocity_x",
    "v": "velocity_y",
    "w": "velocity_z",
    "vel_x": "velocity_x",
    "vel_y": "velocity_y",
    # Heat flux variants
    "q_wall": "heat_flux",
    "heat_flux_W_m2": "heat_flux",
    "q_heat": "heat_flux",
    "q_interface": "heat_flux",
    "q_rad": "radiative_heat_flux",
    "q_linear": "linear_heat_rate",
    # Stress variants
    "max_von_mises": "stress_von_mises",
    "sigma": "stress",
    "von_mises": "stress_von_mises",
    # Displacement
    "deflection": "displacement",
    "disp": "displacement",
    # Turbulence
    "k": "turbulent_kinetic_energy",
    "epsilon": "turbulent_dissipation",
    "mu_t": "turbulent_viscosity",
    # Nuclear
    "phi": "neutron_flux",
    "phi_thermal": "neutron_flux",
    "power_shape": "power_density",
    # Electromagnetics / plasma
    "V": "electric_potential",
    "potential": "electric_potential",
    "phi_e": "electric_potential",
    "Ex": "electric_field_x",
    "Ey": "electric_field_y",
    "E_x": "electric_field_x",
    "E_y": "electric_field_y",
    "E_field": "electric_field",
    "Jx": "current_density_x",
    "Jy": "current_density_y",
    "ne": "electron_density",
    "ni": "ion_density",
    "n_e": "electron_density",
    "n_i": "ion_density",
    "rho_charge": "charge_density",
    # Aerodynamics
    "CL": "lift_coefficient",
    "CD": "drag_coefficient",
    "Cp_distribution": "pressure_coefficient",
    # Radiation
    "G": "incident_radiation",
    "kappa": "absorption_coefficient",
    "dq_dy": "heat_source",
}


class FieldMapper:
    """Maps between local (solver-specific) and canonical field names.

    Each solver can register its own local names as aliases for canonical
    names. The mapper handles bidirectional translation.

    Example
    -------
    >>> mapper = FieldMapper("neutronics")
    >>> mapper.add_alias("phi_thermal", "neutron_flux")
    >>> mapper.add_alias("q_lin", "linear_heat_rate")
    >>> mapper.to_canonical("phi_thermal")
    'neutron_flux'
    >>> mapper.to_local("neutron_flux")
    'phi_thermal'
    """

    def __init__(self, solver_name: str):
        self.solver_name = solver_name
        self._local_to_canonical: Dict[str, str] = {}
        self._canonical_to_local: Dict[str, str] = {}

    def add_alias(self, local_name: str, canonical_name: str) -> None:
        """Register a local -> canonical mapping."""
        if canonical_name not in CANONICAL_FIELDS:
            raise ValueError(
                f"Unknown canonical field '{canonical_name}'. "
                f"Available: {list(CANONICAL_FIELDS.keys())}"
            )
        self._local_to_canonical[local_name] = canonical_name
        # First registered local name becomes the preferred reverse mapping
        if canonical_name not in self._canonical_to_local:
            self._canonical_to_local[canonical_name] = local_name

    def to_canonical(self, local_name: str) -> str:
        """Translate a local field name to its canonical equivalent."""
        if local_name in self._local_to_canonical:
            return self._local_to_canonical[local_name]
        # Check default aliases
        if local_name in DEFAULT_ALIASES:
            return DEFAULT_ALIASES[local_name]
        # If the name is already canonical, return it
        if local_name in CANONICAL_FIELDS:
            return local_name
        raise KeyError(
            f"No canonical mapping for local field '{local_name}' "
            f"in solver '{self.solver_name}'"
        )

    def to_local(self, canonical_name: str) -> str:
        """Translate a canonical field name to this solver's local name."""
        if canonical_name in self._canonical_to_local:
            return self._canonical_to_local[canonical_name]
        # If the canonical name itself is used locally, return it
        if canonical_name in CANONICAL_FIELDS:
            return canonical_name
        raise KeyError(
            f"No local mapping for canonical field '{canonical_name}' "
            f"in solver '{self.solver_name}'"
        )

    def has_field(self, canonical_name: str) -> bool:
        """Check if this solver provides a given canonical field."""
        return canonical_name in self._canonical_to_local


# ---------------------------------------------------------------------------
# PhysicsField: a single named, unit-aware field
# ---------------------------------------------------------------------------

@dataclass
class PhysicsField:
    """A named physical field with unit metadata and spatial information.

    Attributes
    ----------
    name : str
        Canonical field name.
    data : np.ndarray
        Field values.
    unit : str
        Unit key from SI_UNITS.
    grid : np.ndarray, optional
        Spatial coordinates (1-D) or None.
    time : float, optional
        Time stamp [s].
    """
    name: str
    data: np.ndarray
    unit: str
    grid: Optional[np.ndarray] = None
    time: Optional[float] = None

    def to_si(self) -> 'PhysicsField':
        """Return a copy with data converted to SI."""
        if self.unit not in SI_UNITS:
            return PhysicsField(self.name, self.data.copy(), self.unit,
                                self.grid, self.time)
        spec = SI_UNITS[self.unit]
        si_data = spec.to_si(self.data)
        # Find the corresponding SI unit
        defn = CANONICAL_FIELDS.get(self.name)
        si_unit = defn.si_unit if defn else self.unit
        return PhysicsField(self.name, si_data, si_unit, self.grid, self.time)

    def interpolate_to(self, target_grid: np.ndarray) -> 'PhysicsField':
        """Interpolate this field onto a different spatial grid."""
        if self.grid is None:
            raise ValueError(f"Field '{self.name}' has no grid for interpolation")
        interp_data = np.interp(target_grid, self.grid, self.data)
        return PhysicsField(self.name, interp_data, self.unit,
                            target_grid, self.time)


# ---------------------------------------------------------------------------
# PhysicsState: collection of fields representing a solver's output
# ---------------------------------------------------------------------------

@dataclass
class PhysicsState:
    """Collection of physics fields forming a complete solver state.

    This is the standard exchange format between coupled solvers.
    All fields use canonical names and SI units.

    Example
    -------
    >>> state = PhysicsState(solver_name="cfd_turbulence")
    >>> state.set_field("temperature", T_array, "K", grid=x_array)
    >>> state.set_field("pressure", p_array, "Pa", grid=x_array)
    >>> T = state.get("temperature")  # -> PhysicsField
    >>> T_data = state["temperature"]  # -> np.ndarray (shortcut)
    """
    solver_name: str = ""
    time: float = 0.0
    fields: Dict[str, PhysicsField] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_field(self, name: str, data: Union[np.ndarray, float],
                  unit: str, grid: Optional[np.ndarray] = None) -> None:
        """Set a field by canonical name."""
        if isinstance(data, (int, float)):
            data = np.array([data])
        self.fields[name] = PhysicsField(
            name=name, data=np.asarray(data), unit=unit,
            grid=grid, time=self.time,
        )

    def get(self, name: str) -> PhysicsField:
        """Get a field by canonical name."""
        if name not in self.fields:
            raise KeyError(
                f"Field '{name}' not in state from '{self.solver_name}'. "
                f"Available: {list(self.fields.keys())}"
            )
        return self.fields[name]

    def __getitem__(self, name: str) -> np.ndarray:
        """Shortcut: state['temperature'] returns the data array."""
        return self.get(name).data

    def __contains__(self, name: str) -> bool:
        return name in self.fields

    def has(self, name: str) -> bool:
        """Check if a field exists in this state."""
        return name in self.fields

    def field_names(self) -> List[str]:
        """List all available field names."""
        return list(self.fields.keys())

    def to_si(self) -> 'PhysicsState':
        """Return a copy with all fields converted to SI."""
        si_state = PhysicsState(
            solver_name=self.solver_name,
            time=self.time,
            metadata=dict(self.metadata),
        )
        for name, f in self.fields.items():
            si_field = f.to_si()
            si_state.fields[name] = si_field
        return si_state

    def interpolate_to(self, target_grid: np.ndarray,
                       field_names: Optional[List[str]] = None) -> 'PhysicsState':
        """Interpolate specified fields onto a target grid."""
        names = field_names or list(self.fields.keys())
        new_state = PhysicsState(
            solver_name=self.solver_name,
            time=self.time,
            metadata=dict(self.metadata),
        )
        for name in names:
            if name in self.fields:
                f = self.fields[name]
                if f.grid is not None and f.data.ndim == 1:
                    new_state.fields[name] = f.interpolate_to(target_grid)
                else:
                    new_state.fields[name] = f
        return new_state


# ---------------------------------------------------------------------------
# Protocol for solver coupling interface
# ---------------------------------------------------------------------------

@runtime_checkable
class CoupledSolver(Protocol):
    """Protocol that coupled solvers should implement for field exchange.

    Any solver that participates in multi-physics coupling should provide:
    - export_state() to expose its fields in canonical form
    - import_state() to receive fields from other solvers
    """

    def export_state(self) -> PhysicsState:
        """Export the current solver state as canonical PhysicsState."""
        ...

    def import_state(self, state: PhysicsState) -> None:
        """Import external fields into this solver."""
        ...
