"""
Unit System for TRIALITY

Provides a canonical SI-based unit system with:
- Unit metadata that can be attached to any field or result
- Conversion factors between SI and CGS (for nuclear module boundary)
- Automatic boundary adapters for nuclear ↔ SI interfaces
- Dimension checking for coupling safety

The internal canonical system is SI throughout:
    Length: m, Mass: kg, Time: s, Temperature: K,
    Amount: mol, Current: A, Luminous intensity: cd

Nuclear modules historically use CGS (cm, g, s). This module provides
explicit adapters so those modules can keep their internal conventions
while exposing SI at their boundaries.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Union, Any
from enum import Enum


# ---------------------------------------------------------------------------
# Unit dimensions
# ---------------------------------------------------------------------------

class Dimension(Enum):
    """Physical dimensions."""
    LENGTH = "length"
    MASS = "mass"
    TIME = "time"
    TEMPERATURE = "temperature"
    AMOUNT = "amount"
    CURRENT = "current"
    ANGLE = "angle"
    DIMENSIONLESS = "dimensionless"
    # Derived
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    FORCE = "force"
    PRESSURE = "pressure"
    ENERGY = "energy"
    POWER = "power"
    DENSITY = "density"
    HEAT_FLUX = "heat_flux"
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    SPECIFIC_HEAT = "specific_heat"
    DYNAMIC_VISCOSITY = "dynamic_viscosity"
    KINEMATIC_VISCOSITY = "kinematic_viscosity"
    STRESS = "stress"
    STRAIN = "strain"
    AREA = "area"
    VOLUME = "volume"
    MASS_FLUX = "mass_flux"
    FREQUENCY = "frequency"
    ANGULAR_VELOCITY = "angular_velocity"
    ELECTRIC_POTENTIAL = "electric_potential"
    ELECTRIC_FIELD = "electric_field"
    CURRENT_DENSITY = "current_density"
    # Nuclear-specific
    NEUTRON_FLUX = "neutron_flux"           # n/(cm^2·s) in CGS, n/(m^2·s) in SI
    MACROSCOPIC_XS = "macroscopic_xs"       # 1/cm in CGS, 1/m in SI
    LINEAR_HEAT_RATE = "linear_heat_rate"   # W/cm in nuclear, W/m in SI
    NUMBER_DENSITY = "number_density"       # 1/cm^3 in CGS, 1/m^3 in SI
    POWER_DENSITY = "power_density"         # W/cm^3 in CGS, W/m^3 in SI
    DIFFUSION_COEFF = "diffusion_coeff"     # cm in CGS, m in SI
    # Turbulence
    SPECIFIC_ENERGY = "specific_energy"     # m^2/s^2 (J/kg) — turbulent kinetic energy
    DISSIPATION_RATE = "dissipation_rate"   # m^2/s^3 (W/kg) — turbulent dissipation rate


# ---------------------------------------------------------------------------
# Unit metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UnitSpec:
    """Unit specification for a physical quantity.

    Attributes
    ----------
    symbol : str
        Unit symbol, e.g. 'm', 'Pa', 'K', 'W/m^2'.
    dimension : Dimension
        Physical dimension category.
    si_factor : float
        Multiply value in these units by si_factor to get SI.
        For SI units, si_factor = 1.0.
    si_offset : float
        Additive offset for SI conversion (only for temperature).
        SI_value = value * si_factor + si_offset.
    """
    symbol: str
    dimension: Dimension
    si_factor: float = 1.0
    si_offset: float = 0.0

    def to_si(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert value from this unit to SI."""
        return value * self.si_factor + self.si_offset

    def from_si(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert value from SI to this unit."""
        return (value - self.si_offset) / self.si_factor


# ---------------------------------------------------------------------------
# Standard unit registry
# ---------------------------------------------------------------------------

# SI base units
SI_UNITS: Dict[str, UnitSpec] = {
    # Length
    "m": UnitSpec("m", Dimension.LENGTH),
    "cm": UnitSpec("cm", Dimension.LENGTH, 1e-2),
    "mm": UnitSpec("mm", Dimension.LENGTH, 1e-3),
    "km": UnitSpec("km", Dimension.LENGTH, 1e3),
    # Mass
    "kg": UnitSpec("kg", Dimension.MASS),
    "g": UnitSpec("g", Dimension.MASS, 1e-3),
    # Time
    "s": UnitSpec("s", Dimension.TIME),
    "ms": UnitSpec("ms", Dimension.TIME, 1e-3),
    "us": UnitSpec("us", Dimension.TIME, 1e-6),
    # Temperature
    "K": UnitSpec("K", Dimension.TEMPERATURE),
    "degC": UnitSpec("degC", Dimension.TEMPERATURE, 1.0, 273.15),
    # Velocity
    "m/s": UnitSpec("m/s", Dimension.VELOCITY),
    "cm/s": UnitSpec("cm/s", Dimension.VELOCITY, 1e-2),
    "m/s^2": UnitSpec("m/s^2", Dimension.ACCELERATION),
    # Pressure
    "Pa": UnitSpec("Pa", Dimension.PRESSURE),
    "MPa": UnitSpec("MPa", Dimension.PRESSURE, 1e6),
    "GPa": UnitSpec("GPa", Dimension.PRESSURE, 1e9),
    "bar": UnitSpec("bar", Dimension.PRESSURE, 1e5),
    "atm": UnitSpec("atm", Dimension.PRESSURE, 101325.0),
    # Force
    "N": UnitSpec("N", Dimension.FORCE),
    "kN": UnitSpec("kN", Dimension.FORCE, 1e3),
    # Energy
    "J": UnitSpec("J", Dimension.ENERGY),
    "kJ": UnitSpec("kJ", Dimension.ENERGY, 1e3),
    "MJ": UnitSpec("MJ", Dimension.ENERGY, 1e6),
    "eV": UnitSpec("eV", Dimension.ENERGY, 1.602176634e-19),
    "MeV": UnitSpec("MeV", Dimension.ENERGY, 1.602176634e-13),
    # Power
    "W": UnitSpec("W", Dimension.POWER),
    "kW": UnitSpec("kW", Dimension.POWER, 1e3),
    "MW": UnitSpec("MW", Dimension.POWER, 1e6),
    # Density
    "kg/m^3": UnitSpec("kg/m^3", Dimension.DENSITY),
    "g/cm^3": UnitSpec("g/cm^3", Dimension.DENSITY, 1e3),
    # Heat flux
    "W/m^2": UnitSpec("W/m^2", Dimension.HEAT_FLUX),
    "W/cm^2": UnitSpec("W/cm^2", Dimension.HEAT_FLUX, 1e4),
    # Thermal conductivity
    "W/(m*K)": UnitSpec("W/(m*K)", Dimension.THERMAL_CONDUCTIVITY),
    "W/(cm*K)": UnitSpec("W/(cm*K)", Dimension.THERMAL_CONDUCTIVITY, 1e2),
    # Specific heat
    "J/(kg*K)": UnitSpec("J/(kg*K)", Dimension.SPECIFIC_HEAT),
    # Viscosity
    "Pa*s": UnitSpec("Pa*s", Dimension.DYNAMIC_VISCOSITY),
    "m^2/s": UnitSpec("m^2/s", Dimension.KINEMATIC_VISCOSITY),
    # Turbulence
    "m^2/s^2": UnitSpec("m^2/s^2", Dimension.SPECIFIC_ENERGY),
    "m^2/s^3": UnitSpec("m^2/s^3", Dimension.DISSIPATION_RATE),
    # Stress (same dimension as pressure)
    "MPa_stress": UnitSpec("MPa", Dimension.STRESS, 1e6),
    # Area
    "m^2": UnitSpec("m^2", Dimension.AREA),
    "cm^2": UnitSpec("cm^2", Dimension.AREA, 1e-4),
    # Volume
    "m^3": UnitSpec("m^3", Dimension.VOLUME),
    "cm^3": UnitSpec("cm^3", Dimension.VOLUME, 1e-6),
    # Mass flux
    "kg/(m^2*s)": UnitSpec("kg/(m^2*s)", Dimension.MASS_FLUX),
    # Frequency
    "Hz": UnitSpec("Hz", Dimension.FREQUENCY),
    # Electrical
    "V": UnitSpec("V", Dimension.ELECTRIC_POTENTIAL),
    "V/m": UnitSpec("V/m", Dimension.ELECTRIC_FIELD),
    "A/m^2": UnitSpec("A/m^2", Dimension.CURRENT_DENSITY),
    # Angular velocity
    "rad/s": UnitSpec("rad/s", Dimension.ANGULAR_VELOCITY),
    # Nuclear-specific
    "n/(m^2*s)": UnitSpec("n/(m^2*s)", Dimension.NEUTRON_FLUX),
    "n/(cm^2*s)": UnitSpec("n/(cm^2*s)", Dimension.NEUTRON_FLUX, 1e4),
    "1/m": UnitSpec("1/m", Dimension.MACROSCOPIC_XS),
    "1/cm": UnitSpec("1/cm", Dimension.MACROSCOPIC_XS, 1e2),
    "W/m": UnitSpec("W/m", Dimension.LINEAR_HEAT_RATE),
    "W/cm": UnitSpec("W/cm", Dimension.LINEAR_HEAT_RATE, 1e2),
    "1/m^3": UnitSpec("1/m^3", Dimension.NUMBER_DENSITY),
    "1/cm^3": UnitSpec("1/cm^3", Dimension.NUMBER_DENSITY, 1e6),
    "W/m^3": UnitSpec("W/m^3", Dimension.POWER_DENSITY),
    "W/cm^3": UnitSpec("W/cm^3", Dimension.POWER_DENSITY, 1e6),
    "m_diff": UnitSpec("m", Dimension.DIFFUSION_COEFF),
    "cm_diff": UnitSpec("cm", Dimension.DIFFUSION_COEFF, 1e-2),
    # Dimensionless
    "1": UnitSpec("1", Dimension.DIMENSIONLESS),
}


def convert(value: Union[float, np.ndarray],
            from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """Convert a value between two units.

    Parameters
    ----------
    value : float or np.ndarray
        Value(s) in ``from_unit``.
    from_unit : str
        Source unit key from SI_UNITS.
    to_unit : str
        Target unit key from SI_UNITS.

    Returns
    -------
    float or np.ndarray
        Converted value(s).

    Raises
    ------
    KeyError
        If either unit is not in the registry.
    ValueError
        If units have incompatible dimensions.
    """
    src = SI_UNITS[from_unit]
    dst = SI_UNITS[to_unit]
    if src.dimension != dst.dimension:
        raise ValueError(
            f"Cannot convert {from_unit} ({src.dimension.value}) "
            f"to {to_unit} ({dst.dimension.value}): incompatible dimensions"
        )
    si_value = src.to_si(value)
    return dst.from_si(si_value)


# ---------------------------------------------------------------------------
# Field unit metadata
# ---------------------------------------------------------------------------

@dataclass
class FieldUnit:
    """Unit annotation for a named field.

    Attach this to result fields to make unit provenance explicit.

    Attributes
    ----------
    name : str
        Canonical field name (e.g. 'temperature', 'pressure').
    unit : str
        Unit key from SI_UNITS.
    description : str
        Human-readable description.
    """
    name: str
    unit: str
    description: str = ""

    @property
    def spec(self) -> UnitSpec:
        return SI_UNITS[self.unit]

    @property
    def is_si(self) -> bool:
        return self.spec.si_factor == 1.0 and self.spec.si_offset == 0.0


@dataclass
class UnitMetadata:
    """Unit metadata container for a solver result.

    Attach to result dataclasses to declare units of all fields.

    Example
    -------
    >>> meta = UnitMetadata()
    >>> meta.declare("temperature", "K", "Bulk temperature")
    >>> meta.declare("pressure", "Pa", "Static pressure")
    >>> meta.declare("q_linear", "W/cm", "Linear heat rate (nuclear convention)")
    >>> meta.to_si("q_linear", 200.0)  # -> 20000.0 W/m
    """
    fields: Dict[str, FieldUnit] = field(default_factory=dict)

    def declare(self, name: str, unit: str, description: str = "") -> None:
        """Declare the unit of a named field."""
        if unit not in SI_UNITS:
            raise KeyError(f"Unknown unit '{unit}'. Available: {list(SI_UNITS.keys())}")
        self.fields[name] = FieldUnit(name, unit, description)

    def get_unit(self, name: str) -> str:
        """Get the unit string of a field."""
        return self.fields[name].unit

    def is_si(self, name: str) -> bool:
        """Check if a field is in SI units."""
        return self.fields[name].is_si

    def to_si(self, name: str, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert a field value to SI."""
        fu = self.fields[name]
        return fu.spec.to_si(value)

    def from_si(self, name: str, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert a field value from SI to its declared unit."""
        fu = self.fields[name]
        return fu.spec.from_si(value)


# ---------------------------------------------------------------------------
# Nuclear ↔ SI boundary adapters
# ---------------------------------------------------------------------------

class NuclearSIAdapter:
    """Boundary adapter for converting between nuclear CGS conventions and SI.

    Nuclear modules historically use:
        Length: cm, Cross-sections: 1/cm, Flux: n/(cm^2*s),
        Linear heat rate: W/cm, Power density: W/cm^3,
        Diffusion coefficient: cm

    This adapter provides explicit, named conversions at the module boundary
    so that internal code can remain in CGS while external interfaces use SI.

    Example
    -------
    >>> adapter = NuclearSIAdapter()
    >>> # Convert nuclear output to SI for coupling
    >>> q_linear_si = adapter.linear_heat_rate_to_si(200.0)  # W/cm -> W/m
    >>> assert q_linear_si == 20000.0
    >>> # Convert SI input to nuclear convention
    >>> length_cgs = adapter.length_to_cgs(2.0)  # m -> cm
    >>> assert length_cgs == 200.0
    """

    # Length
    @staticmethod
    def length_to_si(cm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """cm -> m"""
        return np.asarray(cm) * 1e-2

    @staticmethod
    def length_to_cgs(m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """m -> cm"""
        return np.asarray(m) * 1e2

    # Area
    @staticmethod
    def area_to_si(cm2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """cm^2 -> m^2"""
        return np.asarray(cm2) * 1e-4

    @staticmethod
    def area_to_cgs(m2: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """m^2 -> cm^2"""
        return np.asarray(m2) * 1e4

    # Volume
    @staticmethod
    def volume_to_si(cm3: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """cm^3 -> m^3"""
        return np.asarray(cm3) * 1e-6

    @staticmethod
    def volume_to_cgs(m3: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """m^3 -> cm^3"""
        return np.asarray(m3) * 1e6

    # Macroscopic cross-section
    @staticmethod
    def macro_xs_to_si(per_cm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """1/cm -> 1/m"""
        return np.asarray(per_cm) * 1e2

    @staticmethod
    def macro_xs_to_cgs(per_m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """1/m -> 1/cm"""
        return np.asarray(per_m) * 1e-2

    # Neutron flux
    @staticmethod
    def flux_to_si(n_per_cm2_s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """n/(cm^2*s) -> n/(m^2*s)"""
        return np.asarray(n_per_cm2_s) * 1e4

    @staticmethod
    def flux_to_cgs(n_per_m2_s: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """n/(m^2*s) -> n/(cm^2*s)"""
        return np.asarray(n_per_m2_s) * 1e-4

    # Linear heat rate
    @staticmethod
    def linear_heat_rate_to_si(W_per_cm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """W/cm -> W/m"""
        return np.asarray(W_per_cm) * 1e2

    @staticmethod
    def linear_heat_rate_to_cgs(W_per_m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """W/m -> W/cm"""
        return np.asarray(W_per_m) * 1e-2

    # Power density
    @staticmethod
    def power_density_to_si(W_per_cm3: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """W/cm^3 -> W/m^3"""
        return np.asarray(W_per_cm3) * 1e6

    @staticmethod
    def power_density_to_cgs(W_per_m3: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """W/m^3 -> W/cm^3"""
        return np.asarray(W_per_m3) * 1e-6

    # Number density
    @staticmethod
    def number_density_to_si(per_cm3: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """1/cm^3 -> 1/m^3"""
        return np.asarray(per_cm3) * 1e6

    @staticmethod
    def number_density_to_cgs(per_m3: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """1/m^3 -> 1/cm^3"""
        return np.asarray(per_m3) * 1e-6

    # Diffusion coefficient
    @staticmethod
    def diffusion_coeff_to_si(cm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """cm -> m"""
        return np.asarray(cm) * 1e-2

    @staticmethod
    def diffusion_coeff_to_cgs(m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """m -> cm"""
        return np.asarray(m) * 1e2

    # Heat transfer coefficient (W/(cm^2*K) -> W/(m^2*K))
    @staticmethod
    def htc_to_si(W_per_cm2_K: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """W/(cm^2*K) -> W/(m^2*K)"""
        return np.asarray(W_per_cm2_K) * 1e4

    @staticmethod
    def htc_to_cgs(W_per_m2_K: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """W/(m^2*K) -> W/(cm^2*K)"""
        return np.asarray(W_per_m2_K) * 1e-4

    def convert_result_to_si(self, data: Dict[str, Any],
                              field_units: Dict[str, str]) -> Dict[str, Any]:
        """Batch-convert a dict of results from nuclear CGS to SI.

        Parameters
        ----------
        data : dict
            Field name -> value mapping.
        field_units : dict
            Field name -> unit string mapping (e.g. {'q_linear': 'W/cm'}).

        Returns
        -------
        dict
            Converted values in SI.
        """
        result = {}
        for name, value in data.items():
            if name in field_units:
                unit_str = field_units[name]
                if unit_str in SI_UNITS:
                    spec = SI_UNITS[unit_str]
                    result[name] = spec.to_si(value)
                else:
                    result[name] = value
            else:
                result[name] = value
        return result

    def convert_input_to_cgs(self, data: Dict[str, Any],
                              field_units: Dict[str, str]) -> Dict[str, Any]:
        """Batch-convert a dict of SI inputs to nuclear CGS.

        Parameters
        ----------
        data : dict
            Field name -> SI value mapping.
        field_units : dict
            Field name -> target CGS unit string mapping.

        Returns
        -------
        dict
            Converted values in CGS.
        """
        result = {}
        for name, value in data.items():
            if name in field_units:
                unit_str = field_units[name]
                if unit_str in SI_UNITS:
                    spec = SI_UNITS[unit_str]
                    result[name] = spec.from_si(value)
                else:
                    result[name] = value
            else:
                result[name] = value
        return result
