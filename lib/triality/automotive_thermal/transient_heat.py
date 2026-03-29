"""
Transient Heat Conduction for Automotive Applications

Solves time-dependent heat equation with multiple sources:
∂T/∂t = α∇²T + Q(x,y,z,t)/ρc

where:
- T: temperature [K]
- α: thermal diffusivity [m²/s]
- Q: volumetric heat generation [W/m³]
- ρ: density [kg/m³]
- c: specific heat [J/(kg·K)]

Key features for automotive:
- Transient thermal response
- Joule heating from current flow
- Multi-source heat aggregation
- Hotspot detection
- Thermal margin tracking
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from enum import Enum


class CoolingMode(Enum):
    """Cooling mechanisms"""
    NATURAL_CONVECTION = "Natural Convection"
    FORCED_CONVECTION = "Forced Convection"
    LIQUID_COOLING = "Liquid Cooling"
    PHASE_CHANGE = "Phase Change (Boiling)"
    NONE = "No Active Cooling"


@dataclass
class ThermalMaterial:
    """Thermal properties of materials"""
    name: str
    thermal_conductivity: float  # W/(m·K)
    density: float  # kg/m³
    specific_heat: float  # J/(kg·K)
    max_temperature: float = 423.0  # K (150°C typical polymer limit)

    @property
    def thermal_diffusivity(self) -> float:
        """Thermal diffusivity α = k/(ρc) [m²/s]"""
        return self.thermal_conductivity / (self.density * self.specific_heat)


# Standard automotive materials
COPPER = ThermalMaterial(
    name="Copper (busbar)",
    thermal_conductivity=385.0,
    density=8960.0,
    specific_heat=385.0,
    max_temperature=473.0  # 200°C for copper
)

ALUMINUM = ThermalMaterial(
    name="Aluminum (cable)",
    thermal_conductivity=205.0,
    density=2700.0,
    specific_heat=900.0,
    max_temperature=453.0  # 180°C for Al
)

SILICON = ThermalMaterial(
    name="Silicon (IGBT)",
    thermal_conductivity=148.0,
    density=2330.0,
    specific_heat=700.0,
    max_temperature=448.0  # 175°C junction temp
)

THERMAL_PAD = ThermalMaterial(
    name="Thermal Interface Material",
    thermal_conductivity=5.0,
    density=2000.0,
    specific_heat=1000.0,
    max_temperature=423.0
)

COOLANT = ThermalMaterial(
    name="Glycol Coolant",
    thermal_conductivity=0.4,
    density=1070.0,
    specific_heat=3300.0,
    max_temperature=383.0  # 110°C
)


@dataclass
class HeatSource:
    """Heat generation source"""
    position: Tuple[float, float, float]  # (x, y, z) [m]
    power: float  # [W]
    volume: float = 1e-6  # [m³] (default 1 cm³)
    time_profile: Optional[Callable[[float], float]] = None  # P(t)/P_rated

    def power_at_time(self, t: float) -> float:
        """Power at time t [W]"""
        if self.time_profile is None:
            return self.power
        return self.power * self.time_profile(t)

    def power_density(self, t: float = 0.0) -> float:
        """Volumetric power density [W/m³]"""
        return self.power_at_time(t) / self.volume


@dataclass
class Hotspot:
    """Detected thermal hotspot"""
    position: Tuple[float, float, float]
    temperature: float  # K
    margin_to_limit: float  # K (negative if over limit)
    power_density: float  # W/m³
    risk_level: str  # "Safe", "Warning", "Critical"


@dataclass
class ThermalResult:
    """Results from thermal analysis"""
    times: np.ndarray  # Time points [s]
    temperatures: np.ndarray  # Temperature field [K] (time, space)
    hotspots: List[Hotspot]
    max_temperature: float  # K
    max_temp_location: Tuple[float, float, float]
    thermal_margin: float  # K (to material limit)
    cooling_adequate: bool


class TransientHeatSolver1D:
    """
    1D transient heat conduction solver for automotive components

    Useful for:
    - Busbar thermal analysis
    - Cable heating
    - Through-thickness thermal gradients
    """

    def __init__(self, length: float, n_points: int = 100,
                 material: ThermalMaterial = COPPER):
        """
        Initialize 1D transient heat solver

        Parameters:
        -----------
        length : float
            Domain length [m]
        n_points : int
            Number of spatial points
        material : ThermalMaterial
            Material properties
        """
        self.length = length
        self.n_points = n_points
        self.material = material

        self.dx = length / (n_points - 1)
        self.x = np.linspace(0, length, n_points)

        # Initialize temperature field
        self.T = np.ones(n_points) * 298.0  # 25°C ambient

    def joule_heating(self, current: float, resistivity: float,
                      cross_section_area: float) -> float:
        """
        Calculate Joule heating power density

        P_joule = ρ_e * J² = ρ_e * (I/A)²

        Parameters:
        -----------
        current : float
            Current [A]
        resistivity : float
            Electrical resistivity [Ω·m]
        cross_section_area : float
            Cross-sectional area [m²]

        Returns:
        --------
        float
            Volumetric power density [W/m³]
        """
        J = current / cross_section_area  # Current density [A/m²]
        Q = resistivity * J**2  # W/m³
        return Q

    def solve_transient(self, t_end: float, dt: float,
                        heat_sources: List[HeatSource] = None,
                        boundary_conditions: Tuple[str, str] = ("convection", "convection"),
                        h_conv: float = 10.0,
                        T_ambient: float = 298.0) -> ThermalResult:
        """
        Solve transient heat equation using implicit finite difference

        ∂T/∂t = α ∂²T/∂x² + Q/ρc

        Parameters:
        -----------
        t_end : float
            End time [s]
        dt : float
            Time step [s]
        heat_sources : List[HeatSource]
            Heat generation sources
        boundary_conditions : tuple
            ("left", "right") BC types: "convection", "adiabatic", "fixed"
        h_conv : float
            Convection coefficient [W/(m²·K)]
        T_ambient : float
            Ambient temperature [K]

        Returns:
        --------
        ThermalResult
        """
        n_steps = int(t_end / dt)
        times = np.linspace(0, t_end, n_steps + 1)

        # Storage
        T_history = np.zeros((n_steps + 1, self.n_points))
        T_history[0, :] = self.T.copy()

        # Thermal diffusivity
        alpha = self.material.thermal_diffusivity

        # Stability criterion: dt ≤ dx²/(2α)
        dt_crit = self.dx**2 / (2 * alpha)
        if dt > dt_crit:
            print(f"Warning: dt={dt:.2e} exceeds stability limit {dt_crit:.2e}")

        # Fourier number
        Fo = alpha * dt / self.dx**2

        # Build tri-diagonal matrix for implicit scheme
        # Crank-Nicolson: (1 + Fo)T^(n+1) - 0.5*Fo*(T^(n+1)_(i-1) + T^(n+1)_(i+1)) = ...
        # For simplicity, use fully implicit (backward Euler)

        main_diag = np.ones(self.n_points) * (1 + 2*Fo)
        off_diag = np.ones(self.n_points - 1) * (-Fo)

        # Boundary conditions modify first and last rows
        if boundary_conditions[0] == "adiabatic":
            main_diag[0] = 1 + Fo
        elif boundary_conditions[0] == "convection":
            # -k dT/dx = h(T - T_amb) at x=0
            Bi = h_conv * self.dx / self.material.thermal_conductivity
            main_diag[0] = 1 + Fo * (1 + Bi)

        if boundary_conditions[1] == "adiabatic":
            main_diag[-1] = 1 + Fo
        elif boundary_conditions[1] == "convection":
            Bi = h_conv * self.dx / self.material.thermal_conductivity
            main_diag[-1] = 1 + Fo * (1 + Bi)

        A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], format='csr')

        # Time stepping
        for step in range(n_steps):
            t = times[step]

            # Heat source term
            Q_vol = np.zeros(self.n_points)
            if heat_sources:
                for source in heat_sources:
                    # Find nearest point
                    x_src = source.position[0]
                    idx = np.argmin(np.abs(self.x - x_src))
                    Q_vol[idx] = source.power_density(t)

            # RHS: T^n + dt*Q/(ρc)
            rhs = self.T + dt * Q_vol / (self.material.density * self.material.specific_heat)

            # Add ambient temperature contribution from convection BC
            if boundary_conditions[0] == "convection":
                Bi = h_conv * self.dx / self.material.thermal_conductivity
                rhs[0] += Fo * Bi * T_ambient
            if boundary_conditions[1] == "convection":
                Bi = h_conv * self.dx / self.material.thermal_conductivity
                rhs[-1] += Fo * Bi * T_ambient

            # Solve
            self.T = spsolve(A, rhs)
            T_history[step + 1, :] = self.T

        # Detect hotspots
        hotspots = self._detect_hotspots(self.T, heat_sources or [])

        # Maximum temperature
        max_T = np.max(self.T)
        max_idx = np.argmax(self.T)
        max_loc = (self.x[max_idx], 0.0, 0.0)

        # Thermal margin
        margin = self.material.max_temperature - max_T

        return ThermalResult(
            times=times,
            temperatures=T_history,
            hotspots=hotspots,
            max_temperature=max_T,
            max_temp_location=max_loc,
            thermal_margin=margin,
            cooling_adequate=(margin > 0)
        )

    def _detect_hotspots(self, T: np.ndarray, sources: List[HeatSource]) -> List[Hotspot]:
        """Detect thermal hotspots"""
        hotspots = []

        # Find local maxima
        for i in range(1, len(T) - 1):
            if T[i] > T[i-1] and T[i] > T[i+1]:
                # Local maximum found
                margin = self.material.max_temperature - T[i]

                if margin < 0:
                    risk = "Critical"
                elif margin < 20:
                    risk = "Warning"
                else:
                    risk = "Safe"

                # Find associated power density
                Q = 0.0
                for src in sources:
                    if abs(self.x[i] - src.position[0]) < self.dx:
                        Q = src.power_density()

                hotspot = Hotspot(
                    position=(self.x[i], 0.0, 0.0),
                    temperature=T[i],
                    margin_to_limit=margin,
                    power_density=Q,
                    risk_level=risk
                )
                hotspots.append(hotspot)

        return hotspots


def calculate_busbar_temperature_rise(current: float, length: float,
                                       width: float, thickness: float,
                                       material: ThermalMaterial = COPPER,
                                       h_conv: float = 10.0,
                                       T_ambient: float = 298.0) -> Dict:
    """
    Calculate steady-state temperature rise in a busbar

    Simplified analytical solution for rectangular busbar

    Parameters:
    -----------
    current : float
        Current [A]
    length : float
        Busbar length [m]
    width : float
        Busbar width [m]
    thickness : float
        Busbar thickness [m]
    material : ThermalMaterial
    h_conv : float
        Convection coefficient [W/(m²·K)]
    T_ambient : float
        Ambient temperature [K]

    Returns:
    --------
    dict : Results including temperature, heat dissipation, margin
    """
    # Electrical resistivity (approximate, temperature-independent)
    if material.name.startswith("Copper"):
        rho_e = 1.68e-8  # Ω·m at 20°C
    elif material.name.startswith("Aluminum"):
        rho_e = 2.65e-8  # Ω·m
    else:
        rho_e = 1e-7  # Default

    # Cross-sectional area
    A_cross = width * thickness  # m²

    # Resistance
    R = rho_e * length / A_cross  # Ω

    # Joule heating power
    P_joule = current**2 * R  # W

    # Surface area for convection (perimeter × length)
    perimeter = 2 * (width + thickness)
    A_surf = perimeter * length  # m²

    # Steady-state energy balance: P_joule = h * A_surf * (T - T_amb)
    delta_T = P_joule / (h_conv * A_surf)  # K

    T_max = T_ambient + delta_T

    # Thermal margin
    margin = material.max_temperature - T_max

    return {
        "temperature_K": T_max,
        "temperature_C": T_max - 273.15,
        "delta_T": delta_T,
        "power_dissipation_W": P_joule,
        "current_density_A_per_mm2": current / (A_cross * 1e6),
        "thermal_margin_K": margin,
        "safe": margin > 0
    }
