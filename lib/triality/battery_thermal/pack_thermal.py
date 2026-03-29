"""
Battery Pack Thermal Analysis for Automotive Applications

Non-electrochemical thermal modeling for battery pack safety and cooling design.

Models:
- Cell-level heat generation (I²R + calendar aging)
- Cell-to-cell thermal propagation
- Thermal runaway trigger and propagation
- Cooling system effectiveness
- Safety spacing logic

NOT modeled (electrochemistry - outside scope):
- Lithium plating
- SEI layer growth
- Capacity fade mechanisms
- Detailed cell chemistry

This is pack-level thermal physics for automotive safety analysis.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum


class CellChemistry(Enum):
    """Battery cell chemistries"""
    NMC = "NMC (Nickel Manganese Cobalt)"
    LFP = "LFP (Lithium Iron Phosphate)"
    NCA = "NCA (Nickel Cobalt Aluminum)"


class CoolingType(Enum):
    """Cooling system types"""
    AIR_NATURAL = "Natural Air Convection"
    AIR_FORCED = "Forced Air Cooling"
    LIQUID_INDIRECT = "Indirect Liquid Cooling (cold plates)"
    LIQUID_IMMERSION = "Direct Immersion Cooling"


@dataclass
class BatteryCell:
    """Individual battery cell model"""
    cell_id: int
    position: Tuple[float, float, float]  # (x, y, z) [m]
    capacity_Ah: float = 50.0  # Ampere-hours
    nominal_voltage: float = 3.7  # V
    internal_resistance: float = 0.002  # Ω (2 mΩ typical for automotive)
    mass_kg: float = 0.8  # kg (typical 50 Ah prismatic cell)
    specific_heat: float = 900.0  # J/(kg·K)
    surface_area: float = 0.03  # m² (cylindrical or prismatic)

    # Thermal state
    temperature: float = 298.0  # K (25°C)
    current: float = 0.0  # A (positive = discharge)
    state_of_charge: float = 0.5  # 0-1

    # Thermal runaway
    in_thermal_runaway: bool = False
    runaway_triggered_time: Optional[float] = None


@dataclass
class PackConfiguration:
    """Battery pack configuration"""
    n_cells: int
    cell_spacing: float  # m (center-to-center)
    cooling_type: CoolingType = CoolingType.AIR_FORCED
    h_convection: float = 50.0  # W/(m²·K)
    T_coolant: float = 298.0  # K
    thermal_barriers: bool = False  # Intumescent material between cells


@dataclass
class ThermalRunawayEvent:
    """Thermal runaway event record"""
    cell_id: int
    trigger_time: float  # s
    trigger_temperature: float  # K
    peak_temperature: float  # K
    propagated_to: List[int] = field(default_factory=list)


@dataclass
class PackThermalResult:
    """Results from pack thermal analysis"""
    times: np.ndarray
    temperatures: np.ndarray  # (time, cell) [K]
    heat_generation: np.ndarray  # (time, cell) [W]
    thermal_runaway_events: List[ThermalRunawayEvent]
    max_temperature: float  # K
    max_temp_delta: float  # K (uniformity)
    cooling_adequate: bool
    cells_in_runaway: int


class BatteryPackThermalModel:
    """
    Pack-level thermal model with cell-to-cell propagation

    Energy balance per cell:
    m·c·dT/dt = Q_generation - Q_cooling - Q_neighbors

    where:
    - Q_generation = I²R + Q_calendar
    - Q_cooling = h·A·(T - T_coolant)
    - Q_neighbors = Σ k_eff·A_contact·(T_i - T_j)
    """

    def __init__(self, config: PackConfiguration):
        """
        Initialize battery pack thermal model

        Parameters:
        -----------
        config : PackConfiguration
            Pack configuration
        """
        self.config = config
        self.cells: List[BatteryCell] = []

        # Create cells in linear arrangement (1D for simplicity)
        for i in range(config.n_cells):
            cell = BatteryCell(
                cell_id=i,
                position=(i * config.cell_spacing, 0, 0)
            )
            self.cells.append(cell)

        # Thermal runaway tracking
        self.runaway_events: List[ThermalRunawayEvent] = []

    def cell_heat_generation(self, cell: BatteryCell) -> float:
        """
        Calculate heat generation in a cell

        Q = I²R + Q_calendar

        Parameters:
        -----------
        cell : BatteryCell

        Returns:
        --------
        float : Heat generation [W]
        """
        # I²R heating (Joule heating)
        Q_joule = cell.current**2 * cell.internal_resistance

        # Calendar aging heat (small, ~0.1 W for automotive cells)
        Q_calendar = 0.1

        return Q_joule + Q_calendar

    def cooling_heat_transfer(self, cell: BatteryCell) -> float:
        """
        Heat removed by cooling system

        Q_cool = h·A·(T_cell - T_coolant)

        Parameters:
        -----------
        cell : BatteryCell

        Returns:
        --------
        float : Heat removed [W]
        """
        return self.config.h_convection * cell.surface_area * (cell.temperature - self.config.T_coolant)

    def cell_to_cell_heat_transfer(self, cell_i: BatteryCell, cell_j: BatteryCell) -> float:
        """
        Heat transfer between adjacent cells

        Q_ij = k_eff·A_contact·(T_i - T_j) / distance

        Parameters:
        -----------
        cell_i, cell_j : BatteryCell

        Returns:
        --------
        float : Heat transfer from i to j [W]
        """
        # Effective thermal conductivity (cell casing + air gap)
        k_eff = 5.0  # W/(m·K) (conservative for cell-to-cell)

        # Contact area (approximate as cell height × width)
        A_contact = 0.001  # m² (small contact area)

        # Distance between cell centers
        distance = self.config.cell_spacing

        # Thermal resistance
        R_thermal = distance / (k_eff * A_contact)  # K/W

        Q_ij = (cell_i.temperature - cell_j.temperature) / R_thermal

        return Q_ij

    def check_thermal_runaway_trigger(self, cell: BatteryCell, t: float) -> bool:
        """
        Check if cell triggers thermal runaway

        Trigger conditions:
        1. Temperature > 130°C (403 K) for NMC/NCA
        2. Temperature > 150°C (423 K) for LFP
        3. Or: Temperature rate > 10 K/min

        Parameters:
        -----------
        cell : BatteryCell
        t : float
            Current time [s]

        Returns:
        --------
        bool : True if thermal runaway triggered
        """
        if cell.in_thermal_runaway:
            return False  # Already in runaway

        # Temperature threshold (NMC chemistry assumed)
        T_runaway_threshold = 403.0  # K (130°C)

        if cell.temperature > T_runaway_threshold:
            cell.in_thermal_runaway = True
            cell.runaway_triggered_time = t

            # Record event
            event = ThermalRunawayEvent(
                cell_id=cell.cell_id,
                trigger_time=t,
                trigger_temperature=cell.temperature,
                peak_temperature=cell.temperature
            )
            self.runaway_events.append(event)

            return True

        return False

    def thermal_runaway_heat_release(self, cell: BatteryCell, t: float) -> float:
        """
        Heat release during thermal runaway

        Simplified model: Exponential heat release
        Q_runaway(t) = Q_max · [1 - exp(-t/τ)]

        Parameters:
        -----------
        cell : BatteryCell
        t : float
            Time since runaway trigger [s]

        Returns:
        --------
        float : Heat release rate [W]
        """
        if not cell.in_thermal_runaway:
            return 0.0

        t_since_trigger = t - cell.runaway_triggered_time

        if t_since_trigger < 0:
            return 0.0

        # Peak heat release (typical: 50-100 kW for 50 Ah automotive cell)
        Q_max = 80000.0  # W

        # Time constant (runaway develops over ~10-30 seconds)
        tau = 15.0  # s

        # Exponential rise
        Q_runaway = Q_max * (1 - np.exp(-t_since_trigger / tau))

        # After peak, decay exponentially
        if t_since_trigger > 5 * tau:
            Q_runaway = Q_max * np.exp(-(t_since_trigger - 5*tau) / (2*tau))

        return Q_runaway

    def solve_transient(self, current_profile: np.ndarray,
                        t_end: float, dt: float = 0.1,
                        progress_callback=None) -> PackThermalResult:
        """
        Solve transient thermal response of battery pack

        Parameters:
        -----------
        current_profile : np.ndarray
            Current vs time [A] (same for all cells, simplified)
        t_end : float
            End time [s]
        dt : float
            Time step [s]

        Returns:
        --------
        PackThermalResult
        """
        n_steps = int(t_end / dt)
        times = np.linspace(0, t_end, n_steps + 1)

        # Storage
        T_history = np.zeros((n_steps + 1, self.config.n_cells))
        Q_history = np.zeros((n_steps + 1, self.config.n_cells))

        # Initial conditions
        for i, cell in enumerate(self.cells):
            T_history[0, i] = cell.temperature

        # Time integration
        _prog_interval = max(n_steps // 50, 1)
        for step in range(n_steps):
            t = times[step]

            # Update currents (interpolate current profile)
            if current_profile is not None and len(current_profile) > 0:
                idx = min(step, len(current_profile) - 1)
                for cell in self.cells:
                    cell.current = current_profile[idx]

            # Calculate heat generation and transfer for each cell
            for i, cell in enumerate(self.cells):
                # Check thermal runaway trigger
                self.check_thermal_runaway_trigger(cell, t)

                # Heat generation
                Q_gen = self.cell_heat_generation(cell)

                # Thermal runaway heat
                if cell.in_thermal_runaway:
                    Q_gen += self.thermal_runaway_heat_release(cell, t)

                Q_history[step, i] = Q_gen

                # Cooling
                Q_cool = self.cooling_heat_transfer(cell)

                # Cell-to-cell heat transfer
                Q_neighbors = 0.0
                if i > 0:  # Left neighbor
                    Q_neighbors += self.cell_to_cell_heat_transfer(cell, self.cells[i-1])
                if i < self.config.n_cells - 1:  # Right neighbor
                    Q_neighbors -= self.cell_to_cell_heat_transfer(self.cells[i+1], cell)

                # Energy balance: m·c·dT/dt = Q_gen - Q_cool - Q_neighbors
                dT_dt = (Q_gen - Q_cool - Q_neighbors) / (cell.mass_kg * cell.specific_heat)

                # Update temperature
                cell.temperature += dT_dt * dt
                T_history[step + 1, i] = cell.temperature

            if progress_callback and step % _prog_interval == 0:
                progress_callback(step, n_steps)

        # Analysis
        max_T = np.max(T_history)
        max_delta_T = np.max(T_history[-1, :]) - np.min(T_history[-1, :])

        # Count cells in runaway
        cells_in_runaway = sum(1 for cell in self.cells if cell.in_thermal_runaway)

        # Cooling adequacy: max temp < 60°C (333 K)
        cooling_adequate = max_T < 333.0

        return PackThermalResult(
            times=times,
            temperatures=T_history,
            heat_generation=Q_history,
            thermal_runaway_events=self.runaway_events,
            max_temperature=max_T,
            max_temp_delta=max_delta_T,
            cooling_adequate=cooling_adequate,
            cells_in_runaway=cells_in_runaway
        )


def evaluate_cooling_adequacy(pack_model: BatteryPackThermalModel,
                              current_A: float, duration_s: float) -> Dict:
    """
    Evaluate if cooling system is adequate for given load

    Parameters:
    -----------
    pack_model : BatteryPackThermalModel
    current_A : float
        Sustained current per cell [A]
    duration_s : float
        Duration [s]

    Returns:
    --------
    dict : Cooling evaluation results
    """
    # Constant current profile
    n_steps = int(duration_s / 0.1)
    current_profile = np.ones(n_steps) * current_A

    result = pack_model.solve_transient(current_profile, duration_s, dt=0.1)

    # Evaluation criteria
    T_max_safe = 333.0  # 60°C
    T_delta_max = 5.0  # 5 K uniformity requirement

    return {
        "cooling_adequate": result.cooling_adequate,
        "max_temperature_K": result.max_temperature,
        "max_temperature_C": result.max_temperature - 273.15,
        "temperature_uniformity_K": result.max_temp_delta,
        "meets_temp_limit": result.max_temperature < T_max_safe,
        "meets_uniformity": result.max_temp_delta < T_delta_max,
        "overall_pass": (result.max_temperature < T_max_safe and
                         result.max_temp_delta < T_delta_max),
        "result": result
    }
