"""
Multi-Physics Time Coupling Infrastructure

Provides the coupling engine that coordinates multiple physics solvers
running at different time scales. Handles:

- Master time-stepping with global dt control
- Sub-cycling for fast solvers within slow solver steps
- Multi-rate integration (different dt per solver)
- Field exchange at synchronization points
- Under-relaxation and Aitken acceleration for partitioned iteration
- Convergence monitoring across coupled solvers

Architecture:
    The CouplingEngine orchestrates a set of SolverAdapter objects. Each
    adapter wraps a physics solver and provides:
    - advance(dt) -> PhysicsState
    - export_state() -> PhysicsState
    - import_state(PhysicsState)

    At each synchronization point the engine:
    1. Collects exported states from all solvers
    2. Exchanges fields according to the coupling map
    3. Iterates (Gauss-Seidel or Jacobi) until convergence
    4. Advances to the next synchronization point
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple, Any, Protocol, runtime_checkable
from enum import Enum

from triality.core.fields import PhysicsState, PhysicsField, CANONICAL_FIELDS


# ---------------------------------------------------------------------------
# Coupling configuration
# ---------------------------------------------------------------------------

class CouplingStrategy(Enum):
    """Strategy for partitioned coupling iteration."""
    GAUSS_SEIDEL = "gauss_seidel"   # Sequential: solve A, pass to B, solve B, pass to A
    JACOBI = "jacobi"               # Parallel: solve A and B simultaneously, exchange
    ONE_WAY = "one_way"             # No iteration: A -> B only


class RelaxationMethod(Enum):
    """Under-relaxation method for coupling iterations."""
    CONSTANT = "constant"           # Fixed omega
    AITKEN = "aitken"               # Aitken adaptive acceleration


@dataclass
class CouplingLink:
    """Defines a field transfer between two solvers.

    Attributes
    ----------
    source_solver : str
        Name of the solver providing the field.
    target_solver : str
        Name of the solver receiving the field.
    source_field : str
        Canonical field name in the source.
    target_field : str
        Canonical field name in the target (often same as source).
    transform : callable, optional
        Optional transformation f(source_data) -> target_data.
    interpolate : bool
        Whether to interpolate between different grids.
    """
    source_solver: str
    target_solver: str
    source_field: str
    target_field: str = ""  # defaults to source_field
    transform: Optional[Callable] = None
    interpolate: bool = True

    def __post_init__(self):
        if not self.target_field:
            self.target_field = self.source_field


# ---------------------------------------------------------------------------
# Solver adapter protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class SolverAdapterProtocol(Protocol):
    """Protocol for wrapping a physics solver for coupling."""

    @property
    def name(self) -> str: ...

    @property
    def time(self) -> float: ...

    def advance(self, dt: float) -> PhysicsState:
        """Advance the solver by dt and return updated state."""
        ...

    def export_state(self) -> PhysicsState:
        """Export current solver state as PhysicsState."""
        ...

    def import_state(self, state: PhysicsState) -> None:
        """Import external fields into the solver."""
        ...

    def get_max_dt(self) -> float:
        """Return the maximum stable time step for this solver."""
        ...


class SolverAdapter:
    """Base adapter wrapping a physics solver for coupling.

    Subclass this and implement the abstract methods for each specific
    solver you want to couple.

    Parameters
    ----------
    name : str
        Unique solver name for the coupling engine.
    solver : object
        The underlying physics solver.
    max_dt : float
        Maximum stable time step [s].
    priority : int
        Solve order (lower = solved first in Gauss-Seidel).
    """

    def __init__(self, name: str, solver: Any, max_dt: float = 1.0,
                 priority: int = 0):
        self._name = name
        self.solver = solver
        self._max_dt = max_dt
        self.priority = priority
        self._time = 0.0
        self._state: Optional[PhysicsState] = None
        self._imported: Dict[str, PhysicsField] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def time(self) -> float:
        return self._time

    def advance(self, dt: float) -> PhysicsState:
        """Advance solver by dt. Override in subclass."""
        raise NotImplementedError

    def export_state(self) -> PhysicsState:
        """Export current state. Override in subclass."""
        if self._state is not None:
            return self._state
        raise NotImplementedError

    def import_state(self, state: PhysicsState) -> None:
        """Import fields from another solver's state."""
        for name, fld in state.fields.items():
            self._imported[name] = fld

    def get_imported_field(self, name: str) -> Optional[PhysicsField]:
        """Retrieve a previously imported field."""
        return self._imported.get(name)

    def get_max_dt(self) -> float:
        return self._max_dt


# ---------------------------------------------------------------------------
# Under-relaxation
# ---------------------------------------------------------------------------

class Relaxation:
    """Under-relaxation with optional Aitken acceleration."""

    def __init__(self, omega_init: float = 0.5,
                 method: RelaxationMethod = RelaxationMethod.CONSTANT,
                 omega_min: float = 0.01, omega_max: float = 1.0):
        self.omega = omega_init
        self.method = method
        self.omega_min = omega_min
        self.omega_max = omega_max
        self._prev_residual: Optional[np.ndarray] = None
        self._prev_delta: Optional[np.ndarray] = None

    def relax(self, x_new: np.ndarray, x_old: np.ndarray) -> np.ndarray:
        """Apply under-relaxation: x_relaxed = omega * x_new + (1-omega) * x_old."""
        delta = x_new - x_old

        if self.method == RelaxationMethod.AITKEN and self._prev_delta is not None:
            # Aitken acceleration
            ddelta = delta - self._prev_delta
            denom = np.dot(ddelta.ravel(), ddelta.ravel())
            if denom > 1e-30:
                self.omega = -self.omega * float(
                    np.dot(self._prev_delta.ravel(), ddelta.ravel()) / denom
                )
                self.omega = np.clip(self.omega, self.omega_min, self.omega_max)

        self._prev_delta = delta.copy()
        return x_old + self.omega * delta

    def reset(self):
        """Reset Aitken state for a new time step."""
        self._prev_delta = None


# ---------------------------------------------------------------------------
# Convergence monitor
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceInfo:
    """Convergence information for a coupling iteration."""
    iteration: int = 0
    residual_norm: float = float('inf')
    field_residuals: Dict[str, float] = field(default_factory=dict)
    converged: bool = False


class ConvergenceMonitor:
    """Monitors convergence of coupling iterations.

    Parameters
    ----------
    tolerance : float
        Absolute convergence tolerance.
    rtol : float
        Relative convergence tolerance.
    max_iterations : int
        Maximum coupling iterations per time step.
    """

    def __init__(self, tolerance: float = 1e-4, rtol: float = 1e-3,
                 max_iterations: int = 50):
        self.tolerance = tolerance
        self.rtol = rtol
        self.max_iterations = max_iterations
        self._prev_fields: Dict[str, np.ndarray] = {}
        self._iteration = 0

    def check(self, current_states: Dict[str, PhysicsState],
              monitor_fields: Optional[List[str]] = None) -> ConvergenceInfo:
        """Check convergence by comparing field changes.

        Parameters
        ----------
        current_states : dict
            solver_name -> PhysicsState mapping.
        monitor_fields : list of str, optional
            Canonical field names to monitor. If None, monitors all.

        Returns
        -------
        ConvergenceInfo
        """
        self._iteration += 1
        field_residuals: Dict[str, float] = {}
        max_residual = 0.0

        for solver_name, state in current_states.items():
            for fname, fld in state.fields.items():
                if monitor_fields and fname not in monitor_fields:
                    continue

                key = f"{solver_name}.{fname}"
                if key in self._prev_fields:
                    prev = self._prev_fields[key]
                    if prev.shape == fld.data.shape:
                        diff = np.max(np.abs(fld.data - prev))
                        scale = max(np.max(np.abs(fld.data)), 1e-30)
                        rel_diff = diff / scale
                        field_residuals[key] = float(rel_diff)
                        max_residual = max(max_residual, rel_diff)

                self._prev_fields[key] = fld.data.copy()

        converged = (max_residual < self.rtol or
                     max_residual < self.tolerance) and self._iteration > 1

        return ConvergenceInfo(
            iteration=self._iteration,
            residual_norm=max_residual,
            field_residuals=field_residuals,
            converged=converged,
        )

    def reset(self):
        """Reset for a new time step."""
        self._prev_fields.clear()
        self._iteration = 0


# ---------------------------------------------------------------------------
# Sub-cycling engine
# ---------------------------------------------------------------------------

class SubCycler:
    """Runs a fast solver with sub-steps within a slow solver's time step.

    Parameters
    ----------
    adapter : SolverAdapter
        The fast solver adapter.
    """

    def __init__(self, adapter: SolverAdapter):
        self.adapter = adapter

    def advance(self, dt_outer: float) -> PhysicsState:
        """Advance the fast solver from t to t + dt_outer using sub-steps.

        The sub-step size is determined by the solver's max_dt.

        Parameters
        ----------
        dt_outer : float
            The outer (slow) time step to cover.

        Returns
        -------
        PhysicsState
            State at t + dt_outer.
        """
        dt_inner = self.adapter.get_max_dt()
        if dt_inner >= dt_outer:
            return self.adapter.advance(dt_outer)

        n_sub = max(int(np.ceil(dt_outer / dt_inner)), 1)
        dt_sub = dt_outer / n_sub

        state = None
        for _ in range(n_sub):
            state = self.adapter.advance(dt_sub)

        return state


# ---------------------------------------------------------------------------
# Field exchange
# ---------------------------------------------------------------------------

class FieldExchange:
    """Handles field transfer between solvers according to coupling links.

    Performs:
    - Field extraction from source state
    - Optional grid interpolation
    - Optional transformation
    - Injection into target solver
    """

    def __init__(self, links: List[CouplingLink]):
        self.links = links

    def exchange(self, states: Dict[str, PhysicsState],
                 adapters: Dict[str, SolverAdapter]) -> None:
        """Execute all field transfers.

        Parameters
        ----------
        states : dict
            solver_name -> PhysicsState mapping.
        adapters : dict
            solver_name -> SolverAdapter mapping.
        """
        for link in self.links:
            if link.source_solver not in states:
                continue
            if link.target_solver not in adapters:
                continue

            src_state = states[link.source_solver]
            if link.source_field not in src_state.fields:
                continue

            src_field = src_state.get(link.source_field)

            # Apply transformation
            if link.transform is not None:
                transformed_data = link.transform(src_field.data)
                target_field = PhysicsField(
                    name=link.target_field,
                    data=transformed_data,
                    unit=src_field.unit,
                    grid=src_field.grid,
                    time=src_field.time,
                )
            else:
                target_field = PhysicsField(
                    name=link.target_field,
                    data=src_field.data.copy(),
                    unit=src_field.unit,
                    grid=src_field.grid,
                    time=src_field.time,
                )

            # Interpolation to target grid
            if link.interpolate:
                tgt_adapter = adapters[link.target_solver]
                tgt_state = states.get(link.target_solver)
                if tgt_state and link.target_field in tgt_state.fields:
                    tgt_grid = tgt_state.get(link.target_field).grid
                    if tgt_grid is not None and target_field.grid is not None:
                        if target_field.data.ndim == 1 and len(tgt_grid) != len(target_field.data):
                            target_field = target_field.interpolate_to(tgt_grid)

            # Build a state containing just this field
            exchange_state = PhysicsState(
                solver_name=link.source_solver,
                time=src_field.time or 0.0,
            )
            exchange_state.fields[link.target_field] = target_field

            adapters[link.target_solver].import_state(exchange_state)


# ---------------------------------------------------------------------------
# Master coupling engine
# ---------------------------------------------------------------------------

@dataclass
class CouplingResult:
    """Result from a coupled multi-physics simulation.

    Attributes
    ----------
    times : list of float
        Time stamps at synchronization points.
    states : dict
        solver_name -> list of PhysicsState (one per sync point).
    convergence_history : list of ConvergenceInfo
        Convergence info at each sync point.
    total_iterations : int
        Total coupling iterations across all time steps.
    """
    times: List[float] = field(default_factory=list)
    states: Dict[str, List[PhysicsState]] = field(default_factory=dict)
    convergence_history: List[ConvergenceInfo] = field(default_factory=list)
    total_iterations: int = 0


class CouplingEngine:
    """Master multi-physics coupling engine.

    Orchestrates multiple physics solvers, handling:
    - Time synchronization
    - Sub-cycling for fast solvers
    - Partitioned iteration with under-relaxation
    - Convergence monitoring

    Parameters
    ----------
    strategy : CouplingStrategy
        Coupling iteration strategy.
    relaxation_omega : float
        Under-relaxation factor.
    relaxation_method : RelaxationMethod
        Under-relaxation method.
    convergence_tol : float
        Coupling convergence tolerance.
    max_coupling_iter : int
        Maximum coupling iterations per sync point.
    dt_sync : float
        Synchronization time step (outer dt).

    Example
    -------
    >>> engine = CouplingEngine(strategy=CouplingStrategy.GAUSS_SEIDEL)
    >>> engine.add_solver(thermal_adapter)
    >>> engine.add_solver(structural_adapter)
    >>> engine.add_link(CouplingLink("thermal", "structural", "temperature"))
    >>> engine.add_link(CouplingLink("structural", "thermal", "displacement"))
    >>> result = engine.run(t_end=1.0, dt_sync=0.01)
    """

    def __init__(
        self,
        strategy: CouplingStrategy = CouplingStrategy.GAUSS_SEIDEL,
        relaxation_omega: float = 0.5,
        relaxation_method: RelaxationMethod = RelaxationMethod.CONSTANT,
        convergence_tol: float = 1e-4,
        max_coupling_iter: int = 50,
        dt_sync: float = 0.01,
    ):
        self.strategy = strategy
        self.adapters: Dict[str, SolverAdapter] = {}
        self.links: List[CouplingLink] = []
        self.sub_cyclers: Dict[str, SubCycler] = {}

        self._relaxation = Relaxation(
            omega_init=relaxation_omega,
            method=relaxation_method,
        )
        self._monitor = ConvergenceMonitor(
            tolerance=convergence_tol,
            max_iterations=max_coupling_iter,
        )
        self._exchange: Optional[FieldExchange] = None
        self.dt_sync = dt_sync
        self.max_coupling_iter = max_coupling_iter

    def add_solver(self, adapter: SolverAdapter, sub_cycle: bool = False) -> None:
        """Register a solver adapter.

        Parameters
        ----------
        adapter : SolverAdapter
            The solver adapter to add.
        sub_cycle : bool
            If True, this solver will be sub-cycled within the sync step.
        """
        self.adapters[adapter.name] = adapter
        if sub_cycle:
            self.sub_cyclers[adapter.name] = SubCycler(adapter)

    def add_link(self, link: CouplingLink) -> None:
        """Add a field transfer link between solvers."""
        self.links.append(link)

    def run(self, t_end: float, dt_sync: Optional[float] = None,
            save_interval: int = 1,
            monitor_fields: Optional[List[str]] = None) -> CouplingResult:
        """Run the coupled simulation.

        Parameters
        ----------
        t_end : float
            End time [s].
        dt_sync : float, optional
            Override synchronization time step.
        save_interval : int
            Save states every N sync steps.
        monitor_fields : list of str, optional
            Canonical field names to monitor for convergence.

        Returns
        -------
        CouplingResult
        """
        dt = dt_sync or self.dt_sync
        n_steps = max(int(np.ceil(t_end / dt)), 1)
        dt_actual = t_end / n_steps

        self._exchange = FieldExchange(self.links)

        result = CouplingResult()
        result.states = {name: [] for name in self.adapters}

        # Sort solvers by priority for Gauss-Seidel ordering
        solve_order = sorted(self.adapters.keys(),
                             key=lambda n: self.adapters[n].priority)

        t = 0.0
        result.times.append(t)
        # Save initial states
        for name in self.adapters:
            try:
                state = self.adapters[name].export_state()
                result.states[name].append(state)
            except NotImplementedError:
                pass

        for step in range(n_steps):
            t_new = (step + 1) * dt_actual
            self._monitor.reset()
            self._relaxation.reset()

            # Coupling iteration loop
            for c_iter in range(self.max_coupling_iter):
                states: Dict[str, PhysicsState] = {}

                if self.strategy == CouplingStrategy.GAUSS_SEIDEL:
                    # Sequential: solve in order, exchanging after each
                    for name in solve_order:
                        adapter = self.adapters[name]

                        # Exchange available fields to this solver
                        self._exchange.exchange(states, self.adapters)

                        # Advance (with sub-cycling if configured)
                        if name in self.sub_cyclers:
                            state = self.sub_cyclers[name].advance(dt_actual)
                        else:
                            state = adapter.advance(dt_actual)

                        states[name] = state

                elif self.strategy == CouplingStrategy.JACOBI:
                    # Parallel: advance all, then exchange
                    for name in solve_order:
                        adapter = self.adapters[name]
                        if name in self.sub_cyclers:
                            state = self.sub_cyclers[name].advance(dt_actual)
                        else:
                            state = adapter.advance(dt_actual)
                        states[name] = state

                    # Exchange after all solvers have advanced
                    self._exchange.exchange(states, self.adapters)

                else:
                    # ONE_WAY: single pass, no iteration
                    for name in solve_order:
                        adapter = self.adapters[name]
                        self._exchange.exchange(states, self.adapters)
                        if name in self.sub_cyclers:
                            state = self.sub_cyclers[name].advance(dt_actual)
                        else:
                            state = adapter.advance(dt_actual)
                        states[name] = state
                    break

                # Check convergence
                conv = self._monitor.check(states, monitor_fields)
                if conv.converged or self.strategy == CouplingStrategy.ONE_WAY:
                    result.total_iterations += conv.iteration
                    result.convergence_history.append(conv)
                    break

            # End of coupling iteration
            t = t_new

            if (step + 1) % save_interval == 0 or step == n_steps - 1:
                result.times.append(t)
                for name in self.adapters:
                    if name in states:
                        result.states[name].append(states[name])

        return result
