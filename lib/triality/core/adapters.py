"""
Generic Solver Adapters for the CouplingEngine

Provides ready-to-use SolverAdapter implementations that wrap any triality
solver conforming to the M3 coupling interface (import_state / advance /
export_state).  Also includes an adapter factory that auto-wraps solvers.

Usage
-----
>>> from triality.core.adapters import GenericAdapter, AdapterFactory
>>> adapter = GenericAdapter("thermal", th_solver, max_dt=0.01)
>>> engine.add_solver(adapter)

Or with the factory:
>>> adapter = AdapterFactory.wrap(th_solver, name="thermal", max_dt=0.01)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from triality.core.coupling import SolverAdapter
from triality.core.fields import PhysicsState


class GenericAdapter(SolverAdapter):
    """Wraps any solver that implements import_state / advance / export_state.

    This is the universal adapter — it delegates all coupling operations
    directly to the underlying solver's methods.

    Parameters
    ----------
    name : str
        Unique solver name for the CouplingEngine.
    solver : object
        Must implement ``advance(dt) -> PhysicsState``,
        ``export_state(...) -> PhysicsState``, and
        ``import_state(state: PhysicsState) -> None``.
    max_dt : float
        Maximum stable time step [s].
    priority : int
        Solve order in Gauss-Seidel (lower = first).
    """

    def __init__(self, name: str, solver: Any, max_dt: float = 1.0,
                 priority: int = 0):
        super().__init__(name, solver, max_dt, priority)

    def advance(self, dt: float) -> PhysicsState:
        """Advance the underlying solver by *dt*.

        Imports any previously received coupled fields into the solver,
        then calls ``solver.advance(dt)``.
        """
        # Push any fields we received from the CouplingEngine
        if self._imported:
            exchange = PhysicsState(solver_name=self._name, time=self._time)
            exchange.fields.update(self._imported)
            self.solver.import_state(exchange)

        state = self.solver.advance(dt)
        self._time += dt
        self._state = state
        return state

    def export_state(self) -> PhysicsState:
        if self._state is not None:
            return self._state
        # Solver may expose a no-arg export_state for initial condition
        return self.solver.export_state()

    def import_state(self, state: PhysicsState) -> None:
        """Cache fields; they are pushed into the solver on the next advance."""
        super().import_state(state)


class SteadyAdapter(SolverAdapter):
    """Adapter for steady-state solvers (no internal time evolution).

    Each ``advance(dt)`` call re-solves the steady problem with whatever
    coupled fields have been imported as boundary conditions.

    The wrapped solver must implement:
    - ``import_state(state)``
    - ``solve(**kwargs) -> result``
    - ``export_state(result) -> PhysicsState``

    Parameters
    ----------
    name : str
        Unique solver name.
    solver : object
        The steady-state solver.
    solve_kwargs : dict
        Extra keyword arguments forwarded to ``solver.solve()`` each call.
    max_dt : float
        Pseudo time step (steady solvers are not dt-limited, but this
        controls how often they re-solve within sub-cycling).
    priority : int
        Solve order.
    """

    def __init__(self, name: str, solver: Any,
                 solve_kwargs: Optional[Dict[str, Any]] = None,
                 max_dt: float = 1.0, priority: int = 0):
        super().__init__(name, solver, max_dt, priority)
        self._solve_kwargs = solve_kwargs or {}

    def advance(self, dt: float) -> PhysicsState:
        # Push coupled fields
        if self._imported:
            exchange = PhysicsState(solver_name=self._name, time=self._time)
            exchange.fields.update(self._imported)
            self.solver.import_state(exchange)

        result = self.solver.solve(**self._solve_kwargs)
        self._time += dt
        self._state = self.solver.export_state(result)
        return self._state

    def export_state(self) -> PhysicsState:
        if self._state is not None:
            return self._state
        raise RuntimeError(f"SteadyAdapter '{self._name}': no state yet — call advance first")


class AdapterFactory:
    """Auto-wraps a triality solver in the appropriate SolverAdapter.

    Inspects the solver to determine the best adapter type:
    - If it has ``advance(dt)``: uses :class:`GenericAdapter`
    - Otherwise if it has ``solve()``: uses :class:`SteadyAdapter`

    Example
    -------
    >>> adapter = AdapterFactory.wrap(my_solver, name="cfd", max_dt=1e-3)
    """

    @staticmethod
    def wrap(solver: Any, name: str, max_dt: float = 1.0,
             priority: int = 0,
             solve_kwargs: Optional[Dict[str, Any]] = None) -> SolverAdapter:
        """Wrap *solver* in the most appropriate adapter.

        Parameters
        ----------
        solver : object
            Any triality solver.
        name : str
            Unique solver name for the CouplingEngine.
        max_dt : float
            Maximum stable time step [s].
        priority : int
            Solve order (lower = first).
        solve_kwargs : dict, optional
            Extra kwargs for steady solvers.

        Returns
        -------
        SolverAdapter
        """
        has_advance = hasattr(solver, 'advance') and callable(getattr(solver, 'advance'))
        has_import = hasattr(solver, 'import_state') and callable(getattr(solver, 'import_state'))

        if has_advance and has_import:
            return GenericAdapter(name, solver, max_dt=max_dt, priority=priority)

        has_solve = hasattr(solver, 'solve') and callable(getattr(solver, 'solve'))
        if has_solve and has_import:
            return SteadyAdapter(name, solver, solve_kwargs=solve_kwargs,
                                 max_dt=max_dt, priority=priority)

        raise TypeError(
            f"Solver {type(solver).__name__} does not conform to the M3 "
            f"coupling interface. Required: import_state() + advance(dt) "
            f"or import_state() + solve()."
        )
