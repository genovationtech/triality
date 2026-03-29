"""
Triality - Automatic PDE Solver

A focused library that does one thing well:
Solves linear elliptic PDEs automatically by classifying problems
and selecting optimal numerical methods.

Example:
    >>> from triality import *
    >>> u = Field("u")
    >>> sol = solve(Eq(laplacian(u), 1), Interval(0, 1), bc={'left': 0, 'right': 0})
    >>> sol.plot()
"""

from triality.core.expressions import Field, Constant, Equation, Eq, sin, cos, exp, sqrt
from triality.core.expressions import grad, div, laplacian, dx, dy, dt
from triality.core.domains import Interval, Rectangle, Square, Circle
from triality.solvers.classify import classify
from triality.solvers.select import select_solver
from triality.solvers.solve import solve, Solution
from triality.runtime import (
    load_module,
    available_runtime_modules,
    BaseRuntimeSolver,
    RuntimeExecutionResult,
    RuntimeDescription,
    RuntimeFieldSpec,
    RuntimeContractError,
    RuntimeExecutionError,
)
from triality.runtime_graph import (
    RuntimeGraph,
    RuntimeNode,
    RuntimeLink,
    GraphConvergenceCriteria,
    RuntimeNodeResult,
    RuntimeGraphResult,
    merge_physics_states,
)
from triality.runtime_templates import available_runtime_templates, load_runtime_template

__version__ = "0.1.0"

__all__ = [
    # Core types
    'Field', 'Constant', 'Equation', 'Eq', 'sin', 'cos', 'exp', 'sqrt',
    # Operators
    'grad', 'div', 'laplacian', 'dx', 'dy', 'dt',
    # Domains
    'Interval', 'Rectangle', 'Square', 'Circle',
    # Main function
    'solve', 'Solution',
    # Analysis (advanced)
    'classify', 'select_solver',
    # Runtime SDK
    'load_module', 'available_runtime_modules', 'BaseRuntimeSolver',
    'RuntimeExecutionResult', 'RuntimeDescription', 'RuntimeFieldSpec',
    'RuntimeContractError', 'RuntimeExecutionError',
    'RuntimeGraph', 'RuntimeNode', 'RuntimeLink', 'GraphConvergenceCriteria',
    'RuntimeNodeResult', 'RuntimeGraphResult', 'merge_physics_states',
    'available_runtime_templates', 'load_runtime_template',
]
