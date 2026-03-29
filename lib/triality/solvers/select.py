"""
Automatic solver selection.

Chooses optimal numerical methods based on problem classification.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class SolverPlan:
    """Complete solver strategy"""
    discretization: str  # 'fdm', 'fem', 'spectral'
    linear_solver: str  # 'direct', 'cg', 'gmres'
    preconditioner: str  # 'none', 'jacobi', 'ilu'
    backend: str  # 'numpy', 'jax', 'petsc'
    reasoning: List[str]

    def __repr__(self):
        lines = [
            "=== Solver Plan ===",
            f"Discretization: {self.discretization}",
            f"Linear Solver: {self.linear_solver}",
        ]
        if self.preconditioner != 'none':
            lines.append(f"Preconditioner: {self.preconditioner}")
        lines.append(f"Backend: {self.backend}")

        if self.reasoning:
            lines.append("\nReasoning:")
            for r in self.reasoning:
                lines.append(f"  • {r}")

        return "\n".join(lines)


def select_solver(classification, size_estimate=None):
    """
    Select optimal solver strategy based on problem classification.

    Args:
        classification: Problem classification
        size_estimate: Rough number of DOFs

    Returns:
        SolverPlan with chosen methods and reasoning
    """

    reasoning = []

    # Discretization
    if classification.pde_type == 'elliptic' and classification.is_linear:
        discretization = 'fdm'
        reasoning.append("FDM: standard choice for elliptic PDEs on regular grids")
    else:
        discretization = 'fdm'
        reasoning.append("FDM: default choice")

    # Linear solver
    if size_estimate and size_estimate < 10000:
        linear_solver = 'direct'
        precond = 'none'
        reasoning.append("Direct solver: problem size < 10k DOFs")
    elif classification.is_linear and classification.has_laplacian:
        # Symmetric positive definite
        linear_solver = 'cg'
        precond = 'none' if (not size_estimate or size_estimate < 50000) else 'ilu'
        reasoning.append("CG: symmetric positive definite operator")
        if precond == 'ilu':
            reasoning.append("ILU preconditioner: large problem")
    else:
        linear_solver = 'gmres'
        precond = 'none' if (not size_estimate or size_estimate < 50000) else 'ilu'
        reasoning.append("GMRES: general iterative solver")
        if precond == 'ilu':
            reasoning.append("ILU preconditioner: large problem")

    # Backend - prefer Rust when available
    try:
        from triality.solvers.rust_backend import rust_available
        has_rust = rust_available()
    except ImportError:
        has_rust = False

    if size_estimate and size_estimate > 1000000:
        backend = 'petsc'
        reasoning.append("PETSc: very large problem (>1M DOFs)")
    elif has_rust:
        backend = 'rust'
        reasoning.append("Rust (triality_engine): native performance with multithreading")
    elif size_estimate and size_estimate > 100000:
        backend = 'jax'
        reasoning.append("JAX: GPU acceleration for medium-large problem")
    else:
        backend = 'numpy'
        reasoning.append("NumPy: suitable for problem size")

    return SolverPlan(
        discretization=discretization,
        linear_solver=linear_solver,
        preconditioner=precond,
        backend=backend,
        reasoning=reasoning
    )
