"""3D FEM interface backed by triality_engine when available."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from triality.solvers.rust_backend import get_backend, rust_available


@dataclass
class Mesh:
    nodes: np.ndarray
    elements: np.ndarray
    _rust_mesh: object | None = None

    @classmethod
    def from_arrays(cls, nodes: Sequence[Sequence[float]], elements: Sequence[Sequence[int]]) -> "Mesh":
        nodes_arr = np.asarray(nodes, dtype=float)
        elems_arr = np.asarray(elements, dtype=int)
        rust_mesh = None
        if rust_available():
            import triality_engine as te
            rust_mesh = te.Mesh3D(
                [tuple(row) for row in nodes_arr],
                [tuple(map(int, row)) for row in elems_arr],
            )
        return cls(nodes_arr, elems_arr, rust_mesh)

    @classmethod
    def from_hex8_arrays(cls, nodes: Sequence[Sequence[float]], elements: Sequence[Sequence[int]]) -> "Mesh":
        nodes_arr = np.asarray(nodes, dtype=float)
        elems_arr = np.asarray(elements, dtype=int)
        rust_mesh = None
        if rust_available():
            import triality_engine as te
            rust_mesh = te.Mesh3D.from_hex8(
                [tuple(row) for row in nodes_arr],
                [tuple(map(int, row)) for row in elems_arr],
            )
        return cls(nodes_arr, elems_arr, rust_mesh)

    @classmethod
    def from_tet10_arrays(cls, nodes: Sequence[Sequence[float]], elements: Sequence[Sequence[int]]) -> "Mesh":
        nodes_arr = np.asarray(nodes, dtype=float)
        elems_arr = np.asarray(elements, dtype=int)
        rust_mesh = None
        if rust_available():
            import triality_engine as te
            rust_mesh = te.Mesh3D.from_tet10(
                [tuple(row) for row in nodes_arr],
                [tuple(map(int, row)) for row in elems_arr],
            )
        return cls(nodes_arr, elems_arr, rust_mesh)

    @classmethod
    def from_gmsh_text(cls, msh_text: str) -> "Mesh":
        if not rust_available():
            raise RuntimeError("Gmsh parser requires triality_engine")
        import triality_engine as te
        rust_mesh = te.Mesh3D.from_gmsh_text(msh_text)
        return cls(np.empty((0, 3)), np.empty((0, 4), dtype=int), rust_mesh)


class PoissonSolver3D:
    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    def solve(
        self,
        source: float = 0.0,
        dirichlet_dofs: Iterable[int] | None = None,
        dirichlet_values: Iterable[float] | None = None,
        method: str = "gmres",
        precond: str = "ilu0",
        tol: float = 1e-8,
    ) -> np.ndarray:
        if not rust_available() or self.mesh._rust_mesh is None:
            raise RuntimeError("3D FEM solve currently requires Rust backend")

        import triality_engine as te

        assembled = te.fem3d_assemble_poisson(self.mesh._rust_mesh, source)
        A = assembled["stiffness"]
        b = np.asarray(assembled["rhs"], dtype=np.float64)

        if dirichlet_dofs is not None and dirichlet_values is not None:
            bc_applied = te.fem3d_apply_dirichlet(
                A,
                b,
                [int(i) for i in dirichlet_dofs],
                [float(v) for v in dirichlet_values],
            )
            A = bc_applied["matrix"]
            b = np.asarray(bc_applied["rhs"], dtype=np.float64)

        result = te.solve_linear(A, b, method=method, precond=precond, tol=tol, max_iter=max(2000, 20 * len(b)))
        return np.asarray(result["x"], dtype=np.float64)

    def export_vtu(self, values: Sequence[float], field_name: str = "u") -> str:
        if self.mesh._rust_mesh is None:
            raise RuntimeError("VTU export requires Rust backend")
        import triality_engine as te

        return te.fem3d_export_vtu(self.mesh._rust_mesh, field_name, np.asarray(values, dtype=np.float64))


__all__ = ["Mesh", "PoissonSolver3D"]
