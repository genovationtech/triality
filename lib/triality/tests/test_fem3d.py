import numpy as np
import pytest

from triality.solvers.rust_backend import rust_available


@pytest.mark.skipif(not rust_available(), reason="triality_engine not available")
def test_tet4_patch_constant_solution():
    import triality_engine as te

    nodes = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    elems = [(0, 1, 2, 3)]
    mesh = te.Mesh3D(nodes, elems)

    assembled = te.fem3d_assemble_poisson(mesh, source=0.0)
    A = assembled["stiffness"]
    b = np.asarray(assembled["rhs"], dtype=float)

    # all Dirichlet = 1 should produce exact constant solution 1
    bc = te.fem3d_apply_dirichlet(A, b, [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])
    out = te.solve_linear(bc["matrix"], np.asarray(bc["rhs"], dtype=np.float64), method="cg", precond="jacobi", tol=1e-12, max_iter=200)
    u = np.asarray(out["x"])
    assert np.allclose(u, 1.0, atol=1e-10)


@pytest.mark.skipif(not rust_available(), reason="triality_engine not available")
def test_fem3d_python_wrapper_exports_vtu():
    from triality.fem3d import Mesh, PoissonSolver3D

    mesh = Mesh.from_arrays(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)],
        [(0, 1, 2, 3)],
    )
    solver = PoissonSolver3D(mesh)
    u = solver.solve(source=0.0, dirichlet_dofs=[0, 1, 2, 3], dirichlet_values=[0.0, 0.0, 0.0, 0.0])
    vtu = solver.export_vtu(u, field_name="u")
    assert "<VTKFile" in vtu
    assert "PointData" in vtu


@pytest.mark.skipif(not rust_available(), reason="triality_engine not available")
def test_tet10_and_hex8_assembly_paths():
    import triality_engine as te

    tet10_nodes = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ]
    tet10 = te.Mesh3D.from_tet10(tet10_nodes, [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)])
    tet_out = te.fem3d_assemble_poisson(tet10, source=2.0)
    tet_rhs = np.asarray(tet_out["rhs"], dtype=float)
    assert tet_rhs.size == 10
    assert np.isclose(tet_rhs.sum(), 2.0 / 6.0, atol=1e-10)

    hex_nodes = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ]
    hex8 = te.Mesh3D.from_hex8(hex_nodes, [(0, 1, 2, 3, 4, 5, 6, 7)])
    hex_out = te.fem3d_assemble_poisson(hex8, source=3.0)
    hex_rhs = np.asarray(hex_out["rhs"], dtype=float)
    assert hex_rhs.size == 8
    assert np.isclose(hex_rhs.sum(), 3.0, atol=1e-10)


@pytest.mark.skipif(not rust_available(), reason="triality_engine not available")
def test_tet10_vtu_export():
    """Tet10 elements should export to VTU with VTK_QUADRATIC_TETRA (type 24)."""
    from triality.fem3d import Mesh, PoissonSolver3D

    tet10_nodes = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ]
    mesh = Mesh.from_tet10_arrays(tet10_nodes, [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)])
    solver = PoissonSolver3D(mesh)
    u = solver.solve(
        source=0.0,
        dirichlet_dofs=list(range(10)),
        dirichlet_values=[1.0] * 10,
    )
    vtu = solver.export_vtu(u, field_name="phi")
    assert "<VTKFile" in vtu
    assert "PointData" in vtu
    assert "phi" in vtu
    # VTK_QUADRATIC_TETRA type code = 24, should appear in the types array
    assert "24" in vtu
