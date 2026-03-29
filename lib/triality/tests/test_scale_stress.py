"""Scale and stress tests for 3D FEM solvers.

Answers three questions:
1. At what DOF count does performance degrade? (1K → 1M elements)
2. Does accuracy hold on irregular meshes with bad element quality?
3. Does memory become a problem?

Requires: gmsh (pip install gmsh)
"""

import gc
import os
import sys
import time
import tempfile
import traceback

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rss_mb():
    """Current process RSS in MB (Linux)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def make_structured_hex_mesh(n):
    """n×n×n structured hex mesh on [0,1]³."""
    nodes, nmap = [], {}
    idx = 0
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                nodes.append((i / n, j / n, k / n))
                nmap[(i, j, k)] = idx
                idx += 1
    hexes = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                hexes.append((
                    nmap[(i, j, k)], nmap[(i + 1, j, k)],
                    nmap[(i + 1, j + 1, k)], nmap[(i, j + 1, k)],
                    nmap[(i, j, k + 1)], nmap[(i + 1, j, k + 1)],
                    nmap[(i + 1, j + 1, k + 1)], nmap[(i, j + 1, k + 1)],
                ))
    bdry = [nid for (i, j, k), nid in nmap.items()
            if i == 0 or i == n or j == 0 or j == n or k == 0 or k == n]
    return nodes, hexes, bdry, nmap


def make_structured_tet_mesh(n):
    """n×n×n structured tet mesh (Kuhn 6-tet conforming)."""
    nodes, nmap = [], {}
    idx = 0
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                nodes.append((i / n, j / n, k / n))
                nmap[(i, j, k)] = idx
                idx += 1
    tets = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                v = [nmap[(i + di, j + dj, k + dk)]
                     for dk in (0, 1) for dj in (0, 1) for di in (0, 1)]
                v0, v1, v2, v3, v4, v5, v6, v7 = v[0], v[1], v[3], v[2], v[4], v[5], v[7], v[6]
                tets += [
                    (v0, v1, v2, v6), (v0, v1, v5, v6), (v0, v4, v5, v6),
                    (v0, v2, v3, v6), (v0, v3, v7, v6), (v0, v4, v7, v6),
                ]
    bdry = [nid for (i, j, k), nid in nmap.items()
            if i == 0 or i == n or j == 0 or j == n or k == 0 or k == n]
    return nodes, tets, bdry, nmap


# ---------------------------------------------------------------------------
# Gmsh mesh generators
# ---------------------------------------------------------------------------

def gmsh_sphere_tet_mesh(target_elems):
    """Mesh a unit sphere.  Returns msh_text string."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("sphere")
    lc = max((25.1 / target_elems) ** (1 / 3.0), 0.005)
    gmsh.model.occ.addSphere(0, 0, 0, 1.0)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.model.mesh.generate(3)
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        path = f.name
    gmsh.write(path)
    gmsh.finalize()
    with open(path) as f:
        msh_text = f.read()
    os.unlink(path)
    return msh_text


def gmsh_ugly_geometry_mesh(target_elems):
    """L-bracket with hole — sharp re-entrant corner, varying element sizes."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("ugly")
    b1 = gmsh.model.occ.addBox(0, 0, 0, 2, 1, 0.3)
    b2 = gmsh.model.occ.addBox(0, 0, 0, 1, 2, 0.3)
    fused = gmsh.model.occ.fuse([(3, b1)], [(3, b2)])
    cyl = gmsh.model.occ.addCylinder(1.5, 0.5, -0.1, 0, 0, 0.5, 0.2)
    gmsh.model.occ.cut(fused[0], [(3, cyl)])
    gmsh.model.occ.synchronize()
    lc = max((1.8 / target_elems) ** (1 / 3.0), 0.003)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.3)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        path = f.name
    gmsh.write(path)
    gmsh.finalize()
    with open(path) as f:
        msh_text = f.read()
    os.unlink(path)
    return msh_text


def gmsh_cylinder_mesh(target_elems):
    """Cylinder r=1, z∈[0,2]."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cylinder")
    gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 2, 1.0)
    gmsh.model.occ.synchronize()
    lc = max((6.28 * 6 / target_elems) ** (1 / 3.0), 0.003)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    gmsh.model.mesh.generate(3)
    with tempfile.NamedTemporaryFile(suffix=".msh", delete=False) as f:
        path = f.name
    gmsh.write(path)
    gmsh.finalize()
    with open(path) as f:
        msh_text = f.read()
    os.unlink(path)
    return msh_text


def parse_gmsh_nodes(msh_text):
    """Return list of (x,y,z) in gmsh node-tag order from Gmsh v4 MSH."""
    lines = msh_text.split("\n")
    nodes = {}
    i = 0
    while i < len(lines):
        if lines[i].strip() == "$Nodes":
            i += 1
            header = lines[i].strip().split()
            n_entity_blocks = int(header[0])
            i += 1
            for _ in range(n_entity_blocks):
                block_header = lines[i].strip().split()
                n_block = int(block_header[3])
                i += 1
                tags = []
                for _ in range(n_block):
                    tags.append(int(lines[i].strip()))
                    i += 1
                for t in range(n_block):
                    coords = lines[i].strip().split()
                    nodes[tags[t]] = (float(coords[0]), float(coords[1]), float(coords[2]))
                    i += 1
            break
        i += 1
    # Return in tag order (1-based → 0-based index)
    max_tag = max(nodes.keys()) if nodes else 0
    result = [(0, 0, 0)] * max_tag
    for tag, xyz in nodes.items():
        result[tag - 1] = xyz
    return result


# ===================================================================
# TEST 1: Scaling — 1K → 1M elements
# ===================================================================

def test_scaling():
    from triality.fem3d import Mesh, PoissonSolver3D
    import triality_engine as te

    print("=" * 90)
    print("TEST 1: SCALING — Assembly & Solve Time vs Element Count")
    print("         Problem: -∇²u = -6, u = x²+y²+z² on boundary (known analytical)")
    print("=" * 90)
    print()
    print(f"{'Type':<6} {'n':>4} {'Elems':>10} {'DOFs':>10} "
          f"{'Asm(s)':>8} {'Slv(s)':>8} {'Iters':>6} "
          f"{'RSS(MB)':>8} {'Error':>10} {'Conv':>5}")
    print("-" * 94)

    # Hex8: n=10→1K, n=22→10K, n=46→100K, n=100→1M
    for n in [10, 22, 46, 100]:
        n_elems = n ** 3
        n_dofs = (n + 1) ** 3
        if n_dofs > 2_500_000:
            break

        gc.collect()
        nodes, hexes, bdry, _ = make_structured_hex_mesh(n)
        mesh = Mesh.from_hex8_arrays(nodes, hexes)
        exact = np.array([x * x + y * y + z * z for (x, y, z) in nodes])

        t0 = time.time()
        assembled = te.fem3d_assemble_poisson(mesh._rust_mesh, -6.0)
        t_asm = time.time() - t0

        A = assembled["stiffness"]
        b = np.asarray(assembled["rhs"], dtype=np.float64)
        bc = te.fem3d_apply_dirichlet(A, b,
                                       [int(i) for i in bdry],
                                       [float(exact[i]) for i in bdry])

        t0 = time.time()
        result = te.solve_linear(bc["matrix"], np.asarray(bc["rhs"], dtype=np.float64),
                                 method="gmres", precond="ilu0",
                                 tol=1e-8, max_iter=min(10000, max(2000, 20 * n_dofs)))
        t_slv = time.time() - t0

        u = np.asarray(result["x"])
        err = np.max(np.abs(u - exact))
        rss = rss_mb()

        print(f"{'Hex8':<6} {n:>4} {n_elems:>10,} {n_dofs:>10,} "
              f"{t_asm:>8.3f} {t_slv:>8.3f} {result['iterations']:>6} "
              f"{rss:>8.0f} {err:>10.2e} {'Y' if result['converged'] else 'N':>5}")

        del mesh, assembled, A, b, bc, u, nodes, hexes, result
        gc.collect()

    print()

    # Tet4: same node counts but 6× more elements
    for n in [10, 22, 46, 100]:
        n_elems = 6 * n ** 3
        n_dofs = (n + 1) ** 3
        if n_dofs > 2_500_000:
            break

        gc.collect()
        nodes, tets, bdry, _ = make_structured_tet_mesh(n)
        mesh = Mesh.from_arrays(nodes, tets)
        exact = np.array([x * x + y * y + z * z for (x, y, z) in nodes])

        t0 = time.time()
        assembled = te.fem3d_assemble_poisson(mesh._rust_mesh, -6.0)
        t_asm = time.time() - t0

        A = assembled["stiffness"]
        b = np.asarray(assembled["rhs"], dtype=np.float64)
        bc = te.fem3d_apply_dirichlet(A, b,
                                       [int(i) for i in bdry],
                                       [float(exact[i]) for i in bdry])

        t0 = time.time()
        result = te.solve_linear(bc["matrix"], np.asarray(bc["rhs"], dtype=np.float64),
                                 method="gmres", precond="ilu0",
                                 tol=1e-8, max_iter=min(10000, max(2000, 20 * n_dofs)))
        t_slv = time.time() - t0

        u = np.asarray(result["x"])
        err = np.max(np.abs(u - exact))
        rss = rss_mb()

        print(f"{'Tet4':<6} {n:>4} {n_elems:>10,} {n_dofs:>10,} "
              f"{t_asm:>8.3f} {t_slv:>8.3f} {result['iterations']:>6} "
              f"{rss:>8.0f} {err:>10.2e} {'Y' if result['converged'] else 'N':>5}")

        del mesh, assembled, A, b, bc, u, nodes, tets, result
        gc.collect()


# ===================================================================
# TEST 2: Preconditioner shootout at 30K DOFs
# ===================================================================

def test_preconditioner_shootout():
    from triality.fem3d import Mesh, PoissonSolver3D
    import triality_engine as te

    print()
    print("=" * 90)
    print("TEST 2: PRECONDITIONER SHOOTOUT — Hex8 30³ = 27K elems, ~30K DOFs")
    print("=" * 90)
    print()

    n = 30
    nodes, hexes, bdry, _ = make_structured_hex_mesh(n)
    mesh = Mesh.from_hex8_arrays(nodes, hexes)
    n_dofs = len(nodes)
    exact = np.array([x * x + y * y + z * z for (x, y, z) in nodes])

    assembled = te.fem3d_assemble_poisson(mesh._rust_mesh, -6.0)
    bc = te.fem3d_apply_dirichlet(
        assembled["stiffness"],
        np.asarray(assembled["rhs"], dtype=np.float64),
        [int(i) for i in bdry],
        [float(exact[i]) for i in bdry])
    A = bc["matrix"]
    b = np.asarray(bc["rhs"], dtype=np.float64)

    print(f"{'Method':<12} {'Precond':<8} {'Time(s)':>8} {'Iters':>6} "
          f"{'Residual':>10} {'Error':>10} {'Conv':>5}")
    print("-" * 68)

    for method in ["cg", "gmres", "bicgstab"]:
        for precond in ["none", "jacobi", "ilu0"]:
            # Skip SSOR — it's O(nnz) per iteration with bad constants
            gc.collect()
            try:
                t0 = time.time()
                result = te.solve_linear(A, b, method=method, precond=precond,
                                         tol=1e-8, max_iter=2000)
                dt = time.time() - t0
                u = np.asarray(result["x"])
                err = np.max(np.abs(u - exact))
                print(f"{method:<12} {precond:<8} {dt:>8.3f} {result['iterations']:>6} "
                      f"{result['residual']:>10.2e} {err:>10.2e} "
                      f"{'Y' if result['converged'] else 'N':>5}")
            except Exception as e:
                print(f"{method:<12} {precond:<8} {'FAIL':>8}   {e}")

    # Also test SSOR but only with CG and low max_iter to show the cost
    print()
    print("  (SSOR with max_iter=500 to show per-iteration cost)")
    for method in ["cg", "gmres"]:
        t0 = time.time()
        result = te.solve_linear(A, b, method=method, precond="ssor",
                                 tol=1e-8, max_iter=500)
        dt = time.time() - t0
        u = np.asarray(result["x"])
        err = np.max(np.abs(u - exact))
        print(f"  {method:<12} {'ssor':<8} {dt:>8.3f} {result['iterations']:>6} "
              f"{result['residual']:>10.2e} {err:>10.2e} "
              f"{'Y' if result['converged'] else 'N':>5}")

    del mesh, assembled, A, b, bc
    gc.collect()


# ===================================================================
# TEST 3: Sphere with analytical solution — convergence under refinement
# ===================================================================

def test_sphere_convergence():
    from triality.fem3d import Mesh, PoissonSolver3D
    import triality_engine as te

    print()
    print("=" * 90)
    print("TEST 3: SPHERE — Convergence on unstructured mesh")
    print("         Analytical: -∇²u = -6 on unit sphere, u = x²+y²+z² = 1 on surface")
    print("=" * 90)
    print()
    print(f"{'Target':>8} {'Elems':>8} {'DOFs':>8} "
          f"{'Asm(s)':>8} {'Slv(s)':>8} {'Iters':>6} "
          f"{'RSS(MB)':>8} {'L∞ Err':>10} {'Conv':>5}")
    print("-" * 84)

    prev_err, prev_h = None, None
    for target in [500, 2_000, 10_000, 50_000, 200_000]:
        gc.collect()
        try:
            msh_text = gmsh_sphere_tet_mesh(target)
            mesh = Mesh.from_gmsh_text(msh_text)
            n_nodes = mesh._rust_mesh.num_nodes
            n_elems = mesh._rust_mesh.num_elements

            node_list = parse_gmsh_nodes(msh_text)
            coords = np.array(node_list[:n_nodes])

            t0 = time.time()
            assembled = te.fem3d_assemble_poisson(mesh._rust_mesh, -6.0)
            t_asm = time.time() - t0

            A = assembled["stiffness"]
            rhs = np.asarray(assembled["rhs"], dtype=np.float64)

            # Surface nodes: r > 0.95
            r = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2)
            surf_dofs = np.where(r > 0.95)[0].tolist()
            exact = coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2

            bc = te.fem3d_apply_dirichlet(
                A, rhs, [int(i) for i in surf_dofs],
                [float(exact[i]) for i in surf_dofs])

            t0 = time.time()
            result = te.solve_linear(
                bc["matrix"], np.asarray(bc["rhs"], dtype=np.float64),
                method="gmres", precond="ilu0",
                tol=1e-8, max_iter=min(10000, max(2000, 20 * n_nodes)))
            t_slv = time.time() - t0

            u = np.asarray(result["x"])
            err = np.max(np.abs(u - exact))
            rss = rss_mb()

            # Estimate h from element count: h ~ (V / n_elems)^(1/3)
            h = (4.189 / n_elems) ** (1 / 3.0)
            rate_str = ""
            if prev_err is not None and err > 1e-14 and prev_err > 1e-14:
                rate = np.log(prev_err / err) / np.log(prev_h / h)
                rate_str = f"  O(h^{rate:.1f})"
            prev_err, prev_h = err, h

            print(f"{target:>8,} {n_elems:>8,} {n_nodes:>8,} "
                  f"{t_asm:>8.3f} {t_slv:>8.3f} {result['iterations']:>6} "
                  f"{rss:>8.0f} {err:>10.2e} "
                  f"{'Y' if result['converged'] else 'N':>5}{rate_str}")

            del mesh, assembled, A, rhs, bc, u
            gc.collect()

        except Exception as e:
            print(f"{target:>8,} {'FAILED':>8}   {e}")
            traceback.print_exc()


# ===================================================================
# TEST 4: Cylinder with analytical solution
# ===================================================================

def test_cylinder_analytical():
    from triality.fem3d import Mesh, PoissonSolver3D
    import triality_engine as te

    print()
    print("=" * 90)
    print("TEST 4: CYLINDER — Analytical T = z, ∇²T = 0, T=0 at z=0, T=2 at z=2")
    print("=" * 90)
    print()
    print(f"{'Target':>8} {'Elems':>8} {'DOFs':>8} "
          f"{'Asm(s)':>8} {'Slv(s)':>8} {'Iters':>6} "
          f"{'L∞ Err':>10} {'Conv':>5}")
    print("-" * 72)

    for target in [500, 2_000, 10_000, 50_000]:
        gc.collect()
        try:
            msh_text = gmsh_cylinder_mesh(target)
            mesh = Mesh.from_gmsh_text(msh_text)
            n_nodes = mesh._rust_mesh.num_nodes
            n_elems = mesh._rust_mesh.num_elements

            node_list = parse_gmsh_nodes(msh_text)
            coords = np.array(node_list[:n_nodes])

            t0 = time.time()
            assembled = te.fem3d_assemble_poisson(mesh._rust_mesh, 0.0)
            t_asm = time.time() - t0

            A = assembled["stiffness"]
            rhs = np.asarray(assembled["rhs"], dtype=np.float64)

            z = coords[:, 2]
            exact = z.copy()
            bc_dofs = np.where((z < 0.01) | (z > 1.99))[0].tolist()
            bc_vals = [float(exact[i]) for i in bc_dofs]

            bc = te.fem3d_apply_dirichlet(
                A, rhs, [int(i) for i in bc_dofs], bc_vals)

            t0 = time.time()
            result = te.solve_linear(
                bc["matrix"], np.asarray(bc["rhs"], dtype=np.float64),
                method="gmres", precond="ilu0",
                tol=1e-8, max_iter=min(10000, max(2000, 20 * n_nodes)))
            t_slv = time.time() - t0

            u = np.asarray(result["x"])
            err = np.max(np.abs(u - exact))

            print(f"{target:>8,} {n_elems:>8,} {n_nodes:>8,} "
                  f"{t_asm:>8.3f} {t_slv:>8.3f} {result['iterations']:>6} "
                  f"{err:>10.2e} {'Y' if result['converged'] else 'N':>5}")

            del mesh, assembled, A, rhs, bc, u
            gc.collect()

        except Exception as e:
            print(f"{target:>8,} {'FAILED':>8}   {e}")
            traceback.print_exc()


# ===================================================================
# TEST 5: Ugly geometry — L-bracket with hole
# ===================================================================

def test_ugly_geometry():
    from triality.fem3d import Mesh, PoissonSolver3D
    import triality_engine as te

    print()
    print("=" * 90)
    print("TEST 5: UGLY GEOMETRY — L-bracket with cylindrical hole")
    print("         Sharp re-entrant corner, thin features, varying element quality")
    print("         Source=1, homogeneous Dirichlet → solution must be non-negative")
    print("=" * 90)
    print()

    for target in [1_000, 5_000, 20_000, 80_000]:
        gc.collect()
        try:
            msh_text = gmsh_ugly_geometry_mesh(target)
            mesh = Mesh.from_gmsh_text(msh_text)
            n_nodes = mesh._rust_mesh.num_nodes
            n_elems = mesh._rust_mesh.num_elements

            node_list = parse_gmsh_nodes(msh_text)
            coords = np.array(node_list[:n_nodes])

            t0 = time.time()
            assembled = te.fem3d_assemble_poisson(mesh._rust_mesh, 1.0)
            t_asm = time.time() - t0

            A = assembled["stiffness"]
            rhs = np.asarray(assembled["rhs"], dtype=np.float64)

            # Boundary detection: use the Gmsh boundary faces
            # Heuristic: nodes at extremes of bounding box or near the hole
            x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
            tol_b = 0.02
            bc_mask = np.zeros(n_nodes, dtype=bool)
            # Outer surfaces
            bc_mask |= (x < tol_b) | (y < tol_b) | (z < tol_b)
            bc_mask |= (z > 0.3 - tol_b)
            bc_mask |= (x > 2.0 - tol_b) & (y < 1.0 + tol_b)
            bc_mask |= (y > 2.0 - tol_b) & (x < 1.0 + tol_b)
            # Re-entrant corner region
            bc_mask |= (x > 1.0 - tol_b) & (y > 1.0 - tol_b)
            # Hole surface
            dist_hole = np.sqrt((x - 1.5)**2 + (y - 0.5)**2)
            bc_mask |= (dist_hole < 0.22)

            bc_dofs = np.where(bc_mask)[0].tolist()
            n_bc = len(bc_dofs)
            n_free = n_nodes - n_bc

            bc_applied = te.fem3d_apply_dirichlet(
                A, rhs, [int(i) for i in bc_dofs], [0.0] * n_bc)

            t0 = time.time()
            result = te.solve_linear(
                bc_applied["matrix"], np.asarray(bc_applied["rhs"], dtype=np.float64),
                method="gmres", precond="ilu0",
                tol=1e-6, max_iter=min(10000, max(3000, 30 * n_nodes)))
            t_slv = time.time() - t0

            u = np.asarray(result["x"])
            rss = rss_mb()

            int_mask = ~bc_mask
            u_int = u[int_mask]
            min_u = np.min(u_int) if len(u_int) > 0 else 0
            max_u = np.max(u)
            physics_ok = min_u > -1e-6

            print(f"  target={target:>6,}  actual={n_elems:>6,} elems  "
                  f"{n_nodes:>6,} DOFs ({n_free:,} free)  "
                  f"asm={t_asm:.3f}s  solve={t_slv:.3f}s  "
                  f"iters={result['iterations']}  "
                  f"conv={'Y' if result['converged'] else 'N'}  "
                  f"u∈[{min_u:.4e},{max_u:.4e}]  "
                  f"physics={'OK' if physics_ok else 'BAD'}  "
                  f"RSS={rss:.0f}MB")

            # Export VTU for largest mesh
            if n_elems > 10000:
                vtu = te.fem3d_export_vtu(mesh._rust_mesh, "source_field",
                                          np.asarray(u, dtype=np.float64))
                print(f"    VTU export: {len(vtu):,} chars")

            del mesh, assembled, A, rhs, bc_applied, u
            gc.collect()

        except Exception as e:
            print(f"  target={target:>6,}: FAILED — {e}")
            traceback.print_exc()


# ===================================================================
# TEST 6: Memory wall — push until it hurts
# ===================================================================

def test_memory_wall():
    from triality.fem3d import Mesh, PoissonSolver3D
    import triality_engine as te

    print()
    print("=" * 90)
    print("TEST 6: MEMORY WALL — Hex8 scaling until OOM or >60s solve")
    print("         Tracking: mesh build, assembly, BC application, solve, peak RSS")
    print("=" * 90)
    print()
    print(f"{'n':>5} {'Elems':>10} {'DOFs':>10} "
          f"{'Mesh(s)':>8} {'Asm(s)':>8} {'BC(s)':>8} {'Slv(s)':>8} "
          f"{'RSS(MB)':>8} {'Iters':>6} {'Conv':>5}")
    print("-" * 100)

    # n: 10→1K, 22→10K, 46→100K, 68→314K, 100→1M, 130→2.2M
    for n in [10, 22, 46, 68, 100, 130, 160]:
        n_elems = n ** 3
        n_dofs = (n + 1) ** 3

        # Pre-flight memory check: CSR ~27 nonzeros/row, 16 bytes each (index+value)
        # Plus ILU0 copy, RHS, solution vectors
        est_mb = n_dofs * 27 * 16 / 1e6 * 2 + n_dofs * 8 * 5 / 1e6
        if est_mb > 12000:  # 12 GB cap on 16 GB system
            print(f"{n:>5} {n_elems:>10,} {n_dofs:>10,} "
                  f"{'SKIP':>8} — estimated {est_mb:.0f}MB exceeds 12GB cap")
            continue

        gc.collect()
        try:
            t0 = time.time()
            nodes, hexes, bdry, _ = make_structured_hex_mesh(n)
            mesh = Mesh.from_hex8_arrays(nodes, hexes)
            t_mesh = time.time() - t0

            t0 = time.time()
            assembled = te.fem3d_assemble_poisson(mesh._rust_mesh, 1.0)
            t_asm = time.time() - t0

            A = assembled["stiffness"]
            rhs = np.asarray(assembled["rhs"], dtype=np.float64)

            t0 = time.time()
            bc = te.fem3d_apply_dirichlet(
                A, rhs, [int(i) for i in bdry], [0.0] * len(bdry))
            t_bc = time.time() - t0

            t0 = time.time()
            result = te.solve_linear(
                bc["matrix"], np.asarray(bc["rhs"], dtype=np.float64),
                method="gmres", precond="ilu0",
                tol=1e-6, max_iter=min(5000, max(1000, 5 * n_dofs)))
            t_slv = time.time() - t0

            rss = rss_mb()
            print(f"{n:>5} {n_elems:>10,} {n_dofs:>10,} "
                  f"{t_mesh:>8.2f} {t_asm:>8.2f} {t_bc:>8.2f} {t_slv:>8.2f} "
                  f"{rss:>8.0f} {result['iterations']:>6} "
                  f"{'Y' if result['converged'] else 'N':>5}")

            del mesh, assembled, A, rhs, bc, nodes, hexes, result
            gc.collect()

        except MemoryError:
            rss = rss_mb()
            print(f"{n:>5} {n_elems:>10,} {n_dofs:>10,} "
                  f"{'OOM':>8} at RSS={rss:.0f}MB")
            break
        except Exception as e:
            print(f"{n:>5} {n_elems:>10,} {n_dofs:>10,} "
                  f"{'FAIL':>8} — {e}")
            traceback.print_exc()
            break


# ===================================================================
# MAIN
# ===================================================================

if __name__ == "__main__":
    print(f"System: {os.cpu_count()} CPUs, ~16 GB RAM")
    print(f"Initial RSS: {rss_mb():.1f} MB")
    print()

    test_scaling()
    test_preconditioner_shootout()
    test_sphere_convergence()
    test_cylinder_analytical()
    test_ugly_geometry()
    test_memory_wall()

    print()
    print("=" * 90)
    print("ALL SCALE TESTS COMPLETE")
    print("=" * 90)
