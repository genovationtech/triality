"""
Finite Element Method solver for static structural analysis.

Provides numerical FEM solutions complementing the analytical models in
static_solver.py and buckling.py. Supports 1D Euler-Bernoulli beam elements
and 2D assemblies with arbitrary boundary conditions and load cases.

Governing equation:  K * u = F
    K = global stiffness matrix (assembled from element stiffnesses)
    u = nodal displacement vector
    F = nodal force vector

Element types:
    - Euler-Bernoulli beam: 4 DOF per element (v1, theta1, v2, theta2)
    - Extensional bar:      2 DOF per element (u1, u2)

Post-processing:
    - Bending moment and shear force diagrams
    - Stress recovery (sigma = E * strain)
    - Von Mises equivalent stress
    - Margin of safety per load case
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .static_solver import Material, MarginOfSafety, LoadType, MATERIALS


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ElementResult:
    """Per-element post-processed results."""
    element_id: int
    axial_stress: float                # sigma_axial  [Pa]
    bending_stress_max: float          # sigma_bending (outer fibre) [Pa]
    von_mises_stress: float            # sigma_vm [Pa]
    shear_force: np.ndarray            # V at element nodes [N]
    bending_moment: np.ndarray         # M at element nodes [N*m]


@dataclass
class FEMResult:
    """Complete FEM solution for one load case."""
    load_case_name: str
    displacements: np.ndarray          # full DOF vector  [m, rad]
    reactions: np.ndarray              # reaction forces at constrained DOFs
    element_results: List[ElementResult]
    max_von_mises: float               # peak sigma_vm across model [Pa]
    margin_of_safety: float            # MS based on material allowable
    converged: bool


class DOFType(Enum):
    """Degree-of-freedom type at a node."""
    DISPLACEMENT = 0   # translational (v)
    ROTATION = 1       # rotational (theta)


# ---------------------------------------------------------------------------
# Beam element
# ---------------------------------------------------------------------------

def euler_bernoulli_stiffness(E: float, I: float, A: float,
                              L: float) -> np.ndarray:
    """
    Local 4x4 stiffness matrix for an Euler-Bernoulli beam element.

    DOF ordering: [v1, theta1, v2, theta2]
    where v = transverse displacement, theta = rotation.

    Returns:
        4x4 numpy array (element stiffness in local coords)
    """
    coeff = E * I / L**3
    k_e = coeff * np.array([
        [ 12,    6*L,   -12,    6*L  ],
        [ 6*L,   4*L**2, -6*L,  2*L**2],
        [-12,   -6*L,    12,   -6*L  ],
        [ 6*L,   2*L**2, -6*L,  4*L**2],
    ])
    return k_e


def bar_stiffness(E: float, A: float, L: float) -> np.ndarray:
    """
    Local 2x2 stiffness matrix for a bar (axial) element.

    DOF ordering: [u1, u2]

    Returns:
        2x2 numpy array
    """
    coeff = E * A / L
    k_e = coeff * np.array([
        [ 1, -1],
        [-1,  1],
    ])
    return k_e


# ---------------------------------------------------------------------------
# Consistent load vector helpers
# ---------------------------------------------------------------------------

def beam_uniform_load_vector(w: float, L: float) -> np.ndarray:
    """
    Consistent nodal load vector for a uniformly distributed transverse
    load *w* [N/m] on an Euler-Bernoulli element of length *L*.

    Returns:
        4x1 numpy array  [F1, M1, F2, M2]
    """
    return np.array([
        w * L / 2.0,
        w * L**2 / 12.0,
        w * L / 2.0,
        -w * L**2 / 12.0,
    ])


# ---------------------------------------------------------------------------
# Main solver class
# ---------------------------------------------------------------------------

class StructuralFEMSolver:
    """
    1-D / beam-assembly FEM solver for static structural analysis.

    Workflow
    -------
    1. Create solver with a material.
    2. Add nodes (coordinates).
    3. Add beam elements connecting nodes.
    4. Apply boundary conditions (fixed, pinned, roller, ...).
    5. Define one or more load cases (point loads, distributed loads).
    6. Call ``solve()`` to obtain displacements, stresses, and margins.

    Example
    -------
    >>> mat = MATERIALS['AL7075-T6']
    >>> solver = StructuralFEMSolver(default_material=mat)
    >>> solver.add_node(0, 0.0)
    >>> solver.add_node(1, 1.0)
    >>> solver.add_node(2, 2.0)
    >>> solver.add_element(0, 0, 1, E=mat.youngs_modulus, I=1e-6, A=1e-4, c=0.01)
    >>> solver.add_element(1, 1, 2, E=mat.youngs_modulus, I=1e-6, A=1e-4, c=0.01)
    >>> solver.fix_node(0)                       # cantilever root
    >>> solver.add_point_load('gravity', 2, force=-1000.0)
    >>> results = solver.solve()
    """

    # DOFs per node for beam: transverse displacement + rotation
    BEAM_DOFS_PER_NODE = 2

    def __init__(self, default_material: Optional[Material] = None):
        """
        Args:
            default_material: Fallback material for stress allowables.
        """
        self.default_material = default_material

        # Geometry
        self._nodes: Dict[int, float] = {}           # node_id -> x-coordinate
        self._elements: List[Dict] = []               # element property dicts
        self._n_nodes: int = 0

        # Boundary conditions: mapping DOF index -> prescribed value
        self._prescribed_dofs: Dict[int, float] = {}

        # Load cases: name -> {dof_index: value}
        self._load_cases: Dict[str, np.ndarray] = {}

        # Distributed loads per load case: name -> [(elem_id, w)]
        self._distributed_loads: Dict[str, List[Tuple[int, float]]] = {}

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def add_node(self, node_id: int, x: float) -> None:
        """Register a node at coordinate *x* along the beam axis."""
        self._nodes[node_id] = x
        self._n_nodes = len(self._nodes)

    def add_element(self, elem_id: int, node_i: int, node_j: int,
                    E: float, I: float, A: float, c: float,
                    material: Optional[Material] = None) -> None:
        """
        Add a beam element between two nodes.

        Args:
            elem_id:  Unique element identifier.
            node_i:   Start-node id.
            node_j:   End-node id.
            E:        Young's modulus [Pa].
            I:        Second moment of area [m^4].
            A:        Cross-section area [m^2].
            c:        Distance from neutral axis to outer fibre [m] (for stress).
            material: Per-element material override.
        """
        xi = self._nodes[node_i]
        xj = self._nodes[node_j]
        L = abs(xj - xi)
        if L < 1e-15:
            raise ValueError(f"Element {elem_id}: zero-length element "
                             f"between nodes {node_i} and {node_j}.")
        self._elements.append({
            'id': elem_id,
            'node_i': node_i,
            'node_j': node_j,
            'E': E,
            'I': I,
            'A': A,
            'L': L,
            'c': c,
            'material': material or self.default_material,
        })

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _dof_index(self, node_id: int, dof_type: DOFType) -> int:
        """Return global DOF index for a given node and DOF type."""
        ordered = sorted(self._nodes.keys())
        local = ordered.index(node_id)
        return local * self.BEAM_DOFS_PER_NODE + dof_type.value

    def fix_node(self, node_id: int, value: float = 0.0) -> None:
        """Fully fix a node (displacement = rotation = *value*)."""
        self._prescribed_dofs[self._dof_index(node_id, DOFType.DISPLACEMENT)] = value
        self._prescribed_dofs[self._dof_index(node_id, DOFType.ROTATION)] = value

    def pin_node(self, node_id: int, value: float = 0.0) -> None:
        """Pin a node (displacement = *value*, rotation free)."""
        self._prescribed_dofs[self._dof_index(node_id, DOFType.DISPLACEMENT)] = value

    def constrain_dof(self, node_id: int, dof_type: DOFType,
                      value: float = 0.0) -> None:
        """Constrain a single DOF to a prescribed value."""
        self._prescribed_dofs[self._dof_index(node_id, dof_type)] = value

    # ------------------------------------------------------------------
    # Loads
    # ------------------------------------------------------------------

    def add_point_load(self, case_name: str, node_id: int,
                       force: float = 0.0, moment: float = 0.0) -> None:
        """
        Apply a point force and/or moment at *node_id* for load case *case_name*.
        """
        n_dofs = self._n_nodes * self.BEAM_DOFS_PER_NODE
        if case_name not in self._load_cases:
            self._load_cases[case_name] = np.zeros(n_dofs)
        f_vec = self._load_cases[case_name]
        f_vec[self._dof_index(node_id, DOFType.DISPLACEMENT)] += force
        f_vec[self._dof_index(node_id, DOFType.ROTATION)] += moment

    def add_distributed_load(self, case_name: str, elem_id: int,
                             w: float) -> None:
        """
        Apply a uniform distributed transverse load *w* [N/m] on element
        *elem_id* for load case *case_name*.
        """
        if case_name not in self._distributed_loads:
            self._distributed_loads[case_name] = []
        self._distributed_loads[case_name].append((elem_id, w))

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def _assemble_global_stiffness(self) -> sp.csc_matrix:
        """Assemble global stiffness matrix from all elements (sparse)."""
        n_dofs = self._n_nodes * self.BEAM_DOFS_PER_NODE
        rows, cols, vals = [], [], []
        ordered = sorted(self._nodes.keys())

        for elem in self._elements:
            k_e = euler_bernoulli_stiffness(elem['E'], elem['I'],
                                            elem['A'], elem['L'])
            ni = ordered.index(elem['node_i'])
            nj = ordered.index(elem['node_j'])
            dof_map = [
                ni * 2, ni * 2 + 1,
                nj * 2, nj * 2 + 1,
            ]
            for ii in range(4):
                for jj in range(4):
                    rows.append(dof_map[ii])
                    cols.append(dof_map[jj])
                    vals.append(k_e[ii, jj])

        K = sp.csc_matrix((vals, (rows, cols)), shape=(n_dofs, n_dofs))
        return K

    def _assemble_load_vector(self, case_name: str) -> np.ndarray:
        """Assemble global force vector including consistent nodal loads."""
        n_dofs = self._n_nodes * self.BEAM_DOFS_PER_NODE
        F = np.zeros(n_dofs)

        # Point loads
        if case_name in self._load_cases:
            F += self._load_cases[case_name]

        # Distributed loads -> consistent nodal loads
        ordered = sorted(self._nodes.keys())
        if case_name in self._distributed_loads:
            for elem_id, w in self._distributed_loads[case_name]:
                elem = next(e for e in self._elements if e['id'] == elem_id)
                f_e = beam_uniform_load_vector(w, elem['L'])
                ni = ordered.index(elem['node_i'])
                nj = ordered.index(elem['node_j'])
                dof_map = [ni * 2, ni * 2 + 1, nj * 2, nj * 2 + 1]
                for ii in range(4):
                    F[dof_map[ii]] += f_e[ii]

        return F

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------

    def solve(self, load_case_names: Optional[List[str]] = None) -> List[FEMResult]:
        """
        Solve K*u = F for each requested load case.

        Args:
            load_case_names: Subset of cases to solve; *None* solves all.

        Returns:
            List of :class:`FEMResult`, one per load case.
        """
        if load_case_names is None:
            all_names = set(self._load_cases.keys()) | set(self._distributed_loads.keys())
            load_case_names = sorted(all_names) if all_names else []

        K_global = self._assemble_global_stiffness()
        n_dofs = self._n_nodes * self.BEAM_DOFS_PER_NODE

        # Identify free / prescribed DOF sets
        prescribed_indices = sorted(self._prescribed_dofs.keys())
        free_indices = [i for i in range(n_dofs) if i not in prescribed_indices]

        results: List[FEMResult] = []

        for case_name in load_case_names:
            F_global = self._assemble_load_vector(case_name)

            # Partition: K_ff * u_f = F_f - K_fp * u_p
            u_full = np.zeros(n_dofs)
            for idx, val in self._prescribed_dofs.items():
                u_full[idx] = val

            free = np.array(free_indices, dtype=int)
            pres = np.array(prescribed_indices, dtype=int)

            K_ff = K_global[np.ix_(free, free)]
            F_f = F_global[free]

            if len(pres) > 0:
                K_fp = K_global[free][:, pres]
                u_p = np.array([self._prescribed_dofs[i] for i in pres])
                F_f = F_f - K_fp.dot(u_p)

            # Solve sparse system
            try:
                u_free = spla.spsolve(K_ff.tocsc(), F_f)
                converged = True
            except Exception:
                u_free = np.zeros(len(free))
                converged = False

            u_full[free] = u_free

            # Reactions at prescribed DOFs
            reactions = np.zeros(n_dofs)
            R_full = K_global.dot(u_full) - F_global
            reactions[pres] = R_full[pres] if len(pres) > 0 else 0.0

            # Post-process elements
            elem_results = self._post_process_elements(u_full)

            max_vm = max((er.von_mises_stress for er in elem_results), default=0.0)

            # Margin of safety against material ultimate
            allowable = float('inf')
            for elem in self._elements:
                mat = elem.get('material')
                if mat is not None:
                    allowable = min(allowable, mat.ultimate_strength)
            if self.default_material is not None:
                allowable = min(allowable, self.default_material.ultimate_strength)

            ms = MarginOfSafety.compute_ms(allowable, max_vm) if max_vm > 0 else float('inf')

            results.append(FEMResult(
                load_case_name=case_name,
                displacements=u_full,
                reactions=reactions,
                element_results=elem_results,
                max_von_mises=max_vm,
                margin_of_safety=ms,
                converged=converged,
            ))

        return results

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _post_process_elements(self, u_global: np.ndarray) -> List[ElementResult]:
        """Recover element-level stresses from the global displacement vector."""
        ordered = sorted(self._nodes.keys())
        elem_results: List[ElementResult] = []

        for elem in self._elements:
            ni = ordered.index(elem['node_i'])
            nj = ordered.index(elem['node_j'])
            dof_map = [ni * 2, ni * 2 + 1, nj * 2, nj * 2 + 1]
            u_e = u_global[dof_map]

            L = elem['L']
            E = elem['E']
            I = elem['I']
            c = elem['c']

            # Element stiffness * local displacements -> element forces
            k_e = euler_bernoulli_stiffness(E, I, elem['A'], L)
            f_e = k_e @ u_e   # [V1, M1, V2, M2]

            shear = np.array([f_e[0], -f_e[2]])
            moment = np.array([-f_e[1], f_e[3]])

            # Bending stress: sigma = M * c / I
            M_max = np.max(np.abs(moment))
            sigma_bending = M_max * c / I if I > 0 else 0.0

            # Axial stress (approximate from extensional strain if any)
            # For pure beam elements axial stress is zero unless bar DOFs added
            sigma_axial = 0.0

            # Von Mises for uniaxial: sigma_vm = sqrt(sigma_x^2 + 3*tau^2)
            V_max = np.max(np.abs(shear))
            # Average shear stress tau = V / A  (approximate)
            A = elem['A']
            tau = V_max / A if A > 0 else 0.0

            sigma_x = sigma_axial + sigma_bending
            sigma_vm = np.sqrt(sigma_x**2 + 3.0 * tau**2)

            elem_results.append(ElementResult(
                element_id=elem['id'],
                axial_stress=sigma_axial,
                bending_stress_max=sigma_bending,
                von_mises_stress=sigma_vm,
                shear_force=shear,
                bending_moment=moment,
            ))

        return elem_results

    # ------------------------------------------------------------------
    # Convenience builders
    # ------------------------------------------------------------------

    @classmethod
    def cantilever_beam(cls, length: float, n_elements: int,
                        material: Material, I: float, A: float,
                        c: float) -> 'StructuralFEMSolver':
        """
        Build a cantilever beam model (fixed at x=0, free at x=length).

        Args:
            length:      Total beam length [m].
            n_elements:  Number of equal-length elements.
            material:    Beam material.
            I:           Second moment of area [m^4].
            A:           Cross-section area [m^2].
            c:           Outer-fibre distance [m].

        Returns:
            Pre-configured solver (add loads, then call ``solve``).
        """
        solver = cls(default_material=material)
        dx = length / n_elements
        for i in range(n_elements + 1):
            solver.add_node(i, i * dx)
        for i in range(n_elements):
            solver.add_element(i, i, i + 1,
                               E=material.youngs_modulus, I=I, A=A, c=c,
                               material=material)
        solver.fix_node(0)
        return solver

    @classmethod
    def simply_supported_beam(cls, length: float, n_elements: int,
                              material: Material, I: float, A: float,
                              c: float) -> 'StructuralFEMSolver':
        """
        Build a simply supported beam (pinned at x=0 and x=length).

        Returns:
            Pre-configured solver.
        """
        solver = cls(default_material=material)
        dx = length / n_elements
        for i in range(n_elements + 1):
            solver.add_node(i, i * dx)
        for i in range(n_elements):
            solver.add_element(i, i, i + 1,
                               E=material.youngs_modulus, I=I, A=A, c=c,
                               material=material)
        solver.pin_node(0)
        solver.pin_node(n_elements)
        return solver
