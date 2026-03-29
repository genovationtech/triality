"""
Integrated structural analysis solver.

Provides a unified ``solve()`` interface that performs static stress analysis,
buckling checks, and composite laminate evaluation by wiring together the
existing :class:`StructuralFEMSolver`, buckling models, and Classical
Lamination Theory (CLT) from this package.

Physics
-------
1. **Static FEM**: Solve K*u = F for beam/frame assemblies using
   Euler-Bernoulli beam elements with sparse direct solvers.

2. **Buckling Assessment**: For each element, compare applied stress
   against Euler column buckling, panel buckling, and crippling limits.

3. **Composite Laminate Evaluation**: If a laminate layup is provided,
   compute ABD matrices, ply stresses, and first-ply-failure margins
   using Tsai-Hill or max-stress criteria.

The solver collects all results into a single
:class:`StructuralAnalysisResult` dataclass.

Dependencies
------------
- numpy
- scipy.sparse
- static_solver, buckling, composite_laminates, fem_solver (this package)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from triality.core.units import UnitMetadata
from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from .static_solver import (
    EulerBernoulliBeam,
    PlateAnalysis,
    Material,
    MarginOfSafety,
    LoadCaseCombination,
    LoadType,
    MATERIALS,
)
from .buckling import (
    EulerColumnBuckling,
    PanelBuckling,
    OrthotropicPanelBuckling,
    CripplingAnalysis,
    BucklingInteraction,
    BucklingMode,
    EndCondition,
)
from .composite_laminates import (
    OrthotropicPly,
    LaminatePly,
    Laminate,
    FailureCriterion,
    COMPOSITE_MATERIALS,
)
from .fem_solver import (
    StructuralFEMSolver,
    FEMResult,
    ElementResult,
    DOFType,
    euler_bernoulli_stiffness,
    bar_stiffness,
    beam_uniform_load_vector,
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class BucklingCheckResult:
    """Result of a buckling check on one structural element.

    Attributes
    ----------
    element_id : int
        Element identifier.
    mode : str
        Governing buckling mode ('column', 'panel', 'crippling').
    critical_stress : float
        Critical buckling stress [Pa].
    applied_stress : float
        Applied compressive stress [Pa].
    margin_of_safety : float
        Buckling margin of safety.
    is_safe : bool
        True if MS >= 0.
    """
    element_id: int = 0
    mode: str = "column"
    critical_stress: float = 0.0
    applied_stress: float = 0.0
    margin_of_safety: float = 0.0
    is_safe: bool = True


@dataclass
class LaminateCheckResult:
    """Result of a composite laminate strength assessment.

    Attributes
    ----------
    critical_ply : int
        Index of the first ply to fail.
    failure_load_multiplier : float
        Load multiplier at first-ply failure (>1 means safe at applied load).
    margin_of_safety : float
        MS = failure_load_multiplier - 1.
    criterion : str
        Failure criterion used.
    is_symmetric : bool
        Whether the laminate has a symmetric layup.
    A_matrix : np.ndarray
        Extensional stiffness matrix [A] (3x3).
    D_matrix : np.ndarray
        Bending stiffness matrix [D] (3x3).
    """
    critical_ply: int = -1
    failure_load_multiplier: float = 0.0
    margin_of_safety: float = 0.0
    criterion: str = "tsai_hill"
    is_symmetric: bool = False
    A_matrix: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    D_matrix: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))


@dataclass
class StructuralAnalysisResult:
    """Complete result from the integrated structural solver.

    Attributes
    ----------
    fem_results : list of FEMResult
        FEM solution for each load case.
    buckling_checks : list of BucklingCheckResult
        Buckling assessment for each element.
    laminate_check : LaminateCheckResult or None
        Composite laminate evaluation (if a layup was provided).
    max_von_mises : float
        Peak von Mises stress across all elements and load cases [Pa].
    min_buckling_ms : float
        Minimum buckling margin of safety across all elements.
    overall_ms : float
        Governing (minimum) margin of safety from all checks.
    is_safe : bool
        True if overall_ms >= 0.
    """
    fem_results: List[FEMResult] = field(default_factory=list)
    buckling_checks: List[BucklingCheckResult] = field(default_factory=list)
    laminate_check: Optional[LaminateCheckResult] = None
    max_von_mises: float = 0.0
    min_buckling_ms: float = float('inf')
    overall_ms: float = float('inf')
    is_safe: bool = True
    units: UnitMetadata = field(default_factory=UnitMetadata)

    def __post_init__(self):
        """Declare unit metadata for all fields."""
        self.units.declare("max_von_mises", "Pa", "Peak von Mises stress")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class StructuralSolver:
    """Integrated structural analysis solver.

    Combines FEM static analysis, buckling checks, and optional composite
    laminate evaluation into a single ``solve()`` call.

    Parameters
    ----------
    material : Material
        Structural material for isotropic analysis.
    length : float
        Beam / column length [m].
    n_elements : int
        Number of FEM elements.
    I : float
        Second moment of area [m^4].
    A : float
        Cross-sectional area [m^2].
    c : float
        Distance from neutral axis to outer fibre [m].
    support : str
        'cantilever' or 'simply_supported'.

    Example
    -------
    >>> mat = MATERIALS['AL7075-T6']
    >>> solver = StructuralSolver(
    ...     material=mat, length=2.0, n_elements=20,
    ...     I=1e-6, A=1e-4, c=0.01, support='cantilever',
    ... )
    >>> solver.add_tip_load('gravity', force=-5000.0)
    >>> result = solver.solve()
    >>> print(f"Max von Mises: {result.max_von_mises:.1f} Pa")
    >>> print(f"Min buckling MS: {result.min_buckling_ms:.2f}")
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M3_COUPLED

    def __init__(self, material: Material, length: float = 1.0,
                 n_elements: int = 10, I: float = 1e-6,
                 A: float = 1e-4, c: float = 0.01,
                 support: str = 'cantilever'):
        self.material = material
        self.length = length
        self.n_elements = n_elements
        self.I = I
        self.A = A
        self.c = c
        self.support = support

        # Build FEM model
        if support == 'cantilever':
            self._fem = StructuralFEMSolver.cantilever_beam(
                length=length, n_elements=n_elements,
                material=material, I=I, A=A, c=c)
        else:
            self._fem = StructuralFEMSolver.simply_supported_beam(
                length=length, n_elements=n_elements,
                material=material, I=I, A=A, c=c)

        # Optional laminate
        self._laminate: Optional[Laminate] = None
        self._laminate_loads: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._laminate_criterion = FailureCriterion.TSAI_HILL

        # Panel geometry for buckling (optional override)
        self._panel_width: Optional[float] = None
        self._panel_thickness: Optional[float] = None

        # Coupling state
        self._coupled_state = None
        self._time = 0.0

    # ------------------------------------------------------------------
    # Load definition
    # ------------------------------------------------------------------

    def add_tip_load(self, case_name: str, force: float = 0.0,
                     moment: float = 0.0) -> None:
        """Apply a point load at the free end (last node).

        Parameters
        ----------
        case_name : str
            Load case label.
        force : float
            Transverse force [N].
        moment : float
            Applied moment [N.m].
        """
        tip_node = self.n_elements
        self._fem.add_point_load(case_name, tip_node,
                                  force=force, moment=moment)

    def add_distributed_load(self, case_name: str, w: float) -> None:
        """Apply a uniform distributed load on all elements.

        Parameters
        ----------
        case_name : str
            Load case label.
        w : float
            Load intensity [N/m] (positive = +y direction).
        """
        for elem_id in range(self.n_elements):
            self._fem.add_distributed_load(case_name, elem_id, w)

    def add_point_load(self, case_name: str, node_id: int,
                       force: float = 0.0, moment: float = 0.0) -> None:
        """Apply a point load at an arbitrary node.

        Parameters
        ----------
        case_name : str
            Load case label.
        node_id : int
            Node index (0 = root, n_elements = tip).
        force, moment : float
            Applied force [N] and moment [N.m].
        """
        self._fem.add_point_load(case_name, node_id,
                                  force=force, moment=moment)

    # ------------------------------------------------------------------
    # Composite laminate setup
    # ------------------------------------------------------------------

    def set_laminate(self, laminate: Laminate,
                     N_applied: np.ndarray,
                     M_applied: np.ndarray,
                     criterion: FailureCriterion = FailureCriterion.TSAI_HILL) -> None:
        """Attach a composite laminate for strength evaluation.

        Parameters
        ----------
        laminate : Laminate
            Composite laminate object with ply definitions.
        N_applied : np.ndarray
            Resultant forces per unit width [N/m], shape (3,).
        M_applied : np.ndarray
            Resultant moments per unit width [N.m/m], shape (3,).
        criterion : FailureCriterion
            Failure criterion to apply.
        """
        self._laminate = laminate
        self._laminate_loads = (np.asarray(N_applied), np.asarray(M_applied))
        self._laminate_criterion = criterion

    # ------------------------------------------------------------------
    # Panel buckling setup
    # ------------------------------------------------------------------

    def set_panel_geometry(self, width: float, thickness: float) -> None:
        """Set panel dimensions for panel-buckling checks.

        Parameters
        ----------
        width : float
            Unsupported panel width [m].
        thickness : float
            Panel skin thickness [m].
        """
        self._panel_width = width
        self._panel_thickness = thickness

    # ------------------------------------------------------------------
    # Buckling check
    # ------------------------------------------------------------------

    def _check_buckling(self, fem_results: List[FEMResult]) -> List[BucklingCheckResult]:
        """Run buckling checks on all elements using FEM stress results."""
        checks: List[BucklingCheckResult] = []
        E = self.material.youngs_modulus
        nu = self.material.poissons_ratio
        sigma_y = self.material.yield_strength

        elem_length = self.length / self.n_elements
        r_gyration = np.sqrt(self.I / self.A)

        for fem_res in fem_results:
            for er in fem_res.element_results:
                # Applied compressive stress (conservative: bending + axial)
                sigma_applied = er.bending_stress_max + abs(er.axial_stress)
                if sigma_applied <= 0:
                    checks.append(BucklingCheckResult(
                        element_id=er.element_id,
                        mode='column',
                        critical_stress=float('inf'),
                        applied_stress=0.0,
                        margin_of_safety=float('inf'),
                        is_safe=True,
                    ))
                    continue

                # 1. Euler column buckling
                P_cr_euler = EulerColumnBuckling.critical_load(
                    E, self.I, elem_length, EndCondition.FIXED_PINNED)
                sigma_cr_euler = P_cr_euler / self.A

                # 2. Panel buckling (if panel geometry set)
                sigma_cr_panel = float('inf')
                if self._panel_width is not None and self._panel_thickness is not None:
                    sigma_cr_panel = PanelBuckling.critical_stress_compression(
                        E, nu, self._panel_thickness, self._panel_width)

                # 3. Crippling
                sigma_cr_crip = CripplingAnalysis.crippling_stress_flat_plate(
                    E, sigma_y,
                    self._panel_thickness or self.c * 2,
                    self._panel_width or self.c * 10)

                # Governing mode
                modes = {
                    'column': sigma_cr_euler,
                    'panel': sigma_cr_panel,
                    'crippling': sigma_cr_crip,
                }
                governing_mode = min(modes, key=modes.get)
                sigma_cr = modes[governing_mode]

                ms = MarginOfSafety.compute_ms(sigma_cr, sigma_applied)

                checks.append(BucklingCheckResult(
                    element_id=er.element_id,
                    mode=governing_mode,
                    critical_stress=sigma_cr,
                    applied_stress=sigma_applied,
                    margin_of_safety=ms,
                    is_safe=ms >= 0,
                ))

        return checks

    # ------------------------------------------------------------------
    # Laminate check
    # ------------------------------------------------------------------

    def _check_laminate(self) -> Optional[LaminateCheckResult]:
        """Evaluate first-ply failure of the attached laminate."""
        if self._laminate is None or self._laminate_loads is None:
            return None

        lam = self._laminate
        N_app, M_app = self._laminate_loads
        criterion = self._laminate_criterion

        crit_ply, flm = lam.first_ply_failure(N_app, M_app, criterion)
        ms = flm - 1.0

        return LaminateCheckResult(
            critical_ply=crit_ply,
            failure_load_multiplier=flm,
            margin_of_safety=ms,
            criterion=criterion.value,
            is_symmetric=lam.is_symmetric(),
            A_matrix=lam.A.copy(),
            D_matrix=lam.D.copy(),
        )

    # ------------------------------------------------------------------
    # Main solve
    # ------------------------------------------------------------------

    def solve(self, load_case_names: Optional[List[str]] = None) -> StructuralAnalysisResult:
        """Run the integrated structural analysis.

        Performs:
        1. FEM static solution (displacements, stresses, reactions).
        2. Buckling check for each element (column, panel, crippling).
        3. Composite laminate first-ply failure (if configured).

        Parameters
        ----------
        load_case_names : list of str, optional
            Subset of load cases to solve.  None solves all defined cases.

        Returns
        -------
        StructuralAnalysisResult
            Combined results from all analysis disciplines.
        """
        # 1. FEM solve
        fem_results = self._fem.solve(load_case_names)

        # 2. Buckling
        buckling_checks = self._check_buckling(fem_results)

        # 3. Laminate
        laminate_check = self._check_laminate()

        # Aggregate metrics
        max_vm = max((fr.max_von_mises for fr in fem_results), default=0.0)

        buck_ms_vals = [bc.margin_of_safety for bc in buckling_checks
                        if bc.margin_of_safety != float('inf')]
        min_buck_ms = min(buck_ms_vals) if buck_ms_vals else float('inf')

        # Overall governing MS
        ms_candidates = []
        for fr in fem_results:
            if fr.margin_of_safety != float('inf'):
                ms_candidates.append(fr.margin_of_safety)
        if min_buck_ms != float('inf'):
            ms_candidates.append(min_buck_ms)
        if laminate_check is not None:
            ms_candidates.append(laminate_check.margin_of_safety)

        overall_ms = min(ms_candidates) if ms_candidates else float('inf')

        return StructuralAnalysisResult(
            fem_results=fem_results,
            buckling_checks=buckling_checks,
            laminate_check=laminate_check,
            max_von_mises=max_vm,
            min_buckling_ms=min_buck_ms,
            overall_ms=overall_ms,
            is_safe=overall_ms >= 0,
        )

    def export_state(self, result: StructuralAnalysisResult) -> PhysicsState:
        """Export structural result as canonical displacement/stress fields."""
        state = PhysicsState(solver_name="structural_analysis")
        x = np.linspace(0, self.length, self.n_elements + 1)

        if result.fem_results:
            fr = result.fem_results[0]
            nodal_displacement = np.asarray(fr.displacements)[::2]
            state.set_field("displacement", nodal_displacement, "m", grid=x)

            vm_by_element = np.array([er.von_mises_stress for er in fr.element_results], dtype=float)
            if vm_by_element.size:
                x_elem = np.linspace(self.length / (2 * self.n_elements),
                                     self.length - self.length / (2 * self.n_elements),
                                     self.n_elements)
                state.set_field("stress_von_mises", vm_by_element, "Pa", grid=x_elem)
            else:
                state.set_field("stress_von_mises", np.array([result.max_von_mises]), "Pa")
        else:
            state.set_field("stress_von_mises", np.array([result.max_von_mises]), "Pa")

        state.metadata["max_von_mises_Pa"] = result.max_von_mises
        state.metadata["margin_of_safety"] = result.overall_ms
        state.metadata["is_safe"] = result.is_safe
        state.metadata["min_buckling_ms"] = result.min_buckling_ms
        state.metadata["buckling_check_count"] = len(result.buckling_checks)
        return state

    def import_state(self, state: PhysicsState) -> None:
        """Import coupled fields from partner solvers."""
        self._coupled_state = state

    def advance(self, dt: float) -> PhysicsState:
        """Re-solve structural analysis with coupled thermal loads."""
        if self._coupled_state is not None:
            if self._coupled_state.has("temperature"):
                T = self._coupled_state.get_field("temperature").data
                # Compute thermal load as equivalent distributed force
                alpha_th = self.material.thermal_expansion if hasattr(self.material, 'thermal_expansion') else 12e-6
                T_ref = 293.15  # reference temperature [K]
                dT = float(np.mean(T)) - T_ref
                thermal_force = self.material.youngs_modulus * alpha_th * dT * self.A
                self.add_distributed_load("thermal", -thermal_force / self.length)
            if self._coupled_state.has("heat_flux"):
                q = self._coupled_state.get_field("heat_flux").data
                thermal_stress_load = float(np.mean(np.abs(q))) * self.c / self.material.youngs_modulus
                self._fem.set_thermal_stress_factor(thermal_stress_load) if hasattr(self._fem, 'set_thermal_stress_factor') else None
        result = self.solve()
        self._time += dt
        return self.export_state(result)


# ---------------------------------------------------------------------------
# Level 3: 2D Plane-Stress Elasticity Solver (Navier-Cauchy)
# ---------------------------------------------------------------------------

@dataclass
class StructuralAnalysis2DResult:
    """Result container for 2D plane-stress/strain elasticity analysis.

    Attributes
    ----------
    ux : np.ndarray
        x-displacement field [m], shape (ny, nx).
    uy : np.ndarray
        y-displacement field [m], shape (ny, nx).
    sigma_xx : np.ndarray
        Normal stress in x [Pa], shape (ny, nx).
    sigma_yy : np.ndarray
        Normal stress in y [Pa], shape (ny, nx).
    sigma_xy : np.ndarray
        Shear stress [Pa], shape (ny, nx).
    epsilon_xx : np.ndarray
        Normal strain in x, shape (ny, nx).
    epsilon_yy : np.ndarray
        Normal strain in y, shape (ny, nx).
    epsilon_xy : np.ndarray
        Shear strain, shape (ny, nx).
    von_mises : np.ndarray
        Von Mises equivalent stress [Pa], shape (ny, nx).
    max_von_mises : float
        Peak von Mises stress [Pa].
    max_displacement : float
        Peak displacement magnitude [m].
    x : np.ndarray
        x-coordinates, shape (nx,).
    y : np.ndarray
        y-coordinates, shape (ny,).
    iterations : int
        Number of solver iterations.
    """
    ux: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    uy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    sigma_xx: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    sigma_yy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    sigma_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    epsilon_xx: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    epsilon_yy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    epsilon_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    von_mises: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    max_von_mises: float = 0.0
    max_displacement: float = 0.0
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    iterations: int = 0


class StructuralAnalysis2DSolver:
    """2D plane-stress elasticity solver using Navier-Cauchy equations.

    Solves the 2D equilibrium equations on a rectangular domain via
    iterative relaxation (Gauss-Seidel / SOR):

        (lambda + 2*mu) * d^2 ux/dx^2 + mu * d^2 ux/dy^2
            + (lambda + mu) * d^2 uy/dxdy + fx = 0

        mu * d^2 uy/dx^2 + (lambda + 2*mu) * d^2 uy/dy^2
            + (lambda + mu) * d^2 ux/dxdy + fy = 0

    Boundary conditions:
        - Left edge: fixed (ux = uy = 0) -- Dirichlet
        - Right edge: applied traction (sigma_xx = tx, sigma_xy = ty)
        - Top/bottom: free surfaces (Neumann: zero traction)

    Parameters
    ----------
    nx, ny : int
        Grid points in x and y.
    Lx, Ly : float
        Domain size [m].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    plane_stress : bool
        If True, plane stress; if False, plane strain.
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M3_COUPLED

    def __init__(
        self,
        nx: int = 41,
        ny: int = 21,
        Lx: float = 1.0,
        Ly: float = 0.5,
        E: float = 70e9,
        nu: float = 0.33,
        plane_stress: bool = True,
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.E = E
        self.nu = nu
        self.plane_stress = plane_stress

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)

        # Lame parameters
        if plane_stress:
            self.lam = E * nu / (1.0 - nu**2)
            self.mu = E / (2.0 * (1.0 + nu))
        else:
            self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            self.mu = E / (2.0 * (1.0 + nu))

        # Coupling state
        self._coupled_state = None
        self._time = 0.0

    def solve(
        self,
        traction_x: float = 1e6,
        traction_y: float = 0.0,
        body_force_x: Optional[np.ndarray] = None,
        body_force_y: Optional[np.ndarray] = None,
        max_iter: int = 5000,
        tol: float = 1e-6,
        omega: float = 1.4,
    ) -> StructuralAnalysis2DResult:
        """Solve the 2D plane-stress/strain elasticity problem.

        Parameters
        ----------
        traction_x : float
            Applied normal traction on right edge [Pa].
        traction_y : float
            Applied shear traction on right edge [Pa].
        body_force_x, body_force_y : np.ndarray, optional
            Body force fields [N/m^3], shape (ny, nx).
        max_iter : int
            Maximum SOR iterations.
        tol : float
            Convergence tolerance on displacement residual.
        omega : float
            SOR relaxation factor (1 < omega < 2).

        Returns
        -------
        StructuralAnalysis2DResult
        """
        nx, ny = self.nx, self.ny
        dx, dy = self.dx, self.dy
        lam, mu = self.lam, self.mu

        ux = np.zeros((ny, nx))
        uy = np.zeros((ny, nx))

        fx = body_force_x if body_force_x is not None else np.zeros((ny, nx))
        fy = body_force_y if body_force_y is not None else np.zeros((ny, nx))

        dx2 = dx**2
        dy2 = dy**2

        converged_iter = max_iter
        for it in range(max_iter):
            ux_old = ux.copy()
            uy_old = uy.copy()

            # Interior point update for ux
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    term_xx = (lam + 2 * mu) * (ux[j, i + 1] + ux[j, i - 1]) / dx2
                    term_yy = mu * (ux[j + 1, i] + ux[j - 1, i]) / dy2
                    term_xy = (lam + mu) * (
                        uy[j + 1, i + 1] - uy[j + 1, i - 1]
                        - uy[j - 1, i + 1] + uy[j - 1, i - 1]
                    ) / (4.0 * dx * dy)
                    denom = 2.0 * (lam + 2 * mu) / dx2 + 2.0 * mu / dy2
                    ux_new = (term_xx + term_yy + term_xy + fx[j, i]) / denom
                    ux[j, i] += omega * (ux_new - ux[j, i])

            # Interior point update for uy
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    term_xx = mu * (uy[j, i + 1] + uy[j, i - 1]) / dx2
                    term_yy = (lam + 2 * mu) * (uy[j + 1, i] + uy[j - 1, i]) / dy2
                    term_xy = (lam + mu) * (
                        ux[j + 1, i + 1] - ux[j + 1, i - 1]
                        - ux[j - 1, i + 1] + ux[j - 1, i - 1]
                    ) / (4.0 * dx * dy)
                    denom = 2.0 * mu / dx2 + 2.0 * (lam + 2 * mu) / dy2
                    uy_new = (term_xx + term_yy + term_xy + fy[j, i]) / denom
                    uy[j, i] += omega * (uy_new - uy[j, i])

            # BC: left edge fixed
            ux[:, 0] = 0.0
            uy[:, 0] = 0.0

            # BC: right edge traction (Neumann)
            # sigma_xx = (lam+2mu)*dux/dx + lam*duy/dy = traction_x
            # Approximate: dux/dx ~ (ux[i] - ux[i-1])/dx
            for j in range(1, ny - 1):
                duy_dy = (uy[j + 1, -1] - uy[j - 1, -1]) / (2.0 * dy)
                ux[j, -1] = ux[j, -2] + dx / (lam + 2 * mu) * (
                    traction_x - lam * duy_dy
                )
                # Shear traction: sigma_xy = mu*(dux/dy + duy/dx) = traction_y
                dux_dy = (ux[j + 1, -1] - ux[j - 1, -1]) / (2.0 * dy)
                uy[j, -1] = uy[j, -2] + dx / mu * traction_y - dx * dux_dy

            # BC: top/bottom free (zero traction in y-direction)
            ux[0, :] = ux[1, :]
            ux[-1, :] = ux[-2, :]
            uy[0, :] = uy[1, :]
            uy[-1, :] = uy[-2, :]

            # Check convergence
            res = np.max(np.abs(ux - ux_old)) + np.max(np.abs(uy - uy_old))
            if res < tol:
                converged_iter = it + 1
                break

        # Compute strains (central differences)
        exx = np.zeros((ny, nx))
        eyy = np.zeros((ny, nx))
        exy = np.zeros((ny, nx))
        exx[:, 1:-1] = (ux[:, 2:] - ux[:, :-2]) / (2.0 * dx)
        eyy[1:-1, :] = (uy[2:, :] - uy[:-2, :]) / (2.0 * dy)
        exy[1:-1, 1:-1] = 0.5 * (
            (ux[2:, 1:-1] - ux[:-2, 1:-1]) / (2.0 * dy)
            + (uy[1:-1, 2:] - uy[1:-1, :-2]) / (2.0 * dx)
        )

        # Compute stresses
        sxx = (lam + 2 * mu) * exx + lam * eyy
        syy = lam * exx + (lam + 2 * mu) * eyy
        sxy = 2.0 * mu * exy

        # Von Mises
        von_mises = np.sqrt(sxx**2 - sxx * syy + syy**2 + 3.0 * sxy**2)

        disp_mag = np.sqrt(ux**2 + uy**2)

        return StructuralAnalysis2DResult(
            ux=ux, uy=uy,
            sigma_xx=sxx, sigma_yy=syy, sigma_xy=sxy,
            epsilon_xx=exx, epsilon_yy=eyy, epsilon_xy=exy,
            von_mises=von_mises,
            max_von_mises=float(np.max(von_mises)),
            max_displacement=float(np.max(disp_mag)),
            x=self.x, y=self.y,
            iterations=converged_iter,
        )

    def export_state(self, result: StructuralAnalysis2DResult) -> PhysicsState:
        """Export 2D structural result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="structural_analysis_2d")
        state.set_field("displacement_x", result.ux, "m")
        state.set_field("displacement_y", result.uy, "m")
        state.set_field("stress_von_mises", result.von_mises, "Pa")
        state.metadata["max_von_mises"] = result.max_von_mises
        state.metadata["max_displacement"] = result.max_displacement
        state.metadata["iterations"] = result.iterations
        return state

    def import_state(self, state: PhysicsState) -> None:
        """Import coupled fields from partner solvers."""
        self._coupled_state = state

    def advance(self, dt: float) -> PhysicsState:
        """Re-solve 2D elasticity with coupled thermal and body force fields."""
        body_fx = None
        body_fy = None
        if self._coupled_state is not None:
            if self._coupled_state.has("temperature"):
                T = self._coupled_state.get_field("temperature").data
                # Thermal stress body force: f = -E*alpha*grad(T)/(1-2*nu)
                alpha_th = 12e-6  # thermal expansion [1/K]
                T_ref = 293.15
                dT = T - T_ref if T.shape == (self.ny, self.nx) else np.full((self.ny, self.nx), float(np.mean(T)) - T_ref)
                coeff = -self.E * alpha_th / (1.0 - 2.0 * self.nu + 1e-30)
                body_fx = np.zeros((self.ny, self.nx))
                body_fy = np.zeros((self.ny, self.nx))
                body_fx[:, 1:-1] = coeff * (dT[:, 2:] - dT[:, :-2]) / (2.0 * self.dx)
                body_fy[1:-1, :] = coeff * (dT[2:, :] - dT[:-2, :]) / (2.0 * self.dy)
            if self._coupled_state.has("body_force_y"):
                bf = self._coupled_state.get_field("body_force_y").data
                body_fy = bf if body_fy is None else body_fy + bf
        result = self.solve(body_force_x=body_fx, body_force_y=body_fy)
        self._time += dt
        return self.export_state(result)
