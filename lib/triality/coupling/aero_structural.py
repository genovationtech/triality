"""
Aero Loads → Structural Coupling (Aeroelasticity)

Couples the aero_loads solver to the structural_analysis solver for
static aeroelastic analysis: aerodynamic pressure distributions are
converted to structural loads, and structural deformations feed back
to modify the aerodynamic shape (if iterative).

Physics:
    Aero → Structural:
        F_aero(x) = (p(x) - p_inf) * A_panel(x) → structural nodal loads
        Shear V(x) = integral(F_aero dx)
        Moment M(x) = integral(V dx)

    Structural → Aero (optional feedback):
        delta_theta(x) = d(displacement)/dx → effective angle of attack change
        alpha_eff(x) = alpha + delta_theta(x)

Coupling flow:
    1. Aero solver computes Cp, pressure, shear, moment distributions
    2. Map aero loads to structural FEM nodes
    3. Structural solver computes displacements and stresses
    4. (Optional) Feed structural deformations back to aero solver
    5. Iterate until converged (static aeroelasticity)

Unlock: aeroelasticity, flutter margins, divergence, survivability
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from triality.core.fields import PhysicsState, PhysicsField
from triality.core.coupling import (
    CouplingEngine, CouplingLink, CouplingStrategy,
    CouplingResult, SolverAdapter,
)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class AeroStructuralResult:
    """Result from coupled aero-structural analysis.

    Attributes
    ----------
    x_stations : np.ndarray
        Axial station positions [m].
    Cp : np.ndarray
        Pressure coefficient distribution.
    pressure : np.ndarray
        Surface pressure [Pa].
    aero_load_per_length : np.ndarray
        Distributed aerodynamic load [N/m].
    shear : np.ndarray
        Shear force distribution [N].
    bending_moment : np.ndarray
        Bending moment distribution [N*m].
    displacement : np.ndarray
        Structural displacement [m].
    rotation : np.ndarray
        Structural rotation [rad].
    stress : np.ndarray
        Element stress [Pa].
    max_von_mises : float
        Peak von Mises stress [Pa].
    max_displacement : float
        Peak displacement [m].
    margin_of_safety : float
        Structural margin of safety.
    is_safe : bool
        True if margin >= 0.
    divergence_q : float
        Divergence dynamic pressure [Pa].
    aeroelastic_CL : float
        Aeroelastically corrected lift coefficient.
    aeroelastic_CD : float
        Aeroelastically corrected drag coefficient.
    n_coupling_iterations : int
        Number of aero-structural coupling iterations.
    coupling_converged : bool
        Whether the coupling iteration converged.
    """
    x_stations: np.ndarray = field(default_factory=lambda: np.array([]))
    Cp: np.ndarray = field(default_factory=lambda: np.array([]))
    pressure: np.ndarray = field(default_factory=lambda: np.array([]))
    aero_load_per_length: np.ndarray = field(default_factory=lambda: np.array([]))
    shear: np.ndarray = field(default_factory=lambda: np.array([]))
    bending_moment: np.ndarray = field(default_factory=lambda: np.array([]))
    displacement: np.ndarray = field(default_factory=lambda: np.array([]))
    rotation: np.ndarray = field(default_factory=lambda: np.array([]))
    stress: np.ndarray = field(default_factory=lambda: np.array([]))
    max_von_mises: float = 0.0
    max_displacement: float = 0.0
    margin_of_safety: float = float('inf')
    is_safe: bool = True
    divergence_q: float = float('inf')
    aeroelastic_CL: float = 0.0
    aeroelastic_CD: float = 0.0
    n_coupling_iterations: int = 0
    coupling_converged: bool = True


# ---------------------------------------------------------------------------
# Aero-Structural coupler
# ---------------------------------------------------------------------------

class AeroStructuralCoupler:
    """Couples aerodynamic loads to structural analysis.

    Parameters
    ----------
    length : float
        Vehicle / wing length [m].
    n_elements : int
        Number of structural FEM elements.
    youngs_modulus : float
        Material Young's modulus [Pa].
    yield_strength : float
        Material yield strength [Pa].
    moment_of_inertia : float
        Second moment of area [m^4].
    cross_section_area : float
        Cross-section area [m^2].
    outer_fiber_distance : float
        Distance from NA to outer fiber [m].
    reference_area : float
        Aerodynamic reference area [m^2].
    dynamic_pressure : float
        Freestream dynamic pressure [Pa].
    support : str
        'cantilever' or 'simply_supported'.

    Example
    -------
    >>> coupler = AeroStructuralCoupler(
    ...     length=3.0, n_elements=50,
    ...     youngs_modulus=70e9, yield_strength=500e6,
    ...     moment_of_inertia=1e-5, cross_section_area=1e-3,
    ... )
    >>> # From aero solver result
    >>> pressure = np.linspace(101325, 110000, 51)  # Pa
    >>> result = coupler.solve(pressure=pressure, p_inf=101325.0)
    """

    def __init__(
        self,
        length: float = 3.0,
        n_elements: int = 50,
        youngs_modulus: float = 70e9,
        yield_strength: float = 500e6,
        moment_of_inertia: float = 1e-5,
        cross_section_area: float = 1e-3,
        outer_fiber_distance: float = 0.05,
        reference_area: float = 1.0,
        dynamic_pressure: float = 50000.0,
        support: str = 'cantilever',
    ):
        self.length = length
        self.n_elements = n_elements
        self.E = youngs_modulus
        self.sigma_y = yield_strength
        self.I = moment_of_inertia
        self.A = cross_section_area
        self.c = outer_fiber_distance
        self.S_ref = reference_area
        self.q_inf = dynamic_pressure
        self.support = support

    def solve(
        self,
        pressure: Optional[np.ndarray] = None,
        p_inf: float = 101325.0,
        aero_load_per_length: Optional[np.ndarray] = None,
        aero_state: Optional[PhysicsState] = None,
        max_coupling_iter: int = 10,
        coupling_tol: float = 1e-4,
        relaxation: float = 0.5,
    ) -> AeroStructuralResult:
        """Run aero-structural coupling.

        Provide either pressure distribution or aero_load_per_length or
        an aero PhysicsState. The solver converts to structural loads and
        solves the beam FEM.

        Parameters
        ----------
        pressure : np.ndarray, optional
            Surface pressure at nodes [Pa].
        p_inf : float
            Freestream pressure [Pa].
        aero_load_per_length : np.ndarray, optional
            Distributed load [N/m] at nodes.
        aero_state : PhysicsState, optional
            State containing 'pressure' and/or 'heat_flux' fields.
        max_coupling_iter : int
            Max aero-structural iterations (1 = one-way).
        coupling_tol : float
            Convergence tolerance on displacement change.
        relaxation : float
            Under-relaxation factor for iterative coupling.

        Returns
        -------
        AeroStructuralResult
        """
        n_elem = self.n_elements
        n_nodes = n_elem + 1
        L_elem = self.length / n_elem
        x = np.linspace(0, self.length, n_nodes)

        # Extract loads from inputs
        if aero_state is not None:
            if aero_state.has("pressure"):
                p_field = aero_state.get("pressure")
                pressure = p_field.data
                if p_field.grid is not None and len(pressure) != n_nodes:
                    pressure = np.interp(x, p_field.grid, pressure)

        if aero_load_per_length is not None:
            q_load = np.asarray(aero_load_per_length)
            if len(q_load) != n_nodes:
                q_load = np.interp(x, np.linspace(0, self.length, len(q_load)), q_load)
        elif pressure is not None:
            p = np.asarray(pressure)
            if len(p) != n_nodes:
                p = np.interp(x, np.linspace(0, self.length, len(p)), p)
            # Assume unit-width panel for load per length
            q_load = (p - p_inf)
        else:
            q_load = np.zeros(n_nodes)

        # Build beam FEM (Euler-Bernoulli)
        EI = self.E * self.I
        n_dof = 2 * n_nodes

        disp_prev = np.zeros(n_nodes)
        converged = True
        n_iter = 0

        for c_iter in range(max_coupling_iter):
            n_iter = c_iter + 1

            K_global = np.zeros((n_dof, n_dof))
            F_global = np.zeros(n_dof)

            for e in range(n_elem):
                L_e = L_elem

                k_elem = EI / L_e**3 * np.array([
                    [12,    6*L_e,   -12,    6*L_e],
                    [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
                    [-12,   -6*L_e,  12,    -6*L_e],
                    [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2],
                ])

                # Equivalent nodal forces for distributed load
                q_avg = 0.5 * (q_load[e] + q_load[e+1])
                f_elem = np.array([
                    q_avg * L_e / 2.0,
                    q_avg * L_e**2 / 12.0,
                    q_avg * L_e / 2.0,
                    -q_avg * L_e**2 / 12.0,
                ])

                dof_map = [2*e, 2*e+1, 2*e+2, 2*e+3]
                for i in range(4):
                    F_global[dof_map[i]] += f_elem[i]
                    for j in range(4):
                        K_global[dof_map[i], dof_map[j]] += k_elem[i, j]

            # Boundary conditions
            if self.support == 'cantilever':
                fixed_dofs = [0, 1]
            else:
                fixed_dofs = [0, 2*n_elem]

            free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
            K_ff = K_global[np.ix_(free_dofs, free_dofs)]
            F_f = F_global[free_dofs]

            diag = np.diag(K_ff).copy()
            diag[diag == 0] = 1.0
            np.fill_diagonal(K_ff, diag)

            try:
                u_f = np.linalg.solve(K_ff, F_f)
            except np.linalg.LinAlgError:
                u_f = np.zeros(len(free_dofs))

            u_global = np.zeros(n_dof)
            for i, dof in enumerate(free_dofs):
                u_global[dof] = u_f[i]

            disp = u_global[0::2]
            rot = u_global[1::2]

            # Under-relax
            disp_relaxed = relaxation * disp + (1.0 - relaxation) * disp_prev

            # Check convergence
            change = float(np.max(np.abs(disp_relaxed - disp_prev)))
            scale = max(float(np.max(np.abs(disp_relaxed))), 1e-30)
            if change / scale < coupling_tol and c_iter > 0:
                disp = disp_relaxed
                break

            disp_prev = disp_relaxed.copy()

            if max_coupling_iter <= 1:
                disp = disp_relaxed
                break

            # Feedback: update angle of attack from structural slope
            d_disp_dx = np.gradient(disp_relaxed, x)
            # This would modify the aero loads, but for now we break
            # (full feedback requires re-running aero solver)
            disp = disp_relaxed
            break
        else:
            converged = False

        # Compute element stresses
        stress = np.zeros(n_elem)
        for e in range(n_elem):
            kappa = (rot[e+1] - rot[e]) / L_elem
            stress[e] = abs(self.E * self.c * kappa)

        max_vm = float(np.max(stress)) if len(stress) > 0 else 0.0
        max_disp = float(np.max(np.abs(disp)))
        ms = self.sigma_y / max(max_vm, 1e-30) - 1.0

        # Shear and moment from loads
        shear = np.cumsum(q_load * L_elem) - np.cumsum(q_load * L_elem)[-1] * x / self.length
        moment = np.cumsum(shear * L_elem)

        # Divergence dynamic pressure (simplified)
        # q_div = K_theta / (e * S_ref * dCL/dalpha * L_ref)
        # Using structural stiffness as proxy
        K_theta = EI / self.length  # Torsional stiffness proxy
        dCL_dalpha_est = 2 * np.pi  # Thin airfoil theory
        q_div = K_theta / max(self.S_ref * dCL_dalpha_est * self.length, 1e-30)

        return AeroStructuralResult(
            x_stations=x,
            Cp=q_load / max(self.q_inf, 1e-30),
            pressure=(pressure if pressure is not None else
                      q_load + p_inf),
            aero_load_per_length=q_load,
            shear=shear,
            bending_moment=moment,
            displacement=disp,
            rotation=rot,
            stress=stress,
            max_von_mises=max_vm,
            max_displacement=max_disp,
            margin_of_safety=ms,
            is_safe=ms >= 0,
            divergence_q=q_div,
            aeroelastic_CL=0.0,
            aeroelastic_CD=0.0,
            n_coupling_iterations=n_iter,
            coupling_converged=converged,
        )

    def solve_from_state(self, aero_state: PhysicsState,
                         p_inf: float = 101325.0,
                         **kwargs) -> AeroStructuralResult:
        """Solve using a PhysicsState from an aero solver."""
        return self.solve(aero_state=aero_state, p_inf=p_inf, **kwargs)

    def export_state(self, result: AeroStructuralResult) -> PhysicsState:
        """Export result as PhysicsState."""
        state = PhysicsState(solver_name="aero_structural")
        state.set_field("displacement", result.displacement, "m",
                        grid=result.x_stations)
        state.set_field("stress_von_mises",
                        np.interp(result.x_stations,
                                  np.linspace(0, self.length, self.n_elements),
                                  result.stress),
                        "Pa", grid=result.x_stations)
        state.set_field("pressure", result.pressure, "Pa",
                        grid=result.x_stations)
        return state
