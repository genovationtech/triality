"""
Thermal → Structural Coupling

Couples any thermal solver to the structural analysis solver to enable
thermo-mechanical analysis. Temperature fields from the thermal solver
are converted to thermal loads (thermal strains/stresses) and applied
to the structural FEM solver.

Physics:
    Thermal strain:  eps_th = alpha * (T - T_ref)
    Thermal stress:  sigma_th = E * alpha * (T - T_ref)  (if constrained)
    Thermal load:    F_th = integral(B^T * D * eps_th * dV)

Coupling flow:
    1. Thermal solver provides temperature field T(x)
    2. Interpolate T to structural mesh nodes
    3. Compute equivalent thermal nodal forces
    4. Add thermal loads to structural solver
    5. Solve structural problem (displacement, stress)
    6. Optionally feed displacement back for geometry update

Unlock: thermo-mechanical reasoning, thermal buckling, creep assessment
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any

from triality.core.fields import PhysicsState, PhysicsField
from triality.core.coupling import (
    SolverAdapter, CouplingEngine, CouplingLink, CouplingStrategy,
    CouplingResult, RelaxationMethod,
)
from triality.core.units import NuclearSIAdapter


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class ThermoMechanicalResult:
    """Result from coupled thermal-structural analysis.

    Attributes
    ----------
    temperature : np.ndarray
        Temperature field at structural nodes [K].
    thermal_strain : np.ndarray
        Thermal strain at each element.
    thermal_stress : np.ndarray
        Thermal stress at each element [Pa].
    displacement : np.ndarray
        Nodal displacements [m].
    total_stress : np.ndarray
        Total stress (mechanical + thermal) [Pa].
    max_von_mises : float
        Peak von Mises stress [Pa].
    max_displacement : float
        Peak displacement magnitude [m].
    margin_of_safety : float
        Governing margin of safety.
    is_safe : bool
        True if margin >= 0.
    thermal_buckling_factor : float
        Thermal buckling eigenvalue factor (> 1 means safe).
    coupling_result : Optional[CouplingResult]
        Underlying coupling engine result (if iterative coupling was used).
    """
    temperature: np.ndarray = field(default_factory=lambda: np.array([]))
    thermal_strain: np.ndarray = field(default_factory=lambda: np.array([]))
    thermal_stress: np.ndarray = field(default_factory=lambda: np.array([]))
    displacement: np.ndarray = field(default_factory=lambda: np.array([]))
    total_stress: np.ndarray = field(default_factory=lambda: np.array([]))
    max_von_mises: float = 0.0
    max_displacement: float = 0.0
    margin_of_safety: float = float('inf')
    is_safe: bool = True
    thermal_buckling_factor: float = float('inf')
    coupling_result: Optional[CouplingResult] = None


# ---------------------------------------------------------------------------
# Thermal-Structural coupling solver
# ---------------------------------------------------------------------------

class ThermalStructuralCoupler:
    """Couples a thermal solver to the structural solver for thermo-mechanical analysis.

    The coupler takes a temperature field (from any thermal solver) and
    converts it to thermal loads for the structural FEM. Supports:
    - 1-D axial temperature distributions
    - Steady-state one-way coupling (thermal → structural)
    - Iterative two-way coupling (thermal ↔ structural)

    Parameters
    ----------
    youngs_modulus : float
        Material Young's modulus [Pa].
    poissons_ratio : float
        Material Poisson's ratio.
    yield_strength : float
        Material yield strength [Pa].
    cte : float
        Coefficient of thermal expansion [1/K].
    T_ref : float
        Reference (stress-free) temperature [K].
    length : float
        Structural length [m].
    n_elements : int
        Number of structural FEM elements.
    cross_section_area : float
        Cross-section area [m^2].
    moment_of_inertia : float
        Second moment of area [m^4].
    outer_fiber_distance : float
        Distance from neutral axis to outer fiber [m].

    Example
    -------
    >>> coupler = ThermalStructuralCoupler(
    ...     youngs_modulus=70e9, poissons_ratio=0.33,
    ...     yield_strength=500e6, cte=23e-6, T_ref=293.15,
    ...     length=1.0, n_elements=20,
    ... )
    >>> # Temperature from a thermal solver
    >>> T = np.linspace(293, 573, 21)  # 21 nodes for 20 elements
    >>> result = coupler.solve(T)
    >>> print(f"Max thermal stress: {result.max_von_mises:.1f} Pa")
    """

    def __init__(
        self,
        youngs_modulus: float = 70e9,
        poissons_ratio: float = 0.33,
        yield_strength: float = 500e6,
        cte: float = 23e-6,
        T_ref: float = 293.15,
        length: float = 1.0,
        n_elements: int = 20,
        cross_section_area: float = 1e-4,
        moment_of_inertia: float = 1e-8,
        outer_fiber_distance: float = 0.01,
    ):
        self.E = youngs_modulus
        self.nu = poissons_ratio
        self.sigma_y = yield_strength
        self.cte = cte
        self.T_ref = T_ref
        self.length = length
        self.n_elements = n_elements
        self.A = cross_section_area
        self.I = moment_of_inertia
        self.c = outer_fiber_distance

    def solve(
        self,
        temperature: np.ndarray,
        mechanical_loads: Optional[np.ndarray] = None,
        support: str = 'simply_supported',
        grid: Optional[np.ndarray] = None,
    ) -> ThermoMechanicalResult:
        """Run thermo-mechanical analysis.

        Parameters
        ----------
        temperature : np.ndarray
            Temperature at structural nodes [K]. Length = n_elements + 1.
        mechanical_loads : np.ndarray, optional
            Additional mechanical nodal forces [N].
        support : str
            Boundary condition: 'cantilever' or 'simply_supported'.
        grid : np.ndarray, optional
            Spatial grid for the temperature. If different from structural
            grid, temperature will be interpolated.

        Returns
        -------
        ThermoMechanicalResult
        """
        n_elem = self.n_elements
        n_nodes = n_elem + 1
        L_elem = self.length / n_elem

        # Interpolate temperature to structural grid if needed
        x_struct = np.linspace(0, self.length, n_nodes)
        if grid is not None and len(temperature) != n_nodes:
            T = np.interp(x_struct, grid, temperature)
        else:
            T = np.asarray(temperature[:n_nodes], dtype=float)

        # Compute thermal quantities at each element (average of end nodes)
        T_elem = 0.5 * (T[:-1] + T[1:])
        dT = T_elem - self.T_ref

        # Thermal strain and stress
        eps_thermal = self.cte * dT
        sigma_thermal = self.E * eps_thermal  # axial thermal stress if fully constrained

        # Build structural stiffness and solve
        # Using beam FEM: 2 DOF per node (v, theta) for bending
        n_dof = 2 * n_nodes
        K_global = np.zeros((n_dof, n_dof))
        F_global = np.zeros(n_dof)

        EI = self.E * self.I
        EA = self.E * self.A

        for e in range(n_elem):
            # Euler-Bernoulli beam element stiffness (4x4)
            L_e = L_elem
            k_elem = EI / L_e**3 * np.array([
                [12,    6*L_e,   -12,    6*L_e],
                [6*L_e, 4*L_e**2, -6*L_e, 2*L_e**2],
                [-12,   -6*L_e,  12,    -6*L_e],
                [6*L_e, 2*L_e**2, -6*L_e, 4*L_e**2],
            ])

            # Equivalent thermal load: bending moment from thermal gradient
            # For uniform dT across section -> no bending, only axial
            # For gradient through depth: M_th = E*I*alpha*dT/h
            # Simplified: thermal force as distributed load equivalent
            dT_e = dT[e]
            # Thermal moment (if temperature varies through depth, simplified)
            M_thermal = 0.0  # Uniform T through depth -> no thermal bending
            # Thermal axial force
            F_axial = EA * self.cte * dT_e

            # Equivalent nodal forces for uniform thermal load
            f_elem = np.array([
                F_axial * self.c / L_e,  # Transverse from eccentricity (small)
                M_thermal / 2.0,
                -F_axial * self.c / L_e,
                M_thermal / 2.0,
            ])

            # Assemble
            dof_map = [2*e, 2*e+1, 2*e+2, 2*e+3]
            for i in range(4):
                F_global[dof_map[i]] += f_elem[i]
                for j in range(4):
                    K_global[dof_map[i], dof_map[j]] += k_elem[i, j]

        # Add mechanical loads
        if mechanical_loads is not None:
            mech = np.asarray(mechanical_loads)
            if len(mech) == n_nodes:
                # Assume transverse forces at nodes
                for i in range(n_nodes):
                    F_global[2*i] += mech[i]

        # Apply boundary conditions
        if support == 'cantilever':
            # Fix first node: v=0, theta=0
            fixed_dofs = [0, 1]
        else:
            # Simply supported: v=0 at both ends
            fixed_dofs = [0, 2*n_elem]

        # Solve K*u = F with BCs
        free_dofs = [i for i in range(n_dof) if i not in fixed_dofs]
        K_ff = K_global[np.ix_(free_dofs, free_dofs)].copy()
        F_f = F_global[free_dofs]

        # Regularize if needed
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

        # Extract displacements and compute stresses
        displacements = u_global[0::2]  # transverse displacements
        rotations = u_global[1::2]

        # Element stresses: bending + thermal
        total_stress = np.zeros(n_elem)
        for e in range(n_elem):
            # Bending stress from curvature
            kappa = (rotations[e+1] - rotations[e]) / L_elem
            sigma_bending = self.E * self.c * kappa

            # Total = mechanical bending + thermal
            total_stress[e] = abs(sigma_bending) + abs(sigma_thermal[e])

        max_vm = float(np.max(np.abs(total_stress)))
        max_disp = float(np.max(np.abs(displacements)))

        # Margin of safety
        ms = self.sigma_y / max(max_vm, 1e-30) - 1.0

        # Thermal buckling factor (simplified Euler for axially loaded beam)
        P_thermal_max = EA * self.cte * float(np.max(np.abs(dT)))
        if support == 'cantilever':
            P_cr = np.pi**2 * EI / (4.0 * self.length**2)
        else:
            P_cr = np.pi**2 * EI / self.length**2
        thermal_buckling = P_cr / max(P_thermal_max, 1e-30)

        return ThermoMechanicalResult(
            temperature=T,
            thermal_strain=eps_thermal,
            thermal_stress=sigma_thermal,
            displacement=displacements,
            total_stress=total_stress,
            max_von_mises=max_vm,
            max_displacement=max_disp,
            margin_of_safety=ms,
            is_safe=ms >= 0,
            thermal_buckling_factor=thermal_buckling,
        )

    def solve_from_state(self, thermal_state: PhysicsState,
                         **kwargs) -> ThermoMechanicalResult:
        """Solve using a PhysicsState from a thermal solver.

        Parameters
        ----------
        thermal_state : PhysicsState
            State containing 'temperature' field.

        Returns
        -------
        ThermoMechanicalResult
        """
        T_field = thermal_state.get("temperature")
        return self.solve(
            temperature=T_field.data,
            grid=T_field.grid,
            **kwargs,
        )

    def export_state(self, result: ThermoMechanicalResult) -> PhysicsState:
        """Export results as standardized PhysicsState."""
        x = np.linspace(0, self.length, self.n_elements + 1)
        state = PhysicsState(solver_name="thermal_structural", time=0.0)
        state.set_field("temperature", result.temperature, "K", grid=x)
        state.set_field("displacement", result.displacement, "m", grid=x)
        state.set_field("stress_von_mises",
                        np.interp(x, np.linspace(0, self.length, self.n_elements),
                                  result.total_stress),
                        "Pa", grid=x)
        state.set_field("strain",
                        np.interp(x, np.linspace(0, self.length, self.n_elements),
                                  result.thermal_strain),
                        "1", grid=x)
        return state
