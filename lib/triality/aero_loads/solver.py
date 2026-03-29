"""
Aero loads solver.

Computes distributed aerodynamic loads (pressures, forces, moments) and
heating over a discretised vehicle surface for a given set of flight
conditions using Newtonian impact theory, flat-plate heating correlations,
and panel-based load integration.

Wires in existing FreestreamConditions, NewtonianFlow, SimpleShapes,
PrandtlMeyerExpansion, HeatingConditions, DistributedHeating,
HeatLoadIntegration, SurfaceHeatingMap, PressurePanel, AeroLoads,
LoadIntegration, StructuralLoadDistribution, PressureCoefficientAnalysis,
and AeroelasticLoads.

Physics:
    Newtonian Cp:      Cp = Cp_max * sin^2(theta)
    Force integration:  F = int (p - p_inf) * n dA
    Moment integration: M = int r x dF
    Flat-plate heating: St_lam = 0.332/Re^0.5 * Pr^(-2/3)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from triality.core.units import UnitMetadata
from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from .newtonian_flow import (
    FreestreamConditions,
    NewtonianFlow,
    SimpleShapes,
    PrandtlMeyerExpansion,
)
from .distributed_heating import (
    HeatingConditions,
    DistributedHeating,
    HeatLoadIntegration,
    SurfaceHeatingMap,
)
from .load_integration import (
    PressurePanel,
    AeroLoads,
    LoadIntegration,
    StructuralLoadDistribution,
    PressureCoefficientAnalysis,
    AeroelasticLoads,
)


@dataclass
class AeroLoadsSolverResult:
    """Result container for the aero loads solver.

    Attributes:
        x_stations: Axial station positions [m].
        Cp_distribution: Pressure coefficient at each station.
        pressure_Pa: Surface pressure at each station [Pa].
        heat_flux_W_m2: Surface heat flux at each station [W/m^2].
        forces: Integrated AeroLoads object.
        CL: Lift coefficient.
        CD: Drag coefficient.
        Cm: Pitching moment coefficient.
        shear_N_m: Shear force distribution [N/m].
        moment_Nm_m: Bending moment distribution [N*m/m].
        total_heat_load_W: Total integrated heat load [W].
        peak_heating_location_m: Location of peak heating [m].
        peak_heat_flux_W_m2: Peak heat flux [W/m^2].
        center_of_pressure_m: Centre of pressure location [m].
        divergence_q_Pa: Divergence dynamic pressure [Pa].
    """
    x_stations: np.ndarray = field(default_factory=lambda: np.array([]))
    Cp_distribution: np.ndarray = field(default_factory=lambda: np.array([]))
    pressure_Pa: np.ndarray = field(default_factory=lambda: np.array([]))
    heat_flux_W_m2: np.ndarray = field(default_factory=lambda: np.array([]))
    forces: Optional[AeroLoads] = None
    CL: float = 0.0
    CD: float = 0.0
    Cm: float = 0.0
    shear_N_m: np.ndarray = field(default_factory=lambda: np.array([]))
    moment_Nm_m: np.ndarray = field(default_factory=lambda: np.array([]))
    total_heat_load_W: float = 0.0
    peak_heating_location_m: float = 0.0
    peak_heat_flux_W_m2: float = 0.0
    center_of_pressure_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    divergence_q_Pa: float = float('inf')
    units: UnitMetadata = field(default_factory=UnitMetadata)

    def __post_init__(self):
        """Declare unit metadata for all fields."""
        self.units.declare("x_stations", "m", "Axial station positions")
        self.units.declare("pressure_Pa", "Pa", "Surface pressure")
        self.units.declare("heat_flux_W_m2", "W/m^2", "Surface heat flux")
        self.units.declare("shear_N_m", "N", "Shear force distribution")
        self.units.declare("moment_Nm_m", "N", "Bending moment distribution")
        self.units.declare("total_heat_load_W", "W", "Total heat load")
        self.units.declare("peak_heat_flux_W_m2", "W/m^2", "Peak heat flux")
        self.units.declare("divergence_q_Pa", "Pa", "Divergence dynamic pressure")


class AeroLoadsSolver:
    """Panel-method aero loads and heating solver.

    Discretises an axisymmetric body into panels, applies Newtonian
    pressure theory and flat-plate heating correlations, integrates
    forces and moments, and computes structural load distributions.

    Parameters
    ----------
    body_length_m : float
        Total body length [m].
    nose_radius_m : float
        Nose (leading edge) radius [m].
    base_radius_m : float
        Base radius [m].
    n_panels : int
        Number of axial panels.
    reference_area_m2 : float
        Aerodynamic reference area [m^2].
    reference_length_m : float
        Aerodynamic reference length [m] (e.g. diameter).
    """

    fidelity_tier = FidelityTier.REDUCED_ORDER
    coupling_maturity = CouplingMaturity.M3_COUPLED

    def __init__(
        self,
        body_length_m: float = 3.0,
        nose_radius_m: float = 0.15,
        base_radius_m: float = 0.15,
        n_panels: int = 100,
        reference_area_m2: Optional[float] = None,
        reference_length_m: Optional[float] = None,
    ):
        self._coupled_state = None
        self._time = 0.0
        self._freestream = None
        self._alpha_deg = 0.0
        self._wall_temperature_K = 300.0
        self.L = body_length_m
        self.R_n = nose_radius_m
        self.R_b = base_radius_m
        self.n_panels = n_panels
        self.S_ref = reference_area_m2 if reference_area_m2 else np.pi * base_radius_m**2
        self.L_ref = reference_length_m if reference_length_m else 2.0 * base_radius_m

        # Generate body geometry (ogive-cylinder)
        self._generate_body()

    # ------------------------------------------------------------------
    def _generate_body(self):
        """Discretise an ogive-cylinder body into panels."""
        x = np.linspace(0, self.L, self.n_panels + 1)
        x_mid = 0.5 * (x[:-1] + x[1:])
        dx = x[1] - x[0]

        # Ogive nose + cylindrical section
        # Nose length ~ 2 * nose radius
        L_nose = min(2.0 * self.R_n, 0.3 * self.L)

        r = np.zeros(self.n_panels)
        theta = np.zeros(self.n_panels)  # surface angle to freestream

        for i in range(self.n_panels):
            xi = x_mid[i]
            if xi < L_nose:
                # Ogive: r(x) = R_n * sqrt(2*x/L_nose - (x/L_nose)^2)
                frac = xi / L_nose
                r[i] = self.R_b * np.sqrt(max(2.0 * frac - frac**2, 0.0))
                # Local surface slope (tangent angle)
                dr_dx = self.R_b * (1.0 - frac) / max(np.sqrt(max(2.0 * frac - frac**2, 1e-10)), 1e-10) / L_nose
                theta[i] = np.arctan(dr_dx)
            else:
                # Cylinder
                r[i] = self.R_b
                theta[i] = 0.0  # parallel to freestream

        self._x_mid = x_mid
        self._r = r
        self._theta = theta
        self._dx = dx
        self._panel_area = 2.0 * np.pi * r * dx  # annular strip area

    # ------------------------------------------------------------------
    def solve(
        self,
        freestream: FreestreamConditions,
        alpha_deg: float = 0.0,
        wall_temperature_K: float = 300.0,
    ) -> AeroLoadsSolverResult:
        """Compute distributed loads and heating.

        Parameters
        ----------
        freestream : FreestreamConditions
            Freestream conditions (velocity, pressure, temperature, density).
        alpha_deg : float
            Angle of attack [deg].
        wall_temperature_K : float
            Surface wall temperature [K].

        Returns
        -------
        AeroLoadsSolverResult
        """
        alpha = np.radians(alpha_deg)
        M = freestream.mach_number
        q_inf = freestream.dynamic_pressure

        # Newtonian Cp_max
        if M >= 1.0:
            Cp_max = NewtonianFlow.modified_newtonian_Cp_max(M, freestream.gamma)
        else:
            Cp_max = 2.0  # fallback for subsonic (not physical for Newtonian)

        n_pan = self.n_panels

        # Pressure coefficients (account for angle of attack on windward/leeward)
        Cp = np.zeros(n_pan)
        for i in range(n_pan):
            # Effective incidence angle at this panel
            theta_eff = self._theta[i] + alpha
            if theta_eff > 0:
                # Windward: Newtonian
                Cp[i] = Cp_max * np.sin(theta_eff)**2
            else:
                # Leeward: Prandtl-Meyer expansion or base pressure
                Cp[i] = 0.0  # vacuum assumption for leeward

        # Surface pressure
        p_surface = freestream.pressure + q_inf * Cp

        # Build pressure panels for load integration
        panels = []
        for i in range(n_pan):
            # Normal vector for axisymmetric panel (simplified: radial + axial)
            nx = -np.sin(self._theta[i])
            nz = np.cos(self._theta[i])
            normal = np.array([nx, 0.0, nz])
            mag = np.linalg.norm(normal)
            if mag > 0:
                normal /= mag

            centroid = np.array([self._x_mid[i], 0.0, self._r[i]])
            panels.append(PressurePanel(
                centroid=centroid,
                area=self._panel_area[i],
                normal=normal,
                pressure=p_surface[i],
            ))

        # Integrate forces and moments
        loads = LoadIntegration.integrate_panels(panels, p_ref=freestream.pressure)
        coeff = LoadIntegration.force_coefficients(loads, q_inf, self.S_ref, alpha)
        mom_coeff = LoadIntegration.moment_coefficients(loads, q_inf, self.S_ref, self.L_ref)

        # Heating distribution
        heat_cond = HeatingConditions(
            velocity=freestream.velocity,
            density=freestream.density,
            temperature=freestream.temperature,
            pressure=freestream.pressure,
        )

        q_heat = np.zeros(n_pan)
        for i in range(n_pan):
            x = self._x_mid[i]
            if x < 1e-6:
                x = 1e-6
            q_heat[i] = DistributedHeating.flat_plate_laminar(
                x, heat_cond, T_wall=wall_temperature_K,
            )

        # Total heat load
        Q_total = HeatLoadIntegration.integrate_1d(
            self._x_mid, q_heat, width=2.0 * np.pi * np.mean(self._r),
        )

        # Peak heating
        x_peak, q_peak = HeatLoadIntegration.peak_location(self._x_mid, q_heat)

        # Structural load distribution (distributed normal load along axis)
        load_per_length = (p_surface - freestream.pressure) * 2.0 * np.pi * self._r
        shear, moment = StructuralLoadDistribution.shear_and_moment_distribution(
            self._x_mid, load_per_length,
        )

        # Divergence dynamic pressure estimate
        K_theta = 1e6  # representative torsional stiffness [N*m/rad]
        dCm_dalpha = mom_coeff.get('Cm', 0.0) / max(abs(alpha), 1e-6) if abs(alpha) > 1e-6 else 0.0
        q_div = AeroelasticLoads.divergence_dynamic_pressure(K_theta, dCm_dalpha, self.L_ref)

        return AeroLoadsSolverResult(
            x_stations=self._x_mid,
            Cp_distribution=Cp,
            pressure_Pa=p_surface,
            heat_flux_W_m2=q_heat,
            forces=loads,
            CL=coeff.get('CL', 0.0),
            CD=coeff.get('CD', 0.0),
            Cm=mom_coeff.get('Cm', 0.0),
            shear_N_m=shear,
            moment_Nm_m=moment,
            total_heat_load_W=Q_total,
            peak_heating_location_m=x_peak,
            peak_heat_flux_W_m2=q_peak,
            center_of_pressure_m=loads.center_of_pressure,
            divergence_q_Pa=q_div,
        )

    def export_state(self, result: AeroLoadsSolverResult) -> PhysicsState:
        """Export aero result as PhysicsState for coupling.

        Returns
        -------
        PhysicsState
            State with canonical pressure, heat flux, and force fields.
        """
        state = PhysicsState(solver_name="aero_loads")
        state.set_field("pressure", result.pressure_Pa, "Pa",
                        grid=result.x_stations)
        state.set_field("heat_flux", result.heat_flux_W_m2, "W/m^2",
                        grid=result.x_stations)
        state.set_field("pressure_coefficient", result.Cp_distribution, "1",
                        grid=result.x_stations)
        state.set_field("lift_coefficient", np.array([result.CL]), "1")
        state.set_field("drag_coefficient", np.array([result.CD]), "1")
        state.metadata["total_heat_load_W"] = result.total_heat_load_W
        state.metadata["divergence_q_Pa"] = result.divergence_q_Pa
        return state

    def import_state(self, state: PhysicsState) -> None:
        """Import coupled fields from partner solvers."""
        self._coupled_state = state

    def advance(self, dt: float) -> PhysicsState:
        """Advance solver by dt for closed-loop coupling (steady re-solve)."""
        T_wall = self._wall_temperature_K
        if self._coupled_state is not None:
            if self._coupled_state.has("wall_temperature"):
                T_data = self._coupled_state.get_field("wall_temperature").data
                T_wall = float(np.mean(T_data))
            elif self._coupled_state.has("temperature"):
                T_data = self._coupled_state.get_field("temperature").data
                T_wall = float(np.mean(T_data))
        if self._freestream is None:
            self._freestream = FreestreamConditions()
        result = self.solve(self._freestream, alpha_deg=self._alpha_deg,
                            wall_temperature_K=T_wall)
        self._time += dt
        return self.export_state(result)

    # ------------------------------------------------------------------
    def sweep_alpha(
        self,
        freestream: FreestreamConditions,
        alpha_range_deg: np.ndarray,
        wall_temperature_K: float = 300.0,
    ) -> Dict[str, np.ndarray]:
        """Sweep angle of attack and collect force coefficients.

        Parameters
        ----------
        freestream : FreestreamConditions
            Freestream conditions.
        alpha_range_deg : np.ndarray
            Array of angles of attack [deg].
        wall_temperature_K : float
            Wall temperature [K].

        Returns
        -------
        dict
            Keys: 'alpha_deg', 'CL', 'CD', 'Cm'.
        """
        n = len(alpha_range_deg)
        CL = np.zeros(n)
        CD = np.zeros(n)
        Cm = np.zeros(n)

        for i, a in enumerate(alpha_range_deg):
            res = self.solve(freestream, alpha_deg=a,
                             wall_temperature_K=wall_temperature_K)
            CL[i] = res.CL
            CD[i] = res.CD
            Cm[i] = res.Cm

        return {
            'alpha_deg': alpha_range_deg,
            'CL': CL,
            'CD': CD,
            'Cm': Cm,
        }


# ======================================================================
# Level 3: 2D Panel Method / Newtonian Flow Solver
# ======================================================================

@dataclass
class AeroLoads2DResult:
    """Result of the 2D aerodynamic loads solver.

    Attributes
    ----------
    Cp : np.ndarray
        Pressure coefficient field on the 2D surface, shape (ny, nx).
    pressure : np.ndarray
        Surface pressure field [Pa], shape (ny, nx).
    x : np.ndarray
        x coordinates [m], shape (nx,).
    y : np.ndarray
        y coordinates [m], shape (ny,).
    body_mask : np.ndarray
        Boolean mask of body cells, shape (ny, nx).
    CL : float
        Integrated lift coefficient.
    CD : float
        Integrated drag coefficient.
    Fx : float
        Integrated x-force (drag direction) [N/m].
    Fy : float
        Integrated y-force (lift direction) [N/m].
    stagnation_pressure : float
        Peak stagnation pressure [Pa].
    """
    Cp: np.ndarray = field(default_factory=lambda: np.array([]))
    pressure: np.ndarray = field(default_factory=lambda: np.array([]))
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    body_mask: np.ndarray = field(default_factory=lambda: np.array([]))
    CL: float = 0.0
    CD: float = 0.0
    Fx: float = 0.0
    Fy: float = 0.0
    stagnation_pressure: float = 0.0


class AeroLoads2DSolver:
    """2D Newtonian/panel-method flow solver over a body shape.

    Computes the pressure coefficient distribution on a 2D grid using
    modified Newtonian impact theory for high-speed flows or an
    incompressible potential flow approximation for low-speed flows.
    The body is defined by a signed-distance function on the grid.

    Integrated lift and drag forces are computed from the surface
    pressure distribution.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    Lx, Ly : float
        Domain size [m].
    mach : float
        Freestream Mach number.
    gamma : float
        Ratio of specific heats.
    rho_inf : float
        Freestream density [kg/m^3].
    p_inf : float
        Freestream pressure [Pa].
    V_inf : float
        Freestream velocity [m/s].
    alpha_deg : float
        Angle of attack [deg].
    body_type : str
        Body shape: 'cylinder', 'ellipse', 'wedge'.
    body_radius : float
        Characteristic body radius [m].
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M3_COUPLED

    def __init__(
        self,
        nx: int = 128,
        ny: int = 128,
        Lx: float = 4.0,
        Ly: float = 4.0,
        mach: float = 2.0,
        gamma: float = 1.4,
        rho_inf: float = 1.225,
        p_inf: float = 101325.0,
        V_inf: float = 680.0,
        alpha_deg: float = 0.0,
        body_type: str = "cylinder",
        body_radius: float = 0.5,
    ):
        self._coupled_state = None
        self._time = 0.0
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.mach = mach
        self.gamma = gamma
        self.rho_inf = rho_inf
        self.p_inf = p_inf
        self.V_inf = V_inf
        self.alpha_rad = np.radians(alpha_deg)
        self.body_type = body_type
        self.body_radius = body_radius
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(-Lx / 2, Lx / 2, nx)
        self.y = np.linspace(-Ly / 2, Ly / 2, ny)

    def _body_sdf(self) -> np.ndarray:
        """Compute signed distance function for the body (negative inside)."""
        XX, YY = np.meshgrid(self.x, self.y)
        if self.body_type == "cylinder":
            sdf = np.sqrt(XX**2 + YY**2) - self.body_radius
        elif self.body_type == "ellipse":
            sdf = np.sqrt((XX / (2.0 * self.body_radius))**2 +
                          (YY / self.body_radius)**2) - 1.0
            sdf *= self.body_radius
        elif self.body_type == "wedge":
            half_angle = np.radians(15.0)
            sdf_top = YY - np.tan(half_angle) * np.abs(XX)
            sdf_bottom = -YY - np.tan(half_angle) * np.abs(XX)
            sdf = np.minimum(sdf_top, sdf_bottom)
            sdf = np.where(np.abs(XX) < 2.0 * self.body_radius,
                           sdf, np.abs(YY) - self.body_radius)
        else:
            sdf = np.sqrt(XX**2 + YY**2) - self.body_radius
        return sdf

    def _surface_normal(self, sdf: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute outward surface normals from the SDF gradient."""
        ny, nx = sdf.shape
        nx_f = np.zeros((ny, nx))
        ny_f = np.zeros((ny, nx))
        nx_f[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) / (2.0 * self.dx)
        ny_f[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) / (2.0 * self.dy)
        nx_f[:, 0] = (sdf[:, 1] - sdf[:, 0]) / self.dx
        nx_f[:, -1] = (sdf[:, -1] - sdf[:, -2]) / self.dx
        ny_f[0, :] = (sdf[1, :] - sdf[0, :]) / self.dy
        ny_f[-1, :] = (sdf[-1, :] - sdf[-2, :]) / self.dy
        mag = np.sqrt(nx_f**2 + ny_f**2) + 1e-30
        return nx_f / mag, ny_f / mag

    def solve(self) -> AeroLoads2DResult:
        """Compute the 2D Cp distribution and integrated forces.

        Returns
        -------
        AeroLoads2DResult
        """
        sdf = self._body_sdf()
        body_mask = sdf <= 0.0
        # Surface band: cells near the body surface
        surface_band = (np.abs(sdf) < 2.0 * max(self.dx, self.dy))

        # Freestream direction
        Vx = self.V_inf * np.cos(self.alpha_rad)
        Vy = self.V_inf * np.sin(self.alpha_rad)
        q_inf = 0.5 * self.rho_inf * self.V_inf**2

        # Surface normals
        snx, sny = self._surface_normal(sdf)

        # Compute Cp using modified Newtonian theory
        # sin^2(theta) where theta = angle between freestream and surface normal
        V_dot_n = Vx * snx + Vy * sny
        sin2_theta = np.clip(V_dot_n / (self.V_inf + 1e-30), 0.0, 1.0)**2

        if self.mach >= 1.0:
            # Modified Newtonian Cp_max for supersonic flow
            gp1 = self.gamma + 1.0
            gm1 = self.gamma - 1.0
            M = self.mach
            # Rayleigh Pitot formula
            term1 = (gp1**2 * M**2) / (4.0 * self.gamma * M**2 - 2.0 * gm1)
            term2 = (1.0 - self.gamma + 2.0 * self.gamma * M**2) / gp1
            Cp_max = (2.0 / (self.gamma * M**2)) * (
                term1 ** (self.gamma / gm1) * term2 - 1.0
            )
        else:
            # Incompressible: Cp_max = 1.0 (stagnation)
            Cp_max = 1.0

        Cp = np.zeros((self.ny, self.nx))
        # Windward: positive V_dot_n (flow impinging on surface)
        windward = V_dot_n > 0.0
        Cp[windward & surface_band] = Cp_max * sin2_theta[windward & surface_band]
        # Leeward: negative Cp (base pressure approximation)
        leeward = (V_dot_n <= 0.0) & surface_band
        Cp[leeward] = -0.1  # base pressure coefficient
        # Inside body: stagnation
        Cp[body_mask] = Cp_max

        # Pressure field
        pressure = self.p_inf + q_inf * Cp

        # Integrate forces on surface cells
        Fx_total = 0.0
        Fy_total = 0.0
        for j in range(1, self.ny - 1):
            for i in range(1, self.nx - 1):
                if surface_band[j, i] and not body_mask[j, i]:
                    dp = (pressure[j, i] - self.p_inf)
                    dA = self.dx * self.dy
                    Fx_total += dp * snx[j, i] * dA
                    Fy_total += dp * sny[j, i] * dA

        # Reference area = body diameter * unit span
        ref_area = 2.0 * self.body_radius * 1.0
        CL = Fy_total / (q_inf * ref_area) if q_inf > 0 else 0.0
        CD = Fx_total / (q_inf * ref_area) if q_inf > 0 else 0.0

        stag_p = float(np.max(pressure[surface_band]))

        return AeroLoads2DResult(
            Cp=Cp,
            pressure=pressure,
            x=self.x,
            y=self.y,
            body_mask=body_mask,
            CL=CL,
            CD=CD,
            Fx=Fx_total,
            Fy=Fy_total,
            stagnation_pressure=stag_p,
        )

    def export_state(self, result: AeroLoads2DResult) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="aero_loads_2d")
        state.set_field("pressure", result.pressure, "Pa")
        state.set_field("pressure_coefficient", result.Cp, "1")
        state.set_field("lift_coefficient", np.array([result.CL]), "1")
        state.set_field("drag_coefficient", np.array([result.CD]), "1")
        state.metadata["Fx_N_per_m"] = result.Fx
        state.metadata["Fy_N_per_m"] = result.Fy
        state.metadata["stagnation_pressure_Pa"] = result.stagnation_pressure
        return state

    def import_state(self, state: PhysicsState) -> None:
        """Import coupled fields from partner solvers."""
        self._coupled_state = state

    def advance(self, dt: float) -> PhysicsState:
        """Advance solver by dt for closed-loop coupling (steady re-solve)."""
        if self._coupled_state is not None:
            if self._coupled_state.has("wall_temperature"):
                T_wall = self._coupled_state.get_field("wall_temperature").data
                # Wall temperature modifies density via ideal gas approximation
                T_mean = float(np.mean(T_wall))
                if T_mean > 0:
                    self.rho_inf *= (300.0 / T_mean)  # simple T correction
            elif self._coupled_state.has("temperature"):
                T_data = self._coupled_state.get_field("temperature").data
                T_mean = float(np.mean(T_data))
                if T_mean > 0:
                    self.rho_inf *= (300.0 / T_mean)
        result = self.solve()
        self._time += dt
        return self.export_state(result)
