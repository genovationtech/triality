"""
Vortex Lattice Method (VLM) Solver for wing aerodynamics.

Discretizes a finite wing into chordwise x spanwise panels, places
horseshoe vortices at the quarter-chord of each panel, enforces the
no-penetration boundary condition at three-quarter-chord control points,
and solves for the circulation distribution.

Governing equation (Neumann BC on each panel i):
    sum_j  AIC_ij * Gamma_j  =  -V_inf . n_hat_i

Post-processing:
    L'(y) = rho * V_inf * Gamma(y)           [Kutta-Joukowski]
    CL    = 2 * sum(Gamma * dy) / (V * S)
    CD_i  = sum(Gamma * w_i * dy) / (0.5 * rho * V^2 * S)
    Cm    = moment about root / (0.5 * rho * V^2 * S * c_ref)

References:
    - Bertin & Cummings, "Aerodynamics for Engineers", 6th ed.
    - Katz & Plotkin, "Low-Speed Aerodynamics", 2nd ed.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from .aero_models import SubsonicAerodynamics, AirfoilType


# -----------------------------------------------------------------------
# Result dataclass
# -----------------------------------------------------------------------
@dataclass
class VLMResult:
    """Results from the Vortex Lattice Method solver.

    CL:               Total lift coefficient
    CD_induced:       Induced drag coefficient
    Cm:               Pitching moment coefficient (about root LE)
    span_stations:    Spanwise y-coordinates of panel centres [m]
    circulation:      Bound circulation at each span station [m^2/s]
    cl_local:         Local section lift coefficient at each station
    downwash:         Induced downwash at each control point [m/s]
    AIC:              Aerodynamic Influence Coefficient matrix
    """
    CL: float = 0.0
    CD_induced: float = 0.0
    Cm: float = 0.0
    span_stations: np.ndarray = field(default_factory=lambda: np.zeros(0))
    circulation: np.ndarray = field(default_factory=lambda: np.zeros(0))
    cl_local: np.ndarray = field(default_factory=lambda: np.zeros(0))
    downwash: np.ndarray = field(default_factory=lambda: np.zeros(0))
    AIC: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


# -----------------------------------------------------------------------
# VLM Solver
# -----------------------------------------------------------------------
class VortexLatticeSolver:
    """Vortex Lattice Method solver for a planar rectangular/tapered wing.

    Parameters
    ----------
    span : float
        Full wingspan [m].
    root_chord : float
        Root chord [m].
    tip_chord : float or None
        Tip chord [m].  If None, equals root_chord (rectangular wing).
    n_span : int
        Number of spanwise panels (half-span is mirrored).
    alpha_deg : float
        Geometric angle of attack [deg].
    V_inf : float
        Freestream speed [m/s].
    rho : float
        Air density [kg/m^3].
    airfoil : SubsonicAerodynamics or None
        Optional airfoil model for zero-lift angle correction.
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        span: float = 10.0,
        root_chord: float = 1.0,
        tip_chord: Optional[float] = None,
        n_span: int = 40,
        alpha_deg: float = 5.0,
        V_inf: float = 30.0,
        rho: float = 1.225,
        airfoil: Optional[SubsonicAerodynamics] = None,
    ):
        self.span = span
        self.root_chord = root_chord
        self.tip_chord = tip_chord if tip_chord is not None else root_chord
        self.n_span = n_span
        self.alpha = np.radians(alpha_deg)
        self.V_inf = V_inf
        self.rho = rho
        self.airfoil = airfoil or SubsonicAerodynamics(
            AirfoilType.NACA_0012, aspect_ratio=span ** 2 / (span * root_chord)
        )

        # Derived
        self.S = 0.5 * (self.root_chord + self.tip_chord) * self.span
        self.c_ref = 0.5 * (self.root_chord + self.tip_chord)
        self.AR = self.span ** 2 / self.S

    # -------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------
    def _chord_at(self, y: float) -> float:
        """Linear taper chord distribution."""
        eta = abs(y) / (0.5 * self.span)
        return self.root_chord + (self.tip_chord - self.root_chord) * eta

    def _build_panels(self):
        """Create panel geometry for the full wing (-b/2 .. +b/2).

        Returns arrays of control-point positions, bound-vortex positions,
        panel normals, and panel widths.
        """
        b2 = 0.5 * self.span
        # Cosine spacing for better tip resolution
        theta = np.linspace(0, np.pi, self.n_span + 1)
        y_edges = -b2 * np.cos(theta)  # from -b/2 to +b/2

        n = self.n_span
        y_cp = np.zeros(n)       # control point y
        x_cp = np.zeros(n)       # control point x (3/4 chord)
        y_bv = np.zeros(n)       # bound vortex y (1/4 chord)
        x_bv = np.zeros(n)       # bound vortex x
        dy = np.zeros(n)         # panel width
        chord = np.zeros(n)      # local chord

        for i in range(n):
            y1, y2 = y_edges[i], y_edges[i + 1]
            yc = 0.5 * (y1 + y2)
            c = self._chord_at(yc)
            y_cp[i] = yc
            x_cp[i] = 0.75 * c   # 3/4-chord control point
            y_bv[i] = yc
            x_bv[i] = 0.25 * c   # 1/4-chord bound vortex
            dy[i] = y2 - y1
            chord[i] = c

        return y_cp, x_cp, y_bv, x_bv, dy, chord

    # -------------------------------------------------------------------
    # Horseshoe vortex induced velocity (z-component)
    # -------------------------------------------------------------------
    @staticmethod
    def _horseshoe_wz(xc, yc, xv, yv1, yv2):
        """Compute z-velocity induced at (xc, yc, 0) by a horseshoe vortex
        of unit circulation with bound segment from (xv, yv1, 0) to
        (xv, yv2, 0) and semi-infinite trailing legs at yv1 and yv2
        extending to x = +inf.

        Returns w_z per unit Gamma (positive downward).
        """
        # Bound segment contribution (Biot-Savart for finite segment)
        dx = xc - xv
        dy1 = yc - yv1
        dy2 = yc - yv2
        r1 = np.sqrt(dx ** 2 + dy1 ** 2 + 1e-12)
        r2 = np.sqrt(dx ** 2 + dy2 ** 2 + 1e-12)

        # Bound vortex: from (xv, yv1) to (xv, yv2)
        # w_bound = Gamma/(4*pi) * (dx/r1 - dx/r2) * ... simplified for z=0 plane
        cross = dx * (dy2 - dy1)
        denom = r1 * r2 * (r1 * r2 + dx ** 2 + dy1 * dy2)
        denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
        w_bound = cross * (r1 + r2) / (4.0 * np.pi * denom)

        # Trailing leg at yv1 (semi-infinite, x from xv to +inf)
        # w_trail1 = -Gamma/(4*pi) * (1 + dx/r1) / (dy1 + 1e-14)
        w_trail1 = -(1.0 + dx / r1) / (4.0 * np.pi * (dy1 + np.sign(dy1) * 1e-10))

        # Trailing leg at yv2 (semi-infinite, x from xv to +inf)
        w_trail2 = (1.0 + dx / r2) / (4.0 * np.pi * (dy2 + np.sign(dy2) * 1e-10))

        return w_bound + w_trail1 + w_trail2

    # -------------------------------------------------------------------
    # Build AIC matrix
    # -------------------------------------------------------------------
    def _build_aic(self, y_cp, x_cp, y_bv, x_bv, dy):
        """Build the Aerodynamic Influence Coefficient matrix.

        AIC_ij = normal-velocity induced at control point i by
                 horseshoe vortex j with unit circulation.
        """
        n = len(y_cp)
        AIC = np.zeros((n, n))

        for j in range(n):
            # Horseshoe vortex j: bound segment from
            # (x_bv[j], y_bv[j] - dy[j]/2) to (x_bv[j], y_bv[j] + dy[j]/2)
            yv1 = y_bv[j] - 0.5 * dy[j]
            yv2 = y_bv[j] + 0.5 * dy[j]

            for i in range(n):
                AIC[i, j] = self._horseshoe_wz(
                    x_cp[i], y_cp[i], x_bv[j], yv1, yv2
                )

        return AIC

    # -------------------------------------------------------------------
    # Solve
    # -------------------------------------------------------------------
    def solve(self) -> VLMResult:
        """Run the VLM solver.

        Steps:
            1. Build panel geometry (cosine-spaced).
            2. Assemble AIC matrix from horseshoe vortex influence.
            3. Form RHS = -V_inf * sin(alpha) at each control point.
            4. Solve AIC @ Gamma = RHS for circulation vector.
            5. Post-process: CL, CD_i, Cm, spanwise loading.
        """
        y_cp, x_cp, y_bv, x_bv, dy, chord = self._build_panels()
        n = self.n_span

        # --- AIC matrix ---
        AIC = self._build_aic(y_cp, x_cp, y_bv, x_bv, dy)

        # --- RHS: no-penetration condition ---
        # Panel normals point in -z for a flat plate at angle alpha;
        # V_inf . n_hat = V_inf * sin(alpha) for small alpha ≈ V_inf * alpha
        rhs = -self.V_inf * np.sin(self.alpha) * np.ones(n)

        # --- Solve linear system ---
        gamma = np.linalg.solve(AIC, rhs)

        # --- Induced downwash at each control point ---
        w_induced = AIC @ gamma  # should equal rhs; recompute physical downwash
        # Physical downwash from all vortices (already encoded in AIC*gamma)
        # but we want the actual w_i for induced drag.  Re-evaluate at
        # control points using trailing vortices only.
        w_i = np.zeros(n)
        for j in range(n):
            yv1 = y_bv[j] - 0.5 * dy[j]
            yv2 = y_bv[j] + 0.5 * dy[j]
            for i in range(n):
                # trailing-leg only contribution
                dx = x_cp[i] - x_bv[j]
                dy1 = y_cp[i] - yv1
                dy2 = y_cp[i] - yv2
                r1 = np.sqrt(dx ** 2 + dy1 ** 2 + 1e-12)
                r2 = np.sqrt(dx ** 2 + dy2 ** 2 + 1e-12)
                wt1 = -(1.0 + dx / r1) / (4.0 * np.pi * (dy1 + np.sign(dy1) * 1e-10))
                wt2 = (1.0 + dx / r2) / (4.0 * np.pi * (dy2 + np.sign(dy2) * 1e-10))
                w_i[i] += gamma[j] * (wt1 + wt2)

        # --- Force computation (Kutta-Joukowski) ---
        # Lift per unit span: L' = rho * V_inf * Gamma
        lift_per_span = self.rho * self.V_inf * gamma
        total_lift = np.sum(lift_per_span * dy)
        q_inf = 0.5 * self.rho * self.V_inf ** 2
        CL = total_lift / (q_inf * self.S)

        # Induced drag per unit span: D_i' = -rho * w_i * Gamma
        drag_i_per_span = -self.rho * w_i * gamma
        total_drag_i = np.sum(drag_i_per_span * dy)
        CD_i = total_drag_i / (q_inf * self.S)

        # Pitching moment about root LE (x=0): M = -sum(L' * x_bv * dy)
        moment = -np.sum(lift_per_span * x_bv * dy)
        Cm = moment / (q_inf * self.S * self.c_ref)

        # Local section CL
        cl_local = np.where(
            chord > 1e-10,
            lift_per_span / (q_inf * chord),
            0.0,
        )

        return VLMResult(
            CL=float(CL),
            CD_induced=float(CD_i),
            Cm=float(Cm),
            span_stations=y_cp,
            circulation=gamma,
            cl_local=cl_local,
            downwash=w_i,
            AIC=AIC,
        )

    def export_state(self, result: VLMResult) -> PhysicsState:
        """Export result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="uav_aerodynamics")
        state.set_field("lift_coefficient", result.cl_local, "1")
        state.set_field("drag_coefficient", np.full_like(result.cl_local, result.CD_induced), "1")
        state.set_field("velocity", result.downwash, "m/s")
        state.metadata["CL"] = result.CL
        state.metadata["CD_induced"] = result.CD_induced
        state.metadata["Cm"] = result.Cm
        return state


# ===========================================================================
# Level 3: 2D Vortex Panel Method for Airfoil Analysis
# ===========================================================================

@dataclass
class UAVAerodynamics2DResult:
    """Result container for 2D vortex panel method airfoil analysis.

    Attributes
    ----------
    Cp : np.ndarray
        Pressure coefficient distribution, shape (n_panels,).
    panel_x : np.ndarray
        Panel midpoint x-coordinates [m], shape (n_panels,).
    panel_y : np.ndarray
        Panel midpoint y-coordinates [m], shape (n_panels,).
    gamma : np.ndarray
        Vortex strength distribution, shape (n_panels,).
    CL : float
        Lift coefficient.
    CD : float
        Drag coefficient (pressure drag from panel method).
    Cm : float
        Pitching moment coefficient about quarter-chord.
    stagnation_point : np.ndarray
        Stagnation point location [x, y] [m].
    circulation : float
        Total circulation [m^2/s].
    velocity_field_x : np.ndarray
        x-velocity on a 2D grid [m/s], shape (ny_grid, nx_grid).
    velocity_field_y : np.ndarray
        y-velocity on a 2D grid [m/s], shape (ny_grid, nx_grid).
    x_grid : np.ndarray
        Grid x-coordinates, shape (nx_grid,).
    y_grid : np.ndarray
        Grid y-coordinates, shape (ny_grid,).
    """
    Cp: np.ndarray = field(default_factory=lambda: np.array([]))
    panel_x: np.ndarray = field(default_factory=lambda: np.array([]))
    panel_y: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma: np.ndarray = field(default_factory=lambda: np.array([]))
    CL: float = 0.0
    CD: float = 0.0
    Cm: float = 0.0
    stagnation_point: np.ndarray = field(default_factory=lambda: np.zeros(2))
    circulation: float = 0.0
    velocity_field_x: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    velocity_field_y: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    x_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    y_grid: np.ndarray = field(default_factory=lambda: np.array([]))


class UAVAerodynamics2DSolver:
    """2D vortex panel method for UAV airfoil analysis.

    Discretizes a 2D airfoil into linear vortex panels, enforces the
    no-penetration condition at panel midpoints, applies the Kutta condition
    at the trailing edge, and solves for the vortex strength distribution.

    The airfoil is defined as a NACA 4-digit profile.

    Governing system:
        A @ gamma = b  (flow tangency at each control point + Kutta)
        A_ij = influence of panel j on control point i
        b_i = -V_inf . n_i

    Post-processing:
        V_t = V_inf . t_hat + sum(gamma_j * influence)
        Cp = 1 - (V_t / V_inf)^2
        CL = -sum(Cp * panel_length * n_y) / chord

    Parameters
    ----------
    naca : str
        NACA 4-digit airfoil designation (e.g., '2412').
    n_panels : int
        Number of panels.
    chord : float
        Chord length [m].
    alpha_deg : float
        Angle of attack [degrees].
    V_inf : float
        Freestream velocity [m/s].
    rho : float
        Air density [kg/m^3].
    """

    fidelity_tier = FidelityTier.ENGINEERING
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        naca: str = '2412',
        n_panels: int = 100,
        chord: float = 1.0,
        alpha_deg: float = 5.0,
        V_inf: float = 30.0,
        rho: float = 1.225,
    ):
        self.naca = naca
        self.n_panels = n_panels
        self.chord = chord
        self.alpha = np.radians(alpha_deg)
        self.V_inf = V_inf
        self.rho = rho

    def _generate_naca4(self) -> tuple:
        """Generate NACA 4-digit airfoil coordinates.

        Returns (x, y) arrays of shape (n_panels+1,) defining panel endpoints,
        ordered from trailing edge along lower surface to leading edge and
        back along upper surface to trailing edge.
        """
        m = int(self.naca[0]) / 100.0
        p = int(self.naca[1]) / 10.0
        t_max = int(self.naca[2:4]) / 100.0

        n = self.n_panels // 2 + 1
        beta = np.linspace(0, np.pi, n)
        xc = 0.5 * (1.0 - np.cos(beta)) * self.chord

        # Thickness distribution
        yt = 5.0 * t_max * self.chord * (
            0.2969 * np.sqrt(xc / self.chord + 1e-12)
            - 0.1260 * (xc / self.chord)
            - 0.3516 * (xc / self.chord)**2
            + 0.2843 * (xc / self.chord)**3
            - 0.1015 * (xc / self.chord)**4
        )

        # Camber line
        yc = np.zeros_like(xc)
        dyc = np.zeros_like(xc)
        if p > 0:
            front = xc <= p * self.chord
            back = ~front
            yc[front] = m * xc[front] / p**2 * (2*p - xc[front] / self.chord)
            yc[back] = m * (self.chord - xc[back]) / (1 - p)**2 * (
                1 + xc[back] / self.chord - 2*p
            )
            dyc[front] = 2*m / p**2 * (p - xc[front] / self.chord)
            dyc[back] = 2*m / (1-p)**2 * (p - xc[back] / self.chord)

        theta = np.arctan(dyc)
        xu = xc - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = xc + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        # Concatenate: TE -> lower -> LE -> upper -> TE
        x = np.concatenate([xl[::-1], xu[1:]])
        y = np.concatenate([yl[::-1], yu[1:]])

        return x, y

    def solve(self, compute_field: bool = True,
              nx_grid: int = 60, ny_grid: int = 40) -> UAVAerodynamics2DResult:
        """Run the vortex panel method.

        Parameters
        ----------
        compute_field : bool
            If True, also compute velocity field on a 2D grid.
        nx_grid, ny_grid : int
            Grid resolution for velocity field.

        Returns
        -------
        UAVAerodynamics2DResult
        """
        x_af, y_af = self._generate_naca4()
        n = len(x_af) - 1  # number of panels

        # Panel midpoints, lengths, normals, tangents
        xm = 0.5 * (x_af[:-1] + x_af[1:])
        ym = 0.5 * (y_af[:-1] + y_af[1:])
        dx_p = x_af[1:] - x_af[:-1]
        dy_p = y_af[1:] - y_af[:-1]
        panel_len = np.sqrt(dx_p**2 + dy_p**2)

        # Outward normals
        nx_n = dy_p / panel_len
        ny_n = -dx_p / panel_len

        # Tangent vectors
        tx = dx_p / panel_len
        ty = dy_p / panel_len

        # Build influence coefficient matrix
        # Using constant-strength vortex panels
        A = np.zeros((n + 1, n + 1))  # extra row for Kutta condition

        for i in range(n):
            for j in range(n):
                # Vortex panel j influence on control point i
                x1, y1 = x_af[j], y_af[j]
                x2, y2 = x_af[j + 1], y_af[j + 1]

                # Relative positions
                r1x = xm[i] - x1
                r1y = ym[i] - y1
                r2x = xm[i] - x2
                r2y = ym[i] - y2

                r1 = np.sqrt(r1x**2 + r1y**2 + 1e-12)
                r2 = np.sqrt(r2x**2 + r2y**2 + 1e-12)

                theta1 = np.arctan2(r1y, r1x)
                theta2 = np.arctan2(r2y, r2x)
                dtheta = theta2 - theta1
                # Wrap angle
                if dtheta > np.pi:
                    dtheta -= 2 * np.pi
                elif dtheta < -np.pi:
                    dtheta += 2 * np.pi

                log_ratio = np.log(r2 / r1) if r1 > 1e-15 and r2 > 1e-15 else 0.0

                # Velocity in panel-local coords, then project to normal
                # Simplified: use point vortex at panel midpoint
                rx = xm[i] - xm[j]
                ry = ym[i] - ym[j]
                r2_sq = rx**2 + ry**2 + 1e-12
                # Induced velocity per unit gamma
                u_ind = ry / (2 * np.pi * r2_sq) * panel_len[j]
                v_ind = -rx / (2 * np.pi * r2_sq) * panel_len[j]

                if i == j:
                    # Self-influence: 0.5 for normal component
                    A[i, j] = 0.5
                else:
                    A[i, j] = (u_ind * nx_n[i] + v_ind * ny_n[i])

        # Kutta condition: gamma[0] + gamma[-1] = 0
        A[n, 0] = 1.0
        A[n, -1] = 1.0

        # RHS: -V_inf . n_hat
        V_inf_x = self.V_inf * np.cos(self.alpha)
        V_inf_y = self.V_inf * np.sin(self.alpha)
        b = np.zeros(n + 1)
        for i in range(n):
            b[i] = -(V_inf_x * nx_n[i] + V_inf_y * ny_n[i])
        b[n] = 0.0  # Kutta

        # Solve
        gamma = np.linalg.lstsq(A[:n+1, :n], b[:n+1], rcond=None)[0]

        # Tangential velocity and Cp
        V_t = np.zeros(n)
        for i in range(n):
            V_t[i] = V_inf_x * tx[i] + V_inf_y * ty[i]
            for j in range(n):
                if i != j:
                    rx = xm[i] - xm[j]
                    ry = ym[i] - ym[j]
                    r2_sq = rx**2 + ry**2 + 1e-12
                    u_ind = gamma[j] * ry / (2 * np.pi * r2_sq) * panel_len[j]
                    v_ind = -gamma[j] * rx / (2 * np.pi * r2_sq) * panel_len[j]
                    V_t[i] += u_ind * tx[i] + v_ind * ty[i]

        Cp = 1.0 - (V_t / self.V_inf)**2

        # Force coefficients
        # CL = -sum(Cp * panel_len * sin(panel_angle)) / chord
        CL = float(-np.sum(Cp * panel_len * ny_n) / self.chord)
        CD = float(-np.sum(Cp * panel_len * nx_n) / self.chord)

        # Moment about quarter-chord
        x_qc = 0.25 * self.chord
        moment_arm = xm - x_qc
        Cm = float(np.sum(Cp * panel_len * ny_n * moment_arm) / self.chord**2)

        # Total circulation
        total_circ = float(np.sum(gamma * panel_len))

        # Stagnation point (where Cp is closest to 1)
        stag_idx = np.argmax(Cp)
        stag_pt = np.array([xm[stag_idx], ym[stag_idx]])

        # Optional velocity field
        vfx = np.zeros((0, 0))
        vfy = np.zeros((0, 0))
        x_grid = np.array([])
        y_grid = np.array([])

        if compute_field:
            x_grid = np.linspace(-0.5 * self.chord, 1.5 * self.chord, nx_grid)
            y_grid = np.linspace(-0.7 * self.chord, 0.7 * self.chord, ny_grid)
            Xg, Yg = np.meshgrid(x_grid, y_grid)
            vfx = np.full_like(Xg, V_inf_x)
            vfy = np.full_like(Yg, V_inf_y)

            for j in range(n):
                rx = Xg - xm[j]
                ry = Yg - ym[j]
                r2_sq = rx**2 + ry**2 + 1e-8
                vfx += gamma[j] * ry / (2 * np.pi * r2_sq) * panel_len[j]
                vfy += -gamma[j] * rx / (2 * np.pi * r2_sq) * panel_len[j]

        return UAVAerodynamics2DResult(
            Cp=Cp,
            panel_x=xm,
            panel_y=ym,
            gamma=gamma,
            CL=CL,
            CD=CD,
            Cm=Cm,
            stagnation_point=stag_pt,
            circulation=total_circ,
            velocity_field_x=vfx,
            velocity_field_y=vfy,
            x_grid=x_grid,
            y_grid=y_grid,
        )

    def export_state(self, result: UAVAerodynamics2DResult) -> PhysicsState:
        """Export 2D airfoil result as PhysicsState for coupling."""
        state = PhysicsState(solver_name="uav_aerodynamics_2d")
        state.set_field("pressure", result.Cp, "1")
        state.set_field("velocity_x", result.velocity_field_x, "m/s")
        state.set_field("velocity_y", result.velocity_field_y, "m/s")
        state.metadata["CL"] = result.CL
        state.metadata["CD"] = result.CD
        state.metadata["Cm"] = result.Cm
        state.metadata["circulation"] = result.circulation
        return state
