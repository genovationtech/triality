"""
UAV Aerodynamics Models.

Subsonic aerodynamic analysis for fixed-wing and rotary-wing UAVs,
including airfoil characteristics, blade element theory, multirotor
hover/forward flight, and fixed-wing trim analysis.

Physics Basis:
--------------
Thin Airfoil Theory (Prandtl, 1918):
    CL = 2*pi*alpha                       (lift curve slope for thin airfoil)
    CL = 2*pi*(alpha - alpha_L0)          (with zero-lift angle of attack)

Parabolic Drag Polar:
    CD = CD0 + CL^2 / (pi * e * AR)      (total drag coefficient)

    where CD0 = zero-lift drag coefficient
          e   = Oswald span efficiency factor (0.7-0.9 typical)
          AR  = aspect ratio = b^2 / S

Blade Element Theory (Glauert, 1926):
    dT = 0.5 * rho * c * (V_T^2 + V_P^2) * CL * dr
    dQ = 0.5 * rho * c * (V_T^2 + V_P^2) * (CD*cos(phi) + CL*sin(phi)) * r * dr

    where V_T = tangential velocity = Omega * r
          V_P = perpendicular velocity (climb + induced)
          phi = inflow angle = atan(V_P / V_T)

Propeller Thrust and Torque Coefficients:
    T  = CT * rho * n^2 * D^4             (thrust)
    Q  = CQ * rho * n^2 * D^5             (torque)
    P  = CP * rho * n^3 * D^5             (power)

    where n = revolutions per second [rps]
          D = propeller diameter [m]

Momentum Theory (Rankine, 1865; Froude, 1889):
    T = 2 * rho * A * v_i * (V_climb + v_i)
    P_induced = T * (V_climb + v_i)

    where v_i = induced velocity
          A   = rotor disk area

Hover Power:
    P_hover = T^(3/2) / sqrt(2 * rho * A)

Maximum Lift-to-Drag Ratio:
    (L/D)_max = 0.5 * sqrt(pi * e * AR / CD0)

    at CL_opt = sqrt(pi * e * AR * CD0)

Betz Limit for Propeller Efficiency (Betz, 1919):
    eta_max = 1 - (a / (1 + a))           (actuator disk theory)

    where a = axial induction factor

References:
-----------
- Prandtl, L. (1918). "Tragflugeltheorie." Nachrichten von der
  Gesellschaft der Wissenschaften zu Gottingen.
- Glauert, H. (1926). "The Elements of Aerofoil and Airscrew Theory."
  Cambridge University Press.
- Betz, A. (1919). "Schraubenpropeller mit geringstem Energieverlust."
  Nachrichten von der Gesellschaft der Wissenschaften zu Gottingen.
- Rankine, W.J.M. (1865). "On the Mechanical Principles of the Action
  of Propellers." Transactions, Institution of Naval Architects.
- Froude, R.E. (1889). "On the Part Played in Propulsion by Differences
  in Fluid Pressure." Transactions, Institution of Naval Architects.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AirfoilType(Enum):
    """Standard airfoil types for UAV applications."""
    NACA_0012 = 'naca_0012'
    CLARK_Y = 'clark_y'
    FLAT_PLATE = 'flat_plate'
    CUSTOM = 'custom'


class FlightRegime(Enum):
    """Flight regime classification."""
    HOVER = 'hover'
    CLIMB = 'climb'
    CRUISE = 'cruise'
    DESCENT = 'descent'
    AUTOROTATION = 'autorotation'


class UAVType(Enum):
    """UAV platform type."""
    FIXED_WING = 'fixed_wing'
    MULTIROTOR = 'multirotor'
    HELICOPTER = 'helicopter'
    TILTROTOR = 'tilt_rotor'


# ---------------------------------------------------------------------------
# Result Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AeroCoefficients:
    """
    Aerodynamic coefficient results.

    Contains lift, drag, and moment coefficients along with key
    aerodynamic parameters for an airfoil or wing section.

    CL:       Lift coefficient [-]
    CD:       Drag coefficient [-]
    CM:       Pitching moment coefficient [-]
    CL_alpha: Lift curve slope [1/rad]
    CD0:      Zero-lift drag coefficient [-]
    e:        Oswald span efficiency factor [-]
    alpha:    Angle of attack [rad]
    """
    CL: float = 0.0
    CD: float = 0.0
    CM: float = 0.0
    CL_alpha: float = 2.0 * np.pi
    CD0: float = 0.01
    e: float = 0.8
    alpha: float = 0.0


@dataclass
class RotorPerformance:
    """
    Rotor performance results from blade element or momentum theory.

    thrust:          Total rotor thrust [N]
    torque:          Total rotor torque [N*m]
    power:           Total rotor power [W]
    CT:              Thrust coefficient [-]
    CQ:              Torque coefficient [-]
    CP:              Power coefficient [-]
    induced_velocity: Mean induced velocity [m/s]
    efficiency:      Propulsive efficiency [-]
    advance_ratio:   J = V / (n * D) [-]
    """
    thrust: float = 0.0
    torque: float = 0.0
    power: float = 0.0
    CT: float = 0.0
    CQ: float = 0.0
    CP: float = 0.0
    induced_velocity: float = 0.0
    efficiency: float = 0.0
    advance_ratio: float = 0.0


@dataclass
class TrimResult:
    """
    Fixed-wing trim condition results.

    CL_trim:        Trim lift coefficient [-]
    alpha_trim:     Trim angle of attack [rad]
    elevator_trim:  Trim elevator deflection [rad]
    CD_trim:        Drag at trim [-]
    thrust_required: Thrust at trim [N]
    L_over_D:       Lift-to-drag ratio at trim [-]
    power_required:  Power required at trim [W]
    """
    CL_trim: float = 0.0
    alpha_trim: float = 0.0
    elevator_trim: float = 0.0
    CD_trim: float = 0.0
    thrust_required: float = 0.0
    L_over_D: float = 0.0
    power_required: float = 0.0


@dataclass
class MultirotorPerformance:
    """
    Multirotor flight performance results.

    total_thrust:     Total thrust from all rotors [N]
    total_power:      Total power consumption [W]
    power_induced:    Induced power [W]
    power_parasite:   Parasite drag power [W]
    power_profile:    Blade profile drag power [W]
    hover_endurance:  Hover endurance estimate [s]
    flight_regime:    Current flight regime
    """
    total_thrust: float = 0.0
    total_power: float = 0.0
    power_induced: float = 0.0
    power_parasite: float = 0.0
    power_profile: float = 0.0
    hover_endurance: float = 0.0
    flight_regime: FlightRegime = FlightRegime.HOVER


# ---------------------------------------------------------------------------
# Airfoil Data Presets
# ---------------------------------------------------------------------------

# NACA 0012: Symmetric airfoil, widely used for helicopter blades and tails.
# alpha_L0 = 0 for symmetric airfoils.
AIRFOIL_PRESETS: Dict[AirfoilType, Dict] = {
    AirfoilType.NACA_0012: {
        'CL_alpha': 2.0 * np.pi,     # Thin airfoil theory [1/rad]
        'alpha_L0': 0.0,              # Zero-lift AoA [rad]
        'CD0': 0.006,                 # Zero-lift drag
        'CM0': 0.0,                   # Zero-lift moment (symmetric)
        'CM_alpha': -0.01,            # Moment slope [1/rad]
        'CL_max': 1.5,               # Max CL before stall
        'alpha_stall': np.radians(16.0),  # Stall angle [rad]
        'description': 'NACA 0012 symmetric airfoil'
    },
    AirfoilType.CLARK_Y: {
        'CL_alpha': 5.8,             # Measured slope [1/rad]
        'alpha_L0': np.radians(-3.6),  # Zero-lift AoA [rad]
        'CD0': 0.0065,               # Zero-lift drag
        'CM0': -0.084,               # Zero-lift moment (cambered)
        'CM_alpha': -0.08,           # Moment slope [1/rad]
        'CL_max': 1.4,              # Max CL
        'alpha_stall': np.radians(15.0),  # Stall angle
        'description': 'Clark Y cambered airfoil (classic propeller section)'
    },
    AirfoilType.FLAT_PLATE: {
        'CL_alpha': 2.0 * np.pi,     # Thin airfoil theory
        'alpha_L0': 0.0,             # Zero-lift AoA
        'CD0': 0.02,                 # Higher friction drag
        'CM0': 0.0,                  # No zero-lift moment
        'CM_alpha': 0.0,             # Neutral moment
        'CL_max': 0.9,              # Lower max CL
        'alpha_stall': np.radians(10.0),
        'description': 'Flat plate airfoil (thin airfoil theory baseline)'
    },
}


# ---------------------------------------------------------------------------
# Subsonic Aerodynamics
# ---------------------------------------------------------------------------

class SubsonicAerodynamics:
    """
    Subsonic airfoil and wing aerodynamic analysis.

    Computes lift, drag, and pitching moment coefficients using
    thin airfoil theory (Prandtl, 1918) and parabolic drag polar.

    Governing Equations:
        CL = CL_alpha * (alpha - alpha_L0)           [Thin Airfoil Theory]
        CD = CD0 + CL^2 / (pi * e * AR)              [Parabolic Drag Polar]
        CM = CM0 + CM_alpha * alpha                   [Pitching Moment]

    Parameters
    ----------
    airfoil_type : AirfoilType
        Predefined airfoil or CUSTOM.
    aspect_ratio : float
        Wing aspect ratio AR = b^2 / S.
    oswald_efficiency : float
        Oswald span efficiency factor e (0.7--0.9 typical).
    custom_params : dict, optional
        Custom airfoil parameters when airfoil_type is CUSTOM.
    """

    def __init__(
        self,
        airfoil_type: AirfoilType = AirfoilType.NACA_0012,
        aspect_ratio: float = 8.0,
        oswald_efficiency: float = 0.8,
        custom_params: Optional[Dict] = None,
    ):
        self.aspect_ratio = aspect_ratio
        self.oswald_efficiency = oswald_efficiency

        if airfoil_type == AirfoilType.CUSTOM and custom_params is not None:
            self.params = custom_params
        elif airfoil_type in AIRFOIL_PRESETS:
            self.params = AIRFOIL_PRESETS[airfoil_type].copy()
        else:
            raise ValueError(f"Unknown airfoil type: {airfoil_type}")

        # Apply finite-wing correction to lift curve slope
        # Prandtl lifting-line correction:
        #   CL_alpha_3D = CL_alpha_2D / (1 + CL_alpha_2D / (pi * AR))
        cl_alpha_2d = self.params['CL_alpha']
        self.CL_alpha_3D = cl_alpha_2d / (
            1.0 + cl_alpha_2d / (np.pi * self.aspect_ratio)
        )

    def lift_coefficient(self, alpha: float) -> float:
        """
        Lift coefficient from thin airfoil theory with stall model.

        CL = CL_alpha * (alpha - alpha_L0)   for alpha < alpha_stall

        Beyond stall, a simple Kirchhoff-type model reduces CL:
            CL_stall = CL_max * sin(2 * alpha) / (2 * sin(alpha_stall))

        Parameters
        ----------
        alpha : float
            Angle of attack [rad].

        Returns
        -------
        float
            Lift coefficient CL.
        """
        alpha_L0 = self.params['alpha_L0']
        alpha_stall = self.params['alpha_stall']
        CL_max = self.params['CL_max']

        # Pre-stall: linear regime
        if abs(alpha - alpha_L0) < abs(alpha_stall - alpha_L0):
            return self.CL_alpha_3D * (alpha - alpha_L0)

        # Post-stall: smooth reduction using flat-plate model
        # CL ~ CL_max * sin(2*alpha) for large alpha (Hoerner, 1965)
        sign = 1.0 if alpha > alpha_L0 else -1.0
        return sign * CL_max * np.sin(2.0 * abs(alpha)) / (
            2.0 * np.sin(alpha_stall) + 1e-12
        )

    def drag_coefficient(self, CL: float) -> float:
        """
        Drag coefficient from parabolic drag polar.

        CD = CD0 + CL^2 / (pi * e * AR)

        Parameters
        ----------
        CL : float
            Lift coefficient.

        Returns
        -------
        float
            Drag coefficient CD.
        """
        CD0 = self.params['CD0']
        induced_drag = CL ** 2 / (
            np.pi * self.oswald_efficiency * self.aspect_ratio
        )
        return CD0 + induced_drag

    def moment_coefficient(self, alpha: float) -> float:
        """
        Pitching moment coefficient about the quarter-chord.

        CM = CM0 + CM_alpha * alpha

        For a thin symmetric airfoil, CM_ac = 0 (aerodynamic center
        at quarter-chord). For cambered airfoils, CM0 != 0.

        Parameters
        ----------
        alpha : float
            Angle of attack [rad].

        Returns
        -------
        float
            Pitching moment coefficient CM.
        """
        CM0 = self.params['CM0']
        CM_alpha = self.params['CM_alpha']
        return CM0 + CM_alpha * alpha

    def compute_coefficients(self, alpha: float) -> AeroCoefficients:
        """
        Compute all aerodynamic coefficients at a given angle of attack.

        Parameters
        ----------
        alpha : float
            Angle of attack [rad].

        Returns
        -------
        AeroCoefficients
            Complete set of aerodynamic coefficients.
        """
        CL = self.lift_coefficient(alpha)
        CD = self.drag_coefficient(CL)
        CM = self.moment_coefficient(alpha)

        return AeroCoefficients(
            CL=CL,
            CD=CD,
            CM=CM,
            CL_alpha=self.CL_alpha_3D,
            CD0=self.params['CD0'],
            e=self.oswald_efficiency,
            alpha=alpha,
        )

    def max_lift_to_drag(self) -> Tuple[float, float]:
        """
        Maximum lift-to-drag ratio and corresponding CL.

        (L/D)_max = 0.5 * sqrt(pi * e * AR / CD0)
        CL_opt    = sqrt(pi * e * AR * CD0)

        Returns
        -------
        tuple of (L_D_max, CL_opt)
            Maximum L/D and optimal lift coefficient.
        """
        CD0 = self.params['CD0']
        e = self.oswald_efficiency
        AR = self.aspect_ratio

        CL_opt = np.sqrt(np.pi * e * AR * CD0)
        L_D_max = 0.5 * np.sqrt(np.pi * e * AR / CD0)

        return L_D_max, CL_opt

    def lift_curve(
        self, alpha_range: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute CL, CD, CM over a range of angles of attack.

        Parameters
        ----------
        alpha_range : ndarray
            Angle of attack values [rad].

        Returns
        -------
        tuple of (CL_array, CD_array, CM_array)
        """
        CL_arr = np.array([self.lift_coefficient(a) for a in alpha_range])
        CD_arr = np.array([self.drag_coefficient(cl) for cl in CL_arr])
        CM_arr = np.array([self.moment_coefficient(a) for a in alpha_range])
        return CL_arr, CD_arr, CM_arr


# ---------------------------------------------------------------------------
# Blade Element Theory
# ---------------------------------------------------------------------------

class BladeElementTheory:
    """
    Blade element theory for rotor and propeller analysis.

    Combines momentum theory (Rankine-Froude) with blade element analysis
    (Glauert, 1926) to compute thrust and torque distributions along the blade.

    Governing Equations:
        dT = 0.5 * rho * c(r) * W^2 * CL * cos(phi) * dr
        dQ = 0.5 * rho * c(r) * W^2 * (CD*cos(phi) + CL*sin(phi)) * r * dr

        where W = sqrt(V_T^2 + V_P^2)   (resultant velocity)
              phi = atan(V_P / V_T)       (inflow angle)
              V_T = Omega * r             (tangential velocity)
              V_P = V_climb + v_i         (axial velocity)

    Momentum theory for induced velocity:
        T = 2 * rho * A * v_i * sqrt((V_climb + v_i)^2 + (Omega*r)^2 * a_t^2)

    Parameters
    ----------
    num_blades : int
        Number of rotor blades.
    blade_radius : float
        Blade tip radius [m].
    chord : float or ndarray
        Blade chord length [m], constant or distribution.
    twist : float or ndarray
        Blade twist angle [rad], root value or distribution.
    airfoil : AirfoilType
        Airfoil type for blade sections.
    root_cutout : float
        Fraction of blade radius for root cutout (hub).
    """

    def __init__(
        self,
        num_blades: int = 2,
        blade_radius: float = 0.5,
        chord: float = 0.05,
        twist: float = np.radians(10.0),
        airfoil: AirfoilType = AirfoilType.NACA_0012,
        root_cutout: float = 0.15,
    ):
        self.num_blades = num_blades
        self.blade_radius = blade_radius
        self.root_cutout = root_cutout

        # Set up blade section airfoil
        self.section_aero = SubsonicAerodynamics(
            airfoil_type=airfoil, aspect_ratio=1e6  # 2D sections
        )

        # Blade geometry arrays
        self.n_elements = 50
        self.r_frac = np.linspace(
            root_cutout, 0.99, self.n_elements
        )
        self.r = self.r_frac * blade_radius

        # Chord distribution (constant or linear taper)
        if np.isscalar(chord):
            self.chord = np.full(self.n_elements, chord)
        else:
            self.chord = np.interp(
                self.r_frac, np.linspace(root_cutout, 1.0, len(chord)), chord
            )

        # Twist distribution (linear from root to tip)
        if np.isscalar(twist):
            # Linear twist: twist at root, 0 at tip
            self.twist = twist * (1.0 - self.r_frac) / (1.0 - root_cutout)
        else:
            self.twist = np.interp(
                self.r_frac, np.linspace(root_cutout, 1.0, len(twist)), twist
            )

        # Blade solidity: sigma = N_b * c / (pi * R)
        self.solidity = (
            num_blades * np.mean(self.chord) / (np.pi * blade_radius)
        )

    def induced_velocity_hover(self, thrust: float, rho: float) -> float:
        """
        Induced velocity in hover from momentum theory.

        v_i = sqrt(T / (2 * rho * A))

        (Rankine, 1865; Froude, 1889)

        Parameters
        ----------
        thrust : float
            Rotor thrust [N].
        rho : float
            Air density [kg/m^3].

        Returns
        -------
        float
            Induced velocity [m/s].
        """
        A = np.pi * self.blade_radius ** 2
        return np.sqrt(max(thrust, 0.0) / (2.0 * rho * A + 1e-12))

    def induced_velocity_climb(
        self, thrust: float, V_climb: float, rho: float
    ) -> float:
        """
        Induced velocity in axial climb from momentum theory.

        Solve: T = 2 * rho * A * v_i * (V_climb + v_i)
        Quadratic: v_i^2 + V_climb * v_i - T/(2*rho*A) = 0

        Parameters
        ----------
        thrust : float
            Rotor thrust [N].
        V_climb : float
            Climb velocity [m/s], positive up.
        rho : float
            Air density [kg/m^3].

        Returns
        -------
        float
            Induced velocity [m/s].
        """
        A = np.pi * self.blade_radius ** 2
        disk_loading = max(thrust, 0.0) / (2.0 * rho * A + 1e-12)

        # v_i^2 + V_climb * v_i - disk_loading = 0
        discriminant = V_climb ** 2 + 4.0 * disk_loading
        v_i = (-V_climb + np.sqrt(max(discriminant, 0.0))) / 2.0
        return max(v_i, 0.0)

    def analyze(
        self,
        rpm: float,
        rho: float = 1.225,
        V_inf: float = 0.0,
        collective: float = 0.0,
    ) -> RotorPerformance:
        """
        Blade element momentum theory (BEMT) analysis.

        Iteratively solves for the induced velocity distribution
        using combined blade element and momentum theory.

        Parameters
        ----------
        rpm : float
            Rotor speed [revolutions per minute].
        rho : float
            Air density [kg/m^3].
        V_inf : float
            Freestream velocity along rotor axis [m/s].
        collective : float
            Additional collective pitch [rad].

        Returns
        -------
        RotorPerformance
            Rotor thrust, torque, power, and coefficients.
        """
        omega = rpm * 2.0 * np.pi / 60.0  # rad/s
        n_rps = rpm / 60.0                  # revolutions per second
        D = 2.0 * self.blade_radius
        dr = np.gradient(self.r)

        # Tip speed
        V_tip = omega * self.blade_radius

        # Initialize induced velocity
        v_i = np.full(self.n_elements, 1.0)

        # Iterative BEMT solution
        for iteration in range(50):
            v_i_old = v_i.copy()

            # Local velocities
            V_T = omega * self.r                      # tangential
            V_P = V_inf + v_i                         # perpendicular (axial)

            # Resultant velocity and inflow angle
            W = np.sqrt(V_T ** 2 + V_P ** 2)
            phi = np.arctan2(V_P, V_T + 1e-12)

            # Effective angle of attack
            alpha_eff = self.twist + collective - phi

            # Section aerodynamic coefficients
            CL_sec = np.array([
                self.section_aero.lift_coefficient(a) for a in alpha_eff
            ])
            CD_sec = np.array([
                self.section_aero.drag_coefficient(cl) for cl in CL_sec
            ])

            # Blade element forces per unit span
            dT_dr = (
                0.5 * rho * self.num_blades * self.chord * W ** 2
                * (CL_sec * np.cos(phi) - CD_sec * np.sin(phi))
            )

            # Momentum theory: dT = 4 * pi * r * rho * v_i * (V_inf + v_i) * dr
            # Solve for v_i from combined BEMT
            # Using Prandtl tip-loss factor F
            f_tip = (self.num_blades / 2.0) * (
                (self.blade_radius - self.r) / (self.r * np.sin(phi + 1e-12) + 1e-12)
            )
            F_tip = (2.0 / np.pi) * np.arccos(
                np.clip(np.exp(-np.abs(f_tip)), -1.0, 1.0)
            )

            # Update induced velocity from momentum balance
            denom = 4.0 * np.pi * self.r * rho * F_tip + 1e-12
            discriminant = (
                V_inf ** 2
                + dT_dr / (denom + 1e-12)
            )
            v_i_new = (
                -V_inf / 2.0
                + 0.5 * np.sqrt(np.maximum(discriminant, 0.0))
            )

            # Relaxation
            v_i = 0.3 * v_i_new + 0.7 * v_i_old

            # Check convergence
            if np.max(np.abs(v_i - v_i_old)) < 1e-6:
                break

        # Final integration
        V_T = omega * self.r
        V_P = V_inf + v_i
        W = np.sqrt(V_T ** 2 + V_P ** 2)
        phi = np.arctan2(V_P, V_T + 1e-12)
        alpha_eff = self.twist + collective - phi

        CL_sec = np.array([
            self.section_aero.lift_coefficient(a) for a in alpha_eff
        ])
        CD_sec = np.array([
            self.section_aero.drag_coefficient(cl) for cl in CL_sec
        ])

        dT = (
            0.5 * rho * self.num_blades * self.chord * W ** 2
            * (CL_sec * np.cos(phi) - CD_sec * np.sin(phi)) * dr
        )
        dQ = (
            0.5 * rho * self.num_blades * self.chord * W ** 2
            * (CD_sec * np.cos(phi) + CL_sec * np.sin(phi))
            * self.r * dr
        )

        thrust = np.sum(dT)
        torque = np.sum(dQ)
        power = torque * omega

        # Non-dimensional coefficients
        CT = thrust / (rho * n_rps ** 2 * D ** 4 + 1e-12)
        CQ = torque / (rho * n_rps ** 2 * D ** 5 + 1e-12)
        CP = power / (rho * n_rps ** 3 * D ** 5 + 1e-12)

        # Advance ratio
        J = V_inf / (n_rps * D + 1e-12) if V_inf > 0 else 0.0

        # Propulsive efficiency
        if power > 0 and V_inf > 0:
            eta = thrust * V_inf / power
        else:
            eta = 0.0

        return RotorPerformance(
            thrust=thrust,
            torque=torque,
            power=power,
            CT=CT,
            CQ=CQ,
            CP=CP,
            induced_velocity=np.mean(v_i),
            efficiency=np.clip(eta, 0.0, 1.0),
            advance_ratio=J,
        )

    def static_thrust(self, rpm: float, rho: float = 1.225) -> float:
        """
        Static thrust (zero forward speed).

        T = CT * rho * n^2 * D^4

        Parameters
        ----------
        rpm : float
            Rotor speed [rev/min].
        rho : float
            Air density [kg/m^3].

        Returns
        -------
        float
            Static thrust [N].
        """
        result = self.analyze(rpm=rpm, rho=rho, V_inf=0.0)
        return result.thrust


# ---------------------------------------------------------------------------
# Multirotor Model
# ---------------------------------------------------------------------------

class MultirotorModel:
    """
    Multirotor aerodynamic and power model.

    Computes hover and forward-flight performance for multi-rotor UAVs.
    Power is decomposed into induced, parasite, and profile components.

    Governing Equations (Hover):
        T_hover = m * g                              [weight support]
        P_hover = T^(3/2) / sqrt(2 * rho * A)        [ideal power]
        P_actual = kappa * P_hover + P_profile        [with corrections]

        where kappa ~ 1.15 accounts for non-uniform inflow
              P_profile = sigma * CD0_blade * rho * A * (Omega*R)^3 / 8

    Forward Flight Power:
        P_total = P_induced + P_parasite + P_profile
        P_parasite = 0.5 * rho * V^3 * f              [equivalent flat plate]
        P_induced  = kappa * T^2 / (2 * rho * A * V)  [Glauert high-speed]

    Parameters
    ----------
    num_rotors : int
        Number of rotors.
    rotor_radius : float
        Individual rotor radius [m].
    mass : float
        Total vehicle mass [kg].
    flat_plate_area : float
        Equivalent flat plate area for parasite drag [m^2].
    blade_solidity : float
        Blade solidity ratio sigma = N_b * c / (pi * R).
    cd0_blade : float
        Mean blade profile drag coefficient.
    """

    def __init__(
        self,
        num_rotors: int = 4,
        rotor_radius: float = 0.15,
        mass: float = 2.0,
        flat_plate_area: float = 0.02,
        blade_solidity: float = 0.05,
        cd0_blade: float = 0.012,
    ):
        self.num_rotors = num_rotors
        self.rotor_radius = rotor_radius
        self.mass = mass
        self.flat_plate_area = flat_plate_area
        self.blade_solidity = blade_solidity
        self.cd0_blade = cd0_blade

        # Total disk area (all rotors)
        self.disk_area_single = np.pi * rotor_radius ** 2
        self.disk_area_total = num_rotors * self.disk_area_single

        # Weight
        self.weight = mass * 9.80665  # [N]

    def hover_performance(
        self, rho: float = 1.225, kappa: float = 1.15
    ) -> MultirotorPerformance:
        """
        Hover performance analysis.

        P_hover = kappa * T^(3/2) / sqrt(2 * rho * A_total)
        P_profile = sigma * CD0 * rho * A * (Omega*R)^3 / 8

        Parameters
        ----------
        rho : float
            Air density [kg/m^3].
        kappa : float
            Induced power correction factor (1.1--1.2 typical).

        Returns
        -------
        MultirotorPerformance
            Hover performance metrics.
        """
        T = self.weight  # Thrust = weight in hover

        # Ideal induced power (momentum theory)
        P_ideal = T ** 1.5 / np.sqrt(2.0 * rho * self.disk_area_total)
        P_induced = kappa * P_ideal

        # Induced velocity in hover
        v_i = np.sqrt(T / (2.0 * rho * self.disk_area_total))

        # Profile power
        # P_profile ~ sigma * CD0 * rho * A * V_tip^3 / 8  per rotor
        # Estimate tip speed from thrust requirement
        # T_per_rotor = CT * rho * n^2 * D^4
        # Use momentum theory: T_per_rotor = 2 * rho * A * v_i^2
        T_per_rotor = T / self.num_rotors
        v_i_single = np.sqrt(
            T_per_rotor / (2.0 * rho * self.disk_area_single)
        )

        # Estimate tip speed: V_tip ~ v_i / lambda, lambda ~ 0.05 for hover
        V_tip = v_i_single / 0.05  # typical inflow ratio

        P_profile = (
            self.num_rotors
            * self.blade_solidity
            * self.cd0_blade
            * rho
            * self.disk_area_single
            * V_tip ** 3
            / 8.0
        )

        P_total = P_induced + P_profile

        return MultirotorPerformance(
            total_thrust=T,
            total_power=P_total,
            power_induced=P_induced,
            power_parasite=0.0,
            power_profile=P_profile,
            hover_endurance=0.0,  # computed externally with battery
            flight_regime=FlightRegime.HOVER,
        )

    def forward_flight_performance(
        self,
        velocity: float,
        rho: float = 1.225,
        kappa: float = 1.15,
    ) -> MultirotorPerformance:
        """
        Forward-flight power breakdown.

        P_total = P_induced + P_parasite + P_profile

        P_parasite = 0.5 * rho * V^3 * f
        P_induced  = kappa * T^2 / (2 * rho * A * V)     [Glauert, 1926]
        P_profile  ~ sigma * CD0 * rho * A * V_tip^3 / 8

        Parameters
        ----------
        velocity : float
            Forward flight speed [m/s].
        rho : float
            Air density [kg/m^3].
        kappa : float
            Induced power correction factor.

        Returns
        -------
        MultirotorPerformance
            Forward flight performance.
        """
        T = self.weight  # Approximate: thrust ~ weight (small tilt angle)
        V = max(velocity, 0.1)  # avoid division by zero

        # Parasite power
        P_parasite = 0.5 * rho * V ** 3 * self.flat_plate_area

        # Induced power (Glauert high-speed approximation)
        P_induced = kappa * T ** 2 / (2.0 * rho * self.disk_area_total * V)

        # Profile power (approximately constant)
        # Estimate from hover
        hover = self.hover_performance(rho, kappa)
        P_profile = hover.power_profile

        P_total = P_induced + P_parasite + P_profile

        regime = FlightRegime.CRUISE if velocity > 1.0 else FlightRegime.HOVER

        return MultirotorPerformance(
            total_thrust=T,
            total_power=P_total,
            power_induced=P_induced,
            power_parasite=P_parasite,
            power_profile=P_profile,
            hover_endurance=0.0,
            flight_regime=regime,
        )

    def best_endurance_speed(self, rho: float = 1.225) -> Tuple[float, float]:
        """
        Find minimum-power forward flight speed for best endurance.

        Minimizes P_total(V) = P_induced(V) + P_parasite(V) + P_profile.

        Returns
        -------
        tuple of (V_be, P_min)
            Best-endurance speed [m/s] and minimum power [W].
        """
        V_range = np.linspace(1.0, 40.0, 200)
        P_arr = np.array([
            self.forward_flight_performance(v, rho).total_power
            for v in V_range
        ])
        idx = np.argmin(P_arr)
        return V_range[idx], P_arr[idx]

    def best_range_speed(self, rho: float = 1.225) -> Tuple[float, float]:
        """
        Find speed for best range (minimum P/V).

        Minimizes P(V)/V, which maximizes distance per unit energy.

        Returns
        -------
        tuple of (V_br, P_V_min)
            Best-range speed [m/s] and P/V ratio [W/(m/s)].
        """
        V_range = np.linspace(2.0, 40.0, 200)
        PV_arr = np.array([
            self.forward_flight_performance(v, rho).total_power / v
            for v in V_range
        ])
        idx = np.argmin(PV_arr)
        return V_range[idx], PV_arr[idx]


# ---------------------------------------------------------------------------
# Fixed-Wing Model
# ---------------------------------------------------------------------------

class FixedWingModel:
    """
    Fixed-wing UAV aerodynamic model with trim solver.

    Computes lift, drag, moment, and solves for trimmed flight conditions
    (CL_trim, alpha_trim, elevator_trim). Includes stall angle
    determination and maximum L/D calculation.

    Governing Equations:
        L = 0.5 * rho * V^2 * S * CL                 [Lift]
        D = 0.5 * rho * V^2 * S * CD                  [Drag]
        M = 0.5 * rho * V^2 * S * c_bar * CM          [Moment]

    Trim Condition (steady level flight):
        L = W       =>  CL_trim = 2*W / (rho * V^2 * S)
        M = 0       =>  CM(alpha_trim, delta_e) = 0
        T = D       =>  T_req = 0.5 * rho * V^2 * S * CD(CL_trim)

    Parameters
    ----------
    wing_area : float
        Wing planform area S [m^2].
    wing_span : float
        Wing span b [m].
    mean_chord : float
        Mean aerodynamic chord c_bar [m].
    mass : float
        Vehicle mass [kg].
    airfoil : AirfoilType
        Wing airfoil.
    oswald_efficiency : float
        Oswald span efficiency factor.
    CM_delta_e : float
        Elevator control effectiveness dCM/d(delta_e) [1/rad].
    CL_delta_e : float
        Lift due to elevator dCL/d(delta_e) [1/rad].
    """

    def __init__(
        self,
        wing_area: float = 0.5,
        wing_span: float = 2.0,
        mean_chord: float = 0.25,
        mass: float = 3.0,
        airfoil: AirfoilType = AirfoilType.CLARK_Y,
        oswald_efficiency: float = 0.8,
        CM_delta_e: float = -1.2,
        CL_delta_e: float = 0.4,
    ):
        self.wing_area = wing_area
        self.wing_span = wing_span
        self.mean_chord = mean_chord
        self.mass = mass
        self.weight = mass * 9.80665
        self.CM_delta_e = CM_delta_e
        self.CL_delta_e = CL_delta_e

        self.aspect_ratio = wing_span ** 2 / wing_area

        self.aero = SubsonicAerodynamics(
            airfoil_type=airfoil,
            aspect_ratio=self.aspect_ratio,
            oswald_efficiency=oswald_efficiency,
        )

    def forces(
        self, alpha: float, velocity: float, rho: float = 1.225,
        delta_e: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Aerodynamic forces (lift, drag, moment) at given conditions.

        L = 0.5 * rho * V^2 * S * CL
        D = 0.5 * rho * V^2 * S * CD
        M = 0.5 * rho * V^2 * S * c * CM

        Parameters
        ----------
        alpha : float
            Angle of attack [rad].
        velocity : float
            Airspeed [m/s].
        rho : float
            Air density [kg/m^3].
        delta_e : float
            Elevator deflection [rad].

        Returns
        -------
        tuple of (Lift, Drag, Moment)
            Forces [N] and moment [N*m].
        """
        q = 0.5 * rho * velocity ** 2  # dynamic pressure

        CL = (
            self.aero.lift_coefficient(alpha)
            + self.CL_delta_e * delta_e
        )
        CD = self.aero.drag_coefficient(CL)
        CM = (
            self.aero.moment_coefficient(alpha)
            + self.CM_delta_e * delta_e
        )

        lift = q * self.wing_area * CL
        drag = q * self.wing_area * CD
        moment = q * self.wing_area * self.mean_chord * CM

        return lift, drag, moment

    def trim_solver(
        self, velocity: float, rho: float = 1.225
    ) -> TrimResult:
        """
        Solve for trimmed level flight conditions.

        Find alpha and delta_e such that:
            L = W   (lift equals weight)
            M = 0   (zero pitching moment)

        Uses Newton-Raphson iteration on the coupled system.

        Parameters
        ----------
        velocity : float
            Trim airspeed [m/s].
        rho : float
            Air density [kg/m^3].

        Returns
        -------
        TrimResult
            Trim angle of attack, elevator, CL, CD, thrust, L/D.
        """
        q = 0.5 * rho * velocity ** 2
        CL_required = self.weight / (q * self.wing_area)

        # Solve for alpha_trim from CL = CL_alpha*(alpha - alpha_L0) + CL_de*de
        # and CM0 + CM_alpha*alpha + CM_de*de = 0

        CL_alpha = self.aero.CL_alpha_3D
        alpha_L0 = self.aero.params['alpha_L0']
        CM0 = self.aero.params['CM0']
        CM_alpha = self.aero.params['CM_alpha']

        # Linear system:
        # CL_alpha*(alpha-alpha_L0) + CL_de*de = CL_req
        # CM0 + CM_alpha*alpha + CM_de*de = 0
        #
        # [ CL_alpha   CL_de ] [alpha]   [CL_req - CL_alpha*(-alpha_L0)]
        # [ CM_alpha   CM_de ] [ de  ] = [-CM0                          ]

        A_mat = np.array([
            [CL_alpha, self.CL_delta_e],
            [CM_alpha, self.CM_delta_e],
        ])
        b_vec = np.array([
            CL_required + CL_alpha * alpha_L0,
            -CM0,
        ])

        det = np.linalg.det(A_mat)
        if abs(det) < 1e-12:
            # Singular; fallback to alpha-only trim
            alpha_trim = (CL_required / CL_alpha) + alpha_L0
            delta_e_trim = 0.0
        else:
            sol = np.linalg.solve(A_mat, b_vec)
            alpha_trim = sol[0]
            delta_e_trim = sol[1]

        # Compute coefficients at trim
        CL_trim = CL_required
        CD_trim = self.aero.drag_coefficient(CL_trim)
        L_over_D = CL_trim / (CD_trim + 1e-12)
        thrust_required = q * self.wing_area * CD_trim
        power_required = thrust_required * velocity

        return TrimResult(
            CL_trim=CL_trim,
            alpha_trim=alpha_trim,
            elevator_trim=delta_e_trim,
            CD_trim=CD_trim,
            thrust_required=thrust_required,
            L_over_D=L_over_D,
            power_required=power_required,
        )

    def stall_speed(self, rho: float = 1.225) -> float:
        """
        Stall speed in level flight.

        V_stall = sqrt(2 * W / (rho * S * CL_max))

        Parameters
        ----------
        rho : float
            Air density [kg/m^3].

        Returns
        -------
        float
            Stall speed [m/s].
        """
        CL_max = self.aero.params['CL_max']
        return np.sqrt(
            2.0 * self.weight / (rho * self.wing_area * CL_max)
        )

    def stall_angle(self) -> float:
        """
        Stall angle of attack from airfoil data.

        Returns
        -------
        float
            Stall angle [rad].
        """
        return self.aero.params['alpha_stall']

    def max_lift_to_drag(self) -> Tuple[float, float]:
        """
        Maximum lift-to-drag ratio.

        (L/D)_max = 0.5 * sqrt(pi * e * AR / CD0)

        Returns
        -------
        tuple of (L_D_max, CL_opt)
        """
        return self.aero.max_lift_to_drag()

    def power_required_curve(
        self, V_range: np.ndarray, rho: float = 1.225
    ) -> np.ndarray:
        """
        Power required vs. airspeed.

        P_req = D * V = 0.5 * rho * V^3 * S * CD0
                      + 2 * W^2 / (rho * V * S * pi * e * AR)

        Parameters
        ----------
        V_range : ndarray
            Airspeed values [m/s].
        rho : float
            Air density [kg/m^3].

        Returns
        -------
        ndarray
            Power required [W] at each airspeed.
        """
        P_req = np.zeros_like(V_range, dtype=float)
        for i, V in enumerate(V_range):
            if V < 0.1:
                P_req[i] = np.inf
                continue
            trim = self.trim_solver(V, rho)
            P_req[i] = trim.power_required
        return P_req

    def best_endurance_speed(self, rho: float = 1.225) -> float:
        """
        Speed for minimum power (best endurance).

        V_be = sqrt(2*W / (rho*S)) * (1 / (3*CD0*pi*e*AR))^(1/4)

        Returns
        -------
        float
            Best endurance speed [m/s].
        """
        CD0 = self.aero.params['CD0']
        e = self.aero.oswald_efficiency
        AR = self.aspect_ratio
        k = 1.0 / (np.pi * e * AR)

        # V_min_power = sqrt(2W/(rho*S)) * (k / (3*CD0))^(1/4)
        V_be = np.sqrt(2.0 * self.weight / (rho * self.wing_area)) * (
            k / (3.0 * CD0)
        ) ** 0.25
        return V_be

    def best_range_speed(self, rho: float = 1.225) -> float:
        """
        Speed for minimum drag (best range).

        At best range: CL = sqrt(CD0 * pi * e * AR) => V_br = V_be * 3^(1/4)

        Returns
        -------
        float
            Best range speed [m/s].
        """
        return self.best_endurance_speed(rho) * 3.0 ** 0.25

    def flight_envelope(
        self,
        rho: float = 1.225,
        thrust_available: float = 20.0,
    ) -> Dict[str, float]:
        """
        Compute key flight envelope parameters.

        Parameters
        ----------
        rho : float
            Air density [kg/m^3].
        thrust_available : float
            Maximum available thrust [N].

        Returns
        -------
        dict
            V_stall, V_max, V_best_endurance, V_best_range,
            L_D_max, CL_opt, max_rate_of_climb.
        """
        V_stall = self.stall_speed(rho)
        L_D_max, CL_opt = self.max_lift_to_drag()
        V_be = self.best_endurance_speed(rho)
        V_br = self.best_range_speed(rho)

        # Maximum speed: T_avail = D at V_max
        # Iterative search
        V_max = V_stall
        for V_test in np.linspace(V_stall, 100.0, 500):
            trim = self.trim_solver(V_test, rho)
            if trim.thrust_required > thrust_available:
                V_max = V_test
                break
            V_max = V_test

        # Max rate of climb at best endurance speed
        trim_be = self.trim_solver(V_be, rho)
        excess_power = (thrust_available - trim_be.thrust_required) * V_be
        roc_max = excess_power / self.weight if excess_power > 0 else 0.0

        return {
            'V_stall': V_stall,
            'V_max': V_max,
            'V_best_endurance': V_be,
            'V_best_range': V_br,
            'L_D_max': L_D_max,
            'CL_opt': CL_opt,
            'max_rate_of_climb': roc_max,
        }


# ---------------------------------------------------------------------------
# Module-level __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    'AirfoilType',
    'FlightRegime',
    'UAVType',
    # Result dataclasses
    'AeroCoefficients',
    'RotorPerformance',
    'TrimResult',
    'MultirotorPerformance',
    # Airfoil presets
    'AIRFOIL_PRESETS',
    # Classes
    'SubsonicAerodynamics',
    'BladeElementTheory',
    'MultirotorModel',
    'FixedWingModel',
]
