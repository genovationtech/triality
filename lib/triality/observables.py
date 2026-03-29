"""
Triality Observable Layer
=========================

Transforms raw solver PhysicsState fields into domain-specific engineering
observables — the quantities that actually answer design questions.

Pipeline position:
    Intent → Solver → Fields → **Observables** → Interpretation

Each module registers an ObservableSet that knows how to derive the
observables that matter for its physics domain.

Usage:
    >>> from triality.observables import compute_observables
    >>> obs = compute_observables("navier_stokes", state, config)
    >>> for o in obs:
    ...     print(f"{o.name}: {o.value:.4g} {o.unit}  — {o.relevance}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np

from triality.core.fields import PhysicsState


# ---------------------------------------------------------------------------
# Observable dataclass
# ---------------------------------------------------------------------------

@dataclass
class Observable:
    """A single derived engineering quantity."""

    name: str                       # e.g. "peak_velocity"
    value: Any                      # float, ndarray, bool, str
    unit: str                       # e.g. "m/s"
    description: str                # human-readable one-liner
    relevance: str = ""             # why it matters for the question
    threshold: Optional[float] = None   # optional pass/fail threshold
    margin: Optional[float] = None      # distance to threshold
    rank: int = 99                  # lower = more important (0 = primary)

    @property
    def is_scalar(self) -> bool:
        return isinstance(self.value, (int, float, bool, np.integer, np.floating))

    def to_dict(self) -> Dict[str, Any]:
        """Serializable dict for JSON transport."""
        v = self.value
        if isinstance(v, np.ndarray):
            v = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            v = float(v)
        d: Dict[str, Any] = {
            "name": self.name,
            "value": v,
            "unit": self.unit,
            "description": self.description,
            "rank": self.rank,
        }
        if self.relevance:
            d["relevance"] = self.relevance
        if self.threshold is not None:
            d["threshold"] = self.threshold
        if self.margin is not None:
            d["margin"] = self.margin
        return d


# ---------------------------------------------------------------------------
# Base class for per-module observable sets
# ---------------------------------------------------------------------------

class BaseObservableSet(ABC):
    """Abstract base for module-specific observable computation."""

    module_name: str = "unknown"
    domain: str = "unknown"

    @abstractmethod
    def compute(
        self,
        state: PhysicsState,
        config: Dict[str, Any],
        native_result: Any = None,
    ) -> List[Observable]:
        """Derive observables from solved state.

        Parameters
        ----------
        state : PhysicsState
            The generated_state from RuntimeExecutionResult.
        config : dict
            The merged config that was passed to the solver.
        native_result : Any, optional
            The raw solver result object (for accessing metadata not in state).

        Returns
        -------
        list of Observable, sorted by rank (most important first).
        """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OBSERVABLE_REGISTRY: Dict[str, Type[BaseObservableSet]] = {}


def register_observables(cls: Type[BaseObservableSet]) -> Type[BaseObservableSet]:
    """Decorator to register an observable set for a module."""
    OBSERVABLE_REGISTRY[cls.module_name] = cls
    return cls


def compute_observables(
    module_name: str,
    state: PhysicsState,
    config: Dict[str, Any],
    native_result: Any = None,
) -> List[Observable]:
    """Compute observables for a given module.

    Returns an empty list if no observable set is registered.
    """
    cls = OBSERVABLE_REGISTRY.get(module_name)
    if cls is None:
        return []
    obs_set = cls()
    observables = obs_set.compute(state, config, native_result)
    observables.sort(key=lambda o: o.rank)
    return observables


# ---------------------------------------------------------------------------
# Helper utilities used by many observable sets
# ---------------------------------------------------------------------------

def _safe_max(arr: np.ndarray) -> float:
    """max of array, handling empty / NaN / Inf without warnings."""
    finite = arr[np.isfinite(arr)] if arr.size > 0 else arr
    return float(np.max(finite)) if finite.size > 0 else 0.0


def _safe_min(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)] if arr.size > 0 else arr
    return float(np.min(finite)) if finite.size > 0 else 0.0


def _safe_mean(arr: np.ndarray) -> float:
    finite = arr[np.isfinite(arr)] if arr.size > 0 else arr
    return float(np.mean(finite)) if finite.size > 0 else 0.0


def _get_field(state: PhysicsState, name: str) -> Optional[np.ndarray]:
    """Get a field's data array by name, or None."""
    if state.has(name):
        return np.asarray(state.fields[name].data)
    return None


def _meta(state: PhysicsState, key: str, default: Any = None) -> Any:
    return state.metadata.get(key, default)


# ===================================================================
# PER-MODULE OBSERVABLE SETS
# ===================================================================

# -------------------------------------------------------------------
# 1. Navier-Stokes
# -------------------------------------------------------------------

@register_observables
class NavierStokesObservables(BaseObservableSet):
    module_name = "navier_stokes"
    domain = "fluid_dynamics"

    def compute(self, state, config, native_result=None):
        obs = []
        u = _get_field(state, "velocity_x")
        v = _get_field(state, "velocity_y")
        p = _get_field(state, "pressure")

        # Velocity magnitude
        if u is not None and v is not None:
            # Handle staggered grids: use overlap region
            min_shape = tuple(min(a, b) for a, b in zip(u.shape, v.shape))
            uc = u[:min_shape[0], :min_shape[1]] if u.ndim == 2 else u[:min(len(u), len(v))]
            vc = v[:min_shape[0], :min_shape[1]] if v.ndim == 2 else v[:min(len(u), len(v))]
            speed = np.sqrt(uc**2 + vc**2)
            max_vel = _safe_max(speed)
            mean_vel = _safe_mean(speed)

            obs.append(Observable("max_velocity", round(max_vel, 6), "m/s",
                                  "Peak flow speed in the domain", rank=0))
            obs.append(Observable("mean_velocity", round(mean_vel, 6), "m/s",
                                  "Domain-averaged flow speed", rank=3))

            # Vorticity (for 2D)
            if u.ndim == 2 and u.shape[0] > 2 and u.shape[1] > 2:
                sc = config.get("solver", config)
                Lx = float(sc.get("Lx", 1.0))
                Ly = float(sc.get("Ly", 1.0))
                nx = u.shape[1]
                ny = u.shape[0]
                dx = Lx / max(nx - 1, 1)
                dy = Ly / max(ny - 1, 1)
                # dvdx - dudy on the interior
                n0, n1 = min(u.shape[0], v.shape[0]), min(u.shape[1], v.shape[1])
                if n0 > 2 and n1 > 2:
                    dvdx = (v[1:n0-1, 2:n1] - v[1:n0-1, :n1-2]) / (2 * dx)
                    dudy = (u[2:n0, 1:n1-1] - u[:n0-2, 1:n1-1]) / (2 * dy)
                    mn = min(dvdx.shape[0], dudy.shape[0]), min(dvdx.shape[1], dudy.shape[1])
                    vort = dvdx[:mn[0], :mn[1]] - dudy[:mn[0], :mn[1]]
                    vort_vals = np.abs(vort)
                    vort_valid = vort_vals[np.isfinite(vort_vals)]
                    if vort_valid.size > 0:
                        obs.append(Observable("vorticity_peak", round(float(np.max(vort_valid)), 4), "1/s",
                                              "Peak vorticity magnitude — indicates mixing intensity", rank=1))

            # Dead zone fraction (velocity < 10% of max)
            if max_vel > 0:
                dead_frac = float(np.mean(speed < 0.1 * max_vel))
                obs.append(Observable("dead_zone_fraction", round(dead_frac, 4), "",
                                      "Fraction of domain with velocity below 10% of peak",
                                      relevance="High dead zone fraction indicates poor mixing", rank=2))

        # Reynolds number
        sc = config.get("solver", config)
        nu = float(sc.get("nu", 0.01))
        U = float(sc.get("U_lid", 0.1))
        L = float(sc.get("Lx", 1.0))
        Re = U * L / nu if nu > 0 else 0
        obs.append(Observable("reynolds_number", round(Re, 1), "",
                              "Flow regime indicator (laminar < 2000, turbulent > 4000)", rank=4))

        # Pressure drop
        if p is not None:
            p_finite = p[np.isfinite(p)]
            if p_finite.size > 0:
                dp = float(np.max(p_finite) - np.min(p_finite))
                obs.append(Observable("pressure_drop", round(dp, 4), "Pa",
                                      "Pressure difference across domain — relates to pumping cost", rank=5))

        # Deep observables
        ke = _meta(state, "kinetic_energy")
        if ke is not None:
            obs.append(Observable("kinetic_energy", round(float(ke), 6), "J/m",
                                  "Domain-integrated kinetic energy per unit depth", rank=6))
        div_rms = _meta(state, "divergence_rms")
        if div_rms is not None:
            obs.append(Observable("divergence_rms", float(div_rms), "1/s",
                                  "RMS velocity divergence — measures incompressibility error", rank=7))
        if u is not None and u.ndim == 2 and u.shape[0] > 2:
            sc = config.get("solver", config)
            Ly = float(sc.get("Ly", 1.0))
            ny = u.shape[0]
            dy = Ly / max(ny - 1, 1)
            # Wall shear at bottom wall (y=0): tau = mu * du/dy|_wall
            du_dy_wall = (u[1, :] - u[0, :]) / dy
            nu_val = float(sc.get("nu", 0.01))
            rho_val = float(sc.get("rho", 1.0))
            mu = rho_val * nu_val
            tau_wall = mu * _safe_mean(np.abs(du_dy_wall))
            obs.append(Observable("wall_shear_stress_mean", round(float(tau_wall), 6), "Pa",
                                  "Mean wall shear stress at bottom boundary", rank=8))
        conv = _meta(state, "converged")
        if conv is not None:
            obs.append(Observable("converged", bool(conv), "", "Whether the solver converged", rank=9))

        return obs


# -------------------------------------------------------------------
# 2. Drift-Diffusion
# -------------------------------------------------------------------

@register_observables
class DriftDiffusionObservables(BaseObservableSet):
    module_name = "drift_diffusion"
    domain = "semiconductor_devices"

    def compute(self, state, config, native_result=None):
        obs = []
        m = state.metadata

        Vbi = m.get("built_in_potential")
        if Vbi is not None:
            obs.append(Observable("built_in_potential", round(float(Vbi), 4), "V",
                                  "Junction built-in potential", rank=0))

        dw = m.get("depletion_width")
        if dw is not None:
            obs.append(Observable("depletion_width", round(float(dw), 6), "cm",
                                  "Depletion region width — determines junction capacitance", rank=1))

        E = _get_field(state, "electric_field")
        if E is not None:
            peak_E = _safe_max(np.abs(E))
            obs.append(Observable("peak_electric_field", round(peak_E, 1), "V/cm",
                                  "Maximum electric field in the junction", rank=2))

        J = _get_field(state, "current_density_total")
        if J is not None and J.size > 0:
            # Terminal current at the right contact
            J_terminal = float(J[-1]) if J.ndim == 1 else _safe_mean(J)
            obs.append(Observable("terminal_current_density", J_terminal, "A/cm^2",
                                  "Current density at device terminal", rank=3))

        cd = m.get("current_density")
        if cd is not None:
            obs.append(Observable("current_density", float(cd), "A/cm^2",
                                  "Net device current density", rank=4))

        conv = m.get("converged")
        if conv is not None:
            obs.append(Observable("converged", bool(conv), "",
                                  "Whether the Gummel iteration converged", rank=10))

        # I-V data
        iv_v = m.get("iv_voltages")
        iv_i = m.get("iv_currents")
        if iv_v is not None and iv_i is not None:
            obs.append(Observable("iv_points", len(iv_v), "",
                                  "Number of I-V curve data points available", rank=8))

        # Deep observables
        T = m.get("temperature")
        if T is not None:
            obs.append(Observable("operating_temperature", float(T), "K",
                                  "Device operating temperature", rank=5))
        mat = m.get("material_name")
        if mat is not None:
            obs.append(Observable("material", str(mat), "", "Semiconductor material", rank=6))
        # I-V derived quantities
        if iv_v is not None and iv_i is not None and len(iv_v) >= 3:
            iv_v_arr = np.array(iv_v)
            iv_i_arr = np.array(iv_i)
            # Forward voltage at threshold (1 mA/cm^2 equivalent)
            threshold = 1e-3
            above = np.where(np.abs(iv_i_arr) > threshold)[0]
            if len(above) > 0:
                Vf = float(iv_v_arr[above[0]])
                obs.append(Observable("forward_voltage_threshold", round(Vf, 4), "V",
                                      "Voltage at which |J| first exceeds 1 mA/cm^2", rank=7))
            # Reverse saturation current (J at V=0 or smallest V)
            J0 = float(np.abs(iv_i_arr[0]))
            if J0 > 0:
                obs.append(Observable("reverse_saturation_current", J0, "A/cm^2",
                                      "Reverse saturation current density J0", rank=8))
            # Ideality factor from slope of ln(J) vs V in forward bias
            fwd_mask = (iv_v_arr > 0.1) & (np.abs(iv_i_arr) > 1e-10)
            if np.sum(fwd_mask) >= 2:
                V_fwd = iv_v_arr[fwd_mask]
                J_fwd = np.abs(iv_i_arr[fwd_mask])
                lnJ = np.log(J_fwd)
                if len(V_fwd) >= 2:
                    slope = (lnJ[-1] - lnJ[0]) / (V_fwd[-1] - V_fwd[0]) if (V_fwd[-1] - V_fwd[0]) > 0 else 0
                    kT = 8.617e-5 * float(T if T else 300)  # eV
                    n_ideality = 1.0 / (slope * kT) if slope > 0 else 0
                    if 0.5 < n_ideality < 5.0:
                        obs.append(Observable("ideality_factor", round(n_ideality, 3), "",
                                              "Diode ideality factor (1=ideal Shockley, 2=recombination dominated)", rank=9))
        # Junction capacitance from depletion width
        if dw is not None and float(dw) > 0:
            eps_Si = 11.7 * 8.854e-14  # F/cm
            C_j = eps_Si / float(dw)  # F/cm^2
            obs.append(Observable("junction_capacitance", C_j, "F/cm^2",
                                  "Depletion capacitance per unit area (C = eps/W)", rank=10))

        return obs


# -------------------------------------------------------------------
# 3. Sensing
# -------------------------------------------------------------------

@register_observables
class SensingObservables(BaseObservableSet):
    module_name = "sensing"
    domain = "sensing"

    def compute(self, state, config, native_result=None):
        obs = []
        Pd = _get_field(state, "detection_probability")
        if Pd is None:
            return obs

        mean_pd = _safe_mean(Pd)
        obs.append(Observable("mean_detection_probability", round(mean_pd, 4), "",
                              "Spatial average detection probability", rank=0))

        cov90 = float(np.mean(Pd > 0.9))
        obs.append(Observable("coverage_fraction_90pct", round(cov90, 4), "",
                              "Fraction of area with Pd > 90%",
                              relevance="Reliable detection coverage", rank=1))

        cov50 = float(np.mean(Pd > 0.5))
        obs.append(Observable("coverage_fraction_50pct", round(cov50, 4), "",
                              "Fraction of area with Pd > 50%", rank=2))

        blind = float(np.mean(Pd < 0.1))
        obs.append(Observable("blind_zone_fraction", round(blind, 4), "",
                              "Fraction of area with Pd < 10%",
                              relevance="Security gap — blind zones need additional sensors", rank=3))

        # Max detection range — prefer metadata (authoritative) over pixel estimate
        meta_range = _meta(state, "max_detection_range_km")
        if isinstance(meta_range, dict):
            # metadata is {sensor_name: range_km}
            for sensor_name, rng in meta_range.items():
                obs.append(Observable(f"max_detection_range_km", round(float(rng), 2), "km",
                                      f"Maximum detection range ({sensor_name})", rank=4))
        elif isinstance(meta_range, (int, float)):
            obs.append(Observable("max_detection_range_km", round(float(meta_range), 2), "km",
                                  "Maximum detection range", rank=4))
        elif Pd.ndim == 2:
            # Fallback: estimate from grid
            ny, nx = Pd.shape
            cy, cx = ny // 2, nx // 2
            yy, xx = np.mgrid[0:ny, 0:nx]
            r = np.sqrt((xx - cx)**2 + (yy - cy)**2)
            mask = Pd > 0.5
            if np.any(mask):
                max_r_pixels = float(np.max(r[mask]))
                grid_cfg = config.get("grid", config)
                gx = config.get("grid_x_km", grid_cfg.get("grid_x_km", (-10, 10)))
                if isinstance(gx, (list, tuple)) and len(gx) == 2:
                    extent_km = float(gx[1] - gx[0])
                    range_km = max_r_pixels / max(nx, 1) * extent_km
                    obs.append(Observable("max_detection_range_km", round(range_km, 2), "km",
                                          "Maximum range at which Pd > 50% (estimated)", rank=4))

        # Deep observables
        snr = _meta(state, "snr_maps_db")
        if snr is not None and isinstance(snr, dict):
            for sensor_name, snr_map in snr.items():
                arr = np.asarray(snr_map)
                peak_snr = _safe_max(arr)
                obs.append(Observable(f"peak_snr_{sensor_name}", round(peak_snr, 1), "dB",
                                      f"Peak SNR for sensor {sensor_name}", rank=5))
        # Detection contour area
        if Pd is not None and Pd.ndim == 2:
            ny, nx = Pd.shape
            grid_cfg = config.get("grid", config)
            gx = config.get("grid_x_km", grid_cfg.get("grid_x_km", (-10, 10)))
            gy = config.get("grid_y_km", grid_cfg.get("grid_y_km", (-10, 10)))
            if isinstance(gx, (list, tuple)) and isinstance(gy, (list, tuple)):
                cell_area = (float(gx[1]-gx[0])/nx) * (float(gy[1]-gy[0])/ny)
                area_90 = float(np.sum(Pd > 0.9)) * cell_area
                obs.append(Observable("detection_area_90pct_km2", round(area_90, 2), "km^2",
                                      "Area with detection probability > 90%", rank=6))
        sensor_count = _meta(state, "sensor_count")
        if sensor_count is not None:
            obs.append(Observable("sensor_count", int(sensor_count), "",
                                  "Number of sensors in the network", rank=7))

        return obs


# -------------------------------------------------------------------
# 4. Electrostatics
# -------------------------------------------------------------------

@register_observables
class ElectrostaticsObservables(BaseObservableSet):
    module_name = "electrostatics"
    domain = "electromagnetism"

    def compute(self, state, config, native_result=None):
        obs = []
        V = _get_field(state, "electric_potential")
        E = _get_field(state, "electric_field")
        Ex = _get_field(state, "electric_field_x")
        Ey = _get_field(state, "electric_field_y")

        if V is not None:
            obs.append(Observable("potential_range", round(_safe_max(V) - _safe_min(V), 2), "V",
                                  "Voltage drop across domain", rank=1))

        E_mag = _get_field(state, "electric_field_magnitude") or E
        if E_mag is None and Ex is not None and Ey is not None:
            E_mag = np.sqrt(Ex**2 + Ey**2)

        if E_mag is not None:
            peak_E = _safe_max(E_mag)
            mean_E = _safe_mean(E_mag)
            obs.append(Observable("peak_field_strength", round(peak_E, 1), "V/m",
                                  "Maximum electric field — determines breakdown risk", rank=0))
            obs.append(Observable("mean_field_strength", round(mean_E, 1), "V/m",
                                  "Average electric field", rank=3))

            if mean_E > 0:
                uniformity = float(np.nanstd(E_mag) / mean_E)
                obs.append(Observable("field_uniformity", round(uniformity, 4), "",
                                      "Field non-uniformity (std/mean) — 0 is perfectly uniform", rank=4))

            # Breakdown margin (air breakdown ~ 3e6 V/m)
            E_bd = 3e6
            margin = (E_bd - peak_E) / E_bd
            obs.append(Observable("breakdown_margin", round(margin, 4), "",
                                  "Margin to air dielectric breakdown (3 MV/m)",
                                  threshold=0.0, margin=round(margin, 4), rank=2))

        # Deep observables from metadata
        mfl = _meta(state, "max_field_x_m")
        mfly = _meta(state, "max_field_y_m")
        if mfl is not None and mfly is not None:
            obs.append(Observable("peak_field_location", f"({float(mfl):.4f}, {float(mfly):.4f})", "m",
                                  "Location of peak electric field", rank=5))
        hfz = _meta(state, "high_field_zone_count")
        if hfz is not None:
            obs.append(Observable("high_field_zone_count", int(hfz), "",
                                  "Number of detected high-field zones (>90th percentile)", rank=6))
        # Energy stored
        if E_mag is not None and V is not None:
            sc = config.get("solver", config)
            eps = float(sc.get("permittivity", 8.854e-12))
            Lx = float(sc.get("x_max", sc.get("x_range", [0, 0.1])[1] if isinstance(sc.get("x_range"), list) else 0.1))
            Ly_e = float(sc.get("y_max", sc.get("y_range", [0, 0.1])[1] if isinstance(sc.get("y_range"), list) else 0.1))
            res = E_mag.shape[0] if E_mag.ndim >= 1 else 50
            dx_e = Lx / max(res - 1, 1)
            dy_e = Ly_e / max(res - 1, 1)
            energy = 0.5 * eps * float(np.nansum(E_mag**2)) * dx_e * dy_e
            obs.append(Observable("energy_stored", energy, "J",
                                  "Total electrostatic energy stored in the field", rank=7))

        return obs


# -------------------------------------------------------------------
# 5. Aero Loads
# -------------------------------------------------------------------

@register_observables
class AeroLoadsObservables(BaseObservableSet):
    module_name = "aero_loads"
    domain = "aerodynamics"

    def compute(self, state, config, native_result=None):
        obs = []
        m = state.metadata

        # Coefficients are in fields for aero_loads
        CL_f = _get_field(state, "lift_coefficient")
        CD_f = _get_field(state, "drag_coefficient")
        CL = _safe_mean(CL_f) if CL_f is not None else float(m.get("lift_coefficient", 0))
        CD = _safe_mean(CD_f) if CD_f is not None else float(m.get("drag_coefficient", 0))

        obs.append(Observable("lift_coefficient", round(float(CL), 5), "", "Aerodynamic lift coefficient", rank=0))
        obs.append(Observable("drag_coefficient", round(float(CD), 5), "", "Aerodynamic drag coefficient", rank=1))

        if CD and CD != 0:
            obs.append(Observable("lift_to_drag_ratio", round(float(CL / CD), 3), "",
                                  "Aerodynamic efficiency L/D", rank=2))

        hf = _get_field(state, "heat_flux")
        if hf is not None and hf.size > 0:
            obs.append(Observable("peak_heat_flux", round(_safe_max(hf), 1), "W/m^2",
                                  "Peak surface heat flux — TPS sizing driver", rank=3))

        p = _get_field(state, "pressure")
        if p is not None and p.size > 0:
            obs.append(Observable("peak_pressure", round(_safe_max(p), 1), "Pa",
                                  "Peak surface pressure", rank=5))

        for key, name, unit, desc, rank in [
            ("total_heat_load_W", "total_heat_load", "W", "Integrated aerothermal heat load", 4),
            ("divergence_q_Pa", "divergence_dynamic_pressure", "Pa", "Aeroelastic divergence dynamic pressure", 6),
        ]:
            v = m.get(key)
            if v is not None:
                obs.append(Observable(name, round(float(v), 1), unit, desc, rank=rank))

        # Deep observables from fields
        Cp = _get_field(state, "pressure_coefficient")
        if Cp is not None and Cp.size > 0:
            obs.append(Observable("max_Cp", round(_safe_max(Cp), 4), "",
                                  "Peak pressure coefficient — indicates stagnation", rank=7))
            obs.append(Observable("mean_Cp", round(_safe_mean(Cp), 4), "",
                                  "Mean pressure coefficient over body surface", rank=8))
        if hf is not None and hf.size > 0:
            mean_hf = _safe_mean(hf)
            obs.append(Observable("mean_heat_flux", round(mean_hf, 1), "W/m^2",
                                  "Average surface heat flux", rank=9))

        return obs


# -------------------------------------------------------------------
# 6. UAV Aerodynamics
# -------------------------------------------------------------------

@register_observables
class UAVAerodynamicsObservables(BaseObservableSet):
    module_name = "uav_aerodynamics"
    domain = "aerodynamics"

    def compute(self, state, config, native_result=None):
        obs = []
        CL_f = _get_field(state, "lift_coefficient")
        CD_f = _get_field(state, "drag_coefficient")

        CL = _safe_mean(CL_f) if CL_f is not None else 0.0
        CD_i = _safe_mean(CD_f) if CD_f is not None else 0.0

        obs.append(Observable("lift_coefficient", round(CL, 5), "",
                              "Total wing lift coefficient", rank=0))
        obs.append(Observable("induced_drag_coefficient", round(CD_i, 6), "",
                              "Induced drag coefficient", rank=1))

        if CD_i > 0:
            LD = CL / CD_i
            obs.append(Observable("lift_to_drag_ratio", round(LD, 2), "",
                                  "Wing L/D ratio — key endurance metric", rank=2))

        sc = config.get("solver", config)
        span = float(sc.get("span", 10.0))
        S = span * float(sc.get("root_chord", 1.0))  # approximate
        AR = span**2 / S if S > 0 else 0
        if CD_i > 0 and AR > 0:
            e = CL**2 / (np.pi * AR * CD_i) if CD_i > 0 else 0
            obs.append(Observable("span_efficiency", round(float(e), 4), "",
                                  "Oswald span efficiency — 1.0 is ideal elliptic loading", rank=3))

        # Deep observables
        obs.append(Observable("aspect_ratio", round(float(AR), 2), "",
                              "Wing aspect ratio (span^2/S)", rank=4))
        Cm = _meta(state, "Cm")
        if Cm is not None:
            obs.append(Observable("pitching_moment_coefficient", round(float(Cm), 5), "",
                                  "Pitching moment coefficient about root LE", rank=5))
        circ = _get_field(state, "velocity")  # circulation stored in velocity field
        if circ is not None and circ.ndim == 1 and circ.size > 0:
            root_load = float(np.abs(circ[len(circ)//2]))
            obs.append(Observable("root_circulation", round(root_load, 4), "m^2/s",
                                  "Bound circulation at wing root — proportional to root bending load", rank=6))

        return obs


# -------------------------------------------------------------------
# 7. Spacecraft Thermal
# -------------------------------------------------------------------

@register_observables
class SpacecraftThermalObservables(BaseObservableSet):
    module_name = "spacecraft_thermal"
    domain = "thermal"

    def compute(self, state, config, native_result=None):
        obs = []
        T = _get_field(state, "temperature")
        m = state.metadata

        if T is not None and T.size > 0:
            obs.append(Observable("peak_temperature", round(_safe_max(T), 1), "K",
                                  "Highest temperature across all nodes", rank=0))
            obs.append(Observable("min_temperature", round(_safe_min(T), 1), "K",
                                  "Lowest temperature across all nodes", rank=1))

            sc = config.get("solver", config)
            T_max_lim = float(sc.get("T_max_limit", 333.0))
            T_min_lim = float(sc.get("T_min_limit", 233.0))
            hot_margin = T_max_lim - _safe_max(T)
            cold_margin = _safe_min(T) - T_min_lim
            obs.append(Observable("hot_margin", round(hot_margin, 1), "K",
                                  f"Margin to upper survival limit ({T_max_lim:.0f}K)",
                                  threshold=0.0, margin=round(hot_margin, 1), rank=2))
            obs.append(Observable("cold_margin", round(cold_margin, 1), "K",
                                  f"Margin to lower survival limit ({T_min_lim:.0f}K)",
                                  threshold=0.0, margin=round(cold_margin, 1), rank=3))

        hf = _get_field(state, "heat_flux")
        if hf is not None and hf.size > 0:
            obs.append(Observable("max_heat_flux", round(_safe_max(hf), 1), "W",
                                  "Peak heat flux at any node", rank=5))

        # Deep observables
        if T is not None and T.ndim >= 1 and T.size >= 2:
            spread = _safe_max(T) - _safe_min(T)
            obs.append(Observable("node_temperature_spread", round(spread, 1), "K",
                                  "Temperature difference between hottest and coldest nodes", rank=4))
            if T.ndim == 2 and T.shape[0] >= 2:
                # Gradient between adjacent nodes at final time
                dT = np.abs(np.diff(T[-1])) if T.ndim == 2 else np.abs(np.diff(T))
                obs.append(Observable("max_node_gradient", round(float(np.max(dT)), 1), "K",
                                      "Maximum temperature difference between adjacent nodes", rank=6))
        node_names = _meta(state, "node_names")
        if node_names and T is not None:
            if T.ndim == 2 and T.shape[1] == len(node_names):
                hottest_idx = int(np.argmax(T[-1]))
                obs.append(Observable("hottest_node", node_names[hottest_idx], "",
                                      "Node with the highest final temperature", rank=7))

        return obs


# -------------------------------------------------------------------
# 8. Automotive Thermal
# -------------------------------------------------------------------

@register_observables
class AutomotiveThermalObservables(BaseObservableSet):
    module_name = "automotive_thermal"
    domain = "thermal"

    def compute(self, state, config, native_result=None):
        obs = []
        T = _get_field(state, "temperature")
        m = state.metadata

        max_T = m.get("max_temperature", _safe_max(T) if T is not None else 0)
        obs.append(Observable("junction_temperature_peak", round(float(max_T), 1), "K",
                              "Peak component temperature", rank=0))

        # Convert to Celsius for display
        max_C = float(max_T) - 273.15
        obs.append(Observable("junction_temperature_peak_C", round(max_C, 1), "°C",
                              "Peak component temperature in Celsius", rank=1))

        cooling = m.get("cooling_adequate")
        if cooling is not None:
            obs.append(Observable("cooling_adequate", bool(cooling), "",
                                  "Whether all components stay within thermal limits", rank=2))

        # Typical IGBT limit is 175°C = 448K
        T_limit = 448.0
        margin = T_limit - float(max_T)
        obs.append(Observable("margin_to_175C_limit", round(margin, 1), "K",
                              "Margin to typical IGBT junction limit (175°C)",
                              threshold=0.0, margin=round(margin, 1), rank=3))

        # Deep observables
        hs = _get_field(state, "heat_source")
        if hs is not None and hs.size > 0:
            total_pwr = _safe_max(np.sum(hs, axis=-1)) if hs.ndim >= 2 else float(np.sum(hs))
            obs.append(Observable("total_power_dissipated", round(float(total_pwr), 2), "W",
                                  "Total heat generation in the assembly", rank=4))
        node_names = m.get("node_names")
        if node_names is not None and T is not None:
            T_arr = np.asarray(T) if not isinstance(T, np.ndarray) else T
            if T_arr.ndim == 2 and T_arr.shape[1] == len(node_names):
                hot_idx = int(np.argmax(T_arr[-1]))
                obs.append(Observable("hotspot_component", node_names[hot_idx], "",
                                      "Component with highest temperature", rank=5))
            elif T_arr.ndim == 1 and len(T_arr) == len(node_names):
                hot_idx = int(np.argmax(T_arr))
                obs.append(Observable("hotspot_component", node_names[hot_idx], "",
                                      "Component with highest temperature", rank=5))
        # Thermal time constant estimate: time to reach 63% of (T_final - T_initial)
        if T is not None:
            T_arr = np.asarray(T) if not isinstance(T, np.ndarray) else T
            if T_arr.ndim == 2 and T_arr.shape[0] >= 3:
                T_node0 = T_arr[:, 0]
                T0 = T_node0[0]
                Tf = T_node0[-1]
                if abs(Tf - T0) > 0.01:
                    T_63 = T0 + 0.632 * (Tf - T0)
                    idx_63 = np.argmin(np.abs(T_node0 - T_63))
                    # Estimate time assuming uniform dt
                    sc = config.get("solve", config)
                    dt = float(sc.get("dt", 0.1))
                    tau = idx_63 * dt
                    obs.append(Observable("thermal_time_constant", round(tau, 2), "s",
                                          "Estimated thermal time constant (63% of final T)", rank=6))

        return obs


# -------------------------------------------------------------------
# 9. Battery Thermal
# -------------------------------------------------------------------

@register_observables
class BatteryThermalObservables(BaseObservableSet):
    module_name = "battery_thermal"
    domain = "thermal"

    def compute(self, state, config, native_result=None):
        obs = []
        m = state.metadata

        for key, name, unit, desc, rank in [
            ("max_cell_temperature", "peak_cell_temperature", "K", "Hottest cell temperature", 0),
            ("temperature_spread", "temperature_spread", "K", "Max - min cell temperature — indicates pack imbalance", 2),
            ("safety_score", "safety_score", "", "Safety score (0 = runaway, 1 = fully safe)", 3),
            ("cooling_power_W", "cooling_power_required", "W", "Average cooling power removed", 5),
        ]:
            v = m.get(key)
            if v is not None:
                obs.append(Observable(name, round(float(v), 2), unit, desc, rank=rank))

        runaway = m.get("runaway_risk")
        if runaway is not None:
            obs.append(Observable("runaway_risk", bool(runaway), "",
                                  "Whether any cell exceeded thermal runaway threshold",
                                  relevance="CRITICAL safety indicator", rank=1))

        # Margin to runaway (NMC: 403K)
        peak_T = m.get("max_cell_temperature", 298)
        T_runaway = 403.0
        margin = T_runaway - float(peak_T)
        obs.append(Observable("margin_to_runaway", round(margin, 1), "K",
                              "Temperature margin to NMC runaway onset (130°C / 403K)",
                              threshold=0.0, margin=round(margin, 1), rank=4))

        # Deep observables
        min_T = m.get("min_cell_temperature")
        if min_T is not None:
            obs.append(Observable("min_cell_temperature", round(float(min_T), 1), "K",
                                  "Coldest cell temperature", rank=6))

        return obs


# -------------------------------------------------------------------
# 10. Structural Analysis
# -------------------------------------------------------------------

@register_observables
class StructuralAnalysisObservables(BaseObservableSet):
    module_name = "structural_analysis"
    domain = "structures"

    def compute(self, state, config, native_result=None):
        obs = []
        m = state.metadata

        disp = _get_field(state, "displacement")
        stress = _get_field(state, "stress_von_mises")

        if stress is not None:
            peak_vm = _safe_max(np.abs(stress))
            obs.append(Observable("max_von_mises_stress", round(peak_vm, 0), "Pa",
                                  "Peak von Mises stress", rank=0))
            obs.append(Observable("max_von_mises_MPa", round(peak_vm / 1e6, 1), "MPa",
                                  "Peak von Mises stress in MPa", rank=1))

        if disp is not None:
            max_def = _safe_max(np.abs(disp))
            obs.append(Observable("max_deflection", max_def, "m",
                                  "Maximum beam deflection", rank=2))
            obs.append(Observable("max_deflection_mm", round(max_def * 1000, 3), "mm",
                                  "Maximum beam deflection in mm", rank=3))

        for key, name, desc, rank in [
            ("min_buckling_ms", "min_buckling_margin", "Minimum buckling margin of safety (>0 = safe)", 4),
            ("overall_ms", "overall_margin_of_safety", "Governing margin of safety", 5),
        ]:
            v = m.get(key)
            if v is not None:
                obs.append(Observable(name, round(float(v), 3), "", desc,
                                      threshold=0.0, margin=round(float(v), 3), rank=rank))

        safe = m.get("is_safe")
        if safe is not None:
            obs.append(Observable("is_safe", bool(safe), "",
                                  "Whether all margins of safety are positive", rank=6))

        # Deep: stress ratio (applied/yield for AL7075-T6 yield = 503 MPa)
        if stress is not None:
            yield_stress = 503e6  # AL7075-T6 typical yield [Pa]
            sc = config.get("solver", config)
            mat_name = sc.get("material_name", "AL7075-T6")
            ratio = peak_vm / yield_stress
            obs.append(Observable("stress_ratio", round(ratio, 4), "",
                                  f"Applied/yield stress ratio for {mat_name} (>1 = yield exceeded)",
                                  threshold=1.0, margin=round(1.0 - ratio, 4), rank=7))
        bc_count = m.get("buckling_check_count")
        if bc_count is not None:
            obs.append(Observable("buckling_checks_performed", int(bc_count), "",
                                  "Number of elements checked for buckling", rank=8))

        return obs


# -------------------------------------------------------------------
# 11. Structural Dynamics
# -------------------------------------------------------------------

@register_observables
class StructuralDynamicsObservables(BaseObservableSet):
    module_name = "structural_dynamics"
    domain = "structures"

    def compute(self, state, config, native_result=None):
        obs = []
        d = _get_field(state, "displacement")
        v = _get_field(state, "velocity")
        a = _get_field(state, "acceleration")

        if d is not None:
            peak_d = _safe_max(np.abs(d))
            obs.append(Observable("peak_displacement", peak_d, "m",
                                  "Maximum displacement over all DOFs and time", rank=0))

        if a is not None:
            peak_a = _safe_max(np.abs(a))
            rms_a = float(np.sqrt(_safe_mean(a**2)))
            obs.append(Observable("peak_acceleration", round(peak_a, 4), "m/s^2",
                                  "Peak acceleration — equipment qualification metric", rank=1))
            obs.append(Observable("rms_acceleration", round(rms_a, 4), "m/s^2",
                                  "RMS acceleration — fatigue-relevant metric", rank=2))
            if peak_a > 0 and rms_a > 0:
                crest = peak_a / rms_a
                obs.append(Observable("crest_factor", round(crest, 2), "",
                                      "Peak-to-RMS ratio — indicates impulsiveness", rank=3))

        # Natural frequencies from metadata
        m = state.metadata
        modes = m.get("natural_frequencies")
        if modes is not None:
            obs.append(Observable("natural_frequencies_Hz", modes, "Hz",
                                  "Natural frequencies from modal analysis", rank=4))

        # Deep: dominant frequency, response amplification
        mode_count = m.get("mode_count")
        if mode_count is not None:
            obs.append(Observable("mode_count", int(mode_count), "",
                                  "Number of modes computed in modal analysis", rank=5))
        if d is not None and d.ndim == 2 and d.shape[0] >= 2:
            # Static displacement estimate: peak force / peak stiffness
            sc = config.get("solver", config)
            k_diag = sc.get("stiffness_diag")
            amp = float(sc.get("force_amplitude", 100.0))
            if k_diag and len(k_diag) > 0:
                static_disp = amp / float(k_diag[0])
                dynamic_amp = peak_d / static_disp if static_disp > 0 else 0
                obs.append(Observable("dynamic_amplification_factor", round(dynamic_amp, 2), "",
                                      "Peak dynamic / static displacement — indicates resonance proximity", rank=6))

        return obs


# -------------------------------------------------------------------
# 12. Flight Mechanics
# -------------------------------------------------------------------

@register_observables
class FlightMechanicsObservables(BaseObservableSet):
    module_name = "flight_mechanics"
    domain = "dynamics"

    def compute(self, state, config, native_result=None):
        obs = []
        m = state.metadata

        for key, name, unit, desc, rank in [
            ("settling_time", "settling_time", "s", "Time to achieve 1° pointing accuracy", 0),
            ("max_angular_rate", "max_angular_rate", "rad/s", "Peak body angular rate during manoeuvre", 1),
            ("fuel_consumed", "fuel_consumed", "kg", "Total RCS propellant consumed", 2),
        ]:
            v = m.get(key)
            if v is not None:
                obs.append(Observable(name, round(float(v), 4), unit, desc, rank=rank))

        vel = _get_field(state, "velocity")
        if vel is not None and vel.size > 0:
            final_speed = float(np.sqrt(np.sum(vel[-1]**2))) if vel.ndim == 2 else float(np.abs(vel[-1]))
            obs.append(Observable("final_speed", round(final_speed, 2), "m/s",
                                  "Speed at end of simulation", rank=3))

        disp = _get_field(state, "displacement")
        if disp is not None and disp.ndim == 2 and disp.shape[1] >= 3:
            max_alt = _safe_max(disp[:, 2])
            obs.append(Observable("max_altitude", round(max_alt, 1), "m",
                                  "Peak altitude (z-component)", rank=4))

        # Deep observables
        settling = m.get("settling_time")
        omega_max = m.get("max_angular_rate")
        if omega_max is not None:
            rpm = float(omega_max) * 60.0 / (2.0 * np.pi)
            obs.append(Observable("max_spin_rate_rpm", round(rpm, 3), "rpm",
                                  "Peak angular rate in RPM", rank=5))
        sc = config.get("solver", config)
        Ixx = float(sc.get("Ixx", 100.0))
        if omega_max is not None:
            H = Ixx * float(omega_max)
            obs.append(Observable("angular_momentum_peak", round(H, 3), "N*m*s",
                                  "Peak angular momentum (Ixx * omega_max)", rank=6))

        return obs


# -------------------------------------------------------------------
# 13. Coupled Physics
# -------------------------------------------------------------------

@register_observables
class CoupledPhysicsObservables(BaseObservableSet):
    module_name = "coupled_physics"
    domain = "nuclear"

    def compute(self, state, config, native_result=None):
        obs = []
        m = state.metadata

        pr = m.get("power_ratio")
        if pr is not None:
            obs.append(Observable("power_ratio", round(float(pr), 3), "",
                                  "Peak power / initial power — excursion severity", rank=0))

        for key, name, unit, desc, rank in [
            ("peak_power_W", "peak_power", "W", "Peak reactor power during transient", 1),
            ("time_to_peak_s", "time_to_peak", "s", "Time from insertion to power peak", 2),
            ("peak_temperature_K", "peak_fuel_temperature", "K", "Peak fuel temperature during transient", 3),
            ("rho_doppler_pcm_final", "doppler_feedback_pcm", "pcm", "Doppler reactivity at end of transient", 5),
            ("rho_moderator_pcm_final", "moderator_feedback_pcm", "pcm", "Moderator reactivity at end", 6),
            ("rho_external_pcm_final", "external_reactivity_pcm", "pcm", "External (inserted) reactivity", 7),
        ]:
            v = m.get(key)
            if v is not None:
                obs.append(Observable(name, round(float(v), 4), unit, desc, rank=rank))

        conv = m.get("converged")
        if conv is not None:
            obs.append(Observable("feedback_arrested_excursion", bool(conv), "",
                                  "Whether feedback mechanisms arrested the power excursion",
                                  relevance="CRITICAL — False means runaway", rank=4))

        # Deep observables
        beta_eff = 0.0065  # standard delayed neutron fraction
        rho_ext = m.get("rho_external_pcm_final", 0)
        if rho_ext is not None:
            margin = (beta_eff * 1e5) - abs(float(rho_ext))
            obs.append(Observable("prompt_critical_margin_pcm", round(margin, 1), "pcm",
                                  f"Margin to prompt critical (beta_eff={beta_eff*1e5:.0f} pcm - |rho_ext|)",
                                  threshold=0.0, margin=round(margin, 1), rank=8))
        P_init = m.get("initial_power_W")
        P_final = m.get("final_power_W")
        if P_init is not None and P_final is not None:
            obs.append(Observable("final_power", round(float(P_final), 0), "W",
                                  "Reactor power at end of transient", rank=9))
            obs.append(Observable("initial_power", round(float(P_init), 0), "W",
                                  "Initial steady-state power", rank=10))
        # Energy released estimate
        P_hist = _get_field(state, "power_history")
        if P_hist is not None and P_hist.size >= 2:
            sc = config.get("solve", config)
            dt_solve = float(sc.get("dt", 0.01))
            energy_J = float(np.sum(P_hist)) * dt_solve
            energy_MJ = energy_J / 1e6
            obs.append(Observable("energy_released_MJ", round(energy_MJ, 3), "MJ",
                                  "Total energy released during transient (integral of P*dt)", rank=11))

        return obs


# -------------------------------------------------------------------
# 14. Neutronics
# -------------------------------------------------------------------

@register_observables
class NeutronicsObservables(BaseObservableSet):
    module_name = "neutronics"
    domain = "nuclear"

    def compute(self, state, config, native_result=None):
        obs = []
        m = state.metadata

        k = m.get("k_eff")
        if k is not None:
            kf = float(k)
            obs.append(Observable("k_effective", round(kf, 5), "",
                                  "Effective multiplication factor", rank=0))
            rho_pcm = (kf - 1.0) / kf * 1e5 if kf > 0 else 0
            obs.append(Observable("excess_reactivity_pcm", round(rho_pcm, 1), "pcm",
                                  "Excess reactivity — positive means supercritical", rank=1))

            subcrit_margin = 1.0 - kf
            obs.append(Observable("subcriticality_margin", round(subcrit_margin, 5), "",
                                  "Distance from criticality (positive = subcritical)",
                                  threshold=0.0, margin=round(subcrit_margin, 5), rank=2))

        pf = m.get("peaking_factor")
        if pf is not None:
            obs.append(Observable("power_peaking_factor", round(float(pf), 3), "",
                                  "Max/mean power density — hot channel factor", rank=3))

        # Deep observables
        flux_fast = _get_field(state, "neutron_flux")
        if flux_fast is not None and flux_fast.size > 0:
            peak_flux = _safe_max(flux_fast)
            mean_flux = _safe_mean(flux_fast)
            obs.append(Observable("peak_neutron_flux", peak_flux, "n/(m^2*s)",
                                  "Peak total neutron flux", rank=4))
            if mean_flux > 0:
                flux_ratio = peak_flux / mean_flux
                obs.append(Observable("flux_peaking", round(flux_ratio, 3), "",
                                      "Peak/mean flux ratio — spatial non-uniformity", rank=5))
        conv = m.get("converged")
        if conv is not None:
            obs.append(Observable("eigenvalue_converged", bool(conv), "",
                                  "Whether the power iteration converged", rank=6))
        core_L = m.get("core_length_m")
        if core_L is not None:
            obs.append(Observable("core_length", round(float(core_L), 3), "m",
                                  "Active core length", rank=7))

        return obs


# -------------------------------------------------------------------
# 15. Geospatial
# -------------------------------------------------------------------

@register_observables
class GeospatialObservables(BaseObservableSet):
    module_name = "geospatial"
    domain = "logistics"

    def compute(self, state, config, native_result=None):
        obs = []
        m = state.metadata

        for key, name, unit, desc, rank in [
            ("coverage_fraction", "coverage_fraction", "", "Fraction of demand population covered", 0),
            ("covered_population_millions", "covered_population", "M people", "Population served within travel time limit", 1),
            ("num_facilities", "facilities_opened", "", "Number of facilities selected", 3),
            ("objective_value", "total_weighted_travel_time", "h", "Optimization objective value", 5),
        ]:
            v = m.get(key)
            if v is not None:
                obs.append(Observable(name, round(float(v), 4) if isinstance(v, float) else v, unit, desc, rank=rank))

        uncov = m.get("uncovered_regions")
        if uncov is not None:
            obs.append(Observable("uncovered_regions", uncov, "",
                                  "Demand centres not served — gap analysis", rank=2))

        # Coverage vs target
        sc = config.get("solver", config)
        target = float(sc.get("target_coverage", 0.95))
        cov = m.get("coverage_fraction", 0)
        if cov is not None:
            meets = float(cov) >= target
            obs.append(Observable("meets_coverage_target", meets, "",
                                  f"Whether coverage >= {target*100:.0f}% target",
                                  threshold=target, margin=round(float(cov) - target, 4), rank=4))

        # Deep observables
        tt = _get_field(state, "travel_time")
        if tt is not None and tt.size > 0:
            avg_tt = _safe_mean(tt[tt > 0]) if np.any(tt > 0) else 0
            max_tt = _safe_max(tt)
            obs.append(Observable("average_travel_time", round(avg_tt, 2), "h",
                                  "Mean travel time across all served demand points", rank=6))
            obs.append(Observable("worst_case_travel_time", round(max_tt, 2), "h",
                                  "Longest travel time to any served demand point", rank=7))
        total_pop = m.get("total_population_millions")
        if total_pop is not None:
            obs.append(Observable("total_demand_population", round(float(total_pop), 1), "M people",
                                  "Total population in the demand region", rank=8))

        return obs


# -------------------------------------------------------------------
# 16. Field-Aware Routing
# -------------------------------------------------------------------

@register_observables
class FieldAwareRoutingObservables(BaseObservableSet):
    module_name = "field_aware_routing"
    domain = "electromagnetism"

    def compute(self, state, config, native_result=None):
        obs = []
        V = _get_field(state, "electric_potential")
        E = _get_field(state, "electric_field_magnitude")
        m = state.metadata

        if E is not None:
            peak_E = _safe_max(E)
            mean_E = _safe_mean(E)
            obs.append(Observable("peak_field_strength", round(peak_E, 1), "V/m",
                                  "Maximum electric field — EMI risk zone", rank=0))
            obs.append(Observable("mean_field_strength", round(mean_E, 1), "V/m",
                                  "Average field strength", rank=2))

            # Low-field corridor (below 30% of peak)
            if peak_E > 0:
                safe_frac = float(np.mean(E < 0.3 * peak_E))
                obs.append(Observable("low_field_corridor_fraction", round(safe_frac, 4), "",
                                      "Fraction of domain with field < 30% of peak — routable area",
                                      relevance="Higher fraction means more routing options", rank=1))

        if V is not None:
            drop = _safe_max(V) - _safe_min(V)
            obs.append(Observable("potential_drop", round(drop, 2), "V",
                                  "Voltage drop across domain", rank=3))

        te = m.get("total_energy")
        if te is not None:
            obs.append(Observable("stored_energy", float(te), "J",
                                  "Electrostatic energy stored in the field", rank=4))

        # Deep observables
        Ex = _get_field(state, "electric_field_x")
        Ey = _get_field(state, "electric_field_y")
        if Ex is not None and Ey is not None:
            # Field gradient (rate of change)
            if Ex.ndim == 2 and Ex.shape[0] > 2 and Ex.shape[1] > 2:
                sc = config.get("solver", config)
                nx_r = int(sc.get("nx", 64))
                x_max = float(sc.get("x_max", 1.0))
                dx_r = x_max / max(nx_r - 1, 1)
                dEdx = np.abs(np.diff(E, axis=1)) / dx_r if E is not None and E.ndim == 2 else None
                if dEdx is not None:
                    obs.append(Observable("max_field_gradient", round(_safe_max(dEdx), 1), "V/m^2",
                                          "Maximum spatial gradient of E-field — indicates coupling hotspot", rank=5))
        conv = m.get("converged")
        if conv is not None:
            obs.append(Observable("solver_converged", bool(conv), "",
                                  "Whether the linear solve converged", rank=6))
        res_norm = m.get("residual_norm")
        if res_norm is not None:
            obs.append(Observable("residual_norm", float(res_norm), "",
                                  "Final residual norm of the linear solve", rank=7))

        return obs
