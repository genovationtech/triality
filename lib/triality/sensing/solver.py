"""Multi-Sensor Performance Solver

Computes detection range, signal-to-noise ratio, resolution, and
coverage maps for radar, sonar, and lidar systems over a 2-D
surveillance area, combining the physics models in the sensing
sub-modules (propagation, targets, noise, detection).

The solver evaluates the radar / sonar equation at every point on
a spatial grid, producing SNR and detection-probability maps that
are useful for sensor placement, coverage analysis, and system
trade-off studies.

Typical workflow:
    1. Build a SensorPerformanceConfig (sensor specs, grid, targets).
    2. Instantiate SensorPerformanceSolver(config).
    3. Call solver.solve() -> SensorPerformanceResult.

(c) 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum

from triality.core.fields import PhysicsState, PhysicsField, FidelityTier, CouplingMaturity

from .signals import (
    WaveParameters,
    SignalCharacteristics,
    BeamCharacteristics,
    C_LIGHT,
)
from .detection import DetectionProbability
from .noise import ThermalNoise
from .radar import RadarSystem, RadarAnalyzer
from .sonar import SonarSystem, SonarAnalyzer
from .tradeoffs import RadarBudget


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class SensorType(Enum):
    RADAR = "radar"
    SONAR = "sonar"


@dataclass
class SensorSpec:
    """Generic sensor specification for the solver.

    Wraps either a RadarSystem or SonarSystem with a location.
    """
    name: str
    sensor_type: SensorType
    location: Tuple[float, float]       # (x_km, y_km) position
    radar: Optional[RadarSystem] = None
    sonar: Optional[SonarSystem] = None


@dataclass
class TargetSpec:
    """Target specification for detection analysis.

    Attributes:
        rcs_m2:            Radar cross-section [m^2] (radar mode).
        target_strength_db: Target strength [dB re 1 m^2] (sonar mode).
    """
    rcs_m2: float = 1.0
    target_strength_db: float = 10.0


@dataclass
class SensorPerformanceConfig:
    """Configuration for the multi-sensor performance solver.

    Attributes:
        sensors:        List of sensor specifications with locations.
        target:         Target characteristics.
        grid_x_km:      x-axis bounds (min, max) in km.
        grid_y_km:      y-axis bounds (min, max) in km.
        grid_nx:        Number of x grid points.
        grid_ny:        Number of y grid points.
        pfa:            Probability of false alarm for Pd calculation.
        weather:        Weather/sea state string ('clear', 'rain', etc.).
        water_temp_c:   Water temperature for sonar propagation [C].
    """
    sensors: List[SensorSpec] = field(default_factory=list)
    target: TargetSpec = field(default_factory=TargetSpec)
    grid_x_km: Tuple[float, float] = (-50.0, 50.0)
    grid_y_km: Tuple[float, float] = (-50.0, 50.0)
    grid_nx: int = 64
    grid_ny: int = 64
    pfa: float = 1e-6
    weather: str = "clear"
    water_temp_c: float = 15.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class SensorPerformanceResult:
    """Result of the multi-sensor performance solver.

    Attributes:
        grid_x_km:          1-D x coordinate array [km].
        grid_y_km:          1-D y coordinate array [km].
        snr_maps:           Dict sensor_name -> 2-D SNR array [dB].
        pd_maps:            Dict sensor_name -> 2-D P_d array (0-1).
        combined_pd:        2-D array of combined detection probability
                            from all sensors (1 - product(1 - pd_i)).
        max_detection_range_km: Dict sensor_name -> max range [km].
        resolution:         Dict sensor_name -> {range_res_m, cross_range_res_m}.
        coverage_fraction:  Fraction of grid area where combined Pd > 0.5.
        sensor_count:       Number of sensors evaluated.
        config:             Solver configuration.
    """
    grid_x_km: np.ndarray
    grid_y_km: np.ndarray
    snr_maps: Dict[str, np.ndarray]
    pd_maps: Dict[str, np.ndarray]
    combined_pd: np.ndarray
    max_detection_range_km: Dict[str, float]
    resolution: Dict[str, Dict[str, float]]
    coverage_fraction: float
    sensor_count: int
    config: SensorPerformanceConfig


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
@dataclass
class PropagationCoverageResult:
    """Result of full 2-D propagation and coverage analysis.

    Attributes:
        grid_x_km: 1-D x coordinate array [km].
        grid_y_km: 1-D y coordinate array [km].
        propagation_loss_db: 2-D one-way propagation loss map [dB].
        snr_map_db: 2-D SNR map [dB] after clutter subtraction.
        pd_map: 2-D detection probability map.
        coverage_fraction: Fraction of grid where Pd > 0.5.
        terrain_height_m: 2-D terrain height profile [m] (generated or supplied).
        refractivity_profile: 1-D modified refractivity M(z) [M-units].
        altitude_axis_m: Altitude axis for refractivity [m].
        prop_factor_db: 2-D propagation factor F^2 [dB] (range x height).
        range_axis_km: Range axis for propagation factor [km].
        height_axis_m: Height axis for propagation factor [m].
        multipath_lobes: Indices of multipath lobe peaks in propagation factor.
        diffraction_loss_db: 2-D diffraction loss contribution [dB].
        clutter_map_db: 2-D surface clutter power map [dB].
        cfar_threshold_map_db: 2-D spatial CFAR threshold [dB].
        detection_map: Boolean 2-D detection map.
        duct_height_m: Estimated duct height from refractivity profile [m].
    """
    grid_x_km: np.ndarray
    grid_y_km: np.ndarray
    propagation_loss_db: np.ndarray
    snr_map_db: np.ndarray
    pd_map: np.ndarray
    coverage_fraction: float
    terrain_height_m: np.ndarray
    refractivity_profile: np.ndarray
    altitude_axis_m: np.ndarray
    prop_factor_db: np.ndarray
    range_axis_km: np.ndarray
    height_axis_m: np.ndarray
    multipath_lobes: np.ndarray
    diffraction_loss_db: np.ndarray
    clutter_map_db: np.ndarray
    cfar_threshold_map_db: np.ndarray
    detection_map: np.ndarray
    duct_height_m: float


class SensorPerformanceSolver:
    """Multi-sensor performance solver.

    For each sensor, evaluates the link budget (radar equation or sonar
    equation) at every grid point to produce SNR and detection-probability
    maps.  Multiple sensors are fused via the independent-detection
    assumption: Pd_combined = 1 - prod(1 - Pd_i).

    Parameters
    ----------
    config : SensorPerformanceConfig
    """

    fidelity_tier = FidelityTier.HEURISTIC
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(self, config: SensorPerformanceConfig):
        self.config = config

    def solve(self) -> SensorPerformanceResult:
        """Run the sensor performance analysis.

        Returns
        -------
        SensorPerformanceResult
        """
        cfg = self.config

        # Build spatial grid
        gx = np.linspace(cfg.grid_x_km[0], cfg.grid_x_km[1], cfg.grid_nx)
        gy = np.linspace(cfg.grid_y_km[0], cfg.grid_y_km[1], cfg.grid_ny)
        XX, YY = np.meshgrid(gx, gy, indexing='ij')  # shape (nx, ny)

        snr_maps: Dict[str, np.ndarray] = {}
        pd_maps: Dict[str, np.ndarray] = {}
        max_ranges: Dict[str, float] = {}
        resolutions: Dict[str, Dict[str, float]] = {}

        for spec in cfg.sensors:
            if spec.sensor_type == SensorType.RADAR and spec.radar is not None:
                snr, pd, max_r, res = self._solve_radar(
                    spec, XX, YY, cfg.target, cfg.pfa
                )
            elif spec.sensor_type == SensorType.SONAR and spec.sonar is not None:
                snr, pd, max_r, res = self._solve_sonar(
                    spec, XX, YY, cfg.target, cfg.pfa, cfg.water_temp_c
                )
            else:
                continue

            snr_maps[spec.name] = snr
            pd_maps[spec.name] = pd
            max_ranges[spec.name] = max_r
            resolutions[spec.name] = res

        # Combined detection probability
        combined_pd = np.zeros_like(XX)
        if pd_maps:
            miss = np.ones_like(XX)
            for pd_arr in pd_maps.values():
                miss *= (1.0 - pd_arr)
            combined_pd = 1.0 - miss

        # Coverage fraction (Pd > 0.5)
        coverage = float(np.mean(combined_pd > 0.5))

        return SensorPerformanceResult(
            grid_x_km=gx,
            grid_y_km=gy,
            snr_maps=snr_maps,
            pd_maps=pd_maps,
            combined_pd=combined_pd,
            max_detection_range_km=max_ranges,
            resolution=resolutions,
            coverage_fraction=coverage,
            sensor_count=len(cfg.sensors),
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Per-sensor solvers
    # ------------------------------------------------------------------
    def _solve_radar(self, spec: SensorSpec,
                     XX: np.ndarray, YY: np.ndarray,
                     tgt: TargetSpec, pfa: float
                     ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
        """Radar SNR and Pd over the grid."""
        sys = spec.radar
        sx, sy = spec.location

        # Derived parameters
        freq_hz = sys.frequency_ghz * 1e9
        wavelength = C_LIGHT / freq_hz
        bandwidth_hz = sys.bandwidth_mhz * 1e6
        aperture_area = np.pi * (sys.aperture_diameter_m / 2.0) ** 2
        antenna_gain_linear = 4.0 * np.pi * aperture_area * 0.55 / (wavelength ** 2)
        antenna_gain_db = 10.0 * np.log10(antenna_gain_linear)

        # Noise power
        k_B = 1.380649e-23
        T0 = 290.0
        noise_figure_lin = 10 ** (sys.noise_figure_db / 10.0)
        N = k_B * T0 * bandwidth_hz * noise_figure_lin
        losses_lin = 10 ** (sys.losses_db / 10.0)

        # Range grid [m]
        R = np.sqrt((XX - sx) ** 2 + (YY - sy) ** 2) * 1000.0  # km -> m
        R = np.maximum(R, 1.0)  # avoid zero

        # Radar equation: SNR = (Pt * G^2 * lambda^2 * sigma) / ((4pi)^3 * R^4 * N * L)
        numerator = (sys.power_w * antenna_gain_linear ** 2
                     * wavelength ** 2 * tgt.rcs_m2)
        denominator = (4.0 * np.pi) ** 3 * R ** 4 * N * losses_lin
        snr_linear = numerator / denominator

        # Pulse integration gain (non-coherent, approximate sqrt(N))
        snr_linear *= np.sqrt(sys.n_pulses_integrated)

        snr_db = 10.0 * np.log10(np.maximum(snr_linear, 1e-30))

        # Detection probability
        pd = np.vectorize(
            lambda s: DetectionProbability.pd_single_pulse(s, pfa)
        )(snr_db)

        # Max detection range (SNR = required_snr at Pd=0.5)
        req_snr_db = 13.0  # ~Pd=0.5 at Pfa=1e-6
        req_snr_lin = 10 ** (req_snr_db / 10.0)
        R_max_m = (numerator / ((4.0 * np.pi) ** 3 * N * losses_lin * req_snr_lin)) ** 0.25
        R_max_km = R_max_m / 1000.0

        # Resolution
        range_res = C_LIGHT / (2.0 * bandwidth_hz)
        beamwidth_rad = 1.22 * wavelength / sys.aperture_diameter_m
        cross_range_1km = beamwidth_rad * 1000.0  # m at 1 km

        return (
            snr_db,
            pd,
            float(R_max_km),
            {'range_resolution_m': range_res,
             'beamwidth_deg': float(np.degrees(beamwidth_rad)),
             'cross_range_m_at_1km': cross_range_1km},
        )

    def export_state(self, result: SensorPerformanceResult) -> PhysicsState:
        """Export detection-performance outputs with canonical sensing labels."""
        state = PhysicsState(solver_name="sensing")
        state.set_field("detection_probability", result.combined_pd, "1")
        state.metadata["coverage_fraction"] = result.coverage_fraction
        state.metadata["sensor_count"] = result.sensor_count
        state.metadata["max_detection_range_km"] = result.max_detection_range_km
        state.metadata["resolution"] = result.resolution
        state.metadata["snr_maps_db"] = result.snr_maps

        # Warn if grid is too small to show detection boundaries
        cfg = result.config
        grid_max_range_km = np.sqrt(
            max(abs(cfg.grid_x_km[0]), abs(cfg.grid_x_km[1])) ** 2
            + max(abs(cfg.grid_y_km[0]), abs(cfg.grid_y_km[1])) ** 2
        )
        warnings = []
        for name, r_max in result.max_detection_range_km.items():
            if grid_max_range_km <= r_max * 1.1:
                warnings.append(
                    f"Grid corner ({grid_max_range_km:.1f} km) does not extend beyond "
                    f"max detection range of {name} ({r_max:.1f} km). "
                    f"Blind zones are NOT visible. Increase grid to at least "
                    f"±{r_max * 1.5:.0f} km."
                )
        if warnings:
            state.metadata["warnings"] = warnings

        return state

    def _solve_sonar(self, spec: SensorSpec,
                     XX: np.ndarray, YY: np.ndarray,
                     tgt: TargetSpec, pfa: float,
                     water_temp_c: float
                     ) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
        """Sonar SNR and Pd over the grid (active sonar)."""
        sys = spec.sonar
        sx, sy = spec.location

        freq_khz = sys.frequency_khz
        SL = sys.source_level_db
        TS = tgt.target_strength_db
        DI = 10.0  # typical directivity index [dB]

        # Range grid [km]
        R_km = np.sqrt((XX - sx) ** 2 + (YY - sy) ** 2)
        R_km = np.maximum(R_km, 0.001)

        # Transmission loss: spherical spreading + absorption
        # TL = 20*log10(R) + alpha*R  (one-way)
        # Absorption (Thorp approximation): alpha [dB/km]
        f = freq_khz
        alpha_db_km = (0.11 * f ** 2 / (1 + f ** 2)
                       + 44.0 * f ** 2 / (4100 + f ** 2)
                       + 2.75e-4 * f ** 2 + 0.003)

        R_m = R_km * 1000.0
        TL_oneway = 20.0 * np.log10(np.maximum(R_m, 1.0)) + alpha_db_km * R_km

        # Ambient noise (simplified sea-state 3)
        NL = 50.0 + 20.0 * np.log10(max(freq_khz, 0.01))  # rough model

        # Active sonar equation: SE = SL - 2*TL + TS - (NL - DI)
        SE = SL - 2.0 * TL_oneway + TS - (NL - DI)

        snr_db = SE  # signal excess ~= SNR above detection threshold

        # Detection probability (map SE to Pd)
        pd = 1.0 / (1.0 + np.exp(-0.5 * SE))  # logistic approximation

        # Max detection range: solve SE = 0
        # SL + TS + DI - NL = 2*TL  ->  TL_max = (SL+TS+DI-NL)/2
        TL_max = (SL + TS + DI - NL) / 2.0
        # Approximate inversion (ignore absorption for initial guess)
        R_max_m = 10.0 ** (TL_max / 20.0)
        R_max_km = float(R_max_m / 1000.0)

        # Resolution
        c_water = 1500.0
        range_res = c_water / (2.0 * sys.bandwidth_khz * 1000.0)
        wl = c_water / (freq_khz * 1000.0)
        beamwidth_rad = 1.22 * wl / sys.array_size_m
        cross_range_1km = beamwidth_rad * 1000.0

        return (
            snr_db,
            pd,
            R_max_km,
            {'range_resolution_m': range_res,
             'beamwidth_deg': float(np.degrees(beamwidth_rad)),
             'cross_range_m_at_1km': cross_range_1km},
        )

    # ------------------------------------------------------------------
    # Level 3: Full 2-D propagation coverage model (existing)
    # ------------------------------------------------------------------
    def solve_propagation_coverage(
        self,
        sensor_index: int = 0,
        frequency_ghz: float = 10.0,
        power_w: float = 1000.0,
        antenna_gain_db: float = 35.0,
        noise_figure_db: float = 5.0,
        bandwidth_mhz: float = 10.0,
        losses_db: float = 5.0,
        rcs_m2: float = 1.0,
        antenna_height_m: float = 15.0,
        target_height_m: float = 10.0,
        terrain_type: str = "hilly",
        terrain_max_height_m: float = 200.0,
        terrain_roughness: float = 0.3,
        surface_refractivity: float = 313.0,
        duct_height_m: float = 0.0,
        duct_strength_m_units: float = 0.0,
        n_pe_height_cells: int = 256,
        n_pe_range_steps: int = 200,
        pe_max_height_m: float = 2000.0,
        clutter_sigma0_db: float = -20.0,
        clutter_patch_size_m: float = 100.0,
        cfar_guard: int = 3,
        cfar_train: int = 10,
        cfar_pfa: float = 1e-6,
        pfa: float = 1e-6,
    ) -> PropagationCoverageResult:
        """Full 2-D propagation model with PE method, multipath, diffraction, and spatial CFAR.

        Implements the split-step parabolic equation (PE) method to propagate
        the radar field through a 2-D range-height plane with a realistic
        refractivity profile (including optional surface ducting).  The
        propagation factor is then projected onto the 2-D surveillance grid
        to produce path-loss, SNR, and detection-probability maps.  Surface
        clutter is modelled from backscatter coefficient sigma-0, and a
        spatial 2-D CFAR detector determines clutter-limited detections.

        Parameters
        ----------
        sensor_index : int
            Index into config.sensors to use (for location).
        frequency_ghz : float
            Operating frequency [GHz].
        power_w : float
            Transmit power [W].
        antenna_gain_db : float
            Antenna gain [dB].
        noise_figure_db : float
            Receiver noise figure [dB].
        bandwidth_mhz : float
            Receiver bandwidth [MHz].
        losses_db : float
            System losses [dB].
        rcs_m2 : float
            Target radar cross section [m^2].
        antenna_height_m : float
            Antenna height above ground [m].
        target_height_m : float
            Target height above ground [m].
        terrain_type : str
            'flat', 'hilly', or 'mountainous' for procedural terrain.
        terrain_max_height_m : float
            Maximum terrain height [m].
        terrain_roughness : float
            Terrain roughness parameter (0-1).
        surface_refractivity : float
            Surface refractivity N_s [N-units] (typically 280-400).
        duct_height_m : float
            Height of evaporation/surface duct [m]. 0 = no duct.
        duct_strength_m_units : float
            Duct strength (delta-M in M-units). 0 = no duct.
        n_pe_height_cells : int
            Number of height grid cells for PE solver.
        n_pe_range_steps : int
            Number of range steps for PE marching.
        pe_max_height_m : float
            Maximum height for PE computation [m].
        clutter_sigma0_db : float
            Surface backscatter coefficient [dB m^2/m^2].
        clutter_patch_size_m : float
            Clutter cell resolution [m].
        cfar_guard : int
            CFAR guard cells.
        cfar_train : int
            CFAR training cells.
        cfar_pfa : float
            CFAR false alarm probability.
        pfa : float
            Probability of false alarm for Pd calculation.

        Returns
        -------
        PropagationCoverageResult
        """
        cfg = self.config
        freq_hz = frequency_ghz * 1e9
        wavelength = C_LIGHT / freq_hz
        k0 = 2.0 * np.pi / wavelength

        # Sensor location
        if cfg.sensors and sensor_index < len(cfg.sensors):
            sx, sy = cfg.sensors[sensor_index].location
        else:
            sx, sy = 0.0, 0.0

        # Build 2-D surveillance grid
        gx = np.linspace(cfg.grid_x_km[0], cfg.grid_x_km[1], cfg.grid_nx)
        gy = np.linspace(cfg.grid_y_km[0], cfg.grid_y_km[1], cfg.grid_ny)
        XX, YY = np.meshgrid(gx, gy, indexing='ij')
        R_km = np.sqrt((XX - sx) ** 2 + (YY - sy) ** 2)
        R_km = np.maximum(R_km, 0.001)
        R_m = R_km * 1000.0

        # ------------------------------------------------------------------
        # 1. Refractivity profile M(z) with optional ducting
        # ------------------------------------------------------------------
        dz_pe = pe_max_height_m / n_pe_height_cells
        height_axis = np.arange(n_pe_height_cells) * dz_pe

        # Standard atmosphere: M(z) = N_s + 0.157 * z (z in metres)
        M_profile = surface_refractivity + 0.157 * height_axis

        # Surface duct (evaporation duct model)
        detected_duct_height = 0.0
        if duct_height_m > 0 and duct_strength_m_units > 0:
            # Log-linear duct model
            z_d = duct_height_m
            delta_M = duct_strength_m_units
            for iz in range(n_pe_height_cells):
                z = height_axis[iz]
                if z <= z_d:
                    if z > 0:
                        M_profile[iz] = surface_refractivity - delta_M * (
                            1.0 - np.log(z / z_d + 1e-10) / np.log(0.001)
                        )
                    else:
                        M_profile[iz] = surface_refractivity + delta_M
            detected_duct_height = z_d

            # Find duct from M-profile (minimum in M)
            m_min_idx = np.argmin(M_profile[:max(1, int(500 / dz_pe))])
            if m_min_idx > 0:
                detected_duct_height = height_axis[m_min_idx]

        # Modified refractivity -> refractive index profile
        # n(z) = 1 + M(z) * 1e-6
        n_profile = 1.0 + M_profile * 1e-6

        # ------------------------------------------------------------------
        # 2. Terrain generation (procedural)
        # ------------------------------------------------------------------
        max_range_km = float(np.max(R_km))
        dr_pe = max_range_km * 1000.0 / n_pe_range_steps
        range_axis_km = np.linspace(0, max_range_km, n_pe_range_steps)
        range_axis_m = range_axis_km * 1000.0

        rng = np.random.default_rng(42)
        if terrain_type == "flat":
            terrain_1d = np.zeros(n_pe_range_steps)
        else:
            # Sum of sinusoids terrain model
            n_harmonics = 8
            terrain_1d = np.zeros(n_pe_range_steps)
            for h in range(1, n_harmonics + 1):
                amp = terrain_max_height_m / (h ** (1.0 + terrain_roughness))
                phase = rng.uniform(0, 2 * np.pi)
                freq_t = h * 2 * np.pi / (max_range_km * 1000.0) * (0.5 + rng.uniform())
                terrain_1d += amp * np.sin(freq_t * range_axis_m + phase)
            terrain_1d = np.maximum(terrain_1d, 0)

        # 2-D terrain: radially project from sensor with azimuthal variation
        terrain_2d = np.zeros((cfg.grid_nx, cfg.grid_ny))
        for ix in range(cfg.grid_nx):
            for iy in range(cfg.grid_ny):
                r = R_km[ix, iy]
                ridx = min(int(r / max_range_km * (n_pe_range_steps - 1)),
                           n_pe_range_steps - 1)
                az_var = 1.0 + 0.2 * np.sin(3 * np.arctan2(YY[ix, iy] - sy, XX[ix, iy] - sx))
                terrain_2d[ix, iy] = terrain_1d[ridx] * az_var

        # ------------------------------------------------------------------
        # 3. Split-step parabolic equation (PE) propagation
        # ------------------------------------------------------------------
        # PE field: u(z) at each range step, complex amplitude
        # Split-step Fourier method for narrow-angle PE:
        #   u(r+dr, z) = exp(j*k0*dr*(n^2-1)/2) * IFFT{ exp(-j*p^2*dr/(2*k0)) * FFT{u} }
        # where p is the spectral variable conjugate to z

        field = np.zeros(n_pe_height_cells, dtype=complex)
        # Initial field: Gaussian beam at antenna height
        ant_idx = min(int(antenna_height_m / dz_pe), n_pe_height_cells - 1)
        beam_width_cells = max(3, int(5.0 * wavelength / dz_pe))
        z_indices = np.arange(n_pe_height_cells)
        field = np.exp(-((z_indices - ant_idx) ** 2) / (2.0 * beam_width_cells ** 2)).astype(complex)
        field /= np.sqrt(np.sum(np.abs(field) ** 2) + 1e-30)

        # Spectral variable
        p = np.fft.fftfreq(n_pe_height_cells, d=dz_pe) * 2 * np.pi
        phase_screen_spectral = np.exp(-1j * p ** 2 * dr_pe / (2.0 * k0))

        # Propagation factor storage (range x height)
        prop_factor = np.zeros((n_pe_range_steps, n_pe_height_cells))
        prop_factor[0, :] = np.abs(field) ** 2

        for ri in range(1, n_pe_range_steps):
            # Refractivity phase screen (z-dependent)
            terrain_h = terrain_1d[ri]
            # Shift effective n_profile for terrain
            n_eff = np.ones(n_pe_height_cells)
            for iz in range(n_pe_height_cells):
                z_eff = height_axis[iz] + terrain_h
                iz_ref = min(int(z_eff / dz_pe), n_pe_height_cells - 1)
                n_eff[iz] = n_profile[iz_ref]

            # Ground boundary condition (image method): zero field below terrain
            ground_idx = max(0, int(terrain_h / dz_pe))
            field[:ground_idx] = 0.0

            # Phase screen: refractivity effect
            refr_phase = np.exp(1j * k0 * dr_pe * (n_eff ** 2 - 1.0) / 2.0)

            # Split-step: spectral propagation then phase screen
            field_spec = np.fft.fft(field)
            field_spec *= phase_screen_spectral
            field = np.fft.ifft(field_spec)
            field *= refr_phase

            # Apply absorbing layer at top of grid (prevent reflections)
            absorb_start = int(0.85 * n_pe_height_cells)
            for iz in range(absorb_start, n_pe_height_cells):
                taper = (iz - absorb_start) / (n_pe_height_cells - absorb_start)
                field[iz] *= np.exp(-3.0 * taper ** 2)

            # Cylindrical spreading compensation
            range_m = range_axis_m[ri]
            if range_m > 0:
                prop_factor[ri, :] = np.abs(field) ** 2 / (range_m + 1.0)
            else:
                prop_factor[ri, :] = np.abs(field) ** 2

        # Propagation factor in dB
        pf_max = np.max(prop_factor)
        if pf_max > 0:
            prop_factor_norm = prop_factor / pf_max
        else:
            prop_factor_norm = prop_factor
        prop_factor_db = 10.0 * np.log10(np.maximum(prop_factor_norm, 1e-30))

        # ------------------------------------------------------------------
        # 4. Multipath lobe detection
        # ------------------------------------------------------------------
        tgt_h_idx = min(int(target_height_m / dz_pe), n_pe_height_cells - 1)
        pf_at_target_h = prop_factor_db[:, tgt_h_idx]
        # Find local maxima
        multipath_lobes = []
        for ri in range(1, n_pe_range_steps - 1):
            if (pf_at_target_h[ri] > pf_at_target_h[ri - 1] and
                    pf_at_target_h[ri] > pf_at_target_h[ri + 1]):
                multipath_lobes.append(ri)
        multipath_lobes = np.array(multipath_lobes, dtype=int)

        # ------------------------------------------------------------------
        # 5. Diffraction loss computation (knife-edge model)
        # ------------------------------------------------------------------
        diffraction_loss_2d = np.zeros((cfg.grid_nx, cfg.grid_ny))
        for ix in range(cfg.grid_nx):
            for iy in range(cfg.grid_ny):
                r_total = R_m[ix, iy]
                if r_total < 100.0:
                    continue
                # Check terrain obstruction along radial path
                n_check = min(20, n_pe_range_steps)
                max_fresnel_param = -10.0
                for ic in range(1, n_check):
                    frac = ic / n_check
                    r1 = frac * r_total
                    r2 = (1.0 - frac) * r_total
                    ridx = min(int(frac * R_km[ix, iy] / max_range_km * (n_pe_range_steps - 1)),
                               n_pe_range_steps - 1)
                    h_terrain = terrain_2d[ix, iy] * frac
                    h_los = antenna_height_m + frac * (target_height_m - antenna_height_m)
                    h_excess = h_terrain - h_los
                    if h_excess > 0 and r1 > 0 and r2 > 0:
                        # Fresnel-Kirchhoff parameter
                        nu_fk = h_excess * np.sqrt(
                            2.0 / (wavelength * r1 * r2 / (r1 + r2))
                        )
                        max_fresnel_param = max(max_fresnel_param, nu_fk)
                if max_fresnel_param > -0.7:
                    # Lee's approximation for knife-edge diffraction loss
                    nu = max_fresnel_param
                    if nu > 1.0:
                        diff_loss = 20.0 * np.log10(nu) + 13.0
                    elif nu > 0:
                        diff_loss = 6.0 + 9.0 * nu + 1.27 * nu ** 2
                    else:
                        diff_loss = 0.0
                    diffraction_loss_2d[ix, iy] = diff_loss

        # ------------------------------------------------------------------
        # 6. 2-D propagation loss and SNR map
        # ------------------------------------------------------------------
        # Free-space path loss
        fspl_db = 20.0 * np.log10(np.maximum(R_m, 1.0)) + \
                  20.0 * np.log10(freq_hz) - 147.55

        # Propagation factor at target height: interpolate from PE results
        pf_target_h = np.zeros((cfg.grid_nx, cfg.grid_ny))
        for ix in range(cfg.grid_nx):
            for iy in range(cfg.grid_ny):
                r_idx = min(
                    int(R_km[ix, iy] / max_range_km * (n_pe_range_steps - 1)),
                    n_pe_range_steps - 1,
                )
                pf_target_h[ix, iy] = prop_factor_db[r_idx, tgt_h_idx]

        # Total one-way propagation loss
        prop_loss_1way = fspl_db + diffraction_loss_2d - pf_target_h
        propagation_loss_db = prop_loss_1way  # one-way

        # Radar equation with propagation model
        # SNR = Pt + 2*Gt + sigma_dBsm - 2*L_prop - kTB - NF - L_sys
        k_B = 1.380649e-23
        T0 = 290.0
        bandwidth_hz = bandwidth_mhz * 1e6
        noise_power_dbw = 10.0 * np.log10(k_B * T0 * bandwidth_hz)
        sigma_dbsm = 10.0 * np.log10(max(rcs_m2, 1e-30))

        snr_map_db = (10.0 * np.log10(power_w) + 2.0 * antenna_gain_db + sigma_dbsm
                      - 2.0 * prop_loss_1way - noise_power_dbw - noise_figure_db - losses_db)

        # ------------------------------------------------------------------
        # 7. Surface clutter map
        # ------------------------------------------------------------------
        sigma0_lin = 10.0 ** (clutter_sigma0_db / 10.0)
        n_clutter_cells = R_m * clutter_patch_size_m  # clutter area approximation
        clutter_rcs = sigma0_lin * n_clutter_cells
        clutter_rcs_dbsm = 10.0 * np.log10(np.maximum(clutter_rcs, 1e-30))

        clutter_snr_db = (10.0 * np.log10(power_w) + 2.0 * antenna_gain_db
                          + clutter_rcs_dbsm - 2.0 * prop_loss_1way
                          - noise_power_dbw - noise_figure_db - losses_db)
        clutter_map_db = clutter_snr_db

        # Signal-to-clutter+noise: SINR
        snr_lin = 10.0 ** (snr_map_db / 10.0)
        clutter_lin = 10.0 ** (clutter_snr_db / 10.0)
        sinr_lin = snr_lin / (1.0 + clutter_lin)
        sinr_db = 10.0 * np.log10(np.maximum(sinr_lin, 1e-30))

        # ------------------------------------------------------------------
        # 8. Spatial 2-D CFAR detection
        # ------------------------------------------------------------------
        nx, ny = cfg.grid_nx, cfg.grid_ny
        cfar_threshold_map = np.full((nx, ny), np.nan)
        detection_map = np.zeros((nx, ny), dtype=bool)

        # CFAR on SINR map (in linear domain)
        outer = cfar_guard + cfar_train
        n_train_total = (2 * outer + 1) ** 2 - (2 * cfar_guard + 1) ** 2
        if n_train_total > 0:
            alpha_cfar = n_train_total * (cfar_pfa ** (-1.0 / n_train_total) - 1.0)
        else:
            alpha_cfar = 1.0

        sinr_padded = np.pad(sinr_lin, outer, mode='edge')

        for ix in range(nx):
            for iy in range(ny):
                pi = ix + outer
                pj = iy + outer
                window = sinr_padded[
                    pi - outer: pi + outer + 1,
                    pj - outer: pj + outer + 1
                ].copy()
                # Zero guard region
                g_s = cfar_train
                g_e = cfar_train + 2 * cfar_guard + 1
                window[g_s:g_e, g_s:g_e] = 0.0
                train_vals = window[window > 0]
                if len(train_vals) == 0:
                    continue
                noise_est = np.mean(train_vals)
                threshold = alpha_cfar * noise_est
                cfar_threshold_map[ix, iy] = 10.0 * np.log10(max(threshold, 1e-30))
                detection_map[ix, iy] = sinr_lin[ix, iy] > threshold

        # ------------------------------------------------------------------
        # 9. Detection probability map (Albersheim-like approximation)
        # ------------------------------------------------------------------
        # Pd ~ logistic function of SINR
        # For SINR >> required, Pd -> 1; for SINR << required, Pd -> 0
        # Approximate required SNR for Pd=0.5 at given Pfa
        snr_req_db = 10.0 * np.log10(-np.log(max(pfa, 1e-30)))  # rough Neyman-Pearson
        pd_map = 1.0 / (1.0 + np.exp(-0.8 * (sinr_db - snr_req_db)))

        coverage = float(np.mean(pd_map > 0.5))

        return PropagationCoverageResult(
            grid_x_km=gx,
            grid_y_km=gy,
            propagation_loss_db=propagation_loss_db,
            snr_map_db=sinr_db,
            pd_map=pd_map,
            coverage_fraction=coverage,
            terrain_height_m=terrain_2d,
            refractivity_profile=M_profile,
            altitude_axis_m=height_axis,
            prop_factor_db=prop_factor_db,
            range_axis_km=range_axis_km,
            height_axis_m=height_axis,
            multipath_lobes=multipath_lobes,
            diffraction_loss_db=diffraction_loss_2d,
            clutter_map_db=clutter_map_db,
            cfar_threshold_map_db=cfar_threshold_map,
            detection_map=detection_map,
            duct_height_m=detected_duct_height,
        )


# ===========================================================================
# Level 3 -- 2-D Sensor Coverage Analysis Solver
# ===========================================================================

@dataclass
class Sensing2DResult:
    """Result of a 2-D sensor coverage / detection probability analysis.

    Attributes
    ----------
    x : np.ndarray
        X-axis coordinates [km], shape (nx,).
    y : np.ndarray
        Y-axis coordinates [km], shape (ny,).
    detection_probability : np.ndarray
        Detection probability field (0-1), shape (ny, nx).
    snr_map_db : np.ndarray
        Signal-to-noise ratio field [dB], shape (ny, nx).
    range_map_km : np.ndarray
        Range from sensor to each grid cell [km], shape (ny, nx).
    coverage_fraction : float
        Fraction of area with Pd > 0.5.
    max_detection_range_km : float
        Maximum range at which Pd > 0.5.
    mean_pd : float
        Mean detection probability over the grid.
    atmospheric_loss_db : np.ndarray
        Atmospheric attenuation [dB], shape (ny, nx).
    """
    x: np.ndarray
    y: np.ndarray
    detection_probability: np.ndarray
    snr_map_db: np.ndarray
    range_map_km: np.ndarray
    coverage_fraction: float
    max_detection_range_km: float
    mean_pd: float
    atmospheric_loss_db: np.ndarray


class Sensing2DSolver:
    """2-D sensor detection probability / coverage solver.

    Evaluates the radar equation over a 2-D surveillance grid,
    accounting for free-space path loss, atmospheric attenuation
    (rain, fog), and terrain masking to produce detection probability
    and SNR maps.

    Parameters
    ----------
    nx, ny : int
        Grid resolution.
    Lx, Ly : float
        Domain half-extent [km].
    sensor_pos : tuple
        Sensor (x, y) position [km].
    power_w : float
        Transmit power [W].
    gain_dBi : float
        Antenna gain [dBi].
    freq_ghz : float
        Operating frequency [GHz].
    bandwidth_mhz : float
        Receiver bandwidth [MHz].
    noise_figure_db : float
        Receiver noise figure [dB].
    losses_db : float
        System losses [dB].
    rcs_m2 : float
        Target RCS [m^2].
    n_pulses : int
        Number of pulses integrated.
    pfa : float
        Probability of false alarm.
    rain_rate_mmhr : float
        Rain rate [mm/hr] for atmospheric model.
    """

    fidelity_tier = FidelityTier.REDUCED_ORDER
    coupling_maturity = CouplingMaturity.M1_CONNECTABLE

    def __init__(
        self,
        nx: int = 100,
        ny: int = 100,
        Lx: float = 50.0,
        Ly: float = 50.0,
        sensor_pos: tuple = (0.0, 0.0),
        power_w: float = 10e3,
        gain_dBi: float = 35.0,
        freq_ghz: float = 10.0,
        bandwidth_mhz: float = 5.0,
        noise_figure_db: float = 5.0,
        losses_db: float = 5.0,
        rcs_m2: float = 1.0,
        n_pulses: int = 16,
        pfa: float = 1e-6,
        rain_rate_mmhr: float = 0.0,
    ):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.sensor_pos = np.array(sensor_pos, dtype=float)
        self.power_w = power_w
        self.gain_dBi = gain_dBi
        self.freq_ghz = freq_ghz
        self.bandwidth_mhz = bandwidth_mhz
        self.noise_figure_db = noise_figure_db
        self.losses_db = losses_db
        self.rcs_m2 = rcs_m2
        self.n_pulses = n_pulses
        self.pfa = pfa
        self.rain_rate_mmhr = rain_rate_mmhr

    def solve(self) -> Sensing2DResult:
        """Compute 2-D detection probability and SNR maps.

        Returns
        -------
        Sensing2DResult
        """
        nx, ny = self.nx, self.ny
        x = np.linspace(-self.Lx, self.Lx, nx)
        y = np.linspace(-self.Ly, self.Ly, ny)
        XX, YY = np.meshgrid(x, y, indexing='ij')  # (nx, ny)

        freq_hz = self.freq_ghz * 1e9
        wavelength = C_LIGHT / freq_hz
        bandwidth_hz = self.bandwidth_mhz * 1e6

        G_lin = 10.0 ** (self.gain_dBi / 10.0)
        NF_lin = 10.0 ** (self.noise_figure_db / 10.0)
        L_lin = 10.0 ** (self.losses_db / 10.0)
        k_B = 1.380649e-23
        T0 = 290.0
        N_power = k_B * T0 * bandwidth_hz * NF_lin

        # Range from sensor [km -> m]
        R_km = np.sqrt(
            (XX - self.sensor_pos[0]) ** 2 + (YY - self.sensor_pos[1]) ** 2
        )
        R_km = np.maximum(R_km, 0.001)
        R_m = R_km * 1e3

        # Atmospheric attenuation (ITU-R P.838 simplified for rain)
        # Specific attenuation [dB/km] ~ k * R^alpha, simplified
        if self.rain_rate_mmhr > 0:
            k_rain = 0.01 * (self.freq_ghz / 10.0) ** 1.5
            alpha_rain = 1.0 + 0.02 * self.freq_ghz
            atm_db_per_km = k_rain * self.rain_rate_mmhr ** alpha_rain
        else:
            atm_db_per_km = 0.01 * self.freq_ghz / 10.0  # clear air

        atm_loss_db = atm_db_per_km * R_km * 2.0  # two-way
        atm_loss_lin = 10.0 ** (-atm_loss_db / 10.0)

        # Radar equation (monostatic)
        numerator = (self.power_w * G_lin ** 2 * wavelength ** 2 *
                     self.rcs_m2 * self.n_pulses * atm_loss_lin)
        denominator = (4.0 * np.pi) ** 3 * R_m ** 4 * N_power * L_lin
        snr_lin = numerator / np.maximum(denominator, 1e-30)
        snr_db = 10.0 * np.log10(np.maximum(snr_lin, 1e-30))

        # Detection probability (Albersheim approximation)
        # SNR_req ~ -5.2 + 6.2 * sqrt(ln(0.62/Pfa)) for Pd=0.5
        snr_req_db = 10.0 * np.log10(-np.log(max(self.pfa, 1e-30)))
        pd = 1.0 / (1.0 + np.exp(-0.8 * (snr_db - snr_req_db)))

        # Transpose to (ny, nx)
        pd_out = pd.T
        snr_out = snr_db.T
        range_out = R_km.T
        atm_out = atm_loss_db.T

        coverage = float(np.mean(pd_out > 0.5))

        # Max detection range
        pd_above = pd > 0.5
        if np.any(pd_above):
            max_det_range = float(np.max(R_km[pd_above]))
        else:
            max_det_range = 0.0

        return Sensing2DResult(
            x=x,
            y=y,
            detection_probability=pd_out,
            snr_map_db=snr_out,
            range_map_km=range_out,
            coverage_fraction=coverage,
            max_detection_range_km=max_det_range,
            mean_pd=float(np.mean(pd_out)),
            atmospheric_loss_db=atm_out,
        )

    def export_state(self, result: Sensing2DResult) -> PhysicsState:
        """Export 2-D sensing results with canonical detection fields."""
        state = PhysicsState(solver_name="sensing_2d")
        state.set_field("detection_probability", result.detection_probability, "1")
        state.metadata["coverage_fraction"] = result.coverage_fraction
        state.metadata["max_detection_range_km"] = result.max_detection_range_km
        state.metadata["mean_pd"] = result.mean_pd
        state.metadata["snr_map_db"] = result.snr_map_db
        return state
