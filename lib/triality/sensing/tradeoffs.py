"""
System-Level Trade-off Engine

Triality's "superpower" - rapid exploration of design trade-offs for sensing systems.
Answers questions like: "If I double the bandwidth, what happens to power/size/cost?"

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class ParameterType(Enum):
    """Types of parameters in trade-off analysis"""
    FREQUENCY = "frequency"
    POWER = "power"
    BANDWIDTH = "bandwidth"
    APERTURE = "aperture"
    RANGE = "range"
    PRF = "prf"
    PULSE_WIDTH = "pulse_width"
    INTEGRATION_TIME = "integration_time"


@dataclass
class SystemParameter:
    """A system parameter with value and constraints"""
    name: str
    value: float
    unit: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    cost_per_unit: Optional[float] = None  # Relative cost
    mass_per_unit: Optional[float] = None  # Relative mass
    power_per_unit: Optional[float] = None  # Relative power consumption


@dataclass
class TradeoffResult:
    """Result of trade-off analysis"""
    baseline_metric: float
    new_metric: float
    improvement_percent: float
    parameter_changed: str
    old_value: float
    new_value: float
    side_effects: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class RadarBudget:
    """Radar link budget and trade-off calculator"""

    @staticmethod
    def max_range(tx_power_w: float, antenna_gain_db: float,
                 wavelength: float, rcs_m2: float,
                 min_snr_db: float, losses_db: float = 3.0,
                 noise_figure_db: float = 3.0,
                 bandwidth_hz: float = 1e6) -> float:
        """Calculate maximum detection range from radar equation

        Args:
            tx_power_w: Transmit power in Watts
            antenna_gain_db: Antenna gain in dB
            wavelength: Wavelength in meters
            rcs_m2: Target RCS in m²
            min_snr_db: Minimum required SNR in dB
            losses_db: System losses in dB
            noise_figure_db: Noise figure in dB
            bandwidth_hz: Receiver bandwidth in Hz

        Returns:
            Maximum range in meters
        """
        from . import noise

        # Convert to linear
        G = 10 ** (antenna_gain_db / 10)
        SNR_min = 10 ** (min_snr_db / 10)
        L = 10 ** (losses_db / 10)

        # Noise power
        T_sys = noise.ThermalNoise.system_noise_temperature(290.0, noise_figure_db)
        N = noise.ThermalNoise.noise_power(T_sys, bandwidth_hz)

        # Radar equation: R_max = [(P_t G² λ² σ) / ((4π)³ SNR N L)]^(1/4)
        numerator = tx_power_w * (G ** 2) * (wavelength ** 2) * rcs_m2
        denominator = ((4 * np.pi) ** 3) * SNR_min * N * L

        R_max = (numerator / denominator) ** 0.25

        return R_max

    @staticmethod
    def required_power(range_m: float, antenna_gain_db: float,
                      wavelength: float, rcs_m2: float,
                      min_snr_db: float, losses_db: float = 3.0,
                      noise_figure_db: float = 3.0,
                      bandwidth_hz: float = 1e6) -> float:
        """Calculate required transmit power for given range

        Args:
            range_m: Target range in meters
            (other parameters same as max_range)

        Returns:
            Required transmit power in Watts
        """
        from . import noise

        # Convert to linear
        G = 10 ** (antenna_gain_db / 10)
        SNR_min = 10 ** (min_snr_db / 10)
        L = 10 ** (losses_db / 10)

        # Noise power
        T_sys = noise.ThermalNoise.system_noise_temperature(290.0, noise_figure_db)
        N = noise.ThermalNoise.noise_power(T_sys, bandwidth_hz)

        # Solve for P_t: P_t = [(4π)³ R⁴ SNR N L] / [G² λ² σ]
        P_t = ((4 * np.pi) ** 3) * (range_m ** 4) * SNR_min * N * L
        P_t /= ((G ** 2) * (wavelength ** 2) * rcs_m2)

        return P_t


class TradeoffEngine:
    """Main trade-off analysis engine"""

    def __init__(self, sensor_type: str = 'radar'):
        """Initialize trade-off engine

        Args:
            sensor_type: 'radar', 'lidar', or 'sonar'
        """
        self.sensor_type = sensor_type
        self.baseline_params = {}
        self.baseline_performance = {}

    def analyze_power_vs_range(self, baseline_power_w: float,
                              baseline_range_m: float,
                              new_power_w: float,
                              **kwargs) -> TradeoffResult:
        """Analyze power-range trade-off

        Args:
            baseline_power_w: Baseline transmit power in Watts
            baseline_range_m: Baseline range in meters
            new_power_w: New transmit power in Watts
            **kwargs: Other radar parameters

        Returns:
            TradeoffResult
        """
        # For radar: R ∝ P^(1/4)
        power_ratio = new_power_w / baseline_power_w
        new_range_m = baseline_range_m * (power_ratio ** 0.25)

        improvement = ((new_range_m - baseline_range_m) / baseline_range_m) * 100

        warnings = []
        side_effects = {}

        # Power increase warnings
        if new_power_w > baseline_power_w:
            power_increase = new_power_w - baseline_power_w
            if power_increase > 100:
                warnings.append(f"Power increase of {power_increase:.0f}W may require cooling")
            if new_power_w > 1000:
                warnings.append("High power (>1kW) requires special components")

            # Heat dissipation
            heat_w = power_increase * 0.5  # Assume 50% efficiency
            side_effects['heat_dissipation'] = f"{heat_w:.1f}W additional heat"

        # Range increase side effects
        if new_range_m > baseline_range_m * 2:
            warnings.append("Large range increase may encounter new propagation effects")

        return TradeoffResult(
            baseline_metric=baseline_range_m,
            new_metric=new_range_m,
            improvement_percent=improvement,
            parameter_changed='power',
            old_value=baseline_power_w,
            new_value=new_power_w,
            side_effects=side_effects,
            warnings=warnings
        )

    def analyze_aperture_vs_resolution(self, baseline_aperture_m: float,
                                      wavelength: float,
                                      new_aperture_m: float,
                                      typical_range_m: float = 1000.0) -> TradeoffResult:
        """Analyze aperture-resolution trade-off

        Args:
            baseline_aperture_m: Baseline aperture size in meters
            wavelength: Wavelength in meters
            new_aperture_m: New aperture size in meters
            typical_range_m: Typical operating range in meters

        Returns:
            TradeoffResult
        """
        from . import signals

        # Angular resolution: θ ≈ λ/D
        baseline_beamwidth = wavelength / baseline_aperture_m
        new_beamwidth = wavelength / new_aperture_m

        # Cross-range resolution at typical range
        baseline_resolution = baseline_beamwidth * typical_range_m
        new_resolution = new_beamwidth * typical_range_m

        improvement = ((baseline_resolution - new_resolution) / baseline_resolution) * 100

        warnings = []
        side_effects = {}

        # Aperture increase side effects
        if new_aperture_m > baseline_aperture_m:
            size_increase = new_aperture_m - baseline_aperture_m
            area_ratio = (new_aperture_m / baseline_aperture_m) ** 2

            side_effects['antenna_area'] = f"{area_ratio:.1f}x larger area"
            side_effects['antenna_mass'] = f"~{area_ratio:.1f}x heavier (approx)"

            if new_aperture_m > 1.0:
                warnings.append(f"Large aperture ({new_aperture_m:.2f}m) may have mechanical constraints")

            # Gain increases too
            gain_improvement_db = 20 * np.log10(new_aperture_m / baseline_aperture_m)
            side_effects['gain_increase'] = f"+{gain_improvement_db:.1f} dB"

        return TradeoffResult(
            baseline_metric=baseline_resolution,
            new_metric=new_resolution,
            improvement_percent=improvement,
            parameter_changed='aperture',
            old_value=baseline_aperture_m,
            new_value=new_aperture_m,
            side_effects=side_effects,
            warnings=warnings
        )

    def analyze_bandwidth_vs_range_resolution(self, baseline_bandwidth_hz: float,
                                             new_bandwidth_hz: float,
                                             wave_speed: float = 3e8) -> TradeoffResult:
        """Analyze bandwidth-range resolution trade-off

        Args:
            baseline_bandwidth_hz: Baseline bandwidth in Hz
            new_bandwidth_hz: New bandwidth in Hz
            wave_speed: Wave speed (3e8 for EM, 1500 for sonar)

        Returns:
            TradeoffResult
        """
        from . import signals

        baseline_resolution = signals.SignalCharacteristics.range_resolution(
            baseline_bandwidth_hz, 'vacuum' if wave_speed == 3e8 else 'water'
        )
        new_resolution = signals.SignalCharacteristics.range_resolution(
            new_bandwidth_hz, 'vacuum' if wave_speed == 3e8 else 'water'
        )

        improvement = ((baseline_resolution - new_resolution) / baseline_resolution) * 100

        warnings = []
        side_effects = {}

        if new_bandwidth_hz > baseline_bandwidth_hz:
            bw_ratio = new_bandwidth_hz / baseline_bandwidth_hz

            # More bandwidth = more noise
            noise_increase_db = 10 * np.log10(bw_ratio)
            side_effects['noise_increase'] = f"+{noise_increase_db:.1f} dB"

            # Data rate increases
            data_rate_increase = bw_ratio
            side_effects['data_rate'] = f"{data_rate_increase:.1f}x higher"

            # Component complexity
            if new_bandwidth_hz > 1e9:  # >1 GHz
                warnings.append("Wideband (>1GHz) requires high-speed ADC")

            if new_bandwidth_hz > 100e6:  # >100 MHz
                warnings.append("Wide bandwidth increases processing load")

        return TradeoffResult(
            baseline_metric=baseline_resolution,
            new_metric=new_resolution,
            improvement_percent=improvement,
            parameter_changed='bandwidth',
            old_value=baseline_bandwidth_hz,
            new_value=new_bandwidth_hz,
            side_effects=side_effects,
            warnings=warnings
        )

    def analyze_frequency_tradeoffs(self, baseline_freq_ghz: float,
                                   new_freq_ghz: float,
                                   range_km: float = 5.0,
                                   rain_rate_mmh: float = 10.0) -> TradeoffResult:
        """Analyze frequency trade-offs (resolution vs atmospheric loss)

        Args:
            baseline_freq_ghz: Baseline frequency in GHz
            new_freq_ghz: New frequency in GHz
            range_km: Range for loss calculation
            rain_rate_mmh: Rain rate in mm/hr

        Returns:
            TradeoffResult
        """
        from . import em_propagation

        # Resolution improves with frequency (smaller wavelength)
        baseline_wavelength = 3e8 / (baseline_freq_ghz * 1e9)
        new_wavelength = 3e8 / (new_freq_ghz * 1e9)

        resolution_improvement = ((baseline_wavelength - new_wavelength) / baseline_wavelength) * 100

        warnings = []
        side_effects = {}

        # Atmospheric losses
        baseline_rain_loss = em_propagation.rain_loss(baseline_freq_ghz, range_km, rain_rate_mmh)
        new_rain_loss = em_propagation.rain_loss(new_freq_ghz, range_km, rain_rate_mmh)

        loss_increase = new_rain_loss - baseline_rain_loss
        side_effects['rain_loss_increase'] = f"+{loss_increase:.1f} dB at {rain_rate_mmh}mm/hr"

        if loss_increase > 10:
            warnings.append(f"⚠ Significant rain loss increase: {loss_increase:.1f} dB")

        if loss_increase > 30:
            warnings.append(f"❌ CRITICAL: Rain loss increase ({loss_increase:.1f} dB) likely unacceptable")

        # Frequency-specific warnings
        if new_freq_ghz > 60 and new_freq_ghz < 63:
            warnings.append("⚠ 60 GHz oxygen absorption band - high atmospheric loss")

        if new_freq_ghz > 90:
            warnings.append("⚠ Millimeter-wave (>90 GHz) - severe weather sensitivity")

        if new_freq_ghz < 1:
            warnings.append("⚠ Low frequency (<1 GHz) - large antenna required")

        return TradeoffResult(
            baseline_metric=baseline_wavelength,
            new_metric=new_wavelength,
            improvement_percent=resolution_improvement,
            parameter_changed='frequency',
            old_value=baseline_freq_ghz,
            new_value=new_freq_ghz,
            side_effects=side_effects,
            warnings=warnings
        )

    def analyze_prf_tradeoffs(self, baseline_prf_hz: float,
                             new_prf_hz: float,
                             wavelength: float,
                             wave_speed: float = 3e8) -> TradeoffResult:
        """Analyze PRF trade-offs (max range vs max velocity)

        Args:
            baseline_prf_hz: Baseline PRF in Hz
            new_prf_hz: New PRF in Hz
            wavelength: Wavelength in meters
            wave_speed: Wave speed

        Returns:
            TradeoffResult
        """
        from . import signals

        # Unambiguous range decreases as PRF increases
        baseline_max_range = signals.SignalCharacteristics.unambiguous_range(
            baseline_prf_hz, 'vacuum' if wave_speed == 3e8 else 'water'
        )
        new_max_range = signals.SignalCharacteristics.unambiguous_range(
            new_prf_hz, 'vacuum' if wave_speed == 3e8 else 'water'
        )

        # Unambiguous velocity increases as PRF increases
        baseline_max_vel = signals.SignalCharacteristics.unambiguous_velocity(
            baseline_prf_hz, wavelength
        )
        new_max_vel = signals.SignalCharacteristics.unambiguous_velocity(
            new_prf_hz, wavelength
        )

        range_change = ((new_max_range - baseline_max_range) / baseline_max_range) * 100
        vel_change = ((new_max_vel - baseline_max_vel) / baseline_max_vel) * 100

        warnings = []
        side_effects = {}

        side_effects['max_unambiguous_range'] = f"{new_max_range/1000:.2f} km (was {baseline_max_range/1000:.2f} km)"
        side_effects['max_unambiguous_velocity'] = f"{new_max_vel:.1f} m/s (was {baseline_max_vel:.1f} m/s)"

        if new_prf_hz > baseline_prf_hz:
            warnings.append("✓ Higher PRF: better velocity coverage")
            warnings.append("⚠ Higher PRF: reduced max range (ambiguities)")
        else:
            warnings.append("✓ Lower PRF: greater max range")
            warnings.append("⚠ Lower PRF: reduced velocity coverage (ambiguities)")

        return TradeoffResult(
            baseline_metric=baseline_max_range,
            new_metric=new_max_range,
            improvement_percent=range_change,
            parameter_changed='prf',
            old_value=baseline_prf_hz,
            new_value=new_prf_hz,
            side_effects=side_effects,
            warnings=warnings
        )

    def generate_report(self, result: TradeoffResult) -> str:
        """Generate human-readable trade-off report

        Args:
            result: TradeoffResult

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append(f"TRADE-OFF ANALYSIS: {result.parameter_changed.upper()}")
        report.append("=" * 70)

        report.append(f"\nParameter Change:")
        report.append(f"  {result.parameter_changed}: {result.old_value} → {result.new_value}")

        report.append(f"\nPrimary Metric:")
        report.append(f"  Baseline: {result.baseline_metric:.6g}")
        report.append(f"  New:      {result.new_metric:.6g}")
        report.append(f"  Change:   {result.improvement_percent:+.1f}%")

        if result.side_effects:
            report.append(f"\nSide Effects:")
            for key, value in result.side_effects.items():
                report.append(f"  • {key}: {value}")

        if result.warnings:
            report.append(f"\nWarnings:")
            for warning in result.warnings:
                report.append(f"  {warning}")

        report.append("=" * 70)

        return "\n".join(report)


# Convenience functions for quick trade-off analysis
def quick_power_range_tradeoff(power_increase_factor: float,
                               baseline_range_m: float = 1000.0) -> Dict[str, float]:
    """Quick power-range trade-off

    Args:
        power_increase_factor: Power multiplier (e.g., 2.0 for 2x power)
        baseline_range_m: Baseline range in meters

    Returns:
        Dictionary with new range and improvement
    """
    # R ∝ P^(1/4)
    new_range = baseline_range_m * (power_increase_factor ** 0.25)
    improvement = ((new_range - baseline_range_m) / baseline_range_m) * 100

    return {
        'new_range_m': new_range,
        'improvement_percent': improvement,
        'power_factor': power_increase_factor
    }


def quick_aperture_resolution_tradeoff(aperture_increase_factor: float,
                                      baseline_resolution_m: float = 1.0) -> Dict[str, float]:
    """Quick aperture-resolution trade-off

    Args:
        aperture_increase_factor: Aperture multiplier
        baseline_resolution_m: Baseline resolution in meters

    Returns:
        Dictionary with new resolution and improvement
    """
    # θ ∝ 1/D, so resolution ∝ 1/D
    new_resolution = baseline_resolution_m / aperture_increase_factor
    improvement = ((baseline_resolution_m - new_resolution) / baseline_resolution_m) * 100

    return {
        'new_resolution_m': new_resolution,
        'improvement_percent': improvement,
        'aperture_factor': aperture_increase_factor,
        'mass_increase_factor': aperture_increase_factor ** 2  # Area scales as D²
    }
