"""
Early Kill-Switch Logic (Feasibility Checker)

Triality's critical value proposition: Kill bad ideas in 5 minutes, not 5 months.
Checks for physics violations, impossible requirements, and provides clear
explanations of WHY a design won't work.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class SeverityLevel(Enum):
    """Severity of feasibility issue"""
    INFO = "info"              # Informational, no concern
    WARNING = "warning"        # Potential concern, review needed
    CRITICAL = "critical"      # Serious issue, likely fatal
    SHOWSTOPPER = "showstopper"  # Physics violation, absolutely impossible


@dataclass
class FeasibilityIssue:
    """A feasibility issue found during analysis"""
    severity: SeverityLevel
    category: str              # e.g., "Propagation", "Power Budget", "Resolution"
    title: str                 # Short description
    explanation: str           # Detailed explanation with physics
    recommendation: str        # What to do about it
    numbers: Dict[str, float] = field(default_factory=dict)  # Relevant numbers


@dataclass
class FeasibilityReport:
    """Complete feasibility analysis report"""
    is_feasible: bool
    overall_assessment: str
    issues: List[FeasibilityIssue] = field(default_factory=list)
    showstoppers: int = 0
    critical_issues: int = 0
    warnings: int = 0

    def add_issue(self, issue: FeasibilityIssue):
        """Add an issue to the report"""
        self.issues.append(issue)

        if issue.severity == SeverityLevel.SHOWSTOPPER:
            self.showstoppers += 1
            self.is_feasible = False
        elif issue.severity == SeverityLevel.CRITICAL:
            self.critical_issues += 1
        elif issue.severity == SeverityLevel.WARNING:
            self.warnings += 1


class FeasibilityChecker:
    """Main feasibility checker for sensing systems"""

    def __init__(self, sensor_type: str = 'radar'):
        """Initialize feasibility checker

        Args:
            sensor_type: 'radar', 'lidar', or 'sonar'
        """
        self.sensor_type = sensor_type
        self.report = FeasibilityReport(
            is_feasible=True,
            overall_assessment="Pending analysis"
        )

    def check_radar_system(self, frequency_ghz: float, power_w: float,
                          aperture_m: float, range_km: float,
                          target_rcs_m2: float,
                          required_snr_db: float = 13.0,
                          weather_condition: str = 'clear',
                          bandwidth_hz: float = 1e6) -> FeasibilityReport:
        """Comprehensive feasibility check for radar system

        Args:
            frequency_ghz: Frequency in GHz
            power_w: Transmit power in Watts
            aperture_m: Antenna aperture in meters
            range_km: Required range in km
            target_rcs_m2: Target RCS in m²
            required_snr_db: Required SNR for detection
            weather_condition: 'clear', 'moderate_rain', 'heavy_rain'
            bandwidth_hz: Receiver bandwidth

        Returns:
            FeasibilityReport with all issues
        """
        self.report = FeasibilityReport(is_feasible=True, overall_assessment="")

        # 1. Check power-aperture-range budget
        self._check_radar_power_budget(frequency_ghz, power_w, aperture_m,
                                       range_km, target_rcs_m2, required_snr_db,
                                       bandwidth_hz)

        # 2. Check propagation feasibility
        self._check_radar_propagation(frequency_ghz, range_km, weather_condition)

        # 3. Check frequency-specific issues
        self._check_frequency_feasibility(frequency_ghz)

        # 4. Check physical constraints
        self._check_radar_physical_constraints(power_w, aperture_m, frequency_ghz)

        # 5. Check resolution vs requirements
        self._check_radar_resolution(frequency_ghz, bandwidth_hz, aperture_m, range_km)

        # Final assessment
        if self.report.showstoppers > 0:
            self.report.overall_assessment = f"❌ NOT FEASIBLE - {self.report.showstoppers} physics violation(s)"
        elif self.report.critical_issues > 0:
            self.report.overall_assessment = f"⚠ MARGINAL - {self.report.critical_issues} critical issue(s) require resolution"
        elif self.report.warnings > 0:
            self.report.overall_assessment = f"✓ FEASIBLE with caveats - {self.report.warnings} warning(s) to address"
        else:
            self.report.overall_assessment = "✓ FEASIBLE - No major issues detected"

        return self.report

    def _check_radar_power_budget(self, frequency_ghz: float, power_w: float,
                                  aperture_m: float, range_km: float,
                                  target_rcs_m2: float, required_snr_db: float,
                                  bandwidth_hz: float):
        """Check if power budget closes"""
        from . import tradeoffs, noise, em_propagation

        wavelength = 3e8 / (frequency_ghz * 1e9)

        # Calculate antenna gain
        aperture_area = np.pi * (aperture_m / 2) ** 2  # Circular aperture
        antenna_gain_linear = (4 * np.pi * aperture_area * 0.6) / (wavelength ** 2)
        antenna_gain_db = 10 * np.log10(antenna_gain_linear)

        # Calculate maximum range with this setup
        max_range_m = tradeoffs.RadarBudget.max_range(
            power_w, antenna_gain_db, wavelength, target_rcs_m2,
            required_snr_db, losses_db=3.0, bandwidth_hz=bandwidth_hz
        )

        required_range_m = range_km * 1000

        # Check if we can meet range requirement
        margin_db = 20 * np.log10(max_range_m / required_range_m)

        if margin_db < 0:
            # Showstopper: can't meet range requirement
            shortage_db = abs(margin_db)

            # Calculate what would be needed
            required_power_w = tradeoffs.RadarBudget.required_power(
                required_range_m, antenna_gain_db, wavelength, target_rcs_m2,
                required_snr_db, bandwidth_hz=bandwidth_hz
            )

            self.report.add_issue(FeasibilityIssue(
                severity=SeverityLevel.SHOWSTOPPER,
                category="Power Budget",
                title="Link budget does not close",
                explanation=(
                    f"At {power_w}W power and {aperture_m}m aperture, maximum achievable "
                    f"range is {max_range_m/1000:.2f} km, but requirement is {range_km:.2f} km. "
                    f"Link budget falls short by {shortage_db:.1f} dB.\n\n"
                    f"Physics: Radar range ∝ (Power)^(1/4), so you need "
                    f"{(required_power_w/power_w):.1f}x more power ({required_power_w:.0f}W) "
                    f"OR {(required_range_m/max_range_m)**2:.1f}x larger aperture to reach {range_km} km."
                ),
                recommendation=(
                    f"OPTION 1: Increase power to {required_power_w:.0f}W\n"
                    f"OPTION 2: Increase aperture by {(required_range_m/max_range_m)**0.5:.1f}x\n"
                    f"OPTION 3: Reduce range requirement to {max_range_m/1000:.2f} km\n"
                    f"OPTION 4: Accept larger target (RCS > {target_rcs_m2*(required_range_m/max_range_m)**4:.1f} m²)"
                ),
                numbers={
                    'max_achievable_range_km': max_range_m / 1000,
                    'required_range_km': range_km,
                    'shortfall_db': shortage_db,
                    'required_power_w': required_power_w
                }
            ))

        elif margin_db < 3:
            # Critical: very tight margin
            self.report.add_issue(FeasibilityIssue(
                severity=SeverityLevel.CRITICAL,
                category="Power Budget",
                title="Insufficient link margin",
                explanation=(
                    f"Link budget barely closes with only {margin_db:.1f} dB margin. "
                    f"Industry standard is 6-10 dB margin for reliable operation.\n\n"
                    f"This leaves no room for: component variations, aging, atmospheric "
                    f"variations, or RCS fluctuations."
                ),
                recommendation=f"Increase power or aperture to achieve 6+ dB margin",
                numbers={'current_margin_db': margin_db}
            ))

        elif margin_db < 6:
            # Warning: marginal
            self.report.add_issue(FeasibilityIssue(
                severity=SeverityLevel.WARNING,
                category="Power Budget",
                title="Low link margin",
                explanation=f"Link margin is {margin_db:.1f} dB. Acceptable but tight.",
                recommendation="Consider 6-10 dB margin for robust operation",
                numbers={'current_margin_db': margin_db}
            ))

    def _check_radar_propagation(self, frequency_ghz: float, range_km: float,
                                weather_condition: str):
        """Check propagation feasibility"""
        from . import em_propagation

        # Rain rates
        rain_rates = {
            'clear': 0.0,
            'light_rain': 2.5,
            'moderate_rain': 10.0,
            'heavy_rain': 50.0
        }

        rain_rate = rain_rates.get(weather_condition, 10.0)

        if rain_rate > 0:
            # Two-way rain loss
            rain_loss_db = em_propagation.rain_loss(frequency_ghz, range_km, rain_rate)

            if rain_loss_db > 50:
                # Showstopper: extreme rain loss
                self.report.add_issue(FeasibilityIssue(
                    severity=SeverityLevel.SHOWSTOPPER,
                    category="Propagation",
                    title="Catastrophic rain attenuation",
                    explanation=(
                        f"At {frequency_ghz} GHz and {rain_rate} mm/hr rain rate, "
                        f"two-way rain loss is {rain_loss_db:.1f} dB over {range_km} km path.\n\n"
                        f"This is EXTREME and effectively blocks the signal. At this frequency, "
                        f"rain droplets are significant scatterers.\n\n"
                        f"Physics: Rain attenuation ∝ f² for millimeter waves. "
                        f"At 94 GHz, heavy rain causes ~15 dB/km loss!"
                    ),
                    recommendation=(
                        f"REQUIRED: Use much lower frequency (<10 GHz) OR dramatically "
                        f"reduce range (<{range_km/5:.1f} km) OR accept system fails in rain"
                    ),
                    numbers={
                        'rain_loss_db': rain_loss_db,
                        'rain_rate_mmh': rain_rate,
                        'frequency_ghz': frequency_ghz
                    }
                ))

            elif rain_loss_db > 20:
                # Critical: severe rain loss
                self.report.add_issue(FeasibilityIssue(
                    severity=SeverityLevel.CRITICAL,
                    category="Propagation",
                    title="Severe rain attenuation",
                    explanation=(
                        f"Rain loss is {rain_loss_db:.1f} dB at {rain_rate} mm/hr. "
                        f"This severely degrades performance in rain."
                    ),
                    recommendation=(
                        f"Consider lower frequency (<{frequency_ghz/2:.0f} GHz) or "
                        f"add {rain_loss_db:.0f} dB additional link margin"
                    ),
                    numbers={'rain_loss_db': rain_loss_db}
                ))

            elif rain_loss_db > 10:
                # Warning: significant rain loss
                self.report.add_issue(FeasibilityIssue(
                    severity=SeverityLevel.WARNING,
                    category="Propagation",
                    title="Significant rain attenuation",
                    explanation=f"Rain loss is {rain_loss_db:.1f} dB in moderate rain",
                    recommendation="Ensure link margin accounts for rain loss",
                    numbers={'rain_loss_db': rain_loss_db}
                ))

    def _check_frequency_feasibility(self, frequency_ghz: float):
        """Check frequency-specific issues"""

        # Oxygen absorption band (60 GHz)
        if 58 <= frequency_ghz <= 63:
            self.report.add_issue(FeasibilityIssue(
                severity=SeverityLevel.CRITICAL,
                category="Frequency",
                title="Oxygen absorption band (60 GHz)",
                explanation=(
                    f"Frequency {frequency_ghz} GHz is in the oxygen absorption band "
                    f"(58-63 GHz). Atmospheric absorption can exceed 15 dB/km.\n\n"
                    f"Physics: Molecular oxygen (O₂) has strong resonance near 60 GHz."
                ),
                recommendation=(
                    "Use this band ONLY for short-range (<1 km) applications, "
                    "or choose different frequency"
                ),
                numbers={'frequency_ghz': frequency_ghz}
            ))

        # Water vapor absorption (183 GHz, 22 GHz)
        if 180 <= frequency_ghz <= 186:
            self.report.add_issue(FeasibilityIssue(
                severity=SeverityLevel.CRITICAL,
                category="Frequency",
                title="Water vapor absorption band (183 GHz)",
                explanation="Strong water vapor absorption at 183 GHz line",
                recommendation="Avoid this band or accept severe weather sensitivity",
                numbers={'frequency_ghz': frequency_ghz}
            ))

        # Millimeter-wave challenges (>30 GHz)
        if frequency_ghz > 30:
            self.report.add_issue(FeasibilityIssue(
                severity=SeverityLevel.WARNING if frequency_ghz < 60 else SeverityLevel.CRITICAL,
                category="Frequency",
                title="Millimeter-wave challenges",
                explanation=(
                    f"Millimeter-wave ({frequency_ghz} GHz) faces multiple challenges: "
                    f"rain attenuation, atmospheric absorption, component complexity, "
                    f"and pointing accuracy requirements."
                ),
                recommendation="Ensure weather sensitivity is acceptable for application",
                numbers={'frequency_ghz': frequency_ghz}
            ))

    def _check_radar_physical_constraints(self, power_w: float, aperture_m: float,
                                         frequency_ghz: float):
        """Check physical/practical constraints"""

        # Power constraints
        if power_w > 10000:  # 10 kW
            self.report.add_issue(FeasibilityIssue(
                severity=SeverityLevel.CRITICAL,
                category="Physical Constraints",
                title="Very high power requirement",
                explanation=(
                    f"Transmit power of {power_w/1000:.1f} kW is extremely high.\n"
                    f"Requires: specialized high-power amplifiers, serious cooling, "
                    f"high voltage power supply, and safety measures."
                ),
                recommendation="Review if power can be reduced via larger aperture",
                numbers={'power_kw': power_w / 1000}
            ))

        elif power_w > 1000:  # 1 kW
            self.report.add_issue(FeasibilityIssue(
                severity=SeverityLevel.WARNING,
                category="Physical Constraints",
                title="High power requirement",
                explanation=f"Power level ({power_w}W) requires careful thermal design",
                recommendation="Plan for adequate cooling and power supply",
                numbers={'power_w': power_w}
            ))

        # Aperture constraints
        if aperture_m > 10.0:
            self.report.add_issue(FeasibilityIssue(
                severity=SeverityLevel.CRITICAL,
                category="Physical Constraints",
                title="Very large aperture",
                explanation=(
                    f"Aperture diameter of {aperture_m} m is extremely large.\n"
                    f"Mechanical challenges: wind loading, pointing accuracy, "
                    f"manufacturing, cost."
                ),
                recommendation="Consider if aperture can be reduced via more power or different approach",
                numbers={'aperture_m': aperture_m}
            ))

        elif aperture_m > 3.0:
            self.report.add_issue(FeasibilityIssue(
                severity=SeverityLevel.WARNING,
                category="Physical Constraints",
                title="Large aperture",
                explanation=f"Aperture of {aperture_m} m requires substantial structure",
                recommendation="Verify mechanical design can accommodate size",
                numbers={'aperture_m': aperture_m}
            ))

    def _check_radar_resolution(self, frequency_ghz: float, bandwidth_hz: float,
                               aperture_m: float, range_km: float):
        """Check resolution capabilities"""
        from . import signals

        wavelength = 3e8 / (frequency_ghz * 1e9)

        # Range resolution
        range_res_m = signals.SignalCharacteristics.range_resolution(bandwidth_hz, 'vacuum')

        # Angular resolution
        beamwidth_rad = signals.BeamCharacteristics.beamwidth_2d(wavelength, aperture_m)
        cross_range_res_m = beamwidth_rad * range_km * 1000

        # Info: report resolution capabilities
        self.report.add_issue(FeasibilityIssue(
            severity=SeverityLevel.INFO,
            category="Resolution",
            title="Resolution capabilities",
            explanation=(
                f"Range resolution: {range_res_m:.2f} m\n"
                f"Cross-range resolution at {range_km} km: {cross_range_res_m:.2f} m\n"
                f"Beamwidth: {np.degrees(beamwidth_rad):.3f}°"
            ),
            recommendation="Verify if resolution meets application requirements",
            numbers={
                'range_resolution_m': range_res_m,
                'cross_range_resolution_m': cross_range_res_m,
                'beamwidth_deg': np.degrees(beamwidth_rad)
            }
        ))

    def generate_report_text(self) -> str:
        """Generate human-readable feasibility report

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append(f"FEASIBILITY ANALYSIS REPORT - {self.sensor_type.upper()} SYSTEM")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"OVERALL ASSESSMENT: {self.report.overall_assessment}")
        lines.append("")
        lines.append(f"Issues Found: {len(self.report.issues)}")
        lines.append(f"  • Showstoppers (physics violations): {self.report.showstoppers}")
        lines.append(f"  • Critical issues: {self.report.critical_issues}")
        lines.append(f"  • Warnings: {self.report.warnings}")
        lines.append("")

        if self.report.showstoppers > 0:
            lines.append("=" * 80)
            lines.append("❌ SHOWSTOPPERS (PHYSICS VIOLATIONS - CANNOT PROCEED)")
            lines.append("=" * 80)
            for issue in self.report.issues:
                if issue.severity == SeverityLevel.SHOWSTOPPER:
                    lines.append(self._format_issue(issue))

        if self.report.critical_issues > 0:
            lines.append("=" * 80)
            lines.append("⚠ CRITICAL ISSUES (MUST RESOLVE)")
            lines.append("=" * 80)
            for issue in self.report.issues:
                if issue.severity == SeverityLevel.CRITICAL:
                    lines.append(self._format_issue(issue))

        if self.report.warnings > 0:
            lines.append("=" * 80)
            lines.append("⚠ WARNINGS (SHOULD ADDRESS)")
            lines.append("=" * 80)
            for issue in self.report.issues:
                if issue.severity == SeverityLevel.WARNING:
                    lines.append(self._format_issue(issue))

        # Info items
        info_issues = [i for i in self.report.issues if i.severity == SeverityLevel.INFO]
        if info_issues:
            lines.append("=" * 80)
            lines.append("ℹ INFORMATION")
            lines.append("=" * 80)
            for issue in info_issues:
                lines.append(self._format_issue(issue))

        lines.append("=" * 80)
        return "\n".join(lines)

    def _format_issue(self, issue: FeasibilityIssue) -> str:
        """Format a single issue"""
        lines = []
        lines.append("")
        lines.append(f"[{issue.category}] {issue.title}")
        lines.append("-" * 80)
        lines.append(f"{issue.explanation}")
        if issue.recommendation:
            lines.append(f"\nRECOMMENDATION:")
            lines.append(f"{issue.recommendation}")
        if issue.numbers:
            lines.append(f"\nKEY NUMBERS:")
            for key, value in issue.numbers.items():
                lines.append(f"  • {key}: {value:.6g}")
        lines.append("")
        return "\n".join(lines)


# Convenience function for quick feasibility check
def quick_radar_check(frequency_ghz: float, power_w: float, aperture_m: float,
                     range_km: float, target_rcs_m2: float = 1.0,
                     weather: str = 'moderate_rain') -> bool:
    """Quick feasibility check for radar

    Args:
        frequency_ghz: Frequency in GHz
        power_w: Transmit power in Watts
        aperture_m: Aperture diameter in meters
        range_km: Required range in km
        target_rcs_m2: Target RCS in m²
        weather: Weather condition

    Returns:
        True if feasible, False otherwise
    """
    checker = FeasibilityChecker('radar')
    report = checker.check_radar_system(
        frequency_ghz, power_w, aperture_m, range_km,
        target_rcs_m2, weather_condition=weather
    )

    print(checker.generate_report_text())

    return report.is_feasible
