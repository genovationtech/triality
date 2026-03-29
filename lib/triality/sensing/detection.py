"""
Detection and Resolution Limits

Probability of detection, false alarm rates, ROC curves, and resolution
limits for sensing systems.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from scipy import special  # For error functions


@dataclass
class DetectionParameters:
    """Detection parameters"""
    snr_db: float                     # Signal-to-noise ratio (dB)
    probability_false_alarm: float    # P_fa (0-1)
    num_pulses: int = 1              # Number of pulses integrated
    swerling_case: int = 0           # Swerling case (0=non-fluctuating)


class DetectionProbability:
    """Detection probability calculations"""

    @staticmethod
    def threshold_from_pfa(pfa: float, num_samples: int = 1) -> float:
        """Calculate detection threshold from false alarm probability

        For single pulse detection in Gaussian noise

        Args:
            pfa: Probability of false alarm (0-1)
            num_samples: Number of independent samples

        Returns:
            Detection threshold (normalized to noise power)
        """
        if pfa <= 0 or pfa >= 1:
            raise ValueError("P_fa must be between 0 and 1")

        # For Gaussian noise: threshold = Q^(-1)(P_fa)
        # where Q is the complementary error function
        # Using approximation: threshold ≈ √(2) * erfc^(-1)(2*P_fa)

        # Simplified: use inverse Q-function
        # For small P_fa: threshold ≈ √(-2 ln(P_fa))
        threshold = np.sqrt(-2 * np.log(pfa))

        return threshold

    @staticmethod
    def pd_single_pulse(snr_db: float, pfa: float) -> float:
        """Probability of detection for single pulse (non-fluctuating target)

        Args:
            snr_db: Signal-to-noise ratio in dB
            pfa: Probability of false alarm

        Returns:
            Probability of detection (0-1)
        """
        snr_linear = 10 ** (snr_db / 10)

        # Detection threshold
        V_t = DetectionProbability.threshold_from_pfa(pfa)

        # Probability of detection (Marcum's formula, approximation)
        # P_d ≈ Q(V_t - √(2*SNR))  for large SNR
        # Using simplified approximation

        if snr_linear <= 0:
            return 0.0

        # Approximate formula
        argument = V_t - np.sqrt(2 * snr_linear)

        if argument <= -5:
            pd = 1.0
        elif argument >= 5:
            pd = 0.0
        else:
            # Q-function approximation
            pd = 0.5 * special.erfc(argument / np.sqrt(2))

        return min(max(pd, 0.0), 1.0)

    @staticmethod
    def pd_with_integration(snr_db: float, pfa: float, n_pulses: int) -> float:
        """Probability of detection with coherent integration

        Args:
            snr_db: Single-pulse SNR in dB
            pfa: Probability of false alarm
            n_pulses: Number of pulses integrated

        Returns:
            Probability of detection (0-1)
        """
        # Coherent integration: SNR increases by 10*log10(n)
        integrated_snr_db = snr_db + 10 * np.log10(n_pulses)

        return DetectionProbability.pd_single_pulse(integrated_snr_db, pfa)

    @staticmethod
    def pd_swerling(snr_db: float, pfa: float, n_pulses: int,
                   swerling_case: int) -> float:
        """Probability of detection for fluctuating targets (Swerling models)

        Args:
            snr_db: Average SNR in dB
            pfa: Probability of false alarm
            n_pulses: Number of pulses
            swerling_case: Swerling case (1, 2, 3, 4)

        Returns:
            Probability of detection (0-1)
        """
        snr_linear = 10 ** (snr_db / 10)

        # Simplified approximations for different Swerling cases
        # Full calculation requires Marcum Q-functions

        if swerling_case == 0:
            # Non-fluctuating
            return DetectionProbability.pd_with_integration(snr_db, pfa, n_pulses)

        elif swerling_case == 1:
            # Case 1: Slow fluctuation (one sample per scan)
            # Rayleigh amplitude
            # Simplified: requires higher SNR than non-fluctuating
            loss_factor = 5.0  # Approximate loss in dB
            return DetectionProbability.pd_single_pulse(snr_db - loss_factor, pfa)

        elif swerling_case == 2:
            # Case 2: Fast fluctuation (many samples)
            # Rayleigh amplitude, pulse-to-pulse decorrelation
            # Integration helps
            loss_factor = 3.0
            return DetectionProbability.pd_with_integration(snr_db - loss_factor, pfa, n_pulses)

        elif swerling_case == 3:
            # Case 3: Slow fluctuation, one dominant reflector
            loss_factor = 2.0
            return DetectionProbability.pd_single_pulse(snr_db - loss_factor, pfa)

        elif swerling_case == 4:
            # Case 4: Fast fluctuation, one dominant reflector
            loss_factor = 1.0
            return DetectionProbability.pd_with_integration(snr_db - loss_factor, pfa, n_pulses)

        else:
            return DetectionProbability.pd_single_pulse(snr_db, pfa)

    @staticmethod
    def required_snr(pd_target: float, pfa: float, n_pulses: int = 1,
                    swerling_case: int = 0) -> float:
        """Calculate required SNR for target P_d and P_fa

        Args:
            pd_target: Target probability of detection
            pfa: Probability of false alarm
            n_pulses: Number of pulses integrated
            swerling_case: Swerling case

        Returns:
            Required SNR in dB
        """
        # Binary search for required SNR
        snr_min = -20.0  # dB
        snr_max = 40.0   # dB

        for _ in range(30):  # Iterations
            snr_test = (snr_min + snr_max) / 2

            pd_achieved = DetectionProbability.pd_swerling(
                snr_test, pfa, n_pulses, swerling_case
            )

            if pd_achieved < pd_target:
                snr_min = snr_test
            else:
                snr_max = snr_test

        return (snr_min + snr_max) / 2


class ROCAnalysis:
    """ROC (Receiver Operating Characteristic) curve analysis"""

    @staticmethod
    def generate_roc_curve(snr_db: float, pfa_range: List[float]) -> Dict[str, List[float]]:
        """Generate ROC curve (P_d vs P_fa)

        Args:
            snr_db: SNR in dB
            pfa_range: List of P_fa values to evaluate

        Returns:
            Dictionary with 'pfa' and 'pd' lists
        """
        pd_values = []

        for pfa in pfa_range:
            pd = DetectionProbability.pd_single_pulse(snr_db, pfa)
            pd_values.append(pd)

        return {
            'pfa': pfa_range,
            'pd': pd_values,
            'snr_db': snr_db
        }

    @staticmethod
    def area_under_roc(snr_db: float, n_points: int = 100) -> float:
        """Calculate area under ROC curve (AUC)

        Args:
            snr_db: SNR in dB
            n_points: Number of points for integration

        Returns:
            AUC (0-1, closer to 1 is better)
        """
        pfa_range = np.logspace(-6, 0, n_points)
        roc = ROCAnalysis.generate_roc_curve(snr_db, pfa_range.tolist())

        # Trapezoidal integration
        auc = np.trapezoid(roc['pd'], roc['pfa'])

        return auc


class ResolutionLimits:
    """Resolution limits for sensing systems"""

    @staticmethod
    def range_resolution_from_bandwidth(bandwidth_hz: float,
                                       wave_speed: float = 3e8) -> float:
        """Range resolution from signal bandwidth

        Δr = c / (2 * B)

        Args:
            bandwidth_hz: Signal bandwidth in Hz
            wave_speed: Wave propagation speed (m/s)

        Returns:
            Range resolution in meters
        """
        return wave_speed / (2 * bandwidth_hz)

    @staticmethod
    def doppler_resolution(coherent_integration_time: float) -> float:
        """Doppler resolution from CPI time

        Δf_d = 1 / T_cpi

        Args:
            coherent_integration_time: CPI time in seconds

        Returns:
            Doppler resolution in Hz
        """
        return 1.0 / coherent_integration_time

    @staticmethod
    def velocity_resolution(coherent_integration_time: float,
                          wavelength: float) -> float:
        """Velocity resolution

        Args:
            coherent_integration_time: CPI time in seconds
            wavelength: Wavelength in meters

        Returns:
            Velocity resolution in m/s
        """
        delta_fd = ResolutionLimits.doppler_resolution(coherent_integration_time)
        return delta_fd * wavelength / 2

    @staticmethod
    def angular_resolution_rayleigh(wavelength: float, aperture: float) -> float:
        """Angular resolution (Rayleigh criterion)

        θ = 1.22 * λ / D

        Args:
            wavelength: Wavelength in meters
            aperture: Aperture size in meters

        Returns:
            Angular resolution in radians
        """
        return 1.22 * wavelength / aperture

    @staticmethod
    def angular_resolution_beamwidth(beamwidth_rad: float) -> float:
        """Angular resolution from beamwidth

        Typically Δθ ≈ beamwidth / 2

        Args:
            beamwidth_rad: 3dB beamwidth in radians

        Returns:
            Angular resolution in radians
        """
        return beamwidth_rad / 2

    @staticmethod
    def cross_range_resolution(range_m: float, angular_resolution_rad: float) -> float:
        """Cross-range resolution at given range

        Args:
            range_m: Range in meters
            angular_resolution_rad: Angular resolution in radians

        Returns:
            Cross-range resolution in meters
        """
        return range_m * angular_resolution_rad

    @staticmethod
    def synthetic_aperture_resolution(wavelength: float,
                                     real_aperture: float) -> float:
        """SAR azimuth resolution (unfocused)

        For focused SAR: Δx ≈ D/2 (independent of range!)

        Args:
            wavelength: Wavelength in meters
            real_aperture: Real antenna aperture in meters

        Returns:
            Azimuth resolution in meters (focused SAR)
        """
        # Focused SAR: resolution = D/2
        return real_aperture / 2


class AccuracyLimits:
    """Measurement accuracy limits"""

    @staticmethod
    def range_accuracy(range_resolution: float, snr_db: float) -> float:
        """Range measurement accuracy (RMS error)

        σ_r ≈ Δr / (2 * √SNR)

        Args:
            range_resolution: Range resolution in meters
            snr_db: SNR in dB

        Returns:
            Range accuracy (1-sigma) in meters
        """
        snr_linear = 10 ** (snr_db / 10)
        return range_resolution / (2 * np.sqrt(snr_linear))

    @staticmethod
    def doppler_accuracy(doppler_resolution: float, snr_db: float) -> float:
        """Doppler measurement accuracy

        σ_fd ≈ Δf_d / √SNR

        Args:
            doppler_resolution: Doppler resolution in Hz
            snr_db: SNR in dB

        Returns:
            Doppler accuracy (1-sigma) in Hz
        """
        snr_linear = 10 ** (snr_db / 10)
        return doppler_resolution / np.sqrt(snr_linear)

    @staticmethod
    def angle_accuracy(beamwidth_rad: float, snr_db: float) -> float:
        """Angle measurement accuracy (monopulse)

        σ_θ ≈ θ_3dB / √SNR

        Args:
            beamwidth_rad: Beamwidth in radians
            snr_db: SNR in dB

        Returns:
            Angle accuracy (1-sigma) in radians
        """
        snr_linear = 10 ** (snr_db / 10)
        return beamwidth_rad / np.sqrt(snr_linear)

    @staticmethod
    def cramer_rao_lower_bound(signal_bandwidth_hz: float,
                              snr_db: float,
                              parameter: str = 'delay') -> float:
        """Cramer-Rao lower bound for parameter estimation

        Args:
            signal_bandwidth_hz: Signal bandwidth in Hz
            snr_db: SNR in dB
            parameter: 'delay' (range), 'frequency' (Doppler), or 'phase'

        Returns:
            CRLB (variance) for the parameter
        """
        snr_linear = 10 ** (snr_db / 10)

        if parameter == 'delay':
            # For delay: CRLB_τ = 1 / (8π² * B² * SNR)
            return 1.0 / (8 * np.pi**2 * signal_bandwidth_hz**2 * snr_linear)

        elif parameter == 'frequency':
            # For frequency: CRLB_f = 1 / (8π² * T² * SNR)
            # Assume observation time T = 1/B
            T = 1.0 / signal_bandwidth_hz
            return 1.0 / (8 * np.pi**2 * T**2 * snr_linear)

        else:
            # Generic
            return 1.0 / snr_linear


class TrackingLimits:
    """Tracking performance limits"""

    @staticmethod
    def track_initiation_probability(pd: float, n_out_of_m: Tuple[int, int]) -> float:
        """Probability of track initiation (N out of M logic)

        Args:
            pd: Single-scan detection probability
            n_out_of_m: Tuple (N, M) - need N detections in M scans

        Returns:
            Track initiation probability
        """
        N, M = n_out_of_m

        # Binomial probability: P(≥N detections in M scans)
        prob = 0.0
        for k in range(N, M + 1):
            # C(M, k) * pd^k * (1-pd)^(M-k)
            binom_coeff = special.comb(M, k, exact=True)
            prob += binom_coeff * (pd ** k) * ((1 - pd) ** (M - k))

        return prob

    @staticmethod
    def track_maintenance_probability(pd: float, n_consecutive_misses: int) -> float:
        """Probability of maintaining track (before dropping)

        Args:
            pd: Single-scan detection probability
            n_consecutive_misses: Number of consecutive misses before track drop

        Returns:
            Probability of track survival per scan
        """
        # Probability of NOT having n consecutive misses
        # Simplified: assume track drops after n consecutive misses
        p_drop_per_scan = (1 - pd) ** n_consecutive_misses

        return 1 - p_drop_per_scan


class AmbiguityFunction:
    """Ambiguity function analysis"""

    @staticmethod
    def range_doppler_ambiguity(tau: float, fd: float,
                               pulse_width: float, bandwidth: float) -> float:
        """Ambiguity function magnitude (simplified)

        Args:
            tau: Delay in seconds
            fd: Doppler shift in Hz
            pulse_width: Pulse width in seconds
            bandwidth: Signal bandwidth in Hz

        Returns:
            Ambiguity function magnitude (0-1)
        """
        # Simplified rectangular pulse ambiguity function
        # |χ(τ, f_d)| = |sinc(π * B * τ)| * |sinc(π * T * f_d)|

        range_term = np.abs(np.sinc(bandwidth * tau))
        doppler_term = np.abs(np.sinc(pulse_width * fd))

        return range_term * doppler_term


# Convenience functions
def required_snr_for_detection(pd: float = 0.9, pfa: float = 1e-6,
                               n_pulses: int = 1) -> float:
    """Quick calculation of required SNR

    Args:
        pd: Target detection probability
        pfa: False alarm probability
        n_pulses: Number of pulses integrated

    Returns:
        Required SNR in dB
    """
    return DetectionProbability.required_snr(pd, pfa, n_pulses)


def detection_probability(snr_db: float, pfa: float = 1e-6) -> float:
    """Quick P_d calculation

    Args:
        snr_db: SNR in dB
        pfa: False alarm probability

    Returns:
        Probability of detection (0-1)
    """
    return DetectionProbability.pd_single_pulse(snr_db, pfa)
