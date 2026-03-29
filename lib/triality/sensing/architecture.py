"""
Multi-Sensor Architecture Reasoning

Helps reason about multi-sensor systems, sensor fusion, and complementary
sensor combinations.

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class SensorModality(Enum):
    """Sensor modality types"""
    RADAR = "radar"
    LIDAR = "lidar"
    SONAR = "sonar"
    CAMERA = "camera"
    INFRARED = "infrared"


@dataclass
class SensorCharacteristics:
    """Characteristics of a sensor"""
    modality: SensorModality
    frequency_hz: Optional[float] = None  # Carrier frequency
    range_m: float = 1000.0              # Maximum range
    angular_resolution_deg: float = 1.0   # Angular resolution
    range_resolution_m: float = 1.0      # Range resolution
    velocity_capable: bool = True        # Can measure velocity
    all_weather: bool = True             # Works in all weather
    day_night: bool = True               # Works day and night
    penetrates_foliage: bool = False     # Can see through foliage
    cost_relative: float = 1.0           # Relative cost (1.0 = baseline)


@dataclass
class OperatingConditions:
    """Operating environment conditions"""
    weather: str = 'clear'               # 'clear', 'rain', 'fog', 'snow'
    lighting: str = 'day'                # 'day', 'night', 'twilight'
    environment: str = 'open'            # 'open', 'urban', 'forest'
    target_type: str = 'vehicle'         # Type of target


class SensorComparison:
    """Compare different sensor modalities"""

    @staticmethod
    def get_typical_characteristics(modality: SensorModality) -> SensorCharacteristics:
        """Get typical characteristics for a sensor modality

        Args:
            modality: Sensor modality

        Returns:
            Typical sensor characteristics
        """
        if modality == SensorModality.RADAR:
            return SensorCharacteristics(
                modality=modality,
                frequency_hz=10e9,  # X-band
                range_m=10000,
                angular_resolution_deg=2.0,
                range_resolution_m=1.0,
                velocity_capable=True,
                all_weather=True,
                day_night=True,
                penetrates_foliage=True,  # Lower frequencies can
                cost_relative=1.0
            )

        elif modality == SensorModality.LIDAR:
            return SensorCharacteristics(
                modality=modality,
                frequency_hz=194e12,  # 1550 nm
                range_m=200,
                angular_resolution_deg=0.1,
                range_resolution_m=0.02,
                velocity_capable=True,  # With Doppler lidar
                all_weather=False,  # Degraded in rain/fog
                day_night=True,
                penetrates_foliage=False,
                cost_relative=2.0
            )

        elif modality == SensorModality.SONAR:
            return SensorCharacteristics(
                modality=modality,
                frequency_hz=50e3,  # 50 kHz
                range_m=5000,  # Underwater
                angular_resolution_deg=5.0,
                range_resolution_m=5.0,
                velocity_capable=True,
                all_weather=True,  # Underwater
                day_night=True,
                penetrates_foliage=False,
                cost_relative=1.5
            )

        elif modality == SensorModality.CAMERA:
            return SensorCharacteristics(
                modality=modality,
                frequency_hz=500e12,  # Visible light
                range_m=500,
                angular_resolution_deg=0.01,  # Very high
                range_resolution_m=None,  # Passive, no direct ranging
                velocity_capable=False,  # Not directly
                all_weather=False,
                day_night=False,  # Daylight only
                penetrates_foliage=False,
                cost_relative=0.1
            )

        elif modality == SensorModality.INFRARED:
            return SensorCharacteristics(
                modality=modality,
                frequency_hz=30e12,  # 10 μm LWIR
                range_m=1000,
                angular_resolution_deg=0.05,
                range_resolution_m=None,  # Passive
                velocity_capable=False,
                all_weather=True,  # Better than camera
                day_night=True,
                penetrates_foliage=False,
                cost_relative=3.0
            )

        else:
            raise ValueError(f"Unknown modality: {modality}")

    @staticmethod
    def compare_sensors(modalities: List[SensorModality],
                       conditions: OperatingConditions) -> Dict:
        """Compare multiple sensors under given conditions

        Args:
            modalities: List of sensor modalities to compare
            conditions: Operating conditions

        Returns:
            Comparison dictionary
        """
        results = {}

        for modality in modalities:
            char = SensorComparison.get_typical_characteristics(modality)
            score = SensorComparison._score_sensor(char, conditions)

            results[modality.value] = {
                'characteristics': char,
                'suitability_score': score,
                'strengths': SensorComparison._get_strengths(char),
                'weaknesses': SensorComparison._get_weaknesses(char),
                'conditions_rating': SensorComparison._rate_for_conditions(char, conditions)
            }

        return results

    @staticmethod
    def _score_sensor(char: SensorCharacteristics,
                     conditions: OperatingConditions) -> float:
        """Score a sensor for given conditions (0-100)

        Args:
            char: Sensor characteristics
            conditions: Operating conditions

        Returns:
            Suitability score (0-100)
        """
        score = 50.0  # Baseline

        # Weather scoring
        if conditions.weather in ['rain', 'fog', 'snow']:
            if not char.all_weather:
                score -= 30
        else:
            score += 10 if char.all_weather else 0

        # Lighting scoring
        if conditions.lighting == 'night':
            if not char.day_night:
                score -= 40
        else:
            score += 10 if char.day_night else 0

        # Environment scoring
        if conditions.environment == 'forest':
            score += 20 if char.penetrates_foliage else -10

        # Velocity capability
        if char.velocity_capable:
            score += 10

        # Resolution bonus
        if char.angular_resolution_deg < 0.5:
            score += 15  # High angular resolution

        return max(0, min(100, score))

    @staticmethod
    def _get_strengths(char: SensorCharacteristics) -> List[str]:
        """Get sensor strengths"""
        strengths = []

        if char.all_weather:
            strengths.append("All-weather operation")
        if char.day_night:
            strengths.append("Day/night operation")
        if char.velocity_capable:
            strengths.append("Direct velocity measurement")
        if char.angular_resolution_deg < 0.5:
            strengths.append("High angular resolution")
        if char.range_m > 5000:
            strengths.append("Long range")
        if char.penetrates_foliage:
            strengths.append("Foliage penetration")

        return strengths

    @staticmethod
    def _get_weaknesses(char: SensorCharacteristics) -> List[str]:
        """Get sensor weaknesses"""
        weaknesses = []

        if not char.all_weather:
            weaknesses.append("Weather-dependent")
        if not char.day_night:
            weaknesses.append("Lighting-dependent")
        if not char.velocity_capable:
            weaknesses.append("No direct velocity measurement")
        if char.angular_resolution_deg > 2.0:
            weaknesses.append("Low angular resolution")
        if char.range_m < 500:
            weaknesses.append("Limited range")
        if char.cost_relative > 2.0:
            weaknesses.append("High cost")

        return weaknesses

    @staticmethod
    def _rate_for_conditions(char: SensorCharacteristics,
                            conditions: OperatingConditions) -> str:
        """Rate sensor for specific conditions"""
        score = SensorComparison._score_sensor(char, conditions)

        if score >= 75:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Poor"


class SensorFusion:
    """Sensor fusion architecture reasoning"""

    @staticmethod
    def complementary_pair(sensor1: SensorModality,
                          sensor2: SensorModality) -> Dict[str, any]:
        """Analyze complementarity of two sensors

        Args:
            sensor1: First sensor modality
            sensor2: Second sensor modality

        Returns:
            Analysis of complementarity
        """
        char1 = SensorComparison.get_typical_characteristics(sensor1)
        char2 = SensorComparison.get_typical_characteristics(sensor2)

        # Find complementary aspects
        complementary = []
        redundant = []

        # Weather capability
        if char1.all_weather and not char2.all_weather:
            complementary.append(f"{sensor1.value} provides weather backup for {sensor2.value}")
        elif char2.all_weather and not char1.all_weather:
            complementary.append(f"{sensor2.value} provides weather backup for {sensor1.value}")
        elif char1.all_weather and char2.all_weather:
            redundant.append("Both all-weather (good redundancy)")

        # Day/night
        if char1.day_night and not char2.day_night:
            complementary.append(f"{sensor1.value} covers night operations")
        elif char2.day_night and not char1.day_night:
            complementary.append(f"{sensor2.value} covers night operations")

        # Resolution trade-offs
        if char1.angular_resolution_deg < char2.angular_resolution_deg / 2:
            complementary.append(f"{sensor1.value} provides fine angular resolution")
        elif char2.angular_resolution_deg < char1.angular_resolution_deg / 2:
            complementary.append(f"{sensor2.value} provides fine angular resolution")

        # Range trade-offs
        if char1.range_m > char2.range_m * 2:
            complementary.append(f"{sensor1.value} provides long-range detection")
        elif char2.range_m > char1.range_m * 2:
            complementary.append(f"{sensor2.value} provides long-range detection")

        # Velocity
        if char1.velocity_capable and not char2.velocity_capable:
            complementary.append(f"{sensor1.value} provides velocity measurement")
        elif char2.velocity_capable and not char1.velocity_capable:
            complementary.append(f"{sensor2.value} provides velocity measurement")

        # Synergy score
        synergy_score = len(complementary) * 20 - len(redundant) * 5
        synergy_score = max(0, min(100, synergy_score))

        return {
            'sensor1': sensor1.value,
            'sensor2': sensor2.value,
            'complementary_aspects': complementary,
            'redundant_aspects': redundant,
            'synergy_score': synergy_score,
            'recommendation': SensorFusion._fusion_recommendation(synergy_score)
        }

    @staticmethod
    def _fusion_recommendation(synergy_score: float) -> str:
        """Generate fusion recommendation"""
        if synergy_score >= 60:
            return "Highly complementary - strong fusion candidate"
        elif synergy_score >= 40:
            return "Moderately complementary - consider fusion"
        else:
            return "Limited complementarity - may not justify fusion complexity"

    @staticmethod
    def optimal_sensor_suite(conditions: OperatingConditions,
                            budget_relative: float = 5.0,
                            available_sensors: Optional[List[SensorModality]] = None) -> Dict:
        """Recommend optimal sensor suite for conditions and budget

        Args:
            conditions: Operating conditions
            budget_relative: Budget constraint (relative to radar baseline)
            available_sensors: List of available sensors (default: all)

        Returns:
            Recommended sensor suite
        """
        if available_sensors is None:
            available_sensors = [
                SensorModality.RADAR,
                SensorModality.LIDAR,
                SensorModality.CAMERA,
                SensorModality.INFRARED
            ]

        # Score all sensors
        comparison = SensorComparison.compare_sensors(available_sensors, conditions)

        # Sort by suitability
        sorted_sensors = sorted(
            comparison.items(),
            key=lambda x: x[1]['suitability_score'],
            reverse=True
        )

        # Build suite within budget
        selected = []
        total_cost = 0.0

        for sensor_name, data in sorted_sensors:
            char = data['characteristics']
            if total_cost + char.cost_relative <= budget_relative:
                selected.append({
                    'sensor': sensor_name,
                    'score': data['suitability_score'],
                    'cost': char.cost_relative,
                    'role': SensorFusion._determine_role(sensor_name, selected)
                })
                total_cost += char.cost_relative

        return {
            'recommended_suite': selected,
            'total_cost': total_cost,
            'budget': budget_relative,
            'conditions': conditions.__dict__,
            'rationale': SensorFusion._suite_rationale(selected, conditions)
        }

    @staticmethod
    def _determine_role(sensor_name: str, existing_sensors: List[Dict]) -> str:
        """Determine role of sensor in suite"""
        if len(existing_sensors) == 0:
            return "Primary sensor"
        elif len(existing_sensors) == 1:
            return "Complementary/backup"
        else:
            return "Specialized/niche"

    @staticmethod
    def _suite_rationale(selected: List[Dict], conditions: OperatingConditions) -> str:
        """Generate rationale for sensor suite"""
        if len(selected) == 0:
            return "No sensors within budget"
        elif len(selected) == 1:
            return f"Single {selected[0]['sensor']} provides best value for conditions"
        else:
            primary = selected[0]['sensor']
            secondary = selected[1]['sensor']
            return (f"{primary} as primary (score: {selected[0]['score']:.0f}), "
                   f"{secondary} as complement for enhanced capability")


# Convenience functions
def compare_radar_lidar(conditions: OperatingConditions = None) -> Dict:
    """Quick comparison of radar vs lidar

    Args:
        conditions: Operating conditions (default: clear day)

    Returns:
        Comparison results
    """
    if conditions is None:
        conditions = OperatingConditions()

    return SensorComparison.compare_sensors(
        [SensorModality.RADAR, SensorModality.LIDAR],
        conditions
    )


def best_sensor_for(weather: str = 'clear', lighting: str = 'day') -> str:
    """Find best sensor for conditions

    Args:
        weather: Weather condition
        lighting: Lighting condition

    Returns:
        Recommended sensor name
    """
    conditions = OperatingConditions(weather=weather, lighting=lighting)
    comparison = SensorComparison.compare_sensors(
        [SensorModality.RADAR, SensorModality.LIDAR, SensorModality.CAMERA],
        conditions
    )

    best = max(comparison.items(), key=lambda x: x[1]['suitability_score'])
    return best[0]
