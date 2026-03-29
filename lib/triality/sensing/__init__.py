"""
Triality Sensing Systems Module

Comprehensive radar, lidar, and sonar analysis toolkit with physics-based models,
trade-off analysis, and early feasibility checking.

Key Features:
- Wave & signal fundamentals
- Medium-aware propagation physics (EM + acoustic)
- Target interaction models (RCS, TS, reflectivity)
- Noise, clutter, and interference
- Detection probability and resolution limits
- System-level trade-off engine
- Early kill-switch logic (feasibility checker)
- Multi-sensor architecture reasoning

© 2025 Genovation Technological Solutions Pvt Ltd. All rights reserved. Powered by Mentis OS.
"""

# Core physics modules
from . import signals
from . import em_propagation
from . import acoustic_propagation
from . import targets
from . import noise
from . import detection

# Analysis engines (Triality's superpowers)
from . import tradeoffs
from . import feasibility
from . import architecture

# Convenience interfaces
from . import radar
from . import lidar
from . import sonar

# High-level exports for quick access
from .signals import (
    WaveParameters,
    SignalCharacteristics,
    BeamCharacteristics,
    wavelength,
    frequency,
    beamwidth,
    doppler
)

from .em_propagation import (
    RainAttenuation,
    AtmosphericAbsorption,
    AtmosphericConditions,
    rain_loss,
    atmospheric_loss
)

from .acoustic_propagation import (
    SoundSpeed,
    UnderwaterAbsorption,
    WaterConditions,
    underwater_loss,
    sound_speed_water
)

from .targets import (
    RadarCrossSection,
    TargetStrength,
    OpticalReflectivity,
    rcs,
    target_strength,
    reflectance
)

from .noise import (
    ThermalNoise,
    ClutterModels,
    thermal_noise_dbm,
    clutter_rcs
)

from .detection import (
    DetectionProbability,
    ResolutionLimits,
    required_snr_for_detection,
    detection_probability
)

from .tradeoffs import (
    TradeoffEngine,
    quick_power_range_tradeoff,
    quick_aperture_resolution_tradeoff
)

from .feasibility import (
    FeasibilityChecker,
    quick_radar_check
)

from .architecture import (
    SensorComparison,
    SensorFusion,
    SensorModality,
    OperatingConditions,
    compare_radar_lidar,
    best_sensor_for
)

from .radar import (
    RadarSystem,
    RadarAnalyzer,
    quick_radar_analysis,
    check_radar_feasibility
)

from .lidar import (
    LidarSystem,
    LidarAnalyzer,
    lidar_resolution
)

from .sonar import (
    SonarSystem,
    SonarAnalyzer,
    sonar_detection_range
)

from .solver import (
    SensorPerformanceSolver,
    SensorPerformanceResult,
    SensorPerformanceConfig,
    SensorSpec,
    SensorType,
    TargetSpec,
)


__version__ = "1.0.0"

__all__ = [
    # Core modules
    'signals',
    'em_propagation',
    'acoustic_propagation',
    'targets',
    'noise',
    'detection',
    'tradeoffs',
    'feasibility',
    'architecture',
    'radar',
    'lidar',
    'sonar',

    # Quick access classes
    'WaveParameters',
    'SignalCharacteristics',
    'BeamCharacteristics',
    'RainAttenuation',
    'AtmosphericAbsorption',
    'AtmosphericConditions',
    'SoundSpeed',
    'WaterConditions',
    'RadarCrossSection',
    'TargetStrength',
    'OpticalReflectivity',
    'ThermalNoise',
    'ClutterModels',
    'DetectionProbability',
    'ResolutionLimits',
    'TradeoffEngine',
    'FeasibilityChecker',
    'SensorComparison',
    'SensorFusion',
    'SensorModality',
    'OperatingConditions',
    'RadarSystem',
    'RadarAnalyzer',
    'LidarSystem',
    'LidarAnalyzer',
    'SonarSystem',
    'SonarAnalyzer',

    # Quick access functions
    'wavelength',
    'frequency',
    'beamwidth',
    'doppler',
    'rain_loss',
    'atmospheric_loss',
    'underwater_loss',
    'sound_speed_water',
    'rcs',
    'target_strength',
    'reflectance',
    'thermal_noise_dbm',
    'clutter_rcs',
    'required_snr_for_detection',
    'detection_probability',
    'quick_power_range_tradeoff',
    'quick_aperture_resolution_tradeoff',
    'quick_radar_analysis',
    'check_radar_feasibility',
    'quick_radar_check',
    'compare_radar_lidar',
    'best_sensor_for',
    'lidar_resolution',
    'sonar_detection_range',

    # Solver
    'SensorPerformanceSolver',
    'SensorPerformanceResult',
    'SensorPerformanceConfig',
    'SensorSpec',
    'SensorType',
    'TargetSpec',
]
