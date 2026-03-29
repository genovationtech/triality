"""
Layer 16: Agent-Driven Design Loops

Autonomous design exploration, optimization, and scenario generation for
multi-physics automotive systems.

Key Capabilities:
----------------
1. **Auto-Scenario Generation**:
   - Generate test cases covering operating envelope
   - Stress testing (worst-case combinations)
   - Corner case identification
   - Failure mode enumeration

2. **Parameter Space Exploration**:
   - Latin Hypercube Sampling (LHS)
   - Sobol sequences for quasi-random coverage
   - Adaptive sampling (focus on regions of interest)
   - Design of Experiments (DOE): full factorial, fractional factorial, Taguchi

3. **Multi-Objective Optimization**:
   - NSGA-II (Non-dominated Sorting Genetic Algorithm)
   - Pareto front discovery
   - Constraint handling
   - Trade-off analysis (efficiency vs safety, cost vs performance)

4. **Sensitivity Analysis**:
   - Morris screening (identify important parameters)
   - Sobol indices (variance-based global sensitivity)
   - Local gradient-based sensitivity
   - Interaction effects

5. **Constraint Satisfaction**:
   - Hard constraints (safety limits, regulations)
   - Soft constraints (preferences, targets)
   - Penalty methods
   - Feasibility restoration

6. **Automated Design Iterations**:
   - Iterative refinement loops
   - Convergence criteria
   - Design history tracking
   - Knowledge extraction

Physics Integration:
-------------------
Agents can invoke any layer 1-15:
- Thermal analysis (Layers 7, 11)
- Electrical-thermal coupling (Layer 12)
- EMI/EMC fields (Layer 14)
- Thermo-mechanical stress (Layer 15)
- Battery thermal safety (Layer 13)
- Reactor physics (Layers 1-10) if needed

Applications:
------------
- Busbar geometry optimization (thermal + EMI + stress)
- Power module layout (minimize crosstalk, thermal stress)
- Battery pack cooling design (thermal uniformity + safety)
- Inverter switching parameter tuning (efficiency + EMI)
- Multi-physics system validation

Example Use Cases:
-----------------
1. **Busbar Design Agent**:
   - Objective: Minimize temperature and EMI
   - Constraints: Max stress < yield, max T < 150°C
   - Variables: Width, thickness, spacing, material
   - Output: Pareto front of designs

2. **Battery Thermal Management Agent**:
   - Objective: Maximize cooling uniformity, minimize power
   - Constraints: No cell > 45°C, no thermal runaway propagation
   - Variables: Coolant flow rate, channel geometry
   - Output: Optimal cooling strategy

3. **Power Module Reliability Agent**:
   - Objective: Maximize lifetime (minimize fatigue damage)
   - Constraints: Solder joint SF > 2, warpage < 50 µm
   - Variables: Substrate thickness, die attach material
   - Output: Robust design point
"""

from .scenario_generator import (
    ScenarioGenerator,
    OperatingEnvelope,
    Scenario,
    ScenarioType
)

from .parameter_explorer import (
    ParameterExplorer,
    ParameterSpace,
    SamplingMethod,
    DOE
)

from .optimizer import (
    MultiObjectiveOptimizer,
    Objective,
    Constraint,
    OptimizationResult,
    ParetoFront,
    ObjectiveType
)

from .sensitivity import (
    SensitivityAnalyzer,
    MorrisScreening,
    SobolIndices
)

__all__ = [
    'ScenarioGenerator',
    'OperatingEnvelope',
    'Scenario',
    'ScenarioType',
    'ParameterExplorer',
    'ParameterSpace',
    'SamplingMethod',
    'DOE',
    'MultiObjectiveOptimizer',
    'Objective',
    'Constraint',
    'OptimizationResult',
    'ParetoFront',
    'ObjectiveType',
    'SensitivityAnalyzer',
    'MorrisScreening',
    'SobolIndices'
]
