"""
Goal-Driven Convergence Engine for Triality
=============================================
Transforms Triality from a single-pass analysis tool into a goal-convergent
problem solver. Extracts structured goals from user prompts, evaluates results
against those goals, and iteratively adapts parameters until the answer is found.

Architecture:
    GoalExtractor  — Parses user prompt into AnalysisGoal
    GoalEvaluator  — Checks results against the goal condition
    ConvergenceStrategy — Adapts parameters when goal isn't met
    GoalDrivenRunner — Orchestrates the iterative loop
"""
from __future__ import annotations

import copy
import json
import logging
import math
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("triality.goal_engine")

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
MAX_CONVERGENCE_ITERATIONS = 8
CROSSING_TOLERANCE_PERCENT = 2.0   # Converge to within 2% of threshold
MIN_BISECTION_POINTS = 8           # Points per bisection sweep
EXPANSION_FACTOR = 3.0             # Multiply range by this when expanding
ANALYTICAL_CONFIDENCE_THRESHOLD = 10.0  # Flag if analytical vs numerical differ by >10x


# ---------------------------------------------------------------------------
#  Data Classes
# ---------------------------------------------------------------------------
class GoalType(str, Enum):
    FIND_THRESHOLD = "find_threshold"    # "find force where deflection > 50mm"
    MAXIMIZE = "maximize"                # "maximize the flow rate"
    MINIMIZE = "minimize"                # "minimize temperature"
    COMPARE_SELECT = "compare_select"    # "which design is better?"
    CHARACTERIZE = "characterize"        # "how does X behave?" (fallback)


class ConvergenceAction(str, Enum):
    EXPAND_RANGE = "expand_range"
    BISECT = "bisect"
    REFINE = "refine"
    REPORT_ANSWER = "report_answer"
    REPORT_PARTIAL = "report_partial"
    EXPAND_AND_BISECT = "expand_and_bisect"


@dataclass
class AnalysisGoal:
    """Structured representation of what the user actually wants to know."""
    goal_type: GoalType
    metric: str                          # Field or observable name to evaluate
    operator: Optional[str] = None       # ">", "<", ">=", "<=", "crosses"
    threshold: Optional[float] = None    # Target value
    unit: Optional[str] = None           # Unit of the threshold
    search_variable: Optional[str] = None  # Parameter to vary (param_path)
    search_bounds: Optional[Tuple[float, float]] = None  # Initial bounds
    module_name: Optional[str] = None    # Which module to use
    config: Optional[Dict[str, Any]] = None  # Base config
    confidence: str = "exact"            # "exact", "approximate", "order_of_magnitude"
    raw_prompt: str = ""                 # Original user prompt

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "goal_type": self.goal_type.value,
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
            "unit": self.unit,
            "search_variable": self.search_variable,
            "search_bounds": list(self.search_bounds) if self.search_bounds else None,
            "module_name": self.module_name,
            "confidence": self.confidence,
        }
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class AnalyticalEstimate:
    """Pre-flight analytical estimation before numerical simulation."""
    estimate: Optional[float] = None
    unit: Optional[str] = None
    governing_equation: Optional[str] = None
    suggested_bounds: Optional[Tuple[float, float]] = None
    confidence: str = "order_of_magnitude"  # "exact", "within_2x", "order_of_magnitude"
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate": self.estimate,
            "unit": self.unit,
            "governing_equation": self.governing_equation,
            "suggested_bounds": list(self.suggested_bounds) if self.suggested_bounds else None,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class GoalEvaluation:
    """Result of evaluating simulation results against the goal."""
    satisfied: bool
    answer: Optional[float] = None       # The answer value (e.g., 418kN)
    answer_unit: Optional[str] = None
    accuracy: Optional[str] = None       # "exact", "interpolated", "bounded"
    accuracy_pct: Optional[float] = None  # Accuracy as percentage
    closest_value: Optional[float] = None  # Closest metric value achieved
    gap_ratio: Optional[float] = None    # threshold / closest (how far off)
    action: ConvergenceAction = ConvergenceAction.REPORT_PARTIAL
    suggested_bounds: Optional[Tuple[float, float]] = None  # For next iteration
    crossing_param: Optional[float] = None  # Where the crossing occurs
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "satisfied": self.satisfied,
            "answer": self.answer,
            "answer_unit": self.answer_unit,
            "accuracy": self.accuracy,
            "accuracy_pct": self.accuracy_pct,
            "closest_value": self.closest_value,
            "gap_ratio": self.gap_ratio,
            "action": self.action.value,
            "crossing_param": self.crossing_param,
            "details": self.details,
        }
        if self.suggested_bounds:
            d["suggested_bounds"] = list(self.suggested_bounds)
        return {k: v for k, v in d.items() if v is not None}


@dataclass
class ConvergenceRecord:
    """One iteration of the convergence loop."""
    iteration: int
    action_taken: str
    bounds: Tuple[float, float]
    n_points: int
    best_metric: Optional[float] = None
    crossing_found: bool = False
    crossing_value: Optional[float] = None
    accuracy_pct: Optional[float] = None


# ---------------------------------------------------------------------------
#  Goal Extraction
# ---------------------------------------------------------------------------
GOAL_EXTRACTION_PROMPT = """Analyze the user's engineering question and extract a structured goal.

User's question: "{prompt}"

You must determine:
1. **goal_type**: What kind of answer does the user want?
   - "find_threshold": Find the parameter value where a metric crosses a threshold (e.g., "find force where deflection > 50mm")
   - "maximize": Find the parameter value that maximizes a metric (e.g., "maximize flow rate")
   - "minimize": Find the parameter value that minimizes a metric (e.g., "minimize peak temperature")
   - "compare_select": Choose the best option from alternatives (e.g., "which material is better?")
   - "characterize": Understand general behavior, no specific target (e.g., "how does viscosity affect flow?")

2. **metric**: The physical quantity to evaluate (use field names like: max_velocity, von_mises_stress, max_deflection, peak_temperature, pressure_drop, built_in_potential, detection_probability, etc.)

3. **operator**: For find_threshold: ">", "<", ">=", "<=" (which direction triggers the condition)

4. **threshold**: The numeric target value (convert to SI units: metres, Pascals, Kelvin, etc.)

5. **unit**: SI unit of the threshold

6. **search_variable**: The parameter to vary, using dot notation matching the module's config (e.g., "solver.force", "solver.nu", "solver.U_lid", "solver.temperature")

7. **search_bounds**: Initial [min, max] range for the search variable. Use physics judgment to set reasonable bounds.

8. **module_name**: Which Triality module to use (navier_stokes, drift_diffusion, sensing, structural_analysis, battery_thermal, etc.)

Respond with ONLY valid JSON:
{{
    "goal_type": "find_threshold",
    "metric": "max_deflection",
    "operator": ">",
    "threshold": 0.05,
    "unit": "m",
    "search_variable": "solver.force",
    "search_bounds": [10000, 500000],
    "module_name": "structural_analysis",
    "has_clear_goal": true,
    "reasoning": "User wants to find the force that causes >50mm deflection in a beam"
}}

If the question is open-ended with no specific target, set "has_clear_goal": false and "goal_type": "characterize".
If the user asks to maximize or minimize something, set appropriate goal_type and leave threshold as null.
"""


def extract_goal_from_llm_response(response: Dict[str, Any], prompt: str) -> AnalysisGoal:
    """Convert LLM JSON response into an AnalysisGoal."""
    goal_type_str = response.get("goal_type", "characterize")
    try:
        goal_type = GoalType(goal_type_str)
    except ValueError:
        goal_type = GoalType.CHARACTERIZE

    bounds = response.get("search_bounds")
    if bounds and isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        bounds = (float(bounds[0]), float(bounds[1]))
    else:
        bounds = None

    return AnalysisGoal(
        goal_type=goal_type,
        metric=response.get("metric", ""),
        operator=response.get("operator"),
        threshold=_safe_float(response.get("threshold")),
        unit=response.get("unit"),
        search_variable=response.get("search_variable"),
        search_bounds=bounds,
        module_name=response.get("module_name"),
        confidence="exact" if response.get("has_clear_goal") else "approximate",
        raw_prompt=prompt,
    )


def extract_goal_heuristic(prompt: str) -> Optional[AnalysisGoal]:
    """Rule-based fallback goal extraction when LLM is unavailable."""
    text = prompt.lower().strip()

    # Pattern: "find X where/when Y > Z"
    threshold_patterns = [
        r"(?:find|determine|what)\b.*?(?:where|when|at which)\b.*?(\w+)\s*([><=]+)\s*([\d.]+)\s*(\w*)",
        r"(\w+)\s*(?:exceeds?|reaches?|goes? (?:above|below|over|under))\s*([\d.]+)\s*(\w*)",
        r"(?:at what|what)\b.*?(?:force|load|pressure|temperature|voltage|current)\b.*?(\w+)\s*([><=]+)\s*([\d.]+)",
    ]
    for pattern in threshold_patterns:
        m = re.search(pattern, text)
        if m:
            return AnalysisGoal(
                goal_type=GoalType.FIND_THRESHOLD,
                metric=m.group(1) if len(m.groups()) >= 1 else "",
                operator=">" if "exceed" in text or "above" in text or "over" in text else "<",
                threshold=_safe_float(m.group(3) if len(m.groups()) >= 3 else m.group(2)),
                raw_prompt=prompt,
            )

    # Pattern: "maximize/minimize X"
    if any(w in text for w in ["maximize", "max ", "highest", "largest", "greatest"]):
        return AnalysisGoal(goal_type=GoalType.MAXIMIZE, metric="", raw_prompt=prompt)
    if any(w in text for w in ["minimize", "min ", "lowest", "smallest", "least"]):
        return AnalysisGoal(goal_type=GoalType.MINIMIZE, metric="", raw_prompt=prompt)

    # Pattern: "which is better", "compare A vs B"
    if any(w in text for w in ["which is better", "which design", "compare", "versus", " vs "]):
        return AnalysisGoal(goal_type=GoalType.COMPARE_SELECT, metric="", raw_prompt=prompt)

    return None


# ---------------------------------------------------------------------------
#  Analytical Pre-Flight Estimation
# ---------------------------------------------------------------------------
ANALYTICAL_ESTIMATION_PROMPT = """Before running numerical simulation, provide an analytical estimate.

Problem: "{prompt}"
Module: {module_name}
Goal: Find {search_variable} where {metric} {operator} {threshold} {unit}

Use first-principles physics to estimate:
1. The approximate answer (use beam theory, energy methods, dimensional analysis, Bernoulli, Ohm's law, or whatever applies)
2. The key governing equation
3. Reasonable initial search bounds for the numerical solver

Think step by step through the physics, then provide your estimate.

Respond with ONLY valid JSON:
{{
    "estimate": <number>,
    "unit": "<unit>",
    "governing_equation": "<the key equation used>",
    "suggested_bounds": [<lower>, <upper>],
    "confidence": "within_2x",
    "reasoning": "<step-by-step physics reasoning>"
}}

If you cannot estimate analytically, set estimate to null and explain why in reasoning.
"""


def parse_analytical_estimate(response: Dict[str, Any]) -> AnalyticalEstimate:
    """Convert LLM response into AnalyticalEstimate."""
    bounds = response.get("suggested_bounds")
    if bounds and isinstance(bounds, (list, tuple)) and len(bounds) == 2:
        bounds = (float(bounds[0]), float(bounds[1]))
    else:
        bounds = None

    return AnalyticalEstimate(
        estimate=_safe_float(response.get("estimate")),
        unit=response.get("unit"),
        governing_equation=response.get("governing_equation"),
        suggested_bounds=bounds,
        confidence=response.get("confidence", "order_of_magnitude"),
        reasoning=response.get("reasoning", ""),
    )


# ---------------------------------------------------------------------------
#  Goal Evaluation
# ---------------------------------------------------------------------------
class GoalEvaluator:
    """Evaluates simulation results against the analysis goal."""

    @staticmethod
    def evaluate(
        goal: AnalysisGoal,
        results: List[Dict[str, Any]],
        iteration: int = 0,
    ) -> GoalEvaluation:
        """Check if results satisfy the goal. Returns evaluation with next action."""
        if goal.goal_type == GoalType.FIND_THRESHOLD:
            return GoalEvaluator._evaluate_threshold(goal, results, iteration)
        elif goal.goal_type in (GoalType.MAXIMIZE, GoalType.MINIMIZE):
            return GoalEvaluator._evaluate_optimization(goal, results, iteration)
        elif goal.goal_type == GoalType.COMPARE_SELECT:
            return GoalEvaluator._evaluate_comparison(goal, results, iteration)
        else:
            # CHARACTERIZE — single pass is sufficient
            return GoalEvaluation(
                satisfied=True,
                action=ConvergenceAction.REPORT_ANSWER,
                details="Characterization complete — single-pass analysis.",
            )

    @staticmethod
    def _evaluate_threshold(
        goal: AnalysisGoal,
        results: List[Dict[str, Any]],
        iteration: int,
    ) -> GoalEvaluation:
        """Evaluate find_threshold goal against sweep results."""
        sweep_data = _extract_sweep_data(results, goal.metric)
        if not sweep_data:
            return GoalEvaluation(
                satisfied=False,
                action=ConvergenceAction.REPORT_PARTIAL,
                details="No sweep data found matching metric.",
            )

        param_values, metric_values = sweep_data
        if not param_values or not metric_values:
            return GoalEvaluation(
                satisfied=False,
                action=ConvergenceAction.REPORT_PARTIAL,
                details="Empty sweep results.",
            )

        threshold = goal.threshold
        operator = goal.operator or ">"

        # Check if any point crosses the threshold
        crossed_indices = []
        for i, val in enumerate(metric_values):
            if _check_condition(val, operator, threshold):
                crossed_indices.append(i)

        if crossed_indices:
            # Threshold was crossed — find the crossing point
            first_cross_idx = crossed_indices[0]

            if first_cross_idx == 0:
                # Crossed at the very first point — all values exceed threshold
                # Need to search lower
                crossing_param = param_values[0]
                return GoalEvaluation(
                    satisfied=False,
                    answer=crossing_param,
                    closest_value=metric_values[0],
                    action=ConvergenceAction.EXPAND_RANGE,
                    suggested_bounds=(
                        param_values[0] / EXPANSION_FACTOR,
                        param_values[0],
                    ),
                    details=f"Threshold already exceeded at lowest search value ({param_values[0]:.4g}). Expanding search downward.",
                )

            # Interpolate between the point before and after crossing
            idx_before = first_cross_idx - 1
            idx_after = first_cross_idx

            p_before = param_values[idx_before]
            p_after = param_values[idx_after]
            m_before = metric_values[idx_before]
            m_after = metric_values[idx_after]

            # Linear interpolation for crossing point
            if abs(m_after - m_before) > 1e-30:
                frac = (threshold - m_before) / (m_after - m_before)
                crossing_param = p_before + frac * (p_after - p_before)
            else:
                crossing_param = (p_before + p_after) / 2.0

            # Calculate accuracy (how narrow is the bracket?)
            bracket_width = abs(p_after - p_before)
            bracket_center = abs(crossing_param) if abs(crossing_param) > 1e-30 else 1.0
            accuracy_pct = (bracket_width / bracket_center) * 100.0

            if accuracy_pct <= CROSSING_TOLERANCE_PERCENT:
                # Converged — report the answer
                return GoalEvaluation(
                    satisfied=True,
                    answer=crossing_param,
                    answer_unit=goal.unit,
                    accuracy="interpolated",
                    accuracy_pct=accuracy_pct,
                    crossing_param=crossing_param,
                    closest_value=metric_values[first_cross_idx],
                    action=ConvergenceAction.REPORT_ANSWER,
                    details=f"Threshold crossing found at {crossing_param:.6g}. Bracket width: {accuracy_pct:.2f}%.",
                )
            else:
                # Need to refine — bisect around the crossing
                margin = bracket_width * 0.2  # 20% padding
                return GoalEvaluation(
                    satisfied=False,
                    answer=crossing_param,
                    accuracy="coarse",
                    accuracy_pct=accuracy_pct,
                    crossing_param=crossing_param,
                    closest_value=metric_values[first_cross_idx],
                    action=ConvergenceAction.BISECT,
                    suggested_bounds=(
                        p_before - margin,
                        p_after + margin,
                    ),
                    details=f"Crossing found between {p_before:.4g} and {p_after:.4g}. Refining (accuracy: {accuracy_pct:.1f}%).",
                )
        else:
            # Threshold NOT crossed — need to expand range
            closest_value = _closest_to_threshold(metric_values, threshold, operator)
            max_param = max(param_values)
            min_param = min(param_values)

            if threshold is not None and closest_value is not None and abs(closest_value) > 1e-30:
                gap_ratio = abs(threshold / closest_value)
            else:
                gap_ratio = float("inf")

            # Determine expansion direction
            if operator in (">", ">="):
                # Need higher values — expand upward
                new_lower = max_param
                new_upper = max_param * EXPANSION_FACTOR
                if gap_ratio > 5:
                    # Very far off — more aggressive expansion
                    new_upper = max_param * (EXPANSION_FACTOR ** 2)
            else:
                # Need lower values — expand downward
                new_upper = min_param
                new_lower = min_param / EXPANSION_FACTOR
                if gap_ratio > 5:
                    new_lower = min_param / (EXPANSION_FACTOR ** 2)

            return GoalEvaluation(
                satisfied=False,
                closest_value=closest_value,
                gap_ratio=gap_ratio,
                action=ConvergenceAction.EXPAND_RANGE,
                suggested_bounds=(new_lower, new_upper),
                details=(
                    f"Threshold {threshold:.4g} not reached. "
                    f"Closest: {closest_value:.4g} (ratio: {gap_ratio:.2f}x). "
                    f"Expanding search to [{new_lower:.4g}, {new_upper:.4g}]."
                ),
            )

    @staticmethod
    def _evaluate_optimization(
        goal: AnalysisGoal,
        results: List[Dict[str, Any]],
        iteration: int,
    ) -> GoalEvaluation:
        """Evaluate maximize/minimize goal — check for boundary optima."""
        # Look for optimization results
        for r in results:
            if r.get("_tool_name") == "optimize_parameter":
                optimal = r.get("optimal_param_value")
                bounds = r.get("bounds", [])
                evaluations = r.get("evaluations", [])

                if optimal is not None and bounds and len(bounds) == 2:
                    # Check if optimum is at boundary
                    range_width = abs(bounds[1] - bounds[0])
                    at_lower = abs(optimal - bounds[0]) < 0.05 * range_width
                    at_upper = abs(optimal - bounds[1]) < 0.05 * range_width

                    if at_lower or at_upper:
                        # Boundary optimum — expand in that direction
                        if at_lower:
                            new_bounds = (bounds[0] - range_width, bounds[0])
                        else:
                            new_bounds = (bounds[1], bounds[1] + range_width)

                        return GoalEvaluation(
                            satisfied=False,
                            answer=optimal,
                            closest_value=r.get("optimal_metric"),
                            action=ConvergenceAction.EXPAND_RANGE,
                            suggested_bounds=new_bounds,
                            details=f"Optimum at boundary ({optimal:.4g}). Expanding search range.",
                        )
                    else:
                        return GoalEvaluation(
                            satisfied=True,
                            answer=optimal,
                            answer_unit=goal.unit,
                            accuracy="golden_section",
                            closest_value=r.get("optimal_metric"),
                            action=ConvergenceAction.REPORT_ANSWER,
                            details=f"Optimum found at {optimal:.6g} (interior point).",
                        )

        # Look for sweep results to check for monotonic trends
        sweep_data = _extract_sweep_data(results, goal.metric)
        if sweep_data:
            param_values, metric_values = sweep_data
            if len(metric_values) >= 3:
                is_monotonic, direction = _check_monotonicity(metric_values)
                if is_monotonic:
                    max_param = max(param_values)
                    min_param = min(param_values)
                    range_w = max_param - min_param

                    if (goal.goal_type == GoalType.MAXIMIZE and direction == "increasing") or \
                       (goal.goal_type == GoalType.MINIMIZE and direction == "decreasing"):
                        # Optimum is beyond upper bound
                        return GoalEvaluation(
                            satisfied=False,
                            closest_value=metric_values[-1],
                            action=ConvergenceAction.EXPAND_RANGE,
                            suggested_bounds=(max_param, max_param + range_w),
                            details=f"Metric is {direction} — optimum beyond search range. Expanding upward.",
                        )
                    else:
                        return GoalEvaluation(
                            satisfied=False,
                            closest_value=metric_values[0],
                            action=ConvergenceAction.EXPAND_RANGE,
                            suggested_bounds=(min_param - range_w, min_param),
                            details=f"Metric is {direction} — optimum beyond search range. Expanding downward.",
                        )

        return GoalEvaluation(
            satisfied=True,
            action=ConvergenceAction.REPORT_ANSWER,
            details="Optimization evaluation complete.",
        )

    @staticmethod
    def _evaluate_comparison(
        goal: AnalysisGoal,
        results: List[Dict[str, Any]],
        iteration: int,
    ) -> GoalEvaluation:
        """Evaluate compare_select — always satisfied after one comparison run."""
        return GoalEvaluation(
            satisfied=True,
            action=ConvergenceAction.REPORT_ANSWER,
            details="Comparison complete — all scenarios evaluated.",
        )


# ---------------------------------------------------------------------------
#  Convergence Strategy
# ---------------------------------------------------------------------------
class ConvergenceStrategy:
    """Generates adapted tool calls based on goal evaluation results."""

    @staticmethod
    def build_next_plan(
        goal: AnalysisGoal,
        evaluation: GoalEvaluation,
        iteration: int,
        all_results: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Build the next set of tool calls to get closer to the goal.

        Returns None if no further action is needed.
        """
        if evaluation.satisfied:
            return None

        action = evaluation.action

        if action == ConvergenceAction.EXPAND_RANGE:
            return ConvergenceStrategy._build_expansion(goal, evaluation, iteration)
        elif action == ConvergenceAction.BISECT:
            return ConvergenceStrategy._build_bisection(goal, evaluation, iteration)
        elif action == ConvergenceAction.REFINE:
            return ConvergenceStrategy._build_refinement(goal, evaluation, iteration)
        elif action == ConvergenceAction.EXPAND_AND_BISECT:
            return ConvergenceStrategy._build_expansion(goal, evaluation, iteration)
        else:
            return None

    @staticmethod
    def _build_expansion(
        goal: AnalysisGoal,
        evaluation: GoalEvaluation,
        iteration: int,
    ) -> List[Dict[str, Any]]:
        """Expand search range when threshold/optimum not found."""
        bounds = evaluation.suggested_bounds
        if not bounds:
            return []

        # More points on first expansion, fewer on subsequent
        n_points = max(MIN_BISECTION_POINTS, 10 - iteration)

        if goal.goal_type in (GoalType.MAXIMIZE, GoalType.MINIMIZE):
            return [{
                "tool": "optimize_parameter",
                "args": {
                    "module_name": goal.module_name,
                    "config": goal.config or {},
                    "param_path": goal.search_variable,
                    "bounds": list(bounds),
                    "objective_field": goal.metric,
                    "objective": "maximize" if goal.goal_type == GoalType.MAXIMIZE else "minimize",
                    "n_evals": n_points,
                },
            }]

        return [{
            "tool": "sweep_parameter",
            "args": {
                "module_name": goal.module_name,
                "config": goal.config or {},
                "param_path": goal.search_variable,
                "start": bounds[0],
                "end": bounds[1],
                "steps": n_points,
            },
        }]

    @staticmethod
    def _build_bisection(
        goal: AnalysisGoal,
        evaluation: GoalEvaluation,
        iteration: int,
    ) -> List[Dict[str, Any]]:
        """Bisect around a known crossing point for precision."""
        bounds = evaluation.suggested_bounds
        if not bounds:
            return []

        n_points = max(MIN_BISECTION_POINTS, 10 - iteration)

        return [{
            "tool": "sweep_parameter",
            "args": {
                "module_name": goal.module_name,
                "config": goal.config or {},
                "param_path": goal.search_variable,
                "start": bounds[0],
                "end": bounds[1],
                "steps": n_points,
            },
        }]

    @staticmethod
    def _build_refinement(
        goal: AnalysisGoal,
        evaluation: GoalEvaluation,
        iteration: int,
    ) -> List[Dict[str, Any]]:
        """Fine-grained refinement around a known answer."""
        if evaluation.crossing_param is None:
            return []

        # Very tight sweep around the known crossing
        width = abs(evaluation.crossing_param) * 0.02  # ±1% of value
        bounds = (evaluation.crossing_param - width, evaluation.crossing_param + width)

        return [{
            "tool": "sweep_parameter",
            "args": {
                "module_name": goal.module_name,
                "config": goal.config or {},
                "param_path": goal.search_variable,
                "start": bounds[0],
                "end": bounds[1],
                "steps": MIN_BISECTION_POINTS,
            },
        }]


# ---------------------------------------------------------------------------
#  Goal-Driven Runner
# ---------------------------------------------------------------------------
class GoalDrivenRunner:
    """Orchestrates the iterative goal-convergence loop.

    This is the core engine that transforms Triality from a single-pass
    analysis tool into a goal-convergent problem solver.

    Usage:
        runner = GoalDrivenRunner(
            goal=goal,
            initial_plan=plan,
            execute_fn=execute_tools,
            llm_fn=call_llm,
        )
        for event in runner.run():
            yield event  # SSE events
    """

    def __init__(
        self,
        goal: AnalysisGoal,
        initial_plan: Dict[str, Any],
        execute_fn: Callable,
        sse_fn: Callable,
        insights_fn: Callable,
        analytical_estimate: Optional[AnalyticalEstimate] = None,
    ):
        self.goal = goal
        self.initial_plan = initial_plan
        self.execute_fn = execute_fn
        self.sse = sse_fn
        self.build_insights = insights_fn
        self.analytical = analytical_estimate
        self.all_results: List[Dict[str, Any]] = []
        self.convergence_history: List[ConvergenceRecord] = []
        self.total_solver_runs = 0

    def run(self) -> Generator[str, None, Tuple[List[Dict], str, GoalEvaluation]]:
        """Execute the goal-driven convergence loop.

        Yields SSE events. Returns (all_results, convergence_summary, final_evaluation).
        """
        # Step 1: Execute the initial plan
        yield self.sse("phase", {
            "phase": "goal_converging",
            "message": f"Goal: {self.goal.goal_type.value} — {self.goal.metric}",
            "goal": self.goal.to_dict(),
        })

        if self.analytical:
            yield self.sse("analytical_estimate", self.analytical.to_dict())

        # Execute initial tool calls
        initial_results = yield from self._execute_plan(
            self.initial_plan.get("tool_calls", []),
            iteration=0,
        )
        self.all_results.extend(initial_results)

        # Step 2: Evaluate and iterate
        final_eval = None
        for iteration in range(1, MAX_CONVERGENCE_ITERATIONS + 1):
            evaluation = GoalEvaluator.evaluate(self.goal, self.all_results, iteration)
            final_eval = evaluation

            # Record convergence history
            bounds = evaluation.suggested_bounds or self.goal.search_bounds or (0, 0)
            self.convergence_history.append(ConvergenceRecord(
                iteration=iteration,
                action_taken=evaluation.action.value,
                bounds=bounds,
                n_points=0,
                best_metric=evaluation.closest_value,
                crossing_found=evaluation.satisfied,
                crossing_value=evaluation.answer,
                accuracy_pct=evaluation.accuracy_pct,
            ))

            yield self.sse("convergence_step", {
                "iteration": iteration,
                "max_iterations": MAX_CONVERGENCE_ITERATIONS,
                "evaluation": evaluation.to_dict(),
                "total_solver_runs": self.total_solver_runs,
            })

            if evaluation.satisfied:
                yield self.sse("goal_satisfied", {
                    "answer": evaluation.answer,
                    "answer_unit": evaluation.answer_unit,
                    "accuracy": evaluation.accuracy,
                    "accuracy_pct": evaluation.accuracy_pct,
                    "iterations_used": iteration,
                    "total_solver_runs": self.total_solver_runs,
                    "details": evaluation.details,
                })
                break

            # Build next plan
            next_tools = ConvergenceStrategy.build_next_plan(
                self.goal, evaluation, iteration, self.all_results,
            )

            if not next_tools:
                yield self.sse("convergence_stalled", {
                    "iteration": iteration,
                    "reason": "No further adaptation possible.",
                    "best_so_far": evaluation.to_dict(),
                })
                break

            yield self.sse("convergence_adapting", {
                "iteration": iteration,
                "action": evaluation.action.value,
                "details": evaluation.details,
                "next_bounds": list(evaluation.suggested_bounds) if evaluation.suggested_bounds else None,
            })

            # Execute adapted plan
            iter_results = yield from self._execute_plan(next_tools, iteration=iteration)
            self.all_results.extend(iter_results)
        else:
            # Exhausted iterations without convergence
            if final_eval is None:
                final_eval = GoalEvaluation(
                    satisfied=False,
                    action=ConvergenceAction.REPORT_PARTIAL,
                    details="Maximum iterations reached.",
                )
            yield self.sse("convergence_exhausted", {
                "iterations_used": MAX_CONVERGENCE_ITERATIONS,
                "total_solver_runs": self.total_solver_runs,
                "best_so_far": final_eval.to_dict() if final_eval else {},
            })

        # Build convergence summary
        summary = self._build_convergence_summary(final_eval)

        return self.all_results, summary, final_eval

    def _execute_plan(
        self,
        tool_calls: List[Dict[str, Any]],
        iteration: int,
    ) -> Generator[str, None, List[Dict]]:
        """Execute a list of tool calls, yielding SSE events. Returns results."""
        results = []
        for tc in tool_calls:
            tool_name = tc.get("tool", "")
            args = tc.get("args", {})
            idx = len(self.all_results) + len(results)

            yield self.sse("tool_start", {
                "index": idx,
                "tool": tool_name,
                "args": args,
                "convergence_iteration": iteration,
            })

            # Execute — function may return 2 or 3 values
            exec_result = self.execute_fn(tool_name, args, idx)
            if len(exec_result) == 3:
                result, elapsed, progress_events = exec_result
                # Emit collected progress events
                for pe in progress_events:
                    yield self.sse("tool_progress", pe)
            else:
                result, elapsed = exec_result

            # Count solver runs
            if tool_name == "sweep_parameter":
                n_steps = args.get("steps", len(args.get("values", [])))
                self.total_solver_runs += n_steps
            elif tool_name == "optimize_parameter":
                self.total_solver_runs += args.get("n_evals", 10) * 2 + 1
            elif tool_name == "run_module":
                self.total_solver_runs += 1

            result["_tool_name"] = tool_name
            result["_elapsed_s"] = elapsed
            result["_index"] = idx
            result["_convergence_iteration"] = iteration

            results.append(result)

            yield self.sse("tool_result", {
                "index": idx,
                "tool": tool_name,
                "elapsed_s": round(elapsed, 3),
                "success": not result.get("error"),
                "result": result,
                "convergence_iteration": iteration,
            })

        return results

    def _build_convergence_summary(self, final_eval: Optional[GoalEvaluation]) -> str:
        """Build a structured summary of the convergence process."""
        lines = []
        lines.append("## Convergence Summary\n")

        if final_eval and final_eval.satisfied:
            lines.append(f"**Goal achieved** in {len(self.convergence_history)} iteration(s), "
                         f"{self.total_solver_runs} solver runs.\n")
            if final_eval.answer is not None:
                unit = final_eval.answer_unit or self.goal.unit or ""
                lines.append(f"**Answer: {final_eval.answer:.6g} {unit}**")
                if final_eval.accuracy_pct is not None:
                    lines.append(f"  (accuracy: ±{final_eval.accuracy_pct:.2f}%)")
        else:
            lines.append(f"**Goal not fully achieved** after {len(self.convergence_history)} iteration(s), "
                         f"{self.total_solver_runs} solver runs.\n")
            if final_eval and final_eval.closest_value is not None:
                lines.append(f"Closest value: {final_eval.closest_value:.6g}")
            if final_eval and final_eval.answer is not None:
                lines.append(f"Best estimate: {final_eval.answer:.6g}")

        # Analytical comparison
        if self.analytical and self.analytical.estimate is not None and final_eval and final_eval.answer is not None:
            ratio = final_eval.answer / self.analytical.estimate if abs(self.analytical.estimate) > 1e-30 else float("inf")
            lines.append(f"\n**Analytical vs Numerical:**")
            lines.append(f"  Analytical estimate: {self.analytical.estimate:.4g}")
            lines.append(f"  Numerical result: {final_eval.answer:.4g}")
            lines.append(f"  Ratio: {ratio:.2f}x")
            if abs(ratio) > ANALYTICAL_CONFIDENCE_THRESHOLD or abs(ratio) < 1.0 / ANALYTICAL_CONFIDENCE_THRESHOLD:
                lines.append(f"  ⚠ Large discrepancy — verify problem setup and assumptions.")

        # Iteration log
        if self.convergence_history:
            lines.append("\n**Iteration Log:**")
            for rec in self.convergence_history:
                status = "✓ converged" if rec.crossing_found else rec.action_taken
                accuracy = f" (±{rec.accuracy_pct:.1f}%)" if rec.accuracy_pct else ""
                lines.append(
                    f"  {rec.iteration}. [{rec.bounds[0]:.4g}, {rec.bounds[1]:.4g}] "
                    f"→ {status}{accuracy}"
                )

        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Helper Functions
# ---------------------------------------------------------------------------
def _safe_float(val: Any) -> Optional[float]:
    """Convert to float safely, returning None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _check_condition(value: float, operator: str, threshold: float) -> bool:
    """Check if value satisfies the condition defined by operator and threshold."""
    if operator == ">":
        return value > threshold
    elif operator == ">=":
        return value >= threshold
    elif operator == "<":
        return value < threshold
    elif operator == "<=":
        return value <= threshold
    elif operator == "==" or operator == "crosses":
        return abs(value - threshold) < abs(threshold) * 0.01
    return False


def _closest_to_threshold(
    values: List[float],
    threshold: float,
    operator: str,
) -> Optional[float]:
    """Find the value closest to crossing the threshold in the right direction."""
    if not values:
        return None
    if operator in (">", ">="):
        return max(values)
    elif operator in ("<", "<="):
        return min(values)
    return min(values, key=lambda v: abs(v - threshold))


def _extract_sweep_data(
    results: List[Dict[str, Any]],
    metric_name: str,
) -> Optional[Tuple[List[float], List[float]]]:
    """Extract (param_values, metric_values) from ALL sweep results.

    Aggregates across multiple sweep_parameter tool outputs (from different
    convergence iterations) and returns combined, sorted data. Searches for
    the metric in order:
    1. Observables (exact or partial name match)
    2. Field statistics (max value)
    3. Metadata
    """
    all_param_values = []
    all_metric_values = []

    for r in results:
        tool = r.get("_tool_name", "")
        if tool != "sweep_parameter":
            continue

        sweep_results = r.get("results", [])
        if not sweep_results:
            continue

        metric_lower = metric_name.lower().replace(" ", "_")

        for sr in sweep_results:
            if not sr.get("success", False):
                continue

            pv = sr.get("param_value")
            if pv is None:
                continue

            # Search 1: Observables
            obs = sr.get("observables", {})
            found = False
            for obs_name, obs_data in obs.items():
                if metric_lower in obs_name.lower() or obs_name.lower() in metric_lower:
                    val = obs_data.get("value") if isinstance(obs_data, dict) else None
                    if val is not None and isinstance(val, (int, float)):
                        all_param_values.append(float(pv))
                        all_metric_values.append(float(val))
                        found = True
                        break

            if found:
                continue

            # Search 2: Field stats (use max value)
            fields = sr.get("fields", {})
            for fname, fstats in fields.items():
                if metric_lower in fname.lower() or fname.lower() in metric_lower:
                    # Use max of absolute value for deflection/stress type metrics
                    val = fstats.get("max", fstats.get("mean", 0.0))
                    if "deflect" in metric_lower or "stress" in metric_lower or "displace" in metric_lower:
                        val = max(abs(fstats.get("min", 0.0)), abs(fstats.get("max", 0.0)))
                    all_param_values.append(float(pv))
                    all_metric_values.append(float(val))
                    found = True
                    break

            if found:
                continue

            # Search 3: Metadata
            metadata = sr.get("metadata", {})
            for mk, mv in metadata.items():
                if metric_lower in mk.lower() and isinstance(mv, (int, float)):
                    all_param_values.append(float(pv))
                    all_metric_values.append(float(mv))
                    break

    if not all_param_values:
        return None

    # Sort by parameter value for correct interpolation
    paired = sorted(zip(all_param_values, all_metric_values), key=lambda x: x[0])
    # Deduplicate — keep the latest value for duplicate param_values
    seen = {}
    for p, m in paired:
        seen[p] = m
    sorted_pairs = sorted(seen.items())
    return [p for p, _ in sorted_pairs], [m for _, m in sorted_pairs]


def _check_monotonicity(values: List[float]) -> Tuple[bool, str]:
    """Check if a sequence is monotonically increasing or decreasing."""
    if len(values) < 2:
        return False, "unknown"

    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    n_positive = sum(1 for d in diffs if d > 0)
    n_negative = sum(1 for d in diffs if d < 0)
    total = len(diffs)

    # Allow 10% non-monotonic points (noise)
    threshold = 0.9 * total
    if n_positive >= threshold:
        return True, "increasing"
    elif n_negative >= threshold:
        return True, "decreasing"
    return False, "non_monotonic"


# ---------------------------------------------------------------------------
#  Convergence Detection & Answer Commitment
# ---------------------------------------------------------------------------
BoundarySide = str  # "lower", "upper", "none"


@dataclass
class SearchState:
    """Tracks the state of an iterative search for convergence detection."""
    history: List[Tuple[float, float]] = field(default_factory=list)  # (best_x, best_y)
    current_bounds: Tuple[float, float] = (0.0, 1.0)
    objective: str = "maximize"  # "maximize" or "minimize"
    boundary_hits: int = 0
    last_boundary_side: BoundarySide = "none"
    repeated_best_count: int = 0
    small_improvement_count: int = 0


def detect_boundary(best_x: float, bounds: Tuple[float, float], frac: float = 0.05) -> BoundarySide:
    """Check if best_x is near the boundary of the search range."""
    lo, hi = bounds
    width = abs(hi - lo)
    if width < 1e-30:
        return "none"
    eps = width * frac
    if best_x <= lo + eps:
        return "lower"
    if best_x >= hi - eps:
        return "upper"
    return "none"


def is_stable_best(prev_x: float, curr_x: float, bounds: Tuple[float, float], rel_tol: float = 0.02) -> bool:
    """Check if the best point has stabilized (not moving meaningfully)."""
    width = abs(bounds[1] - bounds[0])
    if width < 1e-30:
        return True
    return abs(curr_x - prev_x) <= width * rel_tol


def improvement_ratio(prev_y: float, curr_y: float, objective: str) -> float:
    """Compute relative improvement between iterations."""
    denom = max(abs(prev_y), 1e-12)
    if objective == "maximize":
        return (curr_y - prev_y) / denom
    return (prev_y - curr_y) / denom


def check_convergence(state: SearchState, best_x: float, best_y: float) -> Dict[str, Any]:
    """Run all convergence checks and return a decision.

    Returns dict with:
        converged: bool
        status: str ("converged_interior", "converged_boundary_plateau", "expand", "continue", "max_reached")
        details: str
    """
    bounds = state.current_bounds
    boundary_side = detect_boundary(best_x, bounds)

    # Update stability tracking
    if state.history:
        prev_x, prev_y = state.history[-1]
        stable = is_stable_best(prev_x, best_x, bounds)
        imp = improvement_ratio(prev_y, best_y, state.objective)

        if stable:
            state.repeated_best_count += 1
        else:
            state.repeated_best_count = 0

        if imp < 0.005:
            state.small_improvement_count += 1
        else:
            state.small_improvement_count = 0
    else:
        imp = float("inf")

    state.history.append((best_x, best_y))

    # Rule A: Interior convergence
    if (boundary_side == "none"
            and state.repeated_best_count >= 2
            and state.small_improvement_count >= 2):
        return {
            "converged": True,
            "status": "converged_interior",
            "details": f"Interior optimum at {best_x:.6g} (stable for {state.repeated_best_count} cycles, "
                       f"<0.5% improvement for {state.small_improvement_count} cycles)",
        }

    # Track boundary hits
    same_boundary = (boundary_side == state.last_boundary_side and boundary_side != "none")
    if boundary_side != "none":
        state.boundary_hits += 1

    # Rule C: Boundary plateau (THIS is the key missing rule)
    if same_boundary and state.boundary_hits >= 2 and state.small_improvement_count >= 2:
        return {
            "converged": True,
            "status": "converged_boundary_plateau",
            "details": f"Boundary plateau at {best_x:.6g} — repeated expansions showed "
                       f"<0.5% improvement. This is the practical optimum.",
            "boundary_side": boundary_side,
        }

    # Rule B: Should expand
    if boundary_side != "none" and (state.boundary_hits <= 1 or imp > 0.005):
        state.last_boundary_side = boundary_side
        return {
            "converged": False,
            "status": "expand",
            "details": f"Best at {boundary_side} boundary ({best_x:.6g}). Expanding search range.",
            "boundary_side": boundary_side,
        }

    # Stable but at boundary with no improvement — practical stop
    if boundary_side != "none":
        return {
            "converged": True,
            "status": "converged_boundary_plateau",
            "details": f"Best at {best_x:.6g} near {boundary_side} boundary. "
                       f"No meaningful improvement after {state.boundary_hits} expansions.",
            "boundary_side": boundary_side,
        }

    return {
        "converged": False,
        "status": "continue",
        "details": f"Best at {best_x:.6g}, still improving. Continuing search.",
    }


def expand_bounds(bounds: Tuple[float, float], side: BoundarySide, factor: float = 1.5) -> Tuple[float, float]:
    """Expand search bounds in the direction of the boundary hit."""
    lo, hi = bounds
    width = hi - lo
    if side == "upper":
        return (lo, hi + width * (factor - 1.0))
    elif side == "lower":
        return (max(1e-10, lo - width * (factor - 1.0)), hi)  # Don't go below 0 for physical params
    return bounds


# ---------------------------------------------------------------------------
#  Deterministic Answer Commitment
# ---------------------------------------------------------------------------
def compute_confidence(
    converged: bool,
    boundary_plateau: bool = False,
    validation_available: bool = False,
    grid_ok: bool = False,
    metric_consistent: bool = True,
) -> str:
    """Compute deterministic confidence level."""
    score = 0
    if converged:
        score += 2
    if boundary_plateau:
        score += 1
    if validation_available:
        score += 2
    if grid_ok:
        score += 1
    if metric_consistent:
        score += 1
    if score >= 6:
        return "high"
    if score >= 4:
        return "moderate"
    return "low"


def commit_answer_optimum(
    param_name: str,
    best_x: float,
    objective_name: str,
    best_y: float,
    status: str,
    confidence: str,
    caveats: Optional[str] = None,
) -> str:
    """Generate deterministic answer text for optimization results."""
    if status == "converged_interior":
        core = (f"**Final answer:** the optimal {param_name} is **{best_x:.6g}**, "
                f"achieving {objective_name} = {best_y:.6g}.")
    elif status == "converged_boundary_plateau":
        core = (f"**Final answer:** the best {param_name} found is approximately **{best_x:.6g}**, "
                f"with {objective_name} = {best_y:.6g}. The optimum sits near the search boundary, "
                f"but repeated expansions showed negligible improvement — this is the practical optimum.")
    elif status == "max_cycles_reached":
        core = (f"**Final answer:** the current best {param_name} is **{best_x:.6g}**, "
                f"giving {objective_name} = {best_y:.6g}. The search did not fully converge "
                f"within the iteration budget.")
    else:
        core = (f"**Final answer:** the best {param_name} identified is **{best_x:.6g}**, "
                f"with {objective_name} = {best_y:.6g}.")

    core += f" Confidence: {confidence}."
    if caveats:
        core += f" Caveats: {caveats}"
    return core


def commit_answer_threshold(
    metric_name: str,
    threshold: float,
    crossing_found: bool,
    crossing_x: Optional[float] = None,
    tested_range: Optional[Tuple[float, float]] = None,
    extrapolated: Optional[float] = None,
    confidence: str = "moderate",
    caveats: Optional[str] = None,
) -> str:
    """Generate deterministic answer text for threshold-finding results."""
    if crossing_found and crossing_x is not None:
        core = (f"**Final answer:** {metric_name} crosses the threshold of {threshold:.4g} "
                f"at approximately **{crossing_x:.6g}**.")
    elif tested_range:
        core = (f"**Final answer:** {metric_name} did not cross the threshold of {threshold:.4g} "
                f"within the tested range [{tested_range[0]:.4g}, {tested_range[1]:.4g}].")
        if extrapolated is not None:
            core += f" Based on the observed trend, the crossing is estimated beyond approximately {extrapolated:.4g}."
    else:
        core = f"**Final answer:** threshold analysis for {metric_name} could not determine the crossing point."

    core += f" Confidence: {confidence}."
    if caveats:
        core += f" Caveats: {caveats}"
    return core


def commit_answer_comparison(
    cases: List[Tuple[str, float]],
    metric_name: str,
    objective: str = "maximize",
    confidence: str = "moderate",
) -> str:
    """Generate deterministic answer text for comparison results."""
    if not cases:
        return "**Final answer:** no comparison data available."

    sorted_cases = sorted(cases, key=lambda c: c[1], reverse=(objective == "maximize"))
    winner_name, winner_val = sorted_cases[0]
    runner_name, runner_val = sorted_cases[1] if len(sorted_cases) > 1 else ("N/A", 0)

    gap = abs(winner_val - runner_val)
    return (f"**Final answer:** {winner_name} performs best on {metric_name} "
            f"({winner_val:.4g} vs {runner_name}: {runner_val:.4g}, "
            f"difference: {gap:.4g}). Confidence: {confidence}.")


# ---------------------------------------------------------------------------
#  Post-Execution Goal Detection
# ---------------------------------------------------------------------------
def infer_goal_from_plan(plan: Dict[str, Any], prompt: str) -> Optional[AnalysisGoal]:
    """Infer a convergible goal from the LLM's tool_calls when explicit extraction fails.

    Detects patterns like:
    - sweep_parameter → could be a threshold search or characterization
    - optimize_parameter → explicit maximize/minimize goal
    - compare_scenarios → compare & select
    """
    tool_calls = plan.get("tool_calls", [])
    if not tool_calls:
        return None

    for tc in tool_calls:
        tool = tc.get("tool", "")
        args = tc.get("args", {})

        if tool == "optimize_parameter":
            objective = args.get("objective", "maximize")
            goal_type = GoalType.MAXIMIZE if objective == "maximize" else GoalType.MINIMIZE
            bounds = args.get("bounds", [])
            return AnalysisGoal(
                goal_type=goal_type,
                metric=args.get("objective_field", ""),
                search_variable=args.get("param_path", ""),
                search_bounds=tuple(bounds) if len(bounds) == 2 else None,
                module_name=args.get("module_name"),
                config=args.get("config"),
                raw_prompt=prompt,
            )

        if tool == "compare_scenarios":
            return AnalysisGoal(
                goal_type=GoalType.COMPARE_SELECT,
                metric="",
                module_name=args.get("module_name"),
                raw_prompt=prompt,
            )

    return None


@dataclass
class UnresolvedFinding:
    """A finding from analysis results that warrants follow-up convergence."""
    finding_type: str    # "boundary_optimum", "monotonic_trend", "threshold_near_miss", "high_sensitivity"
    metric: str          # observable or field name
    current_value: float
    param_path: str      # the parameter that was swept
    module_name: str
    config: Dict[str, Any]
    suggested_action: ConvergenceAction
    suggested_bounds: Optional[Tuple[float, float]] = None
    details: str = ""


def detect_unresolved_findings(results: List[Dict[str, Any]]) -> List[UnresolvedFinding]:
    """Scan results for unresolved patterns that the convergence engine should chase.

    Detects:
    1. Boundary optima in optimization results
    2. Monotonic trends at sweep edges (optimum beyond range)
    3. Observable thresholds nearly crossed (within 20%)
    4. High sensitivity fields that need refinement
    """
    findings: List[UnresolvedFinding] = []

    for r in results:
        tool = r.get("_tool_name", "")
        module = r.get("module_name", "")

        if tool == "optimize_parameter":
            optimal = r.get("optimal_param_value")
            bounds = r.get("bounds", [])
            if optimal is not None and bounds and len(bounds) == 2:
                span = abs(bounds[1] - bounds[0])
                if span > 0:
                    at_lower = abs(optimal - bounds[0]) < 0.05 * span
                    at_upper = abs(optimal - bounds[1]) < 0.05 * span
                    if at_lower or at_upper:
                        if at_lower:
                            new_bounds = (bounds[0] - span, bounds[0])
                        else:
                            new_bounds = (bounds[1], bounds[1] + span)
                        findings.append(UnresolvedFinding(
                            finding_type="boundary_optimum",
                            metric=r.get("objective_field", ""),
                            current_value=r.get("optimal_metric", 0),
                            param_path=r.get("param_path", ""),
                            module_name=module,
                            config=r.get("config", {}),
                            suggested_action=ConvergenceAction.EXPAND_RANGE,
                            suggested_bounds=new_bounds,
                            details=f"Optimum at boundary ({optimal:.4g}). True optimum likely outside [{bounds[0]:.4g}, {bounds[1]:.4g}].",
                        ))

        elif tool == "sweep_parameter":
            sweep_results = r.get("results", [])
            param_path = r.get("param_path", "")
            if not sweep_results or len(sweep_results) < 3:
                continue

            param_values = [sr["param_value"] for sr in sweep_results if sr.get("success")]
            if not param_values:
                continue

            # Check each observable for monotonic trends
            obs_names = set()
            for sr in sweep_results:
                if sr.get("observables"):
                    obs_names.update(sr["observables"].keys())

            for obs_name in obs_names:
                vals = []
                for sr in sweep_results:
                    if sr.get("success") and sr.get("observables", {}).get(obs_name):
                        v = sr["observables"][obs_name].get("value")
                        if isinstance(v, (int, float)):
                            vals.append(v)

                if len(vals) >= 3:
                    is_mono, direction = _check_monotonicity(vals)
                    if is_mono:
                        max_p = max(param_values)
                        min_p = min(param_values)
                        span = max_p - min_p
                        if direction == "increasing":
                            new_bounds = (max_p, max_p + span)
                        else:
                            new_bounds = (min_p - span, min_p)

                        findings.append(UnresolvedFinding(
                            finding_type="monotonic_trend",
                            metric=obs_name,
                            current_value=vals[-1] if direction == "increasing" else vals[0],
                            param_path=param_path,
                            module_name=module,
                            config={},
                            suggested_action=ConvergenceAction.EXPAND_RANGE,
                            suggested_bounds=new_bounds,
                            details=f"{obs_name} is {direction} across entire sweep — optimum beyond range.",
                        ))

            # Check observables with thresholds for near-misses
            for sr in sweep_results:
                for obs_name, obs_data in sr.get("observables", {}).items():
                    if isinstance(obs_data, dict):
                        threshold = obs_data.get("threshold")
                        margin = obs_data.get("margin")
                        value = obs_data.get("value")
                        if threshold is not None and margin is not None and value is not None:
                            if isinstance(margin, (int, float)) and isinstance(value, (int, float)):
                                # Near miss: within 20% of threshold
                                if 0 < abs(margin) < abs(threshold) * 0.2:
                                    findings.append(UnresolvedFinding(
                                        finding_type="threshold_near_miss",
                                        metric=obs_name,
                                        current_value=value,
                                        param_path=param_path,
                                        module_name=module,
                                        config={},
                                        suggested_action=ConvergenceAction.BISECT,
                                        details=f"{obs_name}={value:.4g} is within 20% of threshold {threshold:.4g} (margin={margin:+.4g}).",
                                    ))

    return findings


def build_goal_from_finding(finding: UnresolvedFinding) -> AnalysisGoal:
    """Convert an unresolved finding into an AnalysisGoal for convergence."""
    if finding.finding_type == "boundary_optimum":
        return AnalysisGoal(
            goal_type=GoalType.MAXIMIZE,  # Will be refined by the evaluator
            metric=finding.metric,
            search_variable=finding.param_path,
            search_bounds=finding.suggested_bounds,
            module_name=finding.module_name,
            config=finding.config,
        )
    elif finding.finding_type == "monotonic_trend":
        return AnalysisGoal(
            goal_type=GoalType.MAXIMIZE,
            metric=finding.metric,
            search_variable=finding.param_path,
            search_bounds=finding.suggested_bounds,
            module_name=finding.module_name,
            config=finding.config,
        )
    elif finding.finding_type == "threshold_near_miss":
        return AnalysisGoal(
            goal_type=GoalType.FIND_THRESHOLD,
            metric=finding.metric,
            operator=">",
            threshold=finding.current_value * 1.1,  # 10% beyond current
            search_variable=finding.param_path,
            search_bounds=finding.suggested_bounds,
            module_name=finding.module_name,
            config=finding.config,
        )
    else:
        return AnalysisGoal(
            goal_type=GoalType.CHARACTERIZE,
            metric=finding.metric,
            search_variable=finding.param_path,
            module_name=finding.module_name,
            config=finding.config,
        )


def extract_goal_from_scenario(scenario: Dict[str, Any]) -> Optional[AnalysisGoal]:
    """Extract a goal from a scenario's decision_focus field.

    Analyzes the decision question to determine what kind of answer is needed.
    """
    decision = scenario.get("decision_focus", "").lower()
    if not decision:
        return None

    # Pattern: "at what X does Y..."
    threshold_patterns = [
        r"at what (\w+) does",
        r"how much (\w+) (?:before|until|exists)",
        r"what (\w+) (?:causes?|triggers?|leads? to)",
        r"(?:find|determine) .*?(\w+).*?(?:where|when|threshold)",
    ]
    for pat in threshold_patterns:
        m = re.search(pat, decision)
        if m:
            return AnalysisGoal(
                goal_type=GoalType.FIND_THRESHOLD,
                metric=m.group(1),
                raw_prompt=decision,
            )

    # Pattern: maximize/minimize
    if any(w in decision for w in ["maximize", "optimal", "best", "strongest"]):
        return AnalysisGoal(goal_type=GoalType.MAXIMIZE, metric="", raw_prompt=decision)
    if any(w in decision for w in ["minimize", "reduce", "lowest"]):
        return AnalysisGoal(goal_type=GoalType.MINIMIZE, metric="", raw_prompt=decision)

    # Pattern: compare/select
    if any(w in decision for w in ["which", "better", "compare", "choose"]):
        return AnalysisGoal(goal_type=GoalType.COMPARE_SELECT, metric="", raw_prompt=decision)

    # Pattern: yes/no threshold question
    if any(w in decision for w in ["is ", "does ", "can ", "will "]):
        return AnalysisGoal(goal_type=GoalType.FIND_THRESHOLD, metric="", raw_prompt=decision)

    return None
