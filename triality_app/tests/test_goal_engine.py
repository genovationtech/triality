"""Tests for the Goal-Driven Convergence Engine."""
import numpy as np
import pytest

from triality_app.goal_engine import (
    AnalysisGoal,
    AnalyticalEstimate,
    ConvergenceAction,
    ConvergenceStrategy,
    GoalEvaluator,
    GoalType,
    _check_condition,
    _check_monotonicity,
    _closest_to_threshold,
    _extract_sweep_data,
    _safe_float,
    extract_goal_from_llm_response,
    extract_goal_heuristic,
    parse_analytical_estimate,
    infer_goal_from_plan,
    detect_unresolved_findings,
    build_goal_from_finding,
    extract_goal_from_scenario,
)


# ---------------------------------------------------------------------------
#  Helper: Build mock sweep results
# ---------------------------------------------------------------------------
def _make_sweep(param_values, metric_fn, metric_name="deflection"):
    """Build a mock sweep_parameter result."""
    return {
        "_tool_name": "sweep_parameter",
        "results": [
            {
                "success": True,
                "param_value": pv,
                "fields": {metric_name: {"min": 0.0, "max": metric_fn(pv), "mean": metric_fn(pv) / 2}},
                "observables": {},
            }
            for pv in param_values
        ],
    }


def _make_goal(**overrides):
    """Build a standard find_threshold goal with optional overrides."""
    defaults = dict(
        goal_type=GoalType.FIND_THRESHOLD,
        metric="deflection",
        operator=">",
        threshold=0.05,
        search_variable="solver.force",
        search_bounds=(100, 500),
        module_name="structural_analysis",
        config={"solver": {"material": "steel"}},
    )
    defaults.update(overrides)
    return AnalysisGoal(**defaults)


# ---------------------------------------------------------------------------
#  Tests: _safe_float
# ---------------------------------------------------------------------------
class TestSafeFloat:
    def test_normal(self):
        assert _safe_float(3.14) == 3.14

    def test_string(self):
        assert _safe_float("2.5") == 2.5

    def test_none(self):
        assert _safe_float(None) is None

    def test_invalid(self):
        assert _safe_float("abc") is None

    def test_inf(self):
        assert _safe_float(float("inf")) is None

    def test_int(self):
        assert _safe_float(42) == 42.0


# ---------------------------------------------------------------------------
#  Tests: _check_condition
# ---------------------------------------------------------------------------
class TestCheckCondition:
    def test_gt(self):
        assert _check_condition(10, ">", 5)
        assert not _check_condition(3, ">", 5)

    def test_lt(self):
        assert _check_condition(3, "<", 5)
        assert not _check_condition(10, "<", 5)

    def test_gte(self):
        assert _check_condition(5, ">=", 5)

    def test_lte(self):
        assert _check_condition(5, "<=", 5)

    def test_eq(self):
        assert _check_condition(5.0, "==", 5.01)
        assert not _check_condition(5.0, "==", 10.0)


# ---------------------------------------------------------------------------
#  Tests: _check_monotonicity
# ---------------------------------------------------------------------------
class TestCheckMonotonicity:
    def test_increasing(self):
        is_mono, direction = _check_monotonicity([1, 2, 3, 4, 5])
        assert is_mono and direction == "increasing"

    def test_decreasing(self):
        is_mono, direction = _check_monotonicity([5, 4, 3, 2, 1])
        assert is_mono and direction == "decreasing"

    def test_non_monotonic(self):
        is_mono, direction = _check_monotonicity([1, 3, 2, 4, 0])
        assert not is_mono

    def test_short(self):
        is_mono, _ = _check_monotonicity([1])
        assert not is_mono


# ---------------------------------------------------------------------------
#  Tests: _extract_sweep_data
# ---------------------------------------------------------------------------
class TestExtractSweepData:
    def test_basic_extraction(self):
        results = [_make_sweep([100, 200, 300], lambda v: v * 0.001)]
        data = _extract_sweep_data(results, "deflection")
        assert data is not None
        params, metrics = data
        assert params == [100, 200, 300]
        assert metrics == [0.1, 0.2, 0.3]

    def test_no_matching_field(self):
        results = [_make_sweep([100, 200], lambda v: v * 0.001, metric_name="pressure")]
        data = _extract_sweep_data(results, "deflection")
        assert data is None

    def test_aggregates_across_sweeps(self):
        s1 = _make_sweep([100, 200, 300], lambda v: v * 0.001)
        s2 = _make_sweep([400, 500, 600], lambda v: v * 0.001)
        data = _extract_sweep_data([s1, s2], "deflection")
        assert data is not None
        params, metrics = data
        assert len(params) == 6
        assert params == [100, 200, 300, 400, 500, 600]

    def test_deduplicates_param_values(self):
        s1 = _make_sweep([100, 200, 300], lambda v: v * 0.001)
        s2 = _make_sweep([200, 300, 400], lambda v: v * 0.002)  # overlapping
        data = _extract_sweep_data([s1, s2], "deflection")
        params, metrics = data
        # 200 and 300 appear in both — latest (s2) wins
        assert 200 in params
        idx_200 = params.index(200)
        assert metrics[idx_200] == 0.4  # 200 * 0.002

    def test_sorted_output(self):
        s1 = _make_sweep([500, 100, 300], lambda v: v * 0.001)
        data = _extract_sweep_data([s1], "deflection")
        params, _ = data
        assert params == sorted(params)


# ---------------------------------------------------------------------------
#  Tests: Goal Extraction (Heuristic)
# ---------------------------------------------------------------------------
class TestGoalExtractionHeuristic:
    def test_threshold_pattern(self):
        g = extract_goal_heuristic("find the force where deflection exceeds 50mm")
        assert g is not None
        assert g.goal_type == GoalType.FIND_THRESHOLD
        assert g.operator == ">"

    def test_maximize_pattern(self):
        g = extract_goal_heuristic("maximize the flow rate through the channel")
        assert g is not None
        assert g.goal_type == GoalType.MAXIMIZE

    def test_minimize_pattern(self):
        g = extract_goal_heuristic("minimize the peak temperature in the battery pack")
        assert g is not None
        assert g.goal_type == GoalType.MINIMIZE

    def test_compare_pattern(self):
        g = extract_goal_heuristic("which design is better for heat dissipation?")
        assert g is not None
        assert g.goal_type == GoalType.COMPARE_SELECT

    def test_no_goal(self):
        g = extract_goal_heuristic("hello")
        assert g is None


# ---------------------------------------------------------------------------
#  Tests: Goal Extraction (LLM response)
# ---------------------------------------------------------------------------
class TestGoalExtractionLLM:
    def test_full_response(self):
        resp = {
            "goal_type": "find_threshold",
            "metric": "max_deflection",
            "operator": ">",
            "threshold": 0.05,
            "unit": "m",
            "search_variable": "solver.force",
            "search_bounds": [10000, 500000],
            "module_name": "structural_analysis",
            "has_clear_goal": True,
        }
        goal = extract_goal_from_llm_response(resp, "test prompt")
        assert goal.goal_type == GoalType.FIND_THRESHOLD
        assert goal.metric == "max_deflection"
        assert goal.threshold == 0.05
        assert goal.search_bounds == (10000, 500000)
        assert goal.confidence == "exact"

    def test_no_clear_goal(self):
        resp = {"goal_type": "characterize", "metric": "velocity", "has_clear_goal": False}
        goal = extract_goal_from_llm_response(resp, "how does flow behave?")
        assert goal.goal_type == GoalType.CHARACTERIZE
        assert goal.confidence == "approximate"


# ---------------------------------------------------------------------------
#  Tests: Analytical Estimate Parsing
# ---------------------------------------------------------------------------
class TestAnalyticalEstimate:
    def test_parse(self):
        resp = {
            "estimate": 420000,
            "unit": "N",
            "governing_equation": "delta = F*L^3/(48*E*I)",
            "suggested_bounds": [100000, 1000000],
            "confidence": "within_2x",
            "reasoning": "Using Euler-Bernoulli beam theory...",
        }
        est = parse_analytical_estimate(resp)
        assert est.estimate == 420000
        assert est.unit == "N"
        assert est.suggested_bounds == (100000, 1000000)
        assert est.confidence == "within_2x"


# ---------------------------------------------------------------------------
#  Tests: GoalEvaluator — find_threshold
# ---------------------------------------------------------------------------
class TestGoalEvaluatorThreshold:
    def test_threshold_not_reached_expands(self):
        """When max metric < threshold, should expand range."""
        goal = _make_goal(threshold=0.05, operator=">")
        results = [_make_sweep([100, 200, 300, 400, 500], lambda v: v * 0.00008)]
        # Max metric: 500 * 0.00008 = 0.04 < 0.05

        ev = GoalEvaluator.evaluate(goal, results, 1)
        assert not ev.satisfied
        assert ev.action == ConvergenceAction.EXPAND_RANGE
        assert ev.closest_value == 0.04
        assert ev.suggested_bounds is not None
        assert ev.suggested_bounds[0] >= 500  # should expand upward

    def test_threshold_crossed_bisects(self):
        """When metric crosses threshold, should bisect for precision."""
        goal = _make_goal(threshold=0.05, operator=">")
        results = [_make_sweep(
            [100, 200, 300, 400, 500, 600, 700],
            lambda v: v * 0.0001,
        )]
        # Crosses at 500: 0.05 < 0.05 fails, 600: 0.06 > 0.05 passes

        ev = GoalEvaluator.evaluate(goal, results, 1)
        assert ev.crossing_param is not None
        assert 500 <= ev.crossing_param <= 600

    def test_threshold_converged(self):
        """When bracket is tight enough, should report answer."""
        goal = _make_goal(threshold=0.05, operator=">")
        # Create a very tight sweep around the crossing
        values = np.linspace(498, 502, 10).tolist()
        results = [_make_sweep(values, lambda v: v * 0.0001)]

        ev = GoalEvaluator.evaluate(goal, results, 3)
        assert ev.satisfied
        assert ev.action == ConvergenceAction.REPORT_ANSWER
        assert ev.answer is not None
        assert abs(ev.answer - 500) < 5  # should be close to 500

    def test_threshold_crossed_at_first_point(self):
        """When threshold exceeded at lowest search value, expand downward."""
        goal = _make_goal(threshold=0.01, operator=">")
        results = [_make_sweep([100, 200, 300], lambda v: v * 0.001)]
        # All values exceed 0.01

        ev = GoalEvaluator.evaluate(goal, results, 1)
        assert not ev.satisfied
        assert ev.action == ConvergenceAction.EXPAND_RANGE
        # Should expand downward
        assert ev.suggested_bounds[1] <= 100

    def test_less_than_operator(self):
        """Test with '<' operator."""
        goal = _make_goal(threshold=0.03, operator="<")
        results = [_make_sweep([100, 200, 300, 400, 500], lambda v: 0.05 - v * 0.00005)]
        # Values: 0.045, 0.04, 0.035, 0.03, 0.025 — crosses threshold around 400

        ev = GoalEvaluator.evaluate(goal, results, 1)
        assert ev.crossing_param is not None


# ---------------------------------------------------------------------------
#  Tests: GoalEvaluator — optimization
# ---------------------------------------------------------------------------
class TestGoalEvaluatorOptimization:
    def test_boundary_optimum_expands(self):
        goal = _make_goal(goal_type=GoalType.MAXIMIZE)
        results = [{
            "_tool_name": "optimize_parameter",
            "optimal_param_value": 99.5,  # at lower boundary
            "optimal_metric": 2.5,
            "bounds": [100, 500],
            "evaluations": [{"param_value": 99.5, "metric": 2.5}],
        }]

        ev = GoalEvaluator.evaluate(goal, results, 1)
        assert not ev.satisfied
        assert ev.action == ConvergenceAction.EXPAND_RANGE

    def test_interior_optimum_converges(self):
        goal = _make_goal(goal_type=GoalType.MAXIMIZE)
        results = [{
            "_tool_name": "optimize_parameter",
            "optimal_param_value": 300,  # interior
            "optimal_metric": 5.0,
            "bounds": [100, 500],
            "evaluations": [],
        }]

        ev = GoalEvaluator.evaluate(goal, results, 1)
        assert ev.satisfied
        assert ev.action == ConvergenceAction.REPORT_ANSWER
        assert ev.answer == 300

    def test_monotonic_sweep_expands(self):
        goal = _make_goal(goal_type=GoalType.MAXIMIZE, metric="velocity")
        results = [_make_sweep([100, 200, 300, 400, 500], lambda v: v * 2, metric_name="velocity")]

        ev = GoalEvaluator.evaluate(goal, results, 1)
        assert not ev.satisfied
        assert ev.action == ConvergenceAction.EXPAND_RANGE


# ---------------------------------------------------------------------------
#  Tests: GoalEvaluator — comparison / characterize
# ---------------------------------------------------------------------------
class TestGoalEvaluatorOther:
    def test_comparison_always_satisfied(self):
        goal = _make_goal(goal_type=GoalType.COMPARE_SELECT)
        ev = GoalEvaluator.evaluate(goal, [], 1)
        assert ev.satisfied

    def test_characterize_always_satisfied(self):
        goal = _make_goal(goal_type=GoalType.CHARACTERIZE)
        ev = GoalEvaluator.evaluate(goal, [], 1)
        assert ev.satisfied


# ---------------------------------------------------------------------------
#  Tests: ConvergenceStrategy
# ---------------------------------------------------------------------------
class TestConvergenceStrategy:
    def test_builds_sweep_for_expansion(self):
        goal = _make_goal()
        ev = GoalEvaluator.evaluate(goal, [_make_sweep([100, 200], lambda v: v * 0.0001)], 1)
        plan = ConvergenceStrategy.build_next_plan(goal, ev, 1, [])

        assert plan is not None
        assert len(plan) == 1
        assert plan[0]["tool"] == "sweep_parameter"
        assert plan[0]["args"]["module_name"] == "structural_analysis"

    def test_builds_optimize_for_maximize(self):
        goal = _make_goal(goal_type=GoalType.MAXIMIZE)
        ev = GoalEvaluator._evaluate_optimization(goal, [
            _make_sweep([100, 200, 300], lambda v: v * 2, metric_name="deflection")
        ], 1)
        plan = ConvergenceStrategy.build_next_plan(goal, ev, 1, [])

        assert plan is not None
        assert plan[0]["tool"] == "optimize_parameter"

    def test_returns_none_when_satisfied(self):
        ev = GoalEvaluator.evaluate(
            _make_goal(goal_type=GoalType.CHARACTERIZE), [], 1
        )
        plan = ConvergenceStrategy.build_next_plan(_make_goal(), ev, 1, [])
        assert plan is None


# ---------------------------------------------------------------------------
#  Tests: Full multi-iteration convergence
# ---------------------------------------------------------------------------
class TestFullConvergence:
    def test_converges_in_4_iterations(self):
        """Simulates the full find_threshold convergence loop."""
        goal = _make_goal(threshold=0.05, operator=">", search_bounds=(100, 500))
        metric_fn = lambda v: v * 0.00008  # crosses 0.05 at v=625

        all_results = []
        bounds = goal.search_bounds

        for iteration in range(1, 9):
            # Build sweep for current bounds
            values = np.linspace(bounds[0], bounds[1], 10).tolist()
            sweep = _make_sweep(values, metric_fn)
            all_results.append(sweep)

            ev = GoalEvaluator.evaluate(goal, all_results, iteration)

            if ev.satisfied:
                assert abs(ev.answer - 625) < 10  # should be close to 625
                assert ev.accuracy_pct < 2.0  # within tolerance
                break

            if ev.suggested_bounds:
                bounds = ev.suggested_bounds
            else:
                break
        else:
            pytest.fail("Did not converge within 8 iterations")

    def test_converges_with_less_than_operator(self):
        """Test convergence with '<' operator."""
        goal = _make_goal(
            threshold=0.03,
            operator="<",
            search_bounds=(100, 500),
        )
        metric_fn = lambda v: 0.1 - v * 0.0001  # crosses 0.03 at v=700

        all_results = []
        bounds = goal.search_bounds

        for iteration in range(1, 9):
            values = np.linspace(bounds[0], bounds[1], 10).tolist()
            sweep = _make_sweep(values, metric_fn)
            all_results.append(sweep)

            ev = GoalEvaluator.evaluate(goal, all_results, iteration)

            if ev.satisfied:
                assert ev.answer is not None
                break

            if ev.suggested_bounds:
                bounds = ev.suggested_bounds
            else:
                break


# ---------------------------------------------------------------------------
#  Tests: AnalysisGoal serialization
# ---------------------------------------------------------------------------
class TestAnalysisGoalSerialization:
    def test_to_dict(self):
        goal = _make_goal()
        d = goal.to_dict()
        assert d["goal_type"] == "find_threshold"
        assert d["metric"] == "deflection"
        assert d["threshold"] == 0.05
        assert "config" not in d  # config is not serialized

    def test_to_dict_minimal(self):
        goal = AnalysisGoal(goal_type=GoalType.CHARACTERIZE, metric="velocity")
        d = goal.to_dict()
        assert "threshold" not in d
        assert "operator" not in d


# ---------------------------------------------------------------------------
#  Tests: Plan-based goal inference
# ---------------------------------------------------------------------------
class TestPlanInference:
    def test_optimize_infers_maximize(self):
        plan = {
            "tool_calls": [{
                "tool": "optimize_parameter",
                "args": {
                    "module_name": "navier_stokes",
                    "param_path": "solver.nu",
                    "bounds": [0.005, 0.05],
                    "objective_field": "velocity",
                    "objective": "maximize",
                    "config": {},
                },
            }],
        }
        g = infer_goal_from_plan(plan, "maximize velocity")
        assert g is not None
        assert g.goal_type == GoalType.MAXIMIZE
        assert g.metric == "velocity"
        assert g.search_variable == "solver.nu"

    def test_optimize_infers_minimize(self):
        plan = {
            "tool_calls": [{
                "tool": "optimize_parameter",
                "args": {
                    "module_name": "battery_thermal",
                    "param_path": "solver.coolant_flow",
                    "bounds": [0.1, 10],
                    "objective_field": "temperature",
                    "objective": "minimize",
                    "config": {},
                },
            }],
        }
        g = infer_goal_from_plan(plan, "minimize temperature")
        assert g is not None
        assert g.goal_type == GoalType.MINIMIZE

    def test_compare_infers_compare_select(self):
        plan = {
            "tool_calls": [{
                "tool": "compare_scenarios",
                "args": {"module_name": "navier_stokes", "scenarios": []},
            }],
        }
        g = infer_goal_from_plan(plan, "which is better")
        assert g is not None
        assert g.goal_type == GoalType.COMPARE_SELECT

    def test_sweep_does_not_infer(self):
        plan = {
            "tool_calls": [{
                "tool": "sweep_parameter",
                "args": {"module_name": "navier_stokes", "param_path": "solver.nu"},
            }],
        }
        g = infer_goal_from_plan(plan, "sweep viscosity")
        assert g is None  # Sweep alone doesn't imply a goal

    def test_empty_plan(self):
        g = infer_goal_from_plan({"tool_calls": []}, "hello")
        assert g is None


# ---------------------------------------------------------------------------
#  Tests: Unresolved findings detection
# ---------------------------------------------------------------------------
class TestUnresolvedFindings:
    def test_detects_boundary_optimum(self):
        results = [{
            "_tool_name": "optimize_parameter",
            "module_name": "navier_stokes",
            "optimal_param_value": 0.0498,
            "optimal_metric": 1.5,
            "bounds": [0.005, 0.05],
            "param_path": "solver.nu",
            "objective_field": "velocity",
        }]
        findings = detect_unresolved_findings(results)
        assert len(findings) == 1
        assert findings[0].finding_type == "boundary_optimum"
        assert findings[0].suggested_action == ConvergenceAction.EXPAND_RANGE

    def test_detects_monotonic_trend(self):
        results = [{
            "_tool_name": "sweep_parameter",
            "module_name": "navier_stokes",
            "param_path": "solver.nu",
            "results": [
                {
                    "success": True,
                    "param_value": v,
                    "fields": {},
                    "observables": {"dead_zone": {"value": v * 10, "unit": "%"}},
                }
                for v in [0.01, 0.02, 0.03, 0.04, 0.05]
            ],
        }]
        findings = detect_unresolved_findings(results)
        assert any(f.finding_type == "monotonic_trend" for f in findings)

    def test_no_findings_for_interior_optimum(self):
        results = [{
            "_tool_name": "optimize_parameter",
            "module_name": "navier_stokes",
            "optimal_param_value": 0.025,
            "optimal_metric": 2.0,
            "bounds": [0.005, 0.05],
            "param_path": "solver.nu",
        }]
        findings = detect_unresolved_findings(results)
        assert len(findings) == 0

    def test_build_goal_from_boundary_finding(self):
        results = [{
            "_tool_name": "optimize_parameter",
            "module_name": "navier_stokes",
            "optimal_param_value": 0.0501,
            "optimal_metric": 1.5,
            "bounds": [0.005, 0.05],
            "param_path": "solver.nu",
            "objective_field": "velocity",
            "config": {"solver": {"nx": 32}},
        }]
        findings = detect_unresolved_findings(results)
        assert len(findings) >= 1
        goal = build_goal_from_finding(findings[0])
        assert goal.goal_type == GoalType.MAXIMIZE
        assert goal.search_bounds is not None


# ---------------------------------------------------------------------------
#  Tests: Scenario goal extraction
# ---------------------------------------------------------------------------
class TestScenarioGoalExtraction:
    def test_maximize_scenario(self):
        s = {"decision_focus": "What viscosity setting maximizes vortex strength?"}
        g = extract_goal_from_scenario(s)
        assert g is not None
        assert g.goal_type == GoalType.MAXIMIZE

    def test_threshold_scenario(self):
        s = {"decision_focus": "At what temperature does junction behavior degrade?"}
        g = extract_goal_from_scenario(s)
        assert g is not None
        assert g.goal_type == GoalType.FIND_THRESHOLD

    def test_yes_no_scenario(self):
        s = {"decision_focus": "Is recirculation strong enough to cool all corners?"}
        g = extract_goal_from_scenario(s)
        assert g is not None
        assert g.goal_type == GoalType.FIND_THRESHOLD

    def test_compare_scenario(self):
        s = {"decision_focus": "Which configuration is better for heat dissipation?"}
        g = extract_goal_from_scenario(s)
        assert g is not None
        assert g.goal_type == GoalType.COMPARE_SELECT

    def test_no_decision_focus(self):
        s = {"title": "Some scenario"}
        g = extract_goal_from_scenario(s)
        assert g is None
