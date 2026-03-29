import pytest

from triality import (
    GraphConvergenceCriteria,
    RuntimeContractError,
    RuntimeGraph,
    available_runtime_templates,
    load_runtime_template,
    merge_physics_states,
)
from triality.core.fields import PhysicsState


NAVIER_CONFIG = {
    "solver": {"nx": 10, "ny": 10, "U_lid": 0.15},
    "solve": {"t_end": 0.004, "dt": 0.002, "max_steps": 5, "pressure_iters": 20, "pressure_tol": 1e-4},
}
THERMAL_CONFIG = {
    "solver": {"n_axial": 8, "n_fuel_radial": 6, "mass_flux": 3000.0},
    "solve": {"peak_linear_heat_rate": 140.0, "axial_shape": "uniform"},
}


def test_runtime_graph_executes_in_topological_order_and_propagates_links():
    graph = RuntimeGraph()
    graph.add_node("flow", "navier_stokes", config=NAVIER_CONFIG)
    graph.add_node("thermal", "thermal_hydraulics", config=THERMAL_CONFIG)
    graph.add_link("flow", "thermal", "pressure")

    result = graph.run(scheduler="staged")

    assert result.success is True
    assert result.execution_order == ["flow", "thermal"]
    assert result.execution_stages == [["flow"], ["thermal"]]
    assert result.node_results["thermal"].input_state is not None
    assert "pressure" in result.node_results["thermal"].input_state
    assert result.node_results["flow"].execution.description["construction_mode"] == "config"


def test_runtime_graph_supports_field_aliasing():
    graph = RuntimeGraph()
    graph.add_node("flow", "navier_stokes", config=NAVIER_CONFIG)
    graph.add_node("thermal", "thermal_hydraulics", config=THERMAL_CONFIG)
    graph.add_link("flow", "thermal", "pressure", target_field="linked_pressure")

    result = graph.run()

    assert "linked_pressure" in result.node_results["thermal"].input_state


def test_runtime_graph_rejects_cycles():
    graph = RuntimeGraph()
    graph.add_node("a", "navier_stokes", config=NAVIER_CONFIG)
    graph.add_node("b", "thermal_hydraulics", config=THERMAL_CONFIG)
    graph.add_link("a", "b", "pressure")
    graph.add_link("b", "a", "pressure", required=False)

    with pytest.raises(RuntimeContractError, match="cycle"):
        graph.execution_order()


def test_runtime_graph_rejects_unknown_linked_output_field():
    graph = RuntimeGraph()
    graph.add_node("flow", "navier_stokes", config=NAVIER_CONFIG)
    graph.add_node("thermal", "thermal_hydraulics", config=THERMAL_CONFIG)
    graph.add_link("flow", "thermal", "temperature")

    with pytest.raises(RuntimeContractError, match="advertises outputs"):
        graph.validate()


def test_runtime_graph_rejects_unit_incompatible_links():
    graph = RuntimeGraph()
    graph.add_node("flow", "navier_stokes", config=NAVIER_CONFIG)
    graph.add_node("thermal", "thermal_hydraulics", config=THERMAL_CONFIG)
    graph.add_link("flow", "thermal", "pressure", target_field="temperature")

    with pytest.raises(RuntimeContractError, match="incompatible units"):
        graph.validate()


def test_runtime_graph_iterates_to_graph_level_convergence():
    graph = RuntimeGraph()
    graph.add_node("flow", "navier_stokes", config=NAVIER_CONFIG)
    graph.add_node("thermal", "thermal_hydraulics", config=THERMAL_CONFIG)
    graph.add_link("flow", "thermal", "pressure")

    result = graph.run(
        convergence=GraphConvergenceCriteria(
            monitored_fields=["pressure"],
            abs_tol=1e-12,
            rel_tol=1e-12,
            min_iterations=2,
            max_iterations=3,
        )
    )

    assert result.converged is True
    assert result.iterations_run == 2
    assert result.max_abs_delta == 0.0


def test_runtime_graph_can_persist_and_reload_definitions(tmp_path):
    graph = RuntimeGraph()
    graph.add_node("flow", "navier_stokes", config=NAVIER_CONFIG)
    graph.add_node("thermal", "thermal_hydraulics", config=THERMAL_CONFIG)
    graph.add_link("flow", "thermal", "pressure")

    path = tmp_path / "graph.json"
    graph.save_json(path)
    restored = RuntimeGraph.load_json(path)

    assert restored.execution_order() == ["flow", "thermal"]
    assert restored.links[0].source_field == "pressure"


def test_merge_physics_states_tracks_overrides():
    a = PhysicsState(solver_name="a")
    a.set_field("pressure", [1.0], "Pa")
    b = PhysicsState(solver_name="b")
    b.set_field("pressure", [2.0], "Pa")
    b.set_field("temperature", [300.0], "K")

    merged = merge_physics_states([a, b], solver_name="merged")

    assert merged["pressure"][0] == 2.0
    assert merged["temperature"][0] == 300.0
    assert merged.metadata["overridden_fields"] == ["pressure"]


def test_runtime_graph_report_exposes_machine_readable_introspection():
    graph = RuntimeGraph()
    graph.add_node("flow", "navier_stokes", config=NAVIER_CONFIG)
    graph.add_node("thermal", "thermal_hydraulics", config=THERMAL_CONFIG)
    graph.add_link("flow", "thermal", "pressure", aggregation="mean")

    result = graph.run(convergence=GraphConvergenceCriteria(min_iterations=2, max_iterations=2, monitored_fields=["pressure"]))
    report = result.to_report()

    assert report["execution_plan"] == ["flow", "thermal"]
    assert report["stage_breakdown"] == [["flow"], ["thermal"]]
    assert "flow" in report["node_dependency_map"]
    assert report["link_compatibility_report"][0]["compatible"] is True
    assert len(report["convergence_history"]) == 1
    assert len(report["per_iteration_node_results"]) == 2
    assert report["merge_override_log"]


def test_runtime_graph_link_adaptation_converts_units_and_aggregates():
    graph = RuntimeGraph()
    graph.add_node("thermal", "thermal_hydraulics", config=THERMAL_CONFIG)
    graph.add_node("cht", "conjugate_heat_transfer", config={
        "solver": {"nx": 8, "ny_solid": 4, "ny_fluid": 8, "Q_vol": 150000.0, "T_init": 345.0, "T_fluid_top": 340.0},
        "solve": {"t_end": 0.002, "dt": 0.0005, "max_coupling_iter": 4, "save_interval": 1},
    })
    graph.add_link("thermal", "cht", "temperature", target_field="temperature", target_unit="degC", aggregation="mean")

    result = graph.run()
    field = result.node_results["cht"].input_state.get("temperature")

    assert field.unit == "degC"
    assert field.data.shape == (1,)


def test_runtime_templates_load_and_validate():
    names = available_runtime_templates()

    assert {"fluid_to_thermal", "thermal_to_conjugate_heat", "drift_diffusion_device", "sensing_pipeline"} <= set(names)

    for name in names:
        graph = load_runtime_template(name)
        graph.validate()
