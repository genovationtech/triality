from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional

import numpy as np

from triality.core.fields import CANONICAL_FIELDS, PhysicsField, PhysicsState
from triality.core.units import SI_UNITS, convert
from triality.runtime import (
    RuntimeContractError,
    RuntimeExecutionError,
    RuntimeExecutionResult,
    load_module,
)

SchedulerMode = Literal["topological", "staged"]
GraphErrorPolicy = Literal["raise", "continue"]


@dataclass
class RuntimeNode:
    """A node in a runtime orchestration graph."""

    name: str
    module_name: str
    config: Optional[dict] = None
    initial_state: Optional[PhysicsState] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class RuntimeLink:
    """A directed field link between two runtime nodes."""

    source: str
    target: str
    source_field: str
    target_field: Optional[str] = None
    required: bool = True
    target_unit: Optional[str] = None
    scale: float = 1.0
    offset: float = 0.0
    aggregation: Optional[str] = None
    resample_points: Optional[int] = None


@dataclass
class GraphConvergenceCriteria:
    """Convergence settings for iterative graph execution."""

    abs_tol: float = 1e-9
    rel_tol: float = 1e-6
    min_iterations: int = 2
    max_iterations: int = 10
    monitored_fields: Optional[List[str]] = None


@dataclass
class RuntimeNodeResult:
    """Execution record for a single graph node."""

    node: RuntimeNode
    input_state: Optional[PhysicsState]
    execution: RuntimeExecutionResult
    output_state: Optional[PhysicsState]
    iteration: int


@dataclass
class RuntimeGraphResult:
    """Aggregate result of a runtime graph execution."""

    execution_order: List[str]
    execution_stages: List[List[str]]
    dependency_map: Dict[str, Dict[str, List[str]]]
    link_compatibility_report: List[Dict[str, object]]
    iterations: List[Dict[str, RuntimeNodeResult]]
    convergence_history: List[Dict[str, object]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    converged: bool = False
    max_abs_delta: float = float("inf")
    max_rel_delta: float = float("inf")

    @property
    def node_results(self) -> Dict[str, RuntimeNodeResult]:
        return self.iterations[-1] if self.iterations else {}

    @property
    def iterations_run(self) -> int:
        return len(self.iterations)

    @property
    def success(self) -> bool:
        latest = self.node_results
        return bool(latest) and all(node.execution.success for node in latest.values())

    def to_report(self) -> Dict[str, object]:
        per_iteration = []
        merge_log = []
        for iteration_index, node_map in enumerate(self.iterations, start=1):
            iter_report = {}
            for node_name, node_result in node_map.items():
                input_fields = [] if node_result.input_state is None else node_result.input_state.field_names()
                output_fields = [] if node_result.output_state is None else node_result.output_state.field_names()
                iter_report[node_name] = {
                    "module_name": node_result.node.module_name,
                    "success": node_result.execution.success,
                    "status": node_result.execution.status,
                    "input_fields": input_fields,
                    "output_fields": output_fields,
                    "warnings": list(node_result.execution.warnings),
                    "error": node_result.execution.error,
                }
                if node_result.input_state is not None:
                    merge_log.append({
                        "iteration": iteration_index,
                        "node": node_name,
                        "merged_from": node_result.input_state.metadata.get("merged_from", []),
                        "overridden_fields": node_result.input_state.metadata.get("overridden_fields", []),
                    })
            per_iteration.append(iter_report)
        return {
            "execution_plan": self.execution_order,
            "stage_breakdown": self.execution_stages,
            "node_dependency_map": self.dependency_map,
            "link_compatibility_report": self.link_compatibility_report,
            "convergence_history": self.convergence_history,
            "per_iteration_node_results": per_iteration,
            "merge_override_log": merge_log,
            "warnings": self.warnings,
            "converged": self.converged,
            "iterations_run": self.iterations_run,
            "max_abs_delta": self.max_abs_delta,
            "max_rel_delta": self.max_rel_delta,
        }


class RuntimeGraph:
    """DAG executor for composing runtime SDK modules.

    The graph orchestrates runtime-capable modules by:
    - instantiating each node from config or demo case
    - propagating linked output fields as PhysicsState inputs
    - validating field and unit compatibility up front
    - executing nodes in topological/staged order
    - optionally iterating until graph-level convergence
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, RuntimeNode] = {}
        self._links: List[RuntimeLink] = []

    def add_node(
        self,
        name: str,
        module_name: str,
        config: Optional[dict] = None,
        initial_state: Optional[PhysicsState] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> RuntimeNode:
        if name in self._nodes:
            raise RuntimeContractError(f"Runtime graph already contains node '{name}'.")
        node = RuntimeNode(
            name=name,
            module_name=module_name,
            config=dict(config) if config is not None else None,
            initial_state=initial_state,
            metadata=dict(metadata or {}),
        )
        self._nodes[name] = node
        return node

    def add_link(
        self,
        source: str,
        target: str,
        source_field: str,
        target_field: Optional[str] = None,
        *,
        required: bool = True,
        target_unit: Optional[str] = None,
        scale: float = 1.0,
        offset: float = 0.0,
        aggregation: Optional[str] = None,
        resample_points: Optional[int] = None,
    ) -> RuntimeLink:
        link = RuntimeLink(
            source=source,
            target=target,
            source_field=source_field,
            target_field=target_field,
            required=required,
            target_unit=target_unit,
            scale=scale,
            offset=offset,
            aggregation=aggregation,
            resample_points=resample_points,
        )
        self._links.append(link)
        return link

    @property
    def nodes(self) -> Dict[str, RuntimeNode]:
        return dict(self._nodes)

    @property
    def links(self) -> List[RuntimeLink]:
        return list(self._links)

    def validate(self) -> None:
        if not self._nodes:
            raise RuntimeContractError("Runtime graph has no nodes.")

        for link in self._links:
            if link.source not in self._nodes:
                raise RuntimeContractError(f"Runtime graph link source '{link.source}' is not a known node.")
            if link.target not in self._nodes:
                raise RuntimeContractError(f"Runtime graph link target '{link.target}' is not a known node.")
            if link.source == link.target:
                raise RuntimeContractError(f"Runtime graph link '{link.source}' -> '{link.target}' cannot self-reference.")

            source_handle = load_module(self._nodes[link.source].module_name)
            target_handle = load_module(self._nodes[link.target].module_name)
            source_outputs = {field["name"]: field for field in source_handle.describe()["output_fields"]}
            target_inputs = {field["name"]: field for field in target_handle.describe()["required_inputs"]}

            if link.source_field not in source_outputs:
                raise RuntimeContractError(
                    f"Runtime graph link requests field '{link.source_field}' from '{link.source}', "
                    f"but module '{self._nodes[link.source].module_name}' advertises outputs {sorted(source_outputs)}."
                )

            target_field = link.target_field or link.source_field
            if target_inputs and target_field not in target_inputs:
                raise RuntimeContractError(
                    f"Runtime graph target '{link.target}' does not advertise input '{target_field}'. "
                    f"Available required_inputs: {sorted(target_inputs)}."
                )

            self._validate_unit_compatibility(
                source_unit=source_outputs[link.source_field]["units"],
                target_field=target_field,
                target_unit=link.target_unit or target_inputs.get(target_field, {}).get("units"),
                source=link.source,
                target=link.target,
                source_field=link.source_field,
            )

        self.execution_order()

    def dependency_map(self) -> Dict[str, Dict[str, List[str]]]:
        upstream = {name: [] for name in self._nodes}
        downstream = {name: [] for name in self._nodes}
        for link in self._links:
            upstream[link.target].append(link.source)
            downstream[link.source].append(link.target)
        return {name: {"upstream": sorted(upstream[name]), "downstream": sorted(downstream[name])} for name in self._nodes}

    def link_compatibility_report(self) -> List[Dict[str, object]]:
        report = []
        for link in self._links:
            source_outputs = {field["name"]: field for field in load_module(self._nodes[link.source].module_name).describe()["output_fields"]}
            target_inputs = {field["name"]: field for field in load_module(self._nodes[link.target].module_name).describe()["required_inputs"]}
            target_field = link.target_field or link.source_field
            target_unit = link.target_unit or target_inputs.get(target_field, {}).get("units") or CANONICAL_FIELDS.get(target_field, None).si_unit if target_field in CANONICAL_FIELDS else None
            compatible = True
            if target_unit and source_outputs[link.source_field]["units"] in SI_UNITS and target_unit in SI_UNITS:
                try:
                    convert(1.0, source_outputs[link.source_field]["units"], target_unit)
                except ValueError:
                    compatible = False
            report.append({
                "source": link.source,
                "target": link.target,
                "source_field": link.source_field,
                "target_field": target_field,
                "source_unit": source_outputs[link.source_field]["units"],
                "target_unit": target_unit,
                "compatible": compatible,
                "adaptation": {
                    "target_unit": link.target_unit,
                    "scale": link.scale,
                    "offset": link.offset,
                    "aggregation": link.aggregation,
                    "resample_points": link.resample_points,
                },
            })
        return report

    def explain(self) -> Dict[str, object]:
        return {
            "execution_plan": self.execution_order(),
            "stage_breakdown": self.execution_stages(),
            "node_dependency_map": self.dependency_map(),
            "link_compatibility_report": self.link_compatibility_report(),
        }

    def _validate_unit_compatibility(
        self,
        *,
        source_unit: str,
        target_field: str,
        target_unit: Optional[str],
        source: str,
        target: str,
        source_field: str,
    ) -> None:
        candidate_units = [unit for unit in [target_unit, CANONICAL_FIELDS.get(target_field, None).si_unit if target_field in CANONICAL_FIELDS else None] if unit]
        for unit in candidate_units:
            if source_unit in SI_UNITS and unit in SI_UNITS:
                try:
                    convert(1.0, source_unit, unit)
                    return
                except ValueError as exc:
                    raise RuntimeContractError(
                        f"Runtime graph link {source}:{source_field} -> {target}:{target_field} has incompatible units "
                        f"('{source_unit}' -> '{unit}')."
                    ) from exc
        # If there is no known target unit metadata, accept the link as best-effort.

    def execution_order(self) -> List[str]:
        return [node for stage in self.execution_stages() for node in stage]

    def execution_stages(self) -> List[List[str]]:
        indegree = {name: 0 for name in self._nodes}
        adjacency: Dict[str, List[str]] = {name: [] for name in self._nodes}
        for link in self._links:
            adjacency[link.source].append(link.target)
            indegree[link.target] += 1

        queue = deque(sorted(name for name, degree in indegree.items() if degree == 0))
        stages: List[List[str]] = []
        visited = 0

        while queue:
            stage_size = len(queue)
            stage: List[str] = []
            for _ in range(stage_size):
                node = queue.popleft()
                stage.append(node)
                visited += 1
                for neighbor in adjacency[node]:
                    indegree[neighbor] -= 1
                    if indegree[neighbor] == 0:
                        queue.append(neighbor)
            stages.append(stage)

        if visited != len(self._nodes):
            raise RuntimeContractError("Runtime graph contains a cycle; execution order is undefined.")
        return stages

    def run(
        self,
        external_inputs: Optional[Dict[str, PhysicsState]] = None,
        *,
        strict: bool = True,
        scheduler: SchedulerMode = "topological",
        error_policy: GraphErrorPolicy = "raise",
        convergence: Optional[GraphConvergenceCriteria] = None,
    ) -> RuntimeGraphResult:
        self.validate()
        external_inputs = dict(external_inputs or {})
        stages = self.execution_stages()
        order = self.execution_order()
        warnings: List[str] = []
        iterations: List[Dict[str, RuntimeNodeResult]] = []
        convergence_history: List[Dict[str, object]] = []
        convergence = convergence or GraphConvergenceCriteria(min_iterations=1, max_iterations=1)
        previous_results: Optional[Dict[str, RuntimeNodeResult]] = None
        max_abs_delta = float("inf")
        max_rel_delta = float("inf")
        converged = False

        for iteration in range(1, convergence.max_iterations + 1):
            current_results: Dict[str, RuntimeNodeResult] = {}
            stage_sequence = stages if scheduler == "staged" else [[node] for node in order]

            for stage in stage_sequence:
                for node_name in stage:
                    node = self._nodes[node_name]
                    input_state = self._build_input_state(
                        node_name=node_name,
                        external_inputs=external_inputs,
                        node_results=current_results,
                        previous_results=previous_results,
                        warnings=warnings,
                        error_policy=error_policy,
                    )

                    node_result = self._execute_node(
                        node=node,
                        input_state=input_state,
                        strict=strict,
                        error_policy=error_policy,
                        iteration=iteration,
                    )
                    current_results[node_name] = node_result

            iterations.append(current_results)

            if previous_results is not None:
                max_abs_delta, max_rel_delta, converged = self._check_convergence(
                    previous_results=previous_results,
                    current_results=current_results,
                    criteria=convergence,
                )
                convergence_history.append({
                    "iteration": iteration,
                    "max_abs_delta": max_abs_delta,
                    "max_rel_delta": max_rel_delta,
                    "converged": converged,
                })
                if converged and iteration >= convergence.min_iterations:
                    break
            elif convergence.min_iterations <= 1 and convergence.max_iterations <= 1:
                converged = True
                max_abs_delta = 0.0
                max_rel_delta = 0.0
                break

            previous_results = current_results

        return RuntimeGraphResult(
            execution_order=order,
            execution_stages=stages,
            dependency_map=self.dependency_map(),
            link_compatibility_report=self.link_compatibility_report(),
            iterations=iterations,
            convergence_history=convergence_history,
            warnings=warnings,
            converged=converged,
            max_abs_delta=max_abs_delta,
            max_rel_delta=max_rel_delta,
        )

    def _execute_node(
        self,
        *,
        node: RuntimeNode,
        input_state: Optional[PhysicsState],
        strict: bool,
        error_policy: GraphErrorPolicy,
        iteration: int,
    ) -> RuntimeNodeResult:
        try:
            handle = load_module(node.module_name)
            solver = handle.from_config(node.config) if node.config is not None else handle.from_demo_case()
            if input_state is not None:
                solver.set_input(input_state)
            execution = solver.solve(strict=strict)
            if strict and not execution.success:
                raise RuntimeExecutionError(
                    f"Runtime graph node '{node.name}' ({node.module_name}) failed: {execution.error}"
                )
            return RuntimeNodeResult(
                node=node,
                input_state=input_state,
                execution=execution,
                output_state=execution.generated_state,
                iteration=iteration,
            )
        except Exception as exc:
            if error_policy == "raise":
                raise
            handle = load_module(node.module_name)
            failure = RuntimeExecutionResult(
                module_name=node.module_name,
                success=False,
                status="graph_continue_error",
                warnings=[str(exc)],
                residuals={},
                convergence={},
                elapsed_time_s=0.0,
                result_payload=None,
                generated_state=None,
                description=handle.describe(),
                error=str(exc),
            )
            return RuntimeNodeResult(
                node=node,
                input_state=input_state,
                execution=failure,
                output_state=None,
                iteration=iteration,
            )

    def _build_input_state(
        self,
        *,
        node_name: str,
        external_inputs: Dict[str, PhysicsState],
        node_results: Dict[str, RuntimeNodeResult],
        previous_results: Optional[Dict[str, RuntimeNodeResult]],
        warnings: List[str],
        error_policy: GraphErrorPolicy,
    ) -> Optional[PhysicsState]:
        fragments: List[PhysicsState] = []
        node = self._nodes[node_name]

        if node.initial_state is not None:
            fragments.append(node.initial_state)
        if node_name in external_inputs:
            fragments.append(external_inputs[node_name])

        for link in self._links:
            if link.target != node_name:
                continue
            source_result = node_results.get(link.source)
            if source_result is None and previous_results is not None:
                source_result = previous_results.get(link.source)
            if source_result is None:
                raise RuntimeContractError(
                    f"Runtime graph cannot consume '{link.source}' before it has executed."
                )
            source_state = source_result.output_state
            if source_state is None:
                message = (
                    f"Runtime graph required output from '{link.source}' for '{node_name}', but no state was produced."
                )
                if link.required and error_policy == "raise":
                    raise RuntimeExecutionError(message)
                warnings.append(message)
                continue
            if link.source_field not in source_state:
                message = (
                    f"Runtime graph required field '{link.source_field}' from '{link.source}', "
                    f"but available fields are {source_state.field_names()}."
                )
                if link.required and error_policy == "raise":
                    raise RuntimeContractError(message)
                warnings.append(message)
                continue

            source_field = source_state.get(link.source_field)
            mapped_name = link.target_field or link.source_field
            fragments.append(_field_to_state(link, mapped_name, source_field, source_state.time, link.source, node_name))

        if not fragments:
            return None
        return merge_physics_states(fragments, solver_name=f"graph_input:{node_name}")

    def _check_convergence(
        self,
        *,
        previous_results: Dict[str, RuntimeNodeResult],
        current_results: Dict[str, RuntimeNodeResult],
        criteria: GraphConvergenceCriteria,
    ) -> tuple[float, float, bool]:
        monitored_fields = set(criteria.monitored_fields or self._default_monitored_fields())
        max_abs_delta = 0.0
        max_rel_delta = 0.0
        compared = False

        for node_name, current in current_results.items():
            previous = previous_results.get(node_name)
            if previous is None or previous.output_state is None or current.output_state is None:
                continue
            shared = monitored_fields.intersection(current.output_state.field_names())
            shared = shared.intersection(previous.output_state.field_names())
            for field_name in shared:
                current_field = current.output_state.get(field_name)
                previous_field = previous.output_state.get(field_name)
                if current_field.unit != previous_field.unit and current_field.unit in SI_UNITS and previous_field.unit in SI_UNITS:
                    previous_data = convert(previous_field.data, previous_field.unit, current_field.unit)
                else:
                    previous_data = previous_field.data
                delta = np.asarray(current_field.data) - np.asarray(previous_data)
                abs_delta = float(np.max(np.abs(delta)))
                denom = np.maximum(np.abs(np.asarray(previous_data)), criteria.abs_tol)
                rel_delta = float(np.max(np.abs(delta) / denom))
                max_abs_delta = max(max_abs_delta, abs_delta)
                max_rel_delta = max(max_rel_delta, rel_delta)
                compared = True

        if not compared:
            return float("inf"), float("inf"), False
        converged = max_abs_delta <= criteria.abs_tol or max_rel_delta <= criteria.rel_tol
        return max_abs_delta, max_rel_delta, converged

    def _default_monitored_fields(self) -> List[str]:
        return sorted({link.source_field for link in self._links})

    def to_dict(self) -> dict:
        return {
            "nodes": [
                {
                    "name": node.name,
                    "module_name": node.module_name,
                    "config": node.config,
                    "initial_state": _serialize_state(node.initial_state),
                    "metadata": _jsonable(node.metadata),
                }
                for node in self._nodes.values()
            ],
            "links": [
                {
                    "source": link.source,
                    "target": link.target,
                    "source_field": link.source_field,
                    "target_field": link.target_field,
                    "required": link.required,
                    "target_unit": link.target_unit,
                    "scale": link.scale,
                    "offset": link.offset,
                    "aggregation": link.aggregation,
                    "resample_points": link.resample_points,
                }
                for link in self._links
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RuntimeGraph":
        graph = cls()
        for node in data.get("nodes", []):
            graph.add_node(
                node["name"],
                node["module_name"],
                config=node.get("config"),
                initial_state=_deserialize_state(node.get("initial_state")),
                metadata=node.get("metadata"),
            )
        for link in data.get("links", []):
            graph.add_link(
                link["source"],
                link["target"],
                link["source_field"],
                target_field=link.get("target_field"),
                required=link.get("required", True),
                target_unit=link.get("target_unit"),
                scale=link.get("scale", 1.0),
                offset=link.get("offset", 0.0),
                aggregation=link.get("aggregation"),
                resample_points=link.get("resample_points"),
            )
        return graph

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path: str | Path) -> "RuntimeGraph":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _field_to_state(
    link: RuntimeLink,
    field_name: str,
    source_field: PhysicsField,
    time: float,
    source_node: str,
    target_node: str,
) -> PhysicsState:
    data = np.array(source_field.data, copy=True)
    unit = source_field.unit
    grid = None if source_field.grid is None else np.array(source_field.grid, copy=True)

    if link.target_unit is not None and unit in SI_UNITS and link.target_unit in SI_UNITS:
        data = convert(data, unit, link.target_unit)
        unit = link.target_unit

    data = data * link.scale + link.offset

    if link.resample_points is not None:
        if grid is None or data.ndim != 1:
            raise RuntimeContractError("resample_points requires a 1-D field with an associated grid.")
        target_grid = np.linspace(float(grid[0]), float(grid[-1]), int(link.resample_points))
        data = np.interp(target_grid, grid, data)
        grid = target_grid

    if link.aggregation is not None:
        agg = link.aggregation.lower()
        reducers = {"mean": np.mean, "max": np.max, "min": np.min, "sum": np.sum}
        if agg not in reducers:
            raise RuntimeContractError(f"Unsupported link aggregation '{link.aggregation}'.")
        data = np.array([reducers[agg](data)])
        grid = None

    state = PhysicsState(solver_name=f"link:{source_node}->{target_node}", time=time)
    state.set_field(field_name, data, unit, grid=grid)
    state.metadata["link_adaptation"] = {
        "target_unit": link.target_unit,
        "scale": link.scale,
        "offset": link.offset,
        "aggregation": link.aggregation,
        "resample_points": link.resample_points,
    }
    return state


def merge_physics_states(states: List[PhysicsState], *, solver_name: str) -> PhysicsState:
    """Merge multiple PhysicsState objects for orchestration input preparation."""
    merged = PhysicsState(solver_name=solver_name)
    merged.metadata["merged_from"] = [state.solver_name for state in states]
    overridden_fields: List[str] = []

    for state in states:
        merged.time = max(merged.time, state.time)
        for field_name, field in state.fields.items():
            if field_name in merged.fields:
                overridden_fields.append(field_name)
            merged.fields[field_name] = PhysicsField(
                name=field_name,
                data=np.array(field.data, copy=True),
                unit=field.unit,
                grid=None if field.grid is None else np.array(field.grid, copy=True),
                time=field.time,
            )
        merged.metadata.update(state.metadata)

    if overridden_fields:
        merged.metadata["overridden_fields"] = overridden_fields
    return merged


def _serialize_state(state: Optional[PhysicsState]) -> Optional[dict]:
    if state is None:
        return None
    return {
        "solver_name": state.solver_name,
        "time": state.time,
        "fields": {
            name: {
                "data": np.asarray(field.data).tolist(),
                "unit": field.unit,
                "grid": None if field.grid is None else np.asarray(field.grid).tolist(),
            }
            for name, field in state.fields.items()
        },
        "metadata": _jsonable(state.metadata),
    }


def _deserialize_state(data: Optional[dict]) -> Optional[PhysicsState]:
    if data is None:
        return None
    state = PhysicsState(solver_name=data.get("solver_name", "graph_state"), time=float(data.get("time", 0.0)))
    for name, field in data.get("fields", {}).items():
        grid = field.get("grid")
        state.set_field(
            name,
            np.asarray(field["data"]),
            field["unit"],
            grid=None if grid is None else np.asarray(grid),
        )
    state.metadata.update(data.get("metadata", {}))
    return state


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
