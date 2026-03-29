from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
import time
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Type, TypedDict
import warnings

import numpy as np

from triality.core.fields import PhysicsState


class RuntimeContractError(RuntimeError):
    """Raised when a runtime adapter violates the SDK contract."""


class RuntimeExecutionError(RuntimeError):
    """Raised when a runtime adapter fails while executing a solve path."""


class RuntimeDescription(TypedDict):
    module_name: str
    domain: str
    fidelity_level: str
    coupling_ready: str
    supports_transient: bool
    supports_steady: bool
    supports_demo_case: bool
    construction_mode: str
    required_inputs: list["RuntimeFieldSpec"]
    output_fields: list["RuntimeFieldSpec"]
    validation_status: str
    contract_version: str


class RuntimeFieldSpec(TypedDict):
    name: str
    units: str
    kind: str
    required: bool


@dataclass
class RuntimeExecutionResult:
    """Standardized solve() return type for runtime-capable modules."""

    module_name: str
    success: bool
    status: str
    warnings: list[str]
    residuals: Dict[str, float]
    convergence: Dict[str, Any]
    elapsed_time_s: float
    result_payload: Any
    generated_state: Optional[PhysicsState]
    description: RuntimeDescription
    error: Optional[str] = None

    @property
    def module(self) -> str:
        return self.module_name

    @property
    def native_result(self) -> Any:
        return self.result_payload

    @property
    def state(self) -> Optional[PhysicsState]:
        return self.generated_state


class BaseRuntimeSolver(ABC):
    """Standardized runtime contract for smoke-testable Triality execution.

    Contract requirements for every runtime-capable module:

    - `from_demo_case() -> BaseRuntimeSolver`
    - `from_config(config: dict | None) -> BaseRuntimeSolver`
    - `solve() -> RuntimeExecutionResult`
    - `to_state() -> PhysicsState`
    - `describe() -> RuntimeDescription`

    Failure behavior:
    - contract/schema violations raise `RuntimeContractError`
    - runtime solve failures raise `RuntimeExecutionError`
    """

    contract_version = "1.0"
    module_name: str = "unknown"
    domain: str = "unknown"
    fidelity_level: str = "L1"
    coupling_ready: str = "M1"
    supports_transient: bool = False
    supports_steady: bool = True
    supports_demo_case: bool = True
    validation_status: str = "prototype"
    required_inputs: tuple[RuntimeFieldSpec, ...] = ()
    output_fields: tuple[RuntimeFieldSpec, ...] = ()

    def __init__(self) -> None:
        self._input_state: Optional[PhysicsState] = None
        self._last_result: Optional[RuntimeExecutionResult] = None
        self._runtime_origin: str = "direct"
        self._runtime_config: Dict[str, Any] = {}
        self._progress_log: list[Dict[str, Any]] = []

    def report_progress(self, step: int, total: int, **extra: Any) -> None:
        """Called by solvers during time-stepping to report real progress.

        Thread-safe: appends to a list that can be polled from another thread.
        """
        self._progress_log.append({"step": step, "total": total, **extra})

    def drain_progress(self) -> list[Dict[str, Any]]:
        """Return and clear all queued progress reports."""
        items = self._progress_log[:]
        self._progress_log.clear()
        return items

    @classmethod
    @abstractmethod
    def from_demo_case(cls) -> "BaseRuntimeSolver":
        """Create a deterministic, smoke-testable demo case."""

    @classmethod
    def demo(cls) -> "BaseRuntimeSolver":
        return cls.from_demo_case()

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "BaseRuntimeSolver":
        """Create a solver from keyword config.

        Unknown config keys raise `RuntimeContractError`.
        """
        config = config or {}
        params = inspect.signature(cls.__init__).parameters
        allowed = {name for name in params if name != "self"}
        unknown = sorted(set(config) - allowed)
        if unknown:
            raise RuntimeContractError(
                f"{cls.__name__}.from_config() received unknown keys: {unknown}. "
                f"Allowed keys: {sorted(allowed)}"
            )
        return _mark_runtime_construction(cls(**config), mode="config", config=config)  # type: ignore[misc]

    @classmethod
    def from_state(cls, state: PhysicsState) -> "BaseRuntimeSolver":
        solver = cls.from_demo_case()
        solver.set_input(state)
        return _mark_runtime_construction(solver, mode="state")

    def set_input(self, state: PhysicsState) -> None:
        if not isinstance(state, PhysicsState):
            raise RuntimeContractError("set_input() requires a PhysicsState instance.")
        self._input_state = state

    def solve(self, strict: bool = True) -> RuntimeExecutionResult:
        """Execute the solver and return a standardized runtime result.

        When `strict=True` (default), execution/contract failures raise.
        When `strict=False`, failures are converted into a non-success result.
        """
        start = time.perf_counter()
        captured_warnings: list[str] = []

        try:
            with warnings.catch_warnings(record=True) as warning_list:
                warnings.simplefilter("always")
                native_result = self._solve_native()
            captured_warnings = [str(item.message) for item in warning_list]

            state = self._build_state(native_result)
            if not isinstance(state, PhysicsState):
                raise RuntimeContractError(
                    f"{self.module_name}.to_state() must return PhysicsState, got {type(state).__name__}."
                )

            residuals, convergence = self._extract_runtime_metrics(native_result)
            result = RuntimeExecutionResult(
                module_name=self.module_name,
                success=True,
                status="success",
                warnings=captured_warnings,
                residuals=residuals,
                convergence=convergence,
                elapsed_time_s=time.perf_counter() - start,
                result_payload=native_result,
                generated_state=state,
                description=self.describe(),
            )
            self._last_result = result
            return result
        except Exception as exc:
            failure_kind = "contract_error" if isinstance(exc, RuntimeContractError) else "execution_error"
            result = RuntimeExecutionResult(
                module_name=self.module_name,
                success=False,
                status=failure_kind,
                warnings=captured_warnings,
                residuals={},
                convergence={},
                elapsed_time_s=time.perf_counter() - start,
                result_payload=None,
                generated_state=None,
                description=self.describe(),
                error=str(exc),
            )
            self._last_result = result
            if strict:
                if isinstance(exc, RuntimeContractError):
                    raise
                raise RuntimeExecutionError(f"{self.module_name}.solve() failed.") from exc
            return result

    def solve_safe(self) -> RuntimeExecutionResult:
        return self.solve(strict=False)

    def to_state(self) -> PhysicsState:
        if self._last_result is None or not self._last_result.success or self._last_result.generated_state is None:
            raise RuntimeContractError("to_state() requires a prior successful solve().")
        return self._last_result.generated_state

    def get_output(self) -> PhysicsState:
        return self.to_state()

    def describe(self) -> RuntimeDescription:
        description: RuntimeDescription = {
            "module_name": self.module_name,
            "domain": self.domain,
            "fidelity_level": self.fidelity_level,
            "coupling_ready": self.coupling_ready,
            "supports_transient": self.supports_transient,
            "supports_steady": self.supports_steady,
            "supports_demo_case": self.supports_demo_case,
            "construction_mode": self._runtime_origin,
            "required_inputs": [dict(field) for field in self.required_inputs],
            "output_fields": [dict(field) for field in self.output_fields],
            "validation_status": self.validation_status,
            "contract_version": self.contract_version,
        }
        return description

    def accepts(self, field_name: str) -> bool:
        return any(field["name"] == field_name for field in self.required_inputs)

    def outputs(self, field_name: str) -> bool:
        return any(field["name"] == field_name for field in self.output_fields)

    def _extract_runtime_metrics(self, native_result: Any) -> tuple[Dict[str, float], Dict[str, Any]]:
        residuals: Dict[str, float] = {}
        convergence: Dict[str, Any] = {}
        for attr in ("residual", "divergence_rms", "max_residual"):
            value = getattr(native_result, attr, None)
            if value is not None:
                try:
                    residuals[attr] = float(value)
                except (TypeError, ValueError):
                    pass
        for attr in ("converged", "iterations", "steady_state_reached", "min_dnbr"):
            value = getattr(native_result, attr, None)
            if value is not None:
                convergence[attr] = value
        return residuals, convergence

    @abstractmethod
    def _solve_native(self) -> Any:
        """Run the module-native solve path."""

    @abstractmethod
    def _build_state(self, native_result: Any) -> PhysicsState:
        """Convert the native result into a PhysicsState."""


def _ensure_mapping(config: Optional[Mapping[str, Any]], *, context: str) -> Dict[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise RuntimeContractError(f"{context} must be a mapping/dict, got {type(config).__name__}.")
    return dict(config)


def _reject_unknown_keys(config: Mapping[str, Any], *, allowed: Iterable[str], context: str) -> None:
    unknown = sorted(set(config) - set(allowed))
    if unknown:
        raise RuntimeContractError(
            f"{context} received unknown keys: {unknown}. Allowed keys: {sorted(set(allowed))}"
        )


def _extract_config_section(
    config: Mapping[str, Any],
    *,
    section_name: str,
    section_keys: Iterable[str],
) -> Dict[str, Any]:
    section_keys = set(section_keys)
    nested = _ensure_mapping(config.get(section_name), context=f"{section_name} section")
    _reject_unknown_keys(nested, allowed=section_keys, context=f"{section_name} section")

    flat = {key: value for key, value in config.items() if key in section_keys}
    overlap = sorted(set(nested) & set(flat))
    if overlap:
        raise RuntimeContractError(
            f"Config duplicates keys between top-level and '{section_name}' section: {overlap}."
        )
    return {**flat, **nested}


def _mark_runtime_construction(
    solver: "BaseRuntimeSolver",
    *,
    mode: str,
    config: Optional[Mapping[str, Any]] = None,
) -> "BaseRuntimeSolver":
    solver._runtime_origin = mode
    solver._runtime_config = dict(config or {})
    return solver


@dataclass
class RuntimeModuleHandle:
    """User-facing loader handle returned by `load_module()`."""

    name: str
    solver_cls: Type[BaseRuntimeSolver]
    metadata_factory: Callable[[], Dict[str, Any]] = field(default=lambda: {})

    def from_demo_case(self) -> BaseRuntimeSolver:
        solver = self.solver_cls.from_demo_case()
        validate_runtime_solver(solver)
        return solver

    def demo(self) -> BaseRuntimeSolver:
        return self.from_demo_case()

    def from_config(self, config: Optional[Dict[str, Any]] = None) -> BaseRuntimeSolver:
        solver = self.solver_cls.from_config(config)
        validate_runtime_solver(solver)
        return solver

    def from_state(self, state: PhysicsState) -> BaseRuntimeSolver:
        solver = self.solver_cls.from_state(state)
        validate_runtime_solver(solver)
        return solver

    def describe(self) -> RuntimeDescription:
        solver = self.from_demo_case()
        base = solver.describe()
        extra = self.metadata_factory()
        return RuntimeDescription(**{**base, **extra})


def validate_runtime_solver(solver: BaseRuntimeSolver) -> None:
    """Validate the static parts of the runtime contract."""
    if not isinstance(solver, BaseRuntimeSolver):
        raise RuntimeContractError("Runtime loader must return a BaseRuntimeSolver instance.")
    description = solver.describe()
    required_keys = {
        "module_name",
        "domain",
        "fidelity_level",
        "contract_version",
        "supports_transient",
        "supports_steady",
        "supports_demo_case",
        "construction_mode",
        "coupling_ready",
        "required_inputs",
        "output_fields",
        "validation_status",
    }
    missing = sorted(required_keys - set(description))
    if missing:
        raise RuntimeContractError(f"{solver.module_name}.describe() missing required keys: {missing}")


class NavierStokesRuntimeSolver(BaseRuntimeSolver):
    module_name = "navier_stokes"
    domain = "fluid_dynamics"
    fidelity_level = "L3"
    coupling_ready = "M2"
    supports_transient = True
    supports_steady = False
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "velocity_x", "units": "m/s", "kind": "field", "required": False},
        {"name": "velocity_y", "units": "m/s", "kind": "field", "required": False},
        {"name": "pressure", "units": "Pa", "kind": "field", "required": False},
    )
    _solver_keys = ("nx", "ny", "Lx", "Ly", "rho", "nu", "U_lid", "cfl", "quasi_3d", "z_length")
    _solve_keys = ("t_end", "dt", "max_steps", "pressure_iters", "pressure_tol")

    def __init__(
        self,
        nx: int = 12,
        ny: int = 12,
        Lx: float = 1.0,
        Ly: float = 1.0,
        rho: float = 1.0,
        nu: float = 1.0e-2,
        U_lid: float = 0.1,
        cfl: float = 0.5,
        quasi_3d: bool = False,
        z_length: float = 1.0,
        *,
        solve_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        from triality.navier_stokes.solver import NavierStokes2DSolver

        self._solver = NavierStokes2DSolver(
            nx=nx, ny=ny, Lx=Lx, Ly=Ly, rho=rho, nu=nu, U_lid=U_lid, cfl=cfl,
            quasi_3d=quasi_3d, z_length=z_length,
        )
        default_solve = {
            "t_end": 0.01,
            "dt": 0.002,
            "max_steps": 20,
            "pressure_iters": 40,
            "pressure_tol": 1e-4,
        }
        self._solve_kwargs = {**default_solve, **dict(solve_config or {})}

    @classmethod
    def from_demo_case(cls) -> "NavierStokesRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "NavierStokesRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, *cls._solve_keys, "solver", "solve"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        solve_cfg = _extract_config_section(config, section_name="solve", section_keys=cls._solve_keys)
        return _mark_runtime_construction(cls(**solver_cfg, solve_config=solve_cfg), mode="config", config=config)

    def _solve_native(self):
        return self._solver.solve(**self._solve_kwargs, progress_callback=self.report_progress)

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)




class DriftDiffusionRuntimeSolver(BaseRuntimeSolver):
    module_name = "drift_diffusion"
    domain = "semiconductor_devices"
    fidelity_level = "L3"
    coupling_ready = "M2"
    supports_transient = False
    supports_steady = True
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "electrostatic_potential", "units": "V", "kind": "field", "required": False},
        {"name": "electron_density", "units": "cm^-3", "kind": "field", "required": False},
        {"name": "hole_density", "units": "cm^-3", "kind": "field", "required": False},
    )
    _solver_keys = ("length", "n_points", "material", "temperature", "enable_srh", "enable_field_mobility", "tau_n", "tau_p")
    _solve_keys = ("applied_voltage", "max_iterations", "tolerance", "under_relaxation")
    _doping_keys = ("type", "N_d_level", "N_a_level", "junction_pos")

    def __init__(self, *, doping_config: Optional[Mapping[str, Any]] = None, solve_config: Optional[Mapping[str, Any]] = None, **solver_kwargs: Any):
        super().__init__()
        from triality.drift_diffusion.solver import DriftDiffusionSolverV2

        default_solver = {
            "length": 1e-4,
            "n_points": 60,
            "material": "Silicon",
            "temperature": 300.0,
            "enable_srh": False,
            "enable_field_mobility": False,
        }
        self._solver = DriftDiffusionSolverV2(**{**default_solver, **solver_kwargs})

        default_doping = {"type": "pn_junction", "N_d_level": 1e17, "N_a_level": 5e16, "junction_pos": None}
        self._doping_config = {**default_doping, **dict(doping_config or {})}
        self._apply_doping_config(self._doping_config)

        default_solve = {"applied_voltage": 0.0, "max_iterations": 50, "tolerance": 1e-5, "under_relaxation": 0.4}
        self._solve_kwargs = {**default_solve, **dict(solve_config or {})}

    def _apply_doping_config(self, doping_config: Mapping[str, Any]) -> None:
        doping_type = doping_config.get("type", "pn_junction")
        # Accept common aliases
        if doping_type in ("pn", "pn-junction", "pnjunction"):
            doping_type = "pn_junction"
        if doping_type != "pn_junction":
            raise RuntimeContractError(
                f"{self.__class__.__name__} only supports doping.type='pn_junction' in from_config(), got {doping_type!r}."
            )
        self._solver.set_pn_junction(
            N_d_level=float(doping_config.get("N_d_level", 1e17)),
            N_a_level=float(doping_config.get("N_a_level", 5e16)),
            junction_pos=doping_config.get("junction_pos"),
        )

    @classmethod
    def from_demo_case(cls) -> "DriftDiffusionRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "DriftDiffusionRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        allowed = (*cls._solver_keys, *cls._solve_keys, *cls._doping_keys, "solver", "solve", "doping")
        _reject_unknown_keys(config, allowed=allowed, context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        solve_cfg = _extract_config_section(config, section_name="solve", section_keys=cls._solve_keys)
        doping_cfg = _extract_config_section(config, section_name="doping", section_keys=cls._doping_keys)
        return _mark_runtime_construction(
            cls(doping_config=doping_cfg, solve_config=solve_cfg, **solver_cfg),
            mode="config",
            config=config,
        )

    def _solve_native(self):
        return self._solver.solve(**self._solve_kwargs)

    def _build_state(self, native_result: Any) -> PhysicsState:
        state = PhysicsState(solver_name="drift_diffusion")
        # Core spatial fields
        state.set_field("electrostatic_potential", native_result.V, "V")
        state.set_field("electron_density", native_result.n, "cm^-3")
        state.set_field("hole_density", native_result.p, "cm^-3")
        # Transport fields — these reveal actual device behavior
        state.set_field("electric_field", native_result.E_field, "V/cm")
        state.set_field("current_density_total", native_result.J_total, "A/cm^2")
        state.set_field("electron_current_density", native_result.J_n, "A/cm^2")
        state.set_field("hole_current_density", native_result.J_p, "A/cm^2")
        # Scalar metadata
        state.metadata["built_in_potential"] = native_result.built_in_potential
        state.metadata["depletion_width"] = native_result.depletion_width
        state.metadata["max_field"] = native_result.max_field
        state.metadata["current_density"] = native_result.current_density
        state.metadata["converged"] = native_result.converged
        state.metadata["temperature"] = native_result.temperature
        state.metadata["material_name"] = native_result.material_name
        # Position array for physical x-axis
        state.metadata["position_cm"] = native_result.x.tolist()
        # I-V curve data when available
        if native_result.iv_voltages is not None:
            state.metadata["iv_voltages"] = native_result.iv_voltages.tolist()
            state.metadata["iv_currents"] = native_result.iv_currents.tolist()
        return state


class SensingRuntimeSolver(BaseRuntimeSolver):
    module_name = "sensing"
    domain = "sensing"
    fidelity_level = "L2"
    coupling_ready = "M2"
    supports_transient = False
    supports_steady = True
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "detection_probability", "units": "1", "kind": "field", "required": False},
    )

    def __init__(
        self,
        grid_x_km: tuple[float, float] = (-10.0, 10.0),
        grid_y_km: tuple[float, float] = (-10.0, 10.0),
        grid_nx: int = 32,
        grid_ny: int = 32,
        sensor_name: str = "radar_1",
        sensor_location: tuple[float, float] = (0.0, 0.0),
        *,
        radar_config: Optional[Mapping[str, Any]] = None,
        target_config: Optional[Mapping[str, Any]] = None,
        solve_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        from triality.sensing.radar import RadarSystem
        from triality.sensing.solver import (
            SensorPerformanceConfig,
            SensorPerformanceSolver,
            SensorSpec,
            SensorType,
            TargetSpec,
        )

        radar_defaults = {
            "frequency_ghz": 10.0,
            "power_w": 1000.0,
            "pulse_width_us": 1.0,
            "prf_hz": 1000.0,
            "aperture_diameter_m": 1.0,
            "bandwidth_mhz": 10.0,
            "noise_figure_db": 3.0,
            "n_pulses_integrated": 10,
            "losses_db": 3.0,
        }
        target_defaults = {"rcs_m2": 1.0, "target_strength_db": 10.0}
        cfg_defaults = {
            "pfa": 1e-6,
            "weather": "clear",
            "water_temp_c": 15.0,
        }

        radar = RadarSystem(**{**radar_defaults, **dict(radar_config or {})})
        target = TargetSpec(**{**target_defaults, **dict(target_config or {})})
        cfg = SensorPerformanceConfig(
            sensors=[SensorSpec(name=sensor_name, sensor_type=SensorType.RADAR, location=sensor_location, radar=radar)],
            target=target,
            grid_x_km=grid_x_km,
            grid_y_km=grid_y_km,
            grid_nx=grid_nx,
            grid_ny=grid_ny,
            **{**cfg_defaults, **dict(solve_config or {})},
        )
        self._solver = SensorPerformanceSolver(cfg)

    @classmethod
    def from_demo_case(cls) -> "SensingRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    # Keys that belong at the top level, NOT inside radar/target
    _TOP_LEVEL_KEYS = {
        "grid_x_km", "grid_y_km", "grid_nx", "grid_ny",
        "auto_grid", "sensor_name", "sensor_location",
        "pfa", "weather", "water_temp_c",
    }

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "SensingRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        # Flatten any "grid" or "top_level" section into top-level keys
        for section_key in ("grid", "top_level"):
            nested = config.pop(section_key, None)
            if isinstance(nested, dict):
                config = {**config, **nested}
        # Extract top-level keys that the LLM may have accidentally nested
        # inside "radar" or "target" (e.g., auto_grid inside radar)
        for nested_key in ("radar", "target"):
            nested = config.get(nested_key)
            if isinstance(nested, dict):
                misplaced = {k: v for k, v in nested.items() if k in cls._TOP_LEVEL_KEYS}
                for k in misplaced:
                    config.setdefault(k, nested.pop(k))
        allowed = {
            *cls._TOP_LEVEL_KEYS,
            "radar", "target", "solve",
        }
        _reject_unknown_keys(config, allowed=allowed, context=f"{cls.__name__}.from_config()")

        grid_x_km = tuple(config.get("grid_x_km", (-10.0, 10.0)))
        grid_y_km = tuple(config.get("grid_y_km", (-10.0, 10.0)))
        grid_nx = int(config.get("grid_nx", 32))
        grid_ny = int(config.get("grid_ny", 32))

        # Auto-size grid: estimate max detection range, then set grid to ±1.5*R_max
        if config.get("auto_grid", False):
            grid_x_km, grid_y_km = cls._auto_size_grid(
                config, grid_nx, grid_ny
            )

        return _mark_runtime_construction(
            cls(
                grid_x_km=grid_x_km,
                grid_y_km=grid_y_km,
                grid_nx=grid_nx,
                grid_ny=grid_ny,
                sensor_name=config.get("sensor_name", "radar_1"),
                sensor_location=tuple(config.get("sensor_location", (0.0, 0.0))),
                radar_config=config.get("radar"),
                target_config=config.get("target"),
                solve_config={
                    **dict(config.get("solve") or {}),
                    **{k: config[k] for k in ("pfa", "weather", "water_temp_c") if k in config},
                },
            ),
            mode="config",
            config=config,
        )

    @staticmethod
    def _auto_size_grid(
        config: Dict[str, Any],
        grid_nx: int,
        grid_ny: int,
    ) -> tuple:
        """Estimate max detection range from radar equation and size grid to ±1.5*R_max."""
        import numpy as np
        from triality.sensing.signals import C_LIGHT

        radar_defaults = {
            "frequency_ghz": 10.0, "power_w": 1000.0, "aperture_diameter_m": 1.0,
            "bandwidth_mhz": 10.0, "noise_figure_db": 3.0, "n_pulses_integrated": 10,
            "losses_db": 3.0,
        }
        r = {**radar_defaults, **dict(config.get("radar") or {})}
        rcs = (config.get("target") or {}).get("rcs_m2", 1.0)

        freq_hz = r["frequency_ghz"] * 1e9
        wavelength = C_LIGHT / freq_hz
        aperture_area = np.pi * (r["aperture_diameter_m"] / 2.0) ** 2
        gain_lin = 4.0 * np.pi * aperture_area * 0.55 / (wavelength ** 2)

        k_B = 1.380649e-23
        T0 = 290.0
        bandwidth_hz = r["bandwidth_mhz"] * 1e6
        nf_lin = 10 ** (r["noise_figure_db"] / 10.0)
        N = k_B * T0 * bandwidth_hz * nf_lin
        losses_lin = 10 ** (r["losses_db"] / 10.0)

        numerator = r["power_w"] * gain_lin ** 2 * wavelength ** 2 * rcs
        req_snr_lin = 10 ** (13.0 / 10.0)  # 13 dB threshold
        R_max_m = (numerator / ((4.0 * np.pi) ** 3 * N * losses_lin * req_snr_lin)) ** 0.25
        R_max_km = R_max_m / 1000.0

        margin = R_max_km * 1.5
        margin = max(margin, 5.0)  # minimum 5 km half-extent
        return (-margin, margin), (-margin, margin)

    def _solve_native(self):
        return self._solver.solve()

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# Electrostatics Runtime Adapter
# ---------------------------------------------------------------------------

class ElectrostaticsRuntimeSolver(BaseRuntimeSolver):
    module_name = "electrostatics"
    domain = "electromagnetism"
    fidelity_level = "L2"
    coupling_ready = "M1"
    supports_transient = False
    supports_steady = True
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "electric_potential", "units": "V", "kind": "field", "required": False},
        {"name": "electric_field_x", "units": "V/m", "kind": "field", "required": False},
        {"name": "electric_field_y", "units": "V/m", "kind": "field", "required": False},
        {"name": "electric_field_magnitude", "units": "V/m", "kind": "field", "required": False},
    )
    _solver_keys = ("x_min", "x_max", "y_min", "y_max", "resolution", "permittivity", "mode")
    _solve_keys = ("method", "boundary_value")

    def __init__(
        self,
        x_min: float = 0.0,
        x_max: float = 0.1,
        y_min: float = 0.0,
        y_max: float = 0.1,
        resolution: int = 50,
        permittivity: float = 8.854e-12,
        mode: str = "electrostatic",
        *,
        solve_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        from triality.electrostatics.solver import ElectrostaticsSolver

        self._solver = ElectrostaticsSolver(
            x_range=(x_min, x_max),
            y_range=(y_min, y_max),
            resolution=resolution,
            permittivity=permittivity,
            mode=mode,
        )
        cfg = dict(solve_config or {})
        bv = cfg.pop("boundary_value", 1000.0)
        dx = (x_max - x_min) / max(resolution - 1, 1)
        # Left boundary at boundary_value, right boundary at 0V — creates a field
        self._solver.add_boundary("dirichlet", value=float(bv),
                                  region=lambda x, y, xm=x_min, d=dx: x < xm + d)
        self._solver.add_boundary("grounded",
                                  region=lambda x, y, xM=x_max, d=dx: x > xM - d)
        self._solve_method = cfg.get("method", "gmres")

    @classmethod
    def from_demo_case(cls) -> "ElectrostaticsRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "ElectrostaticsRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, *cls._solve_keys, "solver", "solve"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        solve_cfg = _extract_config_section(config, section_name="solve", section_keys=cls._solve_keys)
        return _mark_runtime_construction(cls(**solver_cfg, solve_config=solve_cfg), mode="config", config=config)

    def _solve_native(self):
        return self._solver.solve(method=self._solve_method)

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# Aero Loads Runtime Adapter
# ---------------------------------------------------------------------------

class AeroLoadsRuntimeSolver(BaseRuntimeSolver):
    module_name = "aero_loads"
    domain = "aerodynamics"
    fidelity_level = "L2"
    coupling_ready = "M3"
    supports_transient = False
    supports_steady = True
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "pressure_coefficient", "units": "1", "kind": "field", "required": False},
        {"name": "heat_flux", "units": "W/m^2", "kind": "field", "required": False},
        {"name": "lift_coefficient", "units": "1", "kind": "scalar", "required": False},
        {"name": "drag_coefficient", "units": "1", "kind": "scalar", "required": False},
    )
    _solver_keys = ("body_length_m", "nose_radius_m", "base_radius_m", "n_panels", "reference_area_m2", "reference_length_m")
    _solve_keys = ("velocity", "density", "temperature", "pressure", "alpha_deg", "wall_temperature_K")

    def __init__(
        self,
        body_length_m: float = 3.0,
        nose_radius_m: float = 0.15,
        base_radius_m: float = 0.15,
        n_panels: int = 100,
        reference_area_m2: Optional[float] = None,
        reference_length_m: Optional[float] = None,
        *,
        solve_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        from triality.aero_loads.solver import AeroLoadsSolver

        self._solver = AeroLoadsSolver(
            body_length_m=body_length_m,
            nose_radius_m=nose_radius_m,
            base_radius_m=base_radius_m,
            n_panels=n_panels,
            reference_area_m2=reference_area_m2,
            reference_length_m=reference_length_m,
        )
        sc = dict(solve_config or {})
        self._velocity = sc.get("velocity", 2000.0)
        self._mach = sc.get("mach_number", 5.0)
        self._density = sc.get("density", 0.1)
        self._temperature = sc.get("temperature", 300.0)
        self._pressure = sc.get("pressure", 1e4)
        self._alpha = sc.get("alpha_deg", 5.0)
        self._wall_T = sc.get("wall_temperature_K", 300.0)

    @classmethod
    def from_demo_case(cls) -> "AeroLoadsRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "AeroLoadsRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, *cls._solve_keys, "solver", "solve"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        solve_cfg = _extract_config_section(config, section_name="solve", section_keys=cls._solve_keys)
        return _mark_runtime_construction(cls(**solver_cfg, solve_config=solve_cfg), mode="config", config=config)

    def _solve_native(self):
        from triality.aero_loads.newtonian_flow import FreestreamConditions

        freestream = FreestreamConditions(
            velocity=self._velocity,
            density=self._density,
            temperature=self._temperature,
            pressure=self._pressure,
        )
        return self._solver.solve(freestream, alpha_deg=self._alpha, wall_temperature_K=self._wall_T)

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# UAV Aerodynamics Runtime Adapter
# ---------------------------------------------------------------------------

class UAVAerodynamicsRuntimeSolver(BaseRuntimeSolver):
    module_name = "uav_aerodynamics"
    domain = "aerodynamics"
    fidelity_level = "L2"
    coupling_ready = "M1"
    supports_transient = False
    supports_steady = True
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "lift_coefficient", "units": "1", "kind": "scalar", "required": False},
        {"name": "induced_drag_coefficient", "units": "1", "kind": "scalar", "required": False},
        {"name": "span_circulation", "units": "m^2/s", "kind": "field", "required": False},
    )
    _solver_keys = ("span", "root_chord", "tip_chord", "n_span", "alpha_deg", "V_inf", "rho")

    def __init__(
        self,
        span: float = 10.0,
        root_chord: float = 1.0,
        tip_chord: Optional[float] = None,
        n_span: int = 40,
        alpha_deg: float = 5.0,
        V_inf: float = 30.0,
        rho: float = 1.225,
    ):
        super().__init__()
        from triality.uav_aerodynamics.solver import VortexLatticeSolver

        self._solver = VortexLatticeSolver(
            span=span,
            root_chord=root_chord,
            tip_chord=tip_chord,
            n_span=n_span,
            alpha_deg=alpha_deg,
            V_inf=V_inf,
            rho=rho,
        )

    @classmethod
    def from_demo_case(cls) -> "UAVAerodynamicsRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "UAVAerodynamicsRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, "solver"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        return _mark_runtime_construction(cls(**solver_cfg), mode="config", config=config)

    def _solve_native(self):
        return self._solver.solve()

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# Spacecraft Thermal Runtime Adapter
# ---------------------------------------------------------------------------

class SpacecraftThermalRuntimeSolver(BaseRuntimeSolver):
    module_name = "spacecraft_thermal"
    domain = "thermal"
    fidelity_level = "L2"
    coupling_ready = "M3"
    supports_transient = True
    supports_steady = False
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "temperature", "units": "K", "kind": "field", "required": False},
        {"name": "heater_power", "units": "W", "kind": "field", "required": False},
    )
    _solver_keys = ("n_nodes", "T_space", "T_min_limit", "T_max_limit", "internal_power_W")
    _solve_keys = ("t_end", "dt")

    def __init__(
        self,
        n_nodes: int = 4,
        T_space: float = 4.0,
        T_min_limit: float = 233.0,
        T_max_limit: float = 333.0,
        internal_power_W: float = 50.0,
        *,
        solve_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        from triality.spacecraft_thermal.solver import (
            SpacecraftThermalSolver,
            ThermalNode,
            ConductiveLink,
        )
        from triality.spacecraft_thermal.radiative_transfer import Surface

        nodes = []
        for i in range(n_nodes):
            pwr = internal_power_W if i == n_nodes - 1 else 0.0
            nodes.append(ThermalNode(
                name=f"node_{i}",
                mass=5.0,
                specific_heat=900.0,
                surface=Surface(area=1.0, emissivity=0.85, absorptivity=0.3, temperature=293.0),
                internal_power=pwr,
            ))
        links = [ConductiveLink(node_a=i, node_b=i + 1, conductance=10.0) for i in range(n_nodes - 1)]
        self._solver = SpacecraftThermalSolver(
            nodes=nodes,
            conductive_links=links,
            T_space=T_space,
            T_min_limit=T_min_limit,
            T_max_limit=T_max_limit,
        )
        sc = dict(solve_config or {})
        self._t_end = sc.get("t_end", 3600.0)
        self._dt = sc.get("dt", 10.0)
        self._sun_angle = sc.get("sun_angle", 0.0)

    @classmethod
    def from_demo_case(cls) -> "SpacecraftThermalRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "SpacecraftThermalRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, *cls._solve_keys, "solver", "solve"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        solve_cfg = _extract_config_section(config, section_name="solve", section_keys=cls._solve_keys)
        return _mark_runtime_construction(cls(**solver_cfg, solve_config=solve_cfg), mode="config", config=config)

    def _solve_native(self):
        return self._solver.solve(t_end=self._t_end, dt=self._dt, progress_callback=self.report_progress)

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# Automotive Thermal Runtime Adapter
# ---------------------------------------------------------------------------

class AutomotiveThermalRuntimeSolver(BaseRuntimeSolver):
    module_name = "automotive_thermal"
    domain = "thermal"
    fidelity_level = "L1"
    coupling_ready = "M1"
    supports_transient = True
    supports_steady = False
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "temperature", "units": "K", "kind": "field", "required": False},
        {"name": "heat_generation", "units": "W", "kind": "field", "required": False},
    )
    _solver_keys = ("n_components", "T_ambient", "current_A", "h_convection")
    _solve_keys = ("t_end", "dt")

    def __init__(
        self,
        n_components: int = 3,
        T_ambient: float = 298.0,
        current_A: float = 200.0,
        h_convection: float = 100.0,
        *,
        solve_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        from triality.automotive_thermal.solver import (
            AutomotiveThermalSolver,
            ComponentNode,
            ThermalContact,
        )
        from triality.automotive_thermal.transient_heat import (
            COPPER,
            HeatSource,
        )

        nodes = []
        for i in range(n_components):
            hs = [HeatSource(position=(0.0, 0.0, 0.0), power=current_A**2 * 0.001)] if i == 0 else []
            nodes.append(ComponentNode(
                name=f"component_{i}",
                material=COPPER,
                mass=0.1 * (i + 1),
                surface_area=0.05,
                heat_sources=hs,
                h_convection=h_convection,
            ))
        contacts = [ThermalContact(node_a=i, node_b=i + 1, conductance=500.0) for i in range(n_components - 1)]
        self._solver = AutomotiveThermalSolver(nodes=nodes, contacts=contacts, T_ambient=T_ambient)
        sc = dict(solve_config or {})
        self._t_end = sc.get("t_end", 60.0)
        self._dt = sc.get("dt", 0.1)

    @classmethod
    def from_demo_case(cls) -> "AutomotiveThermalRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "AutomotiveThermalRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, *cls._solve_keys, "solver", "solve"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        solve_cfg = _extract_config_section(config, section_name="solve", section_keys=cls._solve_keys)
        return _mark_runtime_construction(cls(**solver_cfg, solve_config=solve_cfg), mode="config", config=config)

    def _solve_native(self):
        return self._solver.solve(t_end=self._t_end, dt=self._dt, progress_callback=self.report_progress)

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# Battery Thermal Runtime Adapter
# ---------------------------------------------------------------------------

class BatteryThermalRuntimeSolver(BaseRuntimeSolver):
    module_name = "battery_thermal"
    domain = "thermal"
    fidelity_level = "L2"
    coupling_ready = "M3"
    supports_transient = True
    supports_steady = False
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "temperature", "units": "K", "kind": "field", "required": False},
    )
    _solver_keys = ("n_cells", "cooling_type", "cell_chemistry", "T_init")
    _solve_keys = ("discharge_current_A", "duration_s", "dt")

    def __init__(
        self,
        n_cells: int = 96,
        cooling_type: str = "liquid_indirect",
        cell_chemistry: str = "NMC",
        T_init: float = 298.0,
        *,
        solve_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        from triality.battery_thermal.solver import BatteryThermalSolver, DriveCycleSegment
        from triality.battery_thermal.pack_thermal import PackConfiguration, CellChemistry, CoolingType

        cooling_map = {
            "liquid_indirect": CoolingType.LIQUID_INDIRECT,
            "liquid_immersion": CoolingType.LIQUID_IMMERSION,
            "air_forced": CoolingType.AIR_FORCED,
            "air_natural": CoolingType.AIR_NATURAL,
        }
        chem_map = {
            "NMC": CellChemistry.NMC,
            "NCA": CellChemistry.NCA,
            "LFP": CellChemistry.LFP,
        }
        pack_config = PackConfiguration(
            n_cells=n_cells,
            cell_spacing=0.002,
            cooling_type=cooling_map.get(cooling_type, CoolingType.LIQUID_INDIRECT),
        )
        self._solver = BatteryThermalSolver(
            config=pack_config,
            cell_chemistry=chem_map.get(cell_chemistry, CellChemistry.NMC),
            T_init=T_init,
        )
        sc = dict(solve_config or {})
        self._current = sc.get("discharge_current_A", 50.0)
        self._duration = sc.get("duration_s", 600.0)
        self._dt = sc.get("dt", 0.5)
        self._DriveCycleSegment = DriveCycleSegment

    @classmethod
    def from_demo_case(cls) -> "BatteryThermalRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "BatteryThermalRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, *cls._solve_keys, "solver", "solve"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        solve_cfg = _extract_config_section(config, section_name="solve", section_keys=cls._solve_keys)
        return _mark_runtime_construction(cls(**solver_cfg, solve_config=solve_cfg), mode="config", config=config)

    def _solve_native(self):
        segments = [self._DriveCycleSegment(duration=self._duration, current=self._current, label="runtime")]
        return self._solver.solve(segments=segments, dt=self._dt, progress_callback=self.report_progress)

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# Structural Analysis Runtime Adapter
# ---------------------------------------------------------------------------

class StructuralAnalysisRuntimeSolver(BaseRuntimeSolver):
    module_name = "structural_analysis"
    domain = "structures"
    fidelity_level = "L2"
    coupling_ready = "M3"
    supports_transient = False
    supports_steady = True
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "displacement", "units": "m", "kind": "field", "required": False},
        {"name": "stress", "units": "Pa", "kind": "field", "required": False},
    )
    _solver_keys = ("material_name", "length", "n_elements", "I", "A", "c", "support")
    _solve_keys = ("tip_force_N", "distributed_load_N_m")

    def __init__(
        self,
        material_name: str = "AL7075-T6",
        length: float = 1.0,
        n_elements: int = 20,
        I: float = 1e-6,
        A: float = 1e-4,
        c: float = 0.01,
        support: str = "cantilever",
        *,
        solve_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        from triality.structural_analysis.solver import StructuralSolver
        from triality.structural_analysis.static_solver import MATERIALS

        material = MATERIALS.get(material_name, MATERIALS["AL7075-T6"])
        self._solver = StructuralSolver(
            material=material,
            length=length,
            n_elements=n_elements,
            I=I,
            A=A,
            c=c,
            support=support,
        )
        sc = dict(solve_config or {})
        tip_force = sc.get("tip_force_N", -5000.0)
        dist_load = sc.get("distributed_load_N_m", 0.0)
        if tip_force != 0.0:
            self._solver.add_tip_load("runtime_tip", force=float(tip_force))
        if dist_load != 0.0:
            self._solver.add_distributed_load("runtime_dist", w=float(dist_load))
        if tip_force == 0.0 and dist_load == 0.0:
            self._solver.add_tip_load("runtime_tip", force=-5000.0)

    @classmethod
    def from_demo_case(cls) -> "StructuralAnalysisRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "StructuralAnalysisRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, *cls._solve_keys, "solver", "solve"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        solve_cfg = _extract_config_section(config, section_name="solve", section_keys=cls._solve_keys)
        return _mark_runtime_construction(cls(**solver_cfg, solve_config=solve_cfg), mode="config", config=config)

    def _solve_native(self):
        return self._solver.solve()

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# Structural Dynamics Runtime Adapter
# ---------------------------------------------------------------------------

class StructuralDynamicsRuntimeSolver(BaseRuntimeSolver):
    module_name = "structural_dynamics"
    domain = "structures"
    fidelity_level = "L3"
    coupling_ready = "M3"
    supports_transient = True
    supports_steady = False
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "displacement", "units": "m", "kind": "field", "required": False},
        {"name": "velocity", "units": "m/s", "kind": "field", "required": False},
        {"name": "acceleration", "units": "m/s^2", "kind": "field", "required": False},
    )
    _solver_keys = ("n_dof", "stiffness_diag", "damping_ratio", "force_amplitude", "force_frequency")
    _solve_keys = ("t_end", "dt", "compute_srs")

    def __init__(
        self,
        n_dof: int = 3,
        stiffness_diag: Optional[list] = None,
        damping_ratio: float = 0.02,
        force_amplitude: float = 100.0,
        force_frequency: float = 10.0,
        *,
        solve_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        from triality.structural_dynamics.solver import StructuralDynamicsSolver
        from triality.structural_dynamics.modal_analysis import StructuralModel

        if stiffness_diag is None:
            stiffness_diag = [1000.0 * (i + 1) for i in range(n_dof)]
        M = np.eye(n_dof)
        K = np.diag(stiffness_diag[:n_dof])

        model = StructuralModel(mass_matrix=M, stiffness_matrix=K)
        amp = force_amplitude
        freq = force_frequency

        def force_func(t: float) -> np.ndarray:
            f = np.zeros(n_dof)
            f[0] = amp * np.sin(2.0 * np.pi * freq * t)
            return f

        self._solver = StructuralDynamicsSolver(
            model=model,
            force_func=force_func,
            damping_ratio=damping_ratio,
        )
        sc = dict(solve_config or {})
        self._t_end = sc.get("t_end", 2.0)
        self._dt = sc.get("dt", 0.001)
        self._compute_srs = sc.get("compute_srs", False)

    @classmethod
    def from_demo_case(cls) -> "StructuralDynamicsRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "StructuralDynamicsRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, *cls._solve_keys, "solver", "solve"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        solve_cfg = _extract_config_section(config, section_name="solve", section_keys=cls._solve_keys)
        return _mark_runtime_construction(cls(**solver_cfg, solve_config=solve_cfg), mode="config", config=config)

    def _solve_native(self):
        return self._solver.solve(t_end=self._t_end, dt=self._dt, compute_srs=self._compute_srs, progress_callback=self.report_progress)

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# Flight Mechanics Runtime Adapter
# ---------------------------------------------------------------------------

class FlightMechanicsRuntimeSolver(BaseRuntimeSolver):
    module_name = "flight_mechanics"
    domain = "dynamics"
    fidelity_level = "L2"
    coupling_ready = "M1"
    supports_transient = True
    supports_steady = False
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "position", "units": "m", "kind": "field", "required": False},
        {"name": "velocity", "units": "m/s", "kind": "field", "required": False},
        {"name": "euler_angles", "units": "rad", "kind": "field", "required": False},
    )
    _solver_keys = ("mass", "Ixx", "Iyy", "Izz", "dt", "gravity")
    _solve_keys = ("t_final", "omega_x", "omega_y", "omega_z")

    def __init__(
        self,
        mass: float = 500.0,
        Ixx: float = 100.0,
        Iyy: float = 120.0,
        Izz: float = 80.0,
        dt: float = 0.01,
        gravity: bool = True,
        *,
        solve_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        from triality.flight_mechanics.solver import FlightMechanicsSolver
        from triality.flight_mechanics.rigid_body_dynamics import RigidBodyState

        inertia = np.diag([Ixx, Iyy, Izz])
        sc = dict(solve_config or {})
        omega = np.array([
            sc.get("omega_x", 0.01),
            sc.get("omega_y", -0.02),
            sc.get("omega_z", 0.005),
        ])
        initial_state = RigidBodyState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=np.array([0.0, 0.0, 0.0, 1.0]),
            angular_velocity=omega,
        )
        self._solver = FlightMechanicsSolver(
            mass=mass,
            inertia=inertia,
            initial_state=initial_state,
            dt=dt,
            gravity=gravity,
        )
        self._t_final = sc.get("t_final", 60.0)

    @classmethod
    def from_demo_case(cls) -> "FlightMechanicsRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "FlightMechanicsRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, *cls._solve_keys, "solver", "solve"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        solve_cfg = _extract_config_section(config, section_name="solve", section_keys=cls._solve_keys)
        return _mark_runtime_construction(cls(**solver_cfg, solve_config=solve_cfg), mode="config", config=config)

    def _solve_native(self):
        return self._solver.solve(t_final=self._t_final, progress_callback=self.report_progress)

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# Coupled Physics (Neutronics-Thermal) Runtime Adapter
# ---------------------------------------------------------------------------

class CoupledPhysicsRuntimeSolver(BaseRuntimeSolver):
    module_name = "coupled_physics"
    domain = "nuclear"
    fidelity_level = "L2"
    coupling_ready = "M1"
    supports_transient = True
    supports_steady = False
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "power", "units": "W", "kind": "field", "required": False},
        {"name": "temperature_field", "units": "K", "kind": "field", "required": False},
        {"name": "reactivity", "units": "pcm", "kind": "field", "required": False},
    )
    _solver_keys = ("n_points", "length_cm", "beta_eff", "neutron_lifetime", "lambda_precursor", "feedback_mode")
    _solve_keys = ("t_end", "dt", "initial_power", "reactivity_insertion_pcm")

    def __init__(
        self,
        n_points: int = 50,
        length_cm: float = 200.0,
        beta_eff: float = 0.0065,
        neutron_lifetime: float = 2e-5,
        lambda_precursor: float = 0.08,
        feedback_mode: str = "full",
        *,
        solve_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        from triality.coupled_physics.solver import CoupledPhysicsSolver
        from triality.coupled_physics.neutronics_thermal_coupled import CoupledNeutronicsThermal, FeedbackMode

        mode_map = {
            "full": FeedbackMode.FULL_FEEDBACK,
            "doppler_only": FeedbackMode.DOPPLER_ONLY,
            "no_feedback": FeedbackMode.NO_FEEDBACK,
        }
        ss = CoupledNeutronicsThermal(
            length=length_cm,
            n_points=n_points,
            feedback_mode=mode_map.get(feedback_mode, FeedbackMode.FULL_FEEDBACK),
        )
        self._solver = CoupledPhysicsSolver(
            steady_state_solver=ss,
            beta_eff=beta_eff,
            neutron_lifetime=neutron_lifetime,
            lambda_precursor=lambda_precursor,
        )
        sc = dict(solve_config or {})
        self._t_end = sc.get("t_end", 10.0)
        self._dt = sc.get("dt", 0.01)
        self._initial_power = sc.get("initial_power", 1e6)
        self._rho_insert = sc.get("reactivity_insertion_pcm", 100.0)

    @classmethod
    def from_demo_case(cls) -> "CoupledPhysicsRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "CoupledPhysicsRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, *cls._solve_keys, "solver", "solve"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        solve_cfg = _extract_config_section(config, section_name="solve", section_keys=cls._solve_keys)
        return _mark_runtime_construction(cls(**solver_cfg, solve_config=solve_cfg), mode="config", config=config)

    def _solve_native(self):
        # Convert pcm to dk/k: 1 pcm = 1e-5 dk/k
        rho_dkk = self._rho_insert * 1e-5

        def reactivity_func(t: float) -> float:
            return rho_dkk if t < 1.0 else 0.0

        return self._solver.solve(
            t_end=self._t_end,
            dt=self._dt,
            initial_power=self._initial_power,
            reactivity_insertion=reactivity_func,
            progress_callback=self.report_progress,
        )

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# Geospatial Runtime Adapter
# ---------------------------------------------------------------------------

class GeospatialRuntimeSolver(BaseRuntimeSolver):
    module_name = "geospatial"
    domain = "logistics"
    fidelity_level = "L1"
    coupling_ready = "M1"
    supports_transient = False
    supports_steady = True
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "coverage_fraction", "units": "1", "kind": "scalar", "required": False},
        {"name": "travel_time_matrix", "units": "h", "kind": "field", "required": False},
    )
    _solver_keys = ("max_facilities", "time_limit_hours", "target_coverage", "road_type", "swap_iterations")

    # Default candidate locations (Indian cities for demo)
    _DEFAULT_CANDIDATES = [
        (28.6139, 77.2090),   # Delhi
        (19.0760, 72.8777),   # Mumbai
        (13.0827, 80.2707),   # Chennai
        (22.5726, 88.3639),   # Kolkata
        (12.9716, 77.5946),   # Bangalore
        (17.3850, 78.4867),   # Hyderabad
        (23.0225, 72.5714),   # Ahmedabad
        (26.9124, 75.7873),   # Jaipur
    ]

    def __init__(
        self,
        max_facilities: int = 3,
        time_limit_hours: float = 24.0,
        target_coverage: float = 0.95,
        road_type: str = "state_highway",
        swap_iterations: int = 50,
    ):
        super().__init__()
        from triality.geospatial.solver import GeospatialSolver, FacilityLocationConfig
        from triality.geospatial.travel_time import RoadType

        road_map = {
            "state_highway": RoadType.STATE_HIGHWAY,
            "national_highway": RoadType.NATIONAL_HIGHWAY,
            "rural": RoadType.RURAL,
            "urban": RoadType.URBAN,
            "mountainous": RoadType.MOUNTAINOUS,
        }
        config = FacilityLocationConfig(
            candidate_locations=list(self._DEFAULT_CANDIDATES),
            max_facilities=max_facilities,
            time_limit_hours=time_limit_hours,
            target_coverage=target_coverage,
            road_type=road_map.get(road_type, RoadType.STATE_HIGHWAY),
            swap_iterations=swap_iterations,
        )
        self._solver = GeospatialSolver(config)

    @classmethod
    def from_demo_case(cls) -> "GeospatialRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "GeospatialRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, "solver"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        return _mark_runtime_construction(cls(**solver_cfg), mode="config", config=config)

    def _solve_native(self):
        return self._solver.solve()

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# Field-Aware Routing Runtime Adapter
# ---------------------------------------------------------------------------

class FieldAwareRoutingRuntimeSolver(BaseRuntimeSolver):
    module_name = "field_aware_routing"
    domain = "electromagnetism"
    fidelity_level = "L2"
    coupling_ready = "M2"
    supports_transient = False
    supports_steady = True
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "electric_potential", "units": "V", "kind": "field", "required": False},
        {"name": "electric_field_magnitude", "units": "V/m", "kind": "field", "required": False},
    )
    _solver_keys = ("nx", "ny", "x_max", "y_max", "bc_left_V", "bc_right_V", "mode")

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        x_max: float = 1.0,
        y_max: float = 1.0,
        bc_left_V: float = 1000.0,
        bc_right_V: float = 0.0,
        mode: str = "electrostatic",
    ):
        super().__init__()
        from triality.field_aware_routing.solver import (
            EMFieldSolver,
            FieldSolverConfig,
            DomainConfig,
            BoundaryCondition,
            BCType,
        )

        config = FieldSolverConfig(
            domain=DomainConfig(x_min=0.0, x_max=x_max, y_min=0.0, y_max=y_max, nx=nx, ny=ny),
            bc_left=BoundaryCondition(bc_type=BCType.DIRICHLET, value=bc_left_V),
            bc_right=BoundaryCondition(bc_type=BCType.DIRICHLET, value=bc_right_V),
            mode=mode,
        )
        self._solver = EMFieldSolver(config)

    @classmethod
    def from_demo_case(cls) -> "FieldAwareRoutingRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "FieldAwareRoutingRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, "solver"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        return _mark_runtime_construction(cls(**solver_cfg), mode="config", config=config)

    def _solve_native(self):
        return self._solver.solve()

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# Neutronics Runtime Adapter
# ---------------------------------------------------------------------------

class NeutronicsRuntimeSolver(BaseRuntimeSolver):
    module_name = "neutronics"
    domain = "nuclear"
    fidelity_level = "L2"
    coupling_ready = "M3"
    supports_transient = True
    supports_steady = True
    validation_status = "demo_smoke_tested"
    required_inputs = ()
    output_fields = (
        {"name": "neutron_flux", "units": "n/(m^2*s)", "kind": "field", "required": False},
        {"name": "power_density", "units": "W/m^3", "kind": "field", "required": False},
    )
    _solver_keys = (
        "core_length", "n_spatial", "fuel_material", "reflector_material",
        "reflector_thickness", "fuel_type", "generation_time", "initial_power",
    )
    _solve_keys = ("t_final", "dt", "reactivity_step_pcm")

    def __init__(
        self,
        core_length: float = 200.0,
        n_spatial: int = 50,
        fuel_material: str = "FUEL_UO2_3PCT",
        reflector_material: str = "REFLECTOR_H2O",
        reflector_thickness: float = 30.0,
        fuel_type: str = "U235",
        generation_time: float = 2e-5,
        initial_power: float = 3e9,
        *,
        solve_config: Optional[Mapping[str, Any]] = None,
    ):
        super().__init__()
        from triality.neutronics.solver import NeutronicsSolver
        from triality.neutronics.diffusion_solver import MaterialType

        mat_map = {name: member for name, member in MaterialType.__members__.items()}
        fuel_mat = mat_map.get(fuel_material, MaterialType.FUEL_UO2_3PCT)
        refl_mat = mat_map.get(reflector_material, MaterialType.REFLECTOR_H2O)

        self._solver = NeutronicsSolver(
            core_length=core_length,
            n_spatial=n_spatial,
            fuel_material=fuel_mat,
            reflector_material=refl_mat,
            reflector_thickness=reflector_thickness,
            fuel_type=fuel_type,
            generation_time=generation_time,
            initial_power=initial_power,
        )
        sc = dict(solve_config or {})
        self._t_final = sc.get("t_final", 0.0)
        self._dt = sc.get("dt", 0.01)
        self._rho_step_pcm = sc.get("reactivity_step_pcm", 0.0)

    @classmethod
    def from_demo_case(cls) -> "NeutronicsRuntimeSolver":
        return _mark_runtime_construction(cls(), mode="demo")

    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "NeutronicsRuntimeSolver":
        config = _ensure_mapping(config, context=f"{cls.__name__}.from_config()")
        _reject_unknown_keys(config, allowed=(*cls._solver_keys, *cls._solve_keys, "solver", "solve"), context=f"{cls.__name__}.from_config()")
        solver_cfg = _extract_config_section(config, section_name="solver", section_keys=cls._solver_keys)
        solve_cfg = _extract_config_section(config, section_name="solve", section_keys=cls._solve_keys)
        return _mark_runtime_construction(cls(**solver_cfg, solve_config=solve_cfg), mode="config", config=config)

    def _solve_native(self):
        if self._t_final > 0:
            rho_pcm = self._rho_step_pcm

            def rho_func(t: float) -> float:
                return rho_pcm * 1e-5 if t > 0.5 else 0.0

            return self._solver.solve_transient(
                t_final=self._t_final, dt=self._dt, reactivity_func=rho_func,
            )
        return self._solver.solve()

    def _build_state(self, native_result: Any) -> PhysicsState:
        return self._solver.export_state(native_result)


# ---------------------------------------------------------------------------
# Open Source runtime registry
# ---------------------------------------------------------------------------

RUNTIME_REGISTRY: Dict[str, Type[BaseRuntimeSolver]] = {
    "navier_stokes": NavierStokesRuntimeSolver,
    "drift_diffusion": DriftDiffusionRuntimeSolver,
    "sensing": SensingRuntimeSolver,
    "electrostatics": ElectrostaticsRuntimeSolver,
    "aero_loads": AeroLoadsRuntimeSolver,
    "uav_aerodynamics": UAVAerodynamicsRuntimeSolver,
    "spacecraft_thermal": SpacecraftThermalRuntimeSolver,
    "automotive_thermal": AutomotiveThermalRuntimeSolver,
    "battery_thermal": BatteryThermalRuntimeSolver,
    "structural_analysis": StructuralAnalysisRuntimeSolver,
    "structural_dynamics": StructuralDynamicsRuntimeSolver,
    "flight_mechanics": FlightMechanicsRuntimeSolver,
    "coupled_physics": CoupledPhysicsRuntimeSolver,
    "geospatial": GeospatialRuntimeSolver,
    "field_aware_routing": FieldAwareRoutingRuntimeSolver,
    "neutronics": NeutronicsRuntimeSolver,
}


def load_module(name: str) -> RuntimeModuleHandle:
    try:
        solver_cls = RUNTIME_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(RUNTIME_REGISTRY))
        raise KeyError(f"No runtime SDK adapter registered for '{name}'. Available: {available}") from exc
    return RuntimeModuleHandle(name=name, solver_cls=solver_cls)


def available_runtime_modules() -> list[str]:
    return sorted(RUNTIME_REGISTRY)
