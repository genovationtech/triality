from triality import available_runtime_modules, load_module, RuntimeExecutionResult


CONFIG_CASES = {
    "navier_stokes": {
        "solver": {"nx": 10, "ny": 10, "U_lid": 0.15},
        "solve": {"t_end": 0.004, "dt": 0.002, "max_steps": 5, "pressure_iters": 20, "pressure_tol": 1e-4},
    },
    "thermal_hydraulics": {
        "solver": {"n_axial": 8, "n_fuel_radial": 6, "mass_flux": 3000.0},
        "solve": {"peak_linear_heat_rate": 140.0, "axial_shape": "uniform"},
    },
    "conjugate_heat_transfer": {
        "solver": {"nx": 8, "ny_solid": 4, "ny_fluid": 8, "Q_vol": 1.5e5, "T_init": 345.0, "T_fluid_top": 340.0},
        "solve": {"t_end": 0.002, "dt": 5e-4, "max_coupling_iter": 4, "save_interval": 1},
    },
    "drift_diffusion": {
        "solver": {"length": 1.0e-4, "n_points": 40, "temperature": 305.0},
        "doping": {"type": "pn_junction", "N_d_level": 8.0e16, "N_a_level": 4.0e16},
        "solve": {"applied_voltage": 0.0, "max_iterations": 20, "tolerance": 1e-4, "under_relaxation": 0.35},
    },
    "sensing": {
        "grid_x_km": (-5.0, 5.0),
        "grid_y_km": (-5.0, 5.0),
        "grid_nx": 24,
        "grid_ny": 24,
        "sensor_name": "demo_radar",
        "sensor_location": (0.0, 0.0),
        "radar": {"frequency_ghz": 10.0, "power_w": 500.0, "bandwidth_mhz": 5.0, "aperture_diameter_m": 0.8},
        "target": {"rcs_m2": 1.5, "target_strength_db": 8.0},
        "solve": {"pfa": 1e-6, "weather": "clear", "water_temp_c": 15.0},
    },
}


def _check_solver(module_name, solver, expected_mode: str) -> None:
    result = solver.solve()
    state = solver.to_state()
    meta = solver.describe()
    assert isinstance(result, RuntimeExecutionResult)
    assert result.success is True
    assert result.status == "success"
    assert isinstance(result.warnings, list)
    assert isinstance(result.residuals, dict)
    assert isinstance(result.convergence, dict)
    assert result.elapsed_time_s >= 0.0
    assert result.generated_state is state
    assert result.result_payload is not None
    assert isinstance(meta, dict)
    assert meta["module_name"] == module_name
    assert meta["supports_demo_case"] is True
    assert meta["construction_mode"] == expected_mode
    for key in [
        "module_name",
        "domain",
        "fidelity_level",
        "coupling_ready",
        "supports_transient",
        "supports_steady",
        "supports_demo_case",
        "construction_mode",
        "required_inputs",
        "output_fields",
        "validation_status",
        "contract_version",
    ]:
        assert key in meta
    for field in meta["output_fields"]:
        assert {"name", "units", "kind", "required"} <= set(field)
    print(
        module_name,
        type(result.result_payload).__name__,
        state.solver_name,
        meta["coupling_ready"],
        meta["construction_mode"],
        sorted(field["name"] for field in meta["output_fields"]),
    )


def main() -> None:
    for module_name in available_runtime_modules():
        handle = load_module(module_name)
        _check_solver(module_name, handle.from_demo_case(), expected_mode="demo")
        _check_solver(module_name, handle.from_config(CONFIG_CASES[module_name]), expected_mode="config")


if __name__ == "__main__":
    main()
