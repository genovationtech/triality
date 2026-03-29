import pytest

from triality import RuntimeContractError, available_runtime_modules, load_module


CONFIG_CASES = {
    "navier_stokes": {
        "solver": {"nx": 10, "ny": 10, "U_lid": 0.15},
        "solve": {"t_end": 0.004, "dt": 0.002, "max_steps": 5, "pressure_iters": 20, "pressure_tol": 1e-4},
    },
    # thermal_hydraulics and conjugate_heat_transfer are Pro warehouse modules
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
    "electrostatics": {
        "solver": {"resolution": 30, "mode": "electrostatic"},
        "solve": {"boundary_value": 500.0},
    },
    "aero_loads": {
        "solver": {"body_length_m": 2.0, "n_panels": 50},
        "solve": {"velocity": 1500.0, "density": 0.2, "temperature": 280.0, "pressure": 2e4, "alpha_deg": 3.0},
    },
    "uav_aerodynamics": {
        "solver": {"span": 8.0, "root_chord": 0.8, "alpha_deg": 6.0, "V_inf": 20.0},
    },
    "spacecraft_thermal": {
        "solver": {"n_nodes": 3, "internal_power_W": 30.0},
        "solve": {"t_end": 1800.0, "dt": 20.0},
    },
    "automotive_thermal": {
        "solver": {"n_components": 2, "T_ambient": 300.0, "current_A": 150.0},
        "solve": {"t_end": 30.0, "dt": 0.2},
    },
    "battery_thermal": {
        "solver": {"n_cells": 48, "cell_chemistry": "NMC"},
        "solve": {"discharge_current_A": 80.0, "duration_s": 300.0, "dt": 1.0},
    },
    "structural_analysis": {
        "solver": {"material_name": "AL7075-T6", "length": 0.5, "n_elements": 10},
        "solve": {"tip_force_N": -3000.0},
    },
    "structural_dynamics": {
        "solver": {"n_dof": 2, "damping_ratio": 0.03, "force_amplitude": 50.0, "force_frequency": 5.0},
        "solve": {"t_end": 1.0, "dt": 0.002},
    },
    "flight_mechanics": {
        "solver": {"mass": 300.0, "Ixx": 80.0, "Iyy": 90.0, "Izz": 60.0},
        "solve": {"t_final": 30.0, "omega_x": 0.02},
    },
    "geospatial": {
        "solver": {"max_facilities": 2, "time_limit_hours": 12.0, "target_coverage": 0.90},
    },
    "field_aware_routing": {
        "solver": {"nx": 32, "ny": 32, "bc_left_V": 500.0},
    },
    "coupled_physics": {
        "solver": {"n_points": 30, "length_cm": 150.0, "feedback_mode": "full"},
        "solve": {"t_end": 5.0, "dt": 0.05, "reactivity_insertion_pcm": 50.0},
    },
    "neutronics": {
        "solver": {"core_length": 150.0, "n_spatial": 30, "fuel_material": "FUEL_UO2_3PCT"},
    },
}


@pytest.mark.parametrize("module_name", available_runtime_modules())
def test_runtime_from_config_executes(module_name):
    handle = load_module(module_name)
    solver = handle.from_config(CONFIG_CASES[module_name])

    result = solver.solve()
    state = solver.to_state()
    description = solver.describe()

    assert result.success is True
    assert result.generated_state is state
    assert description["supports_demo_case"] is True
    assert description["construction_mode"] == "config"
    assert description["module_name"] == module_name


@pytest.mark.parametrize("module_name", available_runtime_modules())
def test_runtime_from_config_accepts_flat_legacy_keys(module_name):
    handle = load_module(module_name)
    config = CONFIG_CASES[module_name]
    flat_config = {
        **config.get("solver", {}),
        **config.get("solve", {}),
        **config.get("doping", {}),
    }

    solver = handle.from_config(flat_config)

    assert solver.describe()["construction_mode"] == "config"


def test_runtime_from_config_rejects_unknown_keys():
    with pytest.raises(RuntimeContractError, match="unknown keys"):
        load_module("navier_stokes").from_config({"solver": {"nx": 8}, "unknown": 1})


def test_runtime_from_config_rejects_duplicate_flat_and_nested_keys():
    with pytest.raises(RuntimeContractError, match="duplicates keys"):
        load_module("navier_stokes").from_config({"nx": 8, "solver": {"nx": 10}})


def test_drift_diffusion_from_config_rejects_unsupported_doping_type():
    with pytest.raises(RuntimeContractError, match="pn_junction"):
        load_module("drift_diffusion").from_config({"doping": {"type": "custom_profile"}})
