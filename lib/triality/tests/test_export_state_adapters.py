import numpy as np

from triality.electrostatics.solver import ElectrostaticsResult, ElectrostaticsSolver
from triality.structural_analysis.fem_solver import ElementResult, FEMResult
from triality.structural_analysis.solver import StructuralAnalysisResult, StructuralSolver
from triality.structural_analysis.static_solver import MATERIALS
from triality.structural_dynamics.modal_analysis import StructuralModel
from triality.structural_dynamics.solver import StructuralDynamicsResult, StructuralDynamicsSolver
from triality.plasma_fluid.solver import PlasmaFluidResult, PlasmaFluidSolver
from triality.neutronics.diffusion_solver import MaterialType, NeutronicsResult
from triality.neutronics.solver import NeutronicsSolver, NeutronicsSolverResult
from triality.sensing.solver import SensorPerformanceConfig, SensorPerformanceResult, SensorPerformanceSolver


def test_electrostatics_export_state_uses_electrical_fields():
    solver = ElectrostaticsSolver()
    result = ElectrostaticsResult(
        potential=np.array([[1.0, 0.0], [0.5, -0.5]]),
        grid_x=np.array([0.0, 1.0]),
        grid_y=np.array([0.0, 1.0]),
        E_x=np.ones((2, 2)),
        E_y=2.0 * np.ones((2, 2)),
        E_magnitude=np.sqrt(5.0) * np.ones((2, 2)),
        max_field_location=(1.0, 1.0, np.sqrt(5.0)),
        field_stats={"max": np.sqrt(5.0)},
        high_field_zones=[(1.0, 1.0, np.sqrt(5.0))],
        current_density_x=3.0 * np.ones((2, 2)),
        current_density_y=4.0 * np.ones((2, 2)),
        power_density=5.0 * np.ones((2, 2)),
        mode="conduction",
    )

    state = solver.export_state(result)

    assert state.get("electric_potential").unit == "V"
    assert state.get("electric_field_x").unit == "V/m"
    assert state.get("electric_field_y").unit == "V/m"
    assert state.get("current_density_x").unit == "A/m^2"
    assert state.get("heat_source").unit == "W/m^3"
    assert state.metadata["mode"] == "conduction"


def test_structural_analysis_export_state_uses_translational_displacement_and_element_stress():
    solver = StructuralSolver(material=MATERIALS["AL7075-T6"], length=2.0, n_elements=2)
    fem_result = FEMResult(
        load_case_name="lc",
        displacements=np.array([0.0, 0.1, 0.01, 0.2, 0.02, 0.3]),
        reactions=np.array([1.0]),
        element_results=[
            ElementResult(0, 1.0, 2.0, 10.0, np.array([0.0, 0.0]), np.array([0.0, 0.0])),
            ElementResult(1, 1.5, 2.5, 20.0, np.array([0.0, 0.0]), np.array([0.0, 0.0])),
        ],
        max_von_mises=20.0,
        margin_of_safety=1.2,
        converged=True,
    )
    result = StructuralAnalysisResult(fem_results=[fem_result], max_von_mises=20.0, overall_ms=1.2)

    state = solver.export_state(result)

    np.testing.assert_allclose(state["displacement"], np.array([0.0, 0.01, 0.02]))
    np.testing.assert_allclose(state["stress_von_mises"], np.array([10.0, 20.0]))


def test_structural_dynamics_export_state_uses_acceleration_field():
    model = StructuralModel(mass_matrix=np.eye(2), stiffness_matrix=np.eye(2))
    solver = StructuralDynamicsSolver(model=model, force_func=lambda t: np.zeros(2))
    result = StructuralDynamicsResult(
        time=np.array([0.0, 0.1]),
        displacement=np.array([[0.0, 0.0], [1.0, 2.0]]),
        velocity=np.array([[0.0, 0.0], [3.0, 4.0]]),
        acceleration=np.array([[0.0, 0.0], [5.0, 6.0]]),
        peak_displacement=np.array([1.0, 2.0]),
        peak_acceleration=np.array([5.0, 6.0]),
        rms_acceleration=np.array([2.0, 3.0]),
    )

    state = solver.export_state(result)

    assert state.get("acceleration").unit == "m/s^2"
    np.testing.assert_allclose(state["acceleration"], np.array([5.0, 6.0]))
    assert state.time == 0.1


def test_plasma_fluid_export_state_uses_number_density_and_electrical_fields():
    solver = PlasmaFluidSolver(nx=3)
    result = PlasmaFluidResult(
        time=np.array([1e-6]),
        x=np.array([0.0, 0.5, 1.0]),
        density=np.array([[1.0, 2.0, 3.0]]),
        ion_velocity=np.array([[4.0, 5.0, 6.0]]),
        electron_temperature_eV=np.array([[7.0, 8.0, 9.0]]),
        electric_field=np.array([[10.0, 11.0, 12.0]]),
        potential=np.array([[13.0, 14.0, 15.0]]),
        neutral_density=np.array([[16.0, 17.0, 18.0]]),
        thrust=np.array([19.0]),
        specific_impulse=np.array([20.0]),
        efficiency=np.array([0.3]),
        discharge_current=np.array([21.0]),
    )

    state = solver.export_state(result)

    assert state.get("number_density").unit == "1/m^3"
    assert state.get("electric_field").unit == "V/m"
    assert state.get("electric_potential").unit == "V"
    np.testing.assert_allclose(state["number_density"], np.array([1.0, 2.0, 3.0]))


def test_neutronics_export_state_converts_flux_and_power_density_to_si():
    solver = NeutronicsSolver(n_spatial=3, fuel_material=MaterialType.FUEL_UO2_3PCT)
    diff = NeutronicsResult(
        x=np.array([0.0, 10.0, 20.0]),
        phi_fast=np.array([1.0, 2.0, 3.0]),
        phi_thermal=np.array([0.5, 1.5, 2.5]),
        k_eff=1.01,
        converged=True,
        iterations=4,
        power_density=np.array([2.0, 4.0, 6.0]),
        materials=np.array([0, 1, 0]),
    )
    result = NeutronicsSolverResult(
        k_eff=1.01,
        converged=True,
        diffusion_result=diff,
        time=np.array([0.0, 1.0]),
        power=np.array([100.0, 110.0]),
        reactivity=np.array([0.0, 1e-5]),
        period=np.array([np.inf, 10.0]),
        precursor_source=np.array([0.0, 1.0]),
        peak_power=110.0,
        is_prompt_critical=False,
        power_shape=np.array([0.5, 1.0, 1.5]),
        peaking_factor=1.5,
    )

    state = solver.export_state(result)

    np.testing.assert_allclose(state["neutron_flux"], np.array([1.5e4, 3.5e4, 5.5e4]))
    np.testing.assert_allclose(state["power_density"], np.array([2.0e6, 4.0e6, 6.0e6]))
    assert "power_history_W" in state.metadata


def test_sensing_export_state_uses_detection_probability():
    solver = SensorPerformanceSolver(SensorPerformanceConfig())
    result = SensorPerformanceResult(
        grid_x_km=np.array([0.0, 1.0]),
        grid_y_km=np.array([0.0, 1.0]),
        snr_maps={"radar": np.array([[1.0, 2.0], [3.0, 4.0]])},
        pd_maps={"radar": np.array([[0.2, 0.4], [0.6, 0.8]])},
        combined_pd=np.array([[0.2, 0.4], [0.6, 0.8]]),
        max_detection_range_km={"radar": 12.0},
        resolution={"radar": {"range_resolution_m": 5.0}},
        coverage_fraction=0.5,
        sensor_count=1,
        config=SensorPerformanceConfig(),
    )

    state = solver.export_state(result)

    assert state.get("detection_probability").unit == "1"
    np.testing.assert_allclose(state["detection_probability"], result.combined_pd)
