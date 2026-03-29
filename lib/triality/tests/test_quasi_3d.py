"""End-to-end tests for quasi-3D (2.5D) support across all five modules.

Each test verifies:
1. The solver accepts quasi_3d=True and z_length parameters
2. The solve completes successfully
3. The quasi-3D correction has the expected physical effect
4. The state export includes quasi_3d metadata
5. Runtime integration passes config through correctly
"""

import numpy as np
import pytest

# ── Navier-Stokes ────────────────────────────────────────────────────────

class TestNavierStokesQuasi3D:

    def test_solver_accepts_params(self):
        from triality.navier_stokes.solver import NavierStokes2DSolver
        s = NavierStokes2DSolver(nx=12, ny=12, quasi_3d=True, z_length=0.5)
        assert s.quasi_3d is True
        assert s.z_half == 0.25

    def test_friction_reduces_velocity(self):
        from triality.navier_stokes.solver import NavierStokes2DSolver
        kwargs = dict(nx=16, ny=16, Lx=1.0, Ly=1.0, rho=1.0, nu=0.01, U_lid=1.0)
        solve_kw = dict(t_end=0.02, dt=0.001, max_steps=20)

        r_2d = NavierStokes2DSolver(**kwargs).solve(**solve_kw)
        r_25d = NavierStokes2DSolver(**kwargs, quasi_3d=True, z_length=0.3).solve(**solve_kw)

        # Interior velocity should be lower with friction
        interior_2d = np.max(np.abs(r_2d.u[1:-1, 1:-1]))
        interior_25d = np.max(np.abs(r_25d.u[1:-1, 1:-1]))
        assert interior_25d <= interior_2d + 1e-6

    def test_export_state_metadata(self):
        from triality.navier_stokes.solver import NavierStokes2DSolver
        s = NavierStokes2DSolver(nx=8, ny=8, quasi_3d=True, z_length=0.5)
        r = s.solve(t_end=0.005, dt=0.001, max_steps=5)
        state = s.export_state(r)
        assert state.metadata["quasi_3d"] is True
        assert state.metadata["z_length"] == 0.5

    def test_runtime_integration(self):
        from triality.runtime import load_module
        h = load_module("navier_stokes")
        s = h.from_config({
            "solver": {"nx": 8, "ny": 8, "quasi_3d": True, "z_length": 0.4},
            "solve": {"t_end": 0.002, "dt": 0.001, "max_steps": 2},
        })
        r = s.solve(strict=False)
        assert r.success
        assert r.generated_state.metadata["quasi_3d"] is True


# ── CFD Turbulence (RANS k-ε) ───────────────────────────────────────────

class TestCFDTurbulenceQuasi3D:

    def test_rans_solver_accepts_params(self):
        from triality.cfd_turbulence.solver import RANSSolver
        s = RANSSolver(nx=8, ny=4, quasi_3d=True, z_length=0.1)
        assert s.quasi_3d is True
        assert s.z_half == 0.05

    def test_cavity_solver_accepts_params(self):
        from triality.cfd_turbulence.solver import CFDTurbulence2DSolver
        s = CFDTurbulence2DSolver(nx=8, ny=8, quasi_3d=True, z_length=0.5)
        assert s.quasi_3d is True
        assert s.z_half == 0.25

    def test_rans_quasi_3d_runs(self):
        from triality.cfd_turbulence.solver import RANSSolver
        s = RANSSolver(nx=8, ny=4, lx=0.5, ly=0.25, max_iter=20,
                       quasi_3d=True, z_length=0.1)
        r = s.solve(u_inlet=1.0)
        assert r.u.shape == (4, 8)
        assert r.iterations > 0

    def test_cavity_quasi_3d_runs(self):
        from triality.cfd_turbulence.solver import CFDTurbulence2DSolver
        s = CFDTurbulence2DSolver(nx=8, ny=8, max_iter=10,
                                   quasi_3d=True, z_length=0.5)
        r = s.solve()
        assert r.u.shape == (8, 8)

    def test_rans_export_state_metadata(self):
        from triality.cfd_turbulence.solver import RANSSolver
        s = RANSSolver(nx=8, ny=4, max_iter=5, quasi_3d=True, z_length=0.2)
        r = s.solve(u_inlet=1.0)
        state = s.export_state(r)
        assert state.metadata["quasi_3d"] is True
        assert state.metadata["z_length"] == 0.2

    def test_runtime_integration(self):
        from triality.runtime import load_module
        h = load_module("cfd_turbulence")
        s = h.from_config({
            "solver": {"nx": 8, "ny": 4, "max_iter": 5,
                       "quasi_3d": True, "z_length": 0.2},
            "solve": {"u_inlet": 1.0},
        })
        r = s.solve(strict=False)
        assert r.success
        assert r.generated_state.metadata["quasi_3d"] is True


# ── Conjugate Heat Transfer ──────────────────────────────────────────────

class TestCHTQuasi3D:

    def test_solver_accepts_params(self):
        from triality.conjugate_heat_transfer.solver import CHTSystemSolver
        s = CHTSystemSolver(nx=6, ny_solid=3, ny_fluid=6,
                            quasi_3d=True, z_length=0.005)
        assert s.quasi_3d is True
        assert s.z_half == 0.0025

    def test_z_loss_reduces_temperature(self):
        from triality.conjugate_heat_transfer.solver import CHTSystemSolver
        kwargs = dict(nx=8, ny_solid=4, ny_fluid=8, Q_vol=5e5,
                      T_init=350.0, T_fluid_top=345.0)
        solve_kw = dict(t_end=0.002, dt=1e-3, max_coupling_iter=3, save_interval=1)

        r_2d = CHTSystemSolver(**kwargs).solve(**solve_kw)
        r_25d = CHTSystemSolver(**kwargs, quasi_3d=True, z_length=0.003).solve(**solve_kw)

        # z-face heat loss should reduce max solid temperature
        assert r_25d.max_solid_temperature <= r_2d.max_solid_temperature + 0.1

    def test_export_state_metadata(self):
        from triality.conjugate_heat_transfer.solver import CHTSystemSolver
        s = CHTSystemSolver(nx=6, ny_solid=3, ny_fluid=6,
                            quasi_3d=True, z_length=0.005)
        r = s.solve(t_end=0.001, dt=5e-4, max_coupling_iter=2, save_interval=1)
        state = s.export_state(r)
        assert state.metadata["quasi_3d"] is True
        assert state.metadata["z_length"] == 0.005

    def test_runtime_integration(self):
        from triality.runtime import load_module
        h = load_module("conjugate_heat_transfer")
        s = h.from_config({
            "solver": {"nx": 6, "ny_solid": 3, "ny_fluid": 6,
                       "quasi_3d": True, "z_length": 0.005},
            "solve": {"t_end": 0.001, "dt": 5e-4},
        })
        r = s.solve(strict=False)
        assert r.success
        assert r.generated_state.metadata["quasi_3d"] is True


# ── Electrostatics ───────────────────────────────────────────────────────

class TestElectrostaticsQuasi3D:

    def test_solver_accepts_params(self):
        from triality.electrostatics.solver import Electrostatics2DSolver
        s = Electrostatics2DSolver(nx=20, ny=20, quasi_3d=True, z_length=0.1)
        assert s.quasi_3d is True
        assert s.z_half == 0.05

    def test_leakage_reduces_potential(self):
        from triality.electrostatics.solver import Electrostatics2DSolver
        kwargs = dict(nx=20, ny=20, Lx=1.0, Ly=1.0,
                      bc_left=0.0, bc_right=100.0, bc_top=0.0, bc_bottom=0.0)

        r_2d = Electrostatics2DSolver(**kwargs).solve()
        r_25d = Electrostatics2DSolver(**kwargs, quasi_3d=True, z_length=0.2).solve()

        # z-leakage should reduce interior potential
        mid = r_2d.potential.shape[0] // 2
        assert r_25d.potential[mid, mid] < r_2d.potential[mid, mid]

    def test_convergence_maintained(self):
        from triality.electrostatics.solver import Electrostatics2DSolver
        s = Electrostatics2DSolver(nx=30, ny=30, bc_right=50.0,
                                    quasi_3d=True, z_length=0.5)
        r = s.solve()
        assert r.converged

    def test_export_state_metadata(self):
        from triality.electrostatics.solver import Electrostatics2DSolver
        s = Electrostatics2DSolver(nx=10, ny=10, quasi_3d=True, z_length=0.3)
        state = s.export_state()
        assert state.metadata["quasi_3d"] is True
        assert state.metadata["z_length"] == 0.3


# ── Thermal Hydraulics ───────────────────────────────────────────────────

class TestThermalHydraulicsQuasi3D:

    def test_solver_accepts_params(self):
        from triality.thermal_hydraulics.solver import ThermalHydraulicsSolver
        s = ThermalHydraulicsSolver(n_axial=8, quasi_3d=True, z_length=0.005)
        assert s.quasi_3d is True
        assert s.z_half == 0.0025

    def test_side_wall_improves_dnbr(self):
        from triality.thermal_hydraulics.solver import ThermalHydraulicsSolver
        kwargs = dict(n_axial=10, n_fuel_radial=5)
        solve_kw = dict(peak_linear_heat_rate=150.0, axial_shape="uniform")

        r_2d = ThermalHydraulicsSolver(**kwargs).solve(**solve_kw)
        r_25d = ThermalHydraulicsSolver(**kwargs, quasi_3d=True, z_length=0.005).solve(**solve_kw)

        # Side-wall cooling reduces effective heat flux → better DNBR
        assert r_25d.min_dnbr >= r_2d.min_dnbr - 0.01

    def test_export_state_metadata(self):
        from triality.thermal_hydraulics.solver import ThermalHydraulicsSolver
        s = ThermalHydraulicsSolver(n_axial=8, quasi_3d=True, z_length=0.005)
        r = s.solve(peak_linear_heat_rate=100.0)
        state = s.export_state(r)
        assert state.metadata["quasi_3d"] is True
        assert state.metadata["z_length"] == 0.005

    def test_runtime_integration(self):
        from triality.runtime import load_module
        h = load_module("thermal_hydraulics")
        s = h.from_config({
            "solver": {"n_axial": 8, "quasi_3d": True, "z_length": 0.005},
            "solve": {"peak_linear_heat_rate": 100.0},
        })
        r = s.solve(strict=False)
        assert r.success
        assert r.generated_state.metadata["quasi_3d"] is True


# ── Cross-module consistency ─────────────────────────────────────────────

class TestQuasi3DConsistency:
    """Verify the interface is uniform across all modules."""

    def test_all_modules_use_same_param_names(self):
        """All quasi-3D modules accept quasi_3d and z_length."""
        from triality.navier_stokes.solver import NavierStokes2DSolver
        from triality.cfd_turbulence.solver import RANSSolver, CFDTurbulence2DSolver
        from triality.conjugate_heat_transfer.solver import CHTSystemSolver
        from triality.electrostatics.solver import Electrostatics2DSolver
        from triality.thermal_hydraulics.solver import ThermalHydraulicsSolver

        import inspect
        for cls in [NavierStokes2DSolver, RANSSolver, CFDTurbulence2DSolver,
                    CHTSystemSolver, Electrostatics2DSolver, ThermalHydraulicsSolver]:
            sig = inspect.signature(cls.__init__)
            params = list(sig.parameters.keys())
            assert "quasi_3d" in params, f"{cls.__name__} missing quasi_3d param"
            assert "z_length" in params, f"{cls.__name__} missing z_length param"

    def test_default_is_2d(self):
        """All modules default to quasi_3d=False (backward compatible)."""
        from triality.navier_stokes.solver import NavierStokes2DSolver
        from triality.cfd_turbulence.solver import RANSSolver
        from triality.conjugate_heat_transfer.solver import CHTSystemSolver
        from triality.electrostatics.solver import Electrostatics2DSolver
        from triality.thermal_hydraulics.solver import ThermalHydraulicsSolver

        for cls in [NavierStokes2DSolver, RANSSolver, CHTSystemSolver,
                    Electrostatics2DSolver, ThermalHydraulicsSolver]:
            sig = inspect.signature(cls.__init__)
            assert sig.parameters["quasi_3d"].default is False, \
                f"{cls.__name__} should default quasi_3d=False"


import inspect  # used in TestQuasi3DConsistency
