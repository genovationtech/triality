"""
Incompressible Navier-Stokes Solver for Variable-Density Two-Phase Flows

Production-grade 2D incompressible Navier-Stokes solver using the projection
method (Chorin's splitting) on a staggered MAC grid.  Designed for nucleate
boiling simulations with variable-density liquid-vapor flows.

Physics Basis
=============
Momentum conservation (variable-density incompressible):

    d(rho*u)/dt + div(rho*u (x) u) = -grad(p)
                                     + div(mu*(grad(u) + grad(u)^T))
                                     + rho*g
                                     + F_surface

Continuity with phase-change mass source:

    div(u) = S_mass

where:
    rho       = density [kg/m^3], variable across the liquid-vapor interface
    mu        = dynamic viscosity [Pa.s], variable across the interface
    u         = velocity vector (u, v) [m/s]
    p         = pressure [Pa]
    g         = gravitational acceleration [m/s^2]
    F_surface = surface tension body force (e.g. CSF model) [N/m^3]
    S_mass    = volume source from phase change [1/s]
                (= mdot * (1/rho_v - 1/rho_l) at the interface)

Numerical Method
================
Projection method (Chorin's splitting), three sub-steps per time step:

    1. **Predictor**: Compute intermediate velocity u* from advection
       (explicit Forward Euler), diffusion (Crank-Nicolson, theta=0.5),
       gravity, and body forces.

    2. **Pressure Poisson**: Solve
           div( (1/rho) * grad(p) ) = (div(u*) - S_mass) / dt
       with scipy sparse solvers (BiCGSTAB with direct fallback).

    3. **Corrector**: Project onto the (modified) divergence-free space:
           u^{n+1} = u* - (dt/rho) * grad(p)

Spatial Discretization
======================
Staggered grid (Marker-and-Cell / MAC):
    - u-velocity stored on vertical cell faces:    shape (nx+1, ny)
    - v-velocity stored on horizontal cell faces:  shape (nx, ny+1)
    - Pressure, density, viscosity at cell centers: shape (nx, ny)

This naturally avoids the checkerboard pressure mode and exactly
enforces the discrete divergence constraint.

Coordinate Systems
==================
- **Cartesian 2D**: standard (x, y)
- **Axisymmetric 2D**: (r, z) with extra 1/r metric terms in divergence
  and Laplacian operators.  Suitable for bubble dynamics simulations.

Boundary Conditions
===================
- **Wall** (no-slip): u = 0 at the wall; dp/dn = 0
- **Inlet**: prescribed velocity; dp/dn = 0
- **Outlet**: zero-gradient velocity; prescribed or zero-gradient pressure
- **Symmetry** (axis): u_normal = 0, d(u_tangential)/dn = 0; dp/dn = 0

Features
========
- Variable density and viscosity (two-phase: liquid + vapor)
- Phase-change mass source term (divergence constraint)
- Surface tension body force support (for CSF coupling)
- 2D Cartesian and 2D axisymmetric coordinates
- Semi-implicit diffusion (Crank-Nicolson) for stability
- Harmonic mean density in Poisson operator (sharp interface robustness)
- Callback interface for coupling with VOF / level-set / energy solvers

Applications
============
- Nucleate boiling bubble dynamics
- Film boiling
- Two-phase Rayleigh-Benard convection
- Lid-driven cavity flow (single-phase benchmark)
- Buoyancy-driven natural convection
- Droplet dynamics

Typical Usage
=============
>>> from triality.navier_stokes import (
...     NavierStokesSolver, StaggeredGrid, NavierStokesResult,
...     BoundaryCondition, BCType, CoordinateSystem,
... )
>>>
>>> # Lid-driven cavity benchmark
>>> solver = NavierStokesSolver()
>>> solver.set_domain(x_range=(0, 1), y_range=(0, 1), nx=32, ny=32)
>>> solver.set_fluid_properties(rho=1.0, mu=0.01)  # Re = 100
>>> solver.set_boundary_conditions(
...     left=BoundaryCondition(BCType.WALL),
...     right=BoundaryCondition(BCType.WALL),
...     bottom=BoundaryCondition(BCType.WALL),
...     top=BoundaryCondition(BCType.INLET, velocity=(1.0, 0.0)),
... )
>>> result = solver.solve(t_end=10.0, dt=0.001, store_every=100)
>>> print(f"Max velocity: {result.max_velocity():.4f}")

>>> # Axisymmetric boiling setup
>>> solver = NavierStokesSolver(coord_system=CoordinateSystem.AXISYMMETRIC)
>>> solver.set_domain(x_range=(0, 0.005), y_range=(0, 0.01), nx=50, ny=100)
>>> solver.set_fluid_properties(rho=958.0, mu=2.82e-4)  # water at 100C
>>> solver.set_gravity(gx=0.0, gy=-9.81)
>>> solver.set_boundary_conditions(
...     left=BoundaryCondition(BCType.SYMMETRY),
...     right=BoundaryCondition(BCType.WALL),
...     bottom=BoundaryCondition(BCType.WALL),
...     top=BoundaryCondition(BCType.OUTLET, pressure=0.0),
... )
>>> # In a coupled simulation loop:
>>> # solver.set_fluid_properties(rho_field, mu_field)  # from VOF
>>> # solver.add_body_force(Fsx, Fsy)                   # from surface tension
>>> # solver.add_mass_source(S_phase_change)             # from energy equation
>>> # result = solver.step(dt)

Accuracy Expectations
=====================
- Spatial: 2nd order (centered differences on staggered grid)
- Temporal: 1st order (Forward Euler advection), 2nd order diffusion (CN)
- Pressure: solved to iterative tolerance (default 1e-8)
- Divergence error: O(solver tolerance)
- Suitable for resolved DNS of moderate-Re flows and boiling

Limitations
===========
- Uniform grid spacing only (dx, dy constant)
- No adaptive mesh refinement
- No turbulence modelling (DNS only)
- Explicit advection limits CFL < 1 for stability
- 2D only (no 3D)

Dependencies
============
- numpy (array operations)
- scipy (sparse linear algebra for pressure Poisson equation)

Version: 1.0.0
"""

from .projection_solver import (
    CoordinateSystem,
    BCType,
    BoundaryCondition,
    StaggeredGrid,
    NavierStokesResult,
    NavierStokesSolver,
)

from .solver import (
    FlowSolver,
    FlowSolverResult,
)

__version__ = "1.0.0"

__all__ = [
    # Coordinate system
    'CoordinateSystem',
    # Boundary conditions
    'BCType',
    'BoundaryCondition',
    # Grid
    'StaggeredGrid',
    # Result container
    'NavierStokesResult',
    # Solver
    'NavierStokesSolver',
    # High-level solver
    'FlowSolver',
    'FlowSolverResult',
]
