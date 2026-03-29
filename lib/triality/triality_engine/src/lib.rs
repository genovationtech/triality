//! Triality Engine: High-performance numerical backbone.
//!
//! Rust owns the engine-room work:
//!   - Sparse matrix assembly and SpMV (memory layout, cache efficiency)
//!   - Linear solvers: CG, GMRES, BiCGSTAB, Direct LU (compute-heavy inner loops)
//!   - Preconditioners: Jacobi, ILU(0), SSOR (reuse across all physics modules)
//!   - FDM stencil assembly (eliminates Python loop bottleneck)
//!   - Time integrators: Forward Euler, Backward Euler, Crank-Nicolson, RK4, BDF2
//!   - Grid types with fast indexing
//!
//! Python keeps the brain:
//!   - PDE classification and solver selection
//!   - Domain-specific physics layers (electrostatics, drift-diffusion, etc.)
//!   - Assumption tracking and validation
//!   - Visualization and user-facing API

pub mod fdm;
pub mod vtk_export;
pub mod mesh;
pub mod gmsh_io;
pub mod fem_bc;
pub mod fem_assembly;
pub mod elements;
pub mod grid;
pub mod precond;
pub mod solvers;
pub mod sparse;
pub mod timestepper;

mod python_bindings;

use pyo3::prelude::*;

/// Triality Engine Python module.
#[pymodule]
fn triality_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python_bindings::register(m)?;
    Ok(())
}
