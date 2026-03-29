//! PyO3 bindings: expose Rust engine to Python.
//!
//! Design: keep the binding layer thin. All real logic lives in the Rust modules.
//! Python sends numpy arrays in, gets numpy arrays back.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::fdm;
use crate::grid;
use crate::solvers;
use crate::sparse::CsrMatrix;
use crate::timestepper;

/// Helper: convert Vec<f64> to a numpy array bound to the given Python context.
fn vec_to_pyarray<'py>(py: Python<'py>, v: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
    Array1::from_vec(v).into_pyarray_bound(py)
}

// ─── Sparse Matrix ──────────────────────────────────────────────────────

/// CSR matrix wrapper for Python.
#[pyclass(name = "CsrMatrix")]
pub struct PyCsrMatrix {
    inner: CsrMatrix,
}

#[pymethods]
impl PyCsrMatrix {
    /// Create from raw CSR components.
    #[new]
    fn new(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<f64>,
    ) -> PyResult<Self> {
        let mat = CsrMatrix::new(nrows, ncols, row_ptr, col_idx, values)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(Self { inner: mat })
    }

    /// Create from SciPy CSR components (convenience).
    #[staticmethod]
    fn from_scipy(
        nrows: usize,
        ncols: usize,
        indptr: PyReadonlyArray1<i64>,
        indices: PyReadonlyArray1<i64>,
        data: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let row_ptr: Vec<usize> = indptr.as_slice()?.iter().map(|&x| x as usize).collect();
        let col_idx: Vec<usize> = indices.as_slice()?.iter().map(|&x| x as usize).collect();
        let values: Vec<f64> = data.as_slice()?.to_vec();
        let mat = CsrMatrix::new(nrows, ncols, row_ptr, col_idx, values)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(Self { inner: mat })
    }

    /// Sparse matrix-vector multiply: y = A * x.
    fn spmv<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let x_slice = x.as_slice()?;
        let y = if self.inner.nrows > 2000 {
            self.inner.spmv_par(x_slice)
        } else {
            self.inner.spmv(x_slice)
        };
        Ok(vec_to_pyarray(py, y))
    }

    /// Extract diagonal.
    fn diagonal<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec_to_pyarray(py, self.inner.diagonal())
    }

    /// Number of non-zeros.
    #[getter]
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    #[getter]
    fn nrows(&self) -> usize {
        self.inner.nrows
    }

    #[getter]
    fn ncols(&self) -> usize {
        self.inner.ncols
    }

    fn __repr__(&self) -> String {
        format!(
            "CsrMatrix({}x{}, nnz={})",
            self.inner.nrows,
            self.inner.ncols,
            self.inner.nnz()
        )
    }
}

// ─── Linear Solvers ─────────────────────────────────────────────────────

/// Solve a linear system Ax = b.
#[pyfunction]
#[pyo3(signature = (matrix, rhs, method="auto", precond="none", tol=1e-8, max_iter=0))]
fn solve_linear<'py>(
    py: Python<'py>,
    matrix: &PyCsrMatrix,
    rhs: PyReadonlyArray1<f64>,
    method: &str,
    precond: &str,
    tol: f64,
    max_iter: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let b = rhs.as_slice()?;
    let max_it = if max_iter == 0 {
        b.len() * 10
    } else {
        max_iter
    };

    let result = solvers::solve_linear(&matrix.inner, b, method, precond, tol, max_it);

    let dict = PyDict::new_bound(py);
    dict.set_item("x", vec_to_pyarray(py, result.x))?;
    dict.set_item("converged", result.converged)?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("residual", result.residual)?;
    Ok(dict)
}

// ─── FDM Assembly ───────────────────────────────────────────────────────

/// Assemble 1D Laplacian stencil matrix.
#[pyfunction]
fn assemble_laplacian_1d<'py>(
    py: Python<'py>,
    a: f64,
    b: f64,
    n: usize,
) -> PyResult<(PyCsrMatrix, Bound<'py, PyArray1<f64>>)> {
    let (mat, grid) = fdm::laplacian_1d(a, b, n);
    Ok((PyCsrMatrix { inner: mat }, vec_to_pyarray(py, grid)))
}

/// Assemble 1D Laplacian with variable coefficient k(x).
#[pyfunction]
fn assemble_laplacian_1d_variable<'py>(
    py: Python<'py>,
    a: f64,
    b: f64,
    n: usize,
    k_values: PyReadonlyArray1<'py, f64>,
) -> PyResult<(PyCsrMatrix, Bound<'py, PyArray1<f64>>)> {
    let k = k_values.as_slice()?;
    let (mat, grid) = fdm::laplacian_1d_variable(a, b, n, k);
    Ok((PyCsrMatrix { inner: mat }, vec_to_pyarray(py, grid)))
}

/// Assemble 2D Laplacian stencil matrix.
#[pyfunction]
fn assemble_laplacian_2d<'py>(
    py: Python<'py>,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    nx: usize,
    ny: usize,
) -> PyResult<(
    PyCsrMatrix,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let (mat, gx, gy) = fdm::laplacian_2d(x0, x1, y0, y1, nx, ny);
    Ok((
        PyCsrMatrix { inner: mat },
        vec_to_pyarray(py, gx),
        vec_to_pyarray(py, gy),
    ))
}

/// Assemble 2D Laplacian with variable coefficient k(x,y).
#[pyfunction]
fn assemble_laplacian_2d_variable<'py>(
    py: Python<'py>,
    x0: f64,
    x1: f64,
    y0: f64,
    y1: f64,
    nx: usize,
    ny: usize,
    k_values: PyReadonlyArray1<'py, f64>,
) -> PyResult<(
    PyCsrMatrix,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let k = k_values.as_slice()?;
    let (mat, gx, gy) = fdm::laplacian_2d_variable(x0, x1, y0, y1, nx, ny, k);
    Ok((
        PyCsrMatrix { inner: mat },
        vec_to_pyarray(py, gx),
        vec_to_pyarray(py, gy),
    ))
}

/// Apply 1D Dirichlet boundary conditions to a RHS vector.
#[pyfunction]
fn apply_bc_1d<'py>(
    py: Python<'py>,
    rhs: PyReadonlyArray1<f64>,
    bc_left: f64,
    bc_right: f64,
) -> Bound<'py, PyArray1<f64>> {
    let mut r = rhs.as_slice().unwrap().to_vec();
    fdm::apply_bc_1d(&mut r, bc_left, bc_right);
    vec_to_pyarray(py, r)
}

/// Apply 2D Dirichlet boundary conditions to a RHS vector.
#[pyfunction]
fn apply_bc_2d<'py>(
    py: Python<'py>,
    rhs: PyReadonlyArray1<f64>,
    nx: usize,
    ny: usize,
    bc_left: f64,
    bc_right: f64,
    bc_bottom: f64,
    bc_top: f64,
) -> Bound<'py, PyArray1<f64>> {
    let mut r = rhs.as_slice().unwrap().to_vec();
    fdm::apply_bc_2d(&mut r, nx, ny, (bc_left, bc_right, bc_bottom, bc_top));
    vec_to_pyarray(py, r)
}

// ─── Time Steppers ──────────────────────────────────────────────────────

/// Forward Euler time integration.
#[pyfunction]
#[pyo3(signature = (matrix, u0, f_rhs, dt, n_steps, save_every=1))]
fn timestep_forward_euler<'py>(
    py: Python<'py>,
    matrix: &PyCsrMatrix,
    u0: PyReadonlyArray1<f64>,
    f_rhs: PyReadonlyArray1<f64>,
    dt: f64,
    n_steps: usize,
    save_every: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let result = timestepper::forward_euler(
        &matrix.inner,
        u0.as_slice()?,
        f_rhs.as_slice()?,
        dt,
        n_steps,
        save_every,
    );
    pack_timestep_result(py, &result)
}

/// Backward Euler time integration (implicit, unconditionally stable).
#[pyfunction]
#[pyo3(signature = (matrix, u0, f_rhs, dt, n_steps, save_every=1, solver="cg", precond="none"))]
fn timestep_backward_euler<'py>(
    py: Python<'py>,
    matrix: &PyCsrMatrix,
    u0: PyReadonlyArray1<f64>,
    f_rhs: PyReadonlyArray1<f64>,
    dt: f64,
    n_steps: usize,
    save_every: usize,
    solver: &str,
    precond: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let result = timestepper::backward_euler(
        &matrix.inner,
        u0.as_slice()?,
        f_rhs.as_slice()?,
        dt,
        n_steps,
        save_every,
        solver,
        precond,
    );
    pack_timestep_result(py, &result)
}

/// Crank-Nicolson time integration (implicit, 2nd-order).
#[pyfunction]
#[pyo3(signature = (matrix, u0, f_rhs, dt, n_steps, save_every=1, solver="cg", precond="none"))]
fn timestep_crank_nicolson<'py>(
    py: Python<'py>,
    matrix: &PyCsrMatrix,
    u0: PyReadonlyArray1<f64>,
    f_rhs: PyReadonlyArray1<f64>,
    dt: f64,
    n_steps: usize,
    save_every: usize,
    solver: &str,
    precond: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let result = timestepper::crank_nicolson(
        &matrix.inner,
        u0.as_slice()?,
        f_rhs.as_slice()?,
        dt,
        n_steps,
        save_every,
        solver,
        precond,
    );
    pack_timestep_result(py, &result)
}

/// RK4 time integration (explicit, 4th-order).
#[pyfunction]
#[pyo3(signature = (matrix, u0, f_rhs, dt, n_steps, save_every=1))]
fn timestep_rk4<'py>(
    py: Python<'py>,
    matrix: &PyCsrMatrix,
    u0: PyReadonlyArray1<f64>,
    f_rhs: PyReadonlyArray1<f64>,
    dt: f64,
    n_steps: usize,
    save_every: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let result = timestepper::rk4(
        &matrix.inner,
        u0.as_slice()?,
        f_rhs.as_slice()?,
        dt,
        n_steps,
        save_every,
    );
    pack_timestep_result(py, &result)
}

/// BDF2 time integration (implicit, 2nd-order, A-stable).
#[pyfunction]
#[pyo3(signature = (matrix, u0, f_rhs, dt, n_steps, save_every=1, solver="cg", precond="none"))]
fn timestep_bdf2<'py>(
    py: Python<'py>,
    matrix: &PyCsrMatrix,
    u0: PyReadonlyArray1<f64>,
    f_rhs: PyReadonlyArray1<f64>,
    dt: f64,
    n_steps: usize,
    save_every: usize,
    solver: &str,
    precond: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let result = timestepper::bdf2(
        &matrix.inner,
        u0.as_slice()?,
        f_rhs.as_slice()?,
        dt,
        n_steps,
        save_every,
        solver,
        precond,
    );
    pack_timestep_result(py, &result)
}

fn pack_timestep_result<'py>(
    py: Python<'py>,
    result: &timestepper::TimeStepResult,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    let snapshots: Vec<Bound<'py, PyArray1<f64>>> = result
        .snapshots
        .iter()
        .map(|s| vec_to_pyarray(py, s.clone()))
        .collect();
    dict.set_item("snapshots", snapshots)?;
    dict.set_item("times", vec_to_pyarray(py, result.times.clone()))?;
    dict.set_item("steps", result.steps_taken)?;
    Ok(dict)
}

// ─── Grid Utilities ─────────────────────────────────────────────────────

/// Create a 1D uniform grid.
#[pyfunction]
fn make_grid_1d<'py>(py: Python<'py>, a: f64, b: f64, n: usize) -> Bound<'py, PyArray1<f64>> {
    let g = grid::Grid1D::uniform(a, b, n);
    vec_to_pyarray(py, g.x)
}

/// Version string for checking Rust backend availability.
#[pyfunction]
fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Check if the engine is available (always true if we got here).
#[pyfunction]
fn is_available() -> bool {
    true
}

// ─── 3D FEM Mesh / Assembly ────────────────────────────────────────────

#[pyclass(name = "Mesh3D")]
pub struct PyMesh3D {
    inner: crate::mesh::Mesh3D,
}

#[pymethods]
impl PyMesh3D {
    #[new]
    fn new(
        nodes: Vec<(f64, f64, f64)>,
        tet4_elements: Vec<(usize, usize, usize, usize)>,
    ) -> PyResult<Self> {
        let ns = nodes
            .into_iter()
            .map(|(x, y, z)| crate::mesh::Node3 { x, y, z })
            .collect();
        let es = tet4_elements
            .into_iter()
            .map(|(a, b, c, d)| crate::mesh::Element::Tet4([a, b, c, d]))
            .collect();
        let mesh = crate::mesh::Mesh3D::new(ns, es, Vec::new())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner: mesh })
    }

    #[staticmethod]
    fn from_gmsh_text(msh_text: &str) -> PyResult<Self> {
        let mesh = crate::gmsh_io::parse_msh_v41(msh_text)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner: mesh })
    }

    #[staticmethod]
    fn from_tet10(nodes: Vec<(f64, f64, f64)>, elements: Vec<[usize; 10]>) -> PyResult<Self> {
        let ns = nodes
            .into_iter()
            .map(|(x, y, z)| crate::mesh::Node3 { x, y, z })
            .collect();
        let es = elements
            .into_iter()
            .map(crate::mesh::Element::Tet10)
            .collect();
        let mesh = crate::mesh::Mesh3D::new(ns, es, Vec::new())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner: mesh })
    }

    #[staticmethod]
    fn from_hex8(nodes: Vec<(f64, f64, f64)>, elements: Vec<[usize; 8]>) -> PyResult<Self> {
        let ns = nodes
            .into_iter()
            .map(|(x, y, z)| crate::mesh::Node3 { x, y, z })
            .collect();
        let es = elements
            .into_iter()
            .map(crate::mesh::Element::Hex8)
            .collect();
        let mesh = crate::mesh::Mesh3D::new(ns, es, Vec::new())
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner: mesh })
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.nodes.len()
    }

    #[getter]
    fn num_elements(&self) -> usize {
        self.inner.elements.len()
    }
}

#[pyfunction]
#[pyo3(signature=(mesh, source=0.0))]
fn fem3d_assemble_poisson<'py>(
    py: Python<'py>,
    mesh: &PyMesh3D,
    source: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let out = crate::fem_assembly::assemble_poisson(&mesh.inner, source)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let d = PyDict::new_bound(py);
    let py_stiff = Py::new(
        py,
        PyCsrMatrix {
            inner: out.stiffness,
        },
    )?;
    d.set_item("stiffness", py_stiff)?;
    let py_mass = Py::new(py, PyCsrMatrix { inner: out.mass })?;
    d.set_item("mass", py_mass)?;
    d.set_item("rhs", vec_to_pyarray(py, out.rhs))?;
    Ok(d)
}

#[pyfunction]
fn fem3d_apply_neumann<'py>(
    py: Python<'py>,
    mesh: &PyMesh3D,
    rhs: PyReadonlyArray1<f64>,
    face_tag: i32,
    flux: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let mut r = rhs.as_slice()?.to_vec();
    crate::fem_bc::apply_neumann_flux(&mut r, &mesh.inner, face_tag, flux);
    Ok(vec_to_pyarray(py, r))
}

#[pyfunction]
fn fem3d_apply_dirichlet<'py>(
    py: Python<'py>,
    matrix: &PyCsrMatrix,
    rhs: PyReadonlyArray1<f64>,
    dof: Vec<usize>,
    values: Vec<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    if dof.len() != values.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dof and values lengths differ",
        ));
    }
    let mut bc = std::collections::HashMap::new();
    for (i, v) in dof.into_iter().zip(values.into_iter()) {
        bc.insert(i, v);
    }
    let (a2, b2) = crate::fem_bc::apply_dirichlet_elimination(&matrix.inner, rhs.as_slice()?, &bc)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let d = PyDict::new_bound(py);
    let py_mat = Py::new(py, PyCsrMatrix { inner: a2 })?;
    d.set_item("matrix", py_mat)?;
    d.set_item("rhs", vec_to_pyarray(py, b2))?;
    Ok(d)
}

#[pyfunction]
fn fem3d_export_vtu(
    mesh: &PyMesh3D,
    field_name: &str,
    values: PyReadonlyArray1<f64>,
) -> PyResult<String> {
    crate::vtk_export::to_vtu_ascii(&mesh.inner, field_name, values.as_slice()?)
        .map_err(pyo3::exceptions::PyValueError::new_err)
}

// ─── Module Registration ────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCsrMatrix>()?;
    m.add_class::<PyMesh3D>()?;

    // Solvers
    m.add_function(wrap_pyfunction!(solve_linear, m)?)?;

    // FDM assembly
    m.add_function(wrap_pyfunction!(assemble_laplacian_1d, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_laplacian_1d_variable, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_laplacian_2d, m)?)?;
    m.add_function(wrap_pyfunction!(assemble_laplacian_2d_variable, m)?)?;
    m.add_function(wrap_pyfunction!(apply_bc_1d, m)?)?;
    m.add_function(wrap_pyfunction!(apply_bc_2d, m)?)?;

    // Time steppers
    m.add_function(wrap_pyfunction!(timestep_forward_euler, m)?)?;
    m.add_function(wrap_pyfunction!(timestep_backward_euler, m)?)?;
    m.add_function(wrap_pyfunction!(timestep_crank_nicolson, m)?)?;
    m.add_function(wrap_pyfunction!(timestep_rk4, m)?)?;
    m.add_function(wrap_pyfunction!(timestep_bdf2, m)?)?;

    m.add_function(wrap_pyfunction!(fem3d_assemble_poisson, m)?)?;

    m.add_function(wrap_pyfunction!(fem3d_apply_neumann, m)?)?;

    m.add_function(wrap_pyfunction!(fem3d_apply_dirichlet, m)?)?;

    m.add_function(wrap_pyfunction!(fem3d_export_vtu, m)?)?;

    // Grid utilities
    m.add_function(wrap_pyfunction!(make_grid_1d, m)?)?;

    // Meta
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(is_available, m)?)?;

    Ok(())
}
