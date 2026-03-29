//! FEM boundary condition operators.

use std::collections::HashMap;

use crate::mesh::Mesh3D;
use crate::sparse::CsrMatrix;

pub fn apply_dirichlet_elimination(
    a: &CsrMatrix,
    rhs: &[f64],
    bc: &HashMap<usize, f64>,
) -> Result<(CsrMatrix, Vec<f64>), String> {
    if rhs.len() != a.nrows {
        return Err("RHS length mismatch".to_string());
    }
    if a.nrows != a.ncols {
        return Err("Dirichlet elimination expects a square matrix".to_string());
    }

    let mut is_fixed = vec![false; a.nrows];
    let mut fixed_value = vec![0.0; a.nrows];
    for (&dof, &v) in bc {
        if dof >= a.nrows {
            return Err(format!("Dirichlet dof {} out of range {}", dof, a.nrows));
        }
        is_fixed[dof] = true;
        fixed_value[dof] = v;
    }

    let mut b = rhs.to_vec();
    for i in 0..a.nrows {
        if is_fixed[i] {
            continue;
        }
        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[idx];
            if is_fixed[j] {
                b[i] -= a.values[idx] * fixed_value[j];
            }
        }
    }

    let mut row_ptr = Vec::with_capacity(a.nrows + 1);
    row_ptr.push(0);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    for i in 0..a.nrows {
        if is_fixed[i] {
            col_idx.push(i);
            values.push(1.0);
            b[i] = fixed_value[i];
            row_ptr.push(col_idx.len());
            continue;
        }

        for idx in a.row_ptr[i]..a.row_ptr[i + 1] {
            let j = a.col_idx[idx];
            if is_fixed[j] {
                continue;
            }
            col_idx.push(j);
            values.push(a.values[idx]);
        }
        row_ptr.push(col_idx.len());
    }

    Ok((
        CsrMatrix::new(a.nrows, a.ncols, row_ptr, col_idx, values)?,
        b,
    ))
}

pub fn apply_neumann_flux(rhs: &mut [f64], mesh: &Mesh3D, face_tag: i32, flux: f64) {
    if rhs.is_empty() {
        return;
    }
    for face in &mesh.boundary_faces {
        if face.tag != face_tag {
            continue;
        }
        if face.nodes.iter().any(|&idx| idx >= rhs.len()) {
            continue;
        }
        let n0 = mesh.nodes[face.nodes[0]];
        let n1 = mesh.nodes[face.nodes[1]];
        let n2 = mesh.nodes[face.nodes[2]];
        let a = [n1.x - n0.x, n1.y - n0.y, n1.z - n0.z];
        let b = [n2.x - n0.x, n2.y - n0.y, n2.z - n0.z];
        let cx = a[1] * b[2] - a[2] * b[1];
        let cy = a[2] * b[0] - a[0] * b[2];
        let cz = a[0] * b[1] - a[1] * b[0];
        let area = 0.5 * (cx * cx + cy * cy + cz * cz).sqrt();
        let contrib = flux * area / 3.0;
        rhs[face.nodes[0]] += contrib;
        rhs[face.nodes[1]] += contrib;
        rhs[face.nodes[2]] += contrib;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse::TripletBuilder;

    #[test]
    fn dirichlet_elimination_preserves_sparse_structure() {
        let mut tb = TripletBuilder::new(4, 4);
        for i in 0..4 {
            tb.add(i, i, 2.0);
        }
        tb.add(1, 2, -1.0);
        tb.add(2, 1, -1.0);
        let a = tb.build();
        let rhs = vec![0.0, 1.0, 2.0, 3.0];
        let bc = HashMap::from([(0usize, 5.0), (3usize, 7.0)]);
        let (a2, b2) = apply_dirichlet_elimination(&a, &rhs, &bc).unwrap();
        assert_eq!(a2.nrows, 4);
        assert_eq!(a2.ncols, 4);
        assert_eq!(b2[0], 5.0);
        assert_eq!(b2[3], 7.0);
        assert!(a2.nnz() <= a.nnz() + bc.len());
    }
}
