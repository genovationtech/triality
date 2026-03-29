//! Global FEM assembly for 3D Poisson-like scalar problems.

use crate::elements::{hex8_poisson, tet10_poisson, tet4_poisson};
use crate::mesh::{Element, Mesh3D};
use crate::sparse::{CsrMatrix, TripletBuilder};

pub struct AssemblyResult {
    pub stiffness: CsrMatrix,
    pub mass: CsrMatrix,
    pub rhs: Vec<f64>,
}

pub fn assemble_poisson(mesh: &Mesh3D, source: f64) -> Result<AssemblyResult, String> {
    let n = mesh.ndof();
    let nnz_hint = mesh
        .elements
        .iter()
        .map(|e| match e {
            Element::Tet4(_) => 16,
            Element::Tet10(_) => 100,
            Element::Hex8(_) => 64,
        })
        .sum();
    let mut k_builder = TripletBuilder::with_capacity(n, n, nnz_hint);
    let mut m_builder = TripletBuilder::with_capacity(n, n, nnz_hint);
    let mut rhs = vec![0.0; n];

    for e in &mesh.elements {
        match e {
            Element::Tet4(conn) => {
                let nodes = [
                    mesh.nodes[conn[0]],
                    mesh.nodes[conn[1]],
                    mesh.nodes[conn[2]],
                    mesh.nodes[conn[3]],
                ];
                let em = tet4_poisson(nodes, source)?;
                scatter(conn, &em.ke, em.n, &mut k_builder);
                scatter(conn, &em.me, em.n, &mut m_builder);
                for a in 0..4 {
                    rhs[conn[a]] += em.fe[a];
                }
            }
            Element::Tet10(conn) => {
                let nodes = [
                    mesh.nodes[conn[0]],
                    mesh.nodes[conn[1]],
                    mesh.nodes[conn[2]],
                    mesh.nodes[conn[3]],
                    mesh.nodes[conn[4]],
                    mesh.nodes[conn[5]],
                    mesh.nodes[conn[6]],
                    mesh.nodes[conn[7]],
                    mesh.nodes[conn[8]],
                    mesh.nodes[conn[9]],
                ];
                let em = tet10_poisson(nodes, source)?;
                scatter(conn, &em.ke, em.n, &mut k_builder);
                scatter(conn, &em.me, em.n, &mut m_builder);
                for a in 0..10 {
                    rhs[conn[a]] += em.fe[a];
                }
            }
            Element::Hex8(conn) => {
                let nodes = [
                    mesh.nodes[conn[0]],
                    mesh.nodes[conn[1]],
                    mesh.nodes[conn[2]],
                    mesh.nodes[conn[3]],
                    mesh.nodes[conn[4]],
                    mesh.nodes[conn[5]],
                    mesh.nodes[conn[6]],
                    mesh.nodes[conn[7]],
                ];
                let em = hex8_poisson(nodes, source)?;
                scatter(conn, &em.ke, em.n, &mut k_builder);
                scatter(conn, &em.me, em.n, &mut m_builder);
                for a in 0..8 {
                    rhs[conn[a]] += em.fe[a];
                }
            }
        }
    }

    Ok(AssemblyResult {
        stiffness: k_builder.build(),
        mass: m_builder.build(),
        rhs,
    })
}

fn scatter<const N: usize>(
    conn: &[usize; N],
    local: &[f64],
    nloc: usize,
    global: &mut TripletBuilder,
) {
    for a in 0..nloc {
        for b in 0..nloc {
            global.add(conn[a], conn[b], local[a * nloc + b]);
        }
    }
}
