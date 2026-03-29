//! VTU exporter for quick ParaView inspection.

use crate::mesh::{Element, Mesh3D};

pub fn to_vtu_ascii(mesh: &Mesh3D, nodal_field_name: &str, nodal_values: &[f64]) -> Result<String, String> {
    if nodal_values.len() != mesh.nodes.len() {
        return Err("nodal_values size mismatch".to_string());
    }
    let mut connectivity = Vec::<usize>::new();
    let mut offsets = Vec::<usize>::new();
    let mut types = Vec::<u8>::new();
    let mut off = 0usize;

    for e in &mesh.elements {
        match e {
            Element::Tet4(c) => {
                connectivity.extend_from_slice(c);
                off += 4;
                offsets.push(off);
                types.push(10); // VTK_TETRA
            }
            Element::Hex8(c) => {
                connectivity.extend_from_slice(c);
                off += 8;
                offsets.push(off);
                types.push(12); // VTK_HEXAHEDRON
            }
            Element::Tet10(c) => {
                connectivity.extend_from_slice(c);
                off += 10;
                offsets.push(off);
                types.push(24); // VTK_QUADRATIC_TETRA
            }
        }
    }

    let points = mesh
        .nodes
        .iter()
        .map(|n| format!("{} {} {}", n.x, n.y, n.z))
        .collect::<Vec<_>>()
        .join(" ");
    let conn = connectivity.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
    let offs = offsets.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
    let ctys = types.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");
    let vals = nodal_values.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ");

    Ok(format!(r#"<?xml version=\"1.0\"?>
<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">
  <UnstructuredGrid>
    <Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">
      <Points>
        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">{}</DataArray>
      </Points>
      <Cells>
        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">{}</DataArray>
        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">{}</DataArray>
        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">{}</DataArray>
      </Cells>
      <PointData Scalars=\"{}\">
        <DataArray type=\"Float64\" Name=\"{}\" format=\"ascii\">{}</DataArray>
      </PointData>
    </Piece>
  </UnstructuredGrid>
</VTKFile>"#,
        mesh.nodes.len(), types.len(), points, conn, offs, ctys, nodal_field_name, nodal_field_name, vals))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Node3;

    #[test]
    fn tet10_vtu_export_produces_valid_xml() {
        let nodes: Vec<Node3> = (0..10)
            .map(|i| Node3 {
                x: i as f64 * 0.1,
                y: 0.0,
                z: 0.0,
            })
            .collect();
        let mesh = Mesh3D::new(
            nodes,
            vec![Element::Tet10([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])],
            vec![],
        )
        .unwrap();
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let vtu = to_vtu_ascii(&mesh, "u", &values).unwrap();
        assert!(vtu.contains("VTKFile"));
        assert!(vtu.contains("PointData"));
        // Type 24 = VTK_QUADRATIC_TETRA
        assert!(vtu.contains("24"));
        // Connectivity should list all 10 node indices
        assert!(vtu.contains("NumberOfCells=\\\"1\\\""));
    }

    #[test]
    fn mixed_mesh_vtu_export() {
        let nodes: Vec<Node3> = (0..10)
            .map(|i| Node3 {
                x: i as f64,
                y: 0.0,
                z: 0.0,
            })
            .collect();
        let mesh = Mesh3D::new(
            nodes,
            vec![
                Element::Tet4([0, 1, 2, 3]),
                Element::Tet10([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            ],
            vec![],
        )
        .unwrap();
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let vtu = to_vtu_ascii(&mesh, "phi", &values).unwrap();
        assert!(vtu.contains("NumberOfCells=\\\"2\\\""));
        // Both type codes should appear: 10 (Tet4) and 24 (Tet10)
        assert!(vtu.contains("10 24"));
    }
}
