//! 3D unstructured mesh primitives.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Node3 {
    #[inline]
    pub fn as_array(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Element {
    Tet4([usize; 4]),
    Tet10([usize; 10]),
    Hex8([usize; 8]),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundaryFace {
    pub nodes: [usize; 3],
    pub tag: i32,
}

#[derive(Debug, Clone)]
pub struct Mesh3D {
    pub nodes: Vec<Node3>,
    pub elements: Vec<Element>,
    pub boundary_faces: Vec<BoundaryFace>,
}

impl Mesh3D {
    pub fn new(nodes: Vec<Node3>, elements: Vec<Element>, boundary_faces: Vec<BoundaryFace>) -> Result<Self, String> {
        let mesh = Self {
            nodes,
            elements,
            boundary_faces,
        };
        mesh.validate()?;
        Ok(mesh)
    }

    pub fn validate(&self) -> Result<(), String> {
        let n = self.nodes.len();
        for (eid, e) in self.elements.iter().enumerate() {
            match e {
                Element::Tet4(conn) => {
                    for &i in conn {
                        if i >= n {
                            return Err(format!("Tet4 {} references node {} >= {}", eid, i, n));
                        }
                    }
                }
                Element::Tet10(conn) => {
                    for &i in conn {
                        if i >= n {
                            return Err(format!("Tet10 {} references node {} >= {}", eid, i, n));
                        }
                    }
                }
                Element::Hex8(conn) => {
                    for &i in conn {
                        if i >= n {
                            return Err(format!("Hex8 {} references node {} >= {}", eid, i, n));
                        }
                    }
                }
            }
        }
        for (fid, f) in self.boundary_faces.iter().enumerate() {
            for &i in &f.nodes {
                if i >= n {
                    return Err(format!("Boundary face {} references node {} >= {}", fid, i, n));
                }
            }
        }
        Ok(())
    }

    #[inline]
    pub fn ndof(&self) -> usize {
        self.nodes.len()
    }
}
