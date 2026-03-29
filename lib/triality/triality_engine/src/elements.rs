//! Finite elements and quadrature for 3D Poisson assembly.

use crate::mesh::Node3;

#[derive(Debug, Clone)]
pub struct ElementMatrices {
    pub ke: Vec<f64>,
    pub me: Vec<f64>,
    pub fe: Vec<f64>,
    pub n: usize,
}

#[derive(Clone, Copy)]
struct ShapeData<const N: usize> {
    n: [f64; N],
    dn_dxi: [[f64; 3]; N],
}

fn det3(a: [[f64; 3]; 3]) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

fn inv3(a: [[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let d = det3(a);
    if d.abs() < 1e-18 {
        return None;
    }
    let inv_d = 1.0 / d;
    let m = [
        [
            (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * inv_d,
            (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * inv_d,
            (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * inv_d,
        ],
        [
            (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * inv_d,
            (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * inv_d,
            (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * inv_d,
        ],
        [
            (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * inv_d,
            (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * inv_d,
            (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * inv_d,
        ],
    ];
    Some(m)
}

fn mat_vec3(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

fn tet10_shape(r: f64, s: f64, t: f64) -> ShapeData<10> {
    let l1 = 1.0 - r - s - t;
    let l2 = r;
    let l3 = s;
    let l4 = t;

    let n = [
        l1 * (2.0 * l1 - 1.0),
        l2 * (2.0 * l2 - 1.0),
        l3 * (2.0 * l3 - 1.0),
        l4 * (2.0 * l4 - 1.0),
        4.0 * l1 * l2,
        4.0 * l2 * l3,
        4.0 * l3 * l1,
        4.0 * l1 * l4,
        4.0 * l2 * l4,
        4.0 * l3 * l4,
    ];

    let dl = [
        [-1.0, -1.0, -1.0], // dL1/dr, dL1/ds, dL1/dt
        [1.0, 0.0, 0.0],    // dL2/...
        [0.0, 1.0, 0.0],    // dL3/...
        [0.0, 0.0, 1.0],    // dL4/...
    ];

    let dnd_l = [
        [4.0 * l1 - 1.0, 0.0, 0.0, 0.0], // N1
        [0.0, 4.0 * l2 - 1.0, 0.0, 0.0], // N2
        [0.0, 0.0, 4.0 * l3 - 1.0, 0.0], // N3
        [0.0, 0.0, 0.0, 4.0 * l4 - 1.0], // N4
        [4.0 * l2, 4.0 * l1, 0.0, 0.0],  // N5 = 4 L1L2
        [0.0, 4.0 * l3, 4.0 * l2, 0.0],  // N6 = 4 L2L3
        [4.0 * l3, 0.0, 4.0 * l1, 0.0],  // N7 = 4 L3L1
        [4.0 * l4, 0.0, 0.0, 4.0 * l1],  // N8 = 4 L1L4
        [0.0, 4.0 * l4, 0.0, 4.0 * l2],  // N9 = 4 L2L4
        [0.0, 0.0, 4.0 * l4, 4.0 * l3],  // N10 = 4 L3L4
    ];

    let mut dn_dxi = [[0.0; 3]; 10];
    for a in 0..10 {
        for k in 0..3 {
            dn_dxi[a][k] = dnd_l[a][0] * dl[0][k]
                + dnd_l[a][1] * dl[1][k]
                + dnd_l[a][2] * dl[2][k]
                + dnd_l[a][3] * dl[3][k];
        }
    }

    ShapeData { n, dn_dxi }
}

fn hex8_shape(xi: f64, eta: f64, zeta: f64) -> ShapeData<8> {
    let signs = [
        [-1.0, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let mut n = [0.0; 8];
    let mut dn_dxi = [[0.0; 3]; 8];
    for a in 0..8 {
        let sx = signs[a][0];
        let sy = signs[a][1];
        let sz = signs[a][2];
        n[a] = 0.125 * (1.0 + sx * xi) * (1.0 + sy * eta) * (1.0 + sz * zeta);
        dn_dxi[a][0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta);
        dn_dxi[a][1] = 0.125 * sy * (1.0 + sx * xi) * (1.0 + sz * zeta);
        dn_dxi[a][2] = 0.125 * sz * (1.0 + sx * xi) * (1.0 + sy * eta);
    }
    ShapeData { n, dn_dxi }
}

pub fn tet4_poisson(nodes: [Node3; 4], source: f64) -> Result<ElementMatrices, String> {
    let x1 = nodes[0].as_array();
    let x2 = nodes[1].as_array();
    let x3 = nodes[2].as_array();
    let x4 = nodes[3].as_array();

    let j = [
        [x2[0] - x1[0], x3[0] - x1[0], x4[0] - x1[0]],
        [x2[1] - x1[1], x3[1] - x1[1], x4[1] - x1[1]],
        [x2[2] - x1[2], x3[2] - x1[2], x4[2] - x1[2]],
    ];
    let det_j = det3(j);
    if det_j.abs() < 1e-18 {
        return Err("Degenerate Tet4 element (zero volume)".to_string());
    }
    let vol = det_j.abs() / 6.0;

    let inv_j = inv3(j).ok_or_else(|| "Singular Tet4 Jacobian".to_string())?;

    // Reference gradients for barycentric-linear basis.
    let g_ref = [
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    let mut g_phys = [[0.0; 3]; 4];
    for a in 0..4 {
        for i in 0..3 {
            g_phys[a][i] =
                inv_j[0][i] * g_ref[a][0] + inv_j[1][i] * g_ref[a][1] + inv_j[2][i] * g_ref[a][2];
        }
    }

    let mut ke = vec![0.0; 16];
    let mut me = vec![0.0; 16];
    let mut fe = vec![0.0; 4];

    for a in 0..4 {
        fe[a] = source * vol / 4.0;
        for b in 0..4 {
            let dot = g_phys[a][0] * g_phys[b][0]
                + g_phys[a][1] * g_phys[b][1]
                + g_phys[a][2] * g_phys[b][2];
            ke[a * 4 + b] = dot * vol;
            me[a * 4 + b] = if a == b { vol / 10.0 } else { vol / 20.0 };
        }
    }

    Ok(ElementMatrices { ke, me, fe, n: 4 })
}

pub fn tet10_poisson(nodes: [Node3; 10], source: f64) -> Result<ElementMatrices, String> {
    let mut ke = vec![0.0; 100];
    let mut me = vec![0.0; 100];
    let mut fe = vec![0.0; 10];

    // Symmetric 4-point tetrahedron rule (degree 2 exact).
    let a = 0.5854101966249685;
    let b = 0.1381966011250105;
    let w = 1.0 / 24.0;
    let gps = [[b, b, b], [a, b, b], [b, a, b], [b, b, a]];

    for gp in gps {
        let sh = tet10_shape(gp[0], gp[1], gp[2]);

        let mut j = [[0.0; 3]; 3];
        for a in 0..10 {
            let x = nodes[a].x;
            let y = nodes[a].y;
            let z = nodes[a].z;
            for k in 0..3 {
                j[0][k] += x * sh.dn_dxi[a][k];
                j[1][k] += y * sh.dn_dxi[a][k];
                j[2][k] += z * sh.dn_dxi[a][k];
            }
        }

        let det_j = det3(j);
        if det_j.abs() < 1e-18 {
            return Err("Degenerate Tet10 element (zero volume at quadrature point)".to_string());
        }
        let inv_j = inv3(j).ok_or_else(|| "Singular Tet10 Jacobian".to_string())?;
        let weight = det_j.abs() * w;

        let mut grad_phys = [[0.0; 3]; 10];
        for a in 0..10 {
            grad_phys[a] = mat_vec3(inv_j, sh.dn_dxi[a]);
        }

        for a in 0..10 {
            fe[a] += source * sh.n[a] * weight;
            for b in 0..10 {
                let dot = grad_phys[a][0] * grad_phys[b][0]
                    + grad_phys[a][1] * grad_phys[b][1]
                    + grad_phys[a][2] * grad_phys[b][2];
                ke[a * 10 + b] += dot * weight;
                me[a * 10 + b] += sh.n[a] * sh.n[b] * weight;
            }
        }
    }

    Ok(ElementMatrices { ke, me, fe, n: 10 })
}

pub fn hex8_poisson(nodes: [Node3; 8], source: f64) -> Result<ElementMatrices, String> {
    let mut ke = vec![0.0; 64];
    let mut me = vec![0.0; 64];
    let mut fe = vec![0.0; 8];

    let g = 1.0_f64 / 3.0_f64.sqrt();
    let gauss = [-g, g];

    for &xi in &gauss {
        for &eta in &gauss {
            for &zeta in &gauss {
                let sh = hex8_shape(xi, eta, zeta);
                let mut j = [[0.0; 3]; 3];
                for a in 0..8 {
                    let x = nodes[a].x;
                    let y = nodes[a].y;
                    let z = nodes[a].z;
                    for k in 0..3 {
                        j[0][k] += x * sh.dn_dxi[a][k];
                        j[1][k] += y * sh.dn_dxi[a][k];
                        j[2][k] += z * sh.dn_dxi[a][k];
                    }
                }
                let det_j = det3(j);
                if det_j.abs() < 1e-18 {
                    return Err(
                        "Degenerate Hex8 element (zero volume at quadrature point)".to_string()
                    );
                }
                let inv_j = inv3(j).ok_or_else(|| "Singular Hex8 Jacobian".to_string())?;
                let weight = det_j.abs(); // unit weights for 2x2x2 in [-1,1]^3

                let mut grad_phys = [[0.0; 3]; 8];
                for a in 0..8 {
                    grad_phys[a] = mat_vec3(inv_j, sh.dn_dxi[a]);
                }

                for a in 0..8 {
                    fe[a] += source * sh.n[a] * weight;
                    for b in 0..8 {
                        let dot = grad_phys[a][0] * grad_phys[b][0]
                            + grad_phys[a][1] * grad_phys[b][1]
                            + grad_phys[a][2] * grad_phys[b][2];
                        ke[a * 8 + b] += dot * weight;
                        me[a * 8 + b] += sh.n[a] * sh.n[b] * weight;
                    }
                }
            }
        }
    }

    Ok(ElementMatrices { ke, me, fe, n: 8 })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(v: &[f64]) -> f64 {
        v.iter().sum::<f64>()
    }

    #[test]
    fn tet10_element_assembly_has_expected_sizes() {
        let nodes = [
            Node3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            Node3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            Node3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            Node3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            Node3 {
                x: 0.5,
                y: 0.0,
                z: 0.0,
            },
            Node3 {
                x: 0.5,
                y: 0.5,
                z: 0.0,
            },
            Node3 {
                x: 0.0,
                y: 0.5,
                z: 0.0,
            },
            Node3 {
                x: 0.0,
                y: 0.0,
                z: 0.5,
            },
            Node3 {
                x: 0.5,
                y: 0.0,
                z: 0.5,
            },
            Node3 {
                x: 0.0,
                y: 0.5,
                z: 0.5,
            },
        ];
        let em = tet10_poisson(nodes, 2.0).unwrap();
        assert_eq!(em.ke.len(), 100);
        assert_eq!(em.me.len(), 100);
        assert_eq!(em.fe.len(), 10);
        assert!((sum(&em.fe) - (2.0 / 6.0)).abs() < 1e-10);
    }

    #[test]
    fn hex8_element_assembly_has_expected_sizes() {
        let nodes = [
            Node3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            Node3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            Node3 {
                x: 1.0,
                y: 1.0,
                z: 0.0,
            },
            Node3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            Node3 {
                x: 0.0,
                y: 0.0,
                z: 1.0,
            },
            Node3 {
                x: 1.0,
                y: 0.0,
                z: 1.0,
            },
            Node3 {
                x: 1.0,
                y: 1.0,
                z: 1.0,
            },
            Node3 {
                x: 0.0,
                y: 1.0,
                z: 1.0,
            },
        ];
        let em = hex8_poisson(nodes, 3.0).unwrap();
        assert_eq!(em.ke.len(), 64);
        assert_eq!(em.me.len(), 64);
        assert_eq!(em.fe.len(), 8);
        assert!((sum(&em.fe) - 3.0).abs() < 1e-10);
    }
}
