//! Minimal Gmsh v4.1 ASCII reader for common 3D volume/surface element types.

use std::collections::HashMap;

use crate::mesh::{BoundaryFace, Element, Mesh3D, Node3};

pub fn parse_msh_v41(contents: &str) -> Result<Mesh3D, String> {
    let mut lines = contents.lines().peekable();
    let mut nodes: HashMap<usize, Node3> = HashMap::new();
    let mut elements: Vec<Element> = Vec::new();
    let mut bfaces: Vec<BoundaryFace> = Vec::new();

    while let Some(line) = lines.next() {
        match line.trim() {
            "$Nodes" => parse_nodes(&mut lines, &mut nodes)?,
            "$Elements" => parse_elements(&mut lines, &mut elements, &mut bfaces)?,
            _ => {}
        }
    }

    if nodes.is_empty() {
        return Err("No nodes parsed from msh".to_string());
    }

    let mut sorted: Vec<(usize, Node3)> = nodes.into_iter().collect();
    sorted.sort_by_key(|(id, _)| *id);
    let id_to_idx: HashMap<usize, usize> = sorted
        .iter()
        .enumerate()
        .map(|(i, (id, _))| (*id, i))
        .collect();
    let node_vec: Vec<Node3> = sorted.into_iter().map(|(_, n)| n).collect();

    let reindex = |id: usize| -> Result<usize, String> {
        id_to_idx
            .get(&id)
            .copied()
            .ok_or_else(|| format!("Unknown node id {}", id))
    };

    let mut elems_idx = Vec::new();
    for e in elements {
        match e {
            Element::Tet4(conn) => elems_idx.push(Element::Tet4([
                reindex(conn[0])?,
                reindex(conn[1])?,
                reindex(conn[2])?,
                reindex(conn[3])?,
            ])),
            Element::Tet10(conn) => elems_idx.push(Element::Tet10([
                reindex(conn[0])?,
                reindex(conn[1])?,
                reindex(conn[2])?,
                reindex(conn[3])?,
                reindex(conn[4])?,
                reindex(conn[5])?,
                reindex(conn[6])?,
                reindex(conn[7])?,
                reindex(conn[8])?,
                reindex(conn[9])?,
            ])),
            Element::Hex8(conn) => elems_idx.push(Element::Hex8([
                reindex(conn[0])?,
                reindex(conn[1])?,
                reindex(conn[2])?,
                reindex(conn[3])?,
                reindex(conn[4])?,
                reindex(conn[5])?,
                reindex(conn[6])?,
                reindex(conn[7])?,
            ])),
        }
    }

    let mut face_idx = Vec::new();
    for f in bfaces {
        face_idx.push(BoundaryFace {
            nodes: [
                reindex(f.nodes[0])?,
                reindex(f.nodes[1])?,
                reindex(f.nodes[2])?,
            ],
            tag: f.tag,
        });
    }

    Mesh3D::new(node_vec, elems_idx, face_idx)
}

fn parse_nodes<'a, I>(
    lines: &mut std::iter::Peekable<I>,
    out: &mut HashMap<usize, Node3>,
) -> Result<(), String>
where
    I: Iterator<Item = &'a str>,
{
    let header = lines
        .next()
        .ok_or_else(|| "Missing $Nodes header line".to_string())?;
    let h: Vec<usize> = header
        .split_whitespace()
        .map(|x| x.parse::<usize>().unwrap_or(0))
        .collect();
    if h.len() < 4 {
        return Err("Invalid $Nodes header".to_string());
    }
    let nblocks = h[0];
    for _ in 0..nblocks {
        let bh = lines
            .next()
            .ok_or_else(|| "Missing node block header".to_string())?;
        let b: Vec<usize> = bh
            .split_whitespace()
            .map(|x| x.parse::<usize>().unwrap_or(0))
            .collect();
        if b.len() < 4 {
            return Err("Invalid node block header".to_string());
        }
        let n = b[3];
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            ids.push(
                lines
                    .next()
                    .ok_or_else(|| "Missing node tag".to_string())?
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| "Bad node tag")?,
            );
        }
        for id in ids {
            let c = lines
                .next()
                .ok_or_else(|| "Missing node coordinates".to_string())?;
            let vals: Vec<f64> = c
                .split_whitespace()
                .map(|x| x.parse::<f64>().unwrap_or(0.0))
                .collect();
            if vals.len() < 3 {
                return Err("Invalid node coordinate line".to_string());
            }
            out.insert(
                id,
                Node3 {
                    x: vals[0],
                    y: vals[1],
                    z: vals[2],
                },
            );
        }
    }
    // consume $EndNodes
    while let Some(l) = lines.next() {
        if l.trim() == "$EndNodes" {
            break;
        }
    }
    Ok(())
}

fn parse_elements<'a, I>(
    lines: &mut std::iter::Peekable<I>,
    elems: &mut Vec<Element>,
    faces: &mut Vec<BoundaryFace>,
) -> Result<(), String>
where
    I: Iterator<Item = &'a str>,
{
    let header = lines
        .next()
        .ok_or_else(|| "Missing $Elements header line".to_string())?;
    let h: Vec<usize> = header
        .split_whitespace()
        .map(|x| x.parse::<usize>().unwrap_or(0))
        .collect();
    if h.len() < 4 {
        return Err("Invalid $Elements header".to_string());
    }
    let nblocks = h[0];

    for _ in 0..nblocks {
        let bh = lines
            .next()
            .ok_or_else(|| "Missing element block header".to_string())?;
        let b: Vec<usize> = bh
            .split_whitespace()
            .map(|x| x.parse::<usize>().unwrap_or(0))
            .collect();
        if b.len() < 4 {
            return Err("Invalid element block header".to_string());
        }
        let entity_tag = b[1] as i32;
        let etype = b[2];
        let n = b[3];
        for _ in 0..n {
            let row = lines
                .next()
                .ok_or_else(|| "Missing element row".to_string())?;
            let v: Vec<usize> = row
                .split_whitespace()
                .map(|x| x.parse::<usize>().unwrap_or(0))
                .collect();
            if v.len() < 2 {
                continue;
            }
            match etype {
                2 if v.len() >= 4 => {
                    faces.push(BoundaryFace {
                        nodes: [v[1], v[2], v[3]],
                        tag: entity_tag,
                    });
                }
                3 if v.len() >= 5 => {
                    // Quad face triangulated into two Tri3 entries.
                    faces.push(BoundaryFace {
                        nodes: [v[1], v[2], v[3]],
                        tag: entity_tag,
                    });
                    faces.push(BoundaryFace {
                        nodes: [v[1], v[3], v[4]],
                        tag: entity_tag,
                    });
                }
                4 if v.len() >= 5 => {
                    elems.push(Element::Tet4([v[1], v[2], v[3], v[4]]));
                }
                5 if v.len() >= 9 => {
                    elems.push(Element::Hex8([
                        v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
                    ]));
                }
                11 if v.len() >= 11 => {
                    elems.push(Element::Tet10([
                        v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10],
                    ]));
                }
                _ => {}
            }
        }
    }

    while let Some(l) = lines.next() {
        if l.trim() == "$EndElements" {
            break;
        }
    }
    Ok(())
}
