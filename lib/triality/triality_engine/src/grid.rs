//! Grid types and mesh utilities.
//!
//! Provides structured grid representations used by FDM and physics layers.

/// 1D uniform grid.
#[derive(Debug, Clone)]
pub struct Grid1D {
    pub x: Vec<f64>,
    pub h: f64,
    pub n: usize,
}

impl Grid1D {
    pub fn uniform(a: f64, b: f64, n: usize) -> Self {
        let h = (b - a) / (n as f64 - 1.0);
        let x: Vec<f64> = (0..n).map(|i| a + i as f64 * h).collect();
        Self { x, h, n }
    }

    /// Map a physical coordinate to the nearest grid index.
    pub fn nearest_index(&self, x_val: f64) -> usize {
        let a = self.x[0];
        let idx = ((x_val - a) / self.h).round() as isize;
        idx.max(0).min(self.n as isize - 1) as usize
    }
}

/// 2D uniform grid (tensor product of two 1D grids).
#[derive(Debug, Clone)]
pub struct Grid2D {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub dx: f64,
    pub dy: f64,
    pub nx: usize,
    pub ny: usize,
}

impl Grid2D {
    pub fn uniform(x0: f64, x1: f64, y0: f64, y1: f64, nx: usize, ny: usize) -> Self {
        let dx = (x1 - x0) / (nx as f64 - 1.0);
        let dy = (y1 - y0) / (ny as f64 - 1.0);
        let x: Vec<f64> = (0..nx).map(|i| x0 + i as f64 * dx).collect();
        let y: Vec<f64> = (0..ny).map(|j| y0 + j as f64 * dy).collect();
        Self { x, y, dx, dy, nx, ny }
    }

    /// Total number of degrees of freedom.
    pub fn ndof(&self) -> usize {
        self.nx * self.ny
    }

    /// Convert 2D index (i, j) to flat 1D index.
    #[inline]
    pub fn flat_index(&self, i: usize, j: usize) -> usize {
        i * self.ny + j
    }

    /// Convert flat 1D index to 2D index (i, j).
    #[inline]
    pub fn grid_index(&self, k: usize) -> (usize, usize) {
        (k / self.ny, k % self.ny)
    }

    /// Check if (i, j) is a boundary point.
    #[inline]
    pub fn is_boundary(&self, i: usize, j: usize) -> bool {
        i == 0 || i == self.nx - 1 || j == 0 || j == self.ny - 1
    }

    /// Map physical coordinates to nearest grid index.
    pub fn nearest_index(&self, x_val: f64, y_val: f64) -> (usize, usize) {
        let i = ((x_val - self.x[0]) / self.dx).round() as isize;
        let j = ((y_val - self.y[0]) / self.dy).round() as isize;
        (
            i.max(0).min(self.nx as isize - 1) as usize,
            j.max(0).min(self.ny as isize - 1) as usize,
        )
    }
}

/// 3D uniform grid.
#[derive(Debug, Clone)]
pub struct Grid3D {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
}

impl Grid3D {
    pub fn uniform(
        x0: f64, x1: f64,
        y0: f64, y1: f64,
        z0: f64, z1: f64,
        nx: usize, ny: usize, nz: usize,
    ) -> Self {
        let dx = (x1 - x0) / (nx as f64 - 1.0);
        let dy = (y1 - y0) / (ny as f64 - 1.0);
        let dz = (z1 - z0) / (nz as f64 - 1.0);
        let x: Vec<f64> = (0..nx).map(|i| x0 + i as f64 * dx).collect();
        let y: Vec<f64> = (0..ny).map(|j| y0 + j as f64 * dy).collect();
        let z: Vec<f64> = (0..nz).map(|k| z0 + k as f64 * dz).collect();
        Self { x, y, z, dx, dy, dz, nx, ny, nz }
    }

    pub fn ndof(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    #[inline]
    pub fn flat_index(&self, i: usize, j: usize, k: usize) -> usize {
        (i * self.ny + j) * self.nz + k
    }

    #[inline]
    pub fn is_boundary(&self, i: usize, j: usize, k: usize) -> bool {
        i == 0 || i == self.nx - 1
            || j == 0 || j == self.ny - 1
            || k == 0 || k == self.nz - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_1d() {
        let g = Grid1D::uniform(0.0, 1.0, 11);
        assert_eq!(g.n, 11);
        assert!((g.h - 0.1).abs() < 1e-12);
        assert_eq!(g.nearest_index(0.35), 3); // 0.35 rounds to 0.3 (index 3) with h=0.1
    }

    #[test]
    fn test_grid_2d() {
        let g = Grid2D::uniform(0.0, 1.0, 0.0, 1.0, 5, 5);
        assert_eq!(g.ndof(), 25);
        assert_eq!(g.flat_index(2, 3), 13);
        assert_eq!(g.grid_index(13), (2, 3));
        assert!(g.is_boundary(0, 2));
        assert!(!g.is_boundary(2, 2));
    }

    #[test]
    fn test_grid_3d() {
        let g = Grid3D::uniform(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3, 3, 3);
        assert_eq!(g.ndof(), 27);
        assert!(g.is_boundary(0, 1, 1));
        assert!(!g.is_boundary(1, 1, 1));
    }
}
