//! Compressed Sparse Row (CSR) matrix and operations.
//!
//! This is the core data structure for all Triality linear algebra.
//! CSR is optimal for SpMV (the inner loop of every iterative solver).

use rayon::prelude::*;
use std::fmt;

/// CSR sparse matrix: the workhorse of Triality numerics.
#[derive(Clone)]
pub struct CsrMatrix {
    /// Number of rows.
    pub nrows: usize,
    /// Number of columns.
    pub ncols: usize,
    /// Row pointers (length nrows + 1). row i has entries in indices[row_ptr[i]..row_ptr[i+1]].
    pub row_ptr: Vec<usize>,
    /// Column indices for each non-zero entry.
    pub col_idx: Vec<usize>,
    /// Non-zero values.
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Create a new CSR matrix from raw components. Validates structure.
    pub fn new(
        nrows: usize,
        ncols: usize,
        row_ptr: Vec<usize>,
        col_idx: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self, String> {
        if row_ptr.len() != nrows + 1 {
            return Err(format!(
                "row_ptr length {} != nrows + 1 = {}",
                row_ptr.len(),
                nrows + 1
            ));
        }
        if col_idx.len() != values.len() {
            return Err(format!(
                "col_idx length {} != values length {}",
                col_idx.len(),
                values.len()
            ));
        }
        let nnz = *row_ptr.last().unwrap_or(&0);
        if col_idx.len() != nnz {
            return Err(format!(
                "col_idx length {} != nnz from row_ptr {}",
                col_idx.len(),
                nnz
            ));
        }
        Ok(Self {
            nrows,
            ncols,
            row_ptr,
            col_idx,
            values,
        })
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparse matrix-vector product: y = A * x.
    /// This is the single hottest operation in iterative solvers.
    pub fn spmv(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.ncols, "spmv: x length mismatch");
        let mut y = vec![0.0; self.nrows];
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut sum = 0.0;
            for idx in start..end {
                sum += self.values[idx] * x[self.col_idx[idx]];
            }
            y[i] = sum;
        }
        y
    }

    /// Parallel SpMV using rayon. Wins for nrows > ~2000.
    pub fn spmv_par(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.ncols, "spmv_par: x length mismatch");
        (0..self.nrows)
            .into_par_iter()
            .map(|i| {
                let start = self.row_ptr[i];
                let end = self.row_ptr[i + 1];
                let mut sum = 0.0;
                for idx in start..end {
                    sum += self.values[idx] * x[self.col_idx[idx]];
                }
                sum
            })
            .collect()
    }

    /// Extract diagonal entries (for Jacobi preconditioner).
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.nrows.min(self.ncols);
        let mut diag = vec![0.0; n];
        for i in 0..n {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                if self.col_idx[idx] == i {
                    diag[i] = self.values[idx];
                    break;
                }
            }
        }
        diag
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> CsrMatrix {
        let mut row_ptr_t = vec![0usize; self.ncols + 1];
        // Count entries per column
        for &c in &self.col_idx {
            row_ptr_t[c + 1] += 1;
        }
        // Prefix sum
        for i in 1..=self.ncols {
            row_ptr_t[i] += row_ptr_t[i - 1];
        }
        let nnz = self.nnz();
        let mut col_idx_t = vec![0usize; nnz];
        let mut values_t = vec![0.0f64; nnz];
        let mut pos = row_ptr_t.clone();
        for i in 0..self.nrows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                let c = self.col_idx[idx];
                let p = pos[c];
                col_idx_t[p] = i;
                values_t[p] = self.values[idx];
                pos[c] += 1;
            }
        }
        CsrMatrix {
            nrows: self.ncols,
            ncols: self.nrows,
            row_ptr: row_ptr_t,
            col_idx: col_idx_t,
            values: values_t,
        }
    }
}

impl fmt::Debug for CsrMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CsrMatrix({}x{}, nnz={})",
            self.nrows,
            self.ncols,
            self.nnz()
        )
    }
}

/// COO triplet builder → CSR. This is the natural way to assemble FDM/FEM matrices.
pub struct TripletBuilder {
    nrows: usize,
    ncols: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
    vals: Vec<f64>,
}

impl TripletBuilder {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            rows: Vec::new(),
            cols: Vec::new(),
            vals: Vec::new(),
        }
    }

    pub fn with_capacity(nrows: usize, ncols: usize, capacity: usize) -> Self {
        Self {
            nrows,
            ncols,
            rows: Vec::with_capacity(capacity),
            cols: Vec::with_capacity(capacity),
            vals: Vec::with_capacity(capacity),
        }
    }

    /// Add a single entry. Duplicate (i,j) pairs are summed during build.
    #[inline]
    pub fn add(&mut self, row: usize, col: usize, val: f64) {
        self.rows.push(row);
        self.cols.push(col);
        self.vals.push(val);
    }

    /// Build CSR matrix. Sums duplicate entries.
    pub fn build(self) -> CsrMatrix {
        let nnz_raw = self.rows.len();
        if nnz_raw == 0 {
            return CsrMatrix {
                nrows: self.nrows,
                ncols: self.ncols,
                row_ptr: vec![0; self.nrows + 1],
                col_idx: Vec::new(),
                values: Vec::new(),
            };
        }

        // Sort by (row, col)
        let mut indices: Vec<usize> = (0..nnz_raw).collect();
        indices.sort_by(|&a, &b| {
            self.rows[a]
                .cmp(&self.rows[b])
                .then(self.cols[a].cmp(&self.cols[b]))
        });

        // Deduplicate and sum
        let mut row_ptr = vec![0usize; self.nrows + 1];
        let mut col_idx = Vec::with_capacity(nnz_raw);
        let mut values = Vec::with_capacity(nnz_raw);

        let mut prev_row = self.rows[indices[0]];
        let mut prev_col = self.cols[indices[0]];
        let mut acc = self.vals[indices[0]];

        for &idx in indices.iter().skip(1) {
            let r = self.rows[idx];
            let c = self.cols[idx];
            let v = self.vals[idx];
            if r == prev_row && c == prev_col {
                acc += v;
            } else {
                col_idx.push(prev_col);
                values.push(acc);
                row_ptr[prev_row + 1] += 1;
                prev_row = r;
                prev_col = c;
                acc = v;
            }
        }
        // Flush last
        col_idx.push(prev_col);
        values.push(acc);
        row_ptr[prev_row + 1] += 1;

        // Prefix sum
        for i in 1..=self.nrows {
            row_ptr[i] += row_ptr[i - 1];
        }

        CsrMatrix {
            nrows: self.nrows,
            ncols: self.ncols,
            row_ptr,
            col_idx,
            values,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spmv_identity() {
        // 3x3 identity
        let mat = CsrMatrix::new(
            3, 3,
            vec![0, 1, 2, 3],
            vec![0, 1, 2],
            vec![1.0, 1.0, 1.0],
        ).unwrap();
        let x = vec![3.0, 5.0, 7.0];
        let y = mat.spmv(&x);
        assert_eq!(y, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_triplet_builder() {
        let mut tb = TripletBuilder::new(3, 3);
        // Add entries with duplicates
        tb.add(0, 0, 2.0);
        tb.add(0, 1, -1.0);
        tb.add(1, 0, -1.0);
        tb.add(1, 1, 2.0);
        tb.add(1, 2, -1.0);
        tb.add(2, 1, -1.0);
        tb.add(2, 2, 2.0);
        // Duplicate: add more to (1,1)
        tb.add(1, 1, 0.5);

        let mat = tb.build();
        assert_eq!(mat.nrows, 3);
        assert_eq!(mat.nnz(), 7);

        // Check (1,1) = 2.0 + 0.5 = 2.5
        let diag = mat.diagonal();
        assert!((diag[1] - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_transpose() {
        let mut tb = TripletBuilder::new(2, 3);
        tb.add(0, 0, 1.0);
        tb.add(0, 2, 3.0);
        tb.add(1, 1, 2.0);
        let mat = tb.build();
        let t = mat.transpose();
        assert_eq!(t.nrows, 3);
        assert_eq!(t.ncols, 2);
        // t[2,0] should be 3.0
        let y = t.spmv(&[1.0, 0.0]);
        assert!((y[2] - 3.0).abs() < 1e-12);
    }
}
