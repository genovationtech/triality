# Triality Verification Suite

## Purpose

This directory contains production-grade verification tests. When someone asks:

> **"How do I know this is correct?"**

Point them here.

## Test Categories

### 1. Method of Manufactured Solutions (MMS)
**File:** `mms.py`

The gold standard for PDE solver verification. We manufacture an exact solution,
compute the required forcing term, solve numerically, and verify convergence rates.

**Tests:**
- 1D Polynomial: `u(x) = x(1-x)`
- 1D Sinusoidal: `u(x) = sin(πx)`
- 2D Polynomial: `u(x,y) = x(1-x)y(1-y)`
- 2D Sinusoidal: `u(x,y) = sin(πx)sin(πy)`

**Expected:** 2nd order convergence (`||error|| ~ h²`)

**Current Status:** ❌ FAILING
- **Issue:** Current API only supports constant forcing terms
- **Fix Required:** Implement spatially-varying forcing functions
- **Workaround:** Tests use averaged forcing (not true MMS)

### 2. Grid Convergence Analysis
**File:** `convergence.py`

Verifies that numerical solutions converge to exact solutions as grid spacing
decreases. Computes observed order of accuracy.

**Tests:**
- 1D Poisson equation convergence
- 2D Poisson equation convergence

**Expected:** Order ~2.0

**Current Status:** ❌ FAILING (see MMS issue above)

### 3. Conservation Checks
**File:** `conservation.py`

Verifies that discretization preserves physical conservation laws.

**Tests:**
- 1D Integral Identity: `∫ u'' dx = u'(b) - u'(a)`
- 2D Gauss's Theorem: `∮ ∂u/∂n ds = ∫∫ ∇²u dA`
- Maximum Principle: For `∇²u = f` with `f > 0`, max occurs at boundary
- Symmetry checks

**Current Status:** ❌ FAILING
- **Issue:** Sign convention mismatch in Laplacian operator
- **Fix Required:** Verify FDM discretization sign convention

### 4. Regression Benchmarks
**File:** `regression.py`

Maintains known solutions to prevent regressions. Solutions are stored as checksums.

**Tests:**
- 1D Poisson with constant forcing
- 2D Poisson with constant forcing
- (More to be added as features expand)

**Current Status:** ✅ Can generate benchmarks

## Running Tests

```bash
# Run full verification suite
cd triality/verification
PYTHONPATH=../..:$PYTHONPATH python3 test_verification.py

# Save new regression benchmarks
python3 test_verification.py --save-benchmarks

# Generate convergence plots
python3 test_verification.py --plots
```

## Test Results Interpretation

### MMS Pass Criteria
- L2 convergence rate within 0.15 of expected (2.0)
- Linf convergence rate within 0.15 of expected (2.0)

### Conservation Pass Criteria
- Relative error < 1e-8 for integral identities
- No maximum principle violations

### Regression Pass Criteria
- L2 norm matches within 1e-10 relative error
- Linf norm matches within 1e-10 relative error
- Solution checksum matches exactly

## Known Limitations

### Current Limitations That Prevent Full Verification

1. **Constant Forcing Only**
   - **Problem:** Can only solve with constant RHS: `Eq(laplacian(u), c)`
   - **Impact:** Cannot perform true MMS tests
   - **Fix:** Need to support: `Eq(laplacian(u), f)` where `f` is an expression
   - **Difficulty:** Medium - requires evaluating expression trees on grids

2. **Sign Convention Issue**
   - **Problem:** Possible sign mismatch in Laplacian discretization
   - **Impact:** Conservation tests fail, maximum principle violated
   - **Fix:** Audit FDM implementation in `geometry/fdm.py`
   - **Difficulty:** Easy - just verify signs in stencil

3. **No Variable BC**
   - **Problem:** Only constant Dirichlet BC supported
   - **Impact:** Cannot test non-homogeneous problems rigorously
   - **Fix:** Support spatially-varying BC: `bc={'left': 'sin(y)'}`
   - **Difficulty:** Medium

## Roadmap to Full Verification

### Phase 1: Fix Sign Convention (IMMEDIATE)
- [ ] Audit `geometry/fdm.py` Laplacian discretization
- [ ] Verify signs match mathematical convention
- [ ] Re-run conservation tests
- [ ] Expected: Conservation tests should pass

### Phase 2: Variable Forcing Terms (HIGH PRIORITY)
- [ ] Extend IR to support evaluated expressions
- [ ] Implement expression evaluation on grids
- [ ] Update `solve()` to handle non-constant forcing
- [ ] Re-run MMS tests
- [ ] Expected: MMS convergence ~2.0

### Phase 3: Complete Test Coverage (MEDIUM PRIORITY)
- [ ] Add time-dependent PDE tests (when implemented)
- [ ] Add nonlinear PDE tests (when implemented)
- [ ] Add mixed BC tests
- [ ] Add irregular domain tests

### Phase 4: Performance Benchmarks (LOW PRIORITY)
- [ ] Add timing benchmarks
- [ ] Add memory benchmarks
- [ ] Add scalability tests

## References

### Verification Methodology
- Oberkampf & Roy (2010). *Verification and Validation in Scientific Computing*
- Roache (1998). *Verification of Codes and Calculations*
- Roache (2002). *Code Verification by the Method of Manufactured Solutions*

### Numerical Methods
- LeVeque (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*
- LeVeque (2002). *Finite Volume Methods for Hyperbolic Problems*
- Toro (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*

### Standards
- AIAA Guide for Verification and Validation (G-077-1998)
- ASME V&V 20-2009 Standard

## Contributing New Verification Tests

When adding a new verification test:

1. **Choose the right category:**
   - MMS for convergence rate verification
   - Conservation for physical law preservation
   - Regression for preventing known-solution breakage

2. **Document expectations:**
   - What should pass/fail
   - Why it might fail
   - How to fix failures

3. **Set tolerances appropriately:**
   - MMS: ±0.15 for convergence rates (accounts for asymptotic range)
   - Conservation: ~1e-8 to 1e-10 (depends on problem)
   - Regression: 1e-10 (bit-for-bit reproducibility)

4. **Update this README** with new test descriptions

## Questions?

If verification tests fail:

1. **Check for known limitations above**
2. **Read error messages carefully**
3. **Check if it's a real bug or tolerance issue**
4. **Update IR_SPEC.md if assumptions changed**

---

**Remember:** Verification is not validation. Verification proves we solved the
equations correctly. Validation proves we solved the right equations.
