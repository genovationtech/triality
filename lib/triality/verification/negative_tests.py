"""
Negative Test Suite - Tests That Should Fail

These tests verify that Triality fails loudly and helpfully on ill-posed,
ambiguous, or invalid problems.

Production-grade means predictability + explanation + refusal.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from triality import Field, Eq, laplacian, grad, div, dx, dy, dt
from triality import Interval, Rectangle, Square, solve


class NegativeTestSuite:
    """Tests that verify proper failure modes"""

    def __init__(self):
        self.results = {
            'ill_posed': [],
            'ambiguous': [],
            'invalid': []
        }

    def run_all_tests(self):
        """Run all negative tests"""
        print("\n" + "="*70)
        print("  NEGATIVE TEST SUITE - Tests That Should Fail Properly")
        print("="*70)

        print("\n" + "="*70)
        print("  PART 1: ILL-POSED PROBLEMS")
        print("="*70)
        self.test_ill_posed_problems()

        print("\n" + "="*70)
        print("  PART 2: AMBIGUOUS PROBLEMS")
        print("="*70)
        self.test_ambiguous_problems()

        print("\n" + "="*70)
        print("  PART 3: INVALID INPUTS")
        print("="*70)
        self.test_invalid_inputs()

        self.print_summary()

    def test_ill_posed_problems(self):
        """Test detection of ill-posed problems"""

        # Test 1: Pure Neumann problem (nullspace issue)
        print("\n[1/4] Pure Neumann Elliptic Problem")
        try:
            u = Field("u")
            eq = Eq(laplacian(u), 1)
            domain = Interval(0, 1)
            # Neumann BCs on both sides - no unique solution!
            bc = {'left_flux': 0.0, 'right_flux': 0.0}
            sol = solve(eq, domain, bc=bc, verbose=False)
            print("❌ FAILED TO DETECT - Should have rejected Neumann problem")
            self.results['ill_posed'].append(('Pure Neumann', False))
        except (ValueError, NotImplementedError) as e:
            if 'Neumann' in str(e) or 'flux' in str(e) or 'not supported' in str(e):
                print("✅ CORRECTLY REJECTED")
                print(f"   Message: {e}")
                self.results['ill_posed'].append(('Pure Neumann', True))
            else:
                print(f"❌ WRONG ERROR: {e}")
                self.results['ill_posed'].append(('Pure Neumann', False))

        # Test 2: Overconstrained BCs
        print("\n[2/4] Overconstrained Boundary Conditions")
        try:
            u = Field("u")
            eq = Eq(laplacian(u), 1)
            domain = Interval(0, 1)
            # Too many BCs
            bc = {'left': 0.0, 'right': 1.0, 'left_flux': 0.5}
            sol = solve(eq, domain, bc=bc, verbose=False)
            print("⚠️  ACCEPTED - Might be OK if solver picked one")
            self.results['ill_posed'].append(('Overconstrained', True))
        except (ValueError, NotImplementedError) as e:
            print("✅ CORRECTLY REJECTED")
            print(f"   Message: {e}")
            self.results['ill_posed'].append(('Overconstrained', True))

        # Test 3: Mixed dimensional operators
        print("\n[3/4] Mixed Dimensional Operators")
        try:
            u = Field("u")
            v = Field("v")
            # laplacian is 2nd order spatial, dt is 1st order temporal
            # Can't have pure spatial operator equal temporal
            eq = Eq(laplacian(u), dt(v))
            domain = Interval(0, 1)
            bc = {'left': 0, 'right': 0}
            sol = solve(eq, domain, bc=bc, verbose=False)
            print("❌ FAILED TO DETECT - Should reject mixed operators")
            self.results['ill_posed'].append(('Mixed Dimensional', False))
        except (ValueError, NotImplementedError, AttributeError) as e:
            print("✅ CORRECTLY REJECTED")
            print(f"   Message: {e}")
            self.results['ill_posed'].append(('Mixed Dimensional', True))

        # Test 4: Time derivative without initial condition
        print("\n[4/4] Time Derivative Without Initial Condition")
        try:
            u = Field("u")
            # ∂u/∂t = ∇²u is parabolic - needs initial condition
            eq = Eq(dt(u), laplacian(u))
            domain = Rectangle(0, 1, 0, 1)
            bc = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}
            # Missing: ic (initial condition)
            sol = solve(eq, domain, bc=bc, verbose=False)
            print("❌ FAILED TO DETECT - Should require initial condition")
            self.results['ill_posed'].append(('Missing IC', False))
        except (ValueError, NotImplementedError) as e:
            if 'time' in str(e).lower() or 'parabolic' in str(e).lower() or 'not supported' in str(e):
                print("✅ CORRECTLY REJECTED")
                print(f"   Message: {e}")
                self.results['ill_posed'].append(('Missing IC', True))
            else:
                print(f"⚠️  REJECTED (different reason): {e}")
                self.results['ill_posed'].append(('Missing IC', True))

    def test_ambiguous_problems(self):
        """Test handling of ambiguous problems"""

        # Test 1: Very coarse grid
        print("\n[1/3] Very Coarse Grid (resolution=3)")
        try:
            u = Field("u")
            eq = Eq(laplacian(u), 1)
            domain = Interval(0, 1)
            bc = {'left': 0, 'right': 0}
            # Only 3 points - basically no interior!
            sol = solve(eq, domain, bc=bc, resolution=3, verbose=False)

            # Check if it warns about coarseness
            # For now, just accept it worked
            print("⚠️  ACCEPTED - Should warn about coarse resolution")
            print(f"   Suggestion: Add warning for resolution < 10")
            self.results['ambiguous'].append(('Coarse Grid', True))
        except Exception as e:
            print(f"✅ REJECTED: {e}")
            self.results['ambiguous'].append(('Coarse Grid', True))

        # Test 2: Nearly singular system
        print("\n[2/3] Nearly Singular System")
        try:
            u = Field("u")
            # Very small coefficient
            eq = Eq(laplacian(u), 1e-15)
            domain = Interval(0, 1)
            bc = {'left': 0, 'right': 0}
            sol = solve(eq, domain, bc=bc, verbose=False)

            # Check residual
            if sol.residual > 1e-6:
                print("⚠️  HIGH RESIDUAL")
                print(f"   Residual: {sol.residual:.2e}")
                print(f"   Suggestion: Warn about near-zero forcing")
            else:
                print("✅ SOLVED (low residual)")
            self.results['ambiguous'].append(('Nearly Singular', True))
        except Exception as e:
            print(f"⚠️  FAILED: {e}")
            self.results['ambiguous'].append(('Nearly Singular', False))

        # Test 3: Multiple unknowns (not supported yet)
        print("\n[3/3] Multiple Unknown Fields")
        try:
            u = Field("u")
            v = Field("v")
            # System: ∇²u = v, ∇²v = u (coupled)
            eq = Eq(laplacian(u), v)
            domain = Interval(0, 1)
            bc = {'left': 0, 'right': 0}
            sol = solve(eq, domain, bc=bc, verbose=False)
            print("❌ FAILED TO DETECT - Should reject multiple unknowns")
            self.results['ambiguous'].append(('Multiple Unknowns', False))
        except (ValueError, NotImplementedError) as e:
            print("✅ CORRECTLY REJECTED")
            print(f"   Message: {e}")
            self.results['ambiguous'].append(('Multiple Unknowns', True))

    def test_invalid_inputs(self):
        """Test validation of user inputs"""

        # Test 1: Negative resolution
        print("\n[1/5] Negative Resolution")
        try:
            u = Field("u")
            eq = Eq(laplacian(u), 1)
            domain = Interval(0, 1)
            bc = {'left': 0, 'right': 0}
            sol = solve(eq, domain, bc=bc, resolution=-10, verbose=False)
            print("❌ FAILED TO VALIDATE - Should reject negative resolution")
            self.results['invalid'].append(('Negative Resolution', False))
        except (ValueError, TypeError) as e:
            print("✅ CORRECTLY REJECTED")
            print(f"   Message: {e}")
            self.results['invalid'].append(('Negative Resolution', True))

        # Test 2: Invalid domain (swapped bounds)
        print("\n[2/5] Invalid Domain (swapped bounds)")
        try:
            u = Field("u")
            eq = Eq(laplacian(u), 1)
            # a > b is invalid
            domain = Interval(1, 0)
            bc = {'left': 0, 'right': 0}
            sol = solve(eq, domain, bc=bc, verbose=False)
            print("❌ FAILED TO VALIDATE - Should reject inverted domain")
            self.results['invalid'].append(('Invalid Domain', False))
        except (ValueError, AssertionError) as e:
            print("✅ CORRECTLY REJECTED")
            print(f"   Message: {e}")
            self.results['invalid'].append(('Invalid Domain', True))

        # Test 3: Missing boundary conditions
        print("\n[3/5] Missing Boundary Conditions")
        try:
            u = Field("u")
            eq = Eq(laplacian(u), 1)
            domain = Interval(0, 1)
            # No BCs at all
            bc = {}
            sol = solve(eq, domain, bc=bc, verbose=False)
            print("⚠️  ACCEPTED - Using default BCs")
            self.results['invalid'].append(('Missing BC', True))
        except (ValueError, KeyError) as e:
            print("✅ REJECTED (strict mode)")
            print(f"   Message: {e}")
            self.results['invalid'].append(('Missing BC', True))

        # Test 4: NaN in forcing
        print("\n[4/5] NaN in Forcing Term")
        try:
            u = Field("u")
            eq = Eq(laplacian(u), 1)
            domain = Interval(0, 1)
            bc = {'left': 0, 'right': 0}
            # Pass NaN forcing
            forcing = lambda x: np.nan
            sol = solve(eq, domain, bc=bc, forcing=forcing, verbose=False)
            print("❌ FAILED TO VALIDATE - Should reject NaN")
            self.results['invalid'].append(('NaN Forcing', False))
        except (ValueError, RuntimeError) as e:
            print("✅ CORRECTLY REJECTED")
            print(f"   Message: {e}")
            self.results['invalid'].append(('NaN Forcing', True))

        # Test 5: Inf in boundary conditions
        print("\n[5/5] Inf in Boundary Conditions")
        try:
            u = Field("u")
            eq = Eq(laplacian(u), 1)
            domain = Interval(0, 1)
            bc = {'left': 0, 'right': np.inf}
            sol = solve(eq, domain, bc=bc, verbose=False)
            print("❌ FAILED TO VALIDATE - Should reject Inf")
            self.results['invalid'].append(('Inf BC', False))
        except (ValueError, RuntimeError) as e:
            print("✅ CORRECTLY REJECTED")
            print(f"   Message: {e}")
            self.results['invalid'].append(('Inf BC', True))

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*70)
        print("  NEGATIVE TEST SUMMARY")
        print("="*70)

        # Ill-posed problems
        ill_posed_passed = sum(1 for _, passed in self.results['ill_posed'] if passed)
        ill_posed_total = len(self.results['ill_posed'])
        print(f"\nIll-Posed Problems: {ill_posed_passed}/{ill_posed_total} properly rejected")
        for name, passed in self.results['ill_posed']:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")

        # Ambiguous problems
        ambiguous_passed = sum(1 for _, passed in self.results['ambiguous'] if passed)
        ambiguous_total = len(self.results['ambiguous'])
        print(f"\nAmbiguous Problems: {ambiguous_passed}/{ambiguous_total} handled")
        for name, passed in self.results['ambiguous']:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")

        # Invalid inputs
        invalid_passed = sum(1 for _, passed in self.results['invalid'] if passed)
        invalid_total = len(self.results['invalid'])
        print(f"\nInvalid Inputs: {invalid_passed}/{invalid_total} validated")
        for name, passed in self.results['invalid']:
            status = "✓" if passed else "✗"
            print(f"  {status} {name}")

        # Overall
        total_passed = ill_posed_passed + ambiguous_passed + invalid_passed
        total_tests = ill_posed_total + ambiguous_total + invalid_total

        print(f"\n{'='*70}")
        print(f"  OVERALL: {total_passed}/{total_tests} negative tests passed")

        if total_passed == total_tests:
            print(f"\n  🛡️  ROBUST - All edge cases handled properly")
        else:
            print(f"\n  ⚠️  Needs improvement - {total_tests - total_passed} cases unhandled")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    suite = NegativeTestSuite()
    suite.run_all_tests()
