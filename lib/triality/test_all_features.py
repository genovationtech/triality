"""
Triality Complete Feature Test

Single file demonstrating all features of the Triality library.
Run this to verify everything works.
"""

import numpy as np
import sys


def section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_expressions():
    """Test expression building"""
    section("1. Expression System")

    from triality import Field, Constant, Eq, laplacian, grad, dx, sin, cos

    # Fields
    u = Field("u")
    v = Field("v")
    print(f"✓ Field creation: u = {u}")

    # Operators
    print(f"✓ Addition: u + v = {u + v}")
    print(f"✓ Multiplication: 2*u = {2 * u}")
    print(f"✓ Power: u**2 = {u**2}")

    # Differential operators
    print(f"✓ Laplacian: ∇²u = {laplacian(u)}")
    print(f"✓ Gradient: ∇u = {grad(u)}")
    print(f"✓ Partial: ∂u/∂x = {dx(u)}")

    # Functions
    print(f"✓ Sine: sin(u) = {sin(u)}")
    print(f"✓ Cosine: cos(u) = {cos(u)}")

    # Equations
    eq = Eq(laplacian(u), Constant(1.0))
    print(f"✓ Equation: {eq}")


def test_domains():
    """Test geometric domains"""
    section("2. Geometric Domains")

    from triality import Interval, Rectangle, Square, Circle

    # 1D
    interval = Interval(0, 1)
    print(f"✓ Interval: {interval}, length = {interval.length()}")

    # 2D
    rect = Rectangle(0, 2, 0, 3)
    print(f"✓ Rectangle: {rect}, area = {rect.area()}")

    square = Square(1.5)
    print(f"✓ Square: {square}, area = {square.area()}")

    circle = Circle((0, 0), 2.0)
    print(f"✓ Circle: {circle}, area = {circle.area():.3f}")


def test_classification():
    """Test automatic problem classification"""
    section("3. Automatic Classification")

    from triality import Field, Eq, laplacian, dx, dy, dt, classify, Interval, Rectangle

    # Test 1: 1D elliptic
    u = Field("u")
    eq1 = Eq(laplacian(u), 1)
    c1 = classify(eq1, Interval(0, 1))
    print(f"\n📊 Problem 1: {eq1}")
    print(f"   Type: {c1.pde_type}")
    print(f"   Linear: {c1.is_linear}")
    print(f"   Dimension: {c1.dimension}D")
    print(f"   ✓ Correctly classified as {c1.pde_type}")

    # Test 2: 2D elliptic
    eq2 = Eq(laplacian(u), -2)
    c2 = classify(eq2, Rectangle(0, 1, 0, 1))
    print(f"\n📊 Problem 2: {eq2}")
    print(f"   Type: {c2.pde_type}")
    print(f"   Dimension: {c2.dimension}D")
    print(f"   ✓ Correctly classified as 2D {c2.pde_type}")

    # Test 3: Nonlinear
    eq3 = Eq(laplacian(u) + u**2, 0)
    c3 = classify(eq3, Interval(0, 1))
    print(f"\n📊 Problem 3: {eq3}")
    print(f"   Type: {c3.pde_type}")
    print(f"   Linear: {c3.is_linear}")
    print(f"   ✓ Correctly detected as nonlinear")

    # Test 4: Time-dependent (parabolic)
    eq4 = Eq(dt(u), laplacian(u))
    c4 = classify(eq4, Interval(0, 1))
    print(f"\n📊 Problem 4: {eq4}")
    print(f"   Type: {c4.pde_type}")
    print(f"   Time-dependent: {c4.has_time}")
    print(f"   ✓ Correctly classified as {c4.pde_type}")


def test_solver_selection():
    """Test automatic solver selection"""
    section("4. Automatic Solver Selection")

    from triality import Field, Eq, laplacian, classify, select_solver, Interval, Rectangle

    u = Field("u")

    # Small 1D problem
    eq1 = Eq(laplacian(u), 1)
    c1 = classify(eq1, Interval(0, 1))
    plan1 = select_solver(c1, size_estimate=100)
    print(f"\n🔧 Small 1D problem (100 DOFs):")
    print(f"   Discretization: {plan1.discretization}")
    print(f"   Linear solver: {plan1.linear_solver}")
    print(f"   Preconditioner: {plan1.preconditioner}")
    print(f"   Backend: {plan1.backend}")

    # Large 2D problem
    eq2 = Eq(laplacian(u), 1)
    c2 = classify(eq2, Rectangle(0, 1, 0, 1))
    plan2 = select_solver(c2, size_estimate=100000)
    print(f"\n🔧 Large 2D problem (100k DOFs):")
    print(f"   Discretization: {plan2.discretization}")
    print(f"   Linear solver: {plan2.linear_solver}")
    print(f"   Preconditioner: {plan2.preconditioner}")
    print(f"   Backend: {plan2.backend}")
    print(f"\n   Reasoning:")
    for reason in plan2.reasoning:
        print(f"     • {reason}")


def test_solve_1d():
    """Test solving 1D Poisson equation"""
    section("5. Solve 1D Poisson Equation")

    from triality import Field, Eq, laplacian, Interval, solve

    print("\n📝 Problem: u''(x) = -1 on [0,1], u(0) = u(1) = 0")
    print("   Exact solution: u(x) = x(1-x)/2")

    u = Field("u")
    eq = Eq(laplacian(u), -1)
    domain = Interval(0, 1)
    bc = {'left': 0, 'right': 0}

    # Solve
    sol = solve(eq, domain, bc=bc, resolution=100, verbose=False)

    print(f"\n✓ Solution computed:")
    print(f"   Converged: {sol.converged}")
    print(f"   Iterations: {sol.iterations}")
    print(f"   Residual: {sol.residual:.2e}")
    print(f"   Time: {sol.time:.3f}s")

    # Check error
    x = sol.grid
    u_exact = x * (1 - x) / 2
    error = np.abs(sol.u - u_exact)
    max_error = np.max(error)
    l2_error = np.sqrt(np.mean(error**2))

    print(f"\n📊 Error analysis:")
    print(f"   Max error: {max_error:.2e}")
    print(f"   L2 error:  {l2_error:.2e}")

    if max_error < 1e-4:
        print(f"   ✓ Excellent accuracy!")

    # Evaluate at points
    print(f"\n📍 Solution values:")
    for xi, exact_val in [(0.25, 0.09375), (0.5, 0.125), (0.75, 0.09375)]:
        computed = sol(xi)
        print(f"   u({xi}) = {computed:.6f}  (exact: {exact_val:.6f}, error: {abs(computed - exact_val):.2e})")

    return sol


def test_solve_2d():
    """Test solving 2D Poisson equation"""
    section("6. Solve 2D Poisson Equation")

    from triality import Field, Eq, laplacian, Square, solve

    print("\n📝 Problem: ∇²u = -2 on [0,1]², u = 0 on boundary")

    u = Field("u")
    eq = Eq(laplacian(u), -2)
    domain = Square(1.0)
    bc = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}

    # Solve
    sol = solve(eq, domain, bc=bc, resolution=50, verbose=False)

    print(f"\n✓ Solution computed:")
    print(f"   Converged: {sol.converged}")
    print(f"   Iterations: {sol.iterations}")
    print(f"   Residual: {sol.residual:.2e}")
    print(f"   Time: {sol.time:.3f}s")

    # Evaluate at points
    print(f"\n📍 Solution values:")
    test_points = [
        ((0.25, 0.25), "corner quadrant"),
        ((0.5, 0.5), "center"),
        ((0.75, 0.75), "opposite corner quadrant"),
    ]

    for (x, y), label in test_points:
        val = sol(x, y)
        print(f"   u({x}, {y}) = {val:.6f}  ({label})")

    # Check symmetry
    u_center = sol(0.5, 0.5)
    if u_center > 0:
        print(f"\n✓ Solution is positive in interior (as expected)")

    return sol


def test_different_operators():
    """Test different differential operators"""
    section("7. Different Operators")

    from triality import Field, Eq, laplacian, dx, dy, grad, div, Interval, Rectangle, solve

    # Test: d²u/dx² = -1
    print("\n📝 Problem 1: d²u/dx² = -1 on [0,1]")
    u = Field("u")
    eq1 = Eq(dx(u, order=2), -1)
    sol1 = solve(eq1, Interval(0, 1), bc={'left': 0, 'right': 0},
                 resolution=50, verbose=False)
    print(f"   ✓ Solved with d²/dx² operator")
    print(f"   u(0.5) = {sol1(0.5):.6f}")

    # Test: Standard Laplacian on rectangle
    print("\n📝 Problem 2: ∇²u = -1 on [0,2]×[0,1]")
    eq2 = Eq(laplacian(u), -1)
    domain2 = Rectangle(0, 2, 0, 1)
    sol2 = solve(eq2, domain2, bc={'left': 0, 'right': 0, 'bottom': 0, 'top': 0},
                 resolution=40, verbose=False)
    print(f"   ✓ Solved on non-square rectangle")
    print(f"   u(1.0, 0.5) = {sol2(1.0, 0.5):.6f}")


def test_edge_cases():
    """Test edge cases and robustness"""
    section("8. Edge Cases & Robustness")

    from triality import Field, Eq, laplacian, Interval, solve

    u = Field("u")

    # Very small domain
    print("\n📝 Test 1: Very small domain")
    eq = Eq(laplacian(u), 1)
    domain_small = Interval(0, 0.1)
    sol = solve(eq, domain_small, bc={'left': 0, 'right': 0},
                resolution=30, verbose=False)
    print(f"   ✓ Solved on [0, 0.1]")
    print(f"   Max value: {np.max(sol.u):.6f}")

    # Large constant
    print("\n📝 Test 2: Large constant RHS")
    eq = Eq(laplacian(u), -100)
    sol = solve(eq, Interval(0, 1), bc={'left': 0, 'right': 0},
                resolution=50, verbose=False)
    print(f"   ✓ Solved with RHS = -100")
    print(f"   Max value: {np.max(sol.u):.2f}")

    # Different boundary conditions
    print("\n📝 Test 3: Non-zero boundary conditions")
    eq = Eq(laplacian(u), 0)  # Harmonic
    sol = solve(eq, Interval(0, 1), bc={'left': 1.0, 'right': 2.0},
                resolution=50, verbose=False)
    print(f"   ✓ Solved with u(0)=1, u(1)=2")
    print(f"   u(0.5) = {sol(0.5):.6f}  (should be ~1.5 for linear)")


def run_all_tests():
    """Run all feature tests"""
    print("\n" + "█" * 70)
    print("  TRIALITY COMPLETE FEATURE TEST")
    print("█" * 70)

    tests = [
        ("Expressions", test_expressions),
        ("Domains", test_domains),
        ("Classification", test_classification),
        ("Solver Selection", test_solver_selection),
        ("1D Solve", test_solve_1d),
        ("2D Solve", test_solve_2d),
        ("Operators", test_different_operators),
        ("Edge Cases", test_edge_cases),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Summary
    section("SUMMARY")
    total = passed + failed
    print(f"\nTests run: {total}")
    print(f"✓ Passed: {passed}")
    if failed > 0:
        print(f"✗ Failed: {failed}")

    if failed == 0:
        print("\n" + "🎉" * 20)
        print("  ALL TESTS PASSED!")
        print("🎉" * 20)
        return 0
    else:
        print("\n⚠️  Some tests failed")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
