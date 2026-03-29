"""Tests for Triality"""

import pytest
import numpy as np
from triality import *


def test_expressions():
    """Test expression building"""
    u = Field("u")
    v = Field("v")

    # Operators
    assert str(u + v) == "(u + v)"
    assert str(u * 2) == "(u * 2.0)"
    assert "∇²" in str(laplacian(u))

    # Equations
    eq = Eq(laplacian(u), 1)
    assert isinstance(eq, Equation)


def test_domains():
    """Test geometric domains"""
    interval = Interval(0, 1)
    assert interval.length() == 1

    rect = Rectangle(0, 1, 0, 2)
    assert rect.area() == 2

    square = Square(1.0)
    assert square.area() == 1


def test_classification():
    """Test automatic classification"""
    u = Field("u")
    eq = Eq(laplacian(u), 1)
    domain = Interval(0, 1)

    c = classify(eq, domain)
    assert c.pde_type == 'elliptic'
    assert c.is_linear == True
    assert c.dimension == 1


def test_solver_selection():
    """Test automatic solver selection"""
    u = Field("u")
    eq = Eq(laplacian(u), 1)

    c = classify(eq, Interval(0, 1))
    plan = select_solver(c, size_estimate=100)

    assert plan.linear_solver == 'direct'  # Small problem


def test_solve_1d():
    """Test 1D Poisson equation"""
    u = Field("u")
    eq = Eq(laplacian(u), -1)
    domain = Interval(0, 1)
    bc = {'left': 0, 'right': 0}

    sol = solve(eq, domain, bc=bc, resolution=50, verbose=False)

    assert sol.converged
    assert sol.residual < 1e-6

    # Check against exact solution
    x = sol.grid
    u_exact = x * (1 - x) / 2
    error = np.max(np.abs(sol.u - u_exact))
    assert error < 1e-4  # Good accuracy


def test_solve_2d():
    """Test 2D Poisson equation"""
    u = Field("u")
    eq = Eq(laplacian(u), -2)
    domain = Square(1.0)
    bc = {'left': 0, 'right': 0, 'bottom': 0, 'top': 0}

    sol = solve(eq, domain, bc=bc, resolution=30, verbose=False)

    assert sol.converged
    assert sol.residual < 1e-6

    # Solution should be positive in interior
    u_center = sol(0.5, 0.5)
    assert u_center > 0


def test_solution_evaluation():
    """Test solution evaluation at points"""
    u = Field("u")
    eq = Eq(laplacian(u), -1)
    domain = Interval(0, 1)
    bc = {'left': 0, 'right': 0}

    sol = solve(eq, domain, bc=bc, resolution=50, verbose=False)

    # Evaluate at multiple points
    values = [sol(x) for x in [0.25, 0.5, 0.75]]
    assert all(v > 0 for v in values)  # Solution positive in interior


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
