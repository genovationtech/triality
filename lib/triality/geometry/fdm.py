"""Finite Difference Method discretization"""

import numpy as np
from scipy import sparse


def discretize_1d(equation, domain, bc, resolution, forcing=None):
    """
    Discretize 1D PDE using finite differences.

    Args:
        forcing: Optional forcing term override. Can be:
            - None: use equation.rhs (default)
            - callable: function f(x) evaluated at each grid point
            - ndarray: forcing values at grid points
            - scalar: constant forcing

    Returns: (A, b, grid) where Ax = b
    """
    nx = resolution
    x = np.linspace(domain.a, domain.b, nx)
    dx = (domain.b - domain.a) / (nx - 1)

    # Build Laplacian stencil: [1, -2, 1] / dx^2
    A = sparse.lil_matrix((nx, nx))
    b = np.zeros(nx)

    for i in range(1, nx-1):
        A[i, i-1] = 1.0 / dx**2
        A[i, i] = -2.0 / dx**2
        A[i, i+1] = 1.0 / dx**2

        # RHS - support multiple input types
        if forcing is not None:
            if callable(forcing):
                b[i] = forcing(x[i])
            elif isinstance(forcing, np.ndarray):
                b[i] = forcing[i]
            else:
                b[i] = float(forcing)  # Constant
        else:
            # Evaluate from equation
            b[i] = _eval_rhs(equation.rhs, x[i])

    # Apply boundary conditions
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    b[0] = bc.get('left', 0.0)
    b[-1] = bc.get('right', 0.0)

    return A.tocsr(), b, x


def discretize_2d(equation, domain, bc, resolution, forcing=None):
    """
    Discretize 2D PDE using finite differences.

    Args:
        forcing: Optional forcing term override. Can be:
            - None: use equation.rhs (default)
            - callable: function f(x,y) evaluated at each grid point
            - ndarray: forcing values at grid points (flattened or 2D)
            - scalar: constant forcing

    Returns: (A, b, grid) where Ax = b
    """
    nx = ny = resolution

    # Handle different domain types
    if hasattr(domain, 'xmin'):  # Rectangle
        x = np.linspace(domain.xmin, domain.xmax, nx)
        y = np.linspace(domain.ymin, domain.ymax, ny)
        dx = (domain.xmax - domain.xmin) / (nx - 1)
        dy = (domain.ymax - domain.ymin) / (ny - 1)
    elif hasattr(domain, 'L'):  # Square
        x = np.linspace(0, domain.L, nx)
        y = np.linspace(0, domain.L, ny)
        dx = dy = domain.L / (nx - 1)
    else:
        raise ValueError(f"Unsupported domain type: {type(domain)}")

    ndof = nx * ny

    # Helper: 2D index to 1D
    def idx(i, j):
        return i * ny + j

    # Build 5-point Laplacian stencil
    A = sparse.lil_matrix((ndof, ndof))
    b = np.zeros(ndof)

    for i in range(1, nx-1):
        for j in range(1, ny-1):
            k = idx(i, j)

            # Laplacian stencil
            A[k, k] = -2.0/dx**2 - 2.0/dy**2
            A[k, idx(i-1, j)] = 1.0 / dx**2
            A[k, idx(i+1, j)] = 1.0 / dx**2
            A[k, idx(i, j-1)] = 1.0 / dy**2
            A[k, idx(i, j+1)] = 1.0 / dy**2

            # RHS - support multiple input types
            if forcing is not None:
                if callable(forcing):
                    b[k] = forcing(x[i], y[j])
                elif isinstance(forcing, np.ndarray):
                    if forcing.ndim == 1:
                        b[k] = forcing[k]
                    else:
                        b[k] = forcing[i, j]
                else:
                    b[k] = float(forcing)  # Constant
            else:
                # Evaluate from equation
                b[k] = _eval_rhs(equation.rhs, x[i], y[j])

    # Apply boundary conditions (Dirichlet)
    bc_val_left = bc.get('left', 0.0)
    bc_val_right = bc.get('right', 0.0)
    bc_val_bottom = bc.get('bottom', 0.0)
    bc_val_top = bc.get('top', 0.0)

    for i in range(nx):
        A[idx(i, 0), idx(i, 0)] = 1.0
        A[idx(i, ny-1), idx(i, ny-1)] = 1.0
        b[idx(i, 0)] = bc_val_bottom
        b[idx(i, ny-1)] = bc_val_top

    for j in range(ny):
        A[idx(0, j), idx(0, j)] = 1.0
        A[idx(nx-1, j), idx(nx-1, j)] = 1.0
        b[idx(0, j)] = bc_val_left
        b[idx(nx-1, j)] = bc_val_right

    return A.tocsr(), b, (x, y)


def _eval_rhs(rhs_expr, *coords):
    """Evaluate RHS expression at point(s)"""
    from triality.core.expressions import (Constant, Field, BinaryOp, UnaryOp,
                                          DiffOp)
    import numpy as np

    # Constant - return the value
    if isinstance(rhs_expr, Constant):
        return rhs_expr.value

    # Field - not allowed in RHS (RHS shouldn't have unknowns)
    if isinstance(rhs_expr, Field):
        raise ValueError(f"RHS contains unknown field '{rhs_expr.name}' - not supported")

    # Binary operations
    if isinstance(rhs_expr, BinaryOp):
        left = _eval_rhs(rhs_expr.left, *coords)
        right = _eval_rhs(rhs_expr.right, *coords)

        if rhs_expr.op == '+':
            return left + right
        elif rhs_expr.op == '-':
            return left - right
        elif rhs_expr.op == '*':
            return left * right
        elif rhs_expr.op == '/':
            return left / right
        elif rhs_expr.op == '**':
            return left ** right
        else:
            raise ValueError(f"Unknown binary operator: {rhs_expr.op}")

    # Unary operations
    if isinstance(rhs_expr, UnaryOp):
        operand = _eval_rhs(rhs_expr.operand, *coords)

        if rhs_expr.op == '-':
            return -operand
        elif rhs_expr.op == 'sin':
            return np.sin(operand)
        elif rhs_expr.op == 'cos':
            return np.cos(operand)
        elif rhs_expr.op == 'exp':
            return np.exp(operand)
        elif rhs_expr.op == 'log':
            return np.log(operand)
        elif rhs_expr.op == 'sqrt':
            return np.sqrt(operand)
        else:
            raise ValueError(f"Unknown unary operator: {rhs_expr.op}")

    # Differential operators in RHS - not supported yet
    if isinstance(rhs_expr, DiffOp):
        raise ValueError(f"Differential operators in RHS not supported yet")

    # Unknown expression type
    raise ValueError(f"Cannot evaluate expression type: {type(rhs_expr)}")
