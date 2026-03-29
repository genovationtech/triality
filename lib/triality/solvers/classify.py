"""
Automatic PDE classification.

Analyzes equations to determine type, linearity, and properties.
This enables automatic solver selection.
"""

from dataclasses import dataclass
from triality.core.expressions import Equation, find_diff_ops, is_linear


@dataclass
class Classification:
    """Problem classification result"""
    pde_type: str  # 'elliptic', 'parabolic', 'hyperbolic', 'ode'
    is_linear: bool
    dimension: int
    has_laplacian: bool
    has_time: bool
    max_spatial_order: int
    time_order: int

    def __repr__(self):
        lines = [
            f"PDE Type: {self.pde_type}",
            f"Linear: {self.is_linear}",
            f"Dimension: {self.dimension}D",
        ]
        if self.has_time:
            lines.append(f"Time-dependent: order {self.time_order}")
        return "\n".join(lines)


def classify(equation: Equation, domain=None) -> Classification:
    """
    Classify a PDE automatically.

    Determines:
    - PDE type (elliptic/parabolic/hyperbolic)
    - Linearity
    - Spatial dimension
    - Derivative orders
    - Time dependence
    """

    # Get differential operators
    diff_ops = find_diff_ops(equation.lhs) + find_diff_ops(equation.rhs)

    # Check linearity
    linear = is_linear(equation)

    # Check for specific operators
    has_laplacian = any(op.op == 'laplacian' for op in diff_ops)
    has_time = any(op.op == 'dt' for op in diff_ops)

    # Determine spatial dimension
    if domain:
        dimension = domain.dim
    else:
        # Infer from operators
        spatial_ops = [op for op in diff_ops if op.op in ['dx', 'dy', 'dz', 'laplacian', 'grad']]
        if any(op.op == 'dz' for op in spatial_ops):
            dimension = 3
        elif any(op.op == 'dy' for op in spatial_ops):
            dimension = 2
        elif any(op.op == 'dx' for op in spatial_ops) or has_laplacian:
            dimension = 1
        else:
            dimension = 0

    # Get max derivative orders
    spatial_orders = [op.order for op in diff_ops if op.op in ['dx', 'dy', 'dz', 'laplacian']]
    max_spatial_order = max(spatial_orders) if spatial_orders else 0

    time_orders = [op.order for op in diff_ops if op.op == 'dt']
    time_order = max(time_orders) if time_orders else 0

    # Detect nested time derivatives: dt(dt(u)) should be time_order=2
    # Walk the expression tree to find the maximum nesting depth of dt ops
    from triality.core.expressions import DiffOp as _DiffOp
    def _max_dt_depth(expr, depth=0):
        if isinstance(expr, _DiffOp) and expr.op == 'dt':
            child_depth = _max_dt_depth(expr.operand, depth + expr.order)
            return child_depth
        max_child = depth
        for child in expr.children():
            max_child = max(max_child, _max_dt_depth(child, 0))
        return max(depth, max_child)

    nested_time_order = max(_max_dt_depth(equation.lhs), _max_dt_depth(equation.rhs))
    if nested_time_order > time_order:
        time_order = nested_time_order

    # Classify PDE type
    if not has_time:
        # Steady-state → elliptic
        pde_type = 'elliptic'
    elif dimension == 0 and not has_laplacian and max_spatial_order == 0:
        # No spatial derivatives → ODE
        pde_type = 'ode'
    elif time_order >= 2:
        # Second-order in time → hyperbolic (wave)
        pde_type = 'hyperbolic'
    elif time_order == 1:
        # First-order in time
        if max_spatial_order == 2 or has_laplacian:
            # ∂u/∂t = ∇²u → parabolic (diffusion)
            pde_type = 'parabolic'
        else:
            # ∂u/∂t + ... → could be hyperbolic or parabolic
            pde_type = 'parabolic'  # Default
    else:
        pde_type = 'unknown'

    return Classification(
        pde_type=pde_type,
        is_linear=linear,
        dimension=dimension,
        has_laplacian=has_laplacian,
        has_time=has_time,
        max_spatial_order=max_spatial_order,
        time_order=time_order
    )
