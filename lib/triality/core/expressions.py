"""
Expression system for building PDE equations.

All expressions are immutable and lazy - they describe operations without executing them.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set
from enum import Enum
import hashlib


# ============================================================================
# Expression Nodes
# ============================================================================

class NodeType(Enum):
    """Expression node types"""
    FIELD = "field"
    CONSTANT = "constant"
    BINARY_OP = "binary_op"
    UNARY_OP = "unary_op"
    DIFF_OP = "diff_op"
    EQUATION = "equation"


class Expr:
    """Base expression class with operator overloading"""

    def __add__(self, other): return BinaryOp('+', self, _wrap(other))
    def __radd__(self, other): return BinaryOp('+', _wrap(other), self)
    def __sub__(self, other): return BinaryOp('-', self, _wrap(other))
    def __rsub__(self, other): return BinaryOp('-', _wrap(other), self)
    def __mul__(self, other): return BinaryOp('*', self, _wrap(other))
    def __rmul__(self, other): return BinaryOp('*', _wrap(other), self)
    def __truediv__(self, other): return BinaryOp('/', self, _wrap(other))
    def __rtruediv__(self, other): return BinaryOp('/', _wrap(other), self)
    def __pow__(self, other): return BinaryOp('**', self, _wrap(other))
    def __neg__(self): return UnaryOp('-', self)

    def children(self) -> List[Expr]:
        """Return child expressions"""
        return []

    def deps(self) -> Set[str]:
        """Return field dependencies"""
        result = set()
        for child in self.children():
            result.update(child.deps())
        return result


@dataclass(frozen=True)
class Field(Expr):
    """Field variable (e.g., u, v, T)"""
    name: str

    def __repr__(self): return self.name
    def deps(self): return {self.name}


@dataclass(frozen=True)
class Constant(Expr):
    """Constant value"""
    value: float

    def __repr__(self): return str(self.value)


@dataclass(frozen=True)
class BinaryOp(Expr):
    """Binary operation (e.g., u + v, u * v)"""
    op: str
    left: Expr
    right: Expr

    def children(self): return [self.left, self.right]
    def __repr__(self): return f"({self.left} {self.op} {self.right})"


@dataclass(frozen=True)
class UnaryOp(Expr):
    """Unary operation (e.g., -u, sin(u))"""
    op: str
    operand: Expr

    def children(self): return [self.operand]
    def __repr__(self):
        if self.op == '-': return f"(-{self.operand})"
        return f"{self.op}({self.operand})"


@dataclass(frozen=True)
class DiffOp(Expr):
    """Differential operator (e.g., ∇u, ∂u/∂x)"""
    op: str  # 'laplacian', 'grad', 'div', 'dx', 'dy', 'dt'
    operand: Expr
    order: int = 1

    def children(self): return [self.operand]
    def __repr__(self):
        if self.op == 'laplacian': return f"∇²{self.operand}"
        if self.op == 'grad': return f"∇{self.operand}"
        if self.op == 'div': return f"∇·{self.operand}"
        if self.op in ['dx', 'dy', 'dz', 'dt']:
            var = self.op[1:]
            if self.order == 1:
                return f"∂{self.operand}/∂{var}"
            return f"∂^{self.order}{self.operand}/∂{var}^{self.order}"
        return f"{self.op}({self.operand})"


class Equation:
    """Equation (e.g., ∇²u = f)"""
    def __init__(self, lhs: Expr, rhs: Expr):
        self.lhs = lhs
        self.rhs = rhs

    def __repr__(self): return f"{self.lhs} = {self.rhs}"


def Eq(lhs: Expr, rhs) -> Equation:
    """Create an equation: Eq(lhs, rhs) creates 'lhs = rhs'"""
    return Equation(lhs, _wrap(rhs))


def _wrap(value) -> Expr:
    """Convert Python values to expressions"""
    if isinstance(value, Expr): return value
    if isinstance(value, (int, float)): return Constant(float(value))
    raise TypeError(f"Cannot convert {type(value)} to expression")


# ============================================================================
# Math Functions
# ============================================================================

def sin(x: Expr) -> UnaryOp: return UnaryOp('sin', x)
def cos(x: Expr) -> UnaryOp: return UnaryOp('cos', x)
def exp(x: Expr) -> UnaryOp: return UnaryOp('exp', x)
def sqrt(x: Expr) -> UnaryOp: return UnaryOp('sqrt', x)


# ============================================================================
# Differential Operators
# ============================================================================

def laplacian(u: Expr) -> DiffOp:
    """Laplacian: ∇²u"""
    return DiffOp('laplacian', u, order=2)

def grad(u: Expr) -> DiffOp:
    """Gradient: ∇u"""
    return DiffOp('grad', u)

def div(u: Expr) -> DiffOp:
    """Divergence: ∇·u"""
    return DiffOp('div', u)

def dx(u: Expr, order: int = 1) -> DiffOp:
    """Partial derivative: ∂u/∂x"""
    return DiffOp('dx', u, order=order)

def dy(u: Expr, order: int = 1) -> DiffOp:
    """Partial derivative: ∂u/∂y"""
    return DiffOp('dy', u, order=order)

def dz(u: Expr, order: int = 1) -> DiffOp:
    """Partial derivative: ∂u/∂z"""
    return DiffOp('dz', u, order=order)

def dt(u: Expr, order: int = 1) -> DiffOp:
    """Time derivative: ∂u/∂t"""
    return DiffOp('dt', u, order=order)


# ============================================================================
# Analysis
# ============================================================================

def is_linear(eq: Equation) -> bool:
    """Check if equation is linear in its fields"""
    fields = eq.lhs.deps() | eq.rhs.deps()

    def check_node(node: Expr) -> bool:
        if isinstance(node, UnaryOp):
            if node.op in ['sin', 'cos', 'exp', 'sqrt']:
                if node.operand.deps() & fields:
                    return False
        elif isinstance(node, BinaryOp):
            left_deps = node.left.deps() & fields
            right_deps = node.right.deps() & fields
            if node.op == '*' and left_deps and right_deps:
                return False
            if node.op == '**':
                if left_deps and isinstance(node.right, Constant):
                    if node.right.value != 1:
                        return False
                elif left_deps:
                    return False
            if node.op == '/' and right_deps:
                return False

        for child in node.children():
            if not check_node(child):
                return False
        return True

    return check_node(eq.lhs) and check_node(eq.rhs)


def find_diff_ops(expr: Expr) -> List[DiffOp]:
    """Find all differential operators in expression"""
    result = []

    def traverse(node):
        if isinstance(node, DiffOp):
            result.append(node)
        for child in node.children():
            traverse(child)

    traverse(expr)
    return result
