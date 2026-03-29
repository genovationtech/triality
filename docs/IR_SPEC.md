# Triality Intermediate Representation (IR) Specification

**Version:** 1.0
**Status:** Frozen
**Purpose:** Formal specification of the Triality expression IR

This document defines the formal IR for Triality. It is intentionally restrictive to maintain coherence and verifiability.

---

## 1. Design Philosophy

The Triality IR is:
- **Immutable**: All expression nodes are immutable once created
- **Strongly typed**: Every expression has a well-defined mathematical type
- **Verifiable**: Every transformation preserves mathematical correctness
- **Minimal**: Only operations that can be discretized and solved are allowed

Think LLVM IR, not NumPy API.

---

## 2. Expression Node Types

### 2.1 Core Nodes

All expressions inherit from `Expr`. The IR consists of exactly 5 node types:

```
Expr := Field(name: str)
      | Constant(value: float)
      | BinaryOp(op: BinOp, left: Expr, right: Expr)
      | UnaryOp(op: UnOp, operand: Expr)
      | DiffOp(op: DiffType, operand: Expr, var: str, order: int)
```

### 2.2 Node Semantics

**Field(name)**
- Represents an unknown function to be solved for
- Must be unique within an equation system
- Invariant: `name` must be non-empty string
- Example: `Field("u")` represents u(x), u(x,y), or u(x,y,t) depending on domain

**Constant(value)**
- Represents a real scalar constant
- Invariant: `value` must be finite float
- Example: `Constant(3.14)` represents π

**BinaryOp(op, left, right)**
- Binary arithmetic operations
- Allowed ops: `+`, `-`, `*`, `/`, `**`
- Invariant: Both operands must be valid `Expr` nodes
- Semantics:
  - `+`: Pointwise addition
  - `-`: Pointwise subtraction
  - `*`: Pointwise multiplication (scalar or field)
  - `/`: Pointwise division
  - `**`: Pointwise power (constant exponent only)

**UnaryOp(op, operand)**
- Unary operations
- Allowed ops: `-`, `sin`, `cos`, `exp`, `log`, `sqrt`
- Invariant: Operand must be valid `Expr`
- Semantics: Pointwise application of mathematical function

**DiffOp(op, operand, var, order)**
- Differential operators
- Allowed ops: `grad`, `div`, `laplacian`, `∂/∂x`, `∂/∂y`, `∂/∂t`
- Invariant:
  - `operand` must be a `Field`
  - `var` must be in {'x', 'y', 't'}
  - `order` must be positive integer
- Semantics:
  - `grad`: ∇u = (∂u/∂x, ∂u/∂y, ...)
  - `div`: ∇·u (requires vector field)
  - `laplacian`: ∇²u = ∂²u/∂x² + ∂²u/∂y² + ...
  - `∂/∂x`: Partial derivative with respect to x

---

## 3. Equation Type

```
Equation := (lhs: Expr, rhs: Expr)
```

**Semantics:**
- Represents `lhs = rhs`
- Both sides must be valid `Expr` nodes
- No implicit conversions

**Invariants:**
1. Exactly one `Field` appears in the equation (the unknown)
2. The highest differential order is well-defined
3. The equation is spatial (contains spatial derivatives) or temporal (contains ∂/∂t)

---

## 4. Operator Semantics (Mathematical)

### 4.1 Binary Operations

| Op | Math | Domain | Commutativity |
|----|------|--------|---------------|
| `+` | u + v | ℝⁿ → ℝⁿ | Yes |
| `-` | u - v | ℝⁿ → ℝⁿ | No |
| `*` | u × v | ℝⁿ → ℝⁿ | Yes |
| `/` | u / v | ℝⁿ → ℝⁿ (v ≠ 0) | No |
| `**` | uⁿ | ℝⁿ → ℝⁿ (n ∈ ℝ) | No |

### 4.2 Unary Operations

| Op | Math | Domain | Range |
|----|------|--------|-------|
| `-` | -u | ℝ | ℝ |
| `sin` | sin(u) | ℝ | [-1, 1] |
| `cos` | cos(u) | ℝ | [-1, 1] |
| `exp` | eᵘ | ℝ | (0, ∞) |
| `log` | ln(u) | (0, ∞) | ℝ |
| `sqrt` | √u | [0, ∞) | [0, ∞) |

### 4.3 Differential Operations

| Op | Math | Order | Tensor Rank |
|----|------|-------|-------------|
| `∂/∂x` | ∂u/∂x | 1 | Scalar → Scalar |
| `grad` | ∇u | 1 | Scalar → Vector |
| `div` | ∇·v | 1 | Vector → Scalar |
| `laplacian` | ∇²u | 2 | Scalar → Scalar |

**Composition Rules:**
- `laplacian(u)` ≡ `div(grad(u))`
- `∂²u/∂x²` ≡ `∂/∂x(∂u/∂x)`
- Commutative: `∂²u/∂x∂y = ∂²u/∂y∂x`

---

## 5. Type System

Every expression has a **mathematical type**:

```
Type := Scalar(dimension: int, order: int)
      | Vector(dimension: int, order: int)
      | Tensor(dimension: int, order: int, rank: int)
```

**Attributes:**
- `dimension`: Spatial dimension (1, 2, or 3)
- `order`: Maximum differential order (0 for algebraic, 1+ for differential)
- `rank`: Tensor rank (0=scalar, 1=vector, 2=matrix, ...)

**Type Rules:**
1. `Field(name)` → `Scalar(dim=inferred, order=0)`
2. `Constant(v)` → `Scalar(dim=0, order=0)`
3. `BinaryOp('+', s1, s2)` → `Scalar(max(s1.dim, s2.dim), max(s1.order, s2.order))`
4. `DiffOp('laplacian', s)` → `Scalar(s.dim, s.order+2)`
5. `DiffOp('grad', s)` → `Vector(s.dim, s.order+1)`

---

## 6. Allowed vs Forbidden

### ✅ ALLOWED

**Linear operations:**
```python
u + v
2 * u
-u
u / 2
```

**Differential operations:**
```python
laplacian(u)
grad(u)
dx(u)
dy(u)
```

**Simple nonlinear (single field):**
```python
u ** 2
sin(u)
exp(u)
```

**Well-formed equations:**
```python
Eq(laplacian(u), 1)           # ∇²u = 1
Eq(laplacian(u), -sin(u))     # ∇²u = -sin(u)
Eq(dx(dx(u)), u)              # u'' = u
```

### ❌ FORBIDDEN

**Undefined operations:**
```python
u == v                  # Use Eq(u, v), not ==
laplacian(3)           # Can't take derivative of constant
div(u)                 # div requires vector field, u is scalar
```

**Type errors:**
```python
u + grad(u)            # Scalar + Vector = Type Error
laplacian(grad(u))     # Returns vector, invalid
```

**Multiple unknowns (currently):**
```python
Eq(u + v, 1)           # Two unknowns not supported yet
```

**Non-differentiable:**
```python
abs(u)                 # Not differentiable at u=0
max(u, v)              # Not differentiable everywhere
```

**Implicit operations:**
```python
u * grad(u)            # Advection term - not yet supported
grad(u) · grad(u)      # Inner products - not yet supported
```

---

## 7. Invariants (MUST HOLD)

### Expression Invariants

1. **Immutability**: Once created, expression nodes never change
2. **Acyclicity**: Expression trees must be acyclic (no self-reference)
3. **Finite depth**: Maximum tree depth is bounded (prevents stack overflow)
4. **Well-typed**: Every subexpression has a valid type

### Equation Invariants

1. **Single unknown**: Exactly one `Field` appears in the equation
2. **Consistent dimensions**: All terms have compatible dimensions
3. **Bounded order**: Differential order ≤ 2 (current limitation)
4. **Solvable**: The equation can be discretized into a linear system

### Solver Invariants

1. **Conservation**: Discretization preserves conservation laws
2. **Consistency**: Truncation error → 0 as grid spacing → 0
3. **Stability**: Solution remains bounded as grid spacing → 0
4. **Convergence**: Numerical solution → exact solution as resolution → ∞

---

## 8. Extension Rules

When adding new features, you MUST:

1. **Update this spec first** - No code without spec
2. **Prove type safety** - Show new operations preserve type system
3. **Verify solvability** - Demonstrate discretization method
4. **Add verification tests** - MMS, convergence, conservation
5. **Document forbidden cases** - What breaks and why

**Example: Adding advection term `u·∇u`**

Before implementation:
- [ ] Define `DotProduct(vector1, vector2)` node type
- [ ] Prove: DotProduct(Vector, Vector) → Scalar
- [ ] Show upwind discretization method
- [ ] Add MMS test with known solution
- [ ] Update forbidden list (remove if now allowed)

---

## 9. Reference Implementation

The canonical implementation is in `/home/user/general_agents/triality/core/expressions.py`.

**Compliance:**
- Implementation MUST match this spec
- Spec takes precedence over implementation
- Deviations are bugs

---

## 10. Verification Requirements

Every IR transformation MUST be verified by:

1. **Unit tests**: Type checking, invariant preservation
2. **Integration tests**: End-to-end equation solving
3. **Convergence tests**: Order-of-accuracy verification
4. **Conservation tests**: Physical quantities preserved
5. **Regression tests**: Known solutions match exactly

See `triality/verification/` for test suite.

---

## Version History

- **v1.0** (2026-01-27): Initial frozen specification
  - 5 node types: Field, Constant, BinaryOp, UnaryOp, DiffOp
  - Linear elliptic PDEs only
  - 1D and 2D domains
  - Dirichlet boundary conditions

---

## References

- LLVM IR Specification: https://llvm.org/docs/LangRef.html
- FEniCS Form Language: https://fenics.readthedocs.io/
- Firedrake UFL: https://www.firedrakeproject.org/

---

**End of Specification**

This document is frozen. Changes require formal review and version increment.
