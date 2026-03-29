# Physics Guide

This document explains the physics models used in Triality, their mathematical foundations, and their accuracy limitations.

---

## Layer 1: Elliptic PDEs

### Poisson / Laplace Equation

**Governing equation:**
```
∇²u = f(x)
```

For electrostatics: `f = -ρ/ε₀` (charge density)
For conduction: steady-state heat with source term `f = -Q/k`

**Discretization:** Second-order central finite differences on uniform Cartesian grids.

**Solvers selected by problem size:**
| Grid size | Solver | Notes |
|---|---|---|
| < 10,000 DOF | Direct (LU) | Exact to machine precision |
| 10k – 500k DOF | Conjugate Gradient + ILU | ε ≈ 1e-8 tolerance |
| > 500k DOF | GMRES + AMG preconditioner | ε ≈ 1e-6 tolerance |

**Accuracy:**
- Uniform grid, smooth solution: O(h²) convergence
- Typical error vs. FEM: ±2–5% for simple geometries
- Known limitation: staircase approximation for curved boundaries

### Heat Equation (Parabolic)

```
ρ·cp · ∂T/∂t = ∇·(k·∇T) + Q
```

Time integration options:
- **Explicit (Forward Euler):** Simple, stable for Fo = αΔt/Δx² < 0.5
- **Implicit (Backward Euler):** Unconditionally stable, first-order in time
- **Crank-Nicolson:** Second-order in time, default for production use

### Post-Solve: Observable Layer

After any PDE solve, the Observable Layer derives engineering quantities from the solved fields. For elliptic PDEs:
- **Electrostatics**: peak field strength, breakdown margin (distance to 3 MV/m air breakdown), field uniformity, stored energy
- **Heat conduction**: peak temperature, thermal margins to survival limits

Observable computation is algebraic post-processing on the solved field — no additional PDEs are discretized. Overhead: < 0.15% of solver time.

---

## Layer 2: Physics-Aware Routing

### Cost Field Construction

The routing cost at each grid cell is derived from physics fields:

```
c(x,y) = w_field · f(|E(x,y)|) + w_thermal · g(T(x,y)) + w_geometry · h(x,y)
```

Default field transform `f`:
```
f(|E|) = log(1 + |E| / E_ref)
```

This log transform gives moderate cost in moderate fields and very high cost near conductors, which produces routes that respect EMI constraints without being overly conservative.

### A* Routing

The router uses A* search on the cost grid with:
- **Heuristic:** Euclidean distance to goal (admissible)
- **Neighbor connectivity:** 8-connected (orthogonal + diagonal)
- **Smoothing:** Post-processing with cubic spline smoothing

Path cost is the sum of per-cell costs along the path, weighted by cell-center-to-cell-center distance.

### EMI Reduction Mechanism

By routing signal traces through low-field regions:
- Reduces mutual coupling capacitance C_m ∝ ε∫E·E' dV
- Reduces inductive coupling L_m ∝ μ∫B·B' dV
- Measured reduction: **30–70%** vs. shortest-path routing in typical PCB layouts

---

## Electrostatics Module

**Governing equations:**
```
∇·D = ρ_free        (Gauss's law)
D = ε₀·εᵣ·E         (constitutive relation)
E = -∇φ             (from Faraday's law, static)
→ ∇·(ε∇φ) = -ρ      (Poisson equation for electrostatics)
```

**Boundary conditions:**
- Dirichlet (voltage specified): φ = V₀
- Neumann (charge flux): ∂φ/∂n = σ/ε
- Mixed (Robin): aφ + b·∂φ/∂n = c

**Derived quantities:**
- Electric field: E = -∇φ
- Energy density: u = ε|E|²/2
- Force density: f = ρE + (ε₂-ε₁)|E|²∇εᵣ/2

**Assumptions and limitations:**
- Electrostatic (∂B/∂t = 0): valid for frequencies f << c/L (L = geometry size)
- Linear dielectric (εᵣ constant): use nonlinear module for ferroelectrics
- No free charge transport: use conduction module for resistive materials

---

## Thermal Hydraulics

**Governing equations:**
```
Continuity:    ∂ρ/∂t + ∇·(ρu) = 0
Momentum:      ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + ρg
Energy:        ρcp(∂T/∂t + u·∇T) = k∇²T + Q + Φ
```

where Φ = 2μ(Sᵢⱼ Sᵢⱼ) is viscous dissipation.

**Turbulence model:** k-ε with wall functions for Re > 5000.

**Heat transfer correlations:**
- Forced convection (internal): Dittus-Boelter (Nu = 0.023 Re⁰·⁸ Pr^n)
- Natural convection: Churchill-Chu correlation
- Nucleate boiling: Chen correlation

---

## Structural Analysis

**Linear elasticity:**
```
∇·σ + f = ρ·ü
σ = C : ε
ε = (∇u + ∇uᵀ)/2
```

**Failure criteria:**
- Von Mises: σ_vm = √(3J₂) < σ_yield
- Maximum principal stress
- Fatigue: Goodman diagram for alternating/mean stress

---

## Drift-Diffusion Semiconductors (Layer 3)

**System of equations:**
```
Poisson:             ∇·(ε∇φ) = q(n - p - N_D + N_A)
Electron continuity: ∂n/∂t + ∇·Jₙ/q = G - R
Hole continuity:     ∂p/∂t - ∇·Jₚ/q = G - R

Electron current:    Jₙ = qμₙnE + qDₙ∇n
Hole current:        Jₚ = qμₚpE - qDₚ∇p
```

**Recombination models:**
- Shockley-Read-Hall (SRH): for indirect-gap semiconductors (Si, Ge)
- Radiative: B·np for direct-gap (GaAs, InP)
- Auger: Cₙn²p + Cₚnp² for high injection

**Mobility models:**
- Constant (low field)
- Caughey-Thomas (velocity saturation)
- Lombardi (surface scattering for MOSFETs)

**Numerical method:** Scharfetter-Gummel discretization for current terms (avoids numerical oscillations in high-field regions).

**Status:** Framework complete; Gummel iteration convergence being refined for high-voltage devices.

---

## Nuclear / Radiation Modules

### Neutron Transport

**Simplified model (diffusion theory):**
```
-∇·D∇φ + Σₐφ = νΣf φ/k
```

Valid when scattering is isotropic and mean free path << system size. Use full transport (Sₙ method) for small systems or strong absorbers.

### Monte Carlo Particle Transport

Uses analog Monte Carlo for photon/neutron transport:
1. Sample source particle from source distribution
2. Track particle through geometry
3. Sample interaction type at each collision
4. Tally quantities of interest (dose, flux, heating)

**Variance reduction:** Russian roulette and splitting for importance regions.

**Accuracy:** Statistical — error scales as 1/√N. Typical: 1% accuracy with N=10⁶ particles.

---

## Accuracy Summary

| Module | Method | Accuracy vs. Reference |
|---|---|---|
| Electrostatics | FD Poisson | ±2–5% (simple), ±10–15% (complex) |
| Thermal conduction | FD heat eq. | ±3–8% |
| Routing cost fields | Physics-to-cost mapping | ±20–30% vs. full EM |
| Structural (linear) | FEM | ±2–5% |
| Thermal-hydraulics | 1D/2D model | ±10–20% vs. CFD |
| Neutronics (diffusion) | Diffusion theory | ±5–15% for thermal reactors |
| Monte Carlo | Analog MC | 1/√N statistical |

These accuracies are appropriate for **early-stage design screening**. For final design validation, export results to high-fidelity tools (ANSYS, COMSOL, OpenMC, OpenFOAM).
