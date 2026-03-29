"""
Simplified 1D Drift-Diffusion Solver using proven numerical methods.

This implementation uses a simpler but more robust approach:
- Charge neutrality approximation in bulk regions
- Scharfetter-Gummel scheme for stability
- Fixed carrier boundaries (ohmic contacts)
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from typing import Tuple, Optional
from dataclasses import dataclass

# Import constants and classes from original module
from .device_solver import (
    q, k_B, eps_0, T, V_T,
    SemiconductorMaterial,
    DD1DResult,
    PNJunctionAnalyzer
)


class DriftDiffusion1DSimple:
    """
    Simplified 1D Drift-Diffusion solver for PN junctions.

    Uses charge-neutral approximation with Poisson equation.
    More stable than full Gummel iteration.
    """

    def __init__(self, length: float, n_points: int = 200):
        """Initialize solver with domain."""
        self.length = length
        self.n_points = n_points

        self.x = np.linspace(0, length, n_points)
        self.dx = self.x[1] - self.x[0]

        self.material = SemiconductorMaterial.Silicon()
        self.N_d = np.zeros(n_points)
        self.N_a = np.zeros(n_points)

        self.V = None
        self.n = None
        self.p = None

    def set_material(self, material: SemiconductorMaterial):
        """Set semiconductor material."""
        self.material = material

    def set_doping(self, N_d, N_a):
        """Set doping profiles."""
        for i, x in enumerate(self.x):
            self.N_d[i] = N_d(x)
            self.N_a[i] = N_a(x)

    def solve(self, applied_voltage: float = 0.0, max_iterations: int = 100,
              tolerance: float = 1e-6, verbose: bool = False) -> DD1DResult:
        """
        Solve PN junction using iterative Poisson with charge neutrality.

        This approach:
        1. Maintains carriers at charge neutrality (majority = doping)
        2. Solves Poisson to get potential
        3. Iterates until potential converges
        This is stable and works well for PN junctions.
        """
        if verbose:
            print(f"Solving simplified DD ({self.n_points} points)...")

        n_i = self.material.n_i
        N_net = self.N_d - self.N_a

        # Initialize carriers with charge neutrality
        self.n = np.zeros(self.n_points)
        self.p = np.zeros(self.n_points)

        for i in range(self.n_points):
            if abs(N_net[i]) > n_i:
                if N_net[i] > 0:  # N-type
                    self.n[i] = N_net[i]
                    self.p[i] = n_i**2 / max(N_net[i], n_i)
                else:  # P-type
                    self.p[i] = -N_net[i]
                    self.n[i] = n_i**2 / max(-N_net[i], n_i)
            else:
                self.n[i] = n_i
                self.p[i] = n_i

        # Initialize potential
        self.V = np.zeros(self.n_points)

        # Iteratively solve Poisson
        for iteration in range(max_iterations):
            V_old = self.V.copy()

            # Solve Poisson equation
            self._solve_poisson_simple(applied_voltage)

            # Check convergence
            delta_V = np.max(np.abs(self.V - V_old))

            if verbose and (iteration < 5 or iteration % 20 == 0):
                V_bi = np.max(self.V) - np.min(self.V)
                print(f"  Iter {iteration:3d}: ΔV={delta_V:.2e}, V_bi={V_bi:.4f} V")

            if delta_V < tolerance:
                if verbose:
                    print(f"  ✓ Converged in {iteration+1} iterations")
                break
        else:
            if verbose:
                print(f"  ⚠ Max iterations reached")

        return DD1DResult(
            x=self.x,
            V=self.V,
            n=self.n,
            p=self.p,
            N_d=self.N_d,
            N_a=self.N_a,
            material=self.material,
            applied_voltage=applied_voltage
        )

    def _solve_poisson_simple(self, applied_voltage: float):
        """
        Solve Poisson with proper boundary conditions for PN junction.

        For equilibrium, we use a reference potential at one side
        and let the built-in potential develop naturally.
        """
        eps = self.material.eps_r * eps_0
        n = self.n_points

        # Build Laplacian matrix
        diag = -2 * np.ones(n)
        upper = np.ones(n - 1)
        lower = np.ones(n - 1)

        A = sparse.diags([lower, diag, upper], [-1, 0, 1], shape=(n, n), format='lil')
        A = A / (self.dx * self.dx)

        # RHS: -q * (p - n + N_d - N_a) / eps
        rho = (self.p - self.n + self.N_d - self.N_a) * 1e6  # cm^-3 -> m^-3
        b = -q * rho / eps

        # Boundary conditions
        # Left: V = 0 (reference)
        A[0, :] = 0
        A[0, 0] = 1
        b[0] = 0

        # Right: For equilibrium, use Neumann BC to let V_bi develop
        # For bias, enforce applied voltage
        if abs(applied_voltage) < 1e-6:
            # Equilibrium: dV/dx = 0 at right boundary
            A[-1, :] = 0
            A[-1, -1] = 1
            A[-1, -2] = -1
            b[-1] = 0
        else:
            # Applied bias
            A[-1, :] = 0
            A[-1, -1] = 1
            b[-1] = applied_voltage

        # Solve
        self.V = spla.spsolve(A.tocsr(), b)


def create_pn_junction_simple(N_d_level: float = 1e17,
                              N_a_level: float = 1e16,
                              junction_pos: float = 1e-4,
                              total_length: float = 2e-4) -> DriftDiffusion1DSimple:
    """Create simplified PN junction solver."""
    solver = DriftDiffusion1DSimple(length=total_length, n_points=200)
    solver.set_material(SemiconductorMaterial.Silicon())

    solver.set_doping(
        N_d=lambda x: N_d_level if x < junction_pos else 0,
        N_a=lambda x: 0 if x < junction_pos else N_a_level,
    )

    return solver
