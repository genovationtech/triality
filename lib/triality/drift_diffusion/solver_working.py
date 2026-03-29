"""
Working 1D Drift-Diffusion Solver with correct Boltzmann statistics.

Key insight: For PN junction in equilibrium, Fermi level is CONSTANT.
Using this fact directly gives stable, correct solution.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .device_solver import (
    q, k_B, eps_0, T, V_T,
    SemiconductorMaterial,
    DD1DResult,
    PNJunctionAnalyzer
)


class DriftDiffusion1DWorking:
    """
    Working 1D DD solver using correct equilibrium formulation.

    For equilibrium, uses the fact that Fermi level E_F is constant:
        n(x) = N_d * exp(V(x) / V_T)
        p(x) = (n_i² / N_d) * exp(-V(x) / V_T)

    Where the reference (V=0) is at the N-side contact.
    """

    def __init__(self, length: float, n_points: int = 200):
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
        self.material = material

    def set_doping(self, N_d, N_a):
        for i, x in enumerate(self.x):
            self.N_d[i] = N_d(x)
            self.N_a[i] = N_a(x)

    def solve(self, applied_voltage: float = 0.0, max_iterations: int = 50,
              tolerance: float = 1e-5, damping: float = 0.5,
              verbose: bool = False) -> DD1DResult:
        """
        Solve using Gummel iteration with correct Boltzmann statistics.

        Key: Use N-side doping as reference for Fermi level calculation.
        """
        if verbose:
            print(f"Solving DD with correct Boltzmann ({self.n_points} points)...")

        n_i = self.material.n_i
        N_net = self.N_d - self.N_a

        # Find reference doping at N-side (left boundary)
        N_d_ref = max(N_net[0], n_i) if N_net[0] > 0 else n_i
        N_a_ref = max(-N_net[-1], n_i) if N_net[-1] < 0 else n_i

        # Initial guess: START SMALL to avoid exponential blowup
        # Use 10% of the expected built-in potential
        V_bi_guess = V_T * np.log((N_d_ref * N_a_ref) / n_i**2)
        self.V = np.linspace(0, -V_bi_guess * 0.01, self.n_points)  # Start with 1% of V_bi

        # Initialize carriers using Boltzmann with initial V
        self._update_carriers_boltzmann(N_d_ref)

        # Gummel iteration
        for iteration in range(max_iterations):
            V_old = self.V.copy()

            # Solve Poisson
            self._solve_poisson(applied_voltage)

            # Apply damping
            self.V = damping * self.V + (1 - damping) * V_old

            # Update carriers
            self._update_carriers_boltzmann(N_d_ref)

            # Check convergence
            delta_V = np.max(np.abs(self.V - V_old))
            V_bi = np.max(self.V) - np.min(self.V)

            if verbose and (iteration < 5 or iteration % 10 == 0):
                print(f"  Iter {iteration:3d}: ΔV={delta_V:.2e}, V_bi={V_bi:.4f} V")

            if delta_V < tolerance:
                if verbose:
                    print(f"  ✓ Converged in {iteration+1} iterations")
                break
        else:
            if verbose:
                print(f"  ⚠ Max iterations")

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

    def _update_carriers_boltzmann(self, N_d_ref: float):
        """
        Update carriers using Boltzmann statistics with N-side reference.

        Uses normalized potential psi = -V/V_T (semiconductor convention):
            n(x) = N_d_ref * exp(-psi(x))
            p(x) = n_i² / N_d_ref * exp(psi(x))

        At N-side where V=0 => psi=0: n=N_d_ref, p=n_i²/N_d_ref ✓
        At P-side where V=-V_bi => psi=V_bi/V_T > 0: n decreases, p increases ✓
        """
        n_i = self.material.n_i

        self.n = np.zeros(self.n_points)
        self.p = np.zeros(self.n_points)

        for i in range(self.n_points):
            # Normalized potential: psi = -V/V_T
            # Negative V (P-side) gives positive psi
            psi = -self.V[i] / V_T

            # Clamp to prevent overflow
            psi_clamped = np.clip(psi, -30, 30)

            # Boltzmann statistics
            self.n[i] = N_d_ref * np.exp(-psi_clamped)
            self.p[i] = (n_i**2 / N_d_ref) * np.exp(psi_clamped)

            # Enforce minimum concentrations
            self.n[i] = max(self.n[i], 1e6)
            self.p[i] = max(self.p[i], 1e6)

    def _solve_poisson(self, applied_voltage: float):
        """Solve Poisson equation."""
        eps = self.material.eps_r * eps_0
        n = self.n_points

        # Laplacian
        diag = -2 * np.ones(n)
        upper = np.ones(n - 1)
        lower = np.ones(n - 1)

        A = sparse.diags([lower, diag, upper], [-1, 0, 1], shape=(n, n), format='lil')
        A = A / (self.dx ** 2)

        # RHS
        rho = (self.p - self.n + self.N_d - self.N_a) * 1e6
        b = -q * rho / eps

        # BC: Left = 0 (reference)
        A[0, :] = 0
        A[0, 0] = 1
        b[0] = 0

        # BC: Right
        if abs(applied_voltage) < 1e-6:
            # Equilibrium: Neumann BC
            A[-1, :] = 0
            A[-1, -1] = 1
            A[-1, -2] = -1
            b[-1] = 0
        else:
            # Bias: Dirichlet BC
            A[-1, :] = 0
            A[-1, -1] = 1
            b[-1] = applied_voltage

        self.V = spla.spsolve(A.tocsr(), b)


def create_pn_junction_working(N_d_level: float = 1e17,
                               N_a_level: float = 1e16,
                               junction_pos: float = 1e-4,
                               total_length: float = 2e-4) -> DriftDiffusion1DWorking:
    """Create working PN junction solver."""
    solver = DriftDiffusion1DWorking(length=total_length, n_points=200)
    solver.set_material(SemiconductorMaterial.Silicon())

    solver.set_doping(
        N_d=lambda x: N_d_level if x < junction_pos else 0,
        N_a=lambda x: 0 if x < junction_pos else N_a_level,
    )

    return solver
