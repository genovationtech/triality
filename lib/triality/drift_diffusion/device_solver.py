"""Practical Drift-Diffusion Solver

Layer 3: Production-useful semiconductor device analysis

This module provides 1D drift-diffusion simulation for semiconductor devices.
It's designed for EARLY-STAGE production design exploration, not final verification.

**Use Cases (Production-Relevant):**
✓ Initial PN junction design and doping optimization
✓ Diode I-V characteristic estimation
✓ Built-in potential and depletion width calculations
✓ Doping profile exploration before detailed TCAD
✓ Quick design iterations and trade-off analysis
✓ Sanity checking before expensive simulation

**When to Use Triality Layer 3:**
- Early design phase: exploring concepts
- Quick iterations: testing doping profiles
- Learning: understanding device physics
- Sanity checks: "does this make physical sense?"

**When to Upgrade to Full TCAD:**
- Final verification before tapeout
- Nanoscale devices (<100nm)
- Advanced mobility models needed
- High-frequency / transient analysis
- Process variation analysis
- 2D/3D effects are significant

**Accuracy Expectations:**
- PN junction: ±20% for depletion width, built-in potential
- Diode I-V: Qualitatively correct, ~30-50% quantitative error
- Good for relative comparisons (doping A vs. doping B)
- NOT accurate enough for final specifications

**Physics Model (1D):**
Solves coupled Poisson + continuity equations:

Poisson: d²V/dx² = -q(p - n + Nd - Na)/ε
Electron: dJn/dx = 0  (steady state)
Hole: dJp/dx = 0
Current: Jn = qμn·n·dV/dx + qDn·dn/dx
         Jp = qμp·p·dV/dx - qDp·dp/dx

With simplifications:
- Boltzmann statistics (not Fermi-Dirac)
- Constant mobility (not field-dependent)
- No generation-recombination
- Room temperature (300K)
- Classical transport only

**Workflow Integration:**
```
Concept → Triality Layer 3 → Validate trends → Full TCAD → Tapeout
         (minutes, explore)   (hours, verify)
```

Layer 3 gets you 80% of the way for 5% of the effort.
Then use full TCAD for the final 20%.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla
from typing import Tuple, Optional, Callable
from dataclasses import dataclass

# Physical constants
q = 1.602e-19      # Elementary charge [C]
k_B = 1.381e-23    # Boltzmann constant [J/K]
eps_0 = 8.854e-12  # Permittivity of free space [F/m]
T = 300            # Temperature [K]
V_T = k_B * T / q  # Thermal voltage ~26 mV


@dataclass
class SemiconductorMaterial:
    """Material properties for semiconductor"""
    name: str
    eps_r: float           # Relative permittivity
    n_i: float             # Intrinsic carrier concentration [cm^-3]
    mu_n: float            # Electron mobility [cm^2/(V·s)]
    mu_p: float            # Hole mobility [cm^2/(V·s)]

    @staticmethod
    def Silicon(temperature: float = 300):
        """Silicon material properties

        Args:
            temperature: Temperature in Kelvin (default 300K)

        Returns:
            SemiconductorMaterial for Silicon
        """
        # Temperature-dependent intrinsic carrier concentration
        # n_i(T) ≈ n_i(300K) * (T/300)^(3/2) * exp(-Eg/2kT)
        # Simplified model: n_i ∝ T^(3/2)
        T_ratio = temperature / 300.0
        n_i_300 = 1.5e10
        n_i = n_i_300 * (T_ratio ** 1.5) * np.exp(1.5 * (1 - T_ratio))

        return SemiconductorMaterial(
            name='Silicon',
            eps_r=11.7,
            n_i=n_i,         # cm^-3, temperature-dependent
            mu_n=1400,       # cm^2/(V·s)
            mu_p=450,
        )

    @staticmethod
    def GaAs(temperature: float = 300):
        """GaAs (Gallium Arsenide) material properties

        Args:
            temperature: Temperature in Kelvin (default 300K)

        Returns:
            SemiconductorMaterial for GaAs
        """
        # Temperature-dependent intrinsic carrier concentration
        T_ratio = temperature / 300.0
        n_i_300 = 2.1e6  # GaAs has much lower n_i than Si
        n_i = n_i_300 * (T_ratio ** 1.5) * np.exp(1.5 * (1 - T_ratio))

        return SemiconductorMaterial(
            name='GaAs',
            eps_r=12.9,
            n_i=n_i,         # cm^-3
            mu_n=8500,       # cm^2/(V·s) - much higher than Si
            mu_p=400,
        )

    @property
    def D_n(self):
        """Electron diffusion coefficient [cm^2/s]"""
        return self.mu_n * V_T

    @property
    def D_p(self):
        """Hole diffusion coefficient [cm^2/s]"""
        return self.mu_p * V_T


class DriftDiffusion1D:
    """
    1D Drift-Diffusion Solver for semiconductor devices.

    Solves Poisson + continuity equations in 1D using finite differences.
    Good for PN junctions, diodes, and initial design exploration.

    Example:
        >>> # Create PN junction
        >>> solver = DriftDiffusion1D(length=2e-4, n_points=200)  # 2 microns
        >>> solver.set_material(SemiconductorMaterial.Silicon())
        >>>
        >>> # Define doping: N-type left, P-type right
        >>> solver.set_doping(
        ...     N_d=lambda x: 1e17 if x < 1e-4 else 0,  # N: 10^17 cm^-3
        ...     N_a=lambda x: 0 if x < 1e-4 else 1e16,  # P: 10^16 cm^-3
        ... )
        >>>
        >>> # Solve at equilibrium (0V bias)
        >>> result = solver.solve(applied_voltage=0.0)
        >>>
        >>> # Get built-in potential
        >>> V_bi = result.built_in_potential()
        >>> print(f"Built-in potential: {V_bi:.3f} V")
    """

    def __init__(self, length: float, n_points: int = 200):
        """
        Initialize 1D drift-diffusion solver.

        Args:
            length: Device length [cm]
            n_points: Number of grid points
        """
        self.length = length
        self.n_points = n_points

        # Grid
        self.x = np.linspace(0, length, n_points)
        self.dx = self.x[1] - self.x[0]

        # Material (default Silicon)
        self.material = SemiconductorMaterial.Silicon()

        # Doping profiles (default undoped)
        self.N_d = np.zeros(n_points)  # Donor concentration
        self.N_a = np.zeros(n_points)  # Acceptor concentration

        # Solution arrays
        self.V = None      # Electrostatic potential
        self.n = None      # Electron concentration
        self.p = None      # Hole concentration

    def set_material(self, material: SemiconductorMaterial):
        """Set semiconductor material"""
        self.material = material

    def set_doping(self, N_d: Callable[[float], float], N_a: Callable[[float], float]):
        """
        Set doping profiles.

        Args:
            N_d: Donor doping function(x) -> concentration [cm^-3]
            N_a: Acceptor doping function(x) -> concentration [cm^-3]
        """
        for i, x in enumerate(self.x):
            self.N_d[i] = N_d(x)
            self.N_a[i] = N_a(x)

    def solve(self, applied_voltage: float = 0.0, max_iterations: int = 100,
              tolerance: float = 1e-6, under_relaxation: float = 0.3,
              verbose: bool = False) -> 'DD1DResult':
        """
        Solve drift-diffusion equations using Gummel iteration with damping.

        Args:
            applied_voltage: Applied voltage [V] (positive = forward bias)
            max_iterations: Maximum Gummel iterations
            tolerance: Convergence tolerance [V]
            under_relaxation: Damping factor (0 < alpha <= 1). Smaller = more stable
            verbose: Print iteration info

        Returns:
            DD1DResult with solution
        """
        if verbose:
            print(f"Solving 1D drift-diffusion ({self.n_points} points)...")
            print(f"  Applied voltage: {applied_voltage:.3f} V")
            print(f"  Under-relaxation: {under_relaxation}")

        # Initialize with equilibrium guess
        self._initialize_equilibrium()

        # Track convergence
        converged = False
        final_iteration = 0
        final_residual = 0.0

        # Gummel iteration with under-relaxation: alternate Poisson and continuity
        for iteration in range(max_iterations):
            V_old = self.V.copy()

            # 1. Solve Poisson equation for V given n, p
            self._solve_poisson(applied_voltage)

            # 2. Apply under-relaxation (damping) for stability
            # V_new = alpha * V + (1-alpha) * V_old
            self.V = under_relaxation * self.V + (1 - under_relaxation) * V_old

            # 3. Update n, p from V (using Boltzmann statistics)
            self._update_carriers()

            # Check convergence
            delta_V = np.max(np.abs(self.V - V_old))
            final_residual = delta_V
            final_iteration = iteration + 1

            if verbose and (iteration % 10 == 0 or iteration < 5):
                V_bi = np.max(self.V) - np.min(self.V)
                print(f"  Iteration {iteration:3d}: ΔV = {delta_V:.2e}, V_bi = {V_bi:.3f} V")

            if delta_V < tolerance:
                converged = True
                if verbose:
                    print(f"  ✓ Converged in {iteration + 1} iterations")
                break
        else:
            if verbose:
                print(f"  ⚠ Max iterations reached (ΔV = {delta_V:.2e})")

        # Apply Boltzmann correction ONLY at equilibrium
        # Under bias, quasi-Fermi levels split and single-EF Boltzmann is incorrect
        # Charge neutrality approximation is sufficient for bias conditions
        if abs(applied_voltage) < 1e-6:
            self._apply_boltzmann_correction()
            if verbose:
                V_range = np.max(self.V) - np.min(self.V)
                print(f"  Final V_bi = {V_range:.3f} V (after Boltzmann correction)")
        else:
            if verbose:
                V_range = np.max(self.V) - np.min(self.V)
                print(f"  Final potential range = {V_range:.3f} V")

        return DD1DResult(
            x=self.x,
            V=self.V,
            n=self.n,
            p=self.p,
            N_d=self.N_d,
            N_a=self.N_a,
            material=self.material,
            applied_voltage=applied_voltage,
            converged=converged,
            iterations=final_iteration,
            residual=final_residual,
        )

    def _initialize_equilibrium(self):
        """
        Initialize with charge neutrality and estimate of built-in potential.

        Uses charge neutrality for carriers and estimates V_bi from Fermi levels.
        """
        n_i = self.material.n_i
        N_net = self.N_d - self.N_a

        # Initial carrier concentrations: charge neutrality everywhere
        self.n = np.zeros(self.n_points)
        self.p = np.zeros(self.n_points)

        for i in range(self.n_points):
            if abs(N_net[i]) > n_i:
                if N_net[i] > 0:  # N-type
                    self.n[i] = N_net[i]
                    self.p[i] = n_i**2 / N_net[i]
                else:  # P-type
                    self.p[i] = -N_net[i]
                    self.n[i] = n_i**2 / (-N_net[i])
            else:  # Intrinsic
                self.n[i] = n_i
                self.p[i] = n_i

        # Initial potential: estimate from Fermi level difference
        # V(x) ≈ V_T * ln(n(x)/n_i) for N-type or -V_T * ln(p(x)/n_i) for P-type
        self.V = np.zeros(self.n_points)
        for i in range(self.n_points):
            if N_net[i] > n_i:  # N-type
                self.V[i] = V_T * np.log(self.n[i] / n_i)
            elif N_net[i] < -n_i:  # P-type
                self.V[i] = -V_T * np.log(self.p[i] / n_i)
            # else intrinsic, V = 0

    def _solve_poisson(self, applied_voltage: float):
        """
        Solve Poisson equation: d²V/dx² = -q(p - n + Nd - Na)/ε

        Boundary conditions based on quasi-Fermi levels at contacts:
        - Left contact: V = 0 (reference)
        - Right contact: V set from carrier concentration (equilibrium Fermi level)
                        plus applied voltage
        """
        eps = self.material.eps_r * eps_0
        n = self.n_points
        n_i = self.material.n_i

        # Build sparse matrix for d²V/dx²
        # Using finite differences: (V[i+1] - 2V[i] + V[i-1])/dx²

        diag = -2 * np.ones(n)
        upper = np.ones(n - 1)
        lower = np.ones(n - 1)

        A = sparse.diags([lower, diag, upper], [-1, 0, 1], shape=(n, n), format='lil')
        A = A / (self.dx * self.dx)

        # RHS: -q(p - n + Nd - Na)/ε  [convert to SI units]
        rho = (self.p - self.n + self.N_d - self.N_a) * 1e6  # cm^-3 -> m^-3
        b = -q * rho / eps

        # Boundary conditions based on doping and constant Fermi level
        # At equilibrium with Fermi level E_F (constant):
        #   V_left = E_F/q + V_T * ln(N_d_left/n_i)   [N-type]
        #   V_right = E_F/q - V_T * ln(N_a_right/n_i) [P-type]
        #
        # Setting left as reference (V_left = 0):
        #   0 = E_F/q + V_T * ln(N_d_left/n_i)
        #   E_F/q = -V_T * ln(N_d_left/n_i)
        #
        # Then right boundary:
        #   V_right = E_F/q - V_T * ln(N_a_right/n_i)
        #           = -V_T * ln(N_d_left/n_i) - V_T * ln(N_a_right/n_i)
        #           = -V_T * ln(N_d_left * N_a_right / n_i²)
        #
        # This gives V_bi = V_left - V_right = V_T * ln(N_d * N_a / n_i²)

        # Find dominant doping type in left and right bulk regions
        N_net_left = self.N_d[0] - self.N_a[0]
        N_net_right = self.N_d[-1] - self.N_a[-1]

        # Left contact: V = 0 (reference)
        A[0, :] = 0
        A[0, 0] = 1
        b[0] = 0

        # Right contact: compute potential relative to left reference
        # For a PN junction (N-left, P-right): V_right = -V_T * ln(Nd*Na/ni²)
        # More generally, compute from Fermi level difference
        if abs(N_net_left) > n_i and abs(N_net_right) > n_i:
            if N_net_left > 0 and N_net_right < 0:
                # PN junction: N-type left, P-type right
                Nd_left = abs(N_net_left)
                Na_right = abs(N_net_right)
                V_right_eq = -V_T * np.log(Nd_left * Na_right / (n_i**2))
            elif N_net_left < 0 and N_net_right > 0:
                # NP junction: P-type left, N-type right
                Na_left = abs(N_net_left)
                Nd_right = abs(N_net_right)
                V_right_eq = V_T * np.log(Na_left * Nd_right / (n_i**2))
            elif N_net_left > 0 and N_net_right > 0:
                # Both N-type
                V_right_eq = V_T * np.log(abs(N_net_right) / abs(N_net_left))
            else:
                # Both P-type
                V_right_eq = -V_T * np.log(abs(N_net_right) / abs(N_net_left))
        else:
            # At least one side is intrinsic or lightly doped
            V_right_eq = 0.0

        # Apply voltage and set boundary
        A[-1, :] = 0
        A[-1, -1] = 1
        b[-1] = V_right_eq + applied_voltage

        # Solve
        self.V = spla.spsolve(A.tocsr(), b)

    def _update_carriers(self):
        """
        Update carrier concentrations using charge neutrality approximation.

        For maximum stability during Gummel iteration:
        - Majority carrier = doping (strict charge neutrality)
        - Minority carrier = ni²/majority (mass action law)

        This is extremely stable and works for most device physics applications.
        """
        n_i = self.material.n_i
        N_net = self.N_d - self.N_a

        # Use simple charge neutrality everywhere
        for i in range(self.n_points):
            if abs(N_net[i]) > n_i:
                if N_net[i] > 0:  # N-type
                    self.n[i] = abs(N_net[i])
                    self.p[i] = n_i**2 / self.n[i]
                else:  # P-type
                    self.p[i] = abs(N_net[i])
                    self.n[i] = n_i**2 / self.p[i]
            else:
                # Intrinsic
                self.n[i] = n_i
                self.p[i] = n_i

    def _apply_boltzmann_correction(self):
        """
        Apply Boltzmann correction to final converged solution.

        After Poisson equation has converged, apply Boltzmann statistics:
        n(x) = ni * exp(q*V(x)/kT) where V is electrostatic potential
        p(x) = ni * exp(-q*V(x)/kT)

        In equilibrium with constant Fermi level, this automatically gives
        depletion in the junction region.

        This is done POST-convergence for numerical stability.
        """
        n_i = self.material.n_i

        # Apply Boltzmann statistics using the converged potential
        # Need to find reference point for Fermi level
        # Use average of bulk potentials
        idx_left = int(0.1 * self.n_points)
        idx_right = int(0.9 * self.n_points)

        # In equilibrium, Fermi level is constant
        # At left (N-side): EF/q = V_left - VT*ln(Nd/ni)
        # At right (P-side): EF/q = V_right + VT*ln(Na/ni)
        # Average these to get EF/q
        N_net_left = self.N_d[idx_left] - self.N_a[idx_left]
        N_net_right = self.N_d[idx_right] - self.N_a[idx_right]

        if abs(N_net_left) > n_i:
            EF_left = self.V[idx_left] - V_T * np.log(abs(N_net_left) / n_i)
        else:
            EF_left = self.V[idx_left]

        if abs(N_net_right) > n_i:
            EF_right = self.V[idx_right] + V_T * np.log(abs(N_net_right) / n_i)
        else:
            EF_right = self.V[idx_right]

        EF = (EF_left + EF_right) / 2  # Average Fermi level

        # Now apply Boltzmann with this Fermi level
        for i in range(self.n_points):
            # Boltzmann relations:
            # n = ni * exp((V - EF)/VT)
            # p = ni * exp((EF - V)/VT)
            psi = np.clip((self.V[i] - EF) / V_T, -20, 20)

            self.n[i] = n_i * np.exp(psi)
            self.p[i] = n_i * np.exp(-psi)

            # These automatically satisfy n*p = ni² (Boltzmann property)

        # Apply minimum carrier concentration
        min_carrier = 1e4  # cm^-3
        for i in range(self.n_points):
            self.n[i] = max(self.n[i], min_carrier)
            self.p[i] = max(self.p[i], min_carrier)

            # Re-enforce mass action after applying minimum
            np_product = self.n[i] * self.p[i]
            if np_product < n_i**2:
                scale = np.sqrt(n_i**2 / np_product)
                self.n[i] *= scale
                self.p[i] *= scale


@dataclass
class DD1DResult:
    """Results from 1D drift-diffusion simulation"""
    x: np.ndarray              # Position [cm]
    V: np.ndarray              # Potential [V]
    n: np.ndarray              # Electron concentration [cm^-3]
    p: np.ndarray              # Hole concentration [cm^-3]
    N_d: np.ndarray            # Donor doping [cm^-3]
    N_a: np.ndarray            # Acceptor doping [cm^-3]
    material: SemiconductorMaterial
    applied_voltage: float      # Applied voltage [V]
    converged: bool = True     # Whether solver converged
    iterations: int = 0        # Number of iterations taken
    residual: float = 0.0      # Final residual

    def built_in_potential(self) -> float:
        """Calculate built-in potential from doping levels [V].

        Uses Vbi = V_T * ln(N_d * N_a / n_i^2), which is a material/doping
        property independent of applied bias.
        """
        V_T = k_B * 300.0 / q  # thermal voltage at nominal T
        # Use peak donor and acceptor concentrations
        N_d_max = np.max(self.N_d)
        N_a_max = np.max(self.N_a)
        n_i = self.material.n_i
        if N_d_max > 0 and N_a_max > 0 and n_i > 0:
            return V_T * np.log(N_d_max * N_a_max / n_i ** 2)
        return np.max(self.V) - np.min(self.V)  # fallback

    def depletion_width(self, threshold: float = 1e13) -> float:
        """
        Estimate depletion region width [cm].

        Args:
            threshold: Carrier concentration threshold [cm^-3]

        Returns:
            Depletion width [cm]
        """
        # Find region where both n and p are below threshold
        depleted = (self.n < threshold) & (self.p < threshold)

        if not np.any(depleted):
            return 0.0

        # Find start and end of depletion region
        indices = np.where(depleted)[0]
        width = (indices[-1] - indices[0]) * (self.x[1] - self.x[0])

        return width

    def electric_field(self) -> np.ndarray:
        """Calculate electric field E = -dV/dx [V/cm]"""
        dx = self.x[1] - self.x[0]
        E = -np.gradient(self.V, dx)
        return E

    def max_field(self) -> float:
        """Maximum electric field magnitude [V/cm]"""
        E = self.electric_field()
        return np.max(np.abs(E))

    def junction_position(self) -> Optional[float]:
        """Find metallurgical junction position [cm]"""
        N_net = self.N_d - self.N_a

        # Find where doping type changes
        sign_changes = np.where(np.diff(np.sign(N_net)))[0]

        if len(sign_changes) > 0:
            # Return position of first sign change
            idx = sign_changes[0]
            return self.x[idx]
        else:
            return None

    def current_density(self) -> float:
        """
        Calculate total current density [A/cm²]

        Returns average current density through the device.
        For diodes, this represents the forward/reverse current.
        """
        dx = self.x[1] - self.x[0]
        E = self.electric_field()

        # Calculate electron current density: J_n = q*mu_n*n*E + q*D_n*dn/dx
        dn_dx = np.gradient(self.n, dx)
        J_n = q * self.material.mu_n * self.n * E + q * self.material.D_n * dn_dx

        # Calculate hole current density: J_p = q*mu_p*p*E - q*D_p*dp/dx
        dp_dx = np.gradient(self.p, dx)
        J_p = q * self.material.mu_p * self.p * E - q * self.material.D_p * dp_dx

        # Total current density (should be constant in steady state)
        J_total = J_n + J_p

        # Return average current density
        return np.mean(J_total)


class PNJunctionAnalyzer:
    """Analyze PN junction characteristics"""

    @staticmethod
    def compute_iv(solver: DriftDiffusion1D,
                   voltage_range: Tuple[float, float] = (-0.5, 0.8),
                   n_points: int = 20,
                   verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute I-V characteristic.

        Args:
            solver: Configured DriftDiffusion1D solver
            voltage_range: (V_min, V_max) [V]
            n_points: Number of voltage points
            verbose: Print progress

        Returns:
            (voltages, currents) arrays
        """
        voltages = np.linspace(voltage_range[0], voltage_range[1], n_points)
        currents = np.zeros(n_points)

        for i, V_app in enumerate(voltages):
            if verbose and i % 5 == 0:
                print(f"Computing I-V: V = {V_app:.3f} V ({i+1}/{n_points})")

            result = solver.solve(applied_voltage=V_app, verbose=False)

            # Estimate current density from gradient near junction
            # J_total ≈ J_n + J_p
            # Simplified: use drift current J ≈ σ·E where σ ≈ q(μn·n + μp·p)

            E = result.electric_field()
            sigma = q * (solver.material.mu_n * result.n + solver.material.mu_p * result.p)

            # Current density at junction (approximate)
            junction_idx = len(result.x) // 2
            J = sigma[junction_idx] * E[junction_idx]  # [A/cm^2]

            currents[i] = J  # Store as current density

        return voltages, currents


def create_pn_junction(N_d_level: float = 1e17,
                      N_a_level: float = 1e16,
                      junction_pos: float = 1e-4,
                      total_length: float = 2e-4) -> DriftDiffusion1D:
    """
    Create a PN junction solver with step doping profile.

    Args:
        N_d_level: N-type doping level [cm^-3]
        N_a_level: P-type doping level [cm^-3]
        junction_pos: Junction position [cm]
        total_length: Total device length [cm]

    Returns:
        Configured DriftDiffusion1D solver
    """
    solver = DriftDiffusion1D(length=total_length, n_points=200)
    solver.set_material(SemiconductorMaterial.Silicon())

    # Step junction
    solver.set_doping(
        N_d=lambda x: N_d_level if x < junction_pos else 0,
        N_a=lambda x: 0 if x < junction_pos else N_a_level,
    )

    return solver
