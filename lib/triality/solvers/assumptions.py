"""
Assumption Tracking System

Every solve operation tracks what assumptions it makes.
This builds trust through transparency.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Assumptions:
    """Tracks all assumptions made during solving"""

    # Smoothness assumptions
    solution_smoothness: str = "C²"  # Twice continuously differentiable
    forcing_smoothness: str = "C⁰"   # Continuous
    domain_smoothness: str = "C∞"    # Smooth domain (rectangles, circles)

    # Linearity assumptions
    is_linear: bool = True
    linearity_checked: bool = False

    # Symmetry assumptions
    has_symmetry: Optional[str] = None  # None, 'even', 'odd', 'radial'
    symmetry_exploited: bool = False

    # Stability assumptions
    is_wellposed: bool = True
    condition_number_estimate: Optional[float] = None

    # Discretization assumptions
    grid_type: str = "uniform"
    boundary_treatment: str = "dirichlet"
    stencil_order: int = 2  # O(h²) accurate

    # Solver assumptions
    matrix_properties: List[str] = field(default_factory=lambda: ["sparse", "symmetric", "positive-definite"])
    preconditioner_assumptions: List[str] = field(default_factory=list)

    # Physical assumptions
    conservation_laws: List[str] = field(default_factory=list)
    maximum_principle: bool = False

    # Warnings and caveats
    warnings: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    def add_warning(self, msg: str):
        """Add a warning"""
        if msg not in self.warnings:
            self.warnings.append(msg)

    def add_caveat(self, msg: str):
        """Add a caveat"""
        if msg not in self.caveats:
            self.caveats.append(msg)

    def verify_smoothness(self, pde_order: int):
        """Verify solution has enough smoothness"""
        required = f"C^{pde_order}"
        if pde_order == 2 and self.solution_smoothness != "C²":
            self.add_warning(f"PDE requires {required} solution, but smoothness unknown")

    def verify_wellposed(self, classification):
        """Check if problem is well-posed"""
        # Elliptic PDEs with Dirichlet BC are well-posed
        if classification.pde_type == 'elliptic':
            if self.boundary_treatment == 'dirichlet':
                self.is_wellposed = True
            elif self.boundary_treatment == 'neumann':
                self.is_wellposed = False
                self.add_warning("Pure Neumann problem has nullspace - solution not unique")

        # Parabolic PDEs need initial condition
        elif classification.pde_type == 'parabolic':
            self.add_caveat("Parabolic PDE - requires initial condition (not yet supported)")
            self.is_wellposed = False

        # Hyperbolic PDEs need IC and proper boundary treatment
        elif classification.pde_type == 'hyperbolic':
            self.add_caveat("Hyperbolic PDE - requires characteristics analysis (not yet supported)")
            self.is_wellposed = False

    def check_resolution(self, resolution: int, domain_size: float):
        """Check if resolution is adequate"""
        h = domain_size / (resolution - 1)

        if resolution < 5:
            self.add_warning(f"Very coarse grid (N={resolution}) - solution may be inaccurate")
        elif resolution < 10:
            self.add_warning(f"Coarse grid (N={resolution}, h={h:.3f}) - consider refining")
        elif resolution > 1000:
            self.add_caveat(f"Fine grid (N={resolution}) - iterative solver recommended")

    def check_forcing(self, forcing_values):
        """Check forcing term for issues"""
        import numpy as np

        if isinstance(forcing_values, (list, np.ndarray)):
            # Check for NaN/Inf
            if np.any(np.isnan(forcing_values)):
                self.add_warning("Forcing term contains NaN values")
                self.is_wellposed = False

            if np.any(np.isinf(forcing_values)):
                self.add_warning("Forcing term contains Inf values")
                self.is_wellposed = False

            # Check magnitude
            max_forcing = np.max(np.abs(forcing_values))
            if max_forcing < 1e-14:
                self.add_caveat("Very small forcing term - solution may be dominated by BCs")
            elif max_forcing > 1e10:
                self.add_warning("Very large forcing term - solution may have large gradients")

    def to_dict(self):
        """Convert to dictionary for storage"""
        return {
            'smoothness': {
                'solution': self.solution_smoothness,
                'forcing': self.forcing_smoothness,
                'domain': self.domain_smoothness
            },
            'linearity': {
                'is_linear': self.is_linear,
                'checked': self.linearity_checked
            },
            'discretization': {
                'grid': self.grid_type,
                'boundary': self.boundary_treatment,
                'order': self.stencil_order
            },
            'stability': {
                'wellposed': self.is_wellposed,
                'condition_number': self.condition_number_estimate
            },
            'warnings': self.warnings,
            'caveats': self.caveats
        }

    def __str__(self):
        """Human-readable summary"""
        lines = []
        lines.append("Assumptions:")
        lines.append(f"  • Solution smoothness: {self.solution_smoothness}")
        lines.append(f"  • Grid: {self.grid_type}, {self.stencil_order}th order accurate")
        lines.append(f"  • Boundary treatment: {self.boundary_treatment}")
        lines.append(f"  • Well-posed: {self.is_wellposed}")

        if self.matrix_properties:
            props = ", ".join(self.matrix_properties)
            lines.append(f"  • Matrix: {props}")

        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠️  {w}")

        if self.caveats:
            lines.append("\nCaveats:")
            for c in self.caveats:
                lines.append(f"  ℹ️  {c}")

        return "\n".join(lines)


def make_assumptions(equation, domain, classification, resolution):
    """
    Create assumption tracker for a solve operation

    Args:
        equation: The PDE equation
        domain: Geometric domain
        classification: Problem classification
        resolution: Grid resolution

    Returns:
        Assumptions object
    """
    assumptions = Assumptions()

    # Set linearity
    assumptions.is_linear = classification.is_linear
    assumptions.linearity_checked = True

    # Set boundary treatment (currently only Dirichlet supported)
    assumptions.boundary_treatment = "dirichlet"

    # Verify well-posedness
    assumptions.verify_wellposed(classification)

    # Check resolution
    if hasattr(domain, 'length'):
        assumptions.check_resolution(resolution, domain.length())
    elif hasattr(domain, 'L'):
        assumptions.check_resolution(resolution, domain.L)

    # Check for symmetry opportunities
    if hasattr(domain, 'L'):  # Square domain
        assumptions.add_caveat("Square domain - could exploit symmetry for 2x speedup")

    # Set matrix properties based on problem type
    if classification.pde_type == 'elliptic' and classification.is_linear:
        assumptions.matrix_properties = ["sparse", "symmetric", "positive-definite"]
        assumptions.maximum_principle = True
        assumptions.conservation_laws = ["energy"]

    return assumptions
