"""
Well-posedness Validator

Checks if a problem is mathematically well-posed before attempting to solve.
This catches issues like nullspace problems, incompatible BCs, etc.

Called during problem analysis, before solver selection.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class WellPosednessResult:
    """Result of well-posedness check"""
    is_wellposed: bool
    issue: Optional[str] = None
    suggestion: Optional[str] = None


def check_wellposedness(classification, domain, bc) -> WellPosednessResult:
    """
    Check if problem is well-posed

    Args:
        classification: PDE classification
        domain: Geometric domain
        bc: Boundary conditions dict

    Returns:
        WellPosednessResult with diagnosis
    """

    # Check 1: Elliptic with pure Neumann BC (nullspace issue)
    if classification.pde_type == 'elliptic':
        if _is_pure_neumann(bc, domain):
            return WellPosednessResult(
                is_wellposed=False,
                issue="Elliptic operator with pure Neumann boundary conditions has nullspace (solution not unique)",
                suggestion="Fix u at one boundary point or add integral constraint ∫u=0"
            )

    # Check 2: Parabolic without time derivative or IC
    if classification.pde_type == 'parabolic':
        if not classification.has_time:
            return WellPosednessResult(
                is_wellposed=False,
                issue="Parabolic PDE requires time derivative",
                suggestion="Add ∂u/∂t term or reclassify problem"
            )
        # Initial condition check deferred until IC support is implemented

    # Check 3: Hyperbolic without proper characteristics
    if classification.pde_type == 'hyperbolic':
        return WellPosednessResult(
            is_wellposed=False,
            issue="Hyperbolic PDEs not yet supported",
            suggestion="Use time-stepping method (future feature)"
        )

    # Check 4: Overconstrained (too many BCs)
    if _is_overconstrained(bc, domain):
        return WellPosednessResult(
            is_wellposed=False,
            issue="Overconstrained: too many boundary conditions specified",
            suggestion="Remove redundant boundary conditions"
        )

    # All checks passed
    return WellPosednessResult(is_wellposed=True)


def _is_pure_neumann(bc, domain) -> bool:
    """Check if all BCs are Neumann (flux) type"""
    if not bc:
        return False

    # Check for flux-type BC keywords
    flux_keywords = ['flux', 'neumann', 'derivative', 'grad']

    # Check if any BC has flux keyword
    has_flux = any(
        any(keyword in str(key).lower() for keyword in flux_keywords)
        for key in bc.keys()
    )

    # Check if NO Dirichlet (value) BCs are present
    dirichlet_keywords = ['left', 'right', 'top', 'bottom', 'boundary']
    has_dirichlet = any(
        any(keyword in str(key).lower() for keyword in dirichlet_keywords)
        and 'flux' not in str(key).lower()
        for key in bc.keys()
    )

    # Pure Neumann if: has flux BCs AND no Dirichlet BCs
    return has_flux and not has_dirichlet


def _is_overconstrained(bc, domain) -> bool:
    """Check if too many BCs specified"""
    if not bc:
        return False

    # For 1D: should have at most 2 BCs (left, right)
    if hasattr(domain, 'a') and hasattr(domain, 'b'):  # Interval
        expected_max = 2
        # Check for conflicting BCs at same location
        if 'left' in bc and 'left_flux' in bc:
            return True
        if 'right' in bc and 'right_flux' in bc:
            return True

    # For 2D: should have at most 4 BCs (left, right, top, bottom)
    elif hasattr(domain, 'L') or hasattr(domain, 'xmin'):  # Square/Rectangle
        expected_max = 4
        # Check for conflicting BCs at same location
        sides = ['left', 'right', 'top', 'bottom']
        for side in sides:
            if side in bc and f'{side}_flux' in bc:
                return True

    return False
