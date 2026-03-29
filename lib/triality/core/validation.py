"""
Expression Validation Layer

Input sanitation for expressions, boundary conditions, and forcing terms.
Catches NaN/Inf/invalid values before they reach the discretizer.

Called during expression lowering, before discretization.
"""

import numpy as np
from typing import Any, Optional


class ValidationError(ValueError):
    """Raised when input validation fails"""
    pass


def validate_scalar(value: Any, name: str = "value") -> float:
    """
    Validate a scalar value

    Args:
        value: Value to validate
        name: Name for error messages

    Returns:
        Validated float

    Raises:
        ValidationError: If value is invalid
    """
    try:
        val = float(value)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Invalid {name}: cannot convert to float: {e}")

    if np.isnan(val):
        raise ValidationError(
            f"Invalid {name}: NaN detected\n"
            f"  Suggestion: Check your input values and expressions"
        )

    if np.isinf(val):
        raise ValidationError(
            f"Invalid {name}: Inf detected\n"
            f"  Suggestion: Check for division by zero or overflow"
        )

    return val


def validate_array(arr: np.ndarray, name: str = "array") -> np.ndarray:
    """
    Validate array values

    Args:
        arr: Array to validate
        name: Name for error messages

    Returns:
        Validated array

    Raises:
        ValidationError: If array contains invalid values
    """
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.array(arr)
        except Exception as e:
            raise ValidationError(f"Invalid {name}: cannot convert to array: {e}")

    if arr.size == 0:
        raise ValidationError(f"Invalid {name}: empty array")

    # Check for NaN
    if np.any(np.isnan(arr)):
        nan_count = np.sum(np.isnan(arr))
        raise ValidationError(
            f"Invalid {name}: contains {nan_count} NaN value(s)\n"
            f"  Location: indices {np.where(np.isnan(arr))[0][:5]}\n"
            f"  Suggestion: Check your forcing function or input data"
        )

    # Check for Inf
    if np.any(np.isinf(arr)):
        inf_count = np.sum(np.isinf(arr))
        raise ValidationError(
            f"Invalid {name}: contains {inf_count} Inf value(s)\n"
            f"  Location: indices {np.where(np.isinf(arr))[0][:5]}\n"
            f"  Suggestion: Check for division by zero or overflow"
        )

    return arr


def validate_boundary_conditions(bc: dict, domain) -> dict:
    """
    Validate boundary condition values

    Args:
        bc: Boundary conditions dict
        domain: Domain object

    Returns:
        Validated BC dict

    Raises:
        ValidationError: If any BC value is invalid
    """
    if not bc:
        return bc

    validated_bc = {}

    for key, value in bc.items():
        # Validate scalar BC values
        if isinstance(value, (int, float)):
            try:
                validated_bc[key] = validate_scalar(value, name=f"BC '{key}'")
            except ValidationError as e:
                # Add context about which boundary
                raise ValidationError(
                    f"Boundary condition error at '{key}':\n  {str(e)}"
                )
        elif callable(value):
            # Can't validate functions until evaluation
            # Add a wrapper that validates output
            def validated_func(x, original_func=value, bc_name=key):
                result = original_func(x)
                try:
                    if np.isscalar(result):
                        return validate_scalar(result, f"BC '{bc_name}' output")
                    else:
                        return validate_array(np.array(result), f"BC '{bc_name}' output")
                except ValidationError as e:
                    raise ValidationError(
                        f"Boundary condition function '{bc_name}' produced invalid output:\n  {str(e)}"
                    )
            validated_bc[key] = validated_func
        else:
            # Pass through non-numeric values (might be symbolic)
            validated_bc[key] = value

    return validated_bc


def validate_forcing(forcing: Any, name: str = "forcing") -> Any:
    """
    Validate forcing term

    Args:
        forcing: Forcing term (scalar, array, or callable)
        name: Name for error messages

    Returns:
        Validated forcing term

    Raises:
        ValidationError: If forcing is invalid
    """
    if forcing is None:
        return None

    # Scalar forcing
    if isinstance(forcing, (int, float)):
        return validate_scalar(forcing, name)

    # Array forcing
    elif isinstance(forcing, np.ndarray):
        return validate_array(forcing, name)

    # Function forcing - wrap with validation
    elif callable(forcing):
        def validated_forcing_func(*args, original_func=forcing):
            result = original_func(*args)
            try:
                if np.isscalar(result):
                    return validate_scalar(result, f"{name} output")
                else:
                    return validate_array(np.array(result), f"{name} output")
            except ValidationError as e:
                raise ValidationError(
                    f"Forcing function produced invalid output:\n  {str(e)}\n"
                    f"  Args: {args}"
                )
        return validated_forcing_func

    else:
        # Try to convert to array
        try:
            return validate_array(np.array(forcing), name)
        except Exception as e:
            raise ValidationError(
                f"Invalid {name}: unsupported type {type(forcing)}\n"
                f"  Suggestion: Use scalar, array, or callable"
            )


def validate_resolution(resolution: int) -> int:
    """
    Validate grid resolution

    Args:
        resolution: Grid resolution

    Returns:
        Validated resolution

    Raises:
        ValidationError: If resolution is invalid
    """
    if not isinstance(resolution, int):
        try:
            resolution = int(resolution)
        except Exception:
            raise ValidationError(
                f"Invalid resolution: must be integer, got {type(resolution)}"
            )

    if resolution < 3:
        raise ValidationError(
            f"Invalid resolution: {resolution} < 3\n"
            f"  Suggestion: Use at least 3 grid points (minimum for interior points)"
        )

    if resolution > 10000:
        raise ValidationError(
            f"Invalid resolution: {resolution} > 10000\n"
            f"  Suggestion: Such fine grids may cause memory issues\n"
            f"  Consider using adaptive mesh refinement (future feature)"
        )

    return resolution


def validate_domain(domain):
    """
    Validate domain bounds

    Args:
        domain: Domain object

    Raises:
        ValidationError: If domain is invalid
    """
    # 1D Interval
    if hasattr(domain, 'a') and hasattr(domain, 'b'):
        if domain.a >= domain.b:
            raise ValidationError(
                f"Invalid interval: {domain.a} >= {domain.b}\n"
                f"  Suggestion: Ensure left bound < right bound"
            )
        if np.isnan(domain.a) or np.isnan(domain.b):
            raise ValidationError("Invalid interval: bounds contain NaN")
        if np.isinf(domain.a) or np.isinf(domain.b):
            raise ValidationError(
                "Invalid interval: infinite bounds not supported\n"
                "  Suggestion: Use finite domain with appropriate BCs"
            )

    # 2D Rectangle
    elif hasattr(domain, 'xmin') and hasattr(domain, 'xmax'):
        if domain.xmin >= domain.xmax:
            raise ValidationError(
                f"Invalid rectangle: xmin={domain.xmin} >= xmax={domain.xmax}"
            )
        if domain.ymin >= domain.ymax:
            raise ValidationError(
                f"Invalid rectangle: ymin={domain.ymin} >= ymax={domain.ymax}"
            )

    # 2D Square
    elif hasattr(domain, 'L'):
        if domain.L <= 0:
            raise ValidationError(f"Invalid square: side length {domain.L} <= 0")
        if np.isnan(domain.L) or np.isinf(domain.L):
            raise ValidationError("Invalid square: side length is NaN or Inf")
