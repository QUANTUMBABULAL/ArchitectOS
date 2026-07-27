"""
Input validation utilities for the AI Research Operator.

This module provides reusable validation functions for common input
validation scenarios across the application.

Features:
    - Type validation helpers
    - String validation
    - Numeric validation
    - Collection validation
    - Custom validators

Usage:
    from src.utils.validators import validate_string, validate_positive_int

    validate_string(value, min_length=1)
    validate_positive_int(port)
"""

from typing import Any, List, Optional, Type, TypeVar

from src.exceptions import ValidationError

T = TypeVar("T")


def validate_string(
    value: Any,
    field_name: str = "value",
    min_length: int = 0,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
) -> str:
    """
    Validate that value is a string with optional constraints.

    Args:
        value: Value to validate.
        field_name: Name of the field for error messages.
        min_length: Minimum string length.
        max_length: Maximum string length.
        pattern: Regex pattern to match.

    Returns:
        Validated string.

    Raises:
        ValidationError: If validation fails.

    Example:
        >>> validate_string("test", min_length=1, max_length=100)
        'test'
    """
    if not isinstance(value, str):
        raise ValidationError(
            f"{field_name} must be a string, got {type(value).__name__}"
        )

    if len(value) < min_length:
        raise ValidationError(
            f"{field_name} must be at least {min_length} characters"
        )

    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            f"{field_name} must be at most {max_length} characters"
        )

    if pattern is not None:
        import re

        if not re.match(pattern, value):
            raise ValidationError(
                f"{field_name} does not match required pattern: {pattern}"
            )

    return value


def validate_integer(
    value: Any,
    field_name: str = "value",
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> int:
    """
    Validate that value is an integer with optional range constraints.

    Args:
        value: Value to validate.
        field_name: Name of the field for error messages.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.

    Returns:
        Validated integer.

    Raises:
        ValidationError: If validation fails.

    Example:
        >>> validate_integer(42, min_value=0, max_value=100)
        42
    """
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValidationError(
            f"{field_name} must be an integer, got {type(value).__name__}"
        )

    if min_value is not None and value < min_value:
        raise ValidationError(
            f"{field_name} must be at least {min_value}"
        )

    if max_value is not None and value > max_value:
        raise ValidationError(
            f"{field_name} must be at most {max_value}"
        )

    return value


def validate_float(
    value: Any,
    field_name: str = "value",
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """
    Validate that value is a float with optional range constraints.

    Args:
        value: Value to validate.
        field_name: Name of the field for error messages.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.

    Returns:
        Validated float.

    Raises:
        ValidationError: If validation fails.

    Example:
        >>> validate_float(3.14, min_value=0.0, max_value=10.0)
        3.14
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValidationError(
            f"{field_name} must be a number, got {type(value).__name__}"
        )

    float_value = float(value)

    if min_value is not None and float_value < min_value:
        raise ValidationError(
            f"{field_name} must be at least {min_value}"
        )

    if max_value is not None and float_value > max_value:
        raise ValidationError(
            f"{field_name} must be at most {max_value}"
        )

    return float_value


def validate_list(
    value: Any,
    field_name: str = "value",
    min_length: int = 0,
    max_length: Optional[int] = None,
    element_type: Optional[Type[T]] = None,
) -> List[T]:
    """
    Validate that value is a list with optional constraints.

    Args:
        value: Value to validate.
        field_name: Name of the field for error messages.
        min_length: Minimum list length.
        max_length: Maximum list length.
        element_type: Type that all elements must be.

    Returns:
        Validated list.

    Raises:
        ValidationError: If validation fails.

    Example:
        >>> validate_list([1, 2, 3], min_length=1, element_type=int)
        [1, 2, 3]
    """
    if not isinstance(value, list):
        raise ValidationError(
            f"{field_name} must be a list, got {type(value).__name__}"
        )

    if len(value) < min_length:
        raise ValidationError(
            f"{field_name} must have at least {min_length} elements"
        )

    if max_length is not None and len(value) > max_length:
        raise ValidationError(
            f"{field_name} must have at most {max_length} elements"
        )

    if element_type is not None:
        for i, element in enumerate(value):
            if not isinstance(element, element_type):
                raise ValidationError(
                    f"{field_name}[{i}] must be {element_type.__name__}, "
                    f"got {type(element).__name__}"
                )

    return value


def validate_dict(
    value: Any,
    field_name: str = "value",
    required_keys: Optional[List[str]] = None,
) -> dict:
    """
    Validate that value is a dictionary with optional key constraints.

    Args:
        value: Value to validate.
        field_name: Name of the field for error messages.
        required_keys: List of keys that must be present.

    Returns:
        Validated dictionary.

    Raises:
        ValidationError: If validation fails.

    Example:
        >>> validate_dict({"a": 1}, required_keys=["a"])
        {'a': 1}
    """
    if not isinstance(value, dict):
        raise ValidationError(
            f"{field_name} must be a dictionary, got {type(value).__name__}"
        )

    if required_keys:
        missing_keys = set(required_keys) - set(value.keys())
        if missing_keys:
            raise ValidationError(
                f"{field_name} missing required keys: {missing_keys}"
            )

    return value


def validate_type(
    value: Any,
    expected_type: Type[T],
    field_name: str = "value",
) -> T:
    """
    Validate that value is of expected type.

    Args:
        value: Value to validate.
        expected_type: Expected type.
        field_name: Name of the field for error messages.

    Returns:
        Validated value.

    Raises:
        ValidationError: If type doesn't match.

    Example:
        >>> validate_type("test", str, "name")
        'test'
    """
    if not isinstance(value, expected_type):
        raise ValidationError(
            f"{field_name} must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )

    return value


__all__ = [
    "validate_string",
    "validate_integer",
    "validate_float",
    "validate_list",
    "validate_dict",
    "validate_type",
]
