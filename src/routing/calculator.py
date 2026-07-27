"""
Safe arithmetic evaluation for locally-handled calculations.

Requests that are pure arithmetic are answered without any model call.
That requires evaluating user-supplied text, which must never be done with
``eval``: arbitrary input would then be arbitrary code execution.

This evaluator parses the expression into an abstract syntax tree and
walks it with an explicit allowlist of node types. Names, attributes,
subscripts, comprehensions, and calls are all rejected, so there is no
route to reading or invoking anything. Exponentiation is additionally
bounded because a small expression such as ``9**9**9`` is a
denial-of-service rather than a calculation.

Depends only on the standard library.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Final

from src.exceptions import ValidationError

# Binary operators permitted in expressions.
_BINARY_OPERATORS: Final[dict[type[ast.operator], Callable[[Any, Any], Any]]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPERATORS: Final[dict[type[ast.unaryop], Callable[[Any], Any]]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Bounds chosen so that legitimate arithmetic passes while expressions
# whose only purpose is to exhaust memory or CPU are rejected.
_MAX_EXPONENT: Final[int] = 1024
_MAX_EXPRESSION_LENGTH: Final[int] = 256


def _reject(node: ast.AST) -> None:
    """
    Raise for a disallowed syntax node.

    Args:
        node: Offending AST node.

    Raises:
        ValidationError: Always.
    """
    raise ValidationError(
        f"Unsupported expression element: {type(node).__name__}",
        code="CALC_UNSUPPORTED_SYNTAX",
    )


def _evaluate_node(node: ast.AST) -> float | int:
    """
    Recursively evaluate an allowed AST node.

    Args:
        node: Node to evaluate.

    Returns:
        Numeric value of the node.

    Raises:
        ValidationError: If the node type is not allowed or evaluation
            is not arithmetically valid.
    """
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float)
        ):
            _reject(node)
        return node.value

    if isinstance(node, ast.UnaryOp):
        handler = _UNARY_OPERATORS.get(type(node.op))
        if handler is None:
            _reject(node.op)
        return handler(_evaluate_node(node.operand))  # type: ignore[misc]

    if isinstance(node, ast.BinOp):
        handler = _BINARY_OPERATORS.get(type(node.op))
        if handler is None:
            _reject(node.op)

        left = _evaluate_node(node.left)
        right = _evaluate_node(node.right)

        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValidationError(
                f"Exponent {right} exceeds the maximum of {_MAX_EXPONENT}",
                code="CALC_EXPONENT_TOO_LARGE",
            )

        try:
            return handler(left, right)  # type: ignore[misc]
        except ZeroDivisionError as exc:
            raise ValidationError(
                "Division by zero",
                code="CALC_DIVISION_BY_ZERO",
            ) from exc
        except OverflowError as exc:
            raise ValidationError(
                "Result is too large to represent",
                code="CALC_OVERFLOW",
            ) from exc

    _reject(node)
    raise AssertionError("unreachable")  # pragma: no cover


def evaluate_expression(expression: str) -> float | int:
    """
    Evaluate an arithmetic expression safely.

    Args:
        expression: Expression text, for example ``"12 * (3 + 4)"``.

    Returns:
        Numeric result.

    Raises:
        ValidationError: If the expression is empty, too long,
            syntactically invalid, contains disallowed constructs, or
            does not evaluate to a finite number.
    """
    text = (expression or "").strip()

    if not text:
        raise ValidationError(
            "Expression cannot be empty",
            code="CALC_EMPTY",
        )

    if len(text) > _MAX_EXPRESSION_LENGTH:
        raise ValidationError(
            f"Expression exceeds {_MAX_EXPRESSION_LENGTH} characters",
            code="CALC_TOO_LONG",
        )

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValidationError(
            f"Could not parse expression: {exc.msg}",
            code="CALC_SYNTAX_ERROR",
        ) from exc

    result = _evaluate_node(tree)

    if isinstance(result, float) and not math.isfinite(result):
        raise ValidationError(
            "Result is not a finite number",
            code="CALC_NOT_FINITE",
        )
    return result


def format_result(value: float | int) -> str:
    """
    Render a numeric result for display.

    Integral floats are shown without a trailing ``.0`` so ``4 / 2``
    reads as ``2`` rather than ``2.0``. Other floats are trimmed of
    trailing zeros introduced by binary representation.

    Args:
        value: Numeric result.

    Returns:
        Display string.
    """
    if isinstance(value, int):
        return str(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.10g}"


__all__ = [
    "evaluate_expression",
    "format_result",
]
