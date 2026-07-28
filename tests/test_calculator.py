"""
Tests for safe arithmetic evaluation.

The evaluator runs on untrusted user text, so the security tests are the
important ones: no name resolution, no calls, no attribute access, and no
unbounded computation. A regression here would turn a calculator into
arbitrary code execution.
"""

from __future__ import annotations

import pytest

from src.exceptions import ValidationError
from src.routing.calculator import evaluate_expression, format_result


class TestArithmetic:
    """Correct evaluation of permitted expressions."""

    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("2+2", 4),
            ("12 * 7", 84),
            ("(3 + 4) / 2", 3.5),
            ("10 - 3 - 2", 5),
            ("7 // 2", 3),
            ("7 % 3", 1),
            ("2 ** 10", 1024),
            ("-5 + 3", -2),
            ("+4", 4),
            ("1.5 * 4", 6.0),
            ("((2))", 2),
        ],
    )
    def test_evaluates(
        self,
        expression: str,
        expected: float,
    ) -> None:
        """Permitted arithmetic evaluates to the expected value."""
        assert evaluate_expression(expression) == expected

    def test_operator_precedence(self) -> None:
        """Standard precedence is preserved."""
        assert evaluate_expression("2 + 3 * 4") == 14


class TestSecurity:
    """Constructs that must be rejected outright."""

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('echo hi')",
            "open('/etc/passwd').read()",
            "eval('2+2')",
            "os.getcwd()",
            "x",
            "True",
            "[1,2,3]",
            "{'a': 1}",
            "(1).__class__",
            "lambda: 1",
            "[i for i in range(3)]",
            "'abc' * 3",
            "1 if True else 2",
        ],
    )
    def test_rejects_non_arithmetic(self, expression: str) -> None:
        """Anything beyond arithmetic raises rather than evaluating."""
        with pytest.raises(ValidationError):
            evaluate_expression(expression)

    def test_rejects_large_exponent(self) -> None:
        """
        Exponent size is bounded. '9**9**9' is a resource-exhaustion
        attempt, not a calculation.
        """
        with pytest.raises(ValidationError) as excinfo:
            evaluate_expression("9 ** 99999")
        assert excinfo.value.code == "CALC_EXPONENT_TOO_LARGE"

    def test_rejects_overlong_input(self) -> None:
        """Very long expressions are refused before parsing."""
        with pytest.raises(ValidationError) as excinfo:
            evaluate_expression("1+" * 500 + "1")
        assert excinfo.value.code == "CALC_TOO_LONG"


class TestErrorHandling:
    """Arithmetic errors surface as validation errors."""

    def test_division_by_zero(self) -> None:
        """Division by zero is reported, not raised as ZeroDivisionError."""
        with pytest.raises(ValidationError) as excinfo:
            evaluate_expression("1 / 0")
        assert excinfo.value.code == "CALC_DIVISION_BY_ZERO"

    def test_empty_expression(self) -> None:
        """Empty input is rejected."""
        with pytest.raises(ValidationError) as excinfo:
            evaluate_expression("   ")
        assert excinfo.value.code == "CALC_EMPTY"

    def test_syntax_error(self) -> None:
        """Malformed input is reported as a syntax error."""
        with pytest.raises(ValidationError) as excinfo:
            evaluate_expression("2 +")
        assert excinfo.value.code == "CALC_SYNTAX_ERROR"


class TestFormatting:
    """Result rendering."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (4, "4"),
            (3.5, "3.5"),
            (6.0, "6"),
            (-2, "-2"),
            (0.1, "0.1"),
        ],
    )
    def test_formats(self, value: float, expected: str) -> None:
        """Integral floats render without a trailing decimal."""
        assert format_result(value) == expected

    def test_round_trip_with_evaluation(self) -> None:
        """Division producing a whole number reads naturally."""
        assert format_result(evaluate_expression("4 / 2")) == "2"
