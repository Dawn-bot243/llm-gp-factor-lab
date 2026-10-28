from __future__ import annotations

import pytest

from src.factors.expression import (
    ALLOWED_OPERATORS,
    ComplexityError,
    ExpressionSyntaxError,
    LookAheadError,
    UnsupportedFieldError,
    UnsupportedOperatorError,
    slugify,
    validate_expression,
)
from src.factors.io import alpha158_specs


def test_alpha158_expressions_all_validate():
    specs = alpha158_specs()
    assert len(specs) == 158
    for spec in specs:
        validated = validate_expression(spec.expression)
        assert validated.canonical
        assert validated.fields
        assert validated.expression_hash
        assert validated.max_window <= 60


def test_canonical_form_is_whitespace_stable():
    first = validate_expression("Mean( $close,5 ) / $close - 1")
    second = validate_expression("Mean($close, 5)/$close-1")
    assert first.canonical == second.canonical == "Mean($close, 5) / $close - 1"
    assert first.expression_hash == second.expression_hash


def test_lookahead_ref_is_rejected():
    with pytest.raises(LookAheadError):
        validate_expression("Ref($close, -1)")
    with pytest.raises(LookAheadError):
        validate_expression("Ref($close, -5) / $close")


@pytest.mark.parametrize(
    ("expression", "error"),
    [
        ("$foo + $close", UnsupportedFieldError),
        ("Foo($close)", UnsupportedOperatorError),
        ("Feature('close')", UnsupportedOperatorError),
        ("ChangeInstrument('SH000300', $close)", UnsupportedOperatorError),
        ("Mean($close)", UnsupportedOperatorError),
        ("Mean($close, 5.5)", ExpressionSyntaxError),
        ("Mean($close, 0)", ExpressionSyntaxError),
        ("Ref($close, 0)", ExpressionSyntaxError),
        ("Ref($close, 1001)", ComplexityError),
        ("$close and $open", ExpressionSyntaxError),
        ("$close if $open else $close", ExpressionSyntaxError),
        ("lambda: $close", ExpressionSyntaxError),
        ("$close.__class__", ExpressionSyntaxError),
        ("$close[0]", ExpressionSyntaxError),
        ("-$close", ExpressionSyntaxError),
        ("__import__('os')", ExpressionSyntaxError),
        ("$close; $open", ExpressionSyntaxError),
        ("Quantile($close, 20, 1.5)", ExpressionSyntaxError),
        ("EMA($close, 2)", ExpressionSyntaxError),
        ("Log(2)", ExpressionSyntaxError),
        ("Mean(2, 5)", ExpressionSyntaxError),
        ("2 + 3", ExpressionSyntaxError),
        ("If(1, $close, $open)", ExpressionSyntaxError),
        ("Corr($close, 2, 20)", ExpressionSyntaxError),
        ("Mask($close, $open)", UnsupportedOperatorError),
        ("If(Greater($close, $open), $close, $open)", ExpressionSyntaxError),
        ("And(Greater($close, $open), Greater($close, $high))", ExpressionSyntaxError),
        ("Not(Greater($close, $open))", ExpressionSyntaxError),
    ],
)
def test_invalid_expressions_are_rejected(expression: str, error: type[Exception]):
    with pytest.raises(error):
        validate_expression(expression)


def test_raw_field_is_a_valid_factor():
    validated = validate_expression("$close")
    assert validated.canonical == "$close"
    assert validated.fields == ("close",)
    assert validated.operators == ()


def test_arithmetic_comparison_and_conditional_are_allowed():
    validated = validate_expression("If($close > $open, $close, $open)")
    assert validated.canonical == "If($close > $open, $close, $open)"
    assert validated.operators == ("If",)
    assert validate_expression("-1 * $close").canonical == "-1 * $close"
    assert validate_expression("2 ** 3 + $close").canonical == "2 ** 3 + $close"


def test_field_whitelist_can_be_narrowed():
    with pytest.raises(UnsupportedFieldError):
        validate_expression("$vwap / $close", allowed_fields=("close",))
    assert validate_expression("$close / $open", allowed_fields=("close", "open")).fields == (
        "close",
        "open",
    )


def test_slugify():
    assert slugify("MA 5 / Close!") == "ma_5_close"
    assert slugify("   ") == "factor"


def test_allowed_operators_are_registered_in_qlib():
    from qlib.config import C
    from qlib.data.ops import Operators, register_all_ops

    register_all_ops(C)
    assert ALLOWED_OPERATORS <= set(Operators._ops)


def test_alpha158_canonical_forms_parse_in_qlib():
    # Qlib's own parser is eval(parse_field(...)); this test feeds it only
    # canonical strings produced by the validator, never raw user input.
    from qlib.config import C
    from qlib.data.base import Feature
    from qlib.data.ops import Operators, register_all_ops
    from qlib.utils import parse_field

    register_all_ops(C)
    namespace = {"Operators": Operators, "Feature": Feature}
    for spec in alpha158_specs():
        validated = validate_expression(spec.expression)
        parsed = eval(parse_field(validated.canonical), namespace)
        assert parsed is not None


def test_all_allowed_operators_parse_in_qlib():
    from qlib.config import C
    from qlib.data.base import Feature
    from qlib.data.ops import Operators, register_all_ops
    from qlib.utils import parse_field

    register_all_ops(C)
    namespace = {"Operators": Operators, "Feature": Feature}
    expressions = {
        "Abs": "Abs($close)",
        "Sign": "Sign($close)",
        "Log": "Log($close)",
        "Not": "Not($close > $open)",
        "If": "If($close > $open, $close, $open)",
        "Power": "Power($close, 2)",
        "Add": "Add($close, $open)",
        "Sub": "Sub($close, $open)",
        "Mul": "Mul($close, $open)",
        "Div": "Div($close, $open)",
        "Greater": "Greater($close, $open)",
        "Less": "Less($close, $open)",
        "Gt": "Gt($close, $open)",
        "Ge": "Ge($close, $open)",
        "Lt": "Lt($close, $open)",
        "Le": "Le($close, $open)",
        "Eq": "Eq($close, $open)",
        "Ne": "Ne($close, $open)",
        "And": "And($close > $open, $close > $high)",
        "Or": "Or($close > $open, $close > $high)",
        "Ref": "Ref($close, 5)",
        "Mean": "Mean($close, 5)",
        "Sum": "Sum($close, 5)",
        "Std": "Std($close, 5)",
        "Var": "Var($close, 5)",
        "Skew": "Skew($close, 5)",
        "Kurt": "Kurt($close, 5)",
        "Max": "Max($close, 5)",
        "IdxMax": "IdxMax($close, 5)",
        "Min": "Min($close, 5)",
        "IdxMin": "IdxMin($close, 5)",
        "Quantile": "Quantile($close, 20, 0.8)",
        "Med": "Med($close, 5)",
        "Mad": "Mad($close, 5)",
        "Rank": "Rank($close, 5)",
        "Count": "Count($close, 5)",
        "Delta": "Delta($close, 5)",
        "Slope": "Slope($close, 5)",
        "Rsquare": "Rsquare($close, 5)",
        "Resi": "Resi($close, 5)",
        "WMA": "WMA($close, 5)",
        "EMA": "EMA($close, 0.5)",
        "Corr": "Corr($close, $open, 20)",
        "Cov": "Cov($close, $open, 20)",
    }
    assert set(expressions) == set(ALLOWED_OPERATORS)
    for expression in expressions.values():
        canonical = validate_expression(expression).canonical
        assert eval(parse_field(canonical), namespace) is not None
