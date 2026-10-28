"""Validate Qlib factor expressions before they reach the expression engine.

Qlib parses expression strings with ``eval`` after a small regex rewrite.
That is convenient, but LLM/GP-generated text must never be handed to it
unchecked.  This module:

* parses the string with Python's ``ast`` (never ``eval``);
* allows only a whitelist of Qlib fields and operators;
* rejects future-looking shifts such as ``Ref($close, -1)``;
* bounds expression length, nesting and rolling windows;
* returns a canonical, Qlib-compatible expression plus a stable hash.

The canonical form is safe to pass to Qlib because it is generated from the
validated AST, not from the raw input string.
"""
from __future__ import annotations

import ast
import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Fields present in the current investment_data archive.  The list is kept
# explicit so a typo cannot silently load an unrelated provider feature.
DEFAULT_FIELDS: frozenset[str] = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "factor",
        "adjclose",
        "amount",
        "vwap",
        "change",
    }
)

# Qlib 0.9.7 operators that are meaningful for a point-in-time factor.  The
# arity is (min_args, max_args); the first arg is always an expression unless
# noted otherwise.
_OPERATOR_ARITY: dict[str, tuple[int, int]] = {
    # element-wise
    "Abs": (1, 1),
    "Sign": (1, 1),
    "Log": (1, 1),
    "Not": (1, 1),
    "If": (3, 3),
    # pair-wise
    "Power": (2, 2),
    "Add": (2, 2),
    "Sub": (2, 2),
    "Mul": (2, 2),
    "Div": (2, 2),
    "Greater": (2, 2),
    "Less": (2, 2),
    "Gt": (2, 2),
    "Ge": (2, 2),
    "Lt": (2, 2),
    "Le": (2, 2),
    "Eq": (2, 2),
    "Ne": (2, 2),
    "And": (2, 2),
    "Or": (2, 2),
    # rolling
    "Ref": (2, 2),
    "Mean": (2, 2),
    "Sum": (2, 2),
    "Std": (2, 2),
    "Var": (2, 2),
    "Skew": (2, 2),
    "Kurt": (2, 2),
    "Max": (2, 2),
    "IdxMax": (2, 2),
    "Min": (2, 2),
    "IdxMin": (2, 2),
    "Quantile": (2, 3),
    "Med": (2, 2),
    "Mad": (2, 2),
    "Rank": (2, 2),
    "Count": (2, 2),
    "Delta": (2, 2),
    "Slope": (2, 2),
    "Rsquare": (2, 2),
    "Resi": (2, 2),
    "WMA": (2, 2),
    "EMA": (2, 2),
    # pair rolling
    "Corr": (3, 3),
    "Cov": (3, 3),
}

ALLOWED_OPERATORS: frozenset[str] = frozenset(_OPERATOR_ARITY)

# Operators Qlib knows but that are deliberately excluded: they either load
# arbitrary features/instruments, are abstract, or are not point-in-time safe.
BLOCKED_OPERATORS: frozenset[str] = frozenset(
    {
        "Feature",
        "PFeature",
        "Mask",
        "P",
        "PRef",
        "ChangeInstrument",
        "TResample",
        "Rolling",
        "ElemOperator",
        "PairOperator",
        "PairRolling",
        "NpElemOperator",
        "NpPairOperator",
        "ExpressionOps",
        "Expression",
    }
)

_ROLLING_WINDOW = frozenset(
    {
        "Mean",
        "Sum",
        "Std",
        "Var",
        "Skew",
        "Kurt",
        "Max",
        "IdxMax",
        "Min",
        "IdxMin",
        "Med",
        "Mad",
        "Rank",
        "Count",
        "Delta",
        "Slope",
        "Rsquare",
        "Resi",
        "WMA",
    }
)
_PAIR_ROLLING_WINDOW = frozenset({"Corr", "Cov"})

# Operators whose first argument must be a series.  Qlib's element-wise and
# rolling operators call ``feature.load(...)``; a bare scalar makes that fail
# only at evaluation time (e.g. Log(2)), so reject it statically.
_SERIES_FIRST_OPS = frozenset(
    {
        "Abs",
        "Sign",
        "Log",
        "Not",
        "Ref",
        "Mean",
        "Sum",
        "Std",
        "Var",
        "Skew",
        "Kurt",
        "Max",
        "IdxMax",
        "Min",
        "IdxMin",
        "Quantile",
        "Med",
        "Mad",
        "Rank",
        "Count",
        "Delta",
        "Slope",
        "Rsquare",
        "Resi",
        "WMA",
        "EMA",
    }
)
_PAIR_SERIES_OPS = frozenset(
    {
        "Power",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Greater",
        "Less",
        "Gt",
        "Ge",
        "Lt",
        "Le",
        "Eq",
        "Ne",
        "And",
        "Or",
    }
)
_LOGICAL_SERIES_OPS = frozenset({"And", "Or"})
_BOOL_SERIES_OPS = frozenset({"Gt", "Ge", "Lt", "Le", "Eq", "Ne"})
_MIN_WINDOW: dict[str, int] = {
    "Std": 2,
    "Var": 2,
    "Skew": 3,
    "Kurt": 3,
    "Slope": 2,
    "Rsquare": 2,
    "Resi": 2,
}
_MIN_PAIR_WINDOW = 2

MAX_EXPRESSION_LENGTH = 2_000
MAX_COMPLEXITY = 200
MAX_DEPTH = 32
MAX_WINDOW = 1_000

_FIELD_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")
_PLACEHOLDER_PREFIX = "QLIBFIELD_"
_PLACEHOLDER_RE = re.compile(r"\bQLIBFIELD_([A-Za-z_][A-Za-z0-9_]*)\b")
_SLUG_RE = re.compile(r"[^A-Za-z0-9_]+")

_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_ALLOWED_UNARY_CONST = (ast.UAdd, ast.USub)
_ALLOWED_CMPOPS = (ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq)


class ExpressionError(ValueError):
    """Base class for all expression validation errors."""


class ExpressionSyntaxError(ExpressionError):
    """The expression is not valid Qlib/Python-like syntax."""


class UnsupportedFieldError(ExpressionError):
    """The expression references a field outside the allowed set."""


class UnsupportedOperatorError(ExpressionError):
    """The expression references an operator outside the allowed set."""


class LookAheadError(ExpressionError):
    """The expression references future data."""


class ComplexityError(ExpressionError):
    """The expression is too large or too deeply nested."""


class DegenerateFactorError(ExpressionError):
    """A computed factor is empty, constant or almost entirely missing."""


@dataclass(frozen=True)
class ValidatedExpression:
    """A parsed expression that passed all static checks."""

    raw: str
    canonical: str
    fields: tuple[str, ...]
    operators: tuple[str, ...]
    max_window: int
    complexity: int
    expression_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "canonical": self.canonical,
            "fields": list(self.fields),
            "operators": list(self.operators),
            "max_window": self.max_window,
            "complexity": self.complexity,
            "expression_hash": self.expression_hash,
        }


def slugify(value: str, max_length: int = 48) -> str:
    """Return a filesystem-safe slug for a factor name."""
    slug = _SLUG_RE.sub("_", value.strip()).strip("_").lower()
    if not slug:
        slug = "factor"
    return slug[:max_length].rstrip("_") or "factor"


def is_registered_operator(name: str) -> bool:
    """Return True if Qlib registers ``name`` after ``qlib.init``.

    The validator does not depend on a Qlib initialization; this helper is for
    tests and diagnostics only.
    """
    try:
        from qlib.data.ops import Operators

        return name in Operators._ops
    except Exception:
        return False


def validate_expression(
    expression: str,
    *,
    allowed_fields: Iterable[str] | None = None,
    max_window: int = MAX_WINDOW,
    max_complexity: int = MAX_COMPLEXITY,
    max_length: int = MAX_EXPRESSION_LENGTH,
) -> ValidatedExpression:
    """Validate one factor expression and return its canonical form.

    Raises an :class:`ExpressionError` subclass with a human-readable message
    when the expression is rejected.
    """
    if not isinstance(expression, str):
        raise ExpressionSyntaxError(f"expression must be a string, got {type(expression).__name__}")

    raw = expression.strip()
    if not raw:
        raise ExpressionSyntaxError("expression is empty")
    if len(raw) > max_length:
        raise ComplexityError(f"expression length {len(raw)} exceeds limit {max_length}")
    if "\x00" in raw:
        raise ExpressionSyntaxError("expression contains a NUL byte")
    if "__" in raw:
        raise ExpressionSyntaxError("dunder names are not allowed")
    if raw.count("$") != len(_FIELD_RE.findall(raw)):
        raise UnsupportedFieldError("every '$' must start a valid field name, e.g. $close")
    for blocked in BLOCKED_OPERATORS:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(blocked)}\s*\(", raw):
            raise UnsupportedOperatorError(f"operator '{blocked}' is deliberately blocked")

    fields = tuple(dict.fromkeys(match.group(1) for match in _FIELD_RE.finditer(raw)))
    if not fields:
        raise ExpressionSyntaxError("expression must reference at least one field, e.g. $close")
    allowed = _normalize_fields(DEFAULT_FIELDS if allowed_fields is None else allowed_fields)
    unknown = [name for name in fields if name not in allowed]
    if unknown:
        raise UnsupportedFieldError(
            f"unknown field(s): {', '.join('$' + name for name in unknown)}; "
            f"allowed: {', '.join('$' + name for name in sorted(allowed))}"
        )

    placeholders = {
        f"{_PLACEHOLDER_PREFIX}{name}": name for name in fields
    }
    transformed = _FIELD_RE.sub(lambda match: f"{_PLACEHOLDER_PREFIX}{match.group(1)}", raw)
    try:
        tree = ast.parse(transformed, mode="eval")
    except SyntaxError as exc:
        detail = f"{exc.msg} at column {exc.offset}" if exc.offset else str(exc.msg)
        raise ExpressionSyntaxError(f"invalid expression syntax: {detail}") from exc

    state = _ValidationState(
        placeholders=placeholders,
        window_limit=max_window,
        max_complexity=max_complexity,
    )
    state.visit(tree, depth=0)
    if state.complexity > max_complexity:
        raise ComplexityError(f"expression complexity {state.complexity} exceeds limit {max_complexity}")

    canonical = ast.unparse(tree)
    canonical = _PLACEHOLDER_RE.sub(lambda match: "$" + match.group(1), canonical)
    if "QLIBFIELD_" in canonical:
        raise ExpressionSyntaxError("internal placeholder was not restored")
    if raw.count("$") != canonical.count("$"):
        raise ExpressionSyntaxError("canonical form lost a field reference")

    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ValidatedExpression(
        raw=raw,
        canonical=canonical,
        fields=fields,
        operators=tuple(sorted(state.operators)),
        max_window=state.max_window,
        complexity=state.complexity,
        expression_hash=digest,
    )


def _normalize_fields(fields: Iterable[str]) -> frozenset[str]:
    out = set()
    for item in fields:
        name = str(item).strip().removeprefix("$")
        if not _FIELD_RE.fullmatch("$" + name):
            raise UnsupportedFieldError(f"invalid allowed field name: {item!r}")
        out.add(name)
    return frozenset(out)


@dataclass
class _ValidationState:
    placeholders: dict[str, str]
    window_limit: int
    max_complexity: int
    operators: set[str] = field(default_factory=set)
    complexity: int = 0
    max_window: int = 0

    def visit(self, node: ast.AST, *, depth: int) -> None:
        self.complexity += 1
        if self.complexity > self.max_complexity:
            raise ComplexityError(f"expression complexity exceeds limit {self.max_complexity}")
        if depth > MAX_DEPTH:
            raise ComplexityError(f"expression nesting exceeds limit {MAX_DEPTH}")

        if isinstance(node, ast.Expression):
            self.visit(node.body, depth=depth + 1)
            return

        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ExpressionSyntaxError("only numeric constants are allowed")
            if not math.isfinite(float(node.value)):
                raise ExpressionSyntaxError("non-finite numeric constants are not allowed")
            return

        if isinstance(node, ast.Name):
            if node.id not in self.placeholders:
                raise UnsupportedFieldError(
                    f"bare name '{node.id}' is not allowed; fields must use the '$' prefix"
                )
            return

        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _ALLOWED_BINOPS):
                raise ExpressionSyntaxError(f"operator '{type(node.op).__name__}' is not allowed")
            self.visit(node.left, depth=depth + 1)
            self.visit(node.right, depth=depth + 1)
            return

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _ALLOWED_UNARY_CONST):
                raise ExpressionSyntaxError(f"unary operator '{type(node.op).__name__}' is not allowed")
            # Qlib expressions only support unary signs on numeric literals.
            # ``-$close`` would be a Python-level negation of an Expression and
            # Qlib does not implement __neg__.
            if not isinstance(node.operand, ast.Constant) or isinstance(node.operand.value, bool):
                raise ExpressionSyntaxError("unary signs are only allowed on numeric literals")
            self.visit(node.operand, depth=depth + 1)
            return

        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                raise ExpressionSyntaxError("chained comparisons are not allowed")
            if not isinstance(node.ops[0], _ALLOWED_CMPOPS):
                raise ExpressionSyntaxError(f"comparison '{type(node.ops[0]).__name__}' is not allowed")
            self.visit(node.left, depth=depth + 1)
            self.visit(node.comparators[0], depth=depth + 1)
            return

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise UnsupportedOperatorError("only named Qlib operator calls are allowed")
            name = node.func.id
            if name in BLOCKED_OPERATORS:
                raise UnsupportedOperatorError(f"operator '{name}' is deliberately blocked")
            if name not in _OPERATOR_ARITY:
                raise UnsupportedOperatorError(
                    f"unknown operator '{name}'; allowed: {', '.join(sorted(ALLOWED_OPERATORS))}"
                )
            if node.keywords:
                raise UnsupportedOperatorError(f"operator '{name}' does not accept keyword arguments")
            if any(isinstance(arg, ast.Starred) for arg in node.args):
                raise UnsupportedOperatorError(f"operator '{name}' does not accept starred arguments")
            min_args, max_args = _OPERATOR_ARITY[name]
            if not min_args <= len(node.args) <= max_args:
                raise UnsupportedOperatorError(
                    f"operator '{name}' expects {min_args}"
                    + (f"-{max_args}" if max_args != min_args else "")
                    + f" arguments, got {len(node.args)}"
                )
            self.operators.add(name)
            for arg in node.args:
                self.visit(arg, depth=depth + 1)
            self._check_operator_args(name, node.args)
            return

        if isinstance(node, ast.BoolOp):
            raise ExpressionSyntaxError("use And(...)/Or(...) instead of Python 'and'/'or'")
        if isinstance(node, ast.IfExp):
            raise ExpressionSyntaxError("use If(condition, a, b) instead of a Python conditional expression")
        if isinstance(node, ast.Attribute):
            raise ExpressionSyntaxError("attribute access is not allowed")
        if isinstance(node, ast.Subscript):
            raise ExpressionSyntaxError("subscripts are not allowed")
        if isinstance(node, ast.Lambda):
            raise ExpressionSyntaxError("lambda expressions are not allowed")

        raise ExpressionSyntaxError(f"syntax element '{type(node).__name__}' is not allowed")

    def _check_operator_args(self, name: str, args: list[ast.expr]) -> None:
        if name in _SERIES_FIRST_OPS and not self._is_series(args[0]):
            raise ExpressionSyntaxError(
                f"{name} requires a series expression as its first argument, got a scalar constant"
            )
        if name == "If" and not self._is_bool_series(args[0]):
            raise ExpressionSyntaxError(
                "If requires a boolean series condition; use Gt/Ge/Lt/Le/Eq/Ne or >, not Greater/Less"
            )
        if name == "Not" and not self._is_bool_series(args[0]):
            raise ExpressionSyntaxError(
                "Not requires a boolean series; use Gt/Ge/Lt/Le/Eq/Ne or >, not Greater/Less"
            )
        if name in _PAIR_SERIES_OPS:
            left, right = args[0], args[1]
            if name in _LOGICAL_SERIES_OPS:
                if not (self._is_bool_series(left) and self._is_bool_series(right)):
                    raise ExpressionSyntaxError(
                        f"{name} requires two boolean series; use Gt/Ge/Lt/Le/Eq/Ne or >, "
                        "not Greater/Less"
                    )
            elif not (self._is_series(left) or self._is_series(right)):
                raise ExpressionSyntaxError(
                    f"{name} requires at least one series operand, got two scalar constants"
                )
        if name in _PAIR_ROLLING_WINDOW and not (
            self._is_series(args[0]) and self._is_series(args[1])
        ):
            raise ExpressionSyntaxError(f"{name} requires two series expressions")
        if name == "Ref":
            shift = self._int_literal(args[1], "Ref shift")
            if shift < 0:
                raise LookAheadError(
                    f"Ref($field, {shift}) reads future data; use a non-negative shift"
                )
            if shift == 0:
                raise ExpressionSyntaxError("Ref(..., 0) is ambiguous in Qlib; use $field directly")
            self.max_window = max(self.max_window, shift)
            return

        if name in _ROLLING_WINDOW:
            minimum = _MIN_WINDOW.get(name, 1)
            window = self._int_literal(args[1], f"{name} window")
            if window < minimum:
                raise ExpressionSyntaxError(f"{name} window must be >= {minimum}, got {window}")
            self.max_window = max(self.max_window, window)
            return

        if name in _PAIR_ROLLING_WINDOW:
            window = self._int_literal(args[2], f"{name} window")
            if window < _MIN_PAIR_WINDOW:
                raise ExpressionSyntaxError(
                    f"{name} window must be >= {_MIN_PAIR_WINDOW}, got {window}"
                )
            self.max_window = max(self.max_window, window)
            return

        if name == "Quantile":
            window = self._int_literal(args[1], "Quantile window")
            if window < 1:
                raise ExpressionSyntaxError(f"Quantile window must be >= 1, got {window}")
            self.max_window = max(self.max_window, window)
            if len(args) == 3:
                qscore = self._number_literal(args[2], "Quantile qscore")
                if not 0.0 <= qscore <= 1.0:
                    raise ExpressionSyntaxError(f"Quantile qscore must be in [0, 1], got {qscore}")
            return

        if name == "EMA":
            alpha = self._number_literal(args[1], "EMA alpha")
            if not 0.0 < alpha <= 1.0:
                raise ExpressionSyntaxError(f"EMA alpha must be in (0, 1], got {alpha}")
            return

    def _is_series(self, node: ast.expr) -> bool:
        """Best-effort static check: does this subtree produce a Qlib series?"""
        if isinstance(node, ast.Name):
            return True
        if isinstance(node, ast.Constant):
            return False
        if isinstance(node, ast.BinOp):
            return self._is_series(node.left) or self._is_series(node.right)
        if isinstance(node, ast.UnaryOp):
            return self._is_series(node.operand)
        if isinstance(node, ast.Compare):
            return self._is_series(node.left) or any(
                self._is_series(item) for item in node.comparators
            )
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else ""
            if name in _SERIES_FIRST_OPS:
                return bool(node.args) and self._is_series(node.args[0])
            if name in _PAIR_SERIES_OPS:
                return len(node.args) >= 2 and any(
                    self._is_series(item) for item in node.args[:2]
                )
            if name in _PAIR_ROLLING_WINDOW:
                return len(node.args) >= 2 and any(
                    self._is_series(item) for item in node.args[:2]
                )
            if name == "If":
                return bool(node.args) and self._is_bool_series(node.args[0])
        return False

    def _is_bool_series(self, node: ast.expr) -> bool:
        """Best-effort static check: does this subtree produce a boolean series?"""
        if isinstance(node, ast.Compare):
            return self._is_series(node.left) or any(
                self._is_series(item) for item in node.comparators
            )
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else ""
            if name in _BOOL_SERIES_OPS:
                return len(node.args) >= 2 and any(
                    self._is_series(item) for item in node.args[:2]
                )
            if name == "Not":
                return bool(node.args) and self._is_bool_series(node.args[0])
            if name in _LOGICAL_SERIES_OPS:
                return len(node.args) >= 2 and (
                    self._is_bool_series(node.args[0])
                    and self._is_bool_series(node.args[1])
                )
        return False

    def _int_literal(self, node: ast.expr, label: str) -> int:
        value = self._signed_constant(node)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ExpressionSyntaxError(f"{label} must be an integer literal")
        if abs(value) > self.window_limit:
            raise ComplexityError(f"{label} {value} exceeds limit {self.window_limit}")
        return value

    def _number_literal(self, node: ast.expr, label: str) -> float:
        value = self._signed_constant(node)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ExpressionSyntaxError(f"{label} must be a numeric literal")
        return float(value)

    @staticmethod
    def _signed_constant(node: ast.expr) -> int | float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and not isinstance(node.operand.value, bool)
        ):
            value = node.operand.value
            if isinstance(value, (int, float)):
                return -value if isinstance(node.op, ast.USub) else value
        raise ExpressionSyntaxError("expected a numeric literal")
