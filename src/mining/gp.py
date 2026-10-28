"""Typed expression-tree genetic programming for Stage 6.

The tree is deliberately small and typed: numeric series, boolean series and
numeric scalars.  Every generated tree is converted to a Qlib expression and
passed through the Stage 3 Validator before it can enter the population.
"""
from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass

from src.factors.expression import ExpressionError, validate_expression

FIELD_PREFIX = "GPF_"
_FIELD_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)")

_GP_FIELDS = ("close", "open", "high", "low", "volume", "vwap", "amount")
_ROLLING = (
    "Mean",
    "Sum",
    "Std",
    "Var",
    "Skew",
    "Kurt",
    "Max",
    "Min",
    "Med",
    "Mad",
    "Ref",
    "Rank",
    "Count",
    "Delta",
    "Slope",
    "Rsquare",
    "Resi",
    "WMA",
)
_WINDOWS = (5, 10, 15, 20, 30, 40, 60, 90, 120)
_UNARY = ("Abs", "Sign", "Log")
_BINARY = ("Add", "Sub", "Mul", "Div", "Power", "Greater", "Less")
_COMPARE = ("Gt", "Ge", "Lt", "Le", "Eq", "Ne")
_BOOL_BINARY = ("And", "Or")


@dataclass(frozen=True)
class GPNode:
    """One typed subtree.

    ``kind`` is ``series`` (numeric), ``bool`` (boolean series) or ``scalar``.
    """

    op: str
    args: tuple[GPNode, ...] = ()
    field: str | None = None
    const: float | None = None
    window: int | None = None
    alpha: float | None = None
    qscore: float | None = None
    kind: str = "series"

    @property
    def size(self) -> int:
        return 1 + sum(arg.size for arg in self.args)

    @property
    def depth(self) -> int:
        return 1 + max((arg.depth for arg in self.args), default=0)

    def to_expression(self) -> str:
        if self.op == "field":
            return "$" + str(self.field)
        if self.op == "const":
            value = float(self.const or 0.0)
            return f"{value:.10g}"
        if self.op in _ROLLING:
            return f"{self.op}({self.args[0].to_expression()}, {self.window})"
        if self.op == "Quantile":
            return (
                f"Quantile({self.args[0].to_expression()}, {self.window}, "
                f"{float(self.qscore or 0.5):.6g})"
            )
        if self.op == "EMA":
            return f"EMA({self.args[0].to_expression()}, {float(self.alpha or 0.5):.6g})"
        if self.op in {"Corr", "Cov"}:
            return (
                f"{self.op}({self.args[0].to_expression()}, "
                f"{self.args[1].to_expression()}, {self.window})"
            )
        if self.op == "If":
            return (
                f"If({self.args[0].to_expression()}, "
                f"{self.args[1].to_expression()}, {self.args[2].to_expression()})"
            )
        if self.op == "Not":
            return f"Not({self.args[0].to_expression()})"
        return f"{self.op}(" + ", ".join(arg.to_expression() for arg in self.args) + ")"

    def canonical(self) -> str:
        return validate_expression(self.to_expression()).canonical

    def paths(self, kind: str | None = None) -> list[tuple[int, ...]]:
        out: list[tuple[int, ...]] = []
        if kind is None or self.kind == kind:
            out.append(())
        for index, arg in enumerate(self.args):
            for sub in arg.paths(kind):
                out.append((index, *sub))
        return out

    def at(self, path: tuple[int, ...]) -> GPNode:
        node = self
        for index in path:
            node = node.args[index]
        return node

    def replace(self, path: tuple[int, ...], replacement: GPNode) -> GPNode:
        if not path:
            return replacement
        index, *rest = path
        args = list(self.args)
        args[index] = args[index].replace(tuple(rest), replacement)
        return GPNode(
            op=self.op,
            args=tuple(args),
            field=self.field,
            const=self.const,
            window=self.window,
            alpha=self.alpha,
            qscore=self.qscore,
            kind=self.kind,
        )


def parse_seed_expression(expression: str) -> GPNode:
    """Parse a validated Qlib expression into a typed GP tree."""
    validated = validate_expression(expression)
    transformed = _FIELD_RE.sub(lambda match: FIELD_PREFIX + match.group(1), validated.canonical)
    tree = ast.parse(transformed, mode="eval")
    return _from_ast(tree.body)


def _from_ast(node: ast.expr) -> GPNode:
    if isinstance(node, ast.Name):
        if not node.id.startswith(FIELD_PREFIX):
            raise ValueError(f"unknown name: {node.id}")
        return GPNode(op="field", field=node.id[len(FIELD_PREFIX) :], kind="series")
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise TypeError("only numeric constants are supported")
        return GPNode(op="const", const=float(node.value), kind="scalar")
    if isinstance(node, ast.BinOp):
        op = {
            ast.Add: "Add",
            ast.Sub: "Sub",
            ast.Mult: "Mul",
            ast.Div: "Div",
            ast.Pow: "Power",
        }.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported binary op: {type(node.op).__name__}")
        left, right = _from_ast(node.left), _from_ast(node.right)
        kind = "scalar" if left.kind == right.kind == "scalar" else "series"
        return GPNode(op=op, args=(left, right), kind=kind)
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
            value = float(node.operand.value)
            if isinstance(node.op, ast.USub):
                value = -value
            return GPNode(op="const", const=value, kind="scalar")
        if isinstance(node.op, ast.USub):
            return GPNode(
                op="Mul",
                args=(GPNode(op="const", const=-1.0, kind="scalar"), _from_ast(node.operand)),
                kind="series",
            )
        raise ValueError("unsupported unary expression")
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise ValueError("chained comparisons unsupported")
        op = {
            ast.Gt: "Gt",
            ast.GtE: "Ge",
            ast.Lt: "Lt",
            ast.LtE: "Le",
            ast.Eq: "Eq",
            ast.NotEq: "Ne",
        }.get(type(node.ops[0]))
        if op is None:
            raise ValueError("unsupported comparison")
        return GPNode(op=op, args=(_from_ast(node.left), _from_ast(node.comparators[0])), kind="bool")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        args = [_from_ast(arg) for arg in node.args]
        if name == "If":
            return GPNode(op=name, args=tuple(args), kind="series")
        if name in _ROLLING:
            return GPNode(op=name, args=(args[0],), window=_int_const(node.args[1]), kind="series")
        if name == "Quantile":
            return GPNode(
                op=name,
                args=(args[0],),
                window=_int_const(node.args[1]),
                qscore=_float_const(node.args[2]) if len(node.args) > 2 else 0.5,
                kind="series",
            )
        if name == "EMA":
            return GPNode(op=name, args=(args[0],), alpha=_float_const(node.args[1]), kind="series")
        if name in {"Corr", "Cov"}:
            return GPNode(op=name, args=(args[0], args[1]), window=_int_const(node.args[2]), kind="series")
        if name == "Not":
            return GPNode(op=name, args=(args[0],), kind="bool")
        if name in _BOOL_BINARY:
            return GPNode(op=name, args=(args[0], args[1]), kind="bool")
        if name in _COMPARE:
            return GPNode(op=name, args=(args[0], args[1]), kind="bool")
        if name in _UNARY:
            return GPNode(op=name, args=(args[0],), kind="series")
        if name in _BINARY:
            kind = "scalar" if all(arg.kind == "scalar" for arg in args) else "series"
            return GPNode(op=name, args=tuple(args), kind=kind)
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _int_const(node: ast.expr) -> int:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    raise ValueError("expected integer constant")


def _float_const(node: ast.expr) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    raise ValueError("expected numeric constant")


def random_series(rng: random.Random, depth: int = 4, max_nodes: int = 30) -> GPNode:
    if depth <= 0 or rng.random() < 0.25:
        return GPNode(op="field", field=rng.choice(_GP_FIELDS), kind="series")
    choice = rng.choice(("unary", "binary", "rolling", "pair_rolling", "if"))
    if choice == "unary":
        return GPNode(op=rng.choice(_UNARY), args=(random_series(rng, depth - 1, max_nodes),), kind="series")
    if choice == "binary":
        return GPNode(
            op=rng.choice(_BINARY),
            args=(random_series(rng, depth - 1, max_nodes), random_series(rng, depth - 1, max_nodes)),
            kind="series",
        )
    if choice == "rolling":
        return GPNode(
            op=rng.choice(_ROLLING),
            args=(random_series(rng, depth - 1, max_nodes),),
            window=rng.choice(_WINDOWS),
            kind="series",
        )
    if choice == "pair_rolling":
        return GPNode(
            op=rng.choice(("Corr", "Cov")),
            args=(random_series(rng, depth - 1, max_nodes), random_series(rng, depth - 1, max_nodes)),
            window=rng.choice(_WINDOWS),
            kind="series",
        )
    return GPNode(
        op="If",
        args=(
            random_bool(rng, max(0, depth - 1)),
            random_series(rng, max(0, depth - 1), max_nodes),
            random_series(rng, max(0, depth - 1), max_nodes),
        ),
        kind="series",
    )


def random_bool(rng: random.Random, depth: int = 2) -> GPNode:
    if depth <= 0 or rng.random() < 0.6:
        return GPNode(
            op=rng.choice(_COMPARE),
            args=(random_series(rng, 0), random_series(rng, 0)),
            kind="bool",
        )
    if rng.random() < 0.5:
        return GPNode(
            op=rng.choice(_BOOL_BINARY),
            args=(random_bool(rng, depth - 1), random_bool(rng, depth - 1)),
            kind="bool",
        )
    return GPNode(op="Not", args=(random_bool(rng, depth - 1),), kind="bool")


def random_node(rng: random.Random, kind: str, depth: int = 4, max_nodes: int = 30) -> GPNode:
    return random_bool(rng, depth) if kind == "bool" else random_series(rng, depth, max_nodes)


def mutate(node: GPNode, rng: random.Random, max_depth: int = 6, max_nodes: int = 30) -> GPNode:
    path = rng.choice(node.paths())
    target = node.at(path)
    replacement = random_node(rng, target.kind, max_depth, max_nodes)
    return node.replace(path, replacement)


def crossover(left: GPNode, right: GPNode, rng: random.Random, max_depth: int = 6, max_nodes: int = 30) -> GPNode:
    left_paths = left.paths("series") or [()]
    right_paths = right.paths("series") or [()]
    subtree = right.at(rng.choice(right_paths))
    return left.replace(rng.choice(left_paths), subtree)


def validate_tree(node: GPNode, max_depth: int = 6, max_nodes: int = 30) -> tuple[str, GPNode]:
    if node.depth > max_depth or node.size > max_nodes:
        raise ExpressionError(f"tree exceeds depth/nodes limits ({node.depth}/{node.size})")
    expression = node.to_expression()
    validated = validate_expression(expression)
    return validated.canonical, node
