"""Small helpers for reading factor-expression lists from disk."""
from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExpressionSpec:
    """A named expression supplied by a CLI, file or Alpha158 config."""

    name: str
    expression: str


def specs_from_expressions(expressions: Iterable[str]) -> list[ExpressionSpec]:
    return [
        ExpressionSpec(name=f"expr_{i}", expression=expr)
        for i, expr in enumerate(expressions, start=1)
    ]


def read_expression_file(path: str | Path) -> list[ExpressionSpec]:
    """Read CSV, JSON/JSONL or plain-text expression lists.

    Plain text supports one expression per line, optionally as
    ``name<TAB>expression``.  CSV must have an ``expression`` column and may
    have a ``name`` column.  JSON may be a list of objects or one object per
    line.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"expression file not found: {file_path}")
    suffix = file_path.suffix.lower()

    if suffix in {".json", ".jsonl", ".ndjson"}:
        return _read_json(file_path)
    if suffix in {".csv", ".tsv"}:
        return _read_csv(file_path, delimiter="\t" if suffix == ".tsv" else ",")
    return _read_text(file_path)


def alpha158_specs(limit: int | None = None) -> list[ExpressionSpec]:
    """Return the Alpha158 feature expressions as named specs.

    Importing Qlib is intentionally delayed so the validator tests do not need
    a Qlib data provider.
    """
    from qlib.contrib.data.loader import Alpha158DL

    conf = {
        "kbar": {},
        "price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
        "rolling": {},
    }
    expressions, names = Alpha158DL.get_feature_config(conf)
    specs = [ExpressionSpec(name=str(name), expression=str(expr)) for name, expr in zip(names, expressions)]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if limit < len(specs):
            # Evenly spaced sample keeps KBar/price/rolling groups represented.
            step = max(1, len(specs) // limit)
            specs = specs[::step][:limit]
    return specs


def _read_json(path: Path) -> list[ExpressionSpec]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        rows = payload if isinstance(payload, list) else [payload]
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    specs: list[ExpressionSpec] = []
    for i, row in enumerate(rows, start=1):
        if isinstance(row, str):
            specs.append(ExpressionSpec(name=f"expr_{i}", expression=row))
            continue
        if not isinstance(row, dict) or "expression" not in row:
            raise ValueError(f"each JSON object needs an 'expression' key: {row!r}")
        specs.append(
            ExpressionSpec(
                name=str(row.get("name") or f"expr_{i}"),
                expression=str(row["expression"]),
            )
        )
    return specs


def _read_csv(path: Path, delimiter: str) -> list[ExpressionSpec]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None or "expression" not in reader.fieldnames:
            raise ValueError(f"{path} needs an 'expression' column")
        specs: list[ExpressionSpec] = []
        for i, row in enumerate(reader, start=1):
            expression = (row.get("expression") or "").strip()
            if not expression:
                continue
            specs.append(
                ExpressionSpec(
                    name=(row.get("name") or f"expr_{i}").strip(),
                    expression=expression,
                )
            )
    return specs


def _read_text(path: Path) -> list[ExpressionSpec]:
    specs: list[ExpressionSpec] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            name, expression = line.split("\t", 1)
            specs.append(ExpressionSpec(name=name.strip() or f"expr_{i}", expression=expression.strip()))
        else:
            specs.append(ExpressionSpec(name=f"expr_{i}", expression=line))
    return specs
