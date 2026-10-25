#!/usr/bin/env python
"""Validate Qlib factor expressions without evaluating them.

Examples:
    .venv/Scripts/python.exe scripts/validate_expression.py --expr 'Mean($close,5)/$close-1'
    .venv/Scripts/python.exe scripts/validate_expression.py --alpha158
    .venv/Scripts/python.exe scripts/validate_expression.py --file expressions.txt --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.factors.expression import ExpressionError, validate_expression
from src.factors.io import ExpressionSpec, alpha158_specs, read_expression_file, specs_from_expressions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expr", action="append", default=[], help="expression to validate; repeatable")
    parser.add_argument("--file", action="append", default=[], help="CSV/JSON/JSONL/TXT expression file; repeatable")
    parser.add_argument("--alpha158", action="store_true", help="validate every Alpha158 expression")
    parser.add_argument(
        "--alpha158-limit",
        type=int,
        default=None,
        help="validate only an evenly-spaced sample of N Alpha158 expressions",
    )
    parser.add_argument("--fields", nargs="+", default=None, help="override the allowed field whitelist")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--no-fail", action="store_true", help="exit 0 even when some expressions are invalid")
    return parser.parse_args()


def collect_specs(args: argparse.Namespace) -> list[ExpressionSpec]:
    specs: list[ExpressionSpec] = []
    specs.extend(specs_from_expressions(args.expr))
    for path in args.file:
        specs.extend(read_expression_file(path))
    if args.alpha158:
        specs.extend(alpha158_specs())
    elif args.alpha158_limit is not None:
        specs.extend(alpha158_specs(limit=args.alpha158_limit))
    if not specs:
        raise SystemExit("no expressions supplied; use --expr, --file or --alpha158")
    return specs


def validate_specs(specs: list[ExpressionSpec], fields: list[str] | None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for spec in specs:
        try:
            validated = validate_expression(spec.expression, allowed_fields=fields)
        except ExpressionError as exc:
            results.append(
                {
                    "name": spec.name,
                    "expression": spec.expression,
                    "status": "invalid",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        except Exception as exc:
            results.append(
                {
                    "name": spec.name,
                    "expression": spec.expression,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        results.append({"name": spec.name, "status": "valid", **validated.to_dict()})
    return results


def main() -> int:
    args = parse_args()
    specs = collect_specs(args)
    results = validate_specs(specs, args.fields)
    n_valid = sum(1 for row in results if row["status"] == "valid")
    n_invalid = sum(1 for row in results if row["status"] == "invalid")
    n_error = sum(1 for row in results if row["status"] == "error")
    summary = {
        "n_total": len(results),
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "n_error": n_error,
    }

    if args.json:
        print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
    else:
        for row in results:
            if row["status"] == "valid":
                print(
                    f"{row['name']:<12} VALID   {row['expression_hash'][:12]}  "
                    f"w={row['max_window']:<4} c={row['complexity']:<3} {row['canonical']}"
                )
            else:
                print(f"{row['name']:<12} {row['status'].upper():<7} {row['error_type']}: {row['error']}")
        print(
            f"\nSummary: {n_valid} valid, {n_invalid} invalid, {n_error} internal errors "
            f"out of {len(results)}"
        )
    if args.no_fail:
        return 0
    return 0 if n_invalid == 0 and n_error == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
