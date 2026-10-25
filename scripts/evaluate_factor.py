#!/usr/bin/env python
"""Evaluate one or more Qlib factor expressions on the PIT CSI300 panel.

Examples:
    # Single factor
    .venv/Scripts/python.exe scripts/evaluate_factor.py \
      --expr 'Mean($close, 5)/$close - 1' --name ma5

    # A small Alpha158 smoke sample
    .venv/Scripts/python.exe scripts/evaluate_factor.py --alpha158-sample 5

    # Batch from a CSV with columns name,expression
    .venv/Scripts/python.exe scripts/evaluate_factor.py --file factors.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.factors.evaluator import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONSTRAINTS,
    DEFAULT_MEMBERSHIP,
    DEFAULT_SNAPSHOT,
    EvaluationConfig,
    evaluate_batch,
    evaluate_expression,
    write_result,
)
from src.factors.expression import ExpressionError, slugify, validate_expression
from src.factors.io import (
    ExpressionSpec,
    alpha158_specs,
    read_expression_file,
    specs_from_expressions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--expr", action="append", default=[], help="expression to evaluate; repeatable")
    parser.add_argument("--file", action="append", default=[], help="CSV/JSON/JSONL/TXT expression file")
    parser.add_argument("--alpha158", action="store_true", help="evaluate all 158 Alpha158 expressions")
    parser.add_argument("--alpha158-sample", type=int, default=None, help="evaluate an evenly-spaced sample")
    parser.add_argument("--name", default=None, help="name for a single expression")
    parser.add_argument("--limit", type=int, default=None, help="limit the number of expressions")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--provider-uri", default=None)
    parser.add_argument("--instruments", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--membership", default=None, help="PIT membership parquet")
    parser.add_argument("--constraints", default=None, help="PIT constraints parquet")
    parser.add_argument("--no-constraints", action="store_true")
    parser.add_argument("--snapshot-file", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--horizons", default="1,2,5,10,20", help="comma-separated forward horizons")
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--min-names", type=int, default=20)
    parser.add_argument("--min-coverage", type=float, default=0.10)
    parser.add_argument("--exclude-suspended", action="store_true")
    parser.add_argument("--exclude-at-limit", action="store_true")
    parser.add_argument("--fields", nargs="+", default=None, help="override the allowed field whitelist")
    parser.add_argument("--kernels", type=int, default=4)
    parser.add_argument("--save-panel", action="store_true", help="write the joined factor/label panel")
    parser.add_argument("--save-series", action="store_true", help="write per-factor series in batch mode")
    return parser.parse_args()


def _load_defaults(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    data = payload.get("data", {})
    return {
        "provider_uri": data.get("provider_uri"),
        "instruments": data.get("instruments", "csi300"),
        "start": data.get("start_time", "2010-01-01"),
        "end": data.get("end_time", "2020-09-23"),
        "membership": data.get("pit_membership"),
    }


def _resolve(value: str | Path | None, default: Path) -> str:
    if value is None or str(value).strip() == "":
        return str(default)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _parse_horizons(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    if not parts:
        raise SystemExit("--horizons cannot be empty")
    horizons = sorted({int(part) for part in parts})
    if any(horizon < 1 for horizon in horizons):
        raise SystemExit("--horizons values must be >= 1")
    return tuple(horizons)


def _collect_specs(args: argparse.Namespace) -> list[ExpressionSpec]:
    specs: list[ExpressionSpec] = []
    if args.alpha158 and args.alpha158_sample is not None:
        raise SystemExit("use either --alpha158 or --alpha158-sample, not both")
    if args.alpha158:
        specs.extend(alpha158_specs())
    elif args.alpha158_sample is not None:
        specs.extend(alpha158_specs(limit=args.alpha158_sample))
    for path in args.file:
        specs.extend(read_expression_file(path))
    specs.extend(specs_from_expressions(args.expr))
    if args.name and len(specs) == 1:
        specs[0] = ExpressionSpec(name=args.name, expression=specs[0].expression)
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        specs = specs[: args.limit]
    if not specs:
        raise SystemExit("no expressions supplied; use --expr, --file or --alpha158-sample")
    return specs


def _build_config(args: argparse.Namespace, defaults: dict[str, Any]) -> EvaluationConfig:
    membership = _resolve(args.membership or defaults.get("membership"), DEFAULT_MEMBERSHIP)
    constraints = None if args.no_constraints else _resolve(
        args.constraints, DEFAULT_CONSTRAINTS
    )
    return EvaluationConfig(
        provider_uri=args.provider_uri or defaults.get("provider_uri") or "",
        instruments=args.instruments or defaults.get("instruments") or "csi300",
        start=args.start or defaults.get("start") or "2010-01-01",
        end=args.end or defaults.get("end") or "2020-09-23",
        membership_path=membership,
        constraints_path=constraints,
        snapshot_file=_resolve(args.snapshot_file, DEFAULT_SNAPSHOT),
        horizons=_parse_horizons(args.horizons),
        quantiles=args.quantiles,
        min_names=args.min_names,
        min_coverage=args.min_coverage,
        exclude_suspended=args.exclude_suspended,
        exclude_at_limit=args.exclude_at_limit,
        cache_dir=_resolve(args.cache_dir, DEFAULT_CACHE_DIR),
        use_cache=not args.no_cache,
        kernels=args.kernels,
    )


def main() -> int:
    args = parse_args()
    defaults = _load_defaults(args.config)
    config = _build_config(args, defaults)
    if not config.provider_uri:
        raise SystemExit("provider_uri is empty; pass --provider-uri or set data.provider_uri in the config")
    specs = _collect_specs(args)

    if len(specs) == 1:
        spec = specs[0]
        try:
            validated = validate_expression(spec.expression, allowed_fields=args.fields)
        except ExpressionError as exc:
            print(f"invalid expression: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        out_dir = (
            Path(args.output_dir)
            if args.output_dir
            else PROJECT_ROOT / "artifacts" / "factor_eval" / f"{slugify(spec.name)}_{validated.expression_hash[:8]}"
        )
        try:
            result = evaluate_expression(
                spec.expression,
                config,
                name=spec.name,
                allowed_fields=args.fields,
            )
        except Exception as exc:
            print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        write_result(result, out_dir, save_panel=args.save_panel, save_series=True)
        metrics = result.metrics
        ic = metrics["ic_decay"][0]
        print(
            json.dumps(
                {
                    "name": spec.name,
                    "canonical": validated.canonical,
                    "expression_hash": validated.expression_hash,
                    "output_dir": str(out_dir),
                    "rank_ic": ic.get("mean"),
                    "icir": ic.get("icir"),
                    "newey_west_t": ic.get("newey_west_t"),
                    "coverage": metrics["data"]["factor_coverage"]["coverage"],
                    "turnover": metrics["turnover"]["top_quantile_one_way"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT
        / "artifacts"
        / "factor_eval"
        / f"batch_{datetime.now(UTC).astimezone().strftime('%Y%m%d_%H%M%S')}"
    )
    summary = evaluate_batch(
        [(spec.name, spec.expression) for spec in specs],
        config,
        out_dir,
        save_series=args.save_series,
        allowed_fields=args.fields,
    )
    print(
        json.dumps(
            {
                "output_dir": str(out_dir),
                "n_ok": summary["n_ok"],
                "n_error": summary["n_error"],
                "n_duplicate": summary["n_duplicate"],
                "leaderboard": str(out_dir / "leaderboard.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if summary["n_ok"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
