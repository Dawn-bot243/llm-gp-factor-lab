#!/usr/bin/env python
"""Stage 4 S1 screen: standardize single-factor evaluation and scorecards.

Examples:
    # Calibrate against the three known factors
    .venv/Scripts/python.exe scripts/screen_factors.py --known

    # Screen a small Alpha158 sample
    .venv/Scripts/python.exe scripts/screen_factors.py --alpha158-sample 10
"""
from __future__ import annotations

import argparse
import dataclasses
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

import pandas as pd
import yaml

from src.factors.evaluator import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONSTRAINTS,
    DEFAULT_MEMBERSHIP,
    DEFAULT_SNAPSHOT,
    EvaluationConfig,
    load_factor,
    membership_row_count,
)
from src.factors.expression import ExpressionError, validate_expression
from src.factors.io import (
    ExpressionSpec,
    alpha158_specs,
    read_expression_file,
    specs_from_expressions,
)
from src.factors.s1 import (
    ForwardReturns,
    S1Config,
    S1Result,
    apply_fdr,
    evaluate_s1,
    scorecard_row,
)
from src.factors.serialization import json_default, jsonable
from src.factors.store import FactorStore

KNOWN_FACTORS: tuple[ExpressionSpec, ...] = (
    # Positive past return; in A-shares this is usually a reversal signal.
    ExpressionSpec("MOM20", "$close / Ref($close, 20) - 1"),
    # Positive 5-day return; a reversal factor has a negative IC.
    ExpressionSpec("REV5", "$close / Ref($close, 5) - 1"),
    # High value = low volatility.
    ExpressionSpec("LOWVOL20", "-1 * Std($close, 20) / $close"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--s1-config", default=str(PROJECT_ROOT / "configs" / "s1.yaml"))
    parser.add_argument("--expr", action="append", default=[], help="expression to screen; repeatable")
    parser.add_argument("--file", action="append", default=[], help="CSV/JSON/JSONL/TXT expression file")
    parser.add_argument("--alpha158-sample", type=int, default=None)
    parser.add_argument("--known", action="store_true", help="add MOM20, REV5, LOWVOL20")
    parser.add_argument("--name", default=None, help="name for a single expression")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts" / "s1"))
    parser.add_argument("--store-dir", default=str(PROJECT_ROOT / "data" / "factor_store"))
    parser.add_argument("--provider-uri", default=None)
    parser.add_argument("--instruments", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--membership", default=None)
    parser.add_argument("--constraints", default=None)
    parser.add_argument("--no-constraints", action="store_true")
    parser.add_argument("--snapshot-file", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--min-names", type=int, default=None)
    parser.add_argument("--decay-horizons", default=None, help="comma-separated override")
    parser.add_argument("--fields", nargs="+", default=None)
    parser.add_argument("--kernels", type=int, default=4)
    parser.add_argument("--no-store", action="store_true")
    return parser.parse_args()


def _load_yaml(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve(value: str | Path | None, default: Path) -> str:
    if value is None or str(value).strip() == "":
        return str(default)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _data_config(args: argparse.Namespace) -> EvaluationConfig:
    payload = _load_yaml(args.config).get("data", {})
    return EvaluationConfig(
        provider_uri=args.provider_uri or payload.get("provider_uri") or "",
        instruments=args.instruments or payload.get("instruments") or "csi300",
        start=args.start or payload.get("start_time") or "2010-01-01",
        end=args.end or payload.get("end_time") or "2020-09-23",
        membership_path=_resolve(args.membership or payload.get("pit_membership"), DEFAULT_MEMBERSHIP),
        constraints_path=None
        if args.no_constraints
        else _resolve(args.constraints, DEFAULT_CONSTRAINTS),
        snapshot_file=_resolve(args.snapshot_file, DEFAULT_SNAPSHOT),
        cache_dir=_resolve(args.cache_dir, DEFAULT_CACHE_DIR),
        use_cache=not args.no_cache,
        kernels=args.kernels,
    )


def _s1_config(args: argparse.Namespace) -> S1Config:
    payload = _load_yaml(args.s1_config)
    config = S1Config.from_dict(payload.get("s1", payload))
    updates: dict[str, Any] = {}
    if args.min_names is not None:
        updates["min_names"] = args.min_names
    if args.decay_horizons:
        updates["decay_horizons"] = tuple(
            sorted({int(part) for part in args.decay_horizons.replace(";", ",").split(",") if part.strip()})
        )
    return dataclasses.replace(config, **updates) if updates else config


def _collect_specs(args: argparse.Namespace) -> list[ExpressionSpec]:
    specs: list[ExpressionSpec] = []
    if args.known:
        specs.extend(KNOWN_FACTORS)
    if args.alpha158_sample is not None:
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
        specs = list(KNOWN_FACTORS)
    return specs


def _error_result(
    name: str,
    expression: str,
    exc: Exception,
    data_config: EvaluationConfig,
) -> S1Result:
    scorecard: dict[str, Any] = {
        "name": name,
        "expression": expression,
        "canonical": None,
        "expression_hash": None,
        "direction": "unknown",
        "status": "error",
        "reasons": [f"ERROR:{type(exc).__name__}"],
        "rank_ic_mean": None,
        "icir": None,
        "newey_west_t": None,
        "p_value": None,
        "q_value": None,
        "fdr_pass": None,
        "coverage": None,
        "turnover": None,
        "monotonicity": None,
        "year_sign_consistency": None,
        "holdout_rank_ic": None,
        "holdout_icir": None,
        "holdout_newey_west_t": None,
        "thresholds": {},
        "error": f"{type(exc).__name__}: {exc}",
    }
    metrics = {
        "schema_version": "stage4.s1.v1",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "name": name,
        "expression": expression,
        "error": scorecard["error"],
        "data": {"snapshot_id": data_config.snapshot_id},
        "scorecard": scorecard,
    }
    return S1Result(metrics=metrics, scorecard=scorecard, rank_ic=pd.Series(dtype=float, name="rank_ic"))


def main() -> int:
    args = parse_args()
    data_config = _data_config(args)
    if not data_config.provider_uri:
        raise SystemExit("provider_uri is empty; pass --provider-uri or set data.provider_uri")
    s1_config = _s1_config(args)
    specs = _collect_specs(args)

    print(f"loading forward returns for {data_config.start}..{data_config.end} ...")
    forward = ForwardReturns.from_config(data_config, s1_config.decay_horizons)
    print(f"factor coverage denominator: {membership_row_count(data_config):,} PIT rows")

    valid: list[S1Result] = []
    errors: list[S1Result] = []
    seen: dict[str, str] = {}
    for index, spec in enumerate(specs, start=1):
        try:
            validated = validate_expression(spec.expression, allowed_fields=args.fields)
        except ExpressionError as exc:
            errors.append(_error_result(spec.name, spec.expression, exc, data_config))
            print(f"[{index}/{len(specs)}] {spec.name}: invalid ({exc})")
            continue
        if validated.canonical in seen:
            print(f"[{index}/{len(specs)}] {spec.name}: duplicate of {seen[validated.canonical]}")
            continue
        seen[validated.canonical] = spec.name
        try:
            validated, factor = load_factor(spec.expression, data_config, allowed_fields=args.fields)
            result = evaluate_s1(
                spec.expression,
                data_config,
                s1_config,
                forward,
                name=spec.name,
                allowed_fields=args.fields,
                validated=validated,
                factor=factor,
            )
        except Exception as exc:
            errors.append(_error_result(spec.name, spec.expression, exc, data_config))
            print(f"[{index}/{len(specs)}] {spec.name}: error ({type(exc).__name__}: {exc})")
            continue
        valid.append(result)
        print(
            f"[{index}/{len(specs)}] {spec.name}: {result.scorecard['status']} "
            f"RankIC={result.scorecard['rank_ic_mean']:.4f} "
            f"ICIR={result.scorecard['icir']:.3f}"
        )

    apply_fdr(valid, alpha=s1_config.thresholds.fdr_alpha)
    # Re-sort after FDR updates.
    valid.sort(
        key=lambda result: (
            {"pass": 0, "review": 1, "fail": 2}.get(result.scorecard["status"], 3),
            -abs(result.scorecard.get("rank_ic_mean") or 0.0),
        )
    )
    all_results = valid + errors

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard = pd.DataFrame([scorecard_row(result) for result in all_results])
    scorecard.to_csv(output_dir / "s1_scorecard.csv", index=False)
    summary = {
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "n_requested": len(specs),
        "n_valid": len(valid),
        "n_error": len(errors),
        "n_pass": sum(1 for result in valid if result.scorecard["status"] == "pass"),
        "n_review": sum(1 for result in valid if result.scorecard["status"] == "review"),
        "n_fail": sum(1 for result in valid if result.scorecard["status"] == "fail"),
        "s1_config": s1_config.to_dict(),
        "data_config": data_config.to_dict(),
    }
    (output_dir / "s1_summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    store_paths: dict[str, str] = {}
    if not args.no_store:
        store_paths = FactorStore(Path(args.store_dir)).write(
            all_results,
            metadata={
                "s1_config": s1_config.to_dict(),
                "data_config": data_config.to_dict(),
                "n_valid": len(valid),
                "n_error": len(errors),
            },
        )

    columns = [
        "name",
        "status",
        "rank_ic_mean",
        "icir",
        "newey_west_t",
        "q_value",
        "coverage",
        "turnover",
        "monotonicity",
        "year_sign_consistency",
    ]
    if not scorecard.empty:
        print("\n" + scorecard[columns].to_string(index=False))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "store": store_paths,
                "n_valid": len(valid),
                "n_error": len(errors),
                "n_pass": summary["n_pass"],
                "n_review": summary["n_review"],
                "n_fail": summary["n_fail"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
