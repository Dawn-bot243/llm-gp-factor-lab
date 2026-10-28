#!/usr/bin/env python
"""Stage 6: run typed-expression GP evolution with a fixed fitness budget.

Examples:
    .venv/Scripts/python.exe scripts/run_gp.py --budget 200 --population 30 --generations 5
    .venv/Scripts/python.exe scripts/run_gp.py --budget 2000
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

from src.factors.evaluator import DEFAULT_MEMBERSHIP
from src.factors.serialization import json_default, jsonable
from src.mining.gp_eval import GPEvalConfig, GPEvaluator
from src.mining.gp_runner import GPConfig, GPRunner, write_gp_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "gp.yaml"))
    parser.add_argument("--baseline-config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--seed-file", default=str(PROJECT_ROOT / "data" / "mining" / "stage5_combined_scorecard.csv"))
    parser.add_argument("--no-seeds", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    return parser.parse_args()


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _seed_expressions(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    frame = pd.read_csv(p, keep_default_na=False)
    if "status" in frame.columns:
        frame = frame[frame["status"] == "pass"]
    if "canonical" not in frame.columns:
        return []
    if "rank_ic_mean" in frame.columns:
        frame = frame.assign(_abs_ic=pd.to_numeric(frame["rank_ic_mean"], errors="coerce").abs())
        frame = frame.sort_values("_abs_ic", ascending=False, na_position="last")
    return [str(value) for value in frame["canonical"].dropna().tolist()]


def main() -> int:
    args = parse_args()
    payload = _load_yaml(args.config)
    gp_payload = payload.get("gp", payload)
    data_payload = payload.get("data", {})
    fitness_payload = payload.get("fitness", {})
    baseline = _load_yaml(args.baseline_config).get("data", {})

    gp_config = GPConfig.from_dict(gp_payload)
    updates: dict[str, Any] = {}
    if args.population is not None:
        updates["population_size"] = args.population
    if args.generations is not None:
        updates["generations"] = args.generations
    if args.budget is not None:
        updates["budget"] = args.budget
    if args.seed is not None:
        updates["seed"] = args.seed
    if updates:
        gp_config = dataclasses.replace(gp_config, **updates)

    provider = baseline.get("provider_uri")
    if not provider:
        raise SystemExit("provider_uri is not configured in the baseline YAML")
    membership = baseline.get("pit_membership") or str(DEFAULT_MEMBERSHIP)
    if not Path(membership).is_absolute():
        membership = str(PROJECT_ROOT / membership)
    sample_size = args.sample_size or int(data_payload.get("sample_size", 100))
    eval_config = GPEvalConfig(
        provider_uri=provider,
        membership_path=membership,
        sample_date=str(data_payload.get("sample_date", "2017-06-30")),
        sample_size=sample_size,
        start=args.start or data_payload.get("start", "2017-01-01"),
        end=args.end or data_payload.get("end", "2018-12-31"),
        max_evaluations=gp_config.budget,
        min_names=int(fitness_payload.get("min_names", 20)),
        min_dates=int(fitness_payload.get("min_dates", 300)),
        quantile=int(fitness_payload.get("quantile", 5)),
        ic_weight=float(fitness_payload.get("ic_weight", 1.0)),
        icir_weight=float(fitness_payload.get("icir_weight", 0.5)),
        turnover_penalty=float(fitness_payload.get("turnover_penalty", 0.05)),
        complexity_penalty=float(fitness_payload.get("complexity_penalty", 0.0005)),
    )
    output_dir = Path(
        args.output_dir
        or PROJECT_ROOT / "data" / "mining" / datetime.now(UTC).astimezone().strftime("gp_run_%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = GPEvaluator(eval_config)
    seeds = [] if args.no_seeds else _seed_expressions(args.seed_file)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "gp": gp_config.to_dict(),
                "sample_names": len(evaluator.names),
                "seed_expressions": len(seeds),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    runner = GPRunner(gp_config, evaluator, seed_expressions=seeds)
    result = runner.run()
    paths = write_gp_outputs(result, output_dir, runner.trials)
    print(json.dumps(jsonable({**result.summary, "paths": paths}), ensure_ascii=False, indent=2, default=json_default))
    return 0 if result.hall_of_fame else 1


if __name__ == "__main__":
    raise SystemExit(main())
