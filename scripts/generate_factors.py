#!/usr/bin/env python
"""Stage 5: generate factor candidates with an LLM, validate and screen them.

Examples:
    # Offline pipeline smoke
    .venv/Scripts/python.exe scripts/generate_factors.py --mock --n 20 --no-screen

    # Live generation with the configured Anthropic-compatible endpoint
    .venv/Scripts/python.exe scripts/generate_factors.py --n 200

    # Resume an existing run directory
    .venv/Scripts/python.exe scripts/generate_factors.py --run-dir data/mining/run_...
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
)
from src.factors.expression import validate_expression
from src.factors.io import alpha158_specs
from src.factors.s1 import S1Config, scorecard_row
from src.factors.serialization import json_default, jsonable
from src.mining.generate import CATEGORIES, generate_candidates
from src.mining.ledger import CandidateRecord, TrialLedger
from src.mining.llm_client import LLMConfig, make_client
from src.mining.prompts import PROMPT_VERSION, prompt_hash
from src.mining.screen import screen_records, screening_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "llm.yaml"))
    parser.add_argument("--baseline-config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--s1-config", default=str(PROJECT_ROOT / "configs" / "s1.yaml"))
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--rpm", type=float, default=None)
    parser.add_argument("--max-cost", type=float, default=None)
    parser.add_argument("--max-calls", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed-factor-store", default=str(PROJECT_ROOT / "data" / "factor_store" / "s1_scorecard.csv"))
    parser.add_argument("--no-seed-store", action="store_true")
    parser.add_argument("--seed-alpha158", action="store_true", help="avoid re-discovering Alpha158 expressions")
    parser.add_argument("--mock", action="store_true", help="use the deterministic offline client")
    parser.add_argument("--no-screen", action="store_true")
    parser.add_argument("--screen-only", action="store_true", help="screen an existing run directory")
    parser.add_argument("--screen-limit", type=int, default=None)
    parser.add_argument("--screen-data-start", default=None)
    parser.add_argument("--screen-data-end", default=None)
    parser.add_argument("--calibration-start", default=None)
    parser.add_argument("--calibration-end", default=None)
    parser.add_argument("--holdout-start", default=None)
    parser.add_argument("--holdout-end", default=None)
    parser.add_argument("--decay-horizons", default=None)
    parser.add_argument("--kernels", type=int, default=4)
    return parser.parse_args()


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve(value: str | Path | None, default: Path) -> str:
    if value is None or str(value).strip() == "":
        return str(default)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


def _build_llm_config(args: argparse.Namespace, payload: dict[str, Any]) -> LLMConfig:
    config = LLMConfig.from_dict(payload)
    overrides = {
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "api_key_env": args.api_key_env,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout,
        "max_retries": args.max_retries,
        "requests_per_minute": args.rpm,
        "max_cost_usd": args.max_cost,
        "max_calls": args.max_calls,
    }
    if args.mock:
        overrides["provider"] = "mock"
    updates = {key: value for key, value in overrides.items() if value is not None}
    return dataclasses.replace(config, **updates) if updates else config


def _build_data_config(args: argparse.Namespace, baseline: dict[str, Any], screening: dict[str, Any]) -> EvaluationConfig:
    data = baseline.get("data", {})
    return EvaluationConfig(
        provider_uri=data.get("provider_uri") or "",
        instruments=data.get("instruments", "csi300"),
        start=args.screen_data_start or screening.get("data_start") or data.get("start_time") or "2016-01-01",
        end=args.screen_data_end or screening.get("data_end") or data.get("end_time") or "2020-09-23",
        membership_path=_resolve(data.get("pit_membership"), DEFAULT_MEMBERSHIP),
        constraints_path=_resolve(None, DEFAULT_CONSTRAINTS),
        snapshot_file=_resolve(None, DEFAULT_SNAPSHOT),
        cache_dir=_resolve(None, DEFAULT_CACHE_DIR),
        kernels=args.kernels,
    )


def _build_s1_config(args: argparse.Namespace, payload: dict[str, Any], screening: dict[str, Any]) -> S1Config:
    config = S1Config.from_dict(payload.get("s1", payload))
    horizons = args.decay_horizons or screening.get("decay_horizons")
    updates: dict[str, Any] = {
        "calibration_start": args.calibration_start or screening.get("calibration_start", config.calibration_start),
        "calibration_end": args.calibration_end or screening.get("calibration_end", config.calibration_end),
        "holdout_start": args.holdout_start or screening.get("holdout_start", config.holdout_start),
        "holdout_end": args.holdout_end or screening.get("holdout_end", config.holdout_end),
        "min_names": int(screening.get("min_names", config.min_names)),
    }
    if horizons:
        updates["decay_horizons"] = tuple(int(h) for h in horizons)
    return dataclasses.replace(config, **updates)


def _records_frame(records: list[CandidateRecord]) -> pd.DataFrame:
    columns = list(CandidateRecord.__dataclass_fields__)
    return pd.DataFrame([record.to_dict() for record in records], columns=columns)


def _load_records(run_dir: Path) -> list[CandidateRecord]:
    path = run_dir / "candidates.parquet"
    if not path.exists():
        raise SystemExit(f"candidates.parquet not found in {run_dir}")
    frame = pd.read_parquet(path)
    fields = CandidateRecord.__dataclass_fields__
    records = []
    for row in frame.to_dict("records"):
        records.append(CandidateRecord(**{key: row.get(key) for key in fields}))
    return records


def _screen_only(args: argparse.Namespace) -> int:
    if not args.run_dir:
        raise SystemExit("--screen-only requires --run-dir")
    run_dir = Path(args.run_dir)
    records = _load_records(run_dir)
    baseline = _load_yaml(args.baseline_config)
    screening = _load_yaml(args.config).get("screening", {})
    data_config = _build_data_config(args, baseline, screening)
    s1_config = _build_s1_config(args, _load_yaml(args.s1_config), screening)
    print(f"screening {len(records)} records on {data_config.start}..{data_config.end} ...")
    results = screen_records(records, data_config, s1_config, max_candidates=args.screen_limit)
    ledger = TrialLedger(run_dir)
    ledger.write_screened(records)
    if results:
        scorecard = pd.DataFrame([scorecard_row(item) for item in results])
        scorecard.to_csv(run_dir / "s1_scorecard.csv", index=False)
        scorecard.to_parquet(run_dir / "s1_scorecard.parquet", index=False)
    frame = _records_frame(records)
    frame.to_csv(run_dir / "candidates.csv", index=False)
    frame.to_parquet(run_dir / "candidates.parquet", index=False)
    summary = {
        "schema_version": "stage5.screening.v1",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "screening": {"enabled": True, **screening_summary(results)},
    }
    (run_dir / "screen_summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=json_default))
    return 0 if results else 1


def main() -> int:
    args = parse_args()
    if args.screen_only:
        return _screen_only(args)
    payload = _load_yaml(args.config)
    llm_cfg = _build_llm_config(args, payload.get("llm", payload))
    generation = payload.get("generation", {})
    screening = payload.get("screening", {})
    n = args.n if args.n is not None else int(generation.get("n", 200))
    batch_size = args.batch_size if args.batch_size is not None else int(generation.get("batch_size", 10))
    prompt_version = str(generation.get("prompt_version", PROMPT_VERSION))
    categories = generation.get("categories", None)

    run_dir = Path(
        args.run_dir
        or PROJECT_ROOT / "data" / "mining" / datetime.now(UTC).astimezone().strftime("run_%Y%m%d_%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    ledger = TrialLedger(run_dir)
    existing = ledger.existing_canonicals() if args.resume else set()
    seed_count = len(existing)
    if not args.no_seed_store:
        store_path = Path(args.seed_factor_store)
        if store_path.exists():
            seeded = pd.read_csv(store_path, usecols=["canonical"])
            existing.update(str(value) for value in seeded["canonical"].dropna())
    if args.seed_alpha158:
        for spec in alpha158_specs():
            existing.add(validate_expression(spec.expression).canonical)
    seed_after = len(existing)

    client = make_client(llm_cfg)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "provider": llm_cfg.provider,
                "model": llm_cfg.resolved_model,
                "base_url": llm_cfg.resolved_base_url,
                "n": n,
                "batch_size": batch_size,
                "prompt_version": prompt_version,
                "prompt_hash": prompt_hash(),
                "resume_existing_canonicals": seed_count,
                "seeded_canonicals_after": seed_after,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    try:
        result = generate_candidates(
            client,
            n=n,
            batch_size=batch_size,
            prompt_version=prompt_version,
            existing_canonicals=existing,
            ledger=ledger,
            categories=list(categories) if categories else list(CATEGORIES),
            run_id=run_dir.name,
        )
    finally:
        client.close()

    records_frame = _records_frame(result.records)
    records_frame.to_csv(run_dir / "candidates.csv", index=False)
    records_frame.to_parquet(run_dir / "candidates.parquet", index=False)

    screen_summary: dict[str, Any] = {"n_screened": 0, "enabled": False}
    if not args.no_screen and screening.get("enabled", True):
        baseline = _load_yaml(args.baseline_config)
        data_config = _build_data_config(args, baseline, screening)
        s1_config = _build_s1_config(args, _load_yaml(args.s1_config), screening)
        print(f"screening candidates on {data_config.start}..{data_config.end} ...")
        s1_results = screen_records(
            result.records,
            data_config,
            s1_config,
            max_candidates=args.screen_limit,
        )
        screen_summary = {"enabled": True, **screening_summary(s1_results)}
        if s1_results:
            scorecard = pd.DataFrame([scorecard_row(item) for item in s1_results])
            scorecard.to_csv(run_dir / "s1_scorecard.csv", index=False)
            scorecard.to_parquet(run_dir / "s1_scorecard.parquet", index=False)
        ledger.write_screened(result.records)
        records_frame = _records_frame(result.records)
        records_frame.to_csv(run_dir / "candidates.csv", index=False)
        records_frame.to_parquet(run_dir / "candidates.parquet", index=False)

    summary = {
        "schema_version": "stage5.generation.v1",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "prompt_version": prompt_version,
        "prompt_hash": prompt_hash(),
        "llm": llm_cfg.to_dict(),
        "generation": result.stats.to_dict(),
        "screening": screen_summary,
        "n_batches": len(result.batches),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=json_default))
    return 0 if result.stats.accepted > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
