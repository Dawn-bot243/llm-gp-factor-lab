#!/usr/bin/env python
"""Stage 10: four-group ablation and locked 2020-2026 OOS.

Groups
------
A random search : 2000 random typed expressions
B GP-only       : 2000 GP evaluations, no LLM seeds
C LLM-only      : 230 accepted Stage 5 LLM candidates
D LLM+GP        : 2000 GP evaluations seeded/injected with LLM candidates

Common downstream protocol
--------------------------
* select the top 30 unique, runtime-valid candidates per group by the same
  2017-2018 fast fitness;
* run the same S1 scorecard on 2010-2018;
* report S1 pass/review/fail, OOS RankIC retention, and incremental IC after
  20 Alpha158 exposures;
* combine S1 pass+review factors equal-weight (train-window signs) and run
  the same fixed TopkDropout S3 simulation on the locked OOS window.

The OOS window 2020-09-24..2023-06-12 is outside every Stage 1-9 sample.
It stops at 2023-06-12 because the Dolt limit source ends there; the Qlib
snapshot extends to 2026 but is not used without PIT constraints.
"""
from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import argparse
import dataclasses
import json
import multiprocessing
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_gp_worker import WorkerEvaluator

from src.evaluation.ablation import (
    SearchCandidate,
    SearchStats,
    load_stage5_candidates,
    load_trial_candidates,
    pass_rate_ci,
    protocol_hash,
    s1_summary,
    select_s3_factors,
)
from src.evaluation.dedupe import Candidate, SamplePanels
from src.evaluation.orthogonalize import (
    Alpha158Exposures,
    ExposureConfig,
    incremental_ic_table,
)
from src.factors.evaluator import (
    DEFAULT_SNAPSHOT,
    EvaluationConfig,
    load_factor,
)
from src.factors.metrics import compute_ic
from src.factors.preprocess import preprocess_factor
from src.factors.s1 import (
    ForwardReturns,
    S1Config,
    S1Result,
    S1Thresholds,
    apply_fdr,
    evaluate_s1,
)
from src.factors.serialization import json_default, jsonable
from src.mining.gp_eval import GPBudgetExhausted, GPEvalConfig, GPEvaluator
from src.portfolio.data import load_market_data
from src.portfolio.simulator import PortfolioConfig, run_portfolio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "stage10.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts" / "stage10"))
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--groups", default=None, help="comma-separated group subset")
    parser.add_argument("--skip-oos", action="store_true", help="smoke mode: no 2020-2026 OOS")
    parser.add_argument("--force", action="store_true", help="overwrite an existing OOS lock")
    parser.add_argument("--smoke", action="store_true", help="small top-n and skip OOS")
    return parser.parse_args()


def _yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _evaluation_config(payload: dict[str, Any]) -> EvaluationConfig:
    baseline = _yaml(PROJECT_ROOT / payload["data"]["baseline_config"]).get("data", {})
    membership = _resolve(payload["data"]["membership"])
    constraints = _resolve(payload["data"].get("constraints"))
    if constraints is None or not constraints.exists():
        constraints = _resolve(payload["data"].get("fallback_constraints"))
    return EvaluationConfig(
        provider_uri=baseline["provider_uri"],
        instruments=baseline.get("instruments", "csi300"),
        start=payload["data"]["start"],
        end=payload["data"]["oos_end"],
        membership_path=str(membership),
        constraints_path=str(constraints) if constraints and constraints.exists() else None,
        snapshot_file=str(DEFAULT_SNAPSHOT),
        cache_dir=str(_resolve(payload["data"]["cache_dir"])),
        use_cache=True,
        kernels=int(payload["data"].get("kernels", 4)),
    )


def _s1_config(payload: dict[str, Any], *, skip_oos: bool) -> S1Config:
    s1 = payload["s1"]
    thresholds = S1Thresholds.from_dict(s1.get("thresholds"))
    return S1Config(
        quantiles=tuple(int(value) for value in s1.get("quantiles", [5, 10])),
        min_names=int(s1.get("min_names", 20)),
        winsorize=None if s1.get("winsorize") is None else float(s1["winsorize"]),
        standardize=bool(s1.get("standardize", True)),
        decay_horizons=tuple(int(value) for value in s1.get("decay_horizons", [1, 2, 3, 5, 10, 20])),
        calibration_start=str(s1["calibration_start"]),
        calibration_end=str(s1["calibration_end"]),
        holdout_start="2019-01-01" if skip_oos else str(s1["holdout_start"]),
        holdout_end="2019-12-31" if skip_oos else str(s1["holdout_end"]),
        thresholds=thresholds,
    )


def _write_protocol(payload: dict[str, Any], output_dir: Path, *, skip_oos: bool, top_n: int) -> dict[str, Any]:
    protocol = {
        "schema_version": "stage10.protocol.v1",
        "created_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "config": payload,
        "top_n": top_n,
        "skip_oos": skip_oos,
        "oos_start": payload["data"]["oos_start"],
        "oos_end": payload["data"]["oos_end"],
        "portfolio": payload["portfolio"],
        "s1": payload["s1"],
    }
    protocol["protocol_hash"] = protocol_hash(protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protocol.json").write_text(
        json.dumps(jsonable(protocol), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    return protocol


def _sample_evaluator(payload: dict[str, Any]) -> GPEvaluator:
    baseline = _yaml(PROJECT_ROOT / payload["data"]["baseline_config"]).get("data", {})
    config = GPEvalConfig(
        provider_uri=baseline["provider_uri"],
        membership_path=str(_resolve(payload["data"]["membership"])),
        sample_date="2017-06-30",
        sample_size=100,
        start="2017-01-01",
        end="2018-12-31",
        max_evaluations=100_000,
        min_names=20,
        min_dates=300,
        quantile=5,
        ic_weight=1.0,
        icir_weight=0.5,
        turnover_penalty=0.05,
        complexity_penalty=0.0005,
        kernels=1,
    )
    return GPEvaluator(config)


def _diverse_select(
    candidates: list[SearchCandidate],
    evaluator: GPEvaluator,
    *,
    top_n: int,
    threshold: float = 0.70,
) -> list[SearchCandidate]:
    """Greedy value-correlation de-duplication on the 100-stock sample."""
    ordered = sorted(candidates, key=lambda item: item.fitness or -1e9, reverse=True)
    ranked: dict[str, pd.Series] = {}
    for candidate in ordered:
        try:
            panel = evaluator.factor_panel(candidate.canonical)
            values = panel["factor"].groupby(level="datetime").rank(pct=True)
            ranked[candidate.canonical] = values.rename(candidate.canonical)
        except Exception:  # noqa: S112
            continue
    if not ranked:
        return ordered[:top_n]
    frame = pd.concat(ranked.values(), axis=1)
    corr = frame.corr(method="spearman")
    selected: list[SearchCandidate] = []
    for candidate in ordered:
        if candidate.canonical not in corr.columns:
            continue
        keep = True
        for previous in selected:
            value = corr.loc[candidate.canonical, previous.canonical]
            if pd.notna(value) and abs(float(value)) > threshold:
                keep = False
                break
        if keep:
            selected.append(candidate)
        if len(selected) >= top_n:
            break
    return selected


def _s1_worker(data_config: EvaluationConfig, s1_config: S1Config, conn) -> None:
    """Worker process: evaluate one factor at a time and return its scorecard."""
    try:
        forward = ForwardReturns.from_config(data_config, s1_config.decay_horizons)
    except Exception as exc:  # pragma: no cover - startup failure
        conn.send({"fatal": f"{type(exc).__name__}: {exc}"})
        return
    while True:
        try:
            item = conn.recv()
        except EOFError:
            break
        if item is None:
            break
        name, expression = item
        try:
            validated, factor = load_factor(expression, data_config)
            result = evaluate_s1(
                expression,
                data_config,
                s1_config,
                forward,
                name=name,
                validated=validated,
                factor=factor,
            )
            conn.send(
                {
                    "name": name,
                    "canonical": result.scorecard["canonical"],
                    "scorecard": result.scorecard,
                }
            )
        except Exception as exc:
            conn.send(
                {
                    "name": name,
                    "canonical": expression,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )


class _S1Worker:
    def __init__(self, data_config: EvaluationConfig, s1_config: S1Config, timeout: float) -> None:
        self.data_config = data_config
        self.s1_config = s1_config
        self.timeout = float(timeout)
        self.ctx = multiprocessing.get_context("spawn")
        self.proc: Any = None
        self.conn: Any = None
        self.restarts = 0
        self._start()

    def _start(self) -> None:
        parent, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(
            target=_s1_worker, args=(self.data_config, self.s1_config, child), daemon=True
        )
        self.proc.start()
        self.conn = parent

    def close(self) -> None:
        if self.conn is not None:
            try:
                self.conn.send(None)
            except Exception:  # noqa: S110
                pass
            self.conn.close()
        if self.proc is not None:
            self.proc.join(timeout=3)
            if self.proc.is_alive():
                self.proc.terminate()
                self.proc.join(timeout=3)

    def evaluate(self, name: str, expression: str) -> dict[str, Any]:
        if self.proc is None or not self.proc.is_alive():
            self._restart()
        assert self.conn is not None
        try:
            self.conn.send((name, expression))
        except (BrokenPipeError, OSError):
            self._restart()
            self.conn.send((name, expression))
        if not self.conn.poll(self.timeout):
            self._restart()
            return {
                "name": name,
                "canonical": expression,
                "error": f"timeout after {self.timeout:.0f}s",
            }
        try:
            return self.conn.recv()
        except EOFError:
            self._restart()
            return {"name": name, "canonical": expression, "error": "worker exited"}

    def _restart(self) -> None:
        self.restarts += 1
        if self.proc is not None:
            self.proc.terminate()
            self.proc.join(timeout=3)
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:  # noqa: S110
                pass
        self._start()


def _apply_fdr_rows(rows: list[dict[str, Any]], alpha: float) -> None:
    evaluated = [row for row in rows if row.get("status") != "error"]
    if not evaluated:
        return
    pvalues = [float(row.get("p_value") or 1.0) for row in evaluated]
    try:
        _, qvalues, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
    except Exception:
        qvalues = pvalues
    for row, q_value in zip(evaluated, qvalues, strict=True):
        q = float(q_value)
        row["q_value"] = q
        row["fdr_pass"] = bool(q <= alpha)
        if q > alpha:
            reasons = row.get("reasons")
            if isinstance(reasons, str):
                reasons = [item for item in reasons.split(";") if item]
            reasons = list(reasons or [])
            if "FDR_FAIL" not in reasons:
                reasons.append("FDR_FAIL")
            row["reasons"] = reasons
            row["status"] = "fail"


def _aggregate_stats(group: str, parts: list[SearchStats], selected: int) -> SearchStats:
    stats = SearchStats(group=group)
    stats.n_returned = sum(part.n_returned for part in parts)
    stats.n_unique = sum(part.n_unique for part in parts)
    stats.n_valid = sum(part.n_valid for part in parts)
    stats.n_eligible = sum(part.n_eligible for part in parts)
    stats.n_coverage_excluded = sum(part.n_coverage_excluded for part in parts)
    stats.n_error = sum(part.n_error for part in parts)
    stats.n_selected = selected
    return stats


def _stage5_coverage(candidates: list[SearchCandidate], payload: dict[str, Any]) -> list[SearchCandidate]:
    path = _resolve(payload["data"].get("stage5_scorecard"))
    if path is None or not Path(path).exists():
        path = PROJECT_ROOT / "data" / "mining" / "stage5_combined_scorecard.csv"
    if not Path(path).exists():
        return candidates
    frame = pd.read_csv(path, keep_default_na=False)
    coverage = {
        str(row["canonical"]): float(row["coverage"])
        for _, row in frame.iterrows()
        if str(row.get("coverage", "")).strip() not in {"", "nan"}
    }
    out = []
    for candidate in candidates:
        value = coverage.get(candidate.canonical)
        extra = dict(candidate.extra)
        extra["coverage"] = value
        out.append(dataclasses.replace(candidate, extra=extra))
    return out


def _load_groups(
    payload: dict[str, Any],
    group_names: list[str],
    top_n: int,
    *,
    data_config: EvaluationConfig,
    output_dir: Path,
) -> tuple[dict[str, list[SearchCandidate]], dict[str, SearchStats]]:
    selected: dict[str, list[SearchCandidate]] = {}
    stats: dict[str, SearchStats] = {}
    downstream = payload.get("downstream", {})
    min_coverage = downstream.get("min_coverage")
    pool_multiplier = int(downstream.get("pool_multiplier", 3))
    correlation_threshold = float(downstream.get("correlation_threshold", 0.70))
    evaluator = _sample_evaluator(payload)
    for group in group_names:
        spec = payload["groups"][group]
        paths = spec.get("paths") or [spec["path"]]
        resolved_paths = [
            resolved for path in paths if (resolved := _resolve(path)) is not None
        ]
        print(f"[{group}] loading {len(resolved_paths)} path(s): {resolved_paths}", flush=True)
        pool: list[SearchCandidate] = []
        raw_rows: list[dict[str, Any]] = []
        parts: list[SearchStats] = []

        if spec["kind"] == "stage5_candidates":
            path = resolved_paths[0]
            candidates, group_stats = load_stage5_candidates(path, group, top_n=10_000)
            group_stats.n_returned = 370
            group_stats.n_unique = len(candidates)
            group_stats.n_valid = len(candidates)
            candidates = _stage5_coverage(candidates, payload)
            fitness_path = output_dir / "llm_only_fitness.csv"
            fallback_fitness = PROJECT_ROOT / "artifacts" / "stage10" / "llm_only_fitness.csv"
            use_fitness = fitness_path if fitness_path.exists() else fallback_fitness
            if use_fitness.exists():
                cached = pd.read_csv(use_fitness, keep_default_na=False)
                by_canonical = {str(row["canonical"]): row for _, row in cached.iterrows()}
                updated = []
                for candidate in candidates:
                    row = by_canonical.get(candidate.canonical)
                    if row is None:
                        updated.append(candidate)
                        continue
                    raw_fitness = row.get("fitness")
                    fitness_value = float(raw_fitness) if str(raw_fitness).strip() not in {"", "nan"} else None
                    raw_rank = row.get("rank_ic")
                    rank_value = float(raw_rank) if str(raw_rank).strip() not in {"", "nan"} else None
                    error = str(row.get("error") or "").strip() or None
                    extra = dict(candidate.extra)
                    raw_coverage = row.get("coverage")
                    if raw_coverage is not None and str(raw_coverage).strip() not in {"", "nan", "None"}:
                        extra["coverage"] = float(raw_coverage)
                    updated.append(
                        dataclasses.replace(
                            candidate,
                            fitness=fitness_value,
                            search_rank_ic=rank_value,
                            error=error,
                            extra=extra,
                        )
                    )
                candidates = updated
                print(f"[{group}] reused {len(by_canonical)} cached fitness rows", flush=True)
            else:
                candidates = _fill_stage5_fitness(candidates, payload, output_dir)
            valid = [item for item in candidates if item.error is None and item.fitness is not None]
            group_stats.n_valid = len(valid)
            eligible = [
                item
                for item in valid
                if min_coverage is None
                or item.extra.get("coverage") is None
                or float(item.extra["coverage"]) >= float(min_coverage)
            ]
            group_stats.n_eligible = len(eligible)
            group_stats.n_coverage_excluded = len(valid) - len(eligible)
            pool = sorted(eligible, key=lambda item: item.fitness or -1e9, reverse=True)[: top_n * pool_multiplier]
            parts.append(group_stats)
        else:
            budget = int(spec.get("budget", 0)) or None
            per_pool = int(spec.get("per_seed_pool", top_n * pool_multiplier))
            for path in resolved_paths:
                candidates, part_stats, raw = load_trial_candidates(
                    path,
                    group,
                    top_n=per_pool,
                    max_trials=budget,
                    max_complexity=spec.get("max_complexity"),
                    min_coverage=min_coverage,
                    max_expression_length=spec.get("max_expression_length"),
                )
                parts.append(part_stats)
                seed_name = Path(path).parent.name
                candidates = [
                    dataclasses.replace(
                        item, extra={**item.extra, "seed": seed_name}
                    )
                    for item in candidates
                ]
                pool.extend(candidates)
                raw_rows.extend(raw)
            trial_by_canonical: dict[str, SearchCandidate] = {}
            for item in pool:
                previous = trial_by_canonical.get(item.canonical)
                if previous is None or (item.fitness or -1e9) > (previous.fitness or -1e9):
                    trial_by_canonical[item.canonical] = item
            pool = sorted(
                trial_by_canonical.values(), key=lambda item: item.fitness or -1e9, reverse=True
            )

        group_stats = _aggregate_stats(group, parts, selected=0)
        selected[group] = _diverse_select(
            pool,
            evaluator,
            top_n=top_n,
            threshold=correlation_threshold,
        )
        group_stats.n_selected = len(selected[group])
        stats[group] = group_stats
        if raw_rows:
            pd.DataFrame(raw_rows).to_csv(output_dir / f"search_trials_{group}.csv", index=False)
        print(
            f"[{group}] returned={group_stats.n_returned} unique={group_stats.n_unique} "
            f"valid={group_stats.n_valid} eligible={group_stats.n_eligible} "
            f"coverage_excluded={group_stats.n_coverage_excluded} selected={group_stats.n_selected}",
            flush=True,
        )
    return selected, stats


def _fill_stage5_fitness(
    candidates: list[SearchCandidate],
    payload: dict[str, Any],
    output_dir: Path,
) -> list[SearchCandidate]:
    baseline = _yaml(PROJECT_ROOT / payload["data"]["baseline_config"]).get("data", {})
    gp_config = GPEvalConfig(
        provider_uri=baseline["provider_uri"],
        membership_path=str(_resolve(payload["data"]["membership"])),
        sample_date="2017-06-30",
        sample_size=100,
        start="2017-01-01",
        end="2018-12-31",
        max_evaluations=len(candidates) + 10,
        min_names=20,
        min_dates=300,
        quantile=5,
        ic_weight=1.0,
        icir_weight=0.5,
        turnover_penalty=0.05,
        complexity_penalty=0.0005,
    )
    evaluator = WorkerEvaluator(gp_config, budget=len(candidates) + 10, timeout=10.0)
    rows = []
    updated = []
    for index, candidate in enumerate(candidates, start=1):
        try:
            metrics = evaluator.evaluate(candidate.canonical)
            updated.append(
                dataclasses.replace(
                    candidate,
                    fitness=metrics.fitness,
                    search_rank_ic=metrics.rank_ic,
                    error=metrics.error,
                )
            )
            rows.append(
                {
                    "group": candidate.group,
                    "name": candidate.name,
                    "canonical": candidate.canonical,
                    "fitness": metrics.fitness,
                    "rank_ic": metrics.rank_ic,
                    "icir": metrics.icir,
                    "error": metrics.error,
                }
            )
        except GPBudgetExhausted:
            break
        except Exception as exc:
            updated.append(dataclasses.replace(candidate, error=f"{type(exc).__name__}: {exc}"))
            rows.append(
                {
                    "group": candidate.group,
                    "name": candidate.name,
                    "canonical": candidate.canonical,
                    "fitness": None,
                    "rank_ic": None,
                    "icir": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if index % 50 == 0:
            print(f"[llm_only] fitness {index}/{len(candidates)}", flush=True)
    evaluator.close()
    pd.DataFrame(rows).to_csv(output_dir / "llm_only_fitness.csv", index=False)
    return updated


def _run_s1_worker(
    selected: dict[str, list[SearchCandidate]],
    payload: dict[str, Any],
    data_config: EvaluationConfig,
    s1_config: S1Config,
    output_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    timeout = float(payload["s1"].get("worker_timeout", 90.0))
    alpha = float(s1_config.thresholds.fdr_alpha)
    for group, candidates in selected.items():
        print(f"[{group}] S1 worker {len(candidates)} candidates", flush=True)
        worker = _S1Worker(data_config, s1_config, timeout)
        rows: list[dict[str, Any]] = []
        try:
            for index, candidate in enumerate(candidates, start=1):
                result = worker.evaluate(candidate.name, candidate.canonical)
                scorecard = result.get("scorecard")
                if scorecard:
                    row = dict(scorecard)
                    row.update(
                        {
                            "group": group,
                            "search_fitness": candidate.fitness,
                            "search_rank_ic": candidate.search_rank_ic,
                            "seed": candidate.extra.get("seed"),
                        }
                    )
                else:
                    row = {
                        "group": group,
                        "name": candidate.name,
                        "canonical": candidate.canonical,
                        "status": "error",
                        "reasons": [result.get("error") or "worker error"],
                        "rank_ic_mean": None,
                        "holdout_rank_ic": None,
                        "coverage": None,
                        "search_fitness": candidate.fitness,
                        "seed": candidate.extra.get("seed"),
                    }
                rows.append(row)
                if index % 10 == 0:
                    print(f"[{group}] S1 worker {index}/{len(candidates)}", flush=True)
        finally:
            worker.close()
        _apply_fdr_rows(rows, alpha)
        frame = pd.DataFrame(rows)
        if not frame.empty and "reasons" in frame.columns:
            frame["reasons"] = frame["reasons"].apply(
                lambda value: ";".join(value) if isinstance(value, list) else value
            )
        frame.to_csv(output_dir / f"s1_{group}.csv", index=False)
        rows_by_group[group] = rows
        summary = s1_summary(rows)
        print(
            f"[{group}] S1 summary {json.dumps(jsonable(summary), ensure_ascii=False)} "
            f"restarts={worker.restarts}",
            flush=True,
        )
    return rows_by_group, {}


def _run_s1(
    selected: dict[str, list[SearchCandidate]],
    payload: dict[str, Any],
    data_config: EvaluationConfig,
    s1_config: S1Config,
    forward: ForwardReturns,
    output_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]]]:
    if payload["s1"].get("worker", False):
        return _run_s1_worker(selected, payload, data_config, s1_config, output_dir)
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    results_by_group: dict[str, list[dict[str, Any]]] = {}
    for group, candidates in selected.items():
        print(f"[{group}] S1 {len(candidates)} candidates", flush=True)
        results: list[S1Result] = []
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates, start=1):
            try:
                validated, factor = load_factor(candidate.canonical, data_config)
                result = evaluate_s1(
                    candidate.canonical,
                    data_config,
                    s1_config,
                    forward,
                    name=candidate.name,
                    validated=validated,
                    factor=factor,
                )
                results.append(result)
                row = dict(result.scorecard)
                row.update(
                    {
                        "group": group,
                        "search_fitness": candidate.fitness,
                        "search_rank_ic": candidate.search_rank_ic,
                        "seed": candidate.extra.get("seed"),
                    }
                )
                rows.append(row)
            except Exception as exc:
                rows.append(
                    {
                        "group": group,
                        "name": candidate.name,
                        "canonical": candidate.canonical,
                        "status": "error",
                        "reasons": f"{type(exc).__name__}: {exc}",
                        "rank_ic_mean": None,
                        "holdout_rank_ic": None,
                        "search_fitness": candidate.fitness,
                        "seed": candidate.extra.get("seed"),
                    }
                )
            if index % 10 == 0:
                print(f"[{group}] S1 {index}/{len(candidates)}", flush=True)
        apply_fdr(results, alpha=float(s1_config.thresholds.fdr_alpha))
        # The scorecard dicts are mutated by apply_fdr; rebuild rows for
        # evaluated results so q-values/status are the final values.
        by_canonical = {result.scorecard["canonical"]: result.scorecard for result in results}
        for row in rows:
            if row.get("canonical") in by_canonical:
                updated = dict(by_canonical[row["canonical"]])
                updated.update(
                    {
                        "group": group,
                        "search_fitness": row.get("search_fitness"),
                        "search_rank_ic": row.get("search_rank_ic"),
                        "seed": row.get("seed"),
                    }
                )
                row.clear()
                row.update(updated)
        rows_by_group[group] = rows
        results_by_group[group] = [dict(result.scorecard) for result in results]
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["reasons"] = frame["reasons"].apply(
                lambda value: ";".join(value) if isinstance(value, list) else value
            )
        frame.to_csv(output_dir / f"s1_{group}.csv", index=False)
        summary = s1_summary(rows)
        print(f"[{group}] S1 summary {json.dumps(jsonable(summary), ensure_ascii=False)}", flush=True)
    return rows_by_group, results_by_group


def _sensitivity_rows(
    group: str,
    rows: list[dict[str, Any]],
    *,
    ic_thresholds: list[float],
    coverage_thresholds: list[float],
    base_thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    out = []
    fdr_alpha = float(base_thresholds.get("fdr_alpha", 0.10))
    for ic_threshold in ic_thresholds:
        for coverage_threshold in coverage_thresholds:
            passes = 0
            for row in rows:
                values = {
                    "rank_ic": row.get("rank_ic_mean"),
                    "icir": row.get("icir"),
                    "nw_t": row.get("newey_west_t"),
                    "coverage": row.get("coverage"),
                    "monotonicity": row.get("monotonicity"),
                    "turnover": row.get("turnover"),
                    "year": row.get("year_sign_consistency"),
                    "q": row.get("q_value"),
                }
                if any(values[key] is None for key in values):
                    continue
                ok = (
                    abs(cast(float, values["rank_ic"])) >= float(ic_threshold)
                    and abs(cast(float, values["icir"])) >= float(base_thresholds.get("min_abs_icir", 0.10))
                    and abs(cast(float, values["nw_t"])) >= float(base_thresholds.get("min_abs_nw_t", 2.0))
                    and cast(float, values["coverage"]) >= float(coverage_threshold)
                    and abs(cast(float, values["monotonicity"]))
                    >= float(base_thresholds.get("min_abs_monotonicity", 0.50))
                    and cast(float, values["turnover"]) <= float(base_thresholds.get("max_turnover", 0.60))
                    and cast(float, values["year"])
                    >= float(base_thresholds.get("min_year_sign_consistency", 0.55))
                    and cast(float, values["q"]) <= fdr_alpha
                )
                if ok:
                    passes += 1
            lo, hi = pass_rate_ci(passes, len(rows))
            out.append(
                {
                    "group": group,
                    "ic_threshold": float(ic_threshold),
                    "coverage_threshold": float(coverage_threshold),
                    "n_pass": passes,
                    "n_total": len(rows),
                    "pass_rate": passes / len(rows) if rows else None,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )
    return out


def _run_incremental(
    rows_by_group: dict[str, list[dict[str, Any]]],
    payload: dict[str, Any],
    data_config: EvaluationConfig,
    output_dir: Path,
) -> dict[str, Any]:
    config = payload["incremental"]
    panels = SamplePanels(
        provider_uri=data_config.provider_uri,
        membership_path=data_config.membership_path,
        sample_date=str(config.get("sample_date", "2017-06-30")),
        sample_size=int(config.get("sample_size", 100)),
        start=str(config.get("start", "2010-01-01")),
        end=str(config.get("end", "2018-12-31")),
        kernels=data_config.kernels,
    )
    exposure_config = ExposureConfig(
        alpha158_count=int(config.get("alpha158_count", 20)),
        ridge_alpha=float(config.get("ridge_alpha", 1.0)),
        min_names=int(payload["s1"].get("min_names", 20)),
    )
    exposures = Alpha158Exposures(panels, exposure_config)
    results: dict[str, Any] = {}
    top_n = int(payload["downstream"].get("incremental_top_n", 10))
    for group, rows in rows_by_group.items():
        accepted = [
            row
            for row in rows
            if str(row.get("status")) in {"pass", "review"}
            and str(row.get("canonical") or "").strip()
        ]
        accepted = sorted(
            accepted,
            key=lambda row: abs(float(row["rank_ic_mean"])) if row.get("rank_ic_mean") is not None else -1.0,
            reverse=True,
        )[:top_n]
        candidates = [
            Candidate(
                name=str(row.get("name") or row["canonical"]),
                expression=str(row.get("expression") or row["canonical"]),
                canonical=str(row["canonical"]),
                source=group,
                score=abs(float(row.get("rank_ic_mean") or 0.0)),
            )
            for row in accepted
        ]
        if not candidates:
            results[group] = {"n": 0, "mean_pre_rank_ic": None, "mean_post_rank_ic": None, "mean_ic_drop": None}
            continue
        try:
            table = incremental_ic_table(candidates, panels, exposures, exposure_config)
            table.to_csv(output_dir / f"incremental_ic_{group}.csv", index=False)
            results[group] = {
                "n": len(table),
                "mean_pre_rank_ic": float(table["pre_rank_ic"].mean()) if "pre_rank_ic" in table else None,
                "mean_post_rank_ic": (
                    float(table["incremental_ic"].mean()) if "incremental_ic" in table else None
                ),
                "mean_ic_drop": (
                    float((table["pre_rank_ic"] - table["incremental_ic"]).mean())
                    if {"pre_rank_ic", "incremental_ic"}.issubset(table.columns)
                    else None
                ),
            }
        except Exception as exc:
            results[group] = {"n": 0, "error": f"{type(exc).__name__}: {exc}"}
        print(f"[{group}] incremental IC {results[group]}", flush=True)
    return results


def _combine_s3_factors(
    factors: list[dict[str, Any]],
    data_config: EvaluationConfig,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    pieces = []
    used = []
    for row in factors:
        expression = str(row.get("expression") or row.get("canonical"))
        validated, panel = load_factor(expression, data_config)
        processed = preprocess_factor(panel, winsorize=0.01, standardize=True)
        sign = 1.0 if float(row.get("rank_ic_mean") or 0.0) >= 0 else -1.0
        pieces.append((sign * processed).rename(row["canonical"]))
        used.append(
            {
                "name": row.get("name"),
                "canonical": row["canonical"],
                "status": row.get("status"),
                "rank_ic_mean": row.get("rank_ic_mean"),
                "holdout_rank_ic": row.get("holdout_rank_ic"),
                "sign": sign,
                "expression_hash": validated.expression_hash,
            }
        )
    if not pieces:
        return pd.Series(dtype=float, name="score"), used
    frame = pd.concat(pieces, axis=1)
    score = frame.mean(axis=1, skipna=True).rename("score")
    return score, used


def _combined_ic(score: pd.Series, label: pd.Series, start: str, end: str, min_names: int) -> dict[str, Any]:
    panel = pd.concat([score, label], axis=1, join="inner").dropna()
    dates = panel.index.get_level_values("datetime")
    panel = panel[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]
    if panel.empty:
        return {"mean": None, "icir": None, "n_dates": 0}
    ic, _ = compute_ic(panel, "score", label.name, min_names=min_names)
    if ic.empty:
        return {"mean": None, "icir": None, "n_dates": 0}
    return {
        "mean": float(ic.mean()),
        "icir": float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) > 0 else None,
        "n_dates": len(ic),
    }


def _run_oos(
    rows_by_group: dict[str, list[dict[str, Any]]],
    payload: dict[str, Any],
    data_config: EvaluationConfig,
    forward: ForwardReturns,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    market = load_market_data(data_config)
    label = forward.get(1)
    portfolio_config = PortfolioConfig.from_dict(payload["portfolio"])
    oos_start = str(payload["data"]["oos_start"])
    oos_end = str(payload["data"]["oos_end"])
    min_names = int(payload["s1"].get("min_names", 20))
    results: dict[str, Any] = {}
    details: dict[str, Any] = {}
    for group, rows in rows_by_group.items():
        factors, fallback = select_s3_factors(
            rows,
            min_factors=int(payload["downstream"].get("min_factors_for_s3", 2)),
            fallback_top=int(payload["downstream"].get("fallback_to_top", 2)),
            allow_fallback=bool(payload["downstream"].get("allow_fallback", True)),
            allowed_statuses=tuple(payload["downstream"].get("s3_statuses", ["pass", "review"])),
        )
        if not factors:
            results[group] = {"error": "insufficient S1 pass factors"}
            continue
        print(f"[{group}] OOS S3 with {len(factors)} factors (fallback={fallback})", flush=True)
        try:
            score, used = _combine_s3_factors(factors, data_config)
            if score.empty:
                raise ValueError("combined score is empty")
            wide = score.unstack("instrument").sort_index()
            simulation = run_portfolio(
                scores=wide,
                open_px=market.open,
                close_px=market.close,
                amount_cny=market.amount_cny,
                benchmark_close=market.benchmark_close,
                config=portfolio_config,
                start=oos_start,
                end=oos_end,
                suspended=market.suspended,
                is_st=market.is_st,
                at_limit_up=market.at_limit_up,
                at_limit_down=market.at_limit_down,
                factor=market.factor,
                collect_weights=False,
            )
            diag = _combined_ic(score, label, oos_start, oos_end, min_names)
            cal = _combined_ic(
                score,
                label,
                str(payload["s1"]["calibration_start"]),
                str(payload["s1"]["calibration_end"]),
                min_names,
            )
            simulation.daily.to_csv(output_dir / f"oos_{group}_daily.csv", index_label="datetime")
            results[group] = {
                "fallback": fallback,
                "n_factors": len(factors),
                "factors": used,
                "oos_rank_ic": diag,
                "calibration_rank_ic": cal,
                **simulation.metrics,
            }
            details[group] = {"factors": used, "portfolio_config": portfolio_config.to_dict()}
            print(
                f"[{group}] OOS excess={simulation.metrics.get('excess_annual_return')} "
                f"IR={simulation.metrics.get('net_excess_ir')}",
                flush=True,
            )
        except Exception as exc:
            results[group] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[{group}] OOS failed: {results[group]['error']}", flush=True)
    return results, details


def main() -> int:
    args = parse_args()
    payload = _yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    top_n = int(args.top_n or payload["downstream"].get("top_n", 30))
    skip_oos = bool(args.skip_oos or args.smoke)
    if args.smoke:
        top_n = min(top_n, 3)
    group_names = list(payload["groups"])
    if args.groups:
        group_names = [item.strip() for item in args.groups.split(",") if item.strip()]
    started = time.time()

    protocol = _write_protocol(payload, output_dir, skip_oos=skip_oos, top_n=top_n)
    lock_path = _resolve(payload["oos"].get("lock_file", "artifacts/stage10/oos_lock.json"))
    if not skip_oos and lock_path is not None:
        if lock_path.exists() and not args.force:
            raise SystemExit(
                f"OOS lock already exists at {lock_path}; refusing a second OOS run. "
                "Use --force only if the protocol changed and you will disclose the rerun."
            )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            json.dumps(
                jsonable(
                    {
                        "status": "started",
                        "protocol_hash": protocol["protocol_hash"],
                        "started_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
                        "oos_start": payload["data"]["oos_start"],
                        "oos_end": payload["data"]["oos_end"],
                    }
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"OOS lock written before any OOS evaluation: {lock_path}", flush=True)

    data_config = _evaluation_config(payload)
    s1_config = _s1_config(payload, skip_oos=skip_oos)
    print(
        json.dumps(
            {
                "data": data_config.to_dict(),
                "s1": s1_config.to_dict(),
                "top_n": top_n,
                "skip_oos": skip_oos,
                "groups": group_names,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        flush=True,
    )

    selected, stats = _load_groups(payload, group_names, top_n, data_config=data_config, output_dir=output_dir)
    pd.DataFrame(
        [
            {
                "group": group,
                "name": candidate.name,
                "canonical": candidate.canonical,
                "fitness": candidate.fitness,
                "search_rank_ic": candidate.search_rank_ic,
                "source": candidate.source,
                "seed": candidate.extra.get("seed"),
            }
            for group in group_names
            for candidate in selected.get(group, [])
        ]
    ).to_csv(output_dir / "selected_candidates.csv", index=False)
    forward = ForwardReturns.from_config(data_config, s1_config.decay_horizons)
    rows_by_group, _results_by_group = _run_s1(
        selected, payload, data_config, s1_config, forward, output_dir
    )
    sensitivity_config = payload["s1"].get("sensitivity", {})
    ic_thresholds = [float(value) for value in sensitivity_config.get("ic_thresholds", [0.015])]
    coverage_thresholds = [
        float(value) for value in sensitivity_config.get("coverage_thresholds", [0.80])
    ]
    sensitivity_rows: list[dict[str, Any]] = []
    for group, rows in rows_by_group.items():
        sensitivity_rows.extend(
            _sensitivity_rows(
                group,
                rows,
                ic_thresholds=ic_thresholds,
                coverage_thresholds=coverage_thresholds,
                base_thresholds=payload["s1"].get("thresholds", {}),
            )
        )
    if sensitivity_rows:
        pd.DataFrame(sensitivity_rows).to_csv(output_dir / "s1_sensitivity.csv", index=False)
    incremental = _run_incremental(rows_by_group, payload, data_config, output_dir)
    oos_results: dict[str, Any] = {}
    oos_details: dict[str, Any] = {}
    if not skip_oos:
        oos_results, oos_details = _run_oos(rows_by_group, payload, data_config, forward, output_dir)
        if lock_path is not None:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            lock["status"] = "completed"
            lock["completed_at"] = datetime.now(UTC).astimezone().isoformat(timespec="seconds")
            lock_path.write_text(json.dumps(jsonable(lock), ensure_ascii=False, indent=2), encoding="utf-8")

    ablation_rows = []
    for group in group_names:
        group_stats = stats[group]
        s1 = s1_summary(rows_by_group.get(group, []))
        inc = incremental.get(group, {})
        oos = oos_results.get(group, {})
        row = {
            "group": group,
            "label": payload["groups"][group].get("label", group),
            "search_budget": int(payload["groups"][group].get("budget", 0)),
            "n_returned": group_stats.n_returned,
            "n_unique": group_stats.n_unique,
            "n_valid": group_stats.n_valid,
            "n_eligible": group_stats.n_eligible,
            "n_coverage_excluded": group_stats.n_coverage_excluded,
            "legal_rate": group_stats.legal_rate,
            "unique_rate": group_stats.unique_rate,
            "n_selected": len(selected.get(group, [])),
            **s1,
            "s1_ci_low": pass_rate_ci(int(s1.get("n_pass") or 0), int(s1.get("n_s1") or 0))[0],
            "s1_ci_high": pass_rate_ci(int(s1.get("n_pass") or 0), int(s1.get("n_s1") or 0))[1],
            "incremental_n": inc.get("n"),
            "incremental_pre_rank_ic": inc.get("mean_pre_rank_ic"),
            "incremental_post_rank_ic": inc.get("mean_post_rank_ic"),
            "incremental_ic_drop": inc.get("mean_ic_drop"),
            "n_s3_factors": oos.get("n_factors"),
            "s3_fallback": oos.get("fallback"),
            "oos_net_annual": oos.get("annual_return"),
            "oos_benchmark_annual": oos.get("benchmark_annual_return"),
            "oos_excess_annual": oos.get("excess_annual_return"),
            "oos_sharpe": oos.get("sharpe"),
            "oos_net_excess_ir": oos.get("net_excess_ir"),
            "oos_turnover_annual": oos.get("turnover_annual"),
            "oos_max_drawdown": oos.get("max_drawdown"),
            "oos_cost_drag_annual": oos.get("cost_drag_annual"),
            "oos_rank_ic": oos.get("oos_rank_ic", {}).get("mean") if isinstance(oos.get("oos_rank_ic"), dict) else None,
            "calibration_rank_ic": oos.get("calibration_rank_ic", {}).get("mean") if isinstance(oos.get("calibration_rank_ic"), dict) else None,
            "error": oos.get("error"),
        }
        ablation_rows.append(row)
    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(output_dir / "ablation_summary.csv", index=False)
    details = {
        "schema_version": "stage10.ablation.v1",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "protocol": protocol,
        "protocol_hash": protocol["protocol_hash"],
        "skip_oos": skip_oos,
        "runtime_seconds": round(time.time() - started, 1),
        "search_stats": {group: stats[group].__dict__ for group in group_names},
        "s1": {group: s1_summary(rows_by_group.get(group, [])) for group in group_names},
        "incremental": incremental,
        "oos": oos_results,
        "oos_details": oos_details,
        "ablation": jsonable(ablation.to_dict(orient="records")),
    }
    (output_dir / "ablation_details.json").write_text(
        json.dumps(jsonable(details), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "stage10.summary.v1",
        "generated_at": details["generated_at"],
        "runtime_seconds": details["runtime_seconds"],
        "skip_oos": skip_oos,
        "protocol_hash": protocol["protocol_hash"],
        "top_n": top_n,
        "groups": group_names,
        "ablation": jsonable(ablation.to_dict(orient="records")),
        "paths": {
            "ablation_csv": str(output_dir / "ablation_summary.csv"),
            "details": str(output_dir / "ablation_details.json"),
            "protocol": str(output_dir / "protocol.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(ablation.to_string(index=False))
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
