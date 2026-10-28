#!/usr/bin/env python
"""GP-only ablation with a worker-backed, timeout-protected evaluator.

The stock GP runner calls the evaluator synchronously.  Qlib can hang on a
pathological expression, so this wrapper sends each expression to a persistent
worker process and kills/restarts the worker on timeout.  The GP loop itself is
unchanged (no LLM seeds, same typed tree operators and fitness).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import yaml
from run_random_search import _Worker

from src.mining.gp_eval import GPBudgetExhausted, GPEvalConfig, GPEvaluator, GPMetrics
from src.mining.gp_runner import GPConfig, GPRunner, write_gp_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "gp.yaml"))
    parser.add_argument("--baseline-config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--membership", default=str(PROJECT_ROOT / "data" / "processed" / "stage10_pit_csi300_membership.parquet"))
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "mining" / "stage10_gp_only"))
    return parser.parse_args()


def _yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


class WorkerEvaluator:
    """Minimal GPEvaluator-shaped facade over the worker process."""

    def __init__(self, config: GPEvalConfig, budget: int, timeout: float) -> None:
        self.config = config
        self.budget = int(budget)
        self.timeout = float(timeout)
        self.n_evaluations = 0
        self.cache: dict[str, GPMetrics] = {}
        self.worker = _Worker(config)

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.n_evaluations)

    def close(self) -> None:
        self.worker.close()

    def evaluate(self, expression: str, validated=None) -> GPMetrics:
        canonical = getattr(validated, "canonical", None) or expression
        if canonical in self.cache:
            return self.cache[canonical]
        if self.n_evaluations >= self.budget:
            raise GPBudgetExhausted(f"evaluation budget {self.budget} exhausted")
        self.n_evaluations += 1
        result = self.worker.evaluate(canonical, self.timeout)
        if "metrics" in result:
            metrics = GPMetrics(**result["metrics"])
        else:
            error = result.get("error") or result.get("fatal") or "worker error"
            metrics = GPMetrics(
                expression=canonical,
                canonical=canonical,
                expression_hash=None,
                fitness=None,
                rank_ic=None,
                icir=None,
                turnover=None,
                coverage=None,
                n_dates=0,
                complexity=None,
                monotonicity=None,
                error=str(error),
            )
        self.cache[canonical] = metrics
        return metrics

    def factor_panel(self, canonical: str):
        # Hall correlation de-duplication is disabled for this ablation; the
        # downstream S1 acts as the common gate for all four groups.
        del canonical


def main() -> int:
    args = parse_args()
    payload = _yaml(args.config)
    gp_payload = payload.get("gp", payload)
    data_payload = payload.get("data", {})
    fitness_payload = payload.get("fitness", {})
    baseline = _yaml(args.baseline_config).get("data", {})
    gp_config = GPConfig.from_dict(gp_payload)
    gp_config = dataclasses.replace(
        gp_config,
        budget=int(args.budget),
        seed_count=0,
        inject_every=0,
        seed=int(args.seed),
        hall_max_corr=1.0,
    )
    eval_config = GPEvalConfig(
        provider_uri=baseline["provider_uri"],
        membership_path=args.membership,
        sample_date=str(data_payload.get("sample_date", "2017-06-30")),
        sample_size=int(data_payload.get("sample_size", 100)),
        start=str(data_payload.get("start", "2017-01-01")),
        end=str(data_payload.get("end", "2018-12-31")),
        max_evaluations=100_000,
        min_names=int(fitness_payload.get("min_names", 20)),
        min_dates=int(fitness_payload.get("min_dates", 300)),
        quantile=int(fitness_payload.get("quantile", 5)),
        ic_weight=float(fitness_payload.get("ic_weight", 1.0)),
        icir_weight=float(fitness_payload.get("icir_weight", 0.5)),
        turnover_penalty=float(fitness_payload.get("turnover_penalty", 0.05)),
        complexity_penalty=float(fitness_payload.get("complexity_penalty", 0.0005)),
        kernels=1,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = WorkerEvaluator(eval_config, budget=args.budget, timeout=args.timeout)
    try:
        runner = GPRunner(gp_config, cast(GPEvaluator, evaluator), seed_expressions=[])
        result = runner.run()
        paths = write_gp_outputs(result, output_dir, runner.trials)
        summary = {
            **result.summary,
            "worker_restarts": evaluator.worker.restarts,
            "paths": paths,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    finally:
        evaluator.close()
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
