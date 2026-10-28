#!/usr/bin/env python
"""Stage 10 ablation A: random typed-expression search with a fixed budget.

This is the random-search control for the LLM/GP ablation.  It uses the same
typed expression space and fast fitness function as Stage 6, but no evolution,
no LLM seeds and no crossover/mutation.

Each expression is evaluated in a persistent worker process.  If Qlib hangs on
a pathological random tree (observed for some deeply nested ``If``/``Corr``
trees), the parent kills and restarts the worker, records a timeout, and
continues.  Completed trials and attempted canonicals are appended
immediately, so ``--resume`` is safe.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import sys
import time
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

from src.factors.serialization import json_default, jsonable
from src.mining.gp import random_series
from src.mining.gp_eval import GPEvalConfig, GPEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "gp.yaml"))
    parser.add_argument("--baseline-config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--membership", default=str(PROJECT_ROOT / "data" / "processed" / "stage10_pit_csi300_membership.parquet"))
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-attempts", type=int, default=20000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "mining" / "stage10_random"))
    return parser.parse_args()


def _yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(row), ensure_ascii=False, default=json_default) + "\n")
        handle.flush()


def _worker(config: GPEvalConfig, conn) -> None:
    try:
        evaluator = GPEvaluator(config)
    except Exception as exc:  # pragma: no cover - startup failure
        conn.send({"fatal": f"{type(exc).__name__}: {exc}"})
        return
    while True:
        try:
            expression = conn.recv()
        except EOFError:
            break
        if expression is None:
            break
        try:
            metrics = evaluator.evaluate(expression)
            conn.send({"canonical": expression, "metrics": metrics.to_dict()})
        except Exception as exc:
            conn.send({"canonical": expression, "error": f"{type(exc).__name__}: {exc}"})


class _Worker:
    def __init__(self, config: GPEvalConfig) -> None:
        self.config = config
        self.ctx = multiprocessing.get_context("spawn")
        self.proc: Any = None
        self.conn: Any = None
        self.restarts = 0
        self._start()

    def _start(self) -> None:
        parent, child = self.ctx.Pipe()
        self.proc = self.ctx.Process(target=_worker, args=(self.config, child), daemon=True)
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

    def evaluate(self, expression: str, timeout: float) -> dict[str, Any]:
        if self.proc is None or not self.proc.is_alive():
            self._restart()
        assert self.conn is not None
        try:
            self.conn.send(expression)
        except (BrokenPipeError, OSError):
            self._restart()
            self.conn.send(expression)
        if not self.conn.poll(timeout):
            self._restart()
            return {"canonical": expression, "error": f"timeout after {timeout:.0f}s"}
        try:
            return self.conn.recv()
        except EOFError:
            self._restart()
            return {"canonical": expression, "error": "worker exited"}

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


def main() -> int:
    args = parse_args()
    payload = _yaml(args.config)
    data_payload = payload.get("data", {})
    fitness_payload = payload.get("fitness", {})
    baseline = _yaml(args.baseline_config).get("data", {})
    config = GPEvalConfig(
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
    attempts_path = output_dir / "attempts.jsonl"
    trials_path = output_dir / "trials.jsonl"
    if not args.resume:
        for path in (attempts_path, trials_path):
            if path.exists():
                path.unlink()

    attempted: set[str] = set()
    trials: list[dict[str, Any]] = []
    if args.resume:
        attempted = {
            str(row.get("canonical") or "").strip()
            for row in _read_jsonl(attempts_path)
            if row.get("canonical")
        }
        trials = [row for row in _read_jsonl(trials_path) if row.get("canonical")]
    seen = set(attempted)
    rng = random.Random(args.seed + len(attempted))
    started = time.time()
    worker = _Worker(config)
    local_attempts = 0
    try:
        while len(attempted) < args.budget and local_attempts < args.max_attempts:
            local_attempts += 1
            node = None
            canonical = ""
            try:
                node = random_series(rng, depth=args.depth, max_nodes=30)
                canonical = node.canonical()
            except Exception:
                node = None
            if node is None or canonical in seen:
                continue
            seen.add(canonical)
            attempted.add(canonical)
            _append_jsonl(attempts_path, {"canonical": canonical})
            result = worker.evaluate(canonical, args.timeout)
            if "metrics" in result:
                metrics = result["metrics"]
                row = {
                    "generation": 0,
                    "origin": "random_search",
                    "expression": canonical,
                    "canonical": canonical,
                    "fitness": metrics.get("fitness"),
                    "rank_ic": metrics.get("rank_ic"),
                    "icir": metrics.get("icir"),
                    "turnover": metrics.get("turnover"),
                    "coverage": metrics.get("coverage"),
                    "n_dates": metrics.get("n_dates"),
                    "complexity": metrics.get("complexity"),
                    "error": metrics.get("error"),
                }
            else:
                row = {
                    "generation": 0,
                    "origin": "random_search",
                    "expression": canonical,
                    "canonical": canonical,
                    "fitness": None,
                    "rank_ic": None,
                    "icir": None,
                    "turnover": None,
                    "coverage": None,
                    "n_dates": None,
                    "complexity": getattr(node, "complexity", None),
                    "error": result.get("error") or result.get("fatal") or "unknown worker error",
                }
            trials.append(row)
            _append_jsonl(trials_path, row)
            if len(attempted) % 50 == 0:
                print(
                    f"[random] attempted={len(attempted)} completed={len(trials)} "
                    f"restarts={worker.restarts} elapsed={time.time() - started:.0f}s",
                    flush=True,
                )
    finally:
        worker.close()

    by_canonical: dict[str, dict[str, Any]] = {}
    for row in trials:
        by_canonical.setdefault(str(row["canonical"]), row)
    trials = list(by_canonical.values())
    frame = pd.DataFrame(trials)
    frame.to_csv(output_dir / "trials.csv", index=False)
    with trials_path.open("w", encoding="utf-8") as handle:
        for row in trials:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, default=json_default) + "\n")
    valid = frame[frame["error"].isna()] if "error" in frame.columns else frame
    best = valid.sort_values("fitness", ascending=False).head(1).to_dict(orient="records")
    summary = {
        "schema_version": "stage10.random_search.v1",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "config": config.__dict__,
        "seed": args.seed,
        "depth": args.depth,
        "budget": args.budget,
        "n_attempts": len(attempted),
        "n_unique_candidates": len(seen),
        "n_completed": len(trials),
        "n_evaluated": len(attempted),
        "n_valid": len(valid),
        "worker_restarts": worker.restarts,
        "runtime_seconds": round(time.time() - started, 1),
        "best": best[0] if best else None,
        "paths": {
            "trials_csv": str(output_dir / "trials.csv"),
            "trials_jsonl": str(trials_path),
            "attempts_jsonl": str(attempts_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
