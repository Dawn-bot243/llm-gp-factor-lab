#!/usr/bin/env python
"""Sanity checks for Stage 1 baseline artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def check(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        print(f"OK   {message}")
    else:
        print(f"FAIL {message}")
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", default=str(PROJECT_ROOT / "artifacts" / "baseline_pit"))
    args = parser.parse_args()
    artifacts = Path(args.artifacts)

    failures: list[str] = []
    metrics_path = artifacts / "metrics.json"
    check(metrics_path.exists(), "metrics.json exists", failures)
    if not metrics_path.exists():
        return 1

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    check(metrics.get("test_rows", 0) > 0, "test_rows > 0", failures)
    check(metrics.get("test_dates", 0) > 20, "test_dates > 20", failures)

    rank = metrics.get("rank_ic", {})
    check(math.isfinite(float(rank.get("mean", float("nan")))), "RankIC mean is finite", failures)
    check(-0.3 < float(rank.get("mean", 0.0)) < 0.3, "RankIC mean is in a plausible range", failures)
    check(float(rank.get("n_dates", 0)) > 20, "RankIC has enough dates", failures)

    quant = metrics.get("quantile", {})
    check(len(quant.get("mean_forward_return", {})) == 5, "5 quantile buckets", failures)

    report_path = artifacts / "report_normal.csv"
    check(report_path.exists(), "portfolio report exists", failures)

    for name in ["metrics.json", "rank_ic.csv", "report_normal.csv", "pred_test.csv"]:
        check((artifacts / name).exists(), f"artifact {name} exists", failures)

    if failures:
        print(f"\n{len(failures)} check(s) failed")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
