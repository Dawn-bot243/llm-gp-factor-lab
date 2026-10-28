#!/usr/bin/env python
"""Circular block bootstrap for the Stage 10C OOS net-excess series.

The OOS is a single window, so point estimates alone cannot distinguish a real
group difference from daily-return noise.  This script resamples blocks of
daily strategy and benchmark returns and reports 95% intervals for the
geometric annualised excess (the same definition as ``excess_annual_return``),
the net-excess IR, and selected group differences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["random", "gp_only", "llm_only", "llm_gp"]
LABELS = {
    "random": "A random",
    "gp_only": "B GP-only",
    "llm_only": "C LLM-only",
    "llm_gp": "D LLM+GP",
}
ANNUALIZATION = 252


def _block_indices(n: int, block: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    return indices.reshape(n_boot, -1)[:, :n]


def _stats(returns: np.ndarray, benchmark: np.ndarray) -> tuple[float, float]:
    n = len(returns)
    port_total = float(np.prod(1.0 + returns) ** (ANNUALIZATION / n) - 1.0)
    bench_total = float(np.prod(1.0 + benchmark) ** (ANNUALIZATION / n) - 1.0)
    excess = returns - benchmark
    std = float(np.std(excess, ddof=1))
    ir = float(np.mean(excess) / std * np.sqrt(ANNUALIZATION)) if std > 0 else float("nan")
    return port_total - bench_total, ir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default=str(PROJECT_ROOT / "artifacts" / "stage10c"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--block", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-days", type=int, default=2, help="match portfolio metric warm-up")
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "artifacts" / "stage10c" / "bootstrap_oos.json")
    )
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    returns: dict[str, pd.Series] = {}
    benchmarks: dict[str, pd.Series] = {}
    for group in GROUPS:
        path = artifacts / f"oos_{group}_daily.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime").sort_index()
        returns[group] = pd.to_numeric(frame["return"], errors="coerce")
        benchmarks[group] = pd.to_numeric(frame["benchmark_return"], errors="coerce")
    if not returns:
        raise RuntimeError("no oos_*_daily.csv files found")

    frame = pd.DataFrame(returns)
    benchmark = pd.DataFrame(benchmarks).bfill(axis=1).iloc[:, 0]
    valid = frame.notna().all(axis=1) & benchmark.notna()
    frame = frame.loc[valid]
    benchmark = benchmark.loc[valid]
    drop = min(max(int(args.drop_days), 0), max(len(frame) - 1, 0))
    frame = frame.iloc[drop:]
    benchmark = benchmark.iloc[drop:]
    rng = np.random.default_rng(args.seed)
    indices = _block_indices(len(frame), args.block, args.n_boot, rng)

    summary: dict[str, Any] = {
        "n_days": len(frame),
        "block": args.block,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "groups": {},
        "differences": {},
    }
    boot: dict[str, np.ndarray] = {}
    for group in frame.columns:
        values = frame[group].to_numpy(dtype=float)
        bench = benchmark.to_numpy(dtype=float)
        point_excess, point_ir = _stats(values, bench)
        boots = np.array([_stats(values[index], bench[index]) for index in indices])
        boot[group] = boots
        summary["groups"][group] = {
            "label": LABELS.get(group, group),
            "annual_excess": point_excess,
            "annual_excess_ci_low": float(np.percentile(boots[:, 0], 2.5)),
            "annual_excess_ci_high": float(np.percentile(boots[:, 0], 97.5)),
            "net_excess_ir": point_ir,
            "net_excess_ir_ci_low": float(np.percentile(boots[:, 1], 2.5)),
            "net_excess_ir_ci_high": float(np.percentile(boots[:, 1], 97.5)),
        }

    pairs = [("llm_gp", "gp_only"), ("random", "llm_only")]
    for left, right in pairs:
        if left not in boot or right not in boot:
            continue
        diff = boot[left][:, 0] - boot[right][:, 0]
        ir_diff = boot[left][:, 1] - boot[right][:, 1]
        summary["differences"][f"{left}_minus_{right}"] = {
            "annual_excess_diff": summary["groups"][left]["annual_excess"]
            - summary["groups"][right]["annual_excess"],
            "annual_excess_diff_ci_low": float(np.percentile(diff, 2.5)),
            "annual_excess_diff_ci_high": float(np.percentile(diff, 97.5)),
            "ir_diff": summary["groups"][left]["net_excess_ir"]
            - summary["groups"][right]["net_excess_ir"],
            "ir_diff_ci_low": float(np.percentile(ir_diff, 2.5)),
            "ir_diff_ci_high": float(np.percentile(ir_diff, 97.5)),
            "share_boot_positive": float(np.mean(diff > 0)),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
