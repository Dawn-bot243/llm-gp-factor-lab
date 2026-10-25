#!/usr/bin/env python
"""Independent pandas check for the MA5 evaluator smoke factor.

This intentionally bypasses Qlib's expression engine: it reads the binary
feature files with :class:`QlibData`, computes ``Mean($close, 5) / $close - 1``
and the next-day label with pandas, then compares daily RankIC against the
evaluator's ``rank_ic.csv``.

Qlib's rolling operators use ``min_periods=1``; the independent calculation
does the same so the comparison is meaningful.  The check exits non-zero when
the maximum per-date RankIC difference exceeds the tolerance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.qlib_io import QlibData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-uri",
        default="D:/qlib_data/cn_data_investment_data/extracted/qlib_bin",
    )
    parser.add_argument(
        "--membership",
        default=str(PROJECT_ROOT / "data" / "processed" / "pit_csi300_membership.parquet"),
    )
    parser.add_argument(
        "--eval-dir",
        default=str(PROJECT_ROOT / "artifacts" / "factor_eval" / "ma5_full"),
    )
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2020-09-23")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-names", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    data = QlibData.load(args.provider_uri, names=("csi300",))
    membership = pd.read_parquet(args.membership)
    membership["datetime"] = pd.to_datetime(membership["datetime"])
    membership["instrument"] = membership["instrument"].astype(str).str.upper()
    keep = set(zip(membership["instrument"], membership["datetime"], strict=True))

    frames: list[pd.DataFrame] = []
    for instrument in sorted(membership["instrument"].unique()):
        close = data.feature(instrument, "close")
        if close.empty:
            continue
        factor = close.rolling(args.window, min_periods=1).mean() / close - 1
        label = close.shift(-2) / close.shift(-1) - 1
        frame = pd.DataFrame({"factor": factor, "fwd_1": label}).loc[start:end]
        frame["instrument"] = instrument
        frames.append(frame.reset_index().rename(columns={"index": "datetime"}))
    if not frames:
        raise SystemExit("no raw close series found")
    panel = pd.concat(frames, ignore_index=True)
    panel["datetime"] = pd.to_datetime(panel["datetime"])
    panel = panel[
        [
            (instrument, date) in keep
            for instrument, date in zip(panel["instrument"], panel["datetime"], strict=True)
        ]
    ]

    independent: dict[pd.Timestamp, float] = {}
    for date, group in panel.dropna(subset=["factor", "fwd_1"]).groupby("datetime"):
        if len(group) < args.min_names:
            continue
        value = group["factor"].corr(group["fwd_1"], method="spearman")
        if pd.notna(value):
            independent[date] = float(value)
    independent_series = pd.Series(independent).sort_index()

    eval_path = Path(args.eval_dir) / "rank_ic.csv"
    if not eval_path.exists():
        raise SystemExit(f"evaluator output not found: {eval_path}")
    evaluated = pd.read_csv(eval_path, parse_dates=["datetime"]).set_index("datetime")["rank_ic"]
    joined = pd.concat({"independent": independent_series, "evaluator": evaluated}, axis=1).dropna()
    if joined.empty:
        raise SystemExit("no common dates between independent and evaluator RankIC")
    diff = (joined["independent"] - joined["evaluator"]).abs()
    result = {
        "n_independent_dates": len(independent_series),
        "n_evaluator_dates": len(evaluated),
        "n_common_dates": len(joined),
        "independent_rank_ic_mean": float(independent_series.mean()),
        "evaluator_rank_ic_mean": float(evaluated.mean()),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "max_diff_date": str(diff.idxmax().date()),
        "tolerance": args.tolerance,
        "passed": bool(diff.max() <= args.tolerance),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
