#!/usr/bin/env python
"""Compare two Qlib providers on returns/turnover for the frozen universe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.qlib_io import QlibData, read_instruments


def _match(left: pd.Series, right: pd.Series, tolerance: float) -> dict[str, Any]:
    diff = (left - right).abs().dropna()
    if diff.empty:
        return {"pairs": 0, "match_rate": None, "p95": None, "max": None}
    passed = diff <= tolerance
    return {
        "pairs": len(diff),
        "match_rate": float(passed.mean()),
        "p95": float(diff.quantile(0.95)),
        "max": float(diff.max()),
    }


def _weighted(frame: pd.DataFrame, rate: str, pairs: str) -> float | None:
    subset = frame[frame[pairs].fillna(0) > 0]
    if subset.empty:
        return None
    total = subset[pairs].sum()
    return float((subset[rate].fillna(0) * subset[pairs]).sum() / total) if total else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", required=True)
    parser.add_argument("--old", default="D:/qlib_data/cn_data_v3")
    parser.add_argument("--universe", default=str(PROJECT_ROOT / "data" / "universes" / "csi300_frozen.txt"))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2020-09-23")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "项目报告" / "provider_comparison.json"))
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    new = QlibData.load(args.new)
    old = QlibData.load(args.old)
    universe = read_instruments(args.universe)["instrument"].astype(str).tolist()
    rows: list[dict[str, Any]] = []

    for code in universe:
        old_frame = old.ohlcv(code)
        new_frame = new.ohlcv(code)
        if old_frame.empty or new_frame.empty:
            rows.append({"instrument": code, "status": "missing_one_side"})
            continue
        old_frame = old_frame.loc[(old_frame.index >= start) & (old_frame.index <= end)]
        new_frame = new_frame.loc[(new_frame.index >= start) & (new_frame.index <= end)]
        common = old_frame.index.intersection(new_frame.index)
        if len(common) == 0:
            rows.append({"instrument": code, "status": "no_overlap"})
            continue
        old_raw = old_frame["close"] / old_frame["factor"].replace(0.0, np.nan)
        new_raw = new_frame["close"] / new_frame["factor"].replace(0.0, np.nan)
        old_turnover = old_frame["volume"] * old_frame["close"]
        new_turnover = new_frame["volume"] * new_frame["close"]
        raw = _match(
            old_raw.reindex(common).pct_change(),
            new_raw.reindex(common).pct_change(),
            0.005,
        )
        # Turnover is a level, so compare relative differences.
        rel = (
            (old_turnover.reindex(common) - new_turnover.reindex(common)).abs()
            / new_turnover.reindex(common).abs().replace(0.0, np.nan)
        ).dropna()
        turnover = {
            "pairs": len(rel),
            "match_rate": float((rel <= 0.01).mean()) if len(rel) else None,
            "p95": float(rel.quantile(0.95)) if len(rel) else None,
            "max": float(rel.max()) if len(rel) else None,
        }
        rows.append(
            {
                "instrument": code,
                "status": "ok",
                "common_days": len(common),
                "old_only": len(old_frame.index.difference(new_frame.index)),
                "new_only": len(new_frame.index.difference(old_frame.index)),
                "raw_pairs": raw["pairs"],
                "raw_return_match_rate": raw["match_rate"],
                "raw_p95": raw["p95"],
                "raw_max": raw["max"],
                "turnover_pairs": turnover["pairs"],
                "turnover_match_rate": turnover["match_rate"],
                "turnover_p95": turnover["p95"],
                "turnover_max": turnover["max"],
            }
        )

    frame = pd.DataFrame(rows)
    ok = frame[frame["status"] == "ok"] if "status" in frame else frame
    report = {
        "new_provider": args.new,
        "old_provider": args.old,
        "new_calendar": [str(new.calendar.min().date()), str(new.calendar.max().date()), len(new.calendar)],
        "old_calendar": [str(old.calendar.min().date()), str(old.calendar.max().date()), len(old.calendar)],
        "new_instruments": {
            name: int(value["instrument"].nunique()) for name, value in new.instruments.items()
        },
        "old_instruments": {
            name: int(value["instrument"].nunique()) for name, value in old.instruments.items()
        },
        "n_compared": len(ok),
        "n_missing_one_side": int((frame["status"] == "missing_one_side").sum()) if "status" in frame else 0,
        "raw_return_match_rate": _weighted(ok, "raw_return_match_rate", "raw_pairs"),
        "turnover_match_rate": _weighted(ok, "turnover_match_rate", "turnover_pairs"),
        "worst_raw": ok.sort_values("raw_return_match_rate").head(10).to_dict("records") if not ok.empty else [],
        "worst_turnover": ok.sort_values("turnover_match_rate").head(10).to_dict("records") if not ok.empty else [],
        "per_stock": frame.to_dict("records"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "n_compared": report["n_compared"],
                "raw_return_match_rate": report["raw_return_match_rate"],
                "turnover_match_rate": report["turnover_match_rate"],
                "new_calendar": report["new_calendar"],
                "old_calendar": report["old_calendar"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
