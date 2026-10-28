#!/usr/bin/env python
"""Build Stage 10.1 OOS constraints for 2023-06-13..2026-10-23.

Dolt's ``final_a_stock_limit`` table ends 2023-06-12, so the new OOS cannot
use exchange-published limit prices.  This script derives *approximate* limit
flags from Qlib raw high/low versus previous raw close, using board-specific
limit ratios:

* main board 10%;
* ChiNext/STAR (SZ300/SZ301/SH688/SH689) 20%;
* Beijing Stock Exchange 30%;
* ST is unknown and is not flagged.

The derived flags are conservative and are documented as a partial execution
constraint in the revised Stage 10 report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.qlib_io import QlibData


def _limit_ratio(instrument: str) -> float:
    code = instrument.upper()
    if code.startswith("BJ"):
        return 0.30
    if code.startswith(("SH688", "SH689", "SZ300", "SZ301")):
        return 0.20
    return 0.10


def _raw(data: QlibData, instrument: str, field: str) -> pd.Series:
    series = data.feature(instrument, field)
    factor = data.feature(instrument, "factor")
    frame = pd.concat({"value": series, "factor": factor}, axis=1)
    return (frame["value"] / frame["factor"].replace(0.0, np.nan)).rename(instrument)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-uri",
        default="D:/qlib_data/cn_data_investment_data/extracted/qlib_bin",
    )
    parser.add_argument(
        "--membership",
        default=str(PROJECT_ROOT / "data" / "processed" / "stage10_pit_csi300_membership.parquet"),
    )
    parser.add_argument(
        "--base-constraints",
        default=str(PROJECT_ROOT / "data" / "processed" / "stage10_pit_csi300_constraints.parquet"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "stage10b_pit_csi300_constraints.parquet"),
    )
    parser.add_argument(
        "--summary",
        default=str(PROJECT_ROOT / "artifacts" / "stage10b" / "constraints_summary.json"),
    )
    parser.add_argument("--start", default="2023-06-13")
    parser.add_argument("--end", default="2026-10-23")
    args = parser.parse_args()

    membership = pd.read_parquet(args.membership)
    membership["instrument"] = membership["instrument"].astype(str).str.upper()
    membership["datetime"] = pd.to_datetime(membership["datetime"])
    membership = membership[
        (membership["datetime"] >= pd.Timestamp(args.start))
        & (membership["datetime"] <= pd.Timestamp(args.end))
    ]
    data = QlibData.load(args.provider_uri)
    rows: list[pd.DataFrame] = []
    for instrument, group in membership.groupby("instrument", sort=True):
        dates = pd.DatetimeIndex(group["datetime"].sort_values())
        close = _raw(data, instrument, "close").reindex(dates)
        high = _raw(data, instrument, "high").reindex(dates)
        low = _raw(data, instrument, "low").reindex(dates)
        # raw close is needed before the first membership date for pre_close.
        close_full = _raw(data, instrument, "close")
        pre_close_full = close_full.shift(1)
        ratio = _limit_ratio(instrument)
        up_limit = pre_close_full * (1.0 + ratio)
        down_limit = pre_close_full * (1.0 - ratio)
        frame = pd.DataFrame(
            {
                "instrument": instrument,
                "date": dates,
                "suspended": close.isna().astype("int8"),
                "is_st": 0,
                "is_st_known": False,
                "pre_close": pre_close_full.reindex(dates).to_numpy(),
                "up_limit": up_limit.reindex(dates).to_numpy(),
                "down_limit": down_limit.reindex(dates).to_numpy(),
                "limit_ratio": ratio,
                "at_limit_up": (
                    high >= (up_limit.reindex(dates) - 0.011)
                ).fillna(False).to_numpy(dtype=bool),
                "at_limit_down": (
                    low <= (down_limit.reindex(dates) + 0.011)
                ).fillna(False).to_numpy(dtype=bool),
                "source": "derived.qlib_raw_price_limit",
            }
        )
        rows.append(frame)

    derived = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    base = pd.read_parquet(args.base_constraints)
    base["date"] = pd.to_datetime(base["date"])
    base = base[base["date"] < pd.Timestamp(args.start)]
    combined = pd.concat([base, derived], ignore_index=True)
    combined = combined.drop_duplicates(["instrument", "date"]).sort_values(["date", "instrument"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    summary = {
        "output": str(output),
        "base_rows": len(base),
        "derived_rows": len(derived),
        "total_rows": len(combined),
        "date_min": str(combined["date"].min().date()),
        "date_max": str(combined["date"].max().date()),
        "derived": {
            "n_instruments": int(derived["instrument"].nunique()) if not derived.empty else 0,
            "n_suspended_days": int(derived["suspended"].sum()) if not derived.empty else 0,
            "n_limit_up_days": int(derived["at_limit_up"].sum()) if not derived.empty else 0,
            "n_limit_down_days": int(derived["at_limit_down"].sum()) if not derived.empty else 0,
            "st_status": "unknown_not_flagged",
            "limit_source": "Qlib raw high/low vs previous raw close; board ratio 10/20/30%",
        },
    }
    path = Path(args.summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
