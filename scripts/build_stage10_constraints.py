#!/usr/bin/env python
"""Build Stage 10 PIT constraints for the extended 2020-2026 OOS.

Stage 2 constraints cover 2010-2020-09-23.  The Stage 10 OOS needs
2020-09-24..2026-10-23; this script combines the existing panel with a freshly
built OOS slice from Dolt limits + Qlib prices.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pit_constraints import build_pit_constraints
from src.data.qlib_io import QlibData


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
        "--old-constraints",
        default=str(PROJECT_ROOT / "data" / "processed" / "pit_csi300_constraints.parquet"),
    )
    parser.add_argument(
        "--oos-limits",
        default=str(PROJECT_ROOT / "data" / "raw" / "stage10" / "dolt" / "final_a_stock_limit_csi300.parquet"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "stage10_pit_csi300_constraints.parquet"),
    )
    parser.add_argument(
        "--summary",
        default=str(PROJECT_ROOT / "artifacts" / "stage10" / "constraints_summary.json"),
    )
    parser.add_argument("--oos-start", default="2020-09-24")
    parser.add_argument("--oos-end", default="2023-06-12")
    args = parser.parse_args()

    membership = pd.read_parquet(args.membership)
    membership["datetime"] = pd.to_datetime(membership["datetime"])
    membership = membership[
        (membership["datetime"] >= pd.Timestamp(args.oos_start))
        & (membership["datetime"] <= pd.Timestamp(args.oos_end))
    ]
    limits = pd.read_parquet(args.oos_limits)
    data = QlibData.load(args.provider_uri)
    oos_panel, oos_summary = build_pit_constraints(data, membership, limits)

    old = pd.read_parquet(args.old_constraints)
    old["date"] = pd.to_datetime(old["date"])
    old = old[old["date"] < pd.Timestamp(args.oos_start)]
    combined = pd.concat([old, oos_panel], ignore_index=True)
    combined = combined.drop_duplicates(["instrument", "date"]).sort_values(["date", "instrument"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    summary = {
        "output": str(output),
        "old_rows": len(old),
        "oos_rows": len(oos_panel),
        "total_rows": len(combined),
        "date_min": str(combined["date"].min().date()),
        "date_max": str(combined["date"].max().date()),
        "oos": oos_summary,
    }
    path = Path(args.summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
