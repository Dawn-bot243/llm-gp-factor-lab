#!/usr/bin/env python
"""Build the Stage 10 extended PIT CSI300 membership, 2010-2026.

The Qlib snapshot's ``instruments/csi300.txt`` contains dated membership
intervals through 2026-10-23.  Stage 1-9 froze the research sample at
2020-09-23; Stage 10 uses 2020-09-24..2026-10-23 as a genuinely untouched OOS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pit import build_pit_membership, membership_summary
from src.data.qlib_io import read_calendar, read_instruments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider-uri",
        default="D:/qlib_data/cn_data_investment_data/extracted/qlib_bin",
    )
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2026-10-23")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "stage10_pit_csi300_membership.parquet"),
    )
    parser.add_argument(
        "--old-membership",
        default=str(PROJECT_ROOT / "data" / "processed" / "pit_csi300_membership.parquet"),
    )
    parser.add_argument(
        "--summary",
        default=str(PROJECT_ROOT / "artifacts" / "stage10" / "universe_summary.json"),
    )
    parser.add_argument(
        "--oos-end",
        default="2023-06-12",
        help="OOS end used in the summary; membership still extends to --end",
    )
    args = parser.parse_args()

    provider = Path(args.provider_uri)
    calendar = read_calendar(provider)
    intervals = read_instruments(provider / "instruments" / "csi300.txt")
    membership = build_pit_membership(intervals, calendar, args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    membership.to_parquet(output, index=False)
    summary = {
        "provider_uri": str(provider),
        "instrument_file": str(provider / "instruments" / "csi300.txt"),
        "start": args.start,
        "end": args.end,
        "output": str(output),
        **membership_summary(membership),
    }

    old_path = Path(args.old_membership)
    if old_path.exists():
        old = pd.read_parquet(old_path)
        old["datetime"] = pd.to_datetime(old["datetime"])
        overlap = membership.merge(old, on=["instrument", "datetime"], how="inner")
        summary["overlap_with_stage2"] = {
            "stage2_rows": len(old),
            "overlap_rows": len(overlap),
            "stage2_only_rows": len(old) - len(overlap),
            "extended_only_rows": len(membership) - len(overlap),
            "stage2_instrument_overlap": len(set(old["instrument"]) & set(membership["instrument"])),
        }

    counts = membership.groupby("datetime").size()
    summary["days_not_300"] = int((counts != 300).sum())
    summary["member_count_distribution"] = counts.value_counts().head(10).to_dict()
    summary["oos_dates"] = {
        "start": "2020-09-24",
        "end": args.oos_end,
        "n_dates": int(
            membership.loc[membership["datetime"] <= pd.Timestamp(args.oos_end), "datetime"]
            .nunique()
            - membership.loc[membership["datetime"] < pd.Timestamp("2020-09-24"), "datetime"].nunique()
        ),
    }
    path = Path(args.summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
