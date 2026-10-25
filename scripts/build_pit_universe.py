#!/usr/bin/env python
"""Build daily PIT CSI300 membership from the new Qlib provider."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pit import build_pit_membership, membership_summary
from src.data.qlib_io import read_calendar, read_instruments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider-uri", default="D:/qlib_data/cn_data_investment_data/extracted/qlib_bin"
    )
    parser.add_argument("--instrument-file", default=None)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2020-09-23")
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "data" / "processed" / "pit_csi300_membership.parquet")
    )
    args = parser.parse_args()
    provider = Path(args.provider_uri)
    instrument_file = Path(args.instrument_file) if args.instrument_file else provider / "instruments" / "csi300.txt"
    calendar = read_calendar(provider)
    intervals = read_instruments(instrument_file)
    membership = build_pit_membership(intervals, calendar, args.start, args.end)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    membership.to_parquet(output, index=False)
    summary = {
        "provider_uri": str(provider),
        "instrument_file": str(instrument_file),
        "start": args.start,
        "end": args.end,
        "output": str(output),
        **membership_summary(membership),
    }
    reports = PROJECT_ROOT / "项目报告" / "pit_universe_summary.json"
    reports.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
