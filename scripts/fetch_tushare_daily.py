#!/usr/bin/env python
"""Fetch A-share raw daily bars from Tushare into the local cache."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.qlib_io import read_instruments
from src.data.tushare_source import fetch_daily


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default=str(PROJECT_ROOT / "data" / "universes" / "csi300_frozen.txt"))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2020-09-23")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-stocks", type=int, default=0)
    args = parser.parse_args()

    universe = read_instruments(args.universe)
    codes = universe["instrument"].astype(str).tolist()
    if args.max_stocks > 0:
        codes = codes[: args.max_stocks]

    def progress(code: str, index: int, total: int) -> None:
        if index == 1 or index % 25 == 0 or index == total:
            print(f"[tushare-daily] {index}/{total} {code}", flush=True)

    frames, status = fetch_daily(
        codes,
        args.start,
        args.end,
        args.raw_dir,
        refresh=args.refresh,
        progress=progress,
    )
    summary = {
        "n_requested": len(status),
        "n_cached": int((status["status"] == "cache").sum()),
        "n_fetched": int((status["status"] == "fetched").sum()),
        "n_empty": int((status["status"] == "empty").sum()),
        "n_errors": int((status["status"] == "error").sum()),
        "n_rows": int(status["rows"].sum()),
        "n_instruments_with_data": len(frames),
        "errors": status.loc[status["status"] == "error", ["instrument", "error"]].head(20).to_dict("records"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
