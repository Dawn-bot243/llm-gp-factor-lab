#!/usr/bin/env python
"""Quick structural probe for a Qlib provider (investment_data candidate)."""
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


def _official_days() -> pd.DatetimeIndex | None:
    path = PROJECT_ROOT / "data" / "raw" / "baostock" / "trade_dates.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    frame["calendar_date"] = pd.to_datetime(frame["calendar_date"])
    return pd.DatetimeIndex(frame.loc[frame["is_trading_day"] == 1, "calendar_date"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", required=True)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2020-09-23")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "项目报告" / "investment_data_probe.json"))
    args = parser.parse_args()
    provider = Path(args.provider_uri)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)

    print(f"[probe] {provider}", flush=True)
    for name in ("calendars", "instruments", "features"):
        path = provider / name
        print(f"[probe] {name}: exists={path.exists()} entries={len(list(path.iterdir())) if path.exists() else 0}")
    data = QlibData.load(provider)
    calendar = data.calendar
    window = calendar[(calendar >= start) & (calendar <= end)]
    all_inst = data.instruments.get("all")
    official = _official_days()
    missing = official.difference(window) if official is not None else None
    missing = missing[(missing >= start) & (missing <= end)] if missing is not None else None

    report: dict[str, Any] = {
        "provider_uri": str(provider),
        "calendar_start": str(calendar.min().date()),
        "calendar_end": str(calendar.max().date()),
        "calendar_days": len(calendar),
        "window_days": len(window),
        "official_trading_days": len(official[(official >= start) & (official <= end)])
        if official is not None
        else None,
        "missing_official_days": len(missing) if missing is not None else None,
        "missing_official_dates": [str(d.date()) for d in missing[:50]] if missing is not None else [],
        "instruments": {},
        "files": sorted(path.name for path in provider.iterdir()),
    }
    if all_inst is not None:
        report["instruments"] = {
            "n": int(all_inst["instrument"].nunique()),
            "duplicates": int(all_inst["instrument"].duplicated().sum()),
            "min_start": str(all_inst["start"].min().date()),
            "max_end": str(all_inst["end"].max().date()),
            "end_counts": all_inst["end"].dt.strftime("%Y-%m-%d").value_counts().head(10).to_dict(),
        }
    for name in ("csi300", "csi300_frozen"):
        path = provider / "instruments" / f"{name}.txt"
        if path.exists():
            frame = read_instruments(path)
            report["instruments"][name] = {
                "n": int(frame["instrument"].nunique()),
                "min_start": str(frame["start"].min().date()),
                "max_end": str(frame["end"].max().date()),
            }
    fields = {}
    for code in ["SH600000", "SH600300", "SZ000001", "SZ002493"]:
        fields[code] = [field for field in ("open", "high", "low", "close", "volume", "factor") if data.feature_path(code, field).exists()]
    report["fields"] = fields
    for code in ["SH600000", "SH000300"]:
        frame = data.ohlcv(code)
        if frame.empty:
            continue
        raw = frame["close"] / frame["factor"].replace(0, np.nan)
        report[f"around_2020_09_28_{code}"] = {
            str(index.date()): {
                "close": float(row["close"]) if pd.notna(row["close"]) else None,
                "factor": float(row["factor"]) if pd.notna(row["factor"]) else None,
                "raw": float(raw.loc[index]) if pd.notna(raw.loc[index]) else None,
            }
            for index, row in frame.loc["2020-09-24":"2020-09-30"].iterrows()
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
