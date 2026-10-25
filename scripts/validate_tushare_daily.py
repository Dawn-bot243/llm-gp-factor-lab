#!/usr/bin/env python
"""Validate the Tushare daily cache against Qlib.

Compares scale-invariant quantities:

* raw returns: Qlib ``$close/$factor`` vs Tushare ``close`` (split dates can differ)
* turnover: Qlib ``$volume*$close`` vs Tushare ``vol*100*close``
  (Tushare ``vol`` is lots, so multiply by 100)
"""
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


def _return_match(qlib_prices: pd.Series, ext_prices: pd.Series, tolerance: float) -> dict[str, Any]:
    q = qlib_prices.pct_change()
    e = ext_prices.pct_change()
    diff = (q - e).abs().dropna()
    if diff.empty:
        return {"pairs": 0, "match_rate": None, "p95_abs_diff": None, "max_abs_diff": None}
    return {
        "pairs": len(diff),
        "match_rate": float((diff <= tolerance).mean()),
        "p95_abs_diff": float(diff.quantile(0.95)),
        "max_abs_diff": float(diff.max()),
    }


def _relative_match(left: pd.Series, right: pd.Series, tolerance: float) -> dict[str, Any]:
    diff = ((left - right).abs() / right.abs().replace(0.0, np.nan)).dropna()
    if diff.empty:
        return {"pairs": 0, "match_rate": None, "p95_rel_diff": None, "max_rel_diff": None}
    return {
        "pairs": len(diff),
        "match_rate": float((diff <= tolerance).mean()),
        "p95_rel_diff": float(diff.quantile(0.95)),
        "max_rel_diff": float(diff.max()),
    }


def _weighted(frame: pd.DataFrame, rate: str, pairs: str) -> float | None:
    subset = frame[frame[pairs].fillna(0) > 0]
    if subset.empty:
        return None
    total = subset[pairs].sum()
    return float((subset[rate].fillna(0) * subset[pairs]).sum() / total) if total else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default=str(PROJECT_ROOT / "data" / "universes" / "csi300_frozen.txt"))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2020-09-23")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "项目报告"))
    parser.add_argument("--return-tolerance", type=float, default=0.005)
    parser.add_argument("--turnover-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    data = QlibData.load("D:/qlib_data/cn_data_v3")
    universe = read_instruments(args.universe)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    rows: list[dict[str, Any]] = []

    for code in universe["instrument"].astype(str):
        path = Path(args.raw_dir) / "tushare" / "daily" / f"{code}.parquet"
        if not path.exists():
            rows.append({"instrument": code, "status": "missing_cache"})
            continue
        ext = pd.read_parquet(path)
        ext["trade_date"] = pd.to_datetime(ext["trade_date"])
        ext = ext.set_index("trade_date").sort_index()
        ext = ext.loc[(ext.index >= start) & (ext.index <= end)]

        ohlcv = data.ohlcv(code)
        if ohlcv.empty:
            rows.append({"instrument": code, "status": "no_qlib_data"})
            continue
        ohlcv = ohlcv.loc[(ohlcv.index >= start) & (ohlcv.index <= end)]
        qlib_raw = ohlcv["close"] / ohlcv["factor"].replace(0.0, np.nan)
        qlib_turnover = ohlcv["volume"] * ohlcv["close"]

        common = qlib_raw.index.intersection(ext.index)
        if len(common) == 0:
            rows.append({"instrument": code, "status": "no_overlap"})
            continue
        q_close = qlib_raw.reindex(common)
        e_close = pd.to_numeric(ext["close"].reindex(common), errors="coerce")
        raw = _return_match(q_close, e_close, args.return_tolerance)
        turnover = _relative_match(
            qlib_turnover.reindex(common),
            e_close * pd.to_numeric(ext["vol"].reindex(common), errors="coerce") * 100.0,
            args.turnover_tolerance,
        )

        qlib_only = qlib_raw.index.difference(ext.index)
        ext_only = ext.index.difference(qlib_raw.index)
        fills = int((qlib_raw.isna() & ext["close"].notna()).sum())
        rows.append(
            {
                "instrument": code,
                "status": "ok",
                "qlib_days": int(qlib_raw.notna().sum()),
                "tushare_days": int(ext["close"].notna().sum()),
                "common_days": len(common),
                "raw_pairs": raw["pairs"],
                "raw_return_match_rate": raw["match_rate"],
                "raw_p95_abs_diff": raw["p95_abs_diff"],
                "raw_max_abs_diff": raw["max_abs_diff"],
                "turnover_pairs": turnover["pairs"],
                "turnover_match_rate": turnover["match_rate"],
                "turnover_p95_rel_diff": turnover["p95_rel_diff"],
                "turnover_max_rel_diff": turnover["max_rel_diff"],
                "qlib_only_days": len(qlib_only),
                "qlib_only_with_price": int(qlib_raw.reindex(qlib_only).notna().sum()),
                "tushare_only_days": len(ext_only),
                "qlib_nan_tushare_fills": fills,
            }
        )

    frame = pd.DataFrame(rows)
    ok = frame[frame["status"] == "ok"] if "status" in frame else frame
    summary = {
        "n_requested": len(frame),
        "n_ok": len(ok),
        "n_missing_cache": int((frame["status"] == "missing_cache").sum()) if "status" in frame else 0,
        "n_no_overlap": int((frame["status"] == "no_overlap").sum()) if "status" in frame else 0,
        "n_instruments_with_fills": int((ok["qlib_nan_tushare_fills"] > 0).sum()) if not ok.empty else 0,
        "total_qlib_nan_tushare_fills": int(ok["qlib_nan_tushare_fills"].sum()) if not ok.empty else 0,
        "total_common_days": int(ok["common_days"].sum()) if not ok.empty else 0,
        "raw_return_match_rate": _weighted(ok, "raw_return_match_rate", "raw_pairs"),
        "turnover_match_rate": _weighted(ok, "turnover_match_rate", "turnover_pairs"),
        "raw_return_tolerance": args.return_tolerance,
        "turnover_tolerance": args.turnover_tolerance,
        "worst_raw": ok.sort_values("raw_return_match_rate").head(10).to_dict("records") if not ok.empty else [],
        "worst_turnover": ok.sort_values("turnover_match_rate").head(10).to_dict("records") if not ok.empty else [],
        "per_stock": frame.to_dict("records"),
    }
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / "tushare_daily_validation.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "n_ok",
                    "n_missing_cache",
                    "n_instruments_with_fills",
                    "total_qlib_nan_tushare_fills",
                    "raw_return_match_rate",
                    "turnover_match_rate",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
