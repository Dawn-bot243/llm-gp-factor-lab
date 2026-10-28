#!/usr/bin/env python
"""Reconcile zzshare limit/ST data with the Dolt panel on the overlap window.

Dolt's ``final_a_stock_limit`` ends 2023-06-12, so the 2020-09-24..2023-06-12
window overlaps the Stage 10 panel.  Fetching a small sample from zzshare for
that window lets us verify the new source before trusting it on 2023-2026.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API = "https://api.zizizaizai.com/v3/market/kline/day"
TOLERANCE = 0.011


def _request(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "llm-gp-factor-lab/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _fetch(ts_code: str, start: str, end: str, cache_dir: Path, retries: int = 8) -> list[dict]:
    path = cache_dir / f"{ts_code}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("rows") or []
    query = urllib.parse.urlencode(
        {
            "get_type": "range",
            "start_date": start.replace("-", ""),
            "end_date": end.replace("-", ""),
            "candle_mode": 0,
        }
    )
    url = f"{API}/{ts_code}?{query}"
    for attempt in range(1, retries + 1):
        try:
            body = _request(url, timeout=60)
            rows = (body.get("data") or {}).get("list") or []
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"ts_code": ts_code, "url": url, "rows": rows}, ensure_ascii=False),
                encoding="utf-8",
            )
            return rows
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries:
                raise
            retry_after = float(exc.headers.get("Retry-After") or 30)
            print(f"  retry {ts_code} in {retry_after:.0f}s", flush=True)
            time.sleep(retry_after)
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instruments",
        default="SZ000001,SH600000,SH600519,SZ300750,SH688981,SH600289",
    )
    parser.add_argument("--start", default="2020-09-24")
    parser.add_argument("--end", default="2023-06-12")
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "data" / "raw" / "zzshare" / "overlap"))
    parser.add_argument(
        "--constraints",
        default=str(PROJECT_ROOT / "data" / "processed" / "stage10_pit_csi300_constraints.parquet"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "artifacts" / "stage10c" / "constraints_validation.json"),
    )
    parser.add_argument("--sleep", type=float, default=6.2)
    args = parser.parse_args()

    instruments = [item.strip().upper() for item in args.instruments.split(",") if item.strip()]
    cache_dir = Path(args.cache_dir)
    base = pd.read_parquet(args.constraints)
    base["date"] = pd.to_datetime(base["date"])
    base = base.set_index(["instrument", "date"]).sort_index()

    details: list[dict] = []
    for index, instrument in enumerate(instruments):
        ts_code = f"{instrument[2:]}.{instrument[:2]}"
        rows = _fetch(ts_code, args.start, args.end, cache_dir)
        if index < len(instruments) - 1:
            time.sleep(args.sleep)
        if not rows:
            details.append({"instrument": instrument, "status": "empty"})
            continue
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d")
        frame["instrument"] = instrument
        frame = frame.set_index(["instrument", "date"]).sort_index()
        joined = frame.join(
            base[
                [
                    "pre_close",
                    "up_limit",
                    "down_limit",
                    "at_limit_up",
                    "at_limit_down",
                    "is_st",
                    "suspended",
                ]
            ],
            how="inner",
            suffixes=("", "_dolt"),
        )
        joined = joined.dropna(subset=["up_limit"])
        if joined.empty:
            details.append({"instrument": instrument, "status": "no_overlap"})
            continue
        up_diff = (joined["high_limit"] - joined["up_limit"]).abs()
        down_diff = (joined["low_limit"] - joined["down_limit"]).abs()
        pre_diff = (joined["prev_close"] - joined["pre_close"]).abs()
        limit_mask = joined["high_limit"].notna() & joined["low_limit"].notna()
        zz_up = (
            (joined["high"] >= joined["high_limit"] - TOLERANCE) & limit_mask
        ).fillna(False)
        zz_down = (
            (joined["low"] <= joined["low_limit"] + TOLERANCE) & limit_mask
        ).fillna(False)
        zz_st = pd.to_numeric(joined["is_st"], errors="coerce").fillna(-1).astype(int)
        dolt_st = pd.to_numeric(joined["is_st_dolt"], errors="coerce").fillna(-1).astype(int)
        details.append(
            {
                "instrument": instrument,
                "status": "ok",
                "n": len(joined),
                "pre_close_match_0.005": float((pre_diff <= 0.0051).mean()),
                "up_limit_match_0.005": float((up_diff <= 0.0051).mean()),
                "down_limit_match_0.005": float((down_diff <= 0.0051).mean()),
                "up_limit_exact": float((up_diff <= 1e-9).mean()),
                "down_limit_exact": float((down_diff <= 1e-9).mean()),
                "max_up_diff": float(up_diff.max()),
                "max_down_diff": float(down_diff.max()),
                "at_limit_up_agreement": float((zz_up.astype(bool) == joined["at_limit_up"].astype(bool)).mean()),
                "at_limit_down_agreement": float((zz_down.astype(bool) == joined["at_limit_down"].astype(bool)).mean()),
                "is_st_agreement": float((zz_st == dolt_st).mean()),
                "suspended_agreement": float(
                    (joined["is_paused"].fillna(0).astype(int) == joined["suspended"].astype(int)).mean()
                ),
                "zz_at_limit_up_days": int(zz_up.sum()),
                "dolt_at_limit_up_days": int(joined["at_limit_up"].sum()),
                "zz_at_limit_down_days": int(zz_down.sum()),
                "dolt_at_limit_down_days": int(joined["at_limit_down"].sum()),
            }
        )
        print(json.dumps(details[-1], ensure_ascii=False), flush=True)

    ok = [item for item in details if item.get("status") == "ok"]

    def _board(name: str) -> str:
        return "gem_20pct" if name.upper().startswith(("SZ30", "SH68")) else "main_board"

    by_board: dict[str, dict[str, float | int | None]] = {}
    for board in ("main_board", "gem_20pct"):
        subset = [item for item in ok if _board(item["instrument"]) == board]
        by_board[board] = {
            "n_instruments": len(subset),
            "n_compared": int(sum(item["n"] for item in subset)),
            "mean_pre_close_match": float(np.mean([item["pre_close_match_0.005"] for item in subset])) if subset else None,
            "mean_up_limit_match": float(np.mean([item["up_limit_match_0.005"] for item in subset])) if subset else None,
            "mean_down_limit_match": float(np.mean([item["down_limit_match_0.005"] for item in subset])) if subset else None,
            "mean_at_limit_up_agreement": float(np.mean([item["at_limit_up_agreement"] for item in subset])) if subset else None,
            "mean_at_limit_down_agreement": float(np.mean([item["at_limit_down_agreement"] for item in subset])) if subset else None,
            "mean_is_st_agreement": float(np.mean([item["is_st_agreement"] for item in subset])) if subset else None,
            "mean_suspended_agreement": float(np.mean([item["suspended_agreement"] for item in subset])) if subset else None,
        }

    summary = {
        "start": args.start,
        "end": args.end,
        "instruments": len(instruments),
        "by_board": by_board,
        "n_compared": int(sum(item["n"] for item in ok)),
        "mean_pre_close_match": float(np.mean([item["pre_close_match_0.005"] for item in ok])) if ok else None,
        "mean_up_limit_match": float(np.mean([item["up_limit_match_0.005"] for item in ok])) if ok else None,
        "mean_down_limit_match": float(np.mean([item["down_limit_match_0.005"] for item in ok])) if ok else None,
        "mean_at_limit_up_agreement": float(np.mean([item["at_limit_up_agreement"] for item in ok])) if ok else None,
        "mean_at_limit_down_agreement": float(np.mean([item["at_limit_down_agreement"] for item in ok])) if ok else None,
        "mean_is_st_agreement": float(np.mean([item["is_st_agreement"] for item in ok])) if ok else None,
        "mean_suspended_agreement": float(np.mean([item["suspended_agreement"] for item in ok])) if ok else None,
        "details": details,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "details"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
