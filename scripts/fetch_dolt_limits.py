#!/usr/bin/env python
"""Fetch exchange limit prices for PIT CSI300 members from DoltHub."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import requests

API = "https://www.dolthub.com/api/v1alpha1/chenditc/investment_data/master?q="
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def _query(query: str) -> dict[str, Any]:
    last_error = None
    for attempt in range(4):
        session = _session()
        try:
            response = session.get(API + urllib.parse.quote(query), timeout=(20, 120))
            if response.status_code == 429:
                raise RuntimeError("rate limited")
            response.raise_for_status()
            payload = response.json()
            status = payload.get("query_execution_status")
            if status not in {"Success", "RowLimit"}:
                raise RuntimeError(f"{status}: {payload.get('query_execution_message')}")
            return payload
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(last_error or "query failed")


def _fetch_chunk(
    dates: list[str],
    membership_groups: dict[str, list[str]],
    cache_dir: Path,
    refresh: bool,
) -> dict[str, Any]:
    missing = [
        date for date in dates
        if refresh or not (cache_dir / f"{date}.json").exists()
    ]
    if not missing:
        rows = 0
        for date in dates:
            payload = json.loads((cache_dir / f"{date}.json").read_text(encoding="utf-8"))
            rows += len(payload.get("rows", []))
        return {"dates": dates, "rows": rows, "status": "cache"}
    codes = sorted({code for date in missing for code in membership_groups[date]})
    start, end = min(missing), max(missing)
    base_query = (
        "SELECT tradedate,symbol,pre_close,up_limit,down_limit "
        f"FROM final_a_stock_limit WHERE tradedate BETWEEN '{start}' AND '{end}' "
        "AND symbol IN (" + ",".join(f"'{code}'" for code in codes) + ")"
    )
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        query = base_query + f" LIMIT 1000 OFFSET {offset}"
        payload = _query(query)
        page = payload.get("rows", [])
        rows.extend(page)
        if len(page) < 1000:
            break
        offset += 1000
        if offset >= 5000:
            break
    by_date: dict[str, list[dict[str, Any]]] = {date: [] for date in missing}
    allowed = {date: set(membership_groups[date]) for date in missing}
    for row in rows:
        date = str(row.get("tradedate"))
        if date in by_date and row.get("symbol") in allowed[date]:
            by_date[date].append(row)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for date in missing:
        (cache_dir / f"{date}.json").write_text(
            json.dumps({"rows": by_date[date]}, ensure_ascii=False), encoding="utf-8"
        )
    return {"dates": dates, "rows": sum(len(value) for value in by_date.values()), "status": "fetched"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--membership", default=str(PROJECT_ROOT / "data" / "processed" / "pit_csi300_membership.parquet")
    )
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2020-09-23")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-days", type=int, default=3)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)

    membership = pd.read_parquet(args.membership)
    membership["datetime"] = pd.to_datetime(membership["datetime"])
    membership = membership.loc[
        (membership["datetime"] >= pd.Timestamp(args.start))
        & (membership["datetime"] <= pd.Timestamp(args.end))
    ]
    groups = {
        str(date.date()): sorted(frame["instrument"].astype(str).tolist())
        for date, frame in membership.groupby("datetime")
    }
    cache_dir = Path(args.raw_dir) / "dolt" / "limits"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(groups)
    chunks = [
        ordered[index:index + args.chunk_days]
        for index in range(0, len(ordered), args.chunk_days)
    ]
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    print(
        f"[dolt] dates={len(groups)} chunks={len(chunks)} workers={args.workers}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_fetch_chunk, chunk, groups, cache_dir, args.refresh): chunk
            for chunk in chunks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()
            except Exception as exc:
                result = {"dates": futures[future], "rows": 0, "status": "error", "error": str(exc)}
            results.append(result)
            if result.get("status") == "error":
                errors.append(result)
            if index == 1 or index % 50 == 0 or index == len(futures):
                print(
                    f"[dolt] {index}/{len(futures)} rows={sum(r.get('rows', 0) for r in results)} "
                    f"errors={len(errors)}",
                    flush=True,
                )

    rows: list[dict[str, Any]] = []
    dates_ok = 0
    for date in ordered:
        path = cache_dir / f"{date}.json"
        if path.exists():
            dates_ok += 1
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(payload.get("rows", []))
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.rename(columns={"tradedate": "date", "symbol": "instrument"})
        for column in ("pre_close", "up_limit", "down_limit"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["limit_ratio"] = frame["up_limit"] / frame["pre_close"].replace(0.0, pd.NA) - 1.0
        frame["is_st_inferred"] = (
            (frame["limit_ratio"] >= 0.045) & (frame["limit_ratio"] <= 0.055)
        ).astype("int8")
        frame["source"] = "dolt.final_a_stock_limit"
        frame = frame.drop_duplicates(subset=["date", "instrument"]).sort_values(["date", "instrument"])
        output = Path(args.raw_dir) / "dolt" / "final_a_stock_limit_csi300.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False)
    else:
        output = Path(args.raw_dir) / "dolt" / "final_a_stock_limit_csi300.parquet"
    manifest = {
        "source": "dolt.final_a_stock_limit",
        "n_dates_requested": len(groups),
        "n_dates_ok": int(dates_ok),
        "n_dates_error": int(len(groups) - dates_ok),
        "n_rows": len(frame),
        "n_instruments": int(frame["instrument"].nunique()) if not frame.empty else 0,
        "output": str(output),
        "errors": errors[:20],
    }
    (Path(args.raw_dir) / "manifests" / "dolt_limits.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
