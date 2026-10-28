#!/usr/bin/env python
"""Fetch daily bars with exchange limit prices from the zzshare open API.

zzshare (https://github.com/zzquant/zzshare) exposes a free, token-optional
daily endpoint that includes ``high_limit``, ``low_limit``, ``is_paused`` and
``is_st``.  It is used here to extend the PIT constraints past 2023-06-12,
where Dolt's ``final_a_stock_limit`` table stops.

The anonymous quota is 10 requests/minute.  The fetcher is deliberately
single-threaded, sleeps between requests, honours ``Retry-After`` on 429 and
caches one JSON file per instrument so a run can be resumed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API = "https://api.zizizaizai.com/v3/market/kline/day"
USER_AGENT = "llm-gp-factor-lab/1.0 (research; zzshare open API)"


def _ts_code(instrument: str) -> str:
    code = instrument.upper()
    if code.startswith(("SH", "SZ", "BJ")):
        return f"{code[2:]}.{code[:2]}"
    return code


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"source": "zzshare", "instruments": {}}


def _write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _request(url: str, timeout: float) -> tuple[dict, int, str | None]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
            return body, response.status, None
    except urllib.error.HTTPError as exc:
        retry_after = exc.headers.get("Retry-After")
        message = exc.read().decode("utf-8", "replace")[:400]
        return {}, exc.code, f"{exc.code}: {message} (retry-after={retry_after})"


def fetch_one(
    ts_code: str,
    start: str,
    end: str,
    cache_dir: Path,
    timeout: float,
    max_retries: int,
    interval: float,
) -> dict:
    query = urllib.parse.urlencode(
        {
            "get_type": "range",
            "start_date": start.replace("-", ""),
            "end_date": end.replace("-", ""),
            "candle_mode": 0,
        }
    )
    url = f"{API}/{ts_code}?{query}"
    cache_path = cache_dir / f"{ts_code}.json"
    attempts = 0
    while True:
        attempts += 1
        body, status, error = _request(url, timeout)
        if status == 200 and isinstance(body, dict) and body.get("code") == 200:
            data = body.get("data") or {}
            rows = data.get("list") or []
            payload = {
                "fetched_at": datetime.now(UTC).isoformat(),
                "source": "zzshare.v3.market.kline.day",
                "url": url,
                "ts_code": ts_code,
                "count": len(rows),
                "rows": rows,
            }
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(cache_path)
            return {
                "status": "ok" if rows else "empty",
                "rows": len(rows),
                "attempts": attempts,
                "first": rows[0].get("trade_date") if rows else None,
                "last": rows[-1].get("trade_date") if rows else None,
                "sha256": _sha256(cache_path),
            }
        if status == 429 or (status or 500) >= 500:
            retry_after = 60.0 / max(interval, 0.1)
            if error and "retry-after=" in error:
                raw = error.split("retry-after=")[-1].rstrip(")")
                try:
                    retry_after = max(retry_after, float(raw))
                except ValueError:
                    pass
            if attempts > max_retries:
                return {"status": "error", "rows": 0, "attempts": attempts, "error": error}
            print(f"  retry {ts_code} in {retry_after:.1f}s ({error})", flush=True)
            time.sleep(retry_after)
            continue
        return {"status": "error", "rows": 0, "attempts": attempts, "error": error or f"http {status}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--membership",
        default=str(PROJECT_ROOT / "data" / "processed" / "stage10_pit_csi300_membership.parquet"),
    )
    parser.add_argument("--start", default="2023-06-13")
    parser.add_argument("--end", default="2026-10-23")
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "data" / "raw" / "zzshare" / "daily"))
    parser.add_argument(
        "--manifest", default=str(PROJECT_ROOT / "data" / "raw" / "zzshare" / "manifest.json")
    )
    parser.add_argument("--rate", type=float, default=10.0, help="requests per minute")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--max-requests", type=int, default=0, help="smoke mode: stop after N requests")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    import pandas as pd

    membership = pd.read_parquet(args.membership)
    membership["datetime"] = pd.to_datetime(membership["datetime"])
    membership = membership[
        (membership["datetime"] >= pd.Timestamp(args.start))
        & (membership["datetime"] <= pd.Timestamp(args.end))
    ]
    instruments = sorted(membership["instrument"].astype(str).str.upper().unique())
    codes = [_ts_code(instrument) for instrument in instruments]
    code_to_instrument = dict(zip(codes, instruments, strict=True))

    cache_dir = Path(args.cache_dir)
    manifest_path = Path(args.manifest)
    manifest = _load_manifest(manifest_path)
    manifest.update(
        {
            "source": "zzshare.v3.market.kline.day",
            "api": API,
            "start": args.start,
            "end": args.end,
            "rate_per_minute": args.rate,
            "requested_instruments": len(codes),
        }
    )
    interval = 60.0 / max(args.rate, 0.1)

    pending: list[str] = []
    for code in codes:
        entry = manifest["instruments"].get(code, {})
        cache_path = cache_dir / f"{code}.json"
        if (
            not args.refresh
            and cache_path.exists()
            and entry.get("status") in {"ok", "empty"}
        ):
            continue
        pending.append(code)

    print(
        f"[zzshare] instruments={len(codes)} cached={len(codes) - len(pending)} "
        f"pending={len(pending)} rate={args.rate}/min",
        flush=True,
    )
    started = time.monotonic()
    for index, code in enumerate(pending, start=1):
        if args.max_requests and index > args.max_requests:
            print(f"[zzshare] stop after {args.max_requests} requests (smoke)", flush=True)
            break
        result = fetch_one(
            code,
            args.start,
            args.end,
            cache_dir,
            args.timeout,
            args.max_retries,
            interval,
        )
        result["instrument"] = code_to_instrument.get(code, "")
        result["fetched_at"] = datetime.now(UTC).isoformat()
        manifest["instruments"][code] = result
        _write_manifest(manifest_path, manifest)
        elapsed = time.monotonic() - started
        eta = (len(pending) - index) * interval / 60.0
        print(
            f"[zzshare] {index}/{len(pending)} {code} {result['status']} "
            f"rows={result['rows']} elapsed={elapsed / 60:.1f}m eta={eta:.1f}m",
            flush=True,
        )
        if index < len(pending):
            time.sleep(interval)

    counts: dict[str, int] = {}
    for entry in manifest["instruments"].values():
        counts[entry.get("status", "unknown")] = counts.get(entry.get("status", "unknown"), 0) + 1
    manifest["status_counts"] = counts
    manifest["completed_at"] = datetime.now(UTC).isoformat()
    _write_manifest(manifest_path, manifest)
    print(json.dumps({"status_counts": counts, "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
