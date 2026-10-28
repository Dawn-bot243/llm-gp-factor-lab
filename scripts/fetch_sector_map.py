#!/usr/bin/env python
"""Fetch a static Sina sector map for Stage 9 Brinson attribution.

Sina's sector taxonomy is a current, coarse classification (49 sectors).  It
is not a PIT industry history and is used only for descriptive attribution;
the Stage 9 report records this limitation.  No credentials are involved.

Usage:
    .venv/Scripts/python.exe -I scripts/fetch_sector_map.py
"""
from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "sector_map.parquet"


def _to_qlib_instrument(symbol: str) -> str | None:
    value = str(symbol).strip().lower()
    if len(value) < 3:
        return None
    exchange = value[:2]
    code = value[2:]
    if exchange not in {"sh", "sz", "bj"} or not code.isdigit():
        return None
    return f"{exchange.upper()}{code}"


def main() -> int:
    os.environ.setdefault("NO_PROXY", "*")
    import akshare as ak

    sectors = ak.stock_sector_spot(indicator="新浪行业")
    if sectors is None or sectors.empty:
        print("no Sina sectors returned")
        return 1
    print(f"sectors: {len(sectors)}")

    rows: list[dict[str, object]] = []
    for index, sector in enumerate(sectors.itertuples(index=False), start=1):
        label = str(sector.label)
        name = str(sector[1])
        try:
            detail = ak.stock_sector_detail(sector=label)
        except Exception as exc:  # pragma: no cover - network dependent
            print(f"[{index}/{len(sectors)}] {label}: {type(exc).__name__}: {exc}")
            continue
        if detail is None or detail.empty:
            continue
        for row in detail.itertuples(index=False):
            instrument = _to_qlib_instrument(getattr(row, "symbol", ""))
            if instrument is None:
                continue
            rows.append(
                {
                    "instrument": instrument,
                    "code": str(getattr(row, "code", "")),
                    "name": str(getattr(row, "name", "")),
                    "sector": name,
                    "sector_label": label,
                    "source": "akshare.sina_sector.current",
                    "fetched_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
                }
            )
        print(f"[{index}/{len(sectors)}] {label} {name}: {len(detail)} rows")
        time.sleep(0.3)

    if not rows:
        print("no sector constituents returned")
        return 1
    frame = pd.DataFrame(rows)
    counts = frame.groupby("instrument")["sector"].nunique().rename("n_sectors")
    frame = frame.drop_duplicates("instrument", keep="first").merge(
        counts, left_on="instrument", right_index=True, how="left"
    )
    frame = frame.sort_values("instrument").reset_index(drop=True)
    frame.to_parquet(DEFAULT_OUTPUT, index=False)
    print(
        f"saved {len(frame)} instruments, {frame['sector'].nunique()} sectors "
        f"(multi-sector: {(frame['n_sectors'] > 1).sum()}) -> {DEFAULT_OUTPUT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
