#!/usr/bin/env python
"""Probe Tushare interface permissions with a handful of calls.

The token is read from ``.env`` and is never printed or written to the report.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(key, None)
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

import tushare as ts


def main() -> None:
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise SystemExit("TUSHARE_TOKEN is missing from .env")
    pro = ts.pro_api(token)

    def probe(name: str, fn: Callable[[], pd.DataFrame]) -> dict[str, Any]:
        try:
            frame = fn()
            result = {
                "interface": name,
                "status": "ok",
                "rows": len(frame),
                "columns": [str(column) for column in frame.columns],
            }
            if not frame.empty:
                result["sample"] = frame.head(2).to_dict("records")
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            if token in message:
                message = message.replace(token, "***")
            result = {"interface": name, "status": "error", "error": message[:500]}
        print(json.dumps(result, ensure_ascii=False, default=str), flush=True)
        time.sleep(0.4)
        return result

    probes: list[tuple[str, Callable[[], pd.DataFrame]]] = [
        ("stock_basic", lambda: pro.stock_basic(list_status="L", fields="ts_code,symbol,name,list_date")),
        ("stock_basic_delisted", lambda: pro.stock_basic(list_status="D", fields="ts_code,symbol,name,list_date,delist_date")),
        ("trade_cal", lambda: pro.trade_cal(exchange="SSE", start_date="20100101", end_date="20200923")),
        ("daily", lambda: pro.daily(ts_code="600000.SH", start_date="20200901", end_date="20200930")),
        ("daily_delisted", lambda: pro.daily(ts_code="600087.SH", start_date="20100101", end_date="20141231")),
        ("adj_factor", lambda: pro.adj_factor(ts_code="600000.SH", start_date="20200901", end_date="20200930")),
        ("stk_limit", lambda: pro.stk_limit(ts_code="600000.SH", start_date="20200901", end_date="20200930")),
        ("suspend_d", lambda: pro.suspend_d(ts_code="600000.SH", start_date="20100101", end_date="20200923")),
        ("namechange", lambda: pro.namechange(ts_code="600000.SH", start_date="20100101", end_date="20200923")),
        ("daily_basic", lambda: pro.daily_basic(ts_code="600000.SH", start_date="20200901", end_date="20200930")),
        ("index_daily", lambda: pro.index_daily(ts_code="000300.SH", start_date="20200901", end_date="20200930")),
        ("index_weight_399300", lambda: pro.index_weight(index_code="399300.SZ", start_date="20200801", end_date="20200831")),
        ("index_weight_000300", lambda: pro.index_weight(index_code="000300.SH", start_date="20200801", end_date="20200831")),
        ("index_member_all", lambda: pro.index_member_all(index_code="399300.SZ")),
    ]

    results = [probe(name, fn) for name, fn in probes]
    out = PROJECT_ROOT / "项目报告" / "tushare_permissions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
