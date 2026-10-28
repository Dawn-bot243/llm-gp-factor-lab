#!/usr/bin/env python
"""Stage 2 data-quality pipeline.

Checks the Qlib dataset internally, supplements ST / limit / suspension data
from Baostock, reconciles a sample against AkShare (Sina), and writes:

* ``项目报告/data_quality.json``
* ``项目报告/data_quality.md``
* ``项目报告/data_quality_flags.csv``
* ``data/processed/constraints_<universe>.parquet``
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.constraints import build_constraints, prepare_akshare_daily
from src.data.external import (
    build_st_intervals,
    fetch_akshare_daily,
    fetch_baostock_daily,
    fetch_baostock_stock_basic,
    fetch_baostock_trade_dates,
    fetch_sina_name_history,
    fetch_szse_name_changes,
)
from src.data.qlib_io import QlibData, read_instruments
from src.data.quality import (
    check_bin_coverage,
    check_calendar,
    early_end_check,
    factor_consistency,
    price_jumps,
    select_sample,
    universe_counts,
)
from src.data.reconcile import reconcile_all
from src.data.report import render_markdown


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qlib data-quality checks and external reconciliation")
    parser.add_argument(
        "--provider-uri", default="D:/qlib_data/cn_data_investment_data/extracted/qlib_bin"
    )
    parser.add_argument(
        "--universe", default=str(PROJECT_ROOT / "data" / "universes" / "csi300_pit_names.txt")
    )
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2020-09-23")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-stocks", type=int, default=0, help="limit the universe for a quick run (0 = all)")
    parser.add_argument("--skip-external", action="store_true", help="only run Qlib-internal checks")
    parser.add_argument("--refresh", action="store_true", help="ignore cached external data")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "项目报告"))
    parser.add_argument("--processed-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def status_summary(status: pd.DataFrame | None) -> dict[str, Any]:
    if status is None or status.empty:
        return {}
    return {
        "n_requested": len(status),
        "n_cached": int((status["status"] == "cache").sum()),
        "n_fetched": int((status["status"] == "fetched").sum()),
        "n_empty": int((status["status"] == "empty").sum()),
        "n_errors": int((status["status"] == "error").sum()),
        "n_rows": int(status["rows"].sum()),
        "errors": status.loc[status["status"] == "error", ["instrument", "error"]].head(20).to_dict("records"),
    }


def progress(label: str):
    def report(code: str, index: int, total: int) -> None:
        if index == 1 or index % 25 == 0 or index == total:
            print(f"[{label}] {index}/{total} {code}", flush=True)

    return report


def main() -> None:
    args = parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    reports_dir = Path(args.reports_dir)
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    provider_uri = Path(args.provider_uri)
    print(f"[load] {provider_uri}", flush=True)
    data = QlibData.load(provider_uri, names=("all",))
    all_instruments = data.instruments["all"]
    universe_path = Path(args.universe)
    frozen = read_instruments(universe_path)
    universe = frozen.head(args.max_stocks) if args.max_stocks > 0 else frozen
    universe_codes = universe["instrument"].astype(str).tolist()

    report: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
            "provider_uri": str(provider_uri),
            "calendar_start": str(data.calendar.min().date()),
            "calendar_end": str(data.calendar.max().date()),
            "start": str(start.date()),
            "end": str(end.date()),
            "universe_path": str(universe_path),
            "universe_size": len(universe),
            "sample_size": args.sample_size,
            "seed": args.seed,
            "skip_external": bool(args.skip_external),
        },
        "external": {"errors": [], "baostock_daily": {}, "akshare_raw": {}, "akshare_qfq": {}},
    }

    official_trading = None
    if not args.skip_external:
        try:
            print("[calendar] fetching Baostock trading calendar", flush=True)
            official_trading = fetch_baostock_trade_dates(start, end, raw_dir, refresh=args.refresh)
        except Exception as exc:
            report["external"]["errors"].append(f"trade_dates: {type(exc).__name__}: {exc}")

    print("[check] calendar", flush=True)
    report["calendar"] = check_calendar(data.calendar, start, end, official_trading)
    print("[check] universe counts", flush=True)
    report["universe"] = {
        "all": universe_counts(all_instruments, data.calendar, start, end),
        "frozen": universe_counts(universe, data.calendar, start, end),
    }

    print("[check] bin coverage", flush=True)
    coverage_summary, coverage_frame = check_bin_coverage(data, universe, start, end)
    report["bin_coverage"] = coverage_summary

    # Scan a few extra days so a vendor break just after the usable range
    # (notably 2020-09-28) is diagnosed rather than silently ignored.
    scan_end = min(data.calendar.max(), end + pd.Timedelta(days=7))
    report["meta"]["price_scan_end"] = str(scan_end.date())

    print("[check] price jumps", flush=True)
    jump_summary, jump_events = price_jumps(data, universe, start, scan_end)
    report["price_jumps"] = jump_summary

    print("[check] factor consistency", flush=True)
    factor_summary, factor_events = factor_consistency(data, universe, start, scan_end)
    report["factor_consistency"] = factor_summary

    early_summary = None
    reconciliation_summary = None
    constraints_summary: dict[str, Any] = {}
    panel = pd.DataFrame()
    baostock_daily: dict[str, pd.DataFrame] = {}
    ak_daily: dict[str, pd.DataFrame] = {}
    baostock_status = None

    if not args.skip_external:
        early_codes = (
            all_instruments.loc[all_instruments["end"] < data.calendar.max(), "instrument"].astype(str).tolist()
        )
        stock_basic = None
        try:
            print(f"[external] Baostock stock_basic for {len(early_codes)} early-ending instruments", flush=True)
            stock_basic = fetch_baostock_stock_basic(early_codes, raw_dir, refresh=args.refresh)
        except Exception as exc:
            report["external"]["errors"].append(f"stock_basic: {type(exc).__name__}: {exc}")
        print("[check] early-ending coverage / delisting", flush=True)
        early_summary, _ = early_end_check(all_instruments, data.calendar.max(), stock_basic, data)
        report["early_end"] = early_summary

        try:
            print(f"[external] Baostock daily for {len(universe_codes)} instruments", flush=True)
            baostock_daily, baostock_status = fetch_baostock_daily(
                universe_codes, start, end, raw_dir, refresh=args.refresh, progress=progress("baostock")
            )
            report["external"]["baostock_daily"] = status_summary(baostock_status)
        except Exception as exc:
            report["external"]["errors"].append(f"baostock_daily: {type(exc).__name__}: {exc}")

        sample_codes = select_sample(universe, args.sample_size, args.seed)
        ak_daily = {}
        ak_qfq: dict[str, pd.DataFrame] = {}
        ak_raw_status = None

        missing_baostock = [code for code in universe_codes if code not in baostock_daily]
        if missing_baostock:
            print(
                f"[external] Baostock missing {len(missing_baostock)} instruments; "
                "using AkShare/SZSE fallback",
                flush=True,
            )
            try:
                ak_daily, ak_raw_status = fetch_akshare_daily(
                    universe_codes,
                    start,
                    end,
                    raw_dir,
                    refresh=args.refresh,
                    adjust="",
                    progress=progress("akshare-universe"),
                )
                report["external"]["akshare_universe"] = status_summary(ak_raw_status)
            except Exception as exc:
                report["external"]["errors"].append(f"akshare_universe: {type(exc).__name__}: {exc}")

            st_intervals: dict[str, pd.Series] = {}
            try:
                sz_names = fetch_szse_name_changes(raw_dir, refresh=args.refresh)
                st_intervals = build_st_intervals(sz_names, data.calendar)
                report["external"]["szse_name_changes"] = {
                    "n_rows": len(sz_names),
                    "n_instruments": int(sz_names["instrument"].nunique()) if not sz_names.empty else 0,
                    "n_with_intervals": len(st_intervals),
                }
            except Exception as exc:
                report["external"]["errors"].append(f"szse_name_changes: {type(exc).__name__}: {exc}")

            sh_codes = [code for code in missing_baostock if code.startswith("SH")]
            try:
                sh_names = fetch_sina_name_history(
                    sh_codes,
                    raw_dir,
                    refresh=args.refresh,
                    progress=progress("sina-names"),
                )
                ever_st = sorted(sh_names.loc[sh_names["ever_st_name"], "instrument"].tolist())
                report["external"]["sina_name_history"] = {
                    "n_requested": len(sh_codes),
                    "n_rows": len(sh_names),
                    "n_ever_st": len(ever_st),
                    "ever_st_instruments": ever_st,
                }
            except Exception as exc:
                report["external"]["errors"].append(f"sina_name_history: {type(exc).__name__}: {exc}")

            fallback = prepare_akshare_daily(data, universe, ak_daily, st_intervals, start, end)
            added = 0
            for code, frame in fallback.items():
                if code not in baostock_daily:
                    baostock_daily[code] = frame
                    added += 1
            report["external"]["fallback_instruments"] = added

        if not ak_daily:
            try:
                print(f"[external] AkShare Sina raw daily for {len(sample_codes)} sampled instruments", flush=True)
                ak_daily, ak_raw_status = fetch_akshare_daily(
                    sample_codes,
                    start,
                    end,
                    raw_dir,
                    refresh=args.refresh,
                    adjust="",
                    progress=progress("akshare-raw"),
                )
            except Exception as exc:
                report["external"]["errors"].append(f"akshare_raw: {type(exc).__name__}: {exc}")
        report["external"]["akshare_raw"] = status_summary(ak_raw_status)

        try:
            print(f"[external] AkShare Sina qfq daily for {len(sample_codes)} sampled instruments", flush=True)
            ak_qfq, ak_qfq_status = fetch_akshare_daily(
                sample_codes,
                start,
                end,
                raw_dir,
                refresh=args.refresh,
                adjust="qfq",
                progress=progress("akshare-qfq"),
            )
            report["external"]["akshare_qfq"] = status_summary(ak_qfq_status)
        except Exception as exc:
            report["external"]["errors"].append(f"akshare_qfq: {type(exc).__name__}: {exc}")

        raw_map: dict[str, pd.DataFrame] = {}
        qfq_map: dict[str, pd.DataFrame] = {}
        for code in sample_codes:
            frame = ak_daily.get(code)
            if frame is None or frame.empty:
                frame = baostock_daily.get(code)
            if frame is not None and not frame.empty:
                raw_map[code] = frame
            adjusted = ak_qfq.get(code)
            if adjusted is not None and not adjusted.empty:
                qfq_map[code] = adjusted

        print("[check] reconciling sample against external data", flush=True)
        reconciliation_summary, _ = reconcile_all(
            data, sample_codes, raw_map, qfq_map, start, end
        )
        report["reconciliation"] = reconciliation_summary

        print("[build] ST / limit / suspension panel", flush=True)
        panel, constraints_summary = build_constraints(data, universe, baostock_daily, start, end)
        report["constraints"] = constraints_summary
        if not panel.empty:
            panel_path = processed_dir / f"constraints_{universe_path.stem}.parquet"
            panel.to_parquet(panel_path, index=False)
            report.setdefault("artifacts", {})["constraints_panel"] = str(panel_path)

    limitations: list[str] = []
    missing_official = report["calendar"].get("missing_official_days") or 0
    if missing_official:
        years = report["calendar"].get("missing_by_year", {})
        detail = "，".join(f"{year}: {count}" for year, count in years.items())
        limitations.append(
            f"Qlib 交易日历缺少 {missing_official} 个官方交易日（{detail}）。这些日期的行情、因子和标签"
            "在 Qlib 中完全缺失；baseline 与因子评估会直接跳过它们。修复需要按正确日历重建 Qlib binary 数据。"
        )
    if (report["price_jumps"].get("n_factor_rebase_days") or 0) > 0:
        broken = report["price_jumps"].get("n_hard_jumps_on_rebase_dates") or 0
        limitations.append(
            "2020-09-28 是复权因子重置：多数标的用 $close/$factor 还原的原始价连续，"
            f"但有 {broken} 只标的即使还原后仍大幅跳变（如 SZ002153 实际约 +3%，Qlib 还原约 −47%），"
            "说明供应商的因子重置口径不一致。当前 baseline 仍截止 2020-09-23；"
            "修复需要用外部原始价重建因子，而不是简单缩放。"
        )
    if early_summary and early_summary.get("n_qlib_truncated"):
        limitations.append(
            f"{early_summary['n_qlib_truncated']} 只标的的 Qlib 覆盖在 2020-09-25 提前结束。"
            "`all.txt` 的 end 是数据覆盖截止日，不是真实退市日；frozen 池内的 SH600068 属于此类。"
        )
    fallback_n = report["external"].get("fallback_instruments") or 0
    st_unknown = constraints_summary.get("n_st_unknown_days") or 0
    if fallback_n:
        limitations.append(
            f"Baostock 在本次运行不可达；{fallback_n} 只标的的 OHLCV/停牌/前收盘使用 AkShare 兜底。"
            f"深交所 ST 由 SZSE 更名表恢复；上交所 ST 无法恢复（{st_unknown} 个标的-日未知），"
            "这些日期按非 ST 规则推导涨跌停，可能低估 5% 限价。兜底行的前收盘由上一交易日收盘推导，"
            "除权日可能与交易所口径不同；涨跌停价仅近似。Baostock 恢复后加 `--refresh` 可补齐。"
        )
    adjusted_below = (reconciliation_summary or {}).get("stocks_adjusted_below_99pct") or 0
    if adjusted_below:
        worst_adjusted = ((reconciliation_summary or {}).get("worst_adjusted") or [{}])[0]
        worst_rate = worst_adjusted.get("adjusted_match_rate")
        if worst_rate is not None:
            limitations.append(
                f"复权收益对账受外部前复权口径影响：{adjusted_below} 只标的匹配率低于 99%"
                f"（最差 {worst_adjusted.get('instrument', '—')} {worst_rate:.1%}）。"
                "原始收益与成交额匹配率更高；这不一定是 Qlib 错误。"
            )
        else:
            limitations.append(
                f"复权收益对账受外部前复权口径影响：{adjusted_below} 只标的匹配率低于 99%。"
            )
    limitations.extend(
        [
            (
                "ST / 停复牌优先使用 Baostock 的 isST / tradestatus；Baostock 不可达时按上文兜底。"
                "涨跌停价由前收盘和板块规则推导，不是交易所逐日公布的 stk_limit，规则存疑的标的-日已单独标记。"
            ),
            (
                "100 只对账优先使用 AkShare 的 Sina 接口；东方财富接口在本机不可达。退市标的 AkShare 可能失败，"
                "Baostock 可用或有缓存时回退（仅价格，无成交量）。"
            ),
            "Qlib cn_data_v3 源自 Yahoo，不是 PIT 基本面数据库；无 $vwap/$amount，也无分钟频与日内执行数据。",
            "csi300_frozen 使用 2020-09-25 的成分信息，不是 point-in-time 股票池；成分前视仍未修复。",
        ]
    )
    report["limitations"] = limitations
    report["artifacts"] = {
        **(report.get("artifacts") or {}),
        "json": str(reports_dir / "data_quality.json"),
        "markdown": str(reports_dir / "data_quality.md"),
        "flags_csv": str(reports_dir / "data_quality_flags.csv"),
    }

    json_path = reports_dir / "data_quality.json"
    json_path.write_text(
        json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    md_path = reports_dir / "data_quality.md"
    md_path.write_text(render_markdown(json_safe(report)), encoding="utf-8")

    flags = coverage_frame.copy()
    if jump_summary.get("per_stock"):
        flags = flags.merge(pd.DataFrame(jump_summary["per_stock"]), on="instrument", how="left")
    if factor_summary.get("per_stock"):
        flags = flags.merge(pd.DataFrame(factor_summary["per_stock"]), on="instrument", how="left")
    if not panel.empty:
        constraints_by_stock = (
            panel.groupby("instrument")
            .agg(
                st_days=("is_st", lambda values: int((values == 1).sum())),
                suspended_days=("suspended", lambda values: int((values == 1).sum())),
                limit_up_days=("at_limit_up", "sum"),
                limit_down_days=("at_limit_down", "sum"),
                limit_uncertain_days=("limit_rule_uncertain", "sum"),
                data_hole_days=("data_hole", "sum"),
            )
            .reset_index()
        )
        flags = flags.merge(constraints_by_stock, on="instrument", how="left")
    flags_path = reports_dir / "data_quality_flags.csv"
    flags.to_csv(flags_path, index=False, encoding="utf-8-sig")
    if not jump_events.empty:
        jump_events.to_csv(reports_dir / "data_quality_price_events.csv", index=False, encoding="utf-8-sig")
    if not factor_events.empty:
        factor_events.to_csv(reports_dir / "data_quality_factor_events.csv", index=False, encoding="utf-8-sig")

    print("\n== summary ==", flush=True)
    print(
        json.dumps(
            {
                "calendar_days": report["calendar"]["n_days"],
                "missing_official_days": report["calendar"].get("missing_official_days"),
                "universe_min_max": [
                    report["universe"]["frozen"].get("min_count"),
                    report["universe"]["frozen"].get("max_count"),
                ],
                "nan_ratio": report["bin_coverage"].get("overall_nan_ratio"),
                "hard_jumps": report["price_jumps"].get("n_hard_jumps"),
                "factor_jumps": report["factor_consistency"].get("n_factor_jumps"),
                "early_end": early_summary.get("n_early_end") if early_summary else None,
                "raw_return_match_rate": (reconciliation_summary or {}).get("raw_return_match_rate"),
                "adjusted_return_match_rate": (reconciliation_summary or {}).get("adjusted_return_match_rate"),
                "turnover_match_rate": (reconciliation_summary or {}).get("turnover_match_rate"),
                "constraint_rows": constraints_summary.get("n_rows"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"report -> {md_path}", flush=True)
    print(f"json   -> {json_path}", flush=True)
    print(f"flags  -> {flags_path}", flush=True)


if __name__ == "__main__":
    main()
