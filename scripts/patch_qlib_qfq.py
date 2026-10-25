#!/usr/bin/env python
"""Second-stage patch for long/factor-change holes using AkShare qfq prices."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.qlib_io import QlibData
from src.data.qlib_patch import apply_patch_plan, build_qfq_patch_plan, write_patch_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", default="D:/qlib_data/cn_data_v3")
    parser.add_argument("--codes", default="SZ002493,SH603501,SZ000568")
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--qfq-dir", default=str(PROJECT_ROOT / "data" / "raw" / "akshare" / "sina_daily_qfq"))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2020-09-23")
    parser.add_argument("--scale-p95-tolerance", type=float, default=0.01)
    parser.add_argument("--include-placeholders", action="store_true")
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--processed-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "项目报告"))
    args = parser.parse_args()

    data = QlibData.load(args.provider_uri)
    codes = [code.strip().upper() for code in args.codes.split(",") if code.strip()]
    plan, skipped = build_qfq_patch_plan(
        data,
        codes,
        args.raw_dir,
        args.qfq_dir,
        args.start,
        args.end,
        scale_p95_tolerance=args.scale_p95_tolerance,
        include_placeholders=args.include_placeholders,
    )
    summary = {
        "dry_run": not args.apply,
        "n_planned": len(plan),
        "n_instruments": int(plan["instrument"].nunique()) if not plan.empty else 0,
        "n_dates": int(plan["date"].nunique()) if not plan.empty else 0,
        "skipped": skipped.to_dict("records") if not skipped.empty else [],
        "skipped_reasons": skipped["reason"].value_counts().to_dict() if not skipped.empty else {},
    }
    if args.apply and not plan.empty:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(args.backup_dir) if args.backup_dir else Path(f"D:/qlib_data/cn_data_v3_patch_backup_{stamp}_qfq")
        backup_dir.mkdir(parents=True, exist_ok=True)
        applied, failures = apply_patch_plan(
            plan, data, backup_dir, allow_overwrite=args.include_placeholders
        )
        summary.update(applied)
        summary["n_failures"] = len(failures)
        if not failures.empty:
            failures.to_csv(
                Path(args.processed_dir) / "qlib_qfq_patch_failures.csv",
                index=False,
                encoding="utf-8-sig",
            )
    write_patch_reports(
        plan,
        skipped,
        summary,
        args.processed_dir,
        args.reports_dir,
        prefix="qlib_qfq_patch",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary -> {Path(args.reports_dir) / 'qlib_qfq_patch_summary.json'}")


if __name__ == "__main__":
    main()
