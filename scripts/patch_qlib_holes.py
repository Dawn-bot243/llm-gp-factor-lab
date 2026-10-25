#!/usr/bin/env python
"""Patch Qlib NaN holes with Tushare daily bars.

Default is a dry run.  Pass ``--apply`` to modify the provider; every modified
bin file is backed up first.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.qlib_io import QlibData, read_instruments
from src.data.qlib_patch import apply_patch_plan, build_patch_plan, write_patch_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", default="D:/qlib_data/cn_data_v3")
    parser.add_argument("--universe", default=str(PROJECT_ROOT / "data" / "universes" / "csi300_frozen.txt"))
    parser.add_argument("--raw-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2020-09-23")
    parser.add_argument("--max-hole-days", type=int, default=30)
    parser.add_argument("--factor-tolerance", type=float, default=0.001)
    parser.add_argument("--scale-tolerance", type=float, default=0.005)
    parser.add_argument("--backup-dir", default=None)
    parser.add_argument("--apply", action="store_true", help="actually modify Qlib bin files")
    parser.add_argument(
        "--processed-dir", default=str(PROJECT_ROOT / "data" / "processed")
    )
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "项目报告"))
    args = parser.parse_args()

    data = QlibData.load(args.provider_uri)
    universe = read_instruments(args.universe)
    plan, skipped = build_patch_plan(
        data,
        universe,
        args.raw_dir,
        args.start,
        args.end,
        max_hole_days=args.max_hole_days,
        factor_tolerance=args.factor_tolerance,
        scale_tolerance=args.scale_tolerance,
    )
    base_summary = {
        "dry_run": not args.apply,
        "n_planned": len(plan),
        "n_skipped_runs": len(skipped),
        "n_instruments": int(plan["instrument"].nunique()) if not plan.empty else 0,
        "n_dates": int(plan["date"].nunique()) if not plan.empty else 0,
        "n_stocks_with_skips": int(skipped["instrument"].nunique()) if not skipped.empty else 0,
        "skipped_reasons": skipped["reason"].value_counts().to_dict() if not skipped.empty else {},
        "skipped": skipped.to_dict("records") if not skipped.empty else [],
    }

    if args.apply and not plan.empty:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(args.backup_dir) if args.backup_dir else Path(f"D:/qlib_data/cn_data_v3_patch_backup_{stamp}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        summary, failures = apply_patch_plan(plan, data, backup_dir)
        base_summary.update(summary)
        if not failures.empty:
            failures.to_csv(
                Path(args.processed_dir) / "qlib_patch_failures.csv",
                index=False,
                encoding="utf-8-sig",
            )
        base_summary["n_failures"] = len(failures)
    write_patch_reports(plan, skipped, base_summary, args.processed_dir, args.reports_dir)
    print(json.dumps(base_summary, ensure_ascii=False, indent=2))
    print(f"summary -> {Path(args.reports_dir) / 'qlib_patch_summary.json'}")


if __name__ == "__main__":
    main()
