#!/usr/bin/env python
"""Build the PIT CSI300 ST/limit/suspension panel from Dolt + Qlib."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.pit_constraints import build_pit_constraints
from src.data.qlib_io import QlibData


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider-uri", default="D:/qlib_data/cn_data_investment_data/extracted/qlib_bin"
    )
    parser.add_argument(
        "--membership", default=str(PROJECT_ROOT / "data" / "processed" / "pit_csi300_membership.parquet")
    )
    parser.add_argument(
        "--limits", default=str(PROJECT_ROOT / "data" / "raw" / "dolt" / "final_a_stock_limit_csi300.parquet")
    )
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "data" / "processed" / "pit_csi300_constraints.parquet")
    )
    args = parser.parse_args()
    import pandas as pd

    membership = pd.read_parquet(args.membership)
    limits = pd.read_parquet(args.limits)
    data = QlibData.load(args.provider_uri)
    panel, summary = build_pit_constraints(data, membership, limits)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    summary["output"] = str(output)
    reports = PROJECT_ROOT / "项目报告" / "pit_constraints_summary.json"
    reports.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
