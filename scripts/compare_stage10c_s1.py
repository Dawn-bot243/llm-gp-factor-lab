#!/usr/bin/env python
"""Compare Stage 10B and Stage 10C S1 scorecards.

The candidate pool and the 2010-2016 calibration window are identical, so S1
status should be stable.  Differences can only come from full-panel coverage
(the Stage 10C evaluation extends to 2026) or worker timeouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["random", "gp_only", "llm_only", "llm_gp"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage10b", default=str(PROJECT_ROOT / "artifacts" / "stage10b"))
    parser.add_argument("--stage10c", default=str(PROJECT_ROOT / "artifacts" / "stage10c"))
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "artifacts" / "stage10c" / "s1_reproduction.json"),
    )
    args = parser.parse_args()

    b_dir, c_dir = Path(args.stage10b), Path(args.stage10c)
    summary: dict[str, Any] = {"groups": {}}
    for group in GROUPS:
        b_path, c_path = b_dir / f"s1_{group}.csv", c_dir / f"s1_{group}.csv"
        if not b_path.exists() or not c_path.exists():
            summary["groups"][group] = {"status": "missing"}
            continue
        before = pd.read_csv(b_path)
        after = pd.read_csv(c_path)
        key = "canonical" if "canonical" in before and "canonical" in after else "expression_hash"
        before = before.drop_duplicates(key).set_index(key)
        after = after.drop_duplicates(key).set_index(key)
        common = before.index.intersection(after.index)
        b = before.loc[common]
        c = after.loc[common]
        b_status = b["status"].astype(str)
        c_status = c["status"].astype(str)
        changes = b_status != c_status
        diff_ic = pd.to_numeric(b["rank_ic_mean"], errors="coerce") - pd.to_numeric(
            c["rank_ic_mean"], errors="coerce"
        )
        diff_cov = pd.to_numeric(b["coverage"], errors="coerce") - pd.to_numeric(
            c["coverage"], errors="coerce"
        )
        changed = [
            {
                "key": str(index),
                "before": str(b_status.loc[index]),
                "after": str(c_status.loc[index]),
                "before_coverage": float(b.loc[index, "coverage"]) if pd.notna(b.loc[index, "coverage"]) else None,
                "after_coverage": float(c.loc[index, "coverage"]) if pd.notna(c.loc[index, "coverage"]) else None,
                "after_reasons": str(c.loc[index, "reasons"]),
            }
            for index in common[changes]
        ]
        summary["groups"][group] = {
            "n_before": len(before),
            "n_after": len(after),
            "n_common": len(common),
            "n_status_changed": int(changes.sum()),
            "pass_before": int((b_status == "pass").sum()),
            "pass_after": int((c_status == "pass").sum()),
            "pass_lost": int(((b_status == "pass") & (c_status != "pass")).sum()),
            "pass_gained": int(((b_status != "pass") & (c_status == "pass")).sum()),
            "mean_abs_rank_ic_diff": float(np.nanmean(np.abs(diff_ic.to_numpy(dtype=float)))),
            "max_abs_rank_ic_diff": float(np.nanmax(np.abs(diff_ic.to_numpy(dtype=float)))),
            "mean_abs_coverage_diff": float(np.nanmean(np.abs(diff_cov.to_numpy(dtype=float)))),
            "max_abs_coverage_diff": float(np.nanmax(np.abs(diff_cov.to_numpy(dtype=float)))),
            "changes": changed,
        }
        print(json.dumps({group: {k: v for k, v in summary["groups"][group].items() if k != "changes"}}, ensure_ascii=False))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
