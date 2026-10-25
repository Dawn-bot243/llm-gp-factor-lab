#!/usr/bin/env python
"""Generate the date/instrument quality mask used by the baseline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mask import build_mask_from_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        default=str(PROJECT_ROOT / "data" / "processed" / "constraints_csi300_frozen.parquet"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "data_quality_mask.json"),
    )
    parser.add_argument("--date-threshold", type=float, default=0.10)
    parser.add_argument("--instrument-threshold", type=float, default=0.10)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel)
    mask = build_mask_from_panel(
        panel,
        date_threshold=args.date_threshold,
        instrument_threshold=args.instrument_threshold,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(mask, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "excluded_dates": mask["excluded_dates"],
                "excluded_instruments": mask["excluded_instruments"],
                "date_ratios": mask["date_ratios"],
                "instrument_ratios": mask["instrument_ratios"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"saved -> {output}")


if __name__ == "__main__":
    main()
