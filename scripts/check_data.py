"""Print and save a basic data-coverage report for the Qlib dataset."""
from __future__ import annotations

import json
from pathlib import Path

import qlib
from qlib.data import D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVIDER_URI = "D:/qlib_data/cn_data_v3"


def main() -> None:
    qlib.init(provider_uri=PROVIDER_URI, region="cn")
    calendar = D.calendar(start_time="2010-01-01", end_time="2026-12-31")
    all_instruments = D.list_instruments(D.instruments("all"), as_list=True)
    csi300 = D.list_instruments(D.instruments("csi300"), as_list=True)
    sample = D.features(
        ["SH600000"],
        ["$open", "$high", "$low", "$close", "$volume", "$factor"],
        start_time=calendar[0],
        end_time=calendar[-1],
    )
    report = {
        "provider_uri": PROVIDER_URI,
        "calendar_start": str(calendar[0].date()),
        "calendar_end": str(calendar[-1].date()),
        "calendar_days": len(calendar),
        "all_instruments": len(all_instruments),
        "csi300_instruments": len(csi300),
        "field_count": int(sample.shape[1]),
        "fields": [str(c) for c in sample.columns],
        "sample_rows": len(sample),
        "sample_nan_ratio": float(sample.isna().mean().mean()),
    }
    out = PROJECT_ROOT / "项目报告" / "data_coverage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
