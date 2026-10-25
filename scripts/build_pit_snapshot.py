#!/usr/bin/env python
"""Freeze a snapshot for the investment_data + PIT CSI300 setup."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider-uri", default="D:/qlib_data/cn_data_investment_data/extracted/qlib_bin"
    )
    parser.add_argument(
        "--release-manifest", default="D:/qlib_data/cn_data_investment_data/qlib_bin.manifest.json"
    )
    parser.add_argument(
        "--membership", default=str(PROJECT_ROOT / "data" / "processed" / "pit_csi300_membership.parquet")
    )
    parser.add_argument(
        "--quality", default=str(PROJECT_ROOT / "项目报告" / "investment_data_check" / "data_quality.json")
    )
    parser.add_argument(
        "--baseline", default=str(PROJECT_ROOT / "artifacts" / "baseline_pit" / "metrics.json")
    )
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "data" / "manifests" / "qlib_snapshot.json")
    )
    parser.add_argument(
        "--public-output", default=str(PROJECT_ROOT / "项目报告" / "qlib_snapshot.json")
    )
    args = parser.parse_args()

    provider = Path(args.provider_uri)
    release = _read_json(Path(args.release_manifest))
    quality = _read_json(Path(args.quality))
    baseline = _read_json(Path(args.baseline))
    membership = Path(args.membership)
    components = {
        "calendar_sha256": _sha256(provider / "calendars" / "day.txt"),
        "all_instruments_sha256": _sha256(provider / "instruments" / "all.txt"),
        "csi300_sha256": _sha256(provider / "instruments" / "csi300.txt"),
        "pit_membership_sha256": _sha256(membership),
        "release_archive_sha256": release.get("archive_sha256"),
    }
    digest = hashlib.sha256(
        "|".join(str(value) for value in components.values()).encode("utf-8")
    ).hexdigest()
    import pandas as pd

    membership_frame = pd.read_parquet(membership)
    membership_counts = membership_frame.groupby("datetime").size()
    snapshot = {
        "snapshot_id": f"qlib-pit-{digest[:12]}",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "provider_uri": str(provider),
        "source": {
            "name": "chenditc/investment_data",
            "release_tag": release.get("release_tag"),
            "target_trade_date": release.get("target_trade_date"),
            "archive_sha256": release.get("archive_sha256"),
            "license": "Apache-2.0 (repository); upstream data sources keep their own terms",
        },
        "calendar": {
            "days": len((provider / "calendars" / "day.txt").read_text(encoding="utf-8").split()),
            **components,
        },
        "universe": {
            "mode": "point_in_time_csi300",
            "membership_path": str(membership),
            "n_rows": len(membership_frame),
            "n_dates": int(membership_frame["datetime"].nunique()),
            "n_instruments": int(membership_frame["instrument"].nunique()),
            "min_members": int(membership_counts.min()),
            "max_members": int(membership_counts.max()),
            "days_not_300": int((membership_counts != 300).sum()),
        },
        "fields": [
            "adjclose",
            "amount",
            "change",
            "close",
            "factor",
            "high",
            "low",
            "open",
            "volume",
            "vwap",
        ],
        "data_quality": {
            "calendar_days_in_range": quality.get("calendar", {}).get("n_days"),
            "missing_official_trading_days": quality.get("calendar", {}).get("missing_official_days"),
            "close_nan_ratio": quality.get("bin_coverage", {}).get("overall_nan_ratio"),
            "data_holes": quality.get("constraints", {}).get("n_data_holes"),
            "hard_jumps": quality.get("price_jumps", {}).get("n_hard_jumps"),
            "split_days": quality.get("price_jumps", {}).get("n_split_days"),
            "factor_rebase_days": quality.get("price_jumps", {}).get("n_factor_rebase_days"),
            "raw_return_match_rate": quality.get("reconciliation", {}).get("raw_return_match_rate"),
            "adjusted_return_match_rate": quality.get("reconciliation", {}).get("adjusted_return_match_rate"),
        },
        "baseline": {
            "test_dates": baseline.get("test_dates"),
            "test_rows": baseline.get("test_rows"),
            "rank_ic_mean": baseline.get("rank_ic", {}).get("mean"),
            "icir": baseline.get("rank_ic", {}).get("icir"),
            "newey_west_t": baseline.get("rank_ic", {}).get("newey_west_t"),
            "excess_with_cost_annualized": baseline.get("backtest", {})
            .get("excess_with_cost", {})
            .get("risk", {})
            .get("annualized_return"),
            "excess_with_cost_ir": baseline.get("backtest", {})
            .get("excess_with_cost", {})
            .get("risk", {})
            .get("information_ratio"),
        },
        "accepted_limitations": [
            "ST / limit / suspension are not in the Qlib archive; limit export from Dolt is pending.",
            "Data is multi-source, not exchange-official; upstream source terms apply.",
            "The research range is still 2010-01-01 to 2020-09-23.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    public = Path(args.public_output)
    public.parent.mkdir(parents=True, exist_ok=True)
    public.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"snapshot_id": snapshot["snapshot_id"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
