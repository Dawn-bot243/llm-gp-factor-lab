#!/usr/bin/env python
"""Freeze a reproducible identifier for the current Qlib data state."""
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
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-uri", default="D:/qlib_data/cn_data_v3")
    parser.add_argument(
        "--universe", default=str(PROJECT_ROOT / "data" / "universes" / "csi300_frozen.txt")
    )
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "项目报告"))
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "data" / "manifests" / "qlib_snapshot.json")
    )
    args = parser.parse_args()

    provider = Path(args.provider_uri)
    reports_dir = Path(args.reports_dir)
    universe = Path(args.universe)
    calendar_path = provider / "calendars" / "day.txt"
    instruments_path = provider / "instruments" / "all.txt"
    patch_summary_path = reports_dir / "qlib_patch_combined_summary.json"
    quality_path = reports_dir / "data_quality.json"
    baseline_paths = {
        "patched": PROJECT_ROOT / "artifacts" / "baseline_patched" / "metrics.json",
        "patched_ex_sz002493": PROJECT_ROOT / "artifacts" / "baseline_patched_ex_sz002493" / "metrics.json",
        "pre_patch_masked": PROJECT_ROOT / "artifacts" / "baseline_masked" / "metrics.json",
        "original": PROJECT_ROOT / "artifacts" / "baseline" / "metrics.json",
    }
    baseline = {name: _read_json(path) for name, path in baseline_paths.items()}
    quality = _read_json(quality_path)
    patch_summary = _read_json(patch_summary_path)

    calendar_days = 0
    if calendar_path.exists():
        calendar_days = len(calendar_path.read_text(encoding="utf-8").split())

    components = {
        "calendar_sha256": _sha256(calendar_path),
        "instruments_sha256": _sha256(instruments_path),
        "universe_sha256": _sha256(universe),
        "patch_summary_sha256": _sha256(patch_summary_path),
    }
    digest = hashlib.sha256(
        "|".join(str(value) for value in components.values()).encode("utf-8")
    ).hexdigest()
    snapshot = {
        "snapshot_id": f"qlib-{digest[:12]}",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "provider_uri": str(provider),
        "calendar": {"days": calendar_days, **components},
        "universe": {
            "path": str(universe),
            "n_instruments": sum(
                1 for line in universe.read_text(encoding="utf-8").splitlines() if line.strip()
            )
            if universe.exists()
            else 0,
            "point_in_time": False,
            "accepted_limitation": (
                "csi300_frozen uses the 2020-09-25 CSI300 membership for the whole sample. "
                "This is a constituent look-ahead and is an accepted, disclosed limitation; "
                "do not describe the universe as point-in-time."
            ),
        },
        "patch": {
            "summary_path": str(patch_summary_path),
            "totals": patch_summary.get("totals"),
            "post_patch": patch_summary.get("post_patch"),
            "backup_dirs": [stage.get("backup_dir") for stage in patch_summary.get("stages", [])],
        },
        "data_quality": {
            "close_nan_ratio": quality.get("bin_coverage", {}).get("overall_nan_ratio"),
            "remaining_data_holes": quality.get("constraints", {}).get("n_data_holes"),
            "missing_official_trading_days": quality.get("calendar", {}).get("missing_official_days"),
        },
        "baseline": {
            name: {
                "test_dates": metrics.get("test_dates"),
                "rank_ic_mean": metrics.get("rank_ic", {}).get("mean"),
                "excess_with_cost_annualized": metrics.get("backtest", {})
                .get("excess_with_cost", {})
                .get("risk", {})
                .get("annualized_return"),
                "excess_with_cost_ir": metrics.get("backtest", {})
                .get("excess_with_cost", {})
                .get("risk", {})
                .get("information_ratio"),
            }
            for name, metrics in baseline.items()
        },
        "accepted_limitations": [
            "csi300_frozen is not point-in-time; constituent look-ahead is disclosed and accepted.",
            "47 official trading days are absent from the Qlib calendar; the sample is frozen as-is.",
            "7 stock-days remain unpatched; the Qlib close NaN ratio is reported above.",
            "ST/limit/suspension coverage is partial because Baostock was unreachable at finalization.",
            "Data is capped at 2020-09-23; post-break data is not used.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"snapshot_id": snapshot["snapshot_id"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
