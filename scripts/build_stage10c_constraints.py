#!/usr/bin/env python
"""Build the Stage 10C PIT constraints for the 2023-2026 OOS.

Inputs are the cached zzshare daily responses written by
``scripts/fetch_zzshare_daily.py``.  zzshare publishes exchange limit prices
(``high_limit`` / ``low_limit``) together with ``is_st`` and ``is_paused``, so
this panel does not need to derive limits from board rules.

The resulting panel uses the same schema and at-limit tolerance (0.011 CNY) as
the Dolt-based Stage 10 panel, which keeps the portfolio simulator semantics
unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AT_LIMIT_TOLERANCE = 0.011


def _instrument_from_ts_code(ts_code: str) -> str:
    code = str(ts_code).upper()
    if "." in code:
        symbol, exchange = code.split(".", 1)
        return f"{exchange}{symbol}"
    return code


def _load_rows(cache_dir: Path, instrument: str) -> list[dict]:
    code = instrument.upper()
    ts_code = f"{code[2:]}.{code[:2]}" if code[:2] in {"SH", "SZ", "BJ"} else code
    path = cache_dir / f"{ts_code}.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("rows") or []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--membership",
        default=str(PROJECT_ROOT / "data" / "processed" / "stage10_pit_csi300_membership.parquet"),
    )
    parser.add_argument("--start", default="2023-06-13")
    parser.add_argument("--end", default="2026-10-23")
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "data" / "raw" / "zzshare" / "daily"))
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "processed" / "stage10c_pit_csi300_constraints.parquet"),
    )
    parser.add_argument(
        "--summary",
        default=str(PROJECT_ROOT / "artifacts" / "stage10c" / "constraints_summary.json"),
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="build even if some membership instruments have no cached response",
    )
    args = parser.parse_args()

    membership = pd.read_parquet(args.membership)
    membership["datetime"] = pd.to_datetime(membership["datetime"])
    membership["instrument"] = membership["instrument"].astype(str).str.upper()
    membership = membership[
        (membership["datetime"] >= pd.Timestamp(args.start))
        & (membership["datetime"] <= pd.Timestamp(args.end))
    ][["instrument", "datetime"]].drop_duplicates()

    cache_dir = Path(args.cache_dir)
    pieces: list[pd.DataFrame] = []
    missing_instruments: list[str] = []
    for instrument, group in membership.groupby("instrument", sort=True):
        rows = _load_rows(cache_dir, instrument)
        if not rows:
            missing_instruments.append(instrument)
            continue
        frame = pd.DataFrame(rows)
        if frame.empty:
            missing_instruments.append(instrument)
            continue
        frame["date"] = pd.to_datetime(frame["trade_date"], format="%Y%m%d")
        frame = frame.drop_duplicates("date").set_index("date")
        dates = pd.DatetimeIndex(group["datetime"].sort_values())
        frame = frame.reindex(dates)

        pre_close = pd.to_numeric(frame.get("prev_close"), errors="coerce")
        up_limit = pd.to_numeric(frame.get("high_limit"), errors="coerce")
        down_limit = pd.to_numeric(frame.get("low_limit"), errors="coerce")
        high = pd.to_numeric(frame.get("high"), errors="coerce")
        low = pd.to_numeric(frame.get("low"), errors="coerce")
        # Dates absent from the cached response (suspension, delisting or a
        # data gap) are treated as suspended: the portfolio must not trade
        # without an explicit limit/ST observation.
        suspended = pd.to_numeric(frame.get("is_paused"), errors="coerce").fillna(1).astype("int8")
        is_st = pd.to_numeric(frame.get("is_st"), errors="coerce")
        is_st_known = is_st.notna()
        is_st = is_st.fillna(-1).astype("int8")

        with np.errstate(divide="ignore", invalid="ignore"):
            limit_ratio = up_limit / pre_close - 1.0
        limit_ratio = limit_ratio.where(pre_close > 0)

        at_limit_up = (
            (high >= (up_limit - AT_LIMIT_TOLERANCE))
            & up_limit.notna()
            & suspended.eq(0)
        ).fillna(False).astype(bool)
        at_limit_down = (
            (low <= (down_limit + AT_LIMIT_TOLERANCE))
            & down_limit.notna()
            & suspended.eq(0)
        ).fillna(False).astype(bool)

        pieces.append(
            pd.DataFrame(
                {
                    "instrument": instrument,
                    "date": dates,
                    "suspended": suspended.to_numpy(),
                    "is_st": is_st.to_numpy(),
                    "is_st_known": is_st_known.to_numpy(),
                    "pre_close": pre_close.to_numpy(),
                    "up_limit": up_limit.to_numpy(),
                    "down_limit": down_limit.to_numpy(),
                    "limit_ratio": limit_ratio.to_numpy(),
                    "at_limit_up": at_limit_up.to_numpy(),
                    "at_limit_down": at_limit_down.to_numpy(),
                    "source": "zzshare.v3.market.kline.day",
                }
            )
        )

    if not pieces:
        raise RuntimeError("no zzshare rows were found; run scripts/fetch_zzshare_daily.py first")
    if missing_instruments and not args.allow_missing:
        raise RuntimeError(
            f"{len(missing_instruments)} membership instruments have no cached zzshare "
            f"response (sample: {missing_instruments[:10]}). Re-run the fetcher or pass "
            "--allow-missing explicitly."
        )
    panel = pd.concat(pieces, ignore_index=True)
    panel = panel.sort_values(["date", "instrument"]).reset_index(drop=True)

    expected = len(membership)
    missing_rows = int(panel["up_limit"].isna().sum())
    covered_instruments = int(panel["instrument"].nunique())
    summary = {
        "output": str(args.output),
        "source": "zzshare.v3.market.kline.day",
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "n_rows": len(panel),
        "n_instruments": covered_instruments,
        "expected_membership_rows": expected,
        "missing_instruments": len(missing_instruments),
        "missing_instrument_sample": missing_instruments[:20],
        "n_rows_with_limits": int(panel["up_limit"].notna().sum()),
        "n_rows_without_limits": missing_rows,
        "n_suspended_days": int(panel["suspended"].sum()),
        "n_st_days": int((panel["is_st"] == 1).sum()),
        "n_st_unknown_days": int((~panel["is_st_known"]).sum()),
        "n_limit_up_days": int(panel["at_limit_up"].sum()),
        "n_limit_down_days": int(panel["at_limit_down"].sum()),
        "limit_ratio_counts": {
            str(round(float(key), 4)): int(value)
            for key, value in panel["limit_ratio"].dropna().round(4).value_counts().head(12).items()
        },
        "at_limit_tolerance": AT_LIMIT_TOLERANCE,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
