"""Market-data loading for Stage 9.

Factor panels stay on the daily PIT membership, but prices/amounts are loaded
for *all* instruments that ever appear in the membership.  This matters when a
holding leaves the index: the portfolio still needs a price to value or sell
it even though it is no longer a scoring candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.factors.evaluator import EvaluationConfig, _ensure_qlib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONSTRAINTS = PROJECT_ROOT / "data" / "processed" / "pit_csi300_constraints.parquet"
DEFAULT_SECTOR_MAP = PROJECT_ROOT / "data" / "processed" / "sector_map.parquet"


@dataclass
class MarketData:
    """Wide market panels aligned to the benchmark trading calendar."""

    dates: pd.DatetimeIndex
    instruments: pd.Index
    open: pd.DataFrame
    close: pd.DataFrame
    amount_cny: pd.DataFrame
    factor: pd.DataFrame
    benchmark_close: pd.Series
    suspended: pd.DataFrame
    is_st: pd.DataFrame
    at_limit_up: pd.DataFrame
    at_limit_down: pd.DataFrame
    membership: pd.DataFrame
    membership_wide: pd.DataFrame


def _wide(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    series = frame[column]
    if not isinstance(series.index, pd.MultiIndex):
        raise TypeError(f"{column} is not a panel")
    if list(series.index.names) == ["datetime", "instrument"]:
        series = series.reorder_levels(["instrument", "datetime"])
    wide = series.unstack("instrument").sort_index()
    return wide.sort_index(axis=1)


def _constraint_wide(
    constraints: pd.DataFrame,
    column: str,
    dates: pd.DatetimeIndex,
    instruments: pd.Index,
    *,
    default: bool | int,
) -> pd.DataFrame:
    if column not in constraints.columns:
        return pd.DataFrame(default, index=dates, columns=instruments)
    series = constraints.set_index(["instrument", "datetime"])[column]
    wide = series.unstack("instrument").reindex(index=dates, columns=instruments)
    return wide.fillna(default)


def load_market_data(
    config: EvaluationConfig,
    *,
    instruments: list[str] | None = None,
    constraints_path: str | Path | None = None,
) -> MarketData:
    """Load prices, amount, benchmark and PIT constraint panels from Qlib."""
    if constraints_path is None:
        constraints_path = config.constraints_path or DEFAULT_CONSTRAINTS
    membership = pd.read_parquet(config.membership_path)
    membership["instrument"] = membership["instrument"].astype(str).str.upper()
    membership["datetime"] = pd.to_datetime(membership["datetime"])
    membership = membership[["instrument", "datetime"]].drop_duplicates()
    membership = membership[
        (membership["datetime"] >= pd.Timestamp(config.start))
        & (membership["datetime"] <= pd.Timestamp(config.end))
    ]
    membership_instruments = pd.Index(sorted(membership["instrument"].unique()))
    if instruments is None:
        instruments = list(membership_instruments)
    instruments = sorted({str(item).upper() for item in instruments})
    if not instruments:
        raise ValueError("no instruments to load")

    _ensure_qlib(config.provider_uri, config.kernels)
    from qlib.data import D

    frame = D.features(
        instruments,
        ["$open", "$close", "$amount", "$factor"],
        start_time=config.start,
        end_time=config.end,
        freq=config.freq,
    )
    if frame is None or frame.empty:
        raise RuntimeError("Qlib returned no market data")
    frame = frame.copy()
    frame.columns = ["open", "close", "amount", "factor"]
    if list(frame.index.names) == ["datetime", "instrument"]:
        frame = frame.reorder_levels(["instrument", "datetime"])
    frame = frame.sort_index()

    benchmark_frame = D.features(
        ["SH000300"],
        ["$close"],
        start_time=config.start,
        end_time=config.end,
        freq=config.freq,
    )
    if benchmark_frame is None or benchmark_frame.empty:
        raise RuntimeError("benchmark SH000300 is missing from the Qlib snapshot")
    benchmark = benchmark_frame["$close"]
    if isinstance(benchmark.index, pd.MultiIndex):
        benchmark = benchmark.droplevel("instrument")
    benchmark = benchmark.sort_index().astype(float)
    benchmark.index = pd.to_datetime(benchmark.index)
    dates = pd.DatetimeIndex(benchmark.index)

    open_wide = _wide(frame, "open").reindex(index=dates, columns=pd.Index(instruments))
    close_wide = _wide(frame, "close").reindex(index=dates, columns=pd.Index(instruments))
    amount_wide = _wide(frame, "amount").reindex(index=dates, columns=pd.Index(instruments))
    amount_cny = amount_wide * 1000.0
    factor_wide = _wide(frame, "factor").reindex(index=dates, columns=pd.Index(instruments)).fillna(1.0)
    factor_wide = factor_wide.where(factor_wide > 0, 1.0)

    constraints = pd.DataFrame()
    if constraints_path is not None and Path(constraints_path).exists():
        constraints = pd.read_parquet(constraints_path)
        constraints = constraints.rename(columns={"date": "datetime"})
        constraints["instrument"] = constraints["instrument"].astype(str).str.upper()
        constraints["datetime"] = pd.to_datetime(constraints["datetime"])
        constraints = constraints.drop_duplicates(["instrument", "datetime"])
    suspended = _constraint_wide(constraints, "suspended", dates, pd.Index(instruments), default=0).astype(int)
    is_st = _constraint_wide(constraints, "is_st", dates, pd.Index(instruments), default=0).eq(1)
    at_limit_up = _constraint_wide(
        constraints, "at_limit_up", dates, pd.Index(instruments), default=False
    ).astype(bool)
    at_limit_down = _constraint_wide(
        constraints, "at_limit_down", dates, pd.Index(instruments), default=False
    ).astype(bool)

    member_series = membership.set_index(["instrument", "datetime"]).assign(value=1)["value"]
    member_wide = member_series.unstack("instrument").reindex(index=dates, columns=pd.Index(instruments))
    membership_wide = member_wide.fillna(0).astype(int)

    return MarketData(
        dates=dates,
        instruments=pd.Index(instruments),
        open=open_wide,
        close=close_wide,
        amount_cny=amount_cny,
        factor=factor_wide,
        benchmark_close=benchmark,
        suspended=suspended,
        is_st=is_st,
        at_limit_up=at_limit_up,
        at_limit_down=at_limit_down,
        membership=membership,
        membership_wide=membership_wide,
    )


def load_sector_map(path: str | Path | None = DEFAULT_SECTOR_MAP) -> pd.Series:
    """Load the static sector map used for descriptive Brinson attribution."""
    if path is None:
        return pd.Series(dtype=object)
    file_path = Path(path)
    if not file_path.exists():
        return pd.Series(dtype=object)
    frame = pd.read_parquet(file_path)
    if "instrument" not in frame.columns or "sector" not in frame.columns:
        return pd.Series(dtype=object)
    frame["instrument"] = frame["instrument"].astype(str).str.upper()
    frame["sector"] = frame["sector"].fillna("UNKNOWN").astype(str)
    return frame.drop_duplicates("instrument").set_index("instrument")["sector"].sort_index()
