"""Cross-sectional factor metrics used by the Stage 3 evaluator.

All functions are pure pandas/numpy helpers so they can be tested without a
Qlib provider.  A panel is expected to have a MultiIndex named
``(instrument, datetime)`` and one column for the factor plus one column per
forward-return horizon.
"""
from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted panel with ``(instrument, datetime)`` index levels."""
    if frame.empty:
        return frame
    names = list(frame.index.names)
    if names == ["datetime", "instrument"]:
        frame = frame.reorder_levels(["instrument", "datetime"])
    elif names != ["instrument", "datetime"]:
        raise ValueError(f"unexpected panel index names: {names}")
    return frame.sort_index()


def clean_panel(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace infinities with NaN and drop rows missing any requested column."""
    if frame.empty:
        return frame
    out = frame[columns].replace([np.inf, -np.inf], np.nan).dropna()
    return out


def compute_ic(
    frame: pd.DataFrame,
    factor_col: str = "factor",
    label_col: str = "fwd_1",
    min_names: int = 20,
) -> tuple[pd.Series, pd.Series]:
    """Return daily Spearman (RankIC) and Pearson IC series."""
    clean = clean_panel(frame, [factor_col, label_col])
    rank_ics: dict[pd.Timestamp, float] = {}
    pearson_ics: dict[pd.Timestamp, float] = {}
    for dt, group in clean.groupby(level="datetime", sort=True):
        if len(group) < min_names:
            continue
        if group[factor_col].nunique() < 2 or group[label_col].nunique() < 2:
            continue
        value = group[factor_col].corr(group[label_col], method="spearman")
        if pd.notna(value):
            rank_ics[dt] = float(value)
        value = group[factor_col].corr(group[label_col], method="pearson")
        if pd.notna(value):
            pearson_ics[dt] = float(value)
    return (
        pd.Series(rank_ics, name="rank_ic", dtype=float).sort_index(),
        pd.Series(pearson_ics, name="ic", dtype=float).sort_index(),
    )


def ic_stats(ic: pd.Series, annualization: int = TRADING_DAYS) -> dict[str, Any]:
    """Summarize an IC series including a Newey-West t statistic."""
    ic = pd.to_numeric(ic, errors="coerce").dropna()
    n = len(ic)
    if n == 0:
        return {
            "n_dates": 0,
            "mean": None,
            "std": None,
            "icir": None,
            "icir_annualized": None,
            "t_stat": None,
            "newey_west_t": None,
            "positive_ratio": None,
        }
    mean = float(ic.mean())
    std = float(ic.std(ddof=1)) if n > 1 else float("nan")
    icir = mean / std if std and math.isfinite(std) and std > 1e-12 else float("nan")
    t_stat = mean / std * math.sqrt(n) if std and math.isfinite(std) and std > 1e-12 else float("nan")
    newey_west_t: float | None = None
    if n >= 3 and std > 1e-12:
        try:
            import statsmodels.api as sm

            x = np.ones((n, 1))
            fit = sm.OLS(ic.to_numpy(dtype=float), x).fit(
                cov_type="HAC", cov_kwds={"maxlags": min(10, n - 1)}
            )
            value = float(fit.tvalues[0])
            if math.isfinite(value):
                newey_west_t = value
        except Exception:
            newey_west_t = None
    return {
        "n_dates": n,
        "mean": mean,
        "std": std if math.isfinite(std) else None,
        "icir": icir if math.isfinite(icir) else None,
        "icir_annualized": float(icir * math.sqrt(annualization)) if math.isfinite(icir) else None,
        "t_stat": t_stat if math.isfinite(t_stat) else None,
        "newey_west_t": newey_west_t,
        "positive_ratio": float((ic > 0).mean()),
    }


def quantile_returns(
    frame: pd.DataFrame,
    factor_col: str = "factor",
    label_col: str = "fwd_1",
    q: int = 5,
    min_names: int | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Return mean return by quantile, daily long-short, and the daily table.

    Bucket 0 is the lowest factor value.  The long-short series is
    ``bucket q-1 minus bucket 0`` on each date.
    """
    if q < 2:
        raise ValueError("q must be >= 2")
    minimum = min_names if min_names is not None else max(20, q * 2)
    clean = clean_panel(frame, [factor_col, label_col])
    rows: list[pd.Series] = []
    for dt, group in clean.groupby(level="datetime", sort=True):
        if len(group) < minimum:
            continue
        ranks = group[factor_col].rank(method="first")
        buckets = pd.qcut(ranks, q, labels=False)
        row = group[label_col].groupby(buckets).mean()
        row.name = dt
        rows.append(row)
    if not rows:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.DataFrame()
    qdf = pd.DataFrame(rows).sort_index()
    qdf.index.name = "datetime"
    qdf.columns = [int(col) for col in qdf.columns]
    qmean = qdf.mean(axis=0, skipna=True)
    long_short = (qdf[q - 1] - qdf[0]).rename("long_short")
    return qmean, long_short, qdf


def quantile_stats(qmean: pd.Series, long_short: pd.Series) -> dict[str, Any]:
    """Summarize quantile averages and the daily long-short return."""
    out: dict[str, Any] = {"mean_returns": [float(v) if pd.notna(v) else None for v in qmean]}
    if len(qmean) >= 2 and qmean.notna().sum() >= 2:
        out["monotonicity"] = float(
            pd.Series(qmean.index, index=qmean.index).corr(qmean, method="spearman")
        )
    else:
        out["monotonicity"] = None
    ls = long_short.dropna()
    if ls.empty:
        out["long_short"] = {"n_dates": 0}
        return out
    mean = float(ls.mean())
    std = float(ls.std(ddof=1)) if len(ls) > 1 else float("nan")
    cumulative = float((1.0 + ls).prod() - 1.0)
    running_max = (1.0 + ls).cumprod().cummax()
    drawdown = (1.0 + ls).cumprod() / running_max - 1.0
    out["long_short"] = {
        "n_dates": len(ls),
        "mean": mean,
        "std": std if math.isfinite(std) else None,
        "t_stat": mean / std * math.sqrt(len(ls)) if std and math.isfinite(std) and std > 1e-12 else None,
        "annualized_return": mean * TRADING_DAYS,
        "cumulative_return": cumulative,
        "max_drawdown": float(drawdown.min()),
        "positive_ratio": float((ls > 0).mean()),
    }
    return out


def top_quantile_turnover(
    frame: pd.DataFrame,
    factor_col: str = "factor",
    q: int = 5,
    min_names: int | None = None,
) -> float | None:
    """Average one-way turnover of the top ``1/q`` names by factor."""
    if q < 2:
        raise ValueError("q must be >= 2")
    minimum = min_names if min_names is not None else max(20, q * 2)
    clean = clean_panel(frame, [factor_col])
    sets: list[set[str]] = []
    for _, group in clean.groupby(level="datetime", sort=True):
        if len(group) < minimum:
            continue
        k = max(1, len(group) // q)
        top = group.nlargest(k, factor_col).index.get_level_values("instrument")
        sets.append(set(top))
    values = []
    for previous, current in itertools.pairwise(sets):
        if previous:
            values.append(1.0 - len(previous & current) / len(previous))
    return float(np.mean(values)) if values else None


def rank_autocorrelation(
    frame: pd.DataFrame,
    factor_col: str = "factor",
    min_names: int = 20,
) -> float | None:
    """Average cross-sectional rank correlation between consecutive dates."""
    clean = clean_panel(frame, [factor_col])
    by_date = {
        dt: group[factor_col].droplevel("datetime").sort_index()
        for dt, group in clean.groupby(level="datetime", sort=True)
        if len(group) >= min_names and group[factor_col].nunique() >= 2
    }
    values: list[float] = []
    for previous, current in itertools.pairwise(by_date.values()):
        common = previous.index.intersection(current.index)
        if len(common) < min_names:
            continue
        value = previous.loc[common].corr(current.loc[common], method="spearman")
        if pd.notna(value):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def distribution_stats(series: pd.Series) -> dict[str, Any]:
    """Return robust distribution statistics for a factor panel."""
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(values)
    if n == 0:
        return {"count": 0}
    quantiles = values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": n,
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if n > 1 else None,
        "min": float(values.min()),
        "max": float(values.max()),
        "skew": float(values.skew()) if n > 2 else None,
        "kurt": float(values.kurt()) if n > 3 else None,
        "zero_ratio": float((values == 0).mean()),
        "percentiles": {f"p{int(q * 100):02d}": float(v) for q, v in quantiles.items()},
    }


def coverage_stats(series: pd.Series, total_rows: int | None = None) -> dict[str, Any]:
    """Count finite, NaN and infinite factor values."""
    numeric = pd.to_numeric(series, errors="coerce")
    total = int(total_rows if total_rows is not None else len(numeric))
    finite = int(np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    inf = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    nan = total - finite - inf
    return {
        "total_rows": total,
        "finite_rows": finite,
        "nan_rows": nan,
        "inf_rows": inf,
        "coverage": finite / total if total else None,
        "unique_values": int(numeric.replace([np.inf, -np.inf], np.nan).nunique()),
    }
