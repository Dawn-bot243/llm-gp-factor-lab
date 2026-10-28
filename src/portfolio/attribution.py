"""Style exposures and Brinson-style attribution for Stage 9.

The data snapshot has no free-float market cap and no PIT industry history.
Stage 9 therefore uses:

* price/volume style proxies: size (log ADV), beta, momentum, volatility,
  illiquidity and short reversal;
* a static, current Sina sector map for descriptive Brinson attribution;
* equal-weight PIT membership as the benchmark portfolio.

The sector map is explicitly *not* PIT and is only used to describe the
portfolio, never to select factors or tune parameters.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _cross_zscore(frame: pd.DataFrame, *, min_names: int = 20) -> pd.DataFrame:
    """Cross-sectional z-score by date, with a minimum cross-section size."""
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=1)
    count = frame.notna().sum(axis=1)
    z = frame.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
    return z.where(count >= min_names, np.nan)


def style_exposures(
    close: pd.DataFrame,
    amount_cny: pd.DataFrame,
    benchmark_close: pd.Series,
    *,
    min_names: int = 20,
) -> pd.DataFrame:
    """Build daily cross-sectional style exposures (z-scored by date).

    All inputs are wide panels.  Exposures use only information available at
    the close of the same date.
    """
    close = close.sort_index()
    returns = close / close.shift(1) - 1.0
    benchmark = benchmark_close.reindex(close.index).astype(float)
    bench_ret = benchmark / benchmark.shift(1) - 1.0

    size = np.log(amount_cny.rolling(20, min_periods=10).mean().replace(0.0, np.nan))
    momentum = close / close.shift(20) - 1.0
    reversal = -(close / close.shift(5) - 1.0)
    volatility = returns.rolling(20, min_periods=10).std()
    illiquidity = (returns.abs() / amount_cny.replace(0.0, np.nan)).rolling(20, min_periods=10).mean()
    illiquidity = np.log1p(illiquidity.replace([np.inf, -np.inf], np.nan))

    bench_mean = bench_ret.rolling(60, min_periods=30).mean()
    stock_mean = returns.rolling(60, min_periods=30).mean()
    cov = (returns.mul(bench_ret, axis=0)).rolling(60, min_periods=30).mean() - stock_mean.mul(
        bench_mean, axis=0
    )
    bench_var = (bench_ret**2).rolling(60, min_periods=30).mean() - bench_mean**2
    beta = cov.div(bench_var.replace(0.0, np.nan), axis=0)

    raw = pd.concat(
        {
            "size": size,
            "beta60": beta,
            "momentum20": momentum,
            "reversal5": reversal,
            "vol20": volatility,
            "illiq20": illiquidity,
        },
        axis=1,
    )
    # concat with a dict creates a MultiIndex; unwrap it to date x (field, stock)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.set_names(["field", "instrument"])
    pieces = {}
    for field in ("size", "beta60", "momentum20", "reversal5", "vol20", "illiq20"):
        block = raw[field] if field in raw.columns.get_level_values(0) else raw.xs(field, axis=1, level=0)
        pieces[field] = _cross_zscore(block, min_names=min_names)
    # Return a long-ish frame with a MultiIndex column (field, instrument).
    result = pd.concat(pieces, axis=1).sort_index(axis=0)
    result.columns = result.columns.set_names(["field", "instrument"])
    return result


def portfolio_exposure(weights: pd.DataFrame, exposures: pd.DataFrame) -> pd.DataFrame:
    """Weighted-average exposure of the portfolio at each date."""
    if isinstance(exposures.columns, pd.MultiIndex):
        fields = list(dict.fromkeys(exposures.columns.get_level_values(0)))
        out = {}
        for field in fields:
            block = exposures[field].reindex(index=weights.index, columns=weights.columns)
            out[field] = (weights * block).sum(axis=1) / weights.sum(axis=1).replace(0.0, np.nan)
        return pd.DataFrame(out)
    return (weights * exposures.reindex_like(weights)).sum(axis=1) / weights.sum(axis=1).replace(0.0, np.nan)


def benchmark_exposure(exposures: pd.DataFrame, membership_wide: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight PIT membership exposure at each date."""
    if isinstance(exposures.columns, pd.MultiIndex):
        fields = list(dict.fromkeys(exposures.columns.get_level_values(0)))
        out = {}
        for field in fields:
            block = exposures[field].reindex(index=membership_wide.index, columns=membership_wide.columns)
            mask = membership_wide.eq(1) & block.notna()
            out[field] = block.where(mask).mean(axis=1)
        return pd.DataFrame(out)
    mask = membership_wide.eq(1) & exposures.notna()
    return exposures.where(mask).mean(axis=1).to_frame("exposure")


def exposure_summary(
    portfolio: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize average active style exposure."""
    common = portfolio.columns.intersection(benchmark.columns)
    rows = []
    for field in common:
        p = portfolio[field].dropna()
        b = benchmark[field].dropna()
        active = p.sub(b, axis=0).dropna()
        rows.append(
            {
                "field": field,
                "portfolio_mean": float(p.mean()) if len(p) else None,
                "benchmark_mean": float(b.mean()) if len(b) else None,
                "active_mean": float(active.mean()) if len(active) else None,
                "active_std": float(active.std(ddof=1)) if len(active) > 1 else None,
                "active_min": float(active.min()) if len(active) else None,
                "active_max": float(active.max()) if len(active) else None,
                "n_dates": len(active),
            }
        )
    return pd.DataFrame(rows)


def _group_labels(groups: pd.Series | pd.DataFrame, date: pd.Timestamp, instruments: pd.Index) -> pd.Series:
    if isinstance(groups, pd.DataFrame):
        if date not in groups.index:
            return pd.Series("UNKNOWN", index=instruments)
        labels = groups.loc[date].reindex(instruments)
    else:
        labels = groups.reindex(instruments)
    return labels.fillna("UNKNOWN").astype(str)


def _group_aggregate(
    weights: pd.Series,
    returns: pd.Series,
    labels: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    frame = pd.DataFrame({"w": weights, "r": returns, "g": labels}).dropna(subset=["g"])
    frame["w"] = frame["w"].fillna(0.0)
    frame["r"] = frame["r"].fillna(0.0)
    group_weight = frame.groupby("g", sort=True)["w"].sum()
    weighted_return = (frame["w"] * frame["r"]).groupby(frame["g"], sort=True).sum()
    group_return = weighted_return / group_weight.replace(0.0, np.nan)
    return group_weight, group_return.fillna(0.0)


def brinson_attribution(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    groups: pd.Series | pd.DataFrame,
    *,
    benchmark_weights: pd.DataFrame | None = None,
    weight_lag: int = 1,
    cash_label: str = "CASH",
) -> dict[str, Any]:
    """Brinson-Fachler attribution by an arbitrary grouping.

    ``weights`` are end-of-day portfolio weights.  With ``weight_lag=1`` they
    are shifted before multiplying today's stock returns, so the attribution
    uses beginning-of-day holdings.  The benchmark defaults to an equal-weight
    portfolio over stocks with a valid return.
    """
    weights = weights.sort_index().fillna(0.0)
    returns = returns.sort_index()
    if weight_lag:
        weights = weights.shift(weight_lag)
    if benchmark_weights is None:
        valid = returns.notna()
        benchmark_weights = valid.div(valid.sum(axis=1).replace(0, np.nan), axis=0)
    else:
        benchmark_weights = benchmark_weights.sort_index().reindex_like(returns).fillna(0.0)
        if weight_lag:
            benchmark_weights = benchmark_weights.shift(weight_lag)

    dates = weights.index.intersection(returns.index)
    daily_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for date in dates:
        w_p = weights.loc[date].reindex(returns.columns).fillna(0.0)
        if w_p.sum() <= 0:
            continue
        r = returns.loc[date].reindex(returns.columns)
        w_b = benchmark_weights.loc[date].reindex(returns.columns).fillna(0.0)
        w_b = w_b.where(r.notna(), 0.0)
        if w_b.sum() <= 0:
            continue
        w_b = w_b / w_b.sum()
        labels = _group_labels(groups, date, returns.columns)
        cash = max(0.0, 1.0 - float(w_p.sum()))
        if cash > 0:
            labels = pd.concat([labels, pd.Series({cash_label: cash_label})])
            w_p = pd.concat([w_p, pd.Series({cash_label: cash})])
            w_b = pd.concat([w_b, pd.Series({cash_label: 0.0})])
            r = pd.concat([r, pd.Series({cash_label: 0.0})])
        p_weight, p_return = _group_aggregate(w_p, r, labels)
        b_weight, b_return = _group_aggregate(w_b, r, labels)
        r_p = float((w_p.reindex(r.index) * r).sum())
        r_b = float((w_b.reindex(r.index) * r).sum())
        all_groups = sorted(set(p_weight.index) | set(b_weight.index))
        allocation = 0.0
        selection = 0.0
        interaction = 0.0
        for group in all_groups:
            wp = float(p_weight.get(group, 0.0))
            wb = float(b_weight.get(group, 0.0))
            rp = float(p_return.get(group, 0.0))
            rb = float(b_return.get(group, 0.0))
            alloc = (wp - wb) * (rb - r_b)
            select = wb * (rp - rb)
            interact = (wp - wb) * (rp - rb)
            allocation += alloc
            selection += select
            interaction += interact
            group_rows.append(
                {
                    "datetime": date,
                    "group": group,
                    "portfolio_weight": wp,
                    "benchmark_weight": wb,
                    "portfolio_return": rp,
                    "benchmark_return": rb,
                    "allocation": alloc,
                    "selection": select,
                    "interaction": interact,
                }
            )
        daily_rows.append(
            {
                "datetime": date,
                "portfolio_return": r_p,
                "benchmark_return": r_b,
                "active_return": r_p - r_b,
                "allocation": allocation,
                "selection": selection,
                "interaction": interaction,
                "cash_weight": cash,
                "reconciliation_error": allocation + selection + interaction - (r_p - r_b),
            }
        )

    if not daily_rows:
        return {"daily": pd.DataFrame(), "groups": pd.DataFrame(), "summary": {}, "by_group": pd.DataFrame()}
    daily = pd.DataFrame(daily_rows).set_index("datetime").sort_index()
    group_daily = pd.DataFrame(group_rows)
    annualization = 252
    summary = {
        "n_days": len(daily),
        "portfolio_mean_annual": float(daily["portfolio_return"].mean() * annualization),
        "benchmark_mean_annual": float(daily["benchmark_return"].mean() * annualization),
        "active_mean_annual": float(daily["active_return"].mean() * annualization),
        "allocation_mean_annual": float(daily["allocation"].mean() * annualization),
        "selection_mean_annual": float(daily["selection"].mean() * annualization),
        "interaction_mean_annual": float(daily["interaction"].mean() * annualization),
        "active_cumulative": float(daily["active_return"].sum()),
        "allocation_cumulative": float(daily["allocation"].sum()),
        "selection_cumulative": float(daily["selection"].sum()),
        "interaction_cumulative": float(daily["interaction"].sum()),
        "cash_weight_mean": float(daily["cash_weight"].mean()),
        "max_reconciliation_error": float(daily["reconciliation_error"].abs().max()),
    }
    by_group = (
        group_daily.groupby("group", sort=True)
        .agg(
            portfolio_weight=("portfolio_weight", "mean"),
            benchmark_weight=("benchmark_weight", "mean"),
            portfolio_return=("portfolio_return", "mean"),
            benchmark_return=("benchmark_return", "mean"),
            allocation=("allocation", "sum"),
            selection=("selection", "sum"),
            interaction=("interaction", "sum"),
        )
        .reset_index()
        .sort_values("allocation", ascending=False)
    )
    return {
        "daily": daily,
        "groups": group_daily,
        "summary": summary,
        "by_group": by_group,
    }


def size_bucket_groups(
    size_exposure: pd.DataFrame,
    *,
    quantiles: int = 5,
) -> pd.DataFrame:
    """Daily size-quantile labels used as a style-sector Brinson grouping."""
    labels = pd.DataFrame(index=size_exposure.index, columns=size_exposure.columns, dtype=object)
    for date, row in size_exposure.iterrows():
        valid = row.dropna()
        if len(valid) < quantiles * 2:
            continue
        buckets = pd.qcut(valid.rank(method="first"), quantiles, labels=False)
        labels.loc[date, valid.index] = [f"SIZE_Q{int(value) + 1}" for value in buckets]
    return labels
