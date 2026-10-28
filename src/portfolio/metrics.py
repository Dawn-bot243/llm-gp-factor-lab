"""Portfolio performance metrics for Stage 9.

The input is the daily frame produced by :func:`src.portfolio.simulator.run_portfolio`.
Returns are close-to-close.  ``drop_days`` skips the first days because the
portfolio starts in cash and the first open-to-close day is not comparable with
a close-to-close benchmark.  The same convention is used for validation and
test, so parameter selection is not biased by the warm-up.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def drawdown_series(nav: pd.Series) -> pd.Series:
    """Drawdown from the running peak."""
    nav = nav.astype(float).dropna()
    if nav.empty:
        return pd.Series(dtype=float)
    return nav / nav.cummax() - 1.0


def performance_metrics(
    daily: pd.DataFrame,
    *,
    drop_days: int = 2,
    annualization: int = 252,
    risk_free: float = 0.0,
) -> dict[str, Any]:
    """Compute net and benchmark-relative metrics from a daily NAV frame."""
    if daily.empty:
        return {"n_days": 0}
    required = {"nav", "return", "benchmark_return", "turnover", "cost", "nav_prev"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily frame is missing columns: {sorted(missing)}")

    frame = daily.sort_index().copy()
    start = min(max(int(drop_days), 0), max(len(frame) - 1, 0))
    returns = pd.to_numeric(frame["return"].iloc[start:], errors="coerce").dropna()
    benchmark = pd.to_numeric(frame["benchmark_return"].iloc[start:], errors="coerce").reindex(
        returns.index
    )
    valid = returns.notna() & benchmark.notna()
    returns = returns[valid]
    benchmark = benchmark[valid]
    n = len(returns)
    if n == 0:
        return {"n_days": 0}

    rf_daily = float(risk_free) / annualization
    port_norm = (1.0 + returns).cumprod()
    bench_norm = (1.0 + benchmark).cumprod()
    total_return = float(port_norm.iloc[-1] - 1.0)
    benchmark_total = float(bench_norm.iloc[-1] - 1.0)
    annual_return = float(port_norm.iloc[-1] ** (annualization / n) - 1.0)
    benchmark_annual = float(bench_norm.iloc[-1] ** (annualization / n) - 1.0)

    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if n > 1 else float("nan")
    annual_vol = std * math.sqrt(annualization) if math.isfinite(std) else float("nan")
    sharpe = (mean - rf_daily) / std * math.sqrt(annualization) if std > 1e-12 else float("nan")

    downside = np.minimum(returns.to_numpy(dtype=float) - rf_daily, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside**2)) * math.sqrt(annualization))
    sortino = (mean - rf_daily) * annualization / downside_dev if downside_dev > 1e-12 else float("nan")

    drawdown = drawdown_series(port_norm)
    max_drawdown = float(drawdown.min()) if not drawdown.empty else float("nan")
    calmar = annual_return / abs(max_drawdown) if max_drawdown < -1e-12 else float("nan")

    excess = returns - benchmark
    excess_mean = float(excess.mean())
    excess_std = float(excess.std(ddof=1)) if n > 1 else float("nan")
    tracking_error = excess_std * math.sqrt(annualization) if math.isfinite(excess_std) else float("nan")
    information_ratio = (
        excess_mean / excess_std * math.sqrt(annualization) if excess_std > 1e-12 else float("nan")
    )
    benchmark_var = float(benchmark.var(ddof=1)) if n > 1 else float("nan")
    beta = (
        float(np.cov(returns, benchmark, ddof=1)[0, 1] / benchmark_var)
        if math.isfinite(benchmark_var) and benchmark_var > 1e-18
        else float("nan")
    )
    alpha_annual = (mean - beta * float(benchmark.mean())) * annualization if math.isfinite(beta) else float("nan")

    turnover = pd.to_numeric(frame["turnover"].iloc[start:], errors="coerce").reindex(returns.index)
    cost = pd.to_numeric(frame["cost"].iloc[start:], errors="coerce").reindex(returns.index)
    nav_prev = pd.to_numeric(frame["nav_prev"].iloc[start:], errors="coerce").reindex(returns.index)
    cost_ratio = cost / nav_prev.replace(0.0, np.nan)
    fill_rate = pd.to_numeric(frame.get("fill_rate", pd.Series(index=frame.index, dtype=float)).iloc[start:], errors="coerce").reindex(returns.index)
    adv_util = pd.to_numeric(
        frame.get("adv_utilization_max", pd.Series(index=frame.index, dtype=float)).iloc[start:],
        errors="coerce",
    ).reindex(returns.index)
    n_holdings = pd.to_numeric(
        frame.get("n_holdings", pd.Series(index=frame.index, dtype=float)).iloc[start:], errors="coerce"
    ).reindex(returns.index)
    cash_ratio = pd.to_numeric(
        frame.get("cash_ratio", pd.Series(index=frame.index, dtype=float)).iloc[start:], errors="coerce"
    ).reindex(returns.index)
    max_weight = pd.to_numeric(
        frame.get("max_weight", pd.Series(index=frame.index, dtype=float)).iloc[start:], errors="coerce"
    ).reindex(returns.index)

    return {
        "n_days": int(n),
        "start": str(returns.index[0].date()),
        "end": str(returns.index[-1].date()),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": _finite(sharpe),
        "sortino": _finite(sortino),
        "max_drawdown": _finite(max_drawdown),
        "calmar": _finite(calmar),
        "hit_rate": float((returns > 0).mean()),
        "skew": _finite(returns.skew()),
        "kurtosis": _finite(returns.kurtosis()),
        "benchmark_total_return": benchmark_total,
        "benchmark_annual_return": benchmark_annual,
        "excess_total_return": total_return - benchmark_total,
        "excess_annual_return": annual_return - benchmark_annual,
        "excess_mean_annual": excess_mean * annualization,
        "tracking_error": _finite(tracking_error),
        "information_ratio": _finite(information_ratio),
        "net_excess_ir": _finite(information_ratio),
        "beta": _finite(beta),
        "alpha_annual": _finite(alpha_annual),
        "turnover_annual": float(turnover.mean() * annualization) if turnover.notna().any() else None,
        "cost_drag_annual": float(cost_ratio.mean() * annualization) if cost_ratio.notna().any() else None,
        "fill_rate_mean": float(fill_rate.mean()) if fill_rate.notna().any() else None,
        "fill_rate_min": float(fill_rate.min()) if fill_rate.notna().any() else None,
        "adv_utilization_max": float(adv_util.max()) if adv_util.notna().any() else None,
        "n_holdings_mean": float(n_holdings.mean()) if n_holdings.notna().any() else None,
        "cash_ratio_mean": float(cash_ratio.mean()) if cash_ratio.notna().any() else None,
        "max_weight_mean": float(max_weight.mean()) if max_weight.notna().any() else None,
        "max_weight_max": float(max_weight.max()) if max_weight.notna().any() else None,
    }


def selection_score(metrics: dict[str, Any], *, turnover_penalty: float = 0.02) -> float:
    """Validation selection score: net excess IR minus a small turnover penalty.

    The penalty only breaks ties between configurations with similar IR; it is
    not a claim that turnover should be minimized at any cost.
    """
    ir = metrics.get("net_excess_ir")
    turnover = metrics.get("turnover_annual")
    if ir is None or not math.isfinite(float(ir)):
        return -1e9
    value = float(ir)
    if turnover is not None and math.isfinite(float(turnover)):
        value -= turnover_penalty * float(turnover)
    fill = metrics.get("fill_rate_mean")
    if fill is not None and float(fill) < 0.95:
        value -= 1.0
    return value


def compound_return(series: pd.Series) -> float:
    """Compound a return series."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float((1.0 + values).prod() - 1.0)


def annualized_return(series: pd.Series, annualization: int = 252) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float((1.0 + values).prod() ** (annualization / len(values)) - 1.0)
