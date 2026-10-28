"""Reconciliation helpers: Qlib TopkDropout reference and NAV comparison."""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

from src.factors.evaluator import EvaluationConfig, _ensure_qlib


def qlib_topkdropout_report(
    scores: pd.DataFrame,
    config: EvaluationConfig,
    *,
    start: str,
    end: str,
    topk: int,
    n_drop: int,
    hold_thresh: int,
    account: float = 100_000_000.0,
    benchmark: str = "SH000300",
    risk_degree: float = 0.95,
) -> pd.DataFrame:
    """Run Qlib's TopkDropoutStrategy with zero costs and no limit checks.

    This is a reference implementation for the simulator's ``sizing='qlib'``
    mode.  It uses Qlib's close-price execution and its own order accounting.
    """
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    _ensure_qlib(config.provider_uri, config.kernels)
    from qlib.backtest import backtest
    from qlib.config import C
    from qlib.contrib.strategy import TopkDropoutStrategy

    C.joblib_backend = "threading"
    C.kernels = 4

    signal = scores.stack(future_stack=True).rename("score")
    signal.index = signal.index.set_names(["datetime", "instrument"])
    signal = signal.reorder_levels(["instrument", "datetime"]).sort_index()
    strategy = TopkDropoutStrategy(
        signal=signal,
        topk=topk,
        n_drop=n_drop,
        hold_thresh=hold_thresh,
        only_tradable=False,
        forbid_all_trade_at_limit=False,
        risk_degree=risk_degree,
    )
    executor = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
    }
    exchange_kwargs = {
        "freq": "day",
        # Qlib treats limit_threshold=None as "use C.limit_threshold"; a 100%
        # threshold is a simple way to disable limit checks in the reference.
        "limit_threshold": 1.0,
        "deal_price": "close",
        "open_cost": 0.0,
        "close_cost": 0.0,
        "min_cost": 0.0,
        "trade_unit": 100,
    }
    metrics, _ = backtest(
        start_time=start,
        end_time=end,
        strategy=strategy,
        executor=executor,
        benchmark=benchmark,
        account=account,
        exchange_kwargs=exchange_kwargs,
    )
    report, _ = metrics["1day"]
    return report.sort_index()


def compare_nav(
    our_daily: pd.DataFrame,
    qlib_report: pd.DataFrame,
    *,
    our_return_col: str = "return",
    qlib_return_col: str = "return",
) -> dict[str, Any]:
    """Compare two daily return streams by cumulative NAV."""
    ours = pd.to_numeric(our_daily[our_return_col], errors="coerce").dropna()
    reference = pd.to_numeric(qlib_report[qlib_return_col], errors="coerce").dropna()
    common = ours.index.intersection(reference.index)
    if len(common) < 3:
        return {"n_days": len(common), "error": "insufficient common dates"}
    ours = ours.loc[common]
    reference = reference.loc[common]
    # Start both NAVs at the first common close; compare returns after it.
    ours_nav = (1.0 + ours.iloc[1:]).cumprod()
    reference_nav = (1.0 + reference.iloc[1:]).cumprod()
    diff = ours_nav - reference_nav
    return {
        "n_days": len(common),
        "start": str(common[0].date()),
        "end": str(common[-1].date()),
        "our_final_nav": float(ours_nav.iloc[-1]),
        "qlib_final_nav": float(reference_nav.iloc[-1]),
        "final_nav_diff": float(ours_nav.iloc[-1] - reference_nav.iloc[-1]),
        "max_abs_nav_diff": float(diff.abs().max()),
        "max_abs_nav_diff_pct": float((diff.abs() / reference_nav.abs()).max()),
        "mean_abs_return_diff": float((ours - reference).abs().mean()),
        "corr": float(np.corrcoef(ours, reference)[0, 1]) if len(ours) > 2 else None,
    }
