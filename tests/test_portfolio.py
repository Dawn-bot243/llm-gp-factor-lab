from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.portfolio.attribution import (
    benchmark_exposure,
    brinson_attribution,
    portfolio_exposure,
    size_bucket_groups,
    style_exposures,
)
from src.portfolio.combine import (
    build_combination,
    forward_returns,
    train_icir_weights,
)
from src.portfolio.costs import CostConfig
from src.portfolio.metrics import performance_metrics
from src.portfolio.simulator import (
    PortfolioConfig,
    TradingConstraints,
    run_portfolio,
)


def _panels(
    n_dates: int = 30,
    n_inst: int = 30,
    *,
    start: str = "2020-01-01",
    seed: int = 7,
) -> tuple[pd.DatetimeIndex, list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    dates = pd.bdate_range(start, periods=n_dates)
    instruments = [f"S{i:02d}" for i in range(n_inst)]
    rng = np.random.default_rng(seed)
    close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=(n_dates, n_inst)), axis=0)),
        index=dates,
        columns=instruments,
    )
    open_px = close.shift(1).fillna(close.iloc[0])
    amount = pd.DataFrame(1e8, index=dates, columns=instruments)
    benchmark = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n_dates))), index=dates)
    return dates, instruments, open_px, close, amount, benchmark


def _simple_sim(
    *,
    topk: int = 1,
    n_drop: int = 0,
    hold_thresh: int = 1,
    account: float = 100_000.0,
    costs: CostConfig | None = None,
    constraints: TradingConstraints | None = None,
    t_plus_one: bool = False,
) -> tuple:
    dates = pd.bdate_range("2020-01-01", periods=7)
    instruments = ["A", "B"]
    open_px = pd.DataFrame(10.0, index=dates, columns=instruments)
    close_px = pd.DataFrame(
        {"A": [10, 11, 12, 13, 14, 15, 16], "B": [10, 10, 10, 10, 10, 10, 10]},
        index=dates,
    )
    amount = pd.DataFrame(1e9, index=dates, columns=instruments)
    benchmark = pd.Series(100.0, index=dates)
    scores = pd.DataFrame({"A": 2.0, "B": 1.0}, index=dates)
    config = PortfolioConfig(
        topk=topk,
        n_drop=n_drop,
        hold_thresh=hold_thresh,
        account=account,
        deal_price="open",
        sizing="equal",
        min_names=2,
        costs=costs or CostConfig.zero(),
        constraints=constraints
        or TradingConstraints(
            exclude_suspended=True,
            exclude_st=True,
            block_buy_at_limit_up=True,
            block_sell_at_limit_down=True,
            t_plus_one=t_plus_one,
            adv_participation=0.0,
        ),
    )
    result = run_portfolio(
        scores,
        open_px,
        close_px,
        amount,
        benchmark,
        config,
        start=dates[0],
        end=dates[-1],
    )
    return result, dates, instruments


def test_cost_model_separates_commission_stamp_and_transfer():
    costs = CostConfig()
    buy = costs.fees(100_000.0, "buy")
    sell = costs.fees(100_000.0, "sell")
    assert buy["commission"] == pytest.approx(25.0)
    assert buy["stamp_duty"] == 0.0
    assert buy["transfer_fee"] == pytest.approx(1.0)
    assert sell["stamp_duty"] == pytest.approx(100.0)
    assert costs.execution_price(10.0, "buy") == pytest.approx(10.005)
    assert costs.execution_price(10.0, "sell") == pytest.approx(9.995)
    assert costs.round_buy_shares(199) == 100
    assert costs.round_sell_shares(199) == 199
    zero = CostConfig.zero()
    assert zero.fees(100_000.0, "sell")["total"] == 0.0


def test_one_stock_buy_and_hold_nav():
    result, dates, _ = _simple_sim()
    # Step 0 has no prior signal; step 1 buys A at 10.0 with 100k.
    assert result.trades.iloc[0]["datetime"] == dates[1]
    assert result.trades.iloc[0]["shares"] == 10_000
    assert result.daily["nav"].iloc[-1] == pytest.approx(160_000.0)
    assert result.daily["n_holdings"].iloc[-1] == 1
    assert result.weights.iloc[-1]["A"] == pytest.approx(1.0)


def test_cost_accounting_matches_hand_calculation():
    result, _, _ = _simple_sim(costs=CostConfig(), account=100_000.0)
    trade = result.trades.iloc[0]
    assert trade["price"] == pytest.approx(10.005)
    assert trade["shares"] == 9_900
    assert trade["commission"] == pytest.approx(max(5.0, 99_049.5 * 2.5 / 10_000))
    assert trade["transfer_fee"] == pytest.approx(99_049.5 * 0.1 / 10_000)
    expected_cash = 100_000.0 - trade["notional"] - trade["commission"] - trade["transfer_fee"]
    assert result.daily.loc[trade["datetime"], "cash"] == pytest.approx(expected_cash)
    assert result.daily["nav"].iloc[-1] < 160_000.0


def test_topkdropout_switches_names_and_respects_t_plus_one():
    dates = pd.bdate_range("2020-01-01", periods=8)
    instruments = ["A", "B"]
    open_px = pd.DataFrame(10.0, index=dates, columns=instruments)
    close_px = pd.DataFrame(10.0, index=dates, columns=instruments)
    amount = pd.DataFrame(1e9, index=dates, columns=instruments)
    benchmark = pd.Series(100.0, index=dates)
    scores = pd.DataFrame(
        {"A": [2, 2, 1, 1, 1, 1, 1, 1], "B": [1, 1, 2, 2, 2, 2, 2, 2]},
        index=dates,
    )
    config = PortfolioConfig(
        topk=1,
        n_drop=1,
        hold_thresh=1,
        account=100_000.0,
        deal_price="open",
        min_names=2,
        costs=CostConfig.zero(),
        constraints=TradingConstraints.none(),
    )
    result = run_portfolio(scores, open_px, close_px, amount, benchmark, config, start=dates[0], end=dates[-1])
    assert result.trades.iloc[0]["instrument"] == "A"
    assert result.trades.iloc[0]["side"] == "buy"
    # The signal flips on date index 2 and the trade happens on index 3.
    switch = result.trades[result.trades["side"] == "sell"]
    assert len(switch) == 1
    assert switch.iloc[0]["datetime"] == dates[3]
    assert result.trades.iloc[-1]["instrument"] == "B"


def test_hold_thresh_delays_sell():
    dates = pd.bdate_range("2020-01-01", periods=10)
    instruments = ["A", "B"]
    open_px = pd.DataFrame(10.0, index=dates, columns=instruments)
    close_px = pd.DataFrame(10.0, index=dates, columns=instruments)
    amount = pd.DataFrame(1e9, index=dates, columns=instruments)
    benchmark = pd.Series(100.0, index=dates)
    scores = pd.DataFrame(
        {"A": [2, 2, 1, 1, 1, 1, 1, 1, 1, 1], "B": [1, 1, 2, 2, 2, 2, 2, 2, 2, 2]},
        index=dates,
    )
    config = PortfolioConfig(
        topk=1,
        n_drop=1,
        hold_thresh=5,
        account=100_000.0,
        deal_price="open",
        min_names=2,
        costs=CostConfig.zero(),
        constraints=TradingConstraints.none(),
    )
    result = run_portfolio(scores, open_px, close_px, amount, benchmark, config, start=dates[0], end=dates[-1])
    sell = result.trades[result.trades["side"] == "sell"]
    assert len(sell) == 1
    # Bought at step 1; held_steps=5 at step 6.
    assert sell.iloc[0]["datetime"] == dates[6]


def test_suspension_and_limit_blocks():
    dates = pd.bdate_range("2020-01-01", periods=6)
    instruments = ["A", "B"]
    open_px = pd.DataFrame(10.0, index=dates, columns=instruments)
    close_px = pd.DataFrame(10.0, index=dates, columns=instruments)
    # A is untradable on the first trade date.
    open_px.loc[dates[1], "A"] = np.nan
    close_px.loc[dates[1], "A"] = np.nan
    amount = pd.DataFrame(1e9, index=dates, columns=instruments)
    benchmark = pd.Series(100.0, index=dates)
    scores = pd.DataFrame({"A": 2.0, "B": 1.0}, index=dates)
    config = PortfolioConfig(
        topk=1,
        n_drop=0,
        hold_thresh=1,
        account=100_000.0,
        deal_price="open",
        min_names=2,
        costs=CostConfig.zero(),
        constraints=TradingConstraints(t_plus_one=True, adv_participation=0.0),
        only_tradable=False,
    )
    result = run_portfolio(scores, open_px, close_px, amount, benchmark, config, start=dates[0], end=dates[-1])
    assert result.trades.iloc[0]["datetime"] == dates[2]

    # Limit-up blocks a buy.
    limit_up = pd.DataFrame(False, index=dates, columns=instruments)
    limit_up.loc[dates[1], "A"] = True
    result = run_portfolio(
        scores,
        pd.DataFrame(10.0, index=dates, columns=instruments),
        pd.DataFrame(10.0, index=dates, columns=instruments),
        amount,
        benchmark,
        config,
        start=dates[0],
        end=dates[-1],
        at_limit_up=limit_up,
    )
    assert result.trades.iloc[0]["datetime"] == dates[2]

    # Limit-down blocks a sell.  A is bought first, then B becomes better.
    scores2 = pd.DataFrame(
        {"A": [2, 2, 1, 1, 1, 1], "B": [1, 1, 2, 2, 2, 2]},
        index=dates,
    )
    limit_down = pd.DataFrame(False, index=dates, columns=instruments)
    limit_down.loc[dates[2], "A"] = True
    config2 = PortfolioConfig(
        topk=1,
        n_drop=1,
        hold_thresh=1,
        account=100_000.0,
        deal_price="open",
        min_names=2,
        costs=CostConfig.zero(),
        constraints=TradingConstraints(t_plus_one=True, adv_participation=0.0),
    )
    result = run_portfolio(
        scores2,
        pd.DataFrame(10.0, index=dates, columns=instruments),
        pd.DataFrame(10.0, index=dates, columns=instruments),
        amount,
        benchmark,
        config2,
        start=dates[0],
        end=dates[-1],
        at_limit_down=limit_down,
    )
    sell = result.trades[result.trades["side"] == "sell"]
    assert len(sell) == 1
    assert sell.iloc[0]["datetime"] == dates[3]


def test_adv_cap_limits_buy_size():
    dates = pd.bdate_range("2020-01-01", periods=5)
    instruments = ["A", "B"]
    open_px = pd.DataFrame(10.0, index=dates, columns=instruments)
    close_px = pd.DataFrame(10.0, index=dates, columns=instruments)
    amount = pd.DataFrame(10_000.0, index=dates, columns=instruments)
    benchmark = pd.Series(100.0, index=dates)
    scores = pd.DataFrame({"A": 2.0, "B": 1.0}, index=dates)
    config = PortfolioConfig(
        topk=1,
        n_drop=0,
        hold_thresh=1,
        account=100_000.0,
        deal_price="open",
        min_names=2,
        costs=CostConfig.zero(),
        constraints=TradingConstraints(t_plus_one=False, adv_participation=0.1, adv_window=1),
    )
    result = run_portfolio(scores, open_px, close_px, amount, benchmark, config, start=dates[0], end=dates[-1])
    trade = result.trades.iloc[0]
    assert trade["shares"] == 100
    assert trade["adv_cny"] == pytest.approx(10_000.0)
    assert result.daily.loc[trade["datetime"], "fill_rate"] < 0.02


def test_forward_return_is_tradable_open_to_open():
    dates = pd.bdate_range("2020-01-01", periods=5)
    open_px = pd.DataFrame({"A": [10.0, 11.0, 12.0, 13.0, 14.0]}, index=dates)
    fwd = forward_returns(open_px, horizon=1, lag=1)
    assert fwd.loc[("A", dates[0])] == pytest.approx(12.0 / 11.0 - 1.0)
    assert np.isnan(fwd.loc[("A", dates[-1])])


def test_icir_weights_and_combination():
    dates = pd.bdate_range("2020-01-01", periods=240)
    instruments = [f"S{i:02d}" for i in range(30)]
    index = pd.MultiIndex.from_product([instruments, dates], names=["instrument", "datetime"])
    rng = np.random.default_rng(3)
    label = pd.Series(rng.normal(0, 0.02, len(index)), index=index, name="fwd_1")
    good = label + pd.Series(rng.normal(0, 0.01, len(index)), index=index)
    noise = pd.Series(rng.normal(0, 1, len(index)), index=index)
    factors = pd.DataFrame({"good": good, "noise": noise})
    diagnostics, weights = train_icir_weights(
        factors, label, train_start=dates[0], train_end=dates[-1], min_names=20, min_dates=50
    )
    assert diagnostics.loc["good", "sign"] > 0
    assert weights["good"] > weights["noise"]
    result = build_combination(
        "icir",
        factors,
        label,
        train_start=dates[0],
        train_end=dates[-1],
        min_names=20,
        min_dates=50,
    )
    assert result.scores.shape[0] == len(dates)
    assert result.scores.notna().any().any()


def test_brinson_reconciles_to_active_return():
    dates = pd.bdate_range("2020-01-01", periods=40)
    instruments = [f"S{i:02d}" for i in range(10)]
    rng = np.random.default_rng(4)
    returns = pd.DataFrame(rng.normal(0, 0.01, (len(dates), len(instruments))), index=dates, columns=instruments)
    weights = pd.DataFrame(0.08, index=dates, columns=instruments)
    groups = pd.Series({inst: "G1" if i < 4 else "G2" for i, inst in enumerate(instruments)})
    result = brinson_attribution(weights, returns, groups, weight_lag=0)
    summary = result["summary"]
    assert summary["n_days"] == len(dates)
    assert summary["max_reconciliation_error"] < 1e-10
    assert summary["allocation_cumulative"] + summary["selection_cumulative"] + summary["interaction_cumulative"] == pytest.approx(
        summary["active_cumulative"], abs=1e-9
    )


def test_style_exposures_and_size_buckets():
    dates, instruments, _, close, amount, benchmark = _panels(n_dates=80, n_inst=30)
    style = style_exposures(close, amount, benchmark, min_names=10)
    assert isinstance(style.columns, pd.MultiIndex)
    assert style.columns.names == ["field", "instrument"]
    fields = set(style.columns.get_level_values("field"))
    assert {"size", "beta60", "momentum20", "vol20", "illiq20"}.issubset(fields)
    weights = pd.DataFrame(1.0 / len(instruments), index=dates, columns=instruments)
    port = portfolio_exposure(weights, style)
    bench = benchmark_exposure(style, pd.DataFrame(1, index=dates, columns=instruments))
    assert port.shape == (len(dates), 6)
    assert bench.shape == (len(dates), 6)
    groups = size_bucket_groups(style["size"], quantiles=5)
    assert groups.shape == style["size"].shape
    assert groups.iloc[-1].notna().sum() == len(instruments)


def test_performance_metrics_known_series():
    dates = pd.bdate_range("2020-01-01", periods=10)
    returns = pd.Series(0.001 + 0.0002 * np.sin(np.arange(len(dates))), index=dates)
    bench = pd.Series(0.0005 + 0.0001 * np.cos(np.arange(len(dates))), index=dates)
    nav = 100.0 * (1 + returns).cumprod()
    daily = pd.DataFrame(
        {
            "nav": nav,
            "nav_prev": nav.shift(1).fillna(100.0),
            "return": returns,
            "benchmark_return": bench,
            "turnover": 0.1,
            "cost": 0.0,
            "n_holdings": 10,
            "cash_ratio": 0.0,
            "max_weight": 0.1,
            "fill_rate": 1.0,
            "adv_utilization_max": 0.0,
        },
        index=dates,
    )
    metrics = performance_metrics(daily, drop_days=0)
    assert metrics["n_days"] == len(dates)
    assert metrics["annual_return"] > 0
    assert metrics["sharpe"] > 0
    assert metrics["max_drawdown"] == pytest.approx(0.0)
