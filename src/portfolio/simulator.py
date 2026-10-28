"""Daily TopkDropout portfolio simulator with explicit A-share constraints.

The simulator is intentionally small and auditable.  It is *not* a full
exchange simulator: it has no order book, no intraday path and no market
impact beyond an ADV participation cap.  What it does model explicitly:

* signal at ``t``, trade at ``t+1`` (open by default);
* T+1 (shares bought today cannot be sold today);
* suspension, ST, limit-up/limit-down execution blocks;
* 100-share buy lots, odd-lot sells;
* commission with a minimum, sell-side stamp duty, transfer fee and slippage;
* ADV participation as a hard cap on each order.

Prices are Qlib's adjusted prices.  ``amount`` is converted from Qlib's
thousand-CNY unit to CNY before the ADV cap is applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.portfolio.costs import CostConfig
from src.portfolio.metrics import performance_metrics


@dataclass(frozen=True)
class TradingConstraints:
    """Execution constraints and the ADV cap."""

    exclude_suspended: bool = True
    exclude_st: bool = True
    block_buy_at_limit_up: bool = True
    block_sell_at_limit_down: bool = True
    block_all_at_limit: bool = False
    t_plus_one: bool = True
    adv_participation: float = 0.10
    adv_window: int = 20

    def __post_init__(self) -> None:
        if self.adv_participation < 0:
            raise ValueError("adv_participation must be non-negative")
        if self.adv_window < 1:
            raise ValueError("adv_window must be >= 1")

    @classmethod
    def none(cls) -> TradingConstraints:
        """Disable all explicit constraints (suspension is still untradable)."""
        return cls(
            exclude_suspended=False,
            exclude_st=False,
            block_buy_at_limit_up=False,
            block_sell_at_limit_down=False,
            block_all_at_limit=False,
            t_plus_one=False,
            adv_participation=0.0,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> TradingConstraints:
        payload = payload or {}
        return cls(
            exclude_suspended=bool(payload.get("exclude_suspended", True)),
            exclude_st=bool(payload.get("exclude_st", True)),
            block_buy_at_limit_up=bool(payload.get("block_buy_at_limit_up", True)),
            block_sell_at_limit_down=bool(payload.get("block_sell_at_limit_down", True)),
            block_all_at_limit=bool(payload.get("block_all_at_limit", False)),
            t_plus_one=bool(payload.get("t_plus_one", True)),
            adv_participation=float(payload.get("adv_participation", 0.10)),
            adv_window=int(payload.get("adv_window", 20)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exclude_suspended": self.exclude_suspended,
            "exclude_st": self.exclude_st,
            "block_buy_at_limit_up": self.block_buy_at_limit_up,
            "block_sell_at_limit_down": self.block_sell_at_limit_down,
            "block_all_at_limit": self.block_all_at_limit,
            "t_plus_one": self.t_plus_one,
            "adv_participation": self.adv_participation,
            "adv_window": self.adv_window,
        }


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio construction and execution configuration."""

    topk: int = 50
    n_drop: int = 5
    hold_thresh: int = 1
    account: float = 100_000_000.0
    deal_price: str = "open"
    signal_lag: int = 1
    sizing: str = "equal"
    risk_degree: float = 0.95
    min_names: int = 20
    only_tradable: bool = True
    qlib_compatible: bool = False
    costs: CostConfig = field(default_factory=CostConfig)
    constraints: TradingConstraints = field(default_factory=TradingConstraints)

    def __post_init__(self) -> None:
        if self.topk < 1:
            raise ValueError("topk must be >= 1")
        if self.n_drop < 0:
            raise ValueError("n_drop must be >= 0")
        if self.hold_thresh < 0:
            raise ValueError("hold_thresh must be >= 0")
        if self.account <= 0:
            raise ValueError("account must be positive")
        if self.deal_price not in {"open", "close"}:
            raise ValueError("deal_price must be 'open' or 'close'")
        if self.signal_lag < 0:
            raise ValueError("signal_lag must be >= 0")
        if self.sizing not in {"equal", "qlib"}:
            raise ValueError("sizing must be 'equal' or 'qlib'")
        if not 0.0 < self.risk_degree <= 1.0:
            raise ValueError("risk_degree must be in (0, 1]")
        if self.min_names < 2:
            raise ValueError("min_names must be >= 2")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> PortfolioConfig:
        payload = payload or {}
        return cls(
            topk=int(payload.get("topk", 50)),
            n_drop=int(payload.get("n_drop", 5)),
            hold_thresh=int(payload.get("hold_thresh", 1)),
            account=float(payload.get("account", 100_000_000.0)),
            deal_price=str(payload.get("deal_price", "open")),
            signal_lag=int(payload.get("signal_lag", 1)),
            sizing=str(payload.get("sizing", "equal")),
            risk_degree=float(payload.get("risk_degree", 0.95)),
            min_names=int(payload.get("min_names", 20)),
            only_tradable=bool(payload.get("only_tradable", True)),
            qlib_compatible=bool(payload.get("qlib_compatible", False)),
            costs=CostConfig.from_dict(payload.get("costs")),
            constraints=TradingConstraints.from_dict(payload.get("constraints")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "topk": self.topk,
            "n_drop": self.n_drop,
            "hold_thresh": self.hold_thresh,
            "account": self.account,
            "deal_price": self.deal_price,
            "signal_lag": self.signal_lag,
            "sizing": self.sizing,
            "risk_degree": self.risk_degree,
            "min_names": self.min_names,
            "only_tradable": self.only_tradable,
            "qlib_compatible": self.qlib_compatible,
            "costs": self.costs.to_dict(),
            "constraints": self.constraints.to_dict(),
        }

    def replace(self, **changes: Any) -> PortfolioConfig:
        payload = self.to_dict()
        payload.update(changes)
        return PortfolioConfig.from_dict(payload)


@dataclass
class PortfolioResult:
    """Daily NAV, trades, weights and metrics from one simulation."""

    daily: pd.DataFrame
    trades: pd.DataFrame
    weights: pd.DataFrame
    metrics: dict[str, Any]
    config: dict[str, Any]


@dataclass
class _Position:
    shares: float
    buy_step: int
    buy_date: pd.Timestamp
    last_price: float


def _row(frame: pd.DataFrame | None, date: pd.Timestamp, columns: pd.Index, default: Any) -> pd.Series:
    if frame is None:
        return pd.Series(default, index=columns)
    if date not in frame.index:
        return pd.Series(default, index=columns)
    return frame.loc[date].reindex(columns).fillna(default)


def _prior_signal_date(index: pd.DatetimeIndex, date: pd.Timestamp, lag: int) -> pd.Timestamp | None:
    if lag <= 0:
        return date if date in index else None
    prior = index[index < date]
    if len(prior) < lag:
        return None
    return prior[-lag]


def _value(position: _Position) -> float:
    return float(position.shares) * float(position.last_price)


def run_portfolio(
    scores: pd.DataFrame,
    open_px: pd.DataFrame,
    close_px: pd.DataFrame,
    amount_cny: pd.DataFrame,
    benchmark_close: pd.Series,
    config: PortfolioConfig,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    suspended: pd.DataFrame | None = None,
    is_st: pd.DataFrame | None = None,
    at_limit_up: pd.DataFrame | None = None,
    at_limit_down: pd.DataFrame | None = None,
    factor: pd.DataFrame | None = None,
    collect_weights: bool = True,
) -> PortfolioResult:
    """Run a daily TopkDropout portfolio.

    Parameters
    ----------
    scores:
        Wide frame (datetime x instrument) of signals known at each date.
    open_px, close_px:
        Wide adjusted price panels.  NaNs mean the stock cannot be traded that
        day (suspension or missing data).
    amount_cny:
        Wide CNY turnover panel.  Qlib's ``$amount`` is in thousand CNY; the
        caller must pass ``$amount * 1000``.
    benchmark_close:
        Benchmark close series used for benchmark returns.
    """
    if scores.empty:
        raise ValueError("scores is empty")
    if close_px.empty:
        raise ValueError("close_px is empty")

    benchmark = benchmark_close.sort_index().astype(float)
    dates = pd.DatetimeIndex(benchmark.index)
    if start is not None:
        dates = dates[dates >= pd.Timestamp(start)]
    if end is not None:
        dates = dates[dates <= pd.Timestamp(end)]
    if len(dates) == 0:
        raise ValueError("no trading dates after start/end filter")

    instruments = pd.Index(sorted(set(scores.columns) | set(close_px.columns)))
    score_panel = scores.reindex(index=scores.index.union(dates).sort_values(), columns=instruments)
    open_panel = open_px.reindex(index=dates, columns=instruments)
    close_panel = close_px.reindex(index=dates, columns=instruments)
    amount_panel = amount_cny.reindex(index=dates, columns=instruments)
    adv_window = config.constraints.adv_window
    adv_panel = amount_panel.rolling(adv_window, min_periods=max(1, adv_window // 2)).mean()

    suspended_panel = suspended.reindex(index=dates, columns=instruments) if suspended is not None else None
    st_panel = is_st.reindex(index=dates, columns=instruments) if is_st is not None else None
    limit_up_panel = at_limit_up.reindex(index=dates, columns=instruments) if at_limit_up is not None else None
    limit_down_panel = (
        at_limit_down.reindex(index=dates, columns=instruments) if at_limit_down is not None else None
    )
    factor_panel = (
        factor.reindex(index=dates, columns=instruments).fillna(1.0)
        if factor is not None
        else pd.DataFrame(1.0, index=dates, columns=instruments)
    )

    cash = float(config.account)
    positions: dict[str, _Position] = {}
    daily_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for step, date in enumerate(dates):
        # Mark existing positions to the previous close before today's trades.
        nav_prev = cash + sum(_value(pos) for pos in positions.values())
        benchmark_return = 0.0
        if step > 0:
            previous = dates[step - 1]
            if pd.notna(benchmark.get(previous)) and float(benchmark[previous]) != 0:
                benchmark_return = float(benchmark[date] / benchmark[previous] - 1.0)

        signal_date = _prior_signal_date(score_panel.index, date, config.signal_lag)
        trade_day = signal_date is not None and date in open_panel.index and date in close_panel.index
        if not trade_day:
            daily_rows.append(
                _daily_row(date, cash, positions, nav_prev, benchmark_return, 0.0, 0.0, 1.0, 0.0)
            )
            if collect_weights:
                weight_rows.append(_weight_row(date, cash, positions))
            continue

        score_today = score_panel.loc[signal_date].dropna()
        if len(score_today) < config.min_names:
            daily_rows.append(
                _daily_row(date, cash, positions, nav_prev, benchmark_return, 0.0, 0.0, 1.0, 0.0)
            )
            if collect_weights:
                weight_rows.append(_weight_row(date, cash, positions))
            continue

        base_prices = (open_panel if config.deal_price == "open" else close_panel).loc[date]
        close_today = close_panel.loc[date]
        price_ok = base_prices.notna() & (base_prices > 0)

        susp = _row(suspended_panel, date, instruments, False).astype(bool)
        st = _row(st_panel, date, instruments, False).astype(bool)
        limit_up = _row(limit_up_panel, date, instruments, False).astype(bool)
        limit_down = _row(limit_down_panel, date, instruments, False).astype(bool)
        constraints = config.constraints

        can_buy = price_ok.copy()
        can_sell = price_ok.copy()
        if constraints.exclude_suspended:
            can_buy &= ~susp
            can_sell &= ~susp
        if constraints.exclude_st:
            can_buy &= ~st
        if constraints.block_buy_at_limit_up:
            can_buy &= ~limit_up
        if constraints.block_sell_at_limit_down:
            can_sell &= ~limit_down
        if constraints.block_all_at_limit:
            can_buy &= ~(limit_up | limit_down)
            can_sell &= ~(limit_up | limit_down)

        # Qlib's Position.get_stock_list() uses set order.  Reproducing that
        # order matters for ties among holdings whose score has become NaN
        # (e.g. a stock that left the PIT universe).
        held = list({inst for inst, pos in positions.items() if pos.shares > 0})
        last_index = score_today.reindex(held).sort_values(
            ascending=False, na_position="last"
        ).index
        last = list(last_index)
        candidates = score_today[~score_today.index.isin(held)].sort_values(ascending=False)
        if config.only_tradable and len(candidates):
            candidates = candidates[can_buy.reindex(candidates.index).fillna(False).astype(bool)]
        new_slots = max(0, config.n_drop + config.topk - len(held))
        today = list(candidates.index[:new_slots])
        combined_index = last_index.union(pd.Index(today))
        combined = score_today.reindex(combined_index).sort_values(
            ascending=False, na_position="last"
        )
        bottom = set(combined.index[-config.n_drop :]) if config.n_drop > 0 else set()

        sell_intent = [
            inst for inst in last if inst in bottom and inst in positions
        ]
        sell: list[str] = []
        for inst in sell_intent:
            held_steps = step - positions[inst].buy_step
            if config.constraints.t_plus_one and held_steps < 1:
                continue
            if held_steps < config.hold_thresh:
                continue
            if not bool(can_sell.get(inst, False)):
                continue
            sell.append(inst)

        # Qlib sizes the buy list with *intended* sells, before hold_thresh
        # blocks a sell.  The default Stage 9 path uses actual sells to keep
        # topk strict; qlib_compatible=True reproduces Qlib for reconciliation.
        sell_count = len(sell_intent) if config.qlib_compatible else len(sell)
        buy_slots = max(0, sell_count + config.topk - len(held))
        buy_intent = today[:buy_slots]
        if config.qlib_compatible:
            buy = list(buy_intent)
        else:
            buy = [inst for inst in buy_intent if bool(can_buy.get(inst, False))]

        requested_notional = 0.0
        executed_notional = 0.0
        cost_total = 0.0
        buy_notional_base = 0.0
        sell_notional_base = 0.0
        adv_ratios: list[float] = []
        adv_at_signal = adv_panel.loc[signal_date] if signal_date in adv_panel.index else pd.Series(dtype=float)

        # Sell first.  A blocked sell simply keeps the position; the buy list
        # was already sized after the actual, not intended, sells.
        for inst in sell:
            position = positions.get(inst)
            if position is None or position.shares <= 0:
                continue
            base_price = float(base_prices.get(inst, np.nan))
            if not np.isfinite(base_price) or base_price <= 0:
                continue
            shares = position.shares
            requested = shares * base_price
            requested_notional += requested
            adv = _adv_value(adv_at_signal, inst)
            if constraints.adv_participation > 0 and adv is not None and adv > 0:
                shares = min(shares, constraints.adv_participation * adv / base_price)
            shares = config.costs.round_sell_shares(shares)
            if shares <= 0:
                continue
            exec_price = config.costs.execution_price(base_price, "sell")
            notional = shares * exec_price
            fees = config.costs.fees(notional, "sell")
            slippage_cost = abs(exec_price - base_price) * shares
            cash += notional - fees["total"]
            position.shares -= shares
            if position.shares <= 1e-12:
                positions.pop(inst, None)
            executed_base = shares * base_price
            executed_notional += executed_base
            sell_notional_base += executed_base
            cost_total += fees["total"] + slippage_cost
            if adv is not None and adv > 0:
                adv_ratios.append(executed_base / adv)
            trade_rows.append(
                {
                    "datetime": date,
                    "instrument": inst,
                    "side": "sell",
                    "shares": shares,
                    "base_price": base_price,
                    "price": exec_price,
                    "notional": notional,
                    "commission": fees["commission"],
                    "stamp_duty": fees["stamp_duty"],
                    "transfer_fee": fees["transfer_fee"],
                    "slippage_cost": slippage_cost,
                    "total_cost": fees["total"] + slippage_cost,
                    "requested_notional": requested,
                    "adv_cny": adv,
                    "adv_utilization": executed_base / adv if adv and adv > 0 else None,
                }
            )

        # Buy.  Equal sizing targets equity/topk per name.  Qlib sizing uses
        # cash * risk_degree / len(buy), matching Qlib's TopkDropoutStrategy
        # closely enough for the reconciliation check.
        if buy:
            equity_open = cash + sum(
                pos.shares * _mark_price(base_prices, inst, pos.last_price) for inst, pos in positions.items()
            )
            if config.sizing == "equal":
                target_count = max(1, min(config.topk, len(positions) + len(buy)))
                per_name = equity_open / target_count
            else:
                per_name = cash * config.risk_degree / len(buy)
            remaining = len(buy)
            for inst in buy:
                base_price = float(base_prices.get(inst, np.nan))
                if not bool(can_buy.get(inst, False)) or not np.isfinite(base_price) or base_price <= 0:
                    remaining -= 1
                    continue
                requested = per_name
                requested_notional += requested
                desired = requested
                adv = _adv_value(adv_at_signal, inst)
                if constraints.adv_participation > 0 and adv is not None and adv > 0:
                    desired = min(desired, constraints.adv_participation * adv)
                if config.sizing == "equal" and remaining > 0:
                    desired = min(desired, cash / remaining)
                exec_price = config.costs.execution_price(base_price, "buy")
                factor_value = float(factor_panel.loc[date].get(inst, 1.0))
                if not np.isfinite(factor_value) or factor_value <= 0:
                    factor_value = 1.0
                raw_qty = config.costs.round_buy_raw_shares(desired / exec_price, factor_value)
                while raw_qty > 0:
                    qty = raw_qty / factor_value
                    notional = qty * exec_price
                    fees = config.costs.fees(notional, "buy")
                    if notional + fees["total"] <= cash + 1e-9:
                        break
                    raw_qty -= config.costs.lot_size
                if raw_qty <= 0:
                    remaining -= 1
                    continue
                qty = raw_qty / factor_value
                notional = qty * exec_price
                fees = config.costs.fees(notional, "buy")
                slippage_cost = abs(exec_price - base_price) * qty
                cash -= notional + fees["total"]
                positions[inst] = _Position(
                    shares=qty,
                    buy_step=step,
                    buy_date=date,
                    last_price=float(close_today.get(inst, base_price)),
                )
                executed_base = qty * base_price
                executed_notional += executed_base
                buy_notional_base += executed_base
                cost_total += fees["total"] + slippage_cost
                if adv is not None and adv > 0:
                    adv_ratios.append(executed_base / adv)
                trade_rows.append(
                    {
                        "datetime": date,
                        "instrument": inst,
                        "side": "buy",
                        "shares": qty,
                        "base_price": base_price,
                        "price": exec_price,
                        "notional": notional,
                        "commission": fees["commission"],
                        "stamp_duty": fees["stamp_duty"],
                        "transfer_fee": fees["transfer_fee"],
                        "slippage_cost": slippage_cost,
                        "total_cost": fees["total"] + slippage_cost,
                        "requested_notional": requested,
                        "adv_cny": adv,
                        "adv_utilization": executed_base / adv if adv and adv > 0 else None,
                    }
                )
                remaining -= 1

        # Mark all remaining positions to today's close.
        for inst, position in positions.items():
            close_value = close_today.get(inst, np.nan)
            if pd.notna(close_value) and float(close_value) > 0:
                position.last_price = float(close_value)
        nav = cash + sum(_value(pos) for pos in positions.values())
        fill_rate = executed_notional / requested_notional if requested_notional > 1e-12 else 1.0
        turnover = (buy_notional_base + sell_notional_base) / 2.0 / nav_prev if nav_prev > 0 else 0.0
        daily_rows.append(
            _daily_row(
                date,
                cash,
                positions,
                nav_prev,
                benchmark_return,
                turnover,
                cost_total,
                fill_rate,
                max(adv_ratios) if adv_ratios else 0.0,
                nav=nav,
                return_value=nav / nav_prev - 1.0 if nav_prev > 0 else 0.0,
            )
        )
        if collect_weights:
            weight_rows.append(_weight_row(date, cash, positions, nav=nav))

    daily = pd.DataFrame(daily_rows).set_index("datetime").sort_index()
    trades = pd.DataFrame(trade_rows)
    if collect_weights:
        weights = pd.DataFrame(weight_rows).set_index("datetime").sort_index().fillna(0.0)
    else:
        weights = pd.DataFrame()
    metrics = performance_metrics(daily, drop_days=2)
    return PortfolioResult(
        daily=daily,
        trades=trades,
        weights=weights,
        metrics=metrics,
        config=config.to_dict(),
    )


def _adv_value(adv: pd.Series, instrument: str) -> float | None:
    value = adv.get(instrument, np.nan) if not adv.empty else np.nan
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    return float(value)


def _mark_price(base_prices: pd.Series, instrument: str, fallback: float) -> float:
    value = base_prices.get(instrument, np.nan)
    return float(value) if pd.notna(value) and float(value) > 0 else float(fallback)


def _daily_row(
    date: pd.Timestamp,
    cash: float,
    positions: dict[str, _Position],
    nav_prev: float,
    benchmark_return: float,
    turnover: float,
    cost: float,
    fill_rate: float,
    adv_utilization_max: float,
    *,
    nav: float | None = None,
    return_value: float | None = None,
) -> dict[str, Any]:
    invested = sum(_value(pos) for pos in positions.values())
    nav_value = float(nav) if nav is not None else cash + invested
    weights = [pos.shares * pos.last_price / nav_value for pos in positions.values()] if nav_value > 0 else []
    return {
        "datetime": date,
        "nav": nav_value,
        "nav_prev": nav_prev,
        "cash": cash,
        "invested": invested,
        "return": float(return_value) if return_value is not None else (nav_value / nav_prev - 1.0 if nav_prev > 0 else 0.0),
        "benchmark_return": benchmark_return,
        "turnover": turnover,
        "cost": cost,
        "fill_rate": fill_rate,
        "adv_utilization_max": adv_utilization_max,
        "n_holdings": len(positions),
        "cash_ratio": cash / nav_value if nav_value > 0 else 0.0,
        "max_weight": max(weights) if weights else 0.0,
    }


def _weight_row(
    date: pd.Timestamp,
    cash: float,
    positions: dict[str, _Position],
    *,
    nav: float | None = None,
) -> dict[str, Any]:
    invested = sum(_value(pos) for pos in positions.values())
    nav_value = float(nav) if nav is not None else cash + invested
    row: dict[str, Any] = {"datetime": date}
    for inst, pos in positions.items():
        row[inst] = pos.shares * pos.last_price / nav_value if nav_value > 0 else 0.0
    return row
