"""Transaction-cost model for Stage 9.

A-share costs are separated on purpose:

* commission: both sides, usually 2.5bp with a CNY 5 minimum per order;
* stamp duty: sell side only, 10bp during 2010-2020 (it was halved in 2023);
* transfer fee: both sides, 0.1bp;
* slippage: modelled as a worse execution price, not as an explicit fee.

The model is deliberately simple and linear.  It does not claim to be a
market-impact model; Stage 9's capacity analysis uses the explicit ADV cap in
the simulator instead.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CostConfig:
    """Round-trip cost parameters in basis points (1bp = 0.01%)."""

    commission_bps_buy: float = 2.5
    commission_bps_sell: float = 2.5
    stamp_duty_bps_sell: float = 10.0
    transfer_fee_bps: float = 0.1
    slippage_bps_buy: float = 5.0
    slippage_bps_sell: float = 5.0
    min_commission: float = 5.0
    lot_size: int = 100
    amount_unit: float = 1000.0
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in (
            "commission_bps_buy",
            "commission_bps_sell",
            "stamp_duty_bps_sell",
            "transfer_fee_bps",
            "slippage_bps_buy",
            "slippage_bps_sell",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.min_commission < 0:
            raise ValueError("min_commission must be non-negative")
        if self.lot_size < 1:
            raise ValueError("lot_size must be >= 1")
        if self.amount_unit <= 0:
            raise ValueError("amount_unit must be positive")

    @classmethod
    def zero(cls, *, lot_size: int = 100) -> CostConfig:
        """A gross-cost configuration used for reconciliation and gross runs."""
        return cls(
            commission_bps_buy=0.0,
            commission_bps_sell=0.0,
            stamp_duty_bps_sell=0.0,
            transfer_fee_bps=0.0,
            slippage_bps_buy=0.0,
            slippage_bps_sell=0.0,
            min_commission=0.0,
            lot_size=lot_size,
            enabled=False,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> CostConfig:
        payload = payload or {}
        return cls(
            commission_bps_buy=float(payload.get("commission_bps_buy", 2.5)),
            commission_bps_sell=float(payload.get("commission_bps_sell", 2.5)),
            stamp_duty_bps_sell=float(payload.get("stamp_duty_bps_sell", 10.0)),
            transfer_fee_bps=float(payload.get("transfer_fee_bps", 0.1)),
            slippage_bps_buy=float(payload.get("slippage_bps_buy", 5.0)),
            slippage_bps_sell=float(payload.get("slippage_bps_sell", 5.0)),
            min_commission=float(payload.get("min_commission", 5.0)),
            lot_size=int(payload.get("lot_size", 100)),
            amount_unit=float(payload.get("amount_unit", 1000.0)),
            enabled=bool(payload.get("enabled", True)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "commission_bps_buy": self.commission_bps_buy,
            "commission_bps_sell": self.commission_bps_sell,
            "stamp_duty_bps_sell": self.stamp_duty_bps_sell,
            "transfer_fee_bps": self.transfer_fee_bps,
            "slippage_bps_buy": self.slippage_bps_buy,
            "slippage_bps_sell": self.slippage_bps_sell,
            "min_commission": self.min_commission,
            "lot_size": self.lot_size,
            "amount_unit": self.amount_unit,
            "enabled": self.enabled,
        }

    def execution_price(self, price: float, side: str) -> float:
        """Apply slippage to a base price."""
        if not self.enabled:
            return float(price)
        slip = self.slippage_bps_buy if side == "buy" else self.slippage_bps_sell
        factor = 1.0 + slip / 10_000.0 if side == "buy" else 1.0 - slip / 10_000.0
        return float(price) * factor

    def fees(self, notional: float, side: str) -> dict[str, float]:
        """Return commission, stamp duty, transfer fee and their sum."""
        if not self.enabled or notional <= 0:
            return {"commission": 0.0, "stamp_duty": 0.0, "transfer_fee": 0.0, "total": 0.0}
        commission_bps = self.commission_bps_buy if side == "buy" else self.commission_bps_sell
        commission = max(self.min_commission, notional * commission_bps / 10_000.0)
        stamp = notional * self.stamp_duty_bps_sell / 10_000.0 if side == "sell" else 0.0
        transfer = notional * self.transfer_fee_bps / 10_000.0
        total = commission + stamp + transfer
        return {
            "commission": float(commission),
            "stamp_duty": float(stamp),
            "transfer_fee": float(transfer),
            "total": float(total),
        }

    def round_buy_raw_shares(self, adjusted_shares: float, factor: float = 1.0) -> int:
        """Round an adjusted share amount to whole **raw** lots.

        Qlib stores adjusted prices and adjusted volume.  A raw A-share lot is
        100 shares, which is ``100 / factor`` adjusted shares.  Qlib's
        ``round_amount_by_trade_unit`` applies the same factor conversion, so
        Stage 9 uses it for the reconciliation path and for the primary run.
        """
        factor = float(factor) if math.isfinite(float(factor)) and float(factor) > 0 else 1.0
        raw_shares = max(0.0, float(adjusted_shares)) * factor
        lots = int(raw_shares // self.lot_size)
        return lots * self.lot_size

    def round_buy_shares(self, shares: float, factor: float = 1.0) -> float:
        """Round a buy quantity down to a whole lot in adjusted-share space."""
        factor = float(factor) if math.isfinite(float(factor)) and float(factor) > 0 else 1.0
        raw = self.round_buy_raw_shares(shares, factor)
        return raw / factor

    def round_sell_shares(self, shares: float, factor: float = 1.0) -> float:
        """Sells may contain odd lots; Qlib does not round sell quantities."""
        del factor
        return max(0.0, float(shares))
