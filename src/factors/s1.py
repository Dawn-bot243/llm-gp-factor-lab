"""Stage 4 S1 single-factor standardization and scorecard.

S1 is a statistical screen, not a portfolio backtest.  The scorecard is
computed on a calibration window (2010-2018 by default); the 2019-2020
holdout is reported separately and is not used to tune thresholds.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

from src.factors.evaluator import (
    EvaluationConfig,
    load_close_panel,
    load_factor,
    membership_row_count,
)
from src.factors.expression import DegenerateFactorError, ValidatedExpression
from src.factors.metrics import (
    compute_ic,
    coverage_stats,
    distribution_stats,
    ic_stats,
    quantile_returns,
    quantile_stats,
    rank_autocorrelation,
    top_quantile_turnover,
)
from src.factors.preprocess import preprocess_factor

HARD_REASONS = frozenset({"LOW_COVERAGE", "WEAK_IC", "WEAK_T", "FDR_FAIL"})


@dataclass(frozen=True)
class S1Thresholds:
    """Pass/review/fail thresholds for the S1 scorecard."""

    min_abs_rank_ic: float = 0.015
    min_abs_icir: float = 0.10
    min_abs_nw_t: float = 2.0
    min_coverage: float = 0.80
    min_abs_monotonicity: float = 0.50
    max_turnover: float = 0.60
    min_year_sign_consistency: float = 0.55
    fdr_alpha: float = 0.10

    def __post_init__(self) -> None:
        if min(self.min_abs_rank_ic, self.min_abs_icir, self.min_abs_nw_t) < 0:
            raise ValueError("strength thresholds must be non-negative")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in [0, 1]")
        if not 0.0 <= self.fdr_alpha <= 1.0:
            raise ValueError("fdr_alpha must be in [0, 1]")
        if not 0.0 <= self.min_abs_monotonicity <= 1.0:
            raise ValueError("min_abs_monotonicity must be in [0, 1]")
        if not 0.0 <= self.min_year_sign_consistency <= 1.0:
            raise ValueError("min_year_sign_consistency must be in [0, 1]")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> S1Thresholds:
        payload = payload or {}
        return cls(
            min_abs_rank_ic=float(payload.get("min_abs_rank_ic", 0.015)),
            min_abs_icir=float(payload.get("min_abs_icir", 0.10)),
            min_abs_nw_t=float(payload.get("min_abs_nw_t", 2.0)),
            min_coverage=float(payload.get("min_coverage", 0.80)),
            min_abs_monotonicity=float(payload.get("min_abs_monotonicity", 0.50)),
            max_turnover=float(payload.get("max_turnover", 0.60)),
            min_year_sign_consistency=float(payload.get("min_year_sign_consistency", 0.55)),
            fdr_alpha=float(payload.get("fdr_alpha", 0.10)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "min_abs_rank_ic": self.min_abs_rank_ic,
            "min_abs_icir": self.min_abs_icir,
            "min_abs_nw_t": self.min_abs_nw_t,
            "min_coverage": self.min_coverage,
            "min_abs_monotonicity": self.min_abs_monotonicity,
            "max_turnover": self.max_turnover,
            "min_year_sign_consistency": self.min_year_sign_consistency,
            "fdr_alpha": self.fdr_alpha,
        }


@dataclass(frozen=True)
class S1Config:
    """Evaluation protocol for Stage 4 S1."""

    quantiles: tuple[int, ...] = (5, 10)
    min_names: int = 20
    winsorize: float | None = 0.01
    standardize: bool = True
    decay_horizons: tuple[int, ...] = tuple(range(1, 21))
    calibration_start: str = "2010-01-01"
    calibration_end: str = "2018-12-31"
    holdout_start: str = "2019-01-01"
    holdout_end: str = "2020-09-23"
    thresholds: S1Thresholds = field(default_factory=S1Thresholds)

    def __post_init__(self) -> None:
        if len(self.quantiles) < 1 or any(q < 2 for q in self.quantiles):
            raise ValueError("quantiles must all be >= 2")
        if not self.decay_horizons or any(h < 1 for h in self.decay_horizons):
            raise ValueError("decay_horizons must all be >= 1")
        if self.min_names < 2:
            raise ValueError("min_names must be >= 2")
        if self.winsorize is not None and not 0.0 <= self.winsorize < 0.5:
            raise ValueError("winsorize must be in [0, 0.5) or None")
        if pd.Timestamp(self.calibration_start) > pd.Timestamp(self.calibration_end):
            raise ValueError("calibration_start is after calibration_end")
        if pd.Timestamp(self.holdout_start) > pd.Timestamp(self.holdout_end):
            raise ValueError("holdout_start is after holdout_end")

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> S1Config:
        payload = payload or {}
        calibration = payload.get("calibration", {})
        holdout = payload.get("holdout", {})
        quantiles = payload.get("quantiles", (5, 10))
        horizons = payload.get("decay_horizons", tuple(range(1, 21)))
        winsorize_value = payload.get("winsorize", 0.01)
        return cls(
            quantiles=tuple(int(q) for q in quantiles),
            min_names=int(payload.get("min_names", 20)),
            winsorize=None if winsorize_value is None else float(winsorize_value),
            standardize=bool(payload.get("standardize", True)),
            decay_horizons=tuple(int(h) for h in horizons),
            calibration_start=str(calibration.get("start", "2010-01-01")),
            calibration_end=str(calibration.get("end", "2018-12-31")),
            holdout_start=str(holdout.get("start", "2019-01-01")),
            holdout_end=str(holdout.get("end", "2020-09-23")),
            thresholds=S1Thresholds.from_dict(payload.get("thresholds")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantiles": list(self.quantiles),
            "min_names": self.min_names,
            "winsorize": self.winsorize,
            "standardize": self.standardize,
            "decay_horizons": list(self.decay_horizons),
            "calibration": {"start": self.calibration_start, "end": self.calibration_end},
            "holdout": {"start": self.holdout_start, "end": self.holdout_end},
            "thresholds": self.thresholds.to_dict(),
        }


class ForwardReturns:
    """Lazily materialize forward returns from one adjusted-close panel.

    ``fwd_h = close[t+1+h] / close[t+1] - 1``: signal at t, trade at t+1
    close, hold for h trading days.  This matches the Stage 1/3 label.
    """

    def __init__(self, close: pd.Series, horizons: Iterable[int]) -> None:
        if not isinstance(close.index, pd.MultiIndex):
            raise TypeError("close must have a (instrument, datetime) MultiIndex")
        self._wide = close.unstack("instrument").sort_index()
        self.horizons = tuple(sorted({int(h) for h in horizons}))
        if not self.horizons or any(h < 1 for h in self.horizons):
            raise ValueError("horizons must all be >= 1")
        self._cache: dict[int, pd.Series] = {}

    @classmethod
    def from_config(cls, config: EvaluationConfig, horizons: Iterable[int]) -> ForwardReturns:
        close_panel = load_close_panel(config)
        return cls(close_panel["close"], horizons)

    def get(self, horizon: int) -> pd.Series:
        if horizon not in self.horizons:
            raise KeyError(f"horizon {horizon} was not loaded; available: {self.horizons}")
        if horizon not in self._cache:
            fwd = self._wide.shift(-(horizon + 1)) / self._wide.shift(-1) - 1.0
            try:
                stacked = fwd.stack(future_stack=True)
            except TypeError:  # pragma: no cover - older pandas fallback
                stacked = fwd.stack()
            stacked.index = stacked.index.set_names(["datetime", "instrument"])
            stacked = stacked.reorder_levels(["instrument", "datetime"]).sort_index()
            self._cache[horizon] = stacked.rename(f"fwd_{horizon}")
        return self._cache[horizon]


@dataclass
class S1Result:
    metrics: dict[str, Any]
    scorecard: dict[str, Any]
    rank_ic: pd.Series


def evaluate_s1(
    expression: str,
    data_config: EvaluationConfig,
    s1_config: S1Config,
    forward: ForwardReturns,
    *,
    name: str | None = None,
    allowed_fields: Iterable[str] | None = None,
    validated: ValidatedExpression | None = None,
    factor: pd.DataFrame | None = None,
) -> S1Result:
    """Evaluate one expression and return its S1 metrics and scorecard."""
    if validated is None or factor is None:
        validated, factor = load_factor(expression, data_config, allowed_fields=allowed_fields)
    display_name = name or validated.canonical

    raw_distribution = distribution_stats(factor["factor"])
    processed = preprocess_factor(
        factor,
        winsorize=s1_config.winsorize,
        standardize=s1_config.standardize,
    )
    panel = factor.copy()
    panel["factor"] = processed
    if panel["factor"].notna().sum() == 0:
        raise DegenerateFactorError("factor has no finite values after preprocessing")

    membership_rows = membership_row_count(data_config) or len(factor)
    coverage = coverage_stats(panel["factor"], total_rows=membership_rows)
    if coverage["finite_rows"] == 0:
        raise DegenerateFactorError("factor has no finite values")
    if coverage["unique_values"] <= 1:
        raise DegenerateFactorError("factor is constant")

    fwd_1 = forward.get(1)
    panel = panel[["factor"]].join(fwd_1, how="left")
    calibration = _slice(panel, s1_config.calibration_start, s1_config.calibration_end)
    holdout = _slice(panel, s1_config.holdout_start, s1_config.holdout_end)
    if calibration.empty:
        raise ValueError("calibration window has no rows")
    if holdout.empty:
        raise ValueError("holdout window has no rows")

    cal_ic, cal_pearson = compute_ic(
        calibration, "factor", "fwd_1", min_names=s1_config.min_names
    )
    cal_stats = ic_stats(cal_ic)
    hold_ic, _ = compute_ic(holdout, "factor", "fwd_1", min_names=s1_config.min_names)
    hold_stats = ic_stats(hold_ic)
    yearly = _yearly_ic(cal_ic)

    quantile_results: dict[str, Any] = {}
    for q in s1_config.quantiles:
        qmean, long_short, _ = quantile_returns(
            calibration, "factor", "fwd_1", q=q, min_names=s1_config.min_names
        )
        quantile_results[f"q{q}"] = {"q": q, **quantile_stats(qmean, long_short)}

    turnover_q = 5 if 5 in s1_config.quantiles else s1_config.quantiles[0]
    turnover = top_quantile_turnover(
        calibration, "factor", q=turnover_q, min_names=s1_config.min_names
    )
    autocorr = rank_autocorrelation(calibration, "factor", min_names=s1_config.min_names)

    decay: list[dict[str, Any]] = []
    for horizon in s1_config.decay_horizons:
        h_panel = panel[["factor"]].join(forward.get(horizon), how="left")
        h_cal = _slice(h_panel, s1_config.calibration_start, s1_config.calibration_end)
        h_ic, _ = compute_ic(h_cal, "factor", f"fwd_{horizon}", min_names=s1_config.min_names)
        st = ic_stats(h_ic)
        decay.append(
            {
                "horizon": horizon,
                "n_dates": st["n_dates"],
                "mean": st["mean"],
                "icir": st["icir"],
                "newey_west_t": st["newey_west_t"],
            }
        )

    q5 = quantile_results.get("q5") or quantile_results[f"q{s1_config.quantiles[0]}"]
    monotonicity = q5.get("monotonicity")
    year_consistency = yearly["sign_consistency"]
    reasons = _score_reasons(
        coverage=coverage.get("coverage"),
        rank_ic=cal_stats.get("mean"),
        icir=cal_stats.get("icir"),
        nw_t=cal_stats.get("newey_west_t"),
        monotonicity=monotonicity,
        turnover=turnover,
        year_consistency=year_consistency,
        thresholds=s1_config.thresholds,
    )
    status = _status(reasons)
    p_value = _p_value(cal_stats)
    scorecard = {
        "name": display_name,
        "expression": validated.raw,
        "canonical": validated.canonical,
        "expression_hash": validated.expression_hash,
        "direction": _direction(cal_stats.get("mean")),
        "status": status,
        "reasons": reasons,
        "rank_ic_mean": cal_stats.get("mean"),
        "icir": cal_stats.get("icir"),
        "newey_west_t": cal_stats.get("newey_west_t"),
        "p_value": p_value,
        "q_value": None,
        "fdr_pass": None,
        "coverage": coverage.get("coverage"),
        "turnover": turnover,
        "monotonicity": monotonicity,
        "year_sign_consistency": year_consistency,
        "holdout_rank_ic": hold_stats.get("mean"),
        "holdout_icir": hold_stats.get("icir"),
        "holdout_newey_west_t": hold_stats.get("newey_west_t"),
        "thresholds": s1_config.thresholds.to_dict(),
    }
    metrics = {
        "schema_version": "stage4.s1.v1",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "name": display_name,
        "expression": validated.raw,
        "canonical": validated.canonical,
        "expression_hash": validated.expression_hash,
        "fields": list(validated.fields),
        "operators": list(validated.operators),
        "max_window": validated.max_window,
        "complexity": validated.complexity,
        "data": {
            **data_config.to_dict(),
            "snapshot_id": data_config.snapshot_id,
            "n_membership_rows": int(membership_rows),
            "n_factor_rows": len(factor),
            "factor_coverage": coverage,
        },
        "preprocessing": {
            "winsorize": s1_config.winsorize,
            "standardize": s1_config.standardize,
            "neutralize": False,
        },
        "distribution_raw": raw_distribution,
        "distribution": distribution_stats(panel["factor"]),
        "calibration": {
            "start": s1_config.calibration_start,
            "end": s1_config.calibration_end,
            "n_dates": cal_stats.get("n_dates"),
            "ic": cal_stats,
            "pearson_ic": ic_stats(cal_pearson),
            "yearly": yearly["years"],
            "year_sign_consistency": year_consistency,
            "quantiles": quantile_results,
            "turnover": {"q": turnover_q, "top_quantile_one_way": turnover},
            "rank_autocorrelation": autocorr,
            "decay": decay,
        },
        "holdout": {
            "start": s1_config.holdout_start,
            "end": s1_config.holdout_end,
            "n_dates": hold_stats.get("n_dates"),
            "ic": hold_stats,
        },
        "scorecard": scorecard,
    }
    return S1Result(metrics=metrics, scorecard=scorecard, rank_ic=cal_ic)


def apply_fdr(results: list[S1Result], alpha: float = 0.10) -> None:
    """Add Benjamini-Hochberg q-values and FDR reasons in place."""
    if not results:
        return
    pvalues = [float(r.scorecard.get("p_value") or 1.0) for r in results]
    try:
        _, qvalues, _, _ = multipletests(pvalues, alpha=alpha, method="fdr_bh")
    except Exception:
        qvalues = np.asarray(pvalues, dtype=float)
    for result, p_value, q_value in zip(results, pvalues, qvalues, strict=True):
        q = float(q_value)
        fdr_pass = bool(q <= alpha)
        result.scorecard["p_value"] = p_value
        result.scorecard["q_value"] = q
        result.scorecard["fdr_pass"] = fdr_pass
        if not fdr_pass:
            reasons = list(result.scorecard["reasons"])
            if "FDR_FAIL" not in reasons:
                reasons.append("FDR_FAIL")
            result.scorecard["reasons"] = reasons
            result.scorecard["status"] = "fail"
        result.metrics["scorecard"] = result.scorecard


def scorecard_row(result: S1Result) -> dict[str, Any]:
    """Flatten one scorecard for CSV/Parquet/SQLite."""
    row = dict(result.scorecard)
    row["reasons"] = ";".join(row.get("reasons") or [])
    row["thresholds"] = _compact_json(row.get("thresholds"))
    return row


def _score_reasons(
    *,
    coverage: float | None,
    rank_ic: float | None,
    icir: float | None,
    nw_t: float | None,
    monotonicity: float | None,
    turnover: float | None,
    year_consistency: float | None,
    thresholds: S1Thresholds,
) -> list[str]:
    reasons: list[str] = []
    if coverage is None or coverage < thresholds.min_coverage:
        reasons.append("LOW_COVERAGE")
    if rank_ic is None or abs(rank_ic) < thresholds.min_abs_rank_ic:
        reasons.append("WEAK_IC")
    if icir is None or abs(icir) < thresholds.min_abs_icir:
        reasons.append("WEAK_ICIR")
    if nw_t is None or abs(nw_t) < thresholds.min_abs_nw_t:
        reasons.append("WEAK_T")
    if monotonicity is None or abs(monotonicity) < thresholds.min_abs_monotonicity:
        reasons.append("NON_MONOTONE")
    if turnover is None:
        reasons.append("TURNOVER_UNKNOWN")
    elif turnover > thresholds.max_turnover:
        reasons.append("HIGH_TURNOVER")
    if year_consistency is None:
        reasons.append("YEAR_SIGN_UNKNOWN")
    elif year_consistency < thresholds.min_year_sign_consistency:
        reasons.append("YEAR_SIGN_UNSTABLE")
    return reasons


def _status(reasons: list[str]) -> str:
    if not reasons:
        return "pass"
    if any(reason in HARD_REASONS for reason in reasons):
        return "fail"
    return "review"


def _yearly_ic(rank_ic: pd.Series) -> dict[str, Any]:
    if rank_ic.empty:
        return {"years": [], "sign_consistency": None}
    overall = rank_ic.mean()
    rows: list[dict[str, Any]] = []
    same_sign = 0
    valid_years = 0
    for year, series in rank_ic.groupby(rank_ic.index.year, sort=True):
        st = ic_stats(series)
        rows.append({"year": int(year), **st})
        mean = st.get("mean")
        if mean is not None and overall != 0:
            valid_years += 1
            if math.copysign(1.0, float(mean)) == math.copysign(1.0, float(overall)):
                same_sign += 1
    consistency = same_sign / valid_years if valid_years else None
    return {"years": rows, "sign_consistency": consistency}


def _p_value(stats: dict[str, Any]) -> float:
    t_value = stats.get("newey_west_t")
    if t_value is None:
        t_value = stats.get("t_stat")
    n_dates = stats.get("n_dates") or 0
    if t_value is None or n_dates < 3:
        return 1.0
    return float(2.0 * scipy_stats.t.sf(abs(float(t_value)), df=max(int(n_dates) - 1, 1)))


def _direction(mean: float | None) -> str:
    if mean is None:
        return "unknown"
    if mean > 0:
        return "positive"
    if mean < 0:
        return "negative"
    return "zero"


def _slice(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    if panel.empty:
        return panel
    dates = panel.index.get_level_values("datetime")
    mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return panel.loc[mask]


def _compact_json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
