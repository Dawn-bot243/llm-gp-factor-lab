"""Stage 8 S2 robustness: walk-forward, subsamples, placebo and sensitivity."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.factors.evaluator import EvaluationConfig, load_factor
from src.factors.metrics import compute_ic, ic_stats
from src.factors.s1 import ForwardReturns
from src.mining.gp import GPNode, parse_seed_expression

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class S2Gates:
    min_folds: int = 4
    min_positive_fold_ratio: float = 0.55
    min_mean_signed_ic: float = 0.010
    min_sign_consistency: float = 0.55
    max_placebo_p: float = 0.10
    max_fdr_q: float = 0.10
    min_dsr: float = 0.0
    min_yearly_positive_ratio: float = 0.50
    max_sensitivity_cv: float = 0.80

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> S2Gates:
        payload = payload or {}
        return cls(
            min_folds=int(payload.get("min_folds", 4)),
            min_positive_fold_ratio=float(payload.get("min_positive_fold_ratio", 0.55)),
            min_mean_signed_ic=float(payload.get("min_mean_signed_ic", 0.010)),
            min_sign_consistency=float(payload.get("min_sign_consistency", 0.55)),
            max_placebo_p=float(payload.get("max_placebo_p", 0.10)),
            max_fdr_q=float(payload.get("max_fdr_q", 0.10)),
            min_dsr=float(payload.get("min_dsr", 0.0)),
            min_yearly_positive_ratio=float(payload.get("min_yearly_positive_ratio", 0.50)),
            max_sensitivity_cv=float(payload.get("max_sensitivity_cv", 0.80)),
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "min_folds": self.min_folds,
            "min_positive_fold_ratio": self.min_positive_fold_ratio,
            "min_mean_signed_ic": self.min_mean_signed_ic,
            "min_sign_consistency": self.min_sign_consistency,
            "max_placebo_p": self.max_placebo_p,
            "max_fdr_q": self.max_fdr_q,
            "min_dsr": self.min_dsr,
            "min_yearly_positive_ratio": self.min_yearly_positive_ratio,
            "max_sensitivity_cv": self.max_sensitivity_cv,
        }


@dataclass(frozen=True)
class S2Config:
    train_years: int = 3
    valid_years: int = 1
    test_years: int = 1
    embargo_days: int = 2
    min_names: int = 20
    placebo_iterations: int = 100
    sensitivity_scales: tuple[float, ...] = (0.5, 1.5)
    n_trials: int = 2000
    quantile: int = 5
    gates: S2Gates = field(default_factory=S2Gates)

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> S2Config:
        payload = payload or {}
        return cls(
            train_years=int(payload.get("train_years", 3)),
            valid_years=int(payload.get("valid_years", 1)),
            test_years=int(payload.get("test_years", 1)),
            embargo_days=int(payload.get("embargo_days", 2)),
            min_names=int(payload.get("min_names", 20)),
            placebo_iterations=int(payload.get("placebo_iterations", 100)),
            sensitivity_scales=tuple(
                float(value) for value in payload.get("sensitivity_scales", (0.5, 1.5))
            ),
            n_trials=int(payload.get("n_trials", 2000)),
            quantile=int(payload.get("quantile", 5)),
            gates=S2Gates.from_dict(payload.get("gates")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_years": self.train_years,
            "valid_years": self.valid_years,
            "test_years": self.test_years,
            "embargo_days": self.embargo_days,
            "min_names": self.min_names,
            "placebo_iterations": self.placebo_iterations,
            "sensitivity_scales": list(self.sensitivity_scales),
            "n_trials": self.n_trials,
            "quantile": self.quantile,
            "gates": self.gates.to_dict(),
        }


class RobustnessEvaluator:
    def __init__(self, data_config: EvaluationConfig, config: S2Config) -> None:
        self.data_config = data_config
        self.config = config
        self.forward = ForwardReturns.from_config(data_config, (1,))
        self.amount = _load_amount(data_config)
        self.benchmark = _load_benchmark(data_config)

    def evaluate(self, candidate: Any) -> dict[str, Any]:
        _, factor = load_factor(candidate.expression, self.data_config)
        panel = factor.join(self.forward.get(1), how="left").sort_index()
        panel = panel.dropna(subset=["factor", "fwd_1"])
        if panel.empty:
            raise ValueError("empty factor/label panel")
        full_ic, _ = compute_ic(panel, "factor", "fwd_1", min_names=self.config.min_names)
        full_stats = ic_stats(full_ic)
        folds = walk_forward_folds(panel, self.config)
        yearly = yearly_ic(panel, self.config)
        regime = regime_ic(yearly, self.benchmark) if self.benchmark is not None else {}
        liquidity = liquidity_ic(panel, self.amount, self.config) if self.amount is not None else {}
        placebo = placebo_test(panel, self.config)
        sensitivity = window_sensitivity(
            candidate.expression, self.config, self._evaluate_variant
        )
        long_short = long_short_returns(panel, self.config)
        direction = 1.0 if (full_stats.get("mean") or 0.0) >= 0 else -1.0
        dsr = deflated_sharpe_ratio(direction * long_short, self.config.n_trials)
        row = {
            "name": candidate.name,
            "canonical": candidate.canonical,
            "source": candidate.source,
            "expression": candidate.expression,
            "full_rank_ic": full_stats.get("mean"),
            "full_icir": full_stats.get("icir"),
            "full_n_dates": full_stats.get("n_dates"),
            "direction": "positive" if direction > 0 else "negative",
            "walk_forward": folds,
            "yearly": yearly,
            "regime": regime,
            "liquidity": liquidity,
            "placebo": placebo,
            "sensitivity": sensitivity,
            "dsr": dsr,
            "gates": self.config.gates.to_dict(),
        }
        row["s2"] = _score(row, self.config.gates)
        return row

    def _evaluate_variant(self, expression: str) -> float | None:
        try:
            _, factor = load_factor(expression, self.data_config)
            panel = factor.join(self.forward.get(1), how="left").dropna(subset=["factor", "fwd_1"])
            ic, _ = compute_ic(panel, "factor", "fwd_1", min_names=self.config.min_names)
            return float(ic.mean()) if len(ic) else None
        except Exception:
            return None


def _load_amount(config: EvaluationConfig) -> pd.DataFrame | None:
    try:
        from src.factors.evaluator import load_feature_panel

        return load_feature_panel("$amount", "amount", config)
    except Exception:
        return None


def _load_benchmark(config: EvaluationConfig) -> pd.Series | None:
    try:
        from qlib.data import D

        bench = D.features(
            ["SH000300"],
            ["$close"],
            start_time=config.start,
            end_time=config.end,
            freq="day",
        )
        if bench is None or bench.empty:
            return None
        series = bench.iloc[:, 0]
        if isinstance(series.index, pd.MultiIndex):
            series = series.droplevel("instrument")
        return series.sort_index()
    except Exception:
        return None


def _slice(panel: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = panel.index.get_level_values("datetime")
    return panel[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))]


def walk_forward_folds(panel: pd.DataFrame, config: S2Config) -> dict[str, Any]:
    dates = pd.DatetimeIndex(sorted(panel.index.get_level_values("datetime").unique()))
    if dates.empty:
        return {"n_folds": 0, "positive_ratio": None, "mean_signed_ic": None, "sign_consistency": None, "folds": []}
    folds: list[dict[str, Any]] = []
    for test_year in range(int(dates.min().year), int(dates.max().year) + 1):
        train_start = test_year - config.train_years - config.valid_years
        train_end = test_year - config.valid_years - 1
        valid_year = test_year - config.valid_years
        if train_start < int(dates.min().year):
            continue
        train = _slice(panel, f"{train_start}-01-01", f"{train_end}-12-31")
        valid = _slice(panel, f"{valid_year}-01-01", f"{valid_year}-12-31")
        test = _slice(panel, f"{test_year}-01-01", f"{test_year}-12-31")
        valid = _drop_first_days(valid, config.embargo_days)
        test = _drop_first_days(test, config.embargo_days)
        if train.empty or valid.empty or test.empty:
            continue
        train_ic, _ = compute_ic(train, "factor", "fwd_1", min_names=config.min_names)
        valid_ic, _ = compute_ic(valid, "factor", "fwd_1", min_names=config.min_names)
        test_ic, _ = compute_ic(test, "factor", "fwd_1", min_names=config.min_names)
        if train_ic.empty or test_ic.empty:
            continue
        sign = float(np.sign(pd.concat([train_ic, valid_ic]).mean()))
        if sign == 0:
            sign = 1.0
        raw_test = float(test_ic.mean())
        folds.append(
            {
                "test_year": int(test_year),
                "train_start": str(train_start),
                "train_end": str(train_end),
                "valid_year": int(valid_year),
                "sign": sign,
                "raw_test_mean": raw_test,
                "test_mean": sign * raw_test,
                "test_icir": _icir(test_ic),
                "n_dates": len(test_ic),
                "positive": bool(sign * raw_test > 0),
            }
        )
    if not folds:
        return {"n_folds": 0, "positive_ratio": None, "mean_signed_ic": None, "sign_consistency": None, "folds": []}
    means = np.array([item["test_mean"] for item in folds])
    overall_sign = float(np.sign(means.mean())) or 1.0
    return {
        "n_folds": len(folds),
        "positive_ratio": float(np.mean([item["positive"] for item in folds])),
        "mean_signed_ic": float(means.mean()),
        "sign_consistency": float(np.mean(np.sign(means) == overall_sign)),
        "folds": folds,
    }


def _drop_first_days(panel: pd.DataFrame, days: int) -> pd.DataFrame:
    if panel.empty or days <= 0:
        return panel
    dates = pd.DatetimeIndex(sorted(panel.index.get_level_values("datetime").unique()))
    keep = dates[days:]
    return panel[panel.index.get_level_values("datetime").isin(keep)]


def _icir(ic: pd.Series) -> float | None:
    if len(ic) < 2:
        return None
    std = float(ic.std(ddof=1))
    return float(ic.mean() / std) if std > 0 else None


def yearly_ic(panel: pd.DataFrame, config: S2Config) -> list[dict[str, Any]]:
    ic, _ = compute_ic(panel, "factor", "fwd_1", min_names=config.min_names)
    return [
        {"year": int(year), "mean_ic": float(series.mean()), "n_dates": len(series)}
        for year, series in ic.groupby(ic.index.year, sort=True)
    ]


def regime_ic(yearly: list[dict[str, Any]], benchmark: pd.Series | None) -> dict[str, Any]:
    if benchmark is None or not yearly:
        return {}
    annual = benchmark.groupby(benchmark.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1)
    by_year = {int(item["year"]): float(item["mean_ic"]) for item in yearly}
    buckets: dict[str, list[float]] = {"bull": [], "bear": [], "range": []}
    for year, ret in annual.items():
        if int(year) not in by_year:
            continue
        regime = "bull" if ret > 0.10 else "bear" if ret < -0.10 else "range"
        buckets[regime].append(by_year[int(year)])
    return {key: (float(np.mean(values)) if values else None, len(values)) for key, values in buckets.items()}


def liquidity_ic(panel: pd.DataFrame, amount: pd.DataFrame, config: S2Config) -> dict[str, Any]:
    merged = panel.join(amount, how="inner")
    buckets: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
    for _, group in merged.groupby(level="datetime", sort=True):
        if len(group) < config.min_names * 3:
            continue
        labels = pd.qcut(group["amount"].rank(method="first"), 3, labels=False)
        for bucket, name in enumerate(("low", "mid", "high")):
            sub = group[labels == bucket]
            if len(sub) < config.min_names:
                continue
            value = sub["factor"].corr(sub["fwd_1"], method="spearman")
            if pd.notna(value):
                buckets[name].append(float(value))
    return {key: float(np.mean(values)) if values else None for key, values in buckets.items()}


def placebo_test(panel: pd.DataFrame, config: S2Config) -> dict[str, Any]:
    factor = panel["factor"].unstack("instrument")
    label = panel["fwd_1"].unstack("instrument")
    common = factor.columns.intersection(label.columns)
    factor = factor[common]
    label = label[common]
    observed = _mean_daily_rank_ic(factor, label, config.min_names)
    if observed is None:
        return {"observed": None, "p_value": None, "null_mean": None, "iterations": 0}
    rng = np.random.default_rng(20261027)
    null = []
    for _ in range(config.placebo_iterations):
        values = np.vstack(
            [rng.permutation(row) for row in factor.to_numpy(dtype=float)]
        )
        shuffled = pd.DataFrame(values, index=factor.index, columns=factor.columns)
        value = _mean_daily_rank_ic(shuffled, label, config.min_names)
        if value is not None:
            null.append(value)
    if not null:
        return {"observed": observed, "p_value": None, "null_mean": None, "iterations": 0}
    p_value = (1.0 + sum(abs(value) >= abs(observed) for value in null)) / (len(null) + 1.0)
    return {
        "observed": observed,
        "null_mean": float(np.mean(null)),
        "null_std": float(np.std(null, ddof=1)),
        "p_value": float(p_value),
        "iterations": len(null),
    }


def _mean_daily_rank_ic(factor: pd.DataFrame, label: pd.DataFrame, min_names: int) -> float | None:
    values = []
    for date in factor.index:
        x = factor.loc[date].dropna()
        y = label.loc[date].dropna()
        common = x.index.intersection(y.index)
        if len(common) < min_names:
            continue
        value = x.loc[common].corr(y.loc[common], method="spearman")
        if pd.notna(value):
            values.append(float(value))
    return float(np.mean(values)) if values else None


def window_sensitivity(
    expression: str,
    config: S2Config,
    evaluator,
) -> dict[str, Any]:
    try:
        node = parse_seed_expression(expression)
    except Exception:
        return {"variants": [], "cv": None, "sign_flip": None}
    base_ic = evaluator(expression)
    if base_ic is None:
        return {"variants": [], "cv": None, "sign_flip": None}
    variants = [{"scale": 1.0, "expression": expression, "ic": base_ic}]
    for scale in config.sensitivity_scales:
        try:
            scaled = _scale_windows(node, scale)
            canonical = scaled.canonical()
            ic = evaluator(canonical)
            if ic is not None:
                variants.append({"scale": scale, "expression": canonical, "ic": ic})
        except Exception as exc:
            logger.debug("sensitivity variant failed: %s", exc)
            continue
    ics = [item["ic"] for item in variants]
    sign_flip = any(np.sign(value) != np.sign(base_ic) for value in ics if value != 0)
    mean_abs = float(np.mean([abs(value) for value in ics]))
    cv = (
        float(np.std([abs(value) for value in ics], ddof=1) / mean_abs)
        if len(ics) > 1 and mean_abs > 0
        else None
    )
    return {"variants": variants, "cv": cv, "sign_flip": bool(sign_flip)}


def _scale_windows(node: GPNode, scale: float) -> GPNode:
    from dataclasses import replace

    window = node.window
    if window is not None:
        window = max(2, min(120, round(window * scale)))
    args = tuple(_scale_windows(arg, scale) for arg in node.args)
    return replace(node, args=args, window=window)


def long_short_returns(panel: pd.DataFrame, config: S2Config) -> pd.Series:
    rows = []
    for _, group in panel.groupby(level="datetime", sort=True):
        if len(group) < config.min_names * 2:
            continue
        ranks = group["factor"].rank(method="first")
        buckets = pd.qcut(ranks, config.quantile, labels=False)
        means = group["fwd_1"].groupby(buckets).mean()
        rows.append(float(means.iloc[-1] - means.iloc[0]))
    return pd.Series(rows, dtype=float)


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int) -> float | None:
    returns = returns.dropna()
    n = len(returns)
    if n < 10:
        return None
    std = float(returns.std(ddof=1))
    if std <= 1e-12:
        return None
    sr = float(returns.mean() / std)
    skew = float(returns.skew())
    kurt = float(returns.kurt() + 3.0)
    variance = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n - 1)
    if variance <= 0:
        return None
    euler = 0.5772156649
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    expected_max = math.sqrt(variance) * ((1.0 - euler) * z1 + euler * z2)
    return float(stats.norm.cdf((sr - expected_max) / math.sqrt(variance)))


def apply_fdr(rows: list[dict[str, Any]], alpha: float = 0.10) -> None:
    p_values = [float((row.get("placebo") or {}).get("p_value") or 1.0) for row in rows]
    if not p_values:
        return
    _, q_values, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    for row, q in zip(rows, q_values, strict=True):
        row["fdr_q"] = float(q)
        row["fdr_pass"] = bool(q <= alpha)
        row["s2"] = _score(row, S2Gates.from_dict(row.get("gates")))


def _score(row: dict[str, Any], gates: S2Gates) -> dict[str, Any]:
    wf = row.get("walk_forward") or {}
    placebo = row.get("placebo") or {}
    yearly = row.get("yearly") or []
    sensitivity = row.get("sensitivity") or {}
    direction = 1.0 if row.get("direction") == "positive" else -1.0
    yearly_positive = (
        sum(1 for item in yearly if direction * (item.get("mean_ic") or 0) > 0) / len(yearly)
        if yearly
        else None
    )
    reasons = []
    if (wf.get("n_folds") or 0) < gates.min_folds:
        reasons.append("FEW_FOLDS")
    if (wf.get("positive_ratio") or 0) < gates.min_positive_fold_ratio:
        reasons.append("FOLD_SIGN_WEAK")
    if wf.get("mean_signed_ic") is None or abs(float(wf["mean_signed_ic"])) < gates.min_mean_signed_ic:
        reasons.append("WEAK_WF_IC")
    if (wf.get("sign_consistency") or 0) < gates.min_sign_consistency:
        reasons.append("LOW_SIGN_CONSISTENCY")
    p_value = placebo.get("p_value")
    if p_value is None or p_value > gates.max_placebo_p:
        reasons.append("PLACEBO_FAIL")
    fdr_q = row.get("fdr_q")
    if fdr_q is None or fdr_q > gates.max_fdr_q:
        reasons.append("FDR_FAIL")
    dsr = row.get("dsr")
    if dsr is None or dsr < gates.min_dsr:
        reasons.append("DSR_FAIL")
    if yearly_positive is None or yearly_positive < gates.min_yearly_positive_ratio:
        reasons.append("YEARLY_WEAK")
    cv = sensitivity.get("cv")
    if cv is None or cv > gates.max_sensitivity_cv:
        reasons.append("SENSITIVITY_WEAK")
    if sensitivity.get("sign_flip"):
        reasons.append("SENSITIVITY_SIGN_FLIP")
    hard = {"FEW_FOLDS", "WEAK_WF_IC", "PLACEBO_FAIL", "FDR_FAIL", "DSR_FAIL"}
    status = "pass" if not reasons else ("fail" if any(reason in hard for reason in reasons) else "review")
    return {
        "status": status,
        "reasons": reasons,
        "yearly_positive_ratio": yearly_positive,
    }
