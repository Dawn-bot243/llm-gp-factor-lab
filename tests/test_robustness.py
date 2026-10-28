from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.robustness import (
    S2Config,
    S2Gates,
    apply_fdr,
    deflated_sharpe_ratio,
    long_short_returns,
    placebo_test,
    walk_forward_folds,
    window_sensitivity,
)


def _panel(n_inst: int = 30, start: str = "2010-01-04", end: str = "2020-09-23") -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    index = pd.MultiIndex.from_product(
        [[f"S{i:02d}" for i in range(n_inst)], dates],
        names=["instrument", "datetime"],
    )
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {"factor": rng.normal(size=len(index)), "fwd_1": rng.normal(size=len(index))},
        index=index,
    )


def test_walk_forward_has_multiple_folds():
    panel = _panel()
    result = walk_forward_folds(panel, S2Config())
    assert result["n_folds"] >= 5
    assert result["positive_ratio"] is not None
    assert all(item["n_dates"] > 0 for item in result["folds"])


def test_placebo_detects_predictive_and_noise_factors():
    panel = _panel()
    noise = placebo_test(panel, S2Config(placebo_iterations=8))
    assert noise["p_value"] is not None
    strong = panel.copy()
    strong["factor"] = strong["fwd_1"] + np.random.default_rng(1).normal(scale=0.01, size=len(strong))
    signal = placebo_test(strong, S2Config(placebo_iterations=8))
    assert signal["p_value"] < noise["p_value"]


def test_window_sensitivity_reports_variants_and_cv():
    def evaluator(expression: str) -> float:
        return 0.02

    result = window_sensitivity("Mean($close, 20) / $close - 1", S2Config(), evaluator)
    assert len(result["variants"]) >= 2
    assert result["cv"] is not None
    assert result["sign_flip"] is False


def test_deflated_sharpe_and_fdr():
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0.001, 0.01, size=500))
    value = deflated_sharpe_ratio(returns, n_trials=2000)
    assert value is not None
    assert 0.0 <= value <= 1.0
    rows = [
        {"placebo": {"p_value": 0.01}, "direction": "positive", "walk_forward": {}, "yearly": [], "sensitivity": {}, "dsr": 1.0, "gates": S2Gates().to_dict()},
        {"placebo": {"p_value": 0.9}, "direction": "positive", "walk_forward": {}, "yearly": [], "sensitivity": {}, "dsr": 1.0, "gates": S2Gates().to_dict()},
    ]
    apply_fdr(rows, 0.1)
    assert rows[0]["fdr_pass"] is True
    assert rows[1]["fdr_pass"] is False


def test_long_short_returns_shape():
    panel = _panel(n_inst=40, start="2017-01-01", end="2017-12-31")
    series = long_short_returns(panel, S2Config())
    assert len(series) > 0
    assert np.isfinite(series).all()
