from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.factors.evaluator import EvaluationConfig, evaluate_panel
from src.factors.expression import DegenerateFactorError, validate_expression
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


def _panel(n_dates: int = 5, n_instruments: int = 30) -> pd.DataFrame:
    rows = []
    for date in pd.bdate_range("2020-01-01", periods=n_dates):
        for i in range(n_instruments):
            factor = float(i)
            label = factor * 0.01 + (i % 3) * 0.001
            rows.append((f"S{i:02d}", date, factor, label))
    frame = pd.DataFrame(rows, columns=["instrument", "datetime", "factor", "fwd_1"])
    return frame.set_index(["instrument", "datetime"])


def test_compute_ic_and_stats():
    panel = _panel()
    rank_ic, ic = compute_ic(panel, "factor", "fwd_1")
    assert len(rank_ic) == 5
    assert rank_ic.mean() == pytest.approx(1.0)
    assert ic.mean() == pytest.approx(1.0, rel=1e-3)
    stats = ic_stats(rank_ic)
    assert stats["n_dates"] == 5
    assert stats["mean"] == pytest.approx(1.0)
    assert stats["positive_ratio"] == 1.0


def test_quantile_returns_are_monotone():
    panel = _panel()
    qmean, long_short, qdf = quantile_returns(panel, "factor", "fwd_1", q=5)
    assert len(qmean) == 5
    assert qmean.iloc[-1] > qmean.iloc[0]
    assert long_short.mean() > 0
    assert qdf.shape[0] == 5
    stats = quantile_stats(qmean, long_short)
    assert stats["monotonicity"] == pytest.approx(1.0)
    assert stats["long_short"]["mean"] > 0


def test_turnover_and_rank_autocorrelation():
    panel = _panel()
    assert top_quantile_turnover(panel, "factor", q=5) == pytest.approx(0.0)
    assert rank_autocorrelation(panel, "factor") == pytest.approx(1.0)


def test_distribution_and_coverage():
    panel = _panel()
    distribution = distribution_stats(panel["factor"])
    assert distribution["count"] == 150
    assert distribution["min"] == 0.0
    assert distribution["max"] == 29.0
    coverage = coverage_stats(panel["factor"])
    assert coverage["coverage"] == 1.0
    assert coverage["finite_rows"] == 150


def test_evaluate_panel_produces_metrics():
    config = EvaluationConfig(
        provider_uri="unused",
        membership_path="__none__",
        snapshot_file=None,
        cache_dir=None,
        use_cache=False,
        horizons=(1,),
        constraints_path=None,
    )
    validated = validate_expression("Mean($close, 5) / $close - 1")
    result = evaluate_panel(_panel(), validated, config, name="ma5")
    assert result.metrics["name"] == "ma5"
    assert result.metrics["data"]["n_panel_rows"] == 150
    assert result.metrics["ic"]["fwd_1"]["mean"] == pytest.approx(1.0)
    assert result.metrics["quantiles"]["q"] == 5
    assert result.metrics["turnover"]["top_quantile_one_way"] == pytest.approx(0.0)


def test_constant_factor_is_rejected():
    config = EvaluationConfig(
        provider_uri="unused",
        membership_path="__none__",
        snapshot_file=None,
        cache_dir=None,
        use_cache=False,
        horizons=(1,),
        constraints_path=None,
    )
    panel = _panel()
    panel["factor"] = 1.0
    with pytest.raises(DegenerateFactorError):
        evaluate_panel(panel, validate_expression("$close"), config, name="constant")


def test_ic_stats_handles_constant_series():
    stats = ic_stats(pd.Series([0.1, 0.1, 0.1]))
    assert stats["n_dates"] == 3
    assert stats["icir"] is None
    assert stats["newey_west_t"] is None


def test_missing_values_are_ignored():
    panel = _panel()
    first_date = panel.index.get_level_values("datetime")[0]
    panel.loc[panel.index.get_level_values("datetime") == first_date, "factor"] = np.nan
    rank_ic, _ = compute_ic(panel, "factor", "fwd_1")
    assert len(rank_ic) == 4
    coverage = coverage_stats(panel["factor"])
    assert coverage["coverage"] == pytest.approx(120 / 150)
