from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.factors.evaluator import EvaluationConfig
from src.factors.preprocess import (
    neutralize_cross_section,
    preprocess_factor,
    standardize_cross_section,
    winsorize_cross_section,
)
from src.factors.s1 import (
    ForwardReturns,
    S1Config,
    S1Result,
    S1Thresholds,
    apply_fdr,
    evaluate_s1,
)
from src.factors.serialization import jsonable
from src.factors.store import FactorStore


def _synthetic(n_instruments: int = 40) -> tuple[pd.Series, pd.DataFrame]:
    dates = pd.bdate_range("2010-01-04", "2020-09-23")
    instruments = [f"S{i:02d}" for i in range(n_instruments)]
    index = pd.MultiIndex.from_product(
        [instruments, dates], names=["instrument", "datetime"]
    )
    rng = np.random.default_rng(7)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index)))),
        index=index,
        name="close",
    )
    wide = close.unstack("instrument")
    fwd_1 = (wide.shift(-2) / wide.shift(-1) - 1.0).stack(future_stack=True)
    fwd_1.index = fwd_1.index.set_names(["datetime", "instrument"])
    fwd_1 = fwd_1.reorder_levels(["instrument", "datetime"]).sort_index()
    factor = (fwd_1 + rng.normal(0, 0.8, len(index))).rename("factor")
    return close, factor.to_frame()


def _config() -> EvaluationConfig:
    return EvaluationConfig(
        provider_uri="unused",
        membership_path="__none__",
        snapshot_file=None,
        cache_dir=None,
        use_cache=False,
        constraints_path=None,
    )


def _s1_config(**overrides) -> S1Config:
    payload = {
        "quantiles": (5, 10),
        "min_names": 20,
        "winsorize": 0.01,
        "standardize": True,
        "decay_horizons": (1, 2, 5),
        "calibration_start": "2010-01-01",
        "calibration_end": "2018-12-31",
        "holdout_start": "2019-01-01",
        "holdout_end": "2020-09-23",
        "thresholds": S1Thresholds(),
    }
    payload.update(overrides)
    return S1Config(**payload)


def test_forward_returns_match_label_definition():
    dates = pd.bdate_range("2020-01-01", periods=6)
    index = pd.MultiIndex.from_product([["A"], dates], names=["instrument", "datetime"])
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], index=index, name="close")
    fwd = ForwardReturns(close, (1, 2))
    values = fwd.get(1)
    # fwd_1 at t = close[t+2] / close[t+1] - 1.
    assert values.loc[("A", dates[0])] == pytest.approx(102.0 / 101.0 - 1.0)
    assert np.isnan(values.loc[("A", dates[-1])])
    assert len(fwd.get(2)) == len(values)


def test_winsorize_and_standardize_are_cross_sectional():
    dates = pd.bdate_range("2020-01-01", periods=2)
    index = pd.MultiIndex.from_product([["A", "B", "C", "D"], dates], names=["instrument", "datetime"])
    panel = pd.DataFrame(
        {"factor": [1.0, 2.0, 3.0, 100.0, 10.0, 11.0, 12.0, 13.0]},
        index=index,
    )
    clipped = winsorize_cross_section(panel, lower=0.25, upper=0.75)
    assert clipped.max() < 100.0
    standardized = standardize_cross_section(panel)
    for _, group in standardized.groupby(level="datetime"):
        assert group.mean() == pytest.approx(0.0, abs=1e-12)
        assert group.std() == pytest.approx(1.0, abs=1e-12)


def test_neutralize_removes_exposure():
    dates = pd.bdate_range("2020-01-01", periods=5)
    index = pd.MultiIndex.from_product([["A", "B", "C", "D"], dates], names=["instrument", "datetime"])
    exposure = pd.Series(np.tile([1.0, 2.0, 3.0, 4.0], len(dates)), index=index)
    panel = pd.DataFrame({"factor": exposure * 3.0 + 1.0}, index=index)
    residual = neutralize_cross_section(panel, exposure.to_frame("size"))
    assert residual.abs().max() < 1e-10


def test_preprocess_factor_returns_named_series():
    dates = pd.bdate_range("2020-01-01", periods=2)
    index = pd.MultiIndex.from_product([["A", "B", "C", "D"], dates], names=["instrument", "datetime"])
    panel = pd.DataFrame({"factor": [1.0, 2.0, 3.0, 100.0, 1.0, 2.0, 3.0, 4.0]}, index=index)
    result = preprocess_factor(panel, winsorize=0.01, standardize=True)
    assert result.name == "factor"
    assert result.notna().all()


def test_evaluate_s1_produces_scorecard_and_holdout():
    close, factor = _synthetic(n_instruments=30)
    config = _config()
    s1_config = _s1_config()
    forward = ForwardReturns(close, s1_config.decay_horizons)
    result = evaluate_s1(
        "$close",
        config,
        s1_config,
        forward,
        name="synthetic",
        validated=validate_for_test(),
        factor=factor,
    )
    assert result.scorecard["status"] in {"pass", "review", "fail"}
    assert result.metrics["calibration"]["n_dates"] > 0
    assert result.metrics["holdout"]["n_dates"] > 0
    assert len(result.metrics["calibration"]["decay"]) == 3
    assert result.metrics["calibration"]["quantiles"]["q5"]["q"] == 5
    assert result.scorecard["p_value"] is not None


def test_apply_fdr_updates_q_values_and_reasons():
    def make_result(name: str, p: float) -> S1Result:
        scorecard = {
            "name": name,
            "expression": "$close",
            "canonical": "$close",
            "expression_hash": name,
            "status": "pass",
            "reasons": [],
            "p_value": p,
        }
        return S1Result(metrics={"scorecard": scorecard}, scorecard=scorecard, rank_ic=pd.Series(dtype=float))

    results = [make_result("a", 0.001), make_result("b", 0.9)]
    apply_fdr(results, alpha=0.10)
    assert results[0].scorecard["q_value"] is not None
    assert results[0].scorecard["fdr_pass"] is True
    assert results[1].scorecard["fdr_pass"] is False
    assert "FDR_FAIL" in results[1].scorecard["reasons"]


def test_factor_store_writes_parquet_csv_sqlite(tmp_path):
    close, factor = _synthetic(n_instruments=20)
    s1_config = _s1_config()
    result = evaluate_s1(
        "$close",
        _config(),
        s1_config,
        ForwardReturns(close, s1_config.decay_horizons),
        name="synthetic",
        validated=validate_for_test(),
        factor=factor,
    )
    store = FactorStore(tmp_path / "store")
    paths = store.write([result], metadata={"source": "test"})
    assert (tmp_path / "store" / "s1_scorecard.parquet").exists()
    assert (tmp_path / "store" / "s1_scorecard.csv").exists()
    assert (tmp_path / "store" / "metadata.json").exists()
    with sqlite3.connect(paths["sqlite"]) as connection:
        rows = connection.execute("select count(*) from s1_scorecard").fetchone()
    assert rows[0] == 1


def test_jsonable_handles_numpy_and_nan():
    assert jsonable({"x": np.float64(1.5), "y": np.int64(2), "z": np.nan}) == {
        "x": 1.5,
        "y": 2,
        "z": None,
    }


def validate_for_test():
    from src.factors.expression import validate_expression

    return validate_expression("$close")


def test_s1_config_from_dict_keeps_default_winsorize():
    config = S1Config.from_dict({})
    assert config.winsorize == pytest.approx(0.01)
    assert config.decay_horizons == tuple(range(1, 21))
    explicit_none = S1Config.from_dict({"winsorize": None})
    assert explicit_none.winsorize is None
