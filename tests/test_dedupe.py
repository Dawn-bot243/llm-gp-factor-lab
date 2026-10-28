from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.dedupe import (
    Candidate,
    correlation_clusters,
    rank_matrix,
    representative_table,
    semantic_similarity,
    semantic_vs_correlation_pairs,
)
from src.evaluation.orthogonalize import orthogonalize_factor


def _panel(values: np.ndarray, n_dates: int | None = None) -> pd.DataFrame:
    n_inst = values.shape[0]
    n_dates = values.shape[1] if n_dates is None else n_dates
    dates = pd.bdate_range("2017-01-01", periods=n_dates)
    index = pd.MultiIndex.from_product(
        [[f"S{i:02d}" for i in range(n_inst)], dates],
        names=["instrument", "datetime"],
    )
    # values shape (n_inst, n_dates) -> flatten instrument-major
    return pd.DataFrame({"factor": values.reshape(-1)}, index=index)


def test_correlation_clusters_group_duplicates_and_inverses():
    rng = np.random.default_rng(3)
    n_inst, n_dates = 30, 20
    a = rng.normal(size=(n_inst, n_dates))
    panels = {
        "A": _panel(a),
        "A_copy": _panel(a),
        "A_neg": _panel(-a),
        "B": _panel(rng.normal(size=(n_inst, n_dates))),
    }
    corr = rank_matrix(panels)
    labels, _ = correlation_clusters(corr, threshold=0.3)
    assert labels[list(corr.columns).index("A")] == labels[list(corr.columns).index("A_copy")]
    candidates = [
        Candidate(name=name, expression=name, canonical=name, source="test", score=score)
        for name, score in zip(corr.columns, [0.01, 0.02, 0.03, 0.04], strict=True)
    ]
    reps = representative_table(candidates, labels, corr)
    assert len(reps) >= 2
    assert reps["representative"].is_unique


def test_semantic_vs_correlation_shows_they_are_different():
    expressions = [
        "Mean($close, 20) / $close - 1",
        "Mean($close, 10) / $close - 1",
        "Corr($volume, $close, 20)",
    ]
    sem = semantic_similarity(expressions)
    assert sem.shape == (3, 3)
    assert sem[0, 1] > sem[0, 2]
    corr = pd.DataFrame(
        [[1.0, 0.0, 0.9], [0.0, 1.0, 0.0], [0.9, 0.0, 1.0]],
        index=expressions,
        columns=expressions,
    )
    pairs = semantic_vs_correlation_pairs(sem, corr, expressions)
    assert isinstance(pairs, pd.DataFrame)


def test_orthogonalize_removes_exposure():
    rng = np.random.default_rng(5)
    n_inst, n_dates = 40, 30
    dates = pd.bdate_range("2017-01-01", periods=n_dates)
    index = pd.MultiIndex.from_product(
        [[f"S{i:02d}" for i in range(n_inst)], dates],
        names=["instrument", "datetime"],
    )
    exposure = pd.Series(rng.normal(size=len(index)), index=index, name="exposure")
    factor = (2.0 * exposure + rng.normal(scale=0.1, size=len(index))).rename("factor")
    panel = factor.to_frame()
    exposures = exposure.to_frame()
    residual = orthogonalize_factor(panel, exposures, ridge_alpha=1e-6)
    merged = pd.concat([exposure, residual], axis=1).dropna()
    corr = merged.groupby(level="datetime").apply(
        lambda group: group["exposure"].corr(group["factor"])
    )
    assert abs(float(corr.mean())) < 0.1


def test_rank_matrix_requires_aligned_panels():
    rng = np.random.default_rng(1)
    panels = {"A": _panel(rng.normal(size=(5, 4))), "B": _panel(rng.normal(size=(5, 4)))}
    corr = rank_matrix(panels)
    assert corr.shape == (2, 2)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
