import pandas as pd

from scripts.run_baseline import _normalize_series, ic_stats, quantile_stats


def _panel(values: dict[str, list[float]], dates: list[str]) -> pd.Series:
    tuples = []
    data = []
    for inst, series in values.items():
        for dt, val in zip(dates, series):
            tuples.append((dt, inst))
            data.append(val)
    idx = pd.MultiIndex.from_tuples(tuples, names=["datetime", "instrument"])
    return pd.Series(data, index=idx, name="x")


def test_normalize_series_reorders_levels():
    s = _panel({"A": [1.0, 2.0], "B": [3.0, 4.0]}, ["2020-01-01", "2020-01-02"])
    out = _normalize_series(s, "pred")
    assert list(out.index.names) == ["instrument", "datetime"]
    assert out.loc[("A", "2020-01-01")] == 1.0


def test_ic_stats_positive():
    ic = pd.Series([0.1, 0.2, 0.15, 0.05, 0.12])
    stats = ic_stats(ic)
    assert stats["n_dates"] == 5
    assert stats["mean"] > 0
    assert stats["positive_ratio"] == 1.0


def test_quantile_stats_monotone():
    dates = ["2020-01-01", "2020-01-02"]
    values = {f"S{i}": [float(i), float(i)] for i in range(10)}
    pred = _panel(values, dates)
    # Label increases with the factor value, so quantiles should be monotone.
    label = pd.Series(
        [float(i) for i in range(10)] * 2,
        index=pd.MultiIndex.from_tuples(
            [(d, f"S{i}") for d in dates for i in range(10)],
            names=["datetime", "instrument"],
        ),
    )
    df = pd.concat([pred.rename("pred"), label.rename("label")], axis=1)
    qmean, long_short, _qdf = quantile_stats(df, q=5)
    assert len(qmean) == 5
    assert qmean.iloc[-1] > qmean.iloc[0]
    assert long_short.mean() > 0
