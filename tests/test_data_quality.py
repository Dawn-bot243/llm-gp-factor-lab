from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.constraints import (
    _round_half_up,
    build_constraints,
    limit_ratio_for,
    prepare_akshare_daily,
)
from src.data.mask import build_mask_from_panel
from src.data.pit import build_pit_membership, membership_summary
from src.data.pit_constraints import classify_st
from src.data.qlib_patch import _calendar_runs
from src.data.quality import (
    check_calendar,
    early_end_check,
    factor_consistency,
    price_jumps,
    select_sample,
    universe_counts,
)
from src.data.reconcile import _relative_match, _return_match
from src.data.tushare_source import to_qlib_code, to_ts_code


class FakeData:
    """Minimal stand-in for QlibData used by the pure check functions."""

    def __init__(self, calendar: pd.DatetimeIndex, fields: dict[str, dict[str, pd.Series]]):
        self.calendar = calendar
        self.fields = fields

    def feature(self, instrument: str, name: str = "close") -> pd.Series:
        return self.fields.get(instrument, {}).get(name, pd.Series(dtype="float64"))

    def ohlcv(self, instrument: str) -> pd.DataFrame:
        values = self.fields.get(instrument, {})
        if not values:
            return pd.DataFrame()
        return pd.DataFrame(values)


def _instruments(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["instrument", "start", "end"])
    frame["start"] = pd.to_datetime(frame["start"])
    frame["end"] = pd.to_datetime(frame["end"])
    return frame


def test_check_calendar_flags_duplicates_weekends_and_missing_official_days():
    calendar = pd.DatetimeIndex(
        ["2020-01-02", "2020-01-03", "2020-01-04", "2020-01-06", "2020-01-06"]
    )
    official = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    report = check_calendar(calendar, "2020-01-01", "2020-01-31", official)
    assert report["duplicates"] == 1
    assert report["n_weekend_days"] == 1
    assert report["missing_official_days"] == 1
    assert report["missing_official_dates"] == ["2020-01-07"]
    assert report["monotonic"] is True


def test_universe_counts_uses_interval_events():
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-01-01", "2020-01-10"))
    instruments = _instruments(
        [
            ("SH600000", "2020-01-01", "2020-01-03"),
            ("SH600001", "2020-01-06", "2020-01-10"),
        ]
    )
    report = universe_counts(instruments, calendar, "2020-01-01", "2020-01-10")
    assert report["min_count"] == 1
    assert report["max_count"] == 1
    assert report["n_instruments"] == 2
    assert report["duplicate_instruments"] == 0


def test_price_jumps_detects_hard_move_and_factor_rebase_separately():
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-09-21", "2020-09-30"))
    instruments = _instruments([("SH600000", "2020-09-21", "2020-09-30")])
    close = pd.Series([7.414077, 1.0], index=calendar[[4, 5]])
    factor = pd.Series([0.782901, 0.105708], index=calendar[[4, 5]])
    fake = FakeData(calendar, {"SH600000": {"close": close, "factor": factor}})
    summary, events = price_jumps(fake, instruments, "2020-09-21", "2020-09-30")
    assert summary["n_factor_rebase_days"] == 1
    assert summary["n_hard_jumps"] == 0
    assert set(events["kind"]) == {"factor_rebase"}
    # Raw price is continuous across the vendor reset.
    raw = (close / factor).to_numpy()
    assert raw[0] == pytest.approx(raw[1], abs=0.02)


def test_price_jumps_detects_hard_raw_move():
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-01-01", "2020-01-10"))
    instruments = _instruments([("SH600000", "2020-01-01", "2020-01-10")])
    close = pd.Series([10.0, 20.0], index=calendar[[0, 1]])
    factor = pd.Series([1.0, 1.0], index=calendar[[0, 1]])
    fake = FakeData(calendar, {"SH600000": {"close": close, "factor": factor}})
    summary, events = price_jumps(fake, instruments, "2020-01-01", "2020-01-10")
    assert summary["n_hard_jumps"] == 1
    assert events.iloc[0]["kind"] == "hard"


def test_factor_consistency_flags_rebase_with_raw_continuity():
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-09-21", "2020-09-30"))
    instruments = _instruments([("SH600000", "2020-09-21", "2020-09-30")])
    close = pd.Series([7.414077, 1.0], index=calendar[[4, 5]])
    factor = pd.Series([0.782901, 0.105708], index=calendar[[4, 5]])
    fake = FakeData(calendar, {"SH600000": {"close": close, "factor": factor}})
    summary, _ = factor_consistency(fake, instruments, "2020-09-21", "2020-09-30")
    assert summary["n_factor_jumps"] == 1
    date, details = next(iter(summary["top_dates"].items()))
    assert date == "2020-09-28"
    assert details["n_stocks"] == 1
    assert abs(details["median_raw_ret"]) < 0.01
    assert details["median_adj_ret"] < -0.5


@pytest.mark.parametrize(
    ("instrument", "date", "is_st", "expected"),
    [
        ("SH600000", "2020-01-02", 0, 0.10),
        ("SH600000", "2020-01-02", 1, 0.05),
        ("SZ300001", "2020-08-21", 0, 0.10),
        ("SZ300001", "2020-08-24", 0, 0.20),
        ("SZ300001", "2020-08-24", 1, 0.20),
        ("SH688001", "2020-01-02", 0, 0.20),
        ("SH688001", "2020-01-02", 1, 0.20),
    ],
)
def test_limit_ratio_rules(instrument, date, is_st, expected):
    ratio, _ = limit_ratio_for(instrument, pd.Timestamp(date), is_st)
    assert ratio == pytest.approx(expected)


def test_round_half_up_matches_exchange_convention():
    values = np.array([10.005, 10.004, 9.995])
    assert _round_half_up(values).tolist() == [10.01, 10.0, 10.0]


def test_build_constraints_uses_external_raw_prices_for_limits():
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-01-02", "2020-01-06"))
    qlib_close = pd.Series([5.0, 5.05, 5.10], index=calendar)
    qlib_factor = pd.Series([0.5, 0.5, 0.5], index=calendar)
    fake = FakeData(
        calendar,
        {
            "SH600000": {
                "open": qlib_close,
                "high": qlib_close,
                "low": qlib_close,
                "close": qlib_close,
                "volume": pd.Series([100.0, 100.0, 100.0], index=calendar),
                "factor": qlib_factor,
            }
        },
    )
    ext = pd.DataFrame(
        {
            "date": calendar,
            "open": [10.0, 10.0, 10.0],
            "high": [10.0, 11.0, 10.5],
            "low": [9.5, 11.0, 9.0],
            "close": [10.0, 11.0, 9.0],
            "preclose": [10.0, 10.0, 11.0],
            "volume": [1000, 1000, 1000],
            "tradestatus": [1, 1, 1],
            "isST": [0, 0, 0],
        }
    )
    universe = _instruments([("SH600000", "2020-01-02", "2020-01-06")])
    panel, summary = build_constraints(fake, universe, {"SH600000": ext}, "2020-01-02", "2020-01-06")
    assert summary["n_limit_up_days"] == 1
    assert summary["n_limit_down_days"] == 1
    first = panel.iloc[0]
    assert first["limit_up"] == pytest.approx(11.0)
    assert first["limit_down"] == pytest.approx(9.0)
    assert first["close"] == pytest.approx(10.0)
    assert first["qlib_raw_close"] == pytest.approx(10.0)


def test_prepare_akshare_daily_infers_suspension_and_st():
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-01-02", "2020-01-08"))
    qlib_close = pd.Series([1.0, 1.0, np.nan, 1.0, 1.0], index=calendar)
    fake = FakeData(calendar, {"SZ000001": {"close": qlib_close}})
    ak_frame = pd.DataFrame(
        {
            "date": calendar[[0, 1, 3, 4]],
            "open": [10.0] * 4,
            "high": [10.5] * 4,
            "low": [9.5] * 4,
            "close": [10.0, 10.1, 10.2, 10.3],
            "volume": [1000] * 4,
        }
    )
    st = pd.Series(0, index=calendar, dtype="int8")
    st.iloc[3:] = 1
    universe = _instruments([("SZ000001", "2020-01-02", "2020-01-08")])
    frames = prepare_akshare_daily(
        fake, universe, {"SZ000001": ak_frame}, {"SZ000001": st}, "2020-01-02", "2020-01-08"
    )
    frame = frames["SZ000001"].set_index("date")
    assert frame.loc[calendar[2], "tradestatus"] == 0
    assert frame.loc[calendar[3], "tradestatus"] == 1
    assert frame.loc[calendar[2], "isST"] == 0
    assert frame.loc[calendar[3], "isST"] == 1
    assert frame.loc[calendar[3], "preclose"] == pytest.approx(10.1)


def test_early_end_check_excludes_indices_and_flags_padded_nan():
    instruments = _instruments(
        [
            ("SH000905", "2010-01-01", "2022-06-09"),
            ("SH600000", "2010-01-01", "2020-09-25"),
            ("SH600001", "2010-01-01", "2020-09-25"),
        ]
    )
    basic = pd.DataFrame(
        {
            "instrument": ["SH600000", "SH600001"],
            "name": ["A", "B"],
            "ipo_date": ["2000-01-01", "2000-01-01"],
            "out_date": ["2020-09-20", "2015-01-01"],
        }
    )
    calendar = pd.DatetimeIndex(pd.bdate_range("2010-01-01", "2022-06-09"))
    close = pd.Series([np.nan] * len(calendar), index=calendar)
    fake = FakeData(calendar, {"SH600001": {"close": close}})
    summary, frame = early_end_check(instruments, "2022-12-30", basic, fake)
    assert summary["n_index_benchmarks"] == 1
    classifications = dict(zip(frame["instrument"], frame["classification"]))
    assert classifications["SH000905"] == "index_benchmark"
    assert classifications["SH600000"] == "delisting_matches"
    assert classifications["SH600001"] == "delisting_matches_padded_nan"


def test_return_match_uses_absolute_tolerance():
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-03", "2020-01-06"])
    qlib = pd.Series([100.0, 110.0, 121.0], index=index)
    ext = pd.Series([100.0, 110.0, 121.0], index=index)
    assert _return_match(qlib, ext, 0.005)["match_rate"] == 1.0

    ext_bad = pd.Series([100.0, 111.0, 122.1], index=index)
    result = _return_match(qlib, ext_bad, 0.005)
    assert result["match_rate"] == 0.5


def test_relative_match_tolerance():
    index = pd.DatetimeIndex(["2020-01-02", "2020-01-03"])
    left = pd.Series([100.0, 100.0], index=index)
    right = pd.Series([100.0, 102.0], index=index)
    assert _relative_match(left, right, 0.01)["match_rate"] == 0.5


def test_build_mask_uses_data_hole_ratios():
    panel = pd.DataFrame(
        {
            "instrument": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
            "date": pd.to_datetime(
                ["2020-01-02"] * 3 + ["2020-01-03"] * 3 + ["2020-01-06"] * 3
            ),
            "data_hole": [1, 1, 0, 0, 0, 0, 1, 0, 0],
        }
    )
    mask = build_mask_from_panel(panel, date_threshold=0.5, instrument_threshold=0.5)
    assert mask["excluded_dates"] == ["2020-01-02"]
    assert mask["excluded_instruments"] == ["A"]
    assert mask["date_ratios"]["2020-01-02"] == pytest.approx(2 / 3)
    assert mask["instrument_ratios"]["A"] == pytest.approx(2 / 3)


def test_build_pit_membership_from_intervals():
    calendar = pd.DatetimeIndex(pd.bdate_range("2020-01-01", "2020-01-10"))
    intervals = _instruments(
        [
            ("SH600000", "2020-01-01", "2020-01-03"),
            ("SH600000", "2020-01-06", "2020-01-10"),
            ("SH600001", "2020-01-01", "2020-01-10"),
        ]
    )
    membership = build_pit_membership(intervals, calendar, "2020-01-01", "2020-01-10")
    summary = membership_summary(membership)
    assert summary["n_dates"] == 8
    assert summary["n_instruments"] == 2
    assert summary["min_members"] == 2
    assert summary["max_members"] == 2
    assert len(membership) == 16


def test_classify_st_from_limit_ratio():
    assert classify_st(0.05) == (1, True)
    assert classify_st(0.10) == (0, True)
    assert classify_st(0.20) == (-1, False)
    assert classify_st(float("nan")) == (-1, False)


def test_calendar_runs_split_on_position_gaps():
    runs = _calendar_runs(np.array([2, 3, 4, 7, 8, 10]))
    assert [run.tolist() for run in runs] == [[2, 3, 4], [7, 8], [10]]
    assert _calendar_runs(np.array([], dtype=int)) == []


def test_tushare_code_conversion_roundtrip():
    assert to_ts_code("SH600000") == "600000.SH"
    assert to_ts_code("SZ000001") == "000001.SZ"
    assert to_ts_code("600000.SH") == "600000.SH"
    assert to_qlib_code("600000.SH") == "SH600000"
    assert to_qlib_code("000001.SZ") == "SZ000001"


def test_select_sample_is_deterministic():
    instruments = _instruments([(f"SH{600000 + i}", "2010-01-01", "2020-09-23") for i in range(20)])
    first = select_sample(instruments, 5, seed=7)
    second = select_sample(instruments, 5, seed=7)
    assert first == second
    assert len(first) == 5
