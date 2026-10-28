from __future__ import annotations

import json

from src.evaluation.ablation import (
    load_trial_candidates,
    protocol_hash,
    s1_summary,
    select_s3_factors,
)


def _write_trials(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_trial_loader_dedupes_and_keeps_best(tmp_path):
    path = tmp_path / "trials.jsonl"
    _write_trials(
        path,
        [
            {"canonical": "Mean($close, 5)", "expression": "Mean($close, 5)", "fitness": 0.01, "rank_ic": 0.01},
            {"canonical": "Mean($close, 5)", "expression": "Mean($close, 5)", "fitness": 0.03, "rank_ic": 0.03},
            {"canonical": "Mean($close, 10)", "expression": "Mean($close, 10)", "fitness": 0.02, "rank_ic": 0.02},
            {"canonical": "Bad($close)", "expression": "Bad($close)", "fitness": None, "error": "invalid"},
        ],
    )
    selected, stats, raw = load_trial_candidates(path, "A", top_n=2)
    assert stats.n_returned == 4
    assert stats.n_unique == 3
    assert stats.n_valid == 2
    assert stats.n_error == 1
    assert [item.fitness for item in selected] == [0.03, 0.02]
    assert len(raw) == 4


def test_s1_summary_and_retention():
    rows = [
        {"status": "pass", "rank_ic_mean": 0.03, "holdout_rank_ic": 0.015},
        {"status": "review", "rank_ic_mean": 0.02, "holdout_rank_ic": 0.01},
        {"status": "fail", "rank_ic_mean": 0.001, "holdout_rank_ic": 0.001},
    ]
    summary = s1_summary(rows)
    assert summary["n_s1"] == 3
    assert summary["n_pass"] == 1
    assert summary["n_review"] == 1
    assert summary["n_fail"] == 1
    assert summary["s1_pass_rate"] == 1 / 3
    assert summary["oos_ic_retention"] == 2 / 3


def test_s3_factor_selection_and_fallback():
    rows = [
        {"status": "pass", "canonical": "a", "rank_ic_mean": 0.03},
        {"status": "review", "canonical": "b", "rank_ic_mean": 0.02},
        {"status": "fail", "canonical": "c", "rank_ic_mean": 0.001},
    ]
    factors, fallback = select_s3_factors(rows, min_factors=2)
    assert not fallback
    assert [item["canonical"] for item in factors] == ["a", "b"]
    only_fail = [row for row in rows if row["status"] == "fail"]
    factors, fallback = select_s3_factors(only_fail, min_factors=2, fallback_top=1)
    assert fallback
    assert factors[0]["canonical"] == "c"


def test_protocol_hash_is_stable():
    payload = {"b": 2, "a": 1}
    assert protocol_hash(payload) == protocol_hash({"a": 1, "b": 2})
    assert protocol_hash(payload) != protocol_hash({"a": 2})
