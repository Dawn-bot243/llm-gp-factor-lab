from __future__ import annotations

import json

from src.factors.io import alpha158_specs, read_expression_file, specs_from_expressions


def test_specs_from_expressions():
    specs = specs_from_expressions(["$close", "$open"])
    assert [spec.name for spec in specs] == ["expr_1", "expr_2"]


def test_read_text_tab_names(tmp_path):
    path = tmp_path / "expressions.txt"
    path.write_text("# comment\nma5\tMean($close, 5)/$close - 1\n$close\n", encoding="utf-8")
    specs = read_expression_file(path)
    assert [(spec.name, spec.expression) for spec in specs] == [
        ("ma5", "Mean($close, 5)/$close - 1"),
        ("expr_3", "$close"),
    ]


def test_read_csv_with_comma_inside_expression(tmp_path):
    path = tmp_path / "expressions.csv"
    path.write_text(
        'name,expression\nma5,"Mean($close, 5)/$close - 1"\n',
        encoding="utf-8",
    )
    specs = read_expression_file(path)
    assert specs[0].name == "ma5"
    assert specs[0].expression == "Mean($close, 5)/$close - 1"


def test_read_jsonl_and_json(tmp_path):
    jsonl = tmp_path / "expressions.jsonl"
    jsonl.write_text(
        json.dumps({"name": "ma5", "expression": "Mean($close, 5)/$close - 1"}) + "\n",
        encoding="utf-8",
    )
    assert read_expression_file(jsonl)[0].name == "ma5"

    payload = tmp_path / "expressions.json"
    payload.write_text(json.dumps([{"expression": "$close"}]), encoding="utf-8")
    specs = read_expression_file(payload)
    assert specs[0].name == "expr_1"
    assert specs[0].expression == "$close"


def test_alpha158_limit_is_evenly_spaced():
    specs = alpha158_specs(limit=3)
    assert len(specs) == 3
    assert specs[0].name == "KMID"
    assert specs[-1].name == "CNTN10"
