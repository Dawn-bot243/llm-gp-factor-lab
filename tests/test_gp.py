from __future__ import annotations

import random

import pandas as pd
import pytest

from src.factors.expression import ExpressionError, validate_expression
from src.mining.gp import (
    GPNode,
    crossover,
    mutate,
    parse_seed_expression,
    random_series,
    validate_tree,
)
from src.mining.gp_eval import GPMetrics
from src.mining.gp_runner import GPConfig, GPRunner, write_gp_outputs


def test_random_trees_validate_within_limits():
    rng = random.Random(3)
    ok = 0
    for _ in range(40):
        node = random_series(rng, depth=4, max_nodes=30)
        if node.size > 30 or node.depth > 6:
            continue
        canonical, _ = validate_tree(node)
        assert validate_expression(canonical).canonical == canonical
        ok += 1
    assert ok > 10


def test_seed_parser_handles_rolling_if_and_correlation():
    expressions = [
        "$close / Ref($close, 20) - 1",
        "-1 * Std($close, 20) / $close",
        "If($close > $open, Mean($volume, 5), $open)",
        "Corr($close, $volume, 20)",
    ]
    for expression in expressions:
        node = parse_seed_expression(expression)
        canonical, _ = validate_tree(node)
        assert validate_expression(canonical).canonical == canonical


def test_mutation_and_crossover_are_valid_or_rejectable():
    rng = random.Random(11)
    left = random_series(rng, depth=3)
    right = random_series(rng, depth=3)
    for _ in range(20):
        child = mutate(left, rng)
        try:
            validate_tree(child)
        except ExpressionError:
            pass
    child = crossover(left, right, rng)
    try:
        validate_tree(child)
    except ExpressionError:
        pass


def test_gp_config_from_dict():
    config = GPConfig.from_dict({"population_size": 10, "budget": 50})
    assert config.population_size == 10
    assert config.budget == 50
    assert config.to_dict()["population_size"] == 10


class _FakeEvaluator:
    def __init__(self, max_evaluations: int = 50) -> None:
        self.max_evaluations = max_evaluations
        self.n_evaluations = 0
        self.cache: dict[str, GPMetrics] = {}

    @property
    def remaining(self) -> int:
        return self.max_evaluations - self.n_evaluations

    def evaluate(self, expression: str, validated=None) -> GPMetrics:
        if expression in self.cache:
            return self.cache[expression]
        self.n_evaluations += 1
        value = float(abs(hash(expression)) % 1000) / 1000.0
        metrics = GPMetrics(
            expression=expression,
            canonical=expression,
            expression_hash=str(hash(expression)),
            fitness=value,
            rank_ic=value / 10,
            icir=value,
            turnover=0.2,
            coverage=0.9,
            n_dates=400,
            complexity=5,
            monotonicity=0.8,
        )
        self.cache[expression] = metrics
        return metrics


def test_gp_runner_respects_budget_and_writes_outputs(tmp_path):
    config = GPConfig(population_size=10, generations=3, elite_size=2, budget=20, seed=5)
    runner = GPRunner(config, _FakeEvaluator(20), seed_expressions=[])
    result = runner.run()
    assert result.summary["n_trials"] <= 20
    assert result.hall_of_fame
    paths = write_gp_outputs(result, tmp_path, runner.trials)
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "hall_of_fame.csv").exists()
    assert (tmp_path / "generations.jsonl").exists()
    assert (tmp_path / "trials.jsonl").exists()
    assert pd.read_csv(paths["hall"]).shape[0] >= 1


def test_invalid_tree_rejected():
    node = GPNode(op="Log", args=(GPNode(op="const", const=2.0, kind="scalar"),), kind="series")
    with pytest.raises(ExpressionError):
        validate_tree(node)
