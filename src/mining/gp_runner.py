"""Stage 6 GP evolution loop with typed trees and a fixed fitness budget."""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.factors.expression import ExpressionError
from src.factors.serialization import json_default, jsonable
from src.mining.gp import (
    GPNode,
    crossover,
    mutate,
    parse_seed_expression,
    random_series,
    validate_tree,
)
from src.mining.gp_eval import GPBudgetExhausted, GPEvaluator, GPMetrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GPConfig:
    population_size: int = 100
    generations: int = 20
    elite_size: int = 5
    tournament_size: int = 3
    crossover_rate: float = 0.8
    mutation_rate: float = 0.2
    max_depth: int = 6
    max_nodes: int = 30
    budget: int = 2_000
    seed_count: int = 30
    inject_every: int = 5
    seed: int = 42
    max_child_retries: int = 20
    hall_max_corr: float = 0.80

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> GPConfig:
        payload = payload or {}
        return cls(
            population_size=int(payload.get("population_size", 100)),
            generations=int(payload.get("generations", 20)),
            elite_size=int(payload.get("elite_size", 5)),
            tournament_size=int(payload.get("tournament_size", 3)),
            crossover_rate=float(payload.get("crossover_rate", 0.8)),
            mutation_rate=float(payload.get("mutation_rate", 0.2)),
            max_depth=int(payload.get("max_depth", 6)),
            max_nodes=int(payload.get("max_nodes", 30)),
            budget=int(payload.get("budget", 2_000)),
            seed_count=int(payload.get("seed_count", 30)),
            inject_every=int(payload.get("inject_every", 5)),
            seed=int(payload.get("seed", 42)),
            max_child_retries=int(payload.get("max_child_retries", 20)),
            hall_max_corr=float(payload.get("hall_max_corr", 0.80)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "generations": self.generations,
            "elite_size": self.elite_size,
            "tournament_size": self.tournament_size,
            "crossover_rate": self.crossover_rate,
            "mutation_rate": self.mutation_rate,
            "max_depth": self.max_depth,
            "max_nodes": self.max_nodes,
            "budget": self.budget,
            "seed_count": self.seed_count,
            "inject_every": self.inject_every,
            "seed": self.seed,
            "max_child_retries": self.max_child_retries,
            "hall_max_corr": self.hall_max_corr,
        }


@dataclass
class GPIndividual:
    node: GPNode
    canonical: str
    metrics: GPMetrics
    origin: str
    generation: int

    @property
    def fitness(self) -> float:
        return float(self.metrics.fitness or float("-inf"))


@dataclass
class GPRunResult:
    hall_of_fame: list[dict[str, Any]]
    generations: list[dict[str, Any]]
    summary: dict[str, Any]


class GPRunner:
    def __init__(
        self,
        config: GPConfig,
        evaluator: GPEvaluator,
        seed_expressions: list[str] | None = None,
    ) -> None:
        self.config = config
        self.evaluator = evaluator
        self.rng = random.Random(config.seed)
        self.seed_expressions = list(seed_expressions or [])
        self.population: list[GPIndividual] = []
        self.seen: set[str] = set()
        self.trials: list[dict[str, Any]] = []
        self.generation_stats: list[dict[str, Any]] = []
        self.seed_index = 0

    def run(self) -> GPRunResult:
        started = time.time()
        self._initialize()
        for generation in range(1, self.config.generations + 1):
            if self.evaluator.remaining <= 0:
                break
            self._evolve_generation(generation)
            if self.config.inject_every and generation % self.config.inject_every == 0:
                self._inject_llm_seeds(generation)
            self._record_generation(generation)
        hall = self._hall_of_fame()
        summary = {
            "schema_version": "stage6.gp.v1",
            "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
            "config": self.config.to_dict(),
            "n_trials": self.evaluator.n_evaluations,
            "n_unique_evaluated": len(self.evaluator.cache),
            "n_generations": len(self.generation_stats),
            "n_seed_expressions": len(self.seed_expressions),
            "n_population": len(self.population),
            "runtime_seconds": round(time.time() - started, 1),
            "best": hall[0] if hall else None,
        }
        return GPRunResult(hall_of_fame=hall, generations=self.generation_stats, summary=summary)

    def _initialize(self) -> None:
        seeds = self._seed_nodes()
        for node in seeds:
            if self.evaluator.remaining <= 0:
                break
            individual = self._evaluate_node(node, origin="llm_seed", generation=0)
            if individual and individual.canonical not in self.seen:
                self.seen.add(individual.canonical)
                self.population.append(individual)
            self.seed_index += 1
        while len(self.population) < self.config.population_size and self.evaluator.remaining > 0:
            node = random_series(self.rng, depth=4, max_nodes=self.config.max_nodes)
            individual = self._evaluate_node(node, origin="random", generation=0)
            if individual and individual.canonical not in self.seen:
                self.seen.add(individual.canonical)
                self.population.append(individual)

    def _seed_nodes(self) -> list[GPNode]:
        nodes: list[GPNode] = []
        for expression in self.seed_expressions[: self.config.seed_count]:
            try:
                nodes.append(parse_seed_expression(expression))
            except (ExpressionError, ValueError, TypeError) as exc:
                logger.debug("skipping seed expression: %s", exc)
                continue
        return nodes

    def _evolve_generation(self, generation: int) -> None:
        if not self.population:
            return
        self.population.sort(key=lambda item: item.fitness, reverse=True)
        elites = self.population[: self.config.elite_size]
        children: list[GPIndividual] = []
        target = max(1, self.config.population_size - len(elites))
        attempts = 0
        while len(children) < target and self.evaluator.remaining > 0:
            attempts += 1
            if attempts > target * self.config.max_child_retries:
                break
            parent_a = self._tournament()
            parent_b = self._tournament()
            try:
                if self.rng.random() < self.config.crossover_rate:
                    node = crossover(
                        parent_a.node,
                        parent_b.node,
                        self.rng,
                        max_depth=self.config.max_depth,
                        max_nodes=self.config.max_nodes,
                    )
                    origin = "crossover"
                else:
                    node = parent_a.node
                    origin = "clone"
                if self.rng.random() < self.config.mutation_rate or origin == "clone":
                    node = mutate(
                        node,
                        self.rng,
                        max_depth=self.config.max_depth,
                        max_nodes=self.config.max_nodes,
                    )
                    origin = "mutation"
                canonical, _ = validate_tree(
                    node, max_depth=self.config.max_depth, max_nodes=self.config.max_nodes
                )
            except (ExpressionError, ValueError):
                continue
            if canonical in self.seen:
                continue
            individual = self._evaluate_node(node, origin=origin, generation=generation)
            if individual is None:
                continue
            self.seen.add(individual.canonical)
            children.append(individual)
        self.population = elites + children

    def _inject_llm_seeds(self, generation: int) -> None:
        if not self.seed_expressions:
            return
        added = 0
        while self.seed_index < len(self.seed_expressions) and added < 5:
            expression = self.seed_expressions[self.seed_index]
            self.seed_index += 1
            try:
                node = parse_seed_expression(expression)
                canonical, _ = validate_tree(node, self.config.max_depth, self.config.max_nodes)
            except (ExpressionError, ValueError, TypeError) as exc:
                logger.debug("skipping injected seed: %s", exc)
                continue
            if canonical in self.seen:
                continue
            individual = self._evaluate_node(node, origin="llm_injection", generation=generation)
            if individual is None:
                continue
            self.seen.add(individual.canonical)
            self.population.append(individual)
            added += 1

    def _tournament(self) -> GPIndividual:
        size = min(self.config.tournament_size, len(self.population))
        candidates = self.rng.sample(self.population, size)
        return max(candidates, key=lambda item: item.fitness)

    def _evaluate_node(self, node: GPNode, *, origin: str, generation: int) -> GPIndividual | None:
        try:
            canonical, _ = validate_tree(node, self.config.max_depth, self.config.max_nodes)
            metrics = self.evaluator.evaluate(canonical)
        except (ExpressionError, ValueError, GPBudgetExhausted) as exc:
            self._trial(origin, generation, node.to_expression(), None, None, str(exc))
            return None
        self._trial(origin, generation, node.to_expression(), canonical, metrics, metrics.error)
        if not metrics.valid:
            return None
        return GPIndividual(node=node, canonical=canonical, metrics=metrics, origin=origin, generation=generation)

    def _trial(
        self,
        origin: str,
        generation: int,
        expression: str,
        canonical: str | None,
        metrics: GPMetrics | None,
        error: str | None,
    ) -> None:
        self.trials.append(
            {
                "generation": generation,
                "origin": origin,
                "expression": expression,
                "canonical": canonical,
                "fitness": metrics.fitness if metrics else None,
                "rank_ic": metrics.rank_ic if metrics else None,
                "icir": metrics.icir if metrics else None,
                "turnover": metrics.turnover if metrics else None,
                "coverage": metrics.coverage if metrics else None,
                "error": error,
            }
        )

    def _record_generation(self, generation: int) -> None:
        fitnesses = [item.fitness for item in self.population]
        best = max(self.population, key=lambda item: item.fitness) if self.population else None
        self.generation_stats.append(
            {
                "generation": generation,
                "population": len(self.population),
                "best_fitness": best.fitness if best else None,
                "mean_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else None,
                "best_expression": best.canonical if best else None,
                "n_trials": self.evaluator.n_evaluations,
                "n_unique": len(self.evaluator.cache),
            }
        )

    def _hall_of_fame(self) -> list[dict[str, Any]]:
        by_canonical: dict[str, GPIndividual] = {}
        for individual in self.population:
            previous = by_canonical.get(individual.canonical)
            if previous is None or individual.fitness > previous.fitness:
                by_canonical[individual.canonical] = individual
        ranked = sorted(by_canonical.values(), key=lambda item: item.fitness, reverse=True)
        selected: list[GPIndividual] = []
        panels: list[pd.DataFrame] = []
        for item in ranked:
            if len(selected) >= 20:
                break
            try:
                panel = self.evaluator.factor_panel(item.canonical)
            except Exception:
                panel = None
            if panel is not None and panels and any(
                _panel_corr(panel, previous) > self.config.hall_max_corr for previous in panels
            ):
                continue
            selected.append(item)
            if panel is not None:
                panels.append(panel)
        return [
            {
                "rank": index,
                **item.metrics.to_dict(),
                "origin": item.origin,
                "generation": item.generation,
            }
            for index, item in enumerate(selected, start=1)
        ]


def _panel_corr(left: pd.DataFrame, right: pd.DataFrame) -> float:
    joined = left.join(right, how="inner", rsuffix="_right")
    if joined.empty:
        return 0.0
    cors = []
    for _, group in joined.groupby(level="datetime", sort=True):
        if len(group) < 20:
            continue
        value = group["factor"].corr(group["factor_right"], method="spearman")
        if pd.notna(value):
            cors.append(abs(float(value)))
    return sum(cors) / len(cors) if cors else 0.0


def write_gp_outputs(result: GPRunResult, output_dir: str | Path, trials: list[dict[str, Any]]) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(jsonable(result.summary), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    pd.DataFrame(result.hall_of_fame).to_csv(out / "hall_of_fame.csv", index=False)
    (out / "hall_of_fame.json").write_text(
        json.dumps(jsonable(result.hall_of_fame), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    pd.DataFrame(result.generations).to_json(
        out / "generations.jsonl", orient="records", lines=True, force_ascii=False
    )
    with (out / "trials.jsonl").open("w", encoding="utf-8") as handle:
        for row in trials:
            handle.write(json.dumps(jsonable(row), ensure_ascii=False, default=json_default))
            handle.write("\n")
    return {
        "summary": str(out / "summary.json"),
        "hall": str(out / "hall_of_fame.csv"),
        "generations": str(out / "generations.jsonl"),
        "trials": str(out / "trials.jsonl"),
    }
