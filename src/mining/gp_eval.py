"""Fast Stage 6 fitness evaluator on a fixed 100-stock, two-year sample."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.factors.expression import ValidatedExpression, validate_expression
from src.factors.metrics import (
    compute_ic,
    coverage_stats,
    top_quantile_turnover,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMBERSHIP = PROJECT_ROOT / "data" / "processed" / "pit_csi300_membership.parquet"
_QLIB_PROVIDER: str | None = None


class GPBudgetExhausted(RuntimeError):
    """Raised when the Stage 6 evaluation budget is used up."""


@dataclass(frozen=True)
class GPEvalConfig:
    provider_uri: str
    membership_path: str = str(DEFAULT_MEMBERSHIP)
    sample_date: str = "2017-06-30"
    sample_size: int = 100
    start: str = "2017-01-01"
    end: str = "2018-12-31"
    min_names: int = 20
    quantile: int = 5
    max_evaluations: int = 2000
    min_coverage: float = 0.50
    min_dates: int = 300
    ic_weight: float = 1.0
    icir_weight: float = 0.5
    turnover_penalty: float = 0.05
    complexity_penalty: float = 0.0005
    kernels: int = 4

    def __post_init__(self) -> None:
        if self.sample_size < self.min_names:
            raise ValueError("sample_size must be >= min_names")
        if self.max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive")


@dataclass
class GPMetrics:
    expression: str
    canonical: str | None
    expression_hash: str | None
    fitness: float | None
    rank_ic: float | None
    icir: float | None
    turnover: float | None
    coverage: float | None
    n_dates: int
    complexity: int | None
    monotonicity: float | None
    error: str | None = None

    @property
    def valid(self) -> bool:
        return self.error is None and self.fitness is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "canonical": self.canonical,
            "expression_hash": self.expression_hash,
            "fitness": self.fitness,
            "rank_ic": self.rank_ic,
            "icir": self.icir,
            "turnover": self.turnover,
            "coverage": self.coverage,
            "n_dates": self.n_dates,
            "complexity": self.complexity,
            "monotonicity": self.monotonicity,
            "error": self.error,
        }


class GPEvaluator:
    """Evaluate expressions on a fixed sample, with a hard evaluation budget."""

    def __init__(self, config: GPEvalConfig) -> None:
        self.config = config
        self.n_evaluations = 0
        self.cache: dict[str, GPMetrics] = {}
        self._panel_cache: dict[str, pd.DataFrame] = {}
        self.names = self._sample_names()
        self._ensure_qlib()
        from qlib.data import D

        labels = D.features(
            self.names,
            ["Ref($close, -2) / Ref($close, -1) - 1"],
            start_time=config.start,
            end_time=config.end,
            freq="day",
        )
        if labels is None or labels.empty:
            raise RuntimeError("could not load GP label panel")
        labels.columns = ["fwd_1"]
        labels.index = labels.index.set_names(["instrument", "datetime"])
        self.labels = labels.sort_index()

    def _sample_names(self) -> list[str]:
        membership = pd.read_parquet(self.config.membership_path)
        membership["datetime"] = pd.to_datetime(membership["datetime"])
        day = membership[membership["datetime"] == pd.Timestamp(self.config.sample_date)]
        names = sorted(day["instrument"].astype(str).str.upper().unique().tolist())
        if len(names) < self.config.sample_size:
            raise ValueError(
                f"only {len(names)} members on {self.config.sample_date}, "
                f"need {self.config.sample_size}"
            )
        rng = random.Random(0)
        return sorted(rng.sample(names, self.config.sample_size))

    def _ensure_qlib(self) -> None:
        global _QLIB_PROVIDER
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        resolved = str(Path(self.config.provider_uri).resolve())
        if _QLIB_PROVIDER != resolved:
            import qlib

            qlib.init(provider_uri=resolved, region="cn", logging_level="ERROR")
            _QLIB_PROVIDER = resolved
        from qlib.config import C

        C.joblib_backend = "threading"
        C.kernels = int(self.config.kernels)

    @property
    def remaining(self) -> int:
        return self.config.max_evaluations - self.n_evaluations

    def factor_panel(self, expression: str, validated: ValidatedExpression | None = None) -> pd.DataFrame:
        """Load one factor panel for hall-of-fame correlation checks."""
        if validated is None:
            validated = validate_expression(expression)
        canonical = validated.canonical
        if canonical in self._panel_cache:
            return self._panel_cache[canonical]
        from qlib.data import D

        factor = D.features(
            self.names,
            [canonical],
            start_time=self.config.start,
            end_time=self.config.end,
            freq="day",
        )
        if factor is None or factor.empty:
            raise RuntimeError("Qlib returned an empty factor panel")
        factor.columns = ["factor"]
        factor.index = factor.index.set_names(["instrument", "datetime"])
        if len(self._panel_cache) >= 40:
            self._panel_cache.pop(next(iter(self._panel_cache)))
        self._panel_cache[canonical] = factor
        return factor

    def evaluate(
        self,
        expression: str,
        validated: ValidatedExpression | None = None,
    ) -> GPMetrics:
        if validated is None:
            validated = validate_expression(expression)
        canonical = validated.canonical
        if canonical in self.cache:
            return self.cache[canonical]
        if self.n_evaluations >= self.config.max_evaluations:
            raise GPBudgetExhausted(f"evaluation budget {self.config.max_evaluations} exhausted")
        self.n_evaluations += 1

        try:
            from qlib.data import D

            factor = D.features(
                self.names,
                [canonical],
                start_time=self.config.start,
                end_time=self.config.end,
                freq="day",
            )
            if factor is None or factor.empty:
                raise RuntimeError("Qlib returned an empty factor panel")
            factor.columns = ["factor"]
            factor.index = factor.index.set_names(["instrument", "datetime"])
            panel = factor.join(self.labels, how="inner").sort_index()
            coverage = coverage_stats(panel["factor"], total_rows=len(panel))
            rank_ic, _ = compute_ic(
                panel, "factor", "fwd_1", min_names=self.config.min_names
            )
            if len(rank_ic) < self.config.min_dates:
                raise RuntimeError(f"only {len(rank_ic)} IC dates")
            mean_ic = float(rank_ic.mean())
            std_ic = float(rank_ic.std(ddof=1))
            icir = mean_ic / std_ic if std_ic > 0 else 0.0
            turnover = top_quantile_turnover(
                panel, "factor", q=self.config.quantile, min_names=self.config.min_names
            )
            monotonicity = None
            cov_value = coverage.get("coverage")
            if cov_value is None or cov_value < self.config.min_coverage:
                raise RuntimeError(f"coverage {cov_value} below {self.config.min_coverage}")
            if coverage.get("unique_values", 0) <= 1:
                raise RuntimeError("constant factor")
            fitness = (
                self.config.ic_weight * abs(mean_ic)
                + self.config.icir_weight * abs(icir)
                - self.config.turnover_penalty * (turnover or 0.0)
                - self.config.complexity_penalty * validated.complexity
            )
            metrics = GPMetrics(
                expression=validated.raw,
                canonical=canonical,
                expression_hash=validated.expression_hash,
                fitness=float(fitness),
                rank_ic=mean_ic,
                icir=icir,
                turnover=turnover,
                coverage=cov_value,
                n_dates=len(rank_ic),
                complexity=validated.complexity,
                monotonicity=monotonicity,
            )
        except Exception as exc:
            metrics = GPMetrics(
                expression=expression,
                canonical=canonical,
                expression_hash=validated.expression_hash,
                fitness=None,
                rank_ic=None,
                icir=None,
                turnover=None,
                coverage=None,
                n_dates=0,
                complexity=validated.complexity,
                monotonicity=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        self.cache[canonical] = metrics
        return metrics
