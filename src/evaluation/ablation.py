"""Pure helpers for the Stage 10 LLM/GP ablation."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SearchCandidate:
    group: str
    name: str
    expression: str
    canonical: str
    source: str
    fitness: float | None
    search_rank_ic: float | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchStats:
    group: str
    n_returned: int = 0
    n_unique: int = 0
    n_valid: int = 0
    n_eligible: int = 0
    n_coverage_excluded: int = 0
    n_error: int = 0
    n_selected: int = 0

    @property
    def legal_rate(self) -> float | None:
        return self.n_valid / self.n_returned if self.n_returned else None

    @property
    def unique_rate(self) -> float | None:
        return self.n_unique / self.n_returned if self.n_returned else None


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_float(value: Any) -> float | None:
    return float(value) if _is_finite(value) else None


def load_trial_candidates(
    path: str | Path,
    group: str,
    *,
    top_n: int = 30,
    max_trials: int | None = None,
    max_complexity: int | None = None,
    min_coverage: float | None = None,
    max_expression_length: int | None = None,
) -> tuple[list[SearchCandidate], SearchStats, list[dict[str, Any]]]:
    """Load a Stage 6/random-search trial ledger and select the top-N pool.

    Trials with validation/runtime errors stay in the returned raw rows so the
    legal-rate denominator includes failures, but only finite-fitness unique
    expressions can be selected for downstream S1.
    """
    rows: list[dict[str, Any]] = []
    file_path = Path(path)
    if file_path.exists():
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if max_trials is not None and max_trials > 0:
        rows = rows[:max_trials]
    stats = SearchStats(group=group, n_returned=len(rows))
    by_canonical: dict[str, SearchCandidate] = {}
    errors = 0
    for row in rows:
        canonical = str(row.get("canonical") or row.get("expression") or "").strip()
        if not canonical:
            errors += 1
            continue
        error = row.get("error")
        fitness = row.get("fitness")
        candidate = SearchCandidate(
            group=group,
            name=str(row.get("name") or canonical),
            expression=str(row.get("expression") or canonical),
            canonical=canonical,
            source="trials",
            fitness=_as_float(fitness),
            search_rank_ic=_as_float(row.get("rank_ic")),
            error=str(error) if error else None,
            extra={
                "origin": row.get("origin"),
                "generation": row.get("generation"),
                "icir": row.get("icir"),
                "turnover": row.get("turnover"),
                "coverage": row.get("coverage"),
                "n_dates": row.get("n_dates"),
                "complexity": row.get("complexity"),
            },
        )
        previous = by_canonical.get(canonical)
        if previous is None or (candidate.fitness or -1e9) > (previous.fitness or -1e9):
            by_canonical[canonical] = candidate
    stats.n_unique = len(by_canonical)
    valid = [item for item in by_canonical.values() if item.error is None and item.fitness is not None]
    if max_complexity is not None:
        valid = [
            item
            for item in valid
            if item.extra.get("complexity") is None
            or int(item.extra["complexity"]) <= max_complexity
        ]
    if max_expression_length is not None:
        valid = [item for item in valid if len(item.canonical) <= max_expression_length]
    stats.n_valid = len(valid)
    eligible = valid
    if min_coverage is not None:
        eligible = [
            item
            for item in valid
            if item.extra.get("coverage") is None
            or float(item.extra["coverage"]) >= float(min_coverage)
        ]
        stats.n_coverage_excluded = len(valid) - len(eligible)
    stats.n_eligible = len(eligible)
    stats.n_error = sum(1 for item in by_canonical.values() if item.error is not None)
    selected = sorted(eligible, key=lambda item: item.fitness or -1e9, reverse=True)[:top_n]
    stats.n_selected = len(selected)
    return selected, stats, rows


def load_stage5_candidates(
    path: str | Path,
    group: str,
    *,
    top_n: int = 30,
) -> tuple[list[SearchCandidate], SearchStats]:
    """Load accepted Stage 5 candidates.

    Fast-fitness values are filled later by the caller; this function only
    enforces schema/validator validity and canonical deduplication.
    """
    frame = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    stats = SearchStats(group=group, n_returned=len(frame))
    by_canonical: dict[str, SearchCandidate] = {}
    for _, row in frame.iterrows():
        canonical = str(row.get("canonical") or "").strip()
        if not canonical:
            continue
        if str(row.get("validation_status") or "") != "valid":
            continue
        candidate = SearchCandidate(
            group=group,
            name=str(row.get("name") or canonical),
            expression=str(row.get("raw_expression") or row.get("expression") or canonical),
            canonical=canonical,
            source="stage5",
            fitness=None,
            search_rank_ic=float(row["s1_rank_ic"]) if _is_finite(row.get("s1_rank_ic")) else None,
            extra={
                "category": row.get("category"),
                "s1_status": row.get("s1_status"),
                "s1_icir": row.get("s1_icir"),
            },
        )
        previous = by_canonical.get(canonical)
        if previous is None:
            by_canonical[canonical] = candidate
    stats.n_unique = len(by_canonical)
    values = list(by_canonical.values())
    stats.n_valid = len(values)
    stats.n_error = 0
    selected = sorted(values, key=lambda item: item.search_rank_ic or -1e9, reverse=True)[:top_n]
    stats.n_selected = len(selected)
    return selected, stats


def s1_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize S1 scorecard rows for one ablation group."""
    if not rows:
        return {
            "n_s1": 0,
            "n_pass": 0,
            "n_review": 0,
            "n_fail": 0,
            "n_error": 0,
            "s1_pass_rate": None,
            "s1_accept_rate": None,
            "mean_s1_rank_ic": None,
            "mean_oos_rank_ic": None,
            "oos_ic_retention": None,
        }
    statuses = [str(row.get("status") or "error") for row in rows]
    cal = [value for row in rows if (value := _as_float(row.get("rank_ic_mean"))) is not None]
    oos = [value for row in rows if (value := _as_float(row.get("holdout_rank_ic"))) is not None]
    retention = []
    for row in rows:
        cal_value = _as_float(row.get("rank_ic_mean"))
        oos_value = _as_float(row.get("holdout_rank_ic"))
        if cal_value is not None and oos_value is not None and abs(cal_value) > 1e-12:
            retention.append(oos_value / cal_value)
    n = len(rows)
    n_pass = statuses.count("pass")
    n_review = statuses.count("review")
    n_fail = statuses.count("fail")
    return {
        "n_s1": n,
        "n_pass": n_pass,
        "n_review": n_review,
        "n_fail": n_fail,
        "n_error": statuses.count("error"),
        "s1_pass_rate": n_pass / n if n else None,
        "s1_accept_rate": (n_pass + n_review) / n if n else None,
        "mean_s1_rank_ic": sum(cal) / len(cal) if cal else None,
        "mean_oos_rank_ic": sum(oos) / len(oos) if oos else None,
        "oos_ic_retention": sum(retention) / len(retention) if retention else None,
    }


def protocol_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def select_s3_factors(
    s1_rows: list[dict[str, Any]],
    *,
    min_factors: int = 2,
    fallback_top: int = 2,
    allow_fallback: bool = True,
    allowed_statuses: tuple[str, ...] = ("pass", "review"),
) -> tuple[list[dict[str, Any]], bool]:
    """Select S1 pass+review factors for the S3 ablation.

    If ``allow_fallback`` is false and a group has fewer than
    ``min_factors`` accepted factors, it returns an empty list; the revised
    Stage 10.1 protocol does not compare fallback portfolios.
    """
    accepted = [
        row
        for row in s1_rows
        if str(row.get("status")) in set(allowed_statuses)
        and str(row.get("canonical") or "").strip()
    ]
    if len(accepted) >= min_factors:
        return accepted, False
    if not allow_fallback:
        return [], False
    ranked = sorted(
        [row for row in s1_rows if str(row.get("canonical") or "").strip()],
        key=lambda row: abs(float(row["rank_ic_mean"])) if _is_finite(row.get("rank_ic_mean")) else -1.0,
        reverse=True,
    )
    return ranked[:fallback_top], True


def pass_rate_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float | None, float | None]:
    """Clopper-Pearson exact confidence interval for a binomial proportion."""
    if n <= 0:
        return None, None
    from scipy.stats import beta

    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi
