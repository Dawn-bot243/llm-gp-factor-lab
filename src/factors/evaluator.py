"""Load one Qlib factor expression and compute Stage 3 metrics.

The evaluator is deliberately stricter than a one-off research notebook:

* it validates every expression before Qlib sees it;
* it filters the panel by the daily PIT membership table;
* it records the data snapshot, expression hash and config hash;
* it caches factor and label panels separately, keyed by content;
* it returns a JSON-serializable metrics record plus CSV-friendly series.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.factors.expression import (
    DegenerateFactorError,
    ExpressionError,
    ValidatedExpression,
    slugify,
    validate_expression,
)
from src.factors.metrics import (
    compute_ic,
    coverage_stats,
    distribution_stats,
    ic_stats,
    normalize_panel,
    quantile_returns,
    quantile_stats,
    rank_autocorrelation,
    top_quantile_turnover,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMBERSHIP = PROJECT_ROOT / "data" / "processed" / "pit_csi300_membership.parquet"
DEFAULT_CONSTRAINTS = PROJECT_ROOT / "data" / "processed" / "pit_csi300_constraints.parquet"
DEFAULT_SNAPSHOT = PROJECT_ROOT / "data" / "manifests" / "qlib_snapshot.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "factor_cache"

_QLIB_PROVIDER: str | None = None
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationConfig:
    """Everything that affects the factor values or their evaluation."""

    provider_uri: str
    instruments: str = "csi300"
    start: str = "2010-01-01"
    end: str = "2020-09-23"
    freq: str = "day"
    membership_path: str = str(DEFAULT_MEMBERSHIP)
    constraints_path: str | None = str(DEFAULT_CONSTRAINTS)
    snapshot_file: str | None = str(DEFAULT_SNAPSHOT)
    horizons: tuple[int, ...] = (1, 2, 5, 10, 20)
    quantiles: int = 5
    min_names: int = 20
    min_coverage: float = 0.10
    exclude_suspended: bool = False
    exclude_at_limit: bool = False
    cache_dir: str | None = str(DEFAULT_CACHE_DIR)
    use_cache: bool = True
    kernels: int = 4

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("at least one horizon is required")
        for horizon in self.horizons:
            if int(horizon) < 1:
                raise ValueError(f"horizon must be >= 1, got {horizon}")
        if self.quantiles < 2:
            raise ValueError("quantiles must be >= 2")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in [0, 1]")

    @property
    def label_expressions(self) -> dict[str, str]:
        # Signal at t, trade at t+1 close, then hold for h days.
        return {
            f"fwd_{h}": f"Ref($close, -{h + 1}) / Ref($close, -1) - 1"
            for h in self.horizons
        }

    @property
    def snapshot_id(self) -> str | None:
        if not self.snapshot_file:
            return None
        path = Path(self.snapshot_file)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        value = payload.get("snapshot_id")
        return str(value) if value else None

    def config_hash(self) -> str:
        membership_sha = _file_sha256(self.membership_path) if Path(self.membership_path).exists() else None
        constraints_sha = None
        if self.constraints_path and (self.exclude_suspended or self.exclude_at_limit):
            constraint_file = Path(self.constraints_path)
            if constraint_file.exists():
                constraints_sha = _file_sha256(str(constraint_file))
        payload = {
            "provider_uri": str(Path(self.provider_uri).resolve()),
            "instruments": self.instruments,
            "start": self.start,
            "end": self.end,
            "freq": self.freq,
            "membership_sha256": membership_sha,
            "constraints_sha256": constraints_sha,
            "snapshot_id": self.snapshot_id,
            "horizons": list(self.horizons),
            "exclude_suspended": self.exclude_suspended,
            "exclude_at_limit": self.exclude_at_limit,
        }
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_uri": self.provider_uri,
            "instruments": self.instruments,
            "start": self.start,
            "end": self.end,
            "freq": self.freq,
            "membership_path": self.membership_path,
            "constraints_path": self.constraints_path,
            "snapshot_file": self.snapshot_file,
            "snapshot_id": self.snapshot_id,
            "horizons": list(self.horizons),
            "quantiles": self.quantiles,
            "min_names": self.min_names,
            "min_coverage": self.min_coverage,
            "exclude_suspended": self.exclude_suspended,
            "exclude_at_limit": self.exclude_at_limit,
            "cache_dir": self.cache_dir,
            "use_cache": self.use_cache,
            "kernels": self.kernels,
        }


@dataclass
class EvaluationResult:
    """Metrics plus the series that are useful for plots and debugging."""

    metrics: dict[str, Any]
    rank_ic: pd.Series
    ic: pd.Series
    quantile_daily: pd.DataFrame
    long_short: pd.Series
    panel: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        return str(self.metrics["name"])

    @property
    def expression_hash(self) -> str:
        return str(self.metrics["expression_hash"])


def evaluate_expression(
    expression: str,
    config: EvaluationConfig,
    *,
    name: str | None = None,
    allowed_fields: Iterable[str] | None = None,
) -> EvaluationResult:
    """Validate, load and evaluate one expression."""
    validated = validate_expression(expression, allowed_fields=allowed_fields)
    factor = _load_or_compute_factor(validated, config)
    labels = _load_or_compute_labels(config)
    panel = factor.join(labels, how="left")
    panel = normalize_panel(panel)
    result = evaluate_panel(panel, validated, config, name=name or validated.canonical)
    return result


def evaluate_panel(
    panel: pd.DataFrame,
    validated: ValidatedExpression,
    config: EvaluationConfig,
    *,
    name: str,
) -> EvaluationResult:
    """Compute metrics from an already-loaded ``(instrument, datetime)`` panel."""
    if "factor" not in panel.columns:
        raise ValueError("panel needs a 'factor' column")
    panel = normalize_panel(panel)

    membership_rows = _membership_row_count(config.membership_path, config.start, config.end)
    # When tradability filters are on, coverage should be measured against the
    # rows that survive those filters, not against suspended/limit rows that
    # were deliberately removed.
    coverage_total = (
        len(panel)
        if (config.exclude_suspended or config.exclude_at_limit)
        else (membership_rows or len(panel))
    )
    coverage = coverage_stats(panel["factor"], total_rows=coverage_total)
    if coverage["finite_rows"] == 0:
        raise DegenerateFactorError("factor has no finite values")
    if coverage["unique_values"] <= 1:
        raise DegenerateFactorError("factor is constant")
    if coverage["coverage"] is not None and coverage["coverage"] < config.min_coverage:
        raise DegenerateFactorError(
            f"factor coverage {coverage['coverage']:.1%} is below min_coverage {config.min_coverage:.1%}"
        )

    distribution = distribution_stats(panel["factor"])
    ic_by_horizon: list[dict[str, Any]] = []
    rank_ic = pd.Series(dtype=float, name="rank_ic")
    pearson_ic = pd.Series(dtype=float, name="ic")
    for horizon in config.horizons:
        label_col = f"fwd_{horizon}"
        if label_col not in panel.columns:
            raise ValueError(f"panel is missing label column '{label_col}'")
        rank_h, ic_h = compute_ic(panel, "factor", label_col, min_names=config.min_names)
        stats = ic_stats(rank_h)
        ic_by_horizon.append({"horizon": int(horizon), **stats})
        if horizon == config.horizons[0]:
            rank_ic, pearson_ic = rank_h, ic_h

    qmean, long_short, qdf = quantile_returns(
        panel, "factor", f"fwd_{config.horizons[0]}", q=config.quantiles, min_names=config.min_names
    )
    qstats = quantile_stats(qmean, long_short)
    turnover = top_quantile_turnover(
        panel, "factor", q=config.quantiles, min_names=config.min_names
    )
    autocorr = rank_autocorrelation(panel, "factor", min_names=config.min_names)

    base_ic = ic_by_horizon[0]
    warnings: list[str] = []
    if coverage["coverage"] is not None and coverage["coverage"] < 0.5:
        warnings.append(f"low coverage: {coverage['coverage']:.1%}")
    if base_ic.get("mean") is not None and float(base_ic["mean"]) < 0:
        warnings.append("negative RankIC at the shortest horizon")
    monotonicity = qstats.get("monotonicity")
    if monotonicity is not None and abs(float(monotonicity)) < 0.5:
        warnings.append("quantile means are not monotone")

    metrics: dict[str, Any] = {
        "schema_version": "stage3.factor_eval.v1",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "name": name,
        "expression": validated.raw,
        "canonical": validated.canonical,
        "expression_hash": validated.expression_hash,
        "fields": list(validated.fields),
        "operators": list(validated.operators),
        "max_window": validated.max_window,
        "complexity": validated.complexity,
        "data": {
            **config.to_dict(),
            "snapshot_id": config.snapshot_id,
            "n_membership_rows": int(membership_rows),
            "n_coverage_rows": int(coverage_total),
            "n_panel_rows": len(panel),
            "n_rows_used": int(coverage["finite_rows"]),
            "n_dates": int(panel.index.get_level_values("datetime").nunique()),
            "n_instruments": int(panel.index.get_level_values("instrument").nunique()),
            "factor_coverage": coverage,
        },
        "distribution": distribution,
        "ic": {
            f"fwd_{h}": stats for h, stats in zip(config.horizons, ic_by_horizon, strict=True)
        },
        "ic_decay": ic_by_horizon,
        "quantiles": {
            "q": config.quantiles,
            **qstats,
        },
        "turnover": {
            "top_quantile_one_way": turnover,
            "rank_autocorrelation": autocorr,
        },
        "warnings": warnings,
    }
    return EvaluationResult(
        metrics=metrics,
        rank_ic=rank_ic,
        ic=pearson_ic,
        quantile_daily=qdf,
        long_short=long_short,
        panel=panel,
    )


def evaluate_batch(
    specs: Iterable[tuple[str, str]],
    config: EvaluationConfig,
    output_dir: str | Path,
    *,
    save_series: bool = False,
    allowed_fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Evaluate many expressions, isolating failures per expression."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    factors_dir = out_dir / "factors"
    factors_dir.mkdir(exist_ok=True)

    rows: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for index, (name, expression) in enumerate(specs, start=1):
        try:
            validated = validate_expression(expression, allowed_fields=allowed_fields)
        except ExpressionError as exc:
            rows.append(_error_row(name, expression, exc))
            continue
        if validated.canonical in seen:
            rows.append(
                {
                    "name": name,
                    "expression": expression,
                    "canonical": validated.canonical,
                    "expression_hash": validated.expression_hash,
                    "status": "duplicate",
                    "duplicate_of": seen[validated.canonical],
                    "error": None,
                }
            )
            continue
        seen[validated.canonical] = name
        try:
            result = evaluate_expression(
                expression,
                config,
                name=name,
                allowed_fields=allowed_fields,
            )
        except Exception as exc:
            rows.append(_error_row(name, expression, exc))
            continue
        row = _leaderboard_row(result)
        rows.append(row)
        factor_dir = factors_dir / f"{slugify(name)}_{result.expression_hash[:8]}"
        write_result(result, factor_dir, save_panel=False, save_series=save_series)
        rank_ic = row.get("rank_ic_mean")
        icir = row.get("icir")
        coverage = row.get("coverage")
        rank_text = f"{rank_ic:.4f}" if isinstance(rank_ic, float) else "n/a"
        icir_text = f"{icir:.3f}" if isinstance(icir, float) else "n/a"
        coverage_text = f"{coverage:.3f}" if isinstance(coverage, float) else "n/a"
        print(f"[{index}] {name}: RankIC={rank_text} ICIR={icir_text} coverage={coverage_text}")

    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty and "rank_ic_mean" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(
            "rank_ic_mean", ascending=False, na_position="last"
        ).reset_index(drop=True)
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)
    summary = {
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "config": config.to_dict(),
        "n_requested": len(rows),
        "n_ok": _count_status(leaderboard, "ok"),
        "n_error": _count_status(leaderboard, "error"),
        "n_duplicate": _count_status(leaderboard, "duplicate"),
        "rows": _jsonable(rows),
    }
    (out_dir / "leaderboard.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return summary


def write_result(
    result: EvaluationResult,
    output_dir: str | Path,
    *,
    save_panel: bool = False,
    save_series: bool = True,
) -> Path:
    """Write metrics plus CSV series; return the output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(
        json.dumps(_jsonable(result.metrics), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    if save_series:
        pd.DataFrame({"rank_ic": result.rank_ic, "ic": result.ic}).to_csv(
            out_dir / "rank_ic.csv", index_label="datetime"
        )
        if not result.quantile_daily.empty:
            result.quantile_daily.to_csv(out_dir / "quantile_daily.csv", index_label="datetime")
        result.long_short.rename("long_short").to_csv(
            out_dir / "long_short.csv", index_label="datetime"
        )
        pd.DataFrame(result.metrics["ic_decay"]).to_csv(out_dir / "ic_decay.csv", index=False)
    if save_panel and result.panel is not None:
        result.panel.to_parquet(out_dir / "factor_panel.parquet")
    return out_dir


def load_qlib_panel(
    expressions: list[str],
    columns: list[str],
    config: EvaluationConfig,
) -> pd.DataFrame:
    """Load Qlib expressions and rename columns in order."""
    _ensure_qlib(config.provider_uri, config.kernels)
    from qlib.data import D

    frame = D.features(
        D.instruments(config.instruments),
        expressions,
        start_time=config.start,
        end_time=config.end,
        freq=config.freq,
    )
    if frame is None or frame.empty:
        raise RuntimeError(
            f"Qlib returned no rows for {config.instruments} {config.start}..{config.end}"
        )
    if len(frame.columns) != len(columns):
        raise RuntimeError(f"expected {len(columns)} columns, got {len(frame.columns)}")
    frame = frame.copy()
    frame.columns = columns
    frame = normalize_panel(frame)
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame


def _load_or_compute_factor(
    validated: ValidatedExpression,
    config: EvaluationConfig,
) -> pd.DataFrame:
    cache_key = config.config_hash()
    cache_path, meta_path = _factor_cache_paths(config, validated, cache_key)
    if config.use_cache and _cache_valid(cache_path, meta_path, validated.canonical, cache_key):
        try:
            frame = pd.read_parquet(cache_path)
            return _ensure_panel(frame, ["factor"])
        except Exception:
            logger.warning("ignoring unreadable factor cache: %s", cache_path, exc_info=True)

    frame = load_qlib_panel([validated.canonical], ["factor"], config)
    frame = _filter_membership(frame, config.membership_path)
    frame, _ = _apply_constraints(frame, config)
    frame = normalize_panel(frame)
    if config.use_cache:
        _write_cache(frame, cache_path, meta_path, validated, config, cache_key)
    return frame


def _load_or_compute_labels(config: EvaluationConfig) -> pd.DataFrame:
    cache_key = config.config_hash()
    cache_path, meta_path = _label_cache_paths(config, cache_key)
    label_columns = list(config.label_expressions)
    if config.use_cache and _cache_valid(cache_path, meta_path, None, cache_key):
        try:
            frame = pd.read_parquet(cache_path)
            if set(label_columns).issubset(frame.columns):
                return _ensure_panel(frame, label_columns)
        except Exception:
            logger.warning("ignoring unreadable label cache: %s", cache_path, exc_info=True)

    frame = load_qlib_panel(list(config.label_expressions.values()), label_columns, config)
    frame = _filter_membership(frame, config.membership_path)
    frame = normalize_panel(frame)
    if config.use_cache:
        _write_cache(frame, cache_path, meta_path, None, config, cache_key)
    return frame


def _filter_membership(frame: pd.DataFrame, membership_path: str) -> pd.DataFrame:
    path = Path(membership_path)
    if not path.exists():
        raise FileNotFoundError(f"PIT membership file not found: {path}")
    membership = pd.read_parquet(path)
    if not {"instrument", "datetime"}.issubset(membership.columns):
        raise ValueError(f"{path} needs instrument and datetime columns")
    membership = membership[["instrument", "datetime"]].copy()
    membership["instrument"] = membership["instrument"].astype(str).str.upper()
    membership["datetime"] = pd.to_datetime(membership["datetime"])
    membership = membership.drop_duplicates()

    left = frame.reset_index()
    left["instrument"] = left["instrument"].astype(str).str.upper()
    left["datetime"] = pd.to_datetime(left["datetime"])
    merged = left.merge(membership, on=["instrument", "datetime"], how="inner")
    return normalize_panel(merged.set_index(["instrument", "datetime"]))


def _apply_constraints(
    frame: pd.DataFrame, config: EvaluationConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata: dict[str, Any] = {"applied": False}
    path = Path(config.constraints_path) if config.constraints_path else None
    if path is None or not path.exists():
        return frame, metadata
    constraints = pd.read_parquet(path)
    columns = [
        col
        for col in ("instrument", "date", "suspended", "is_st", "at_limit_up", "at_limit_down")
        if col in constraints.columns
    ]
    if "date" not in columns or "instrument" not in columns:
        return frame, metadata
    constraints = constraints[columns].rename(columns={"date": "datetime"})
    constraints["instrument"] = constraints["instrument"].astype(str).str.upper()
    constraints["datetime"] = pd.to_datetime(constraints["datetime"])
    constraints = constraints.drop_duplicates(["instrument", "datetime"])

    merged = frame.reset_index().merge(constraints, on=["instrument", "datetime"], how="left")
    metadata = {
        "applied": True,
        "n_rows_before": len(merged),
        "n_suspended": int(merged.get("suspended", pd.Series(dtype=float)).fillna(0).sum()),
        "n_at_limit": int(
            (
                merged.get("at_limit_up", pd.Series(dtype=bool)).fillna(False).astype(bool)
                | merged.get("at_limit_down", pd.Series(dtype=bool)).fillna(False).astype(bool)
            ).sum()
        ),
    }
    if config.exclude_suspended and "suspended" in merged.columns:
        merged = merged[merged["suspended"].fillna(0).astype(int) == 0]
    if config.exclude_at_limit:
        if "at_limit_up" in merged.columns:
            merged = merged[~merged["at_limit_up"].fillna(False).astype(bool)]
        if "at_limit_down" in merged.columns:
            merged = merged[~merged["at_limit_down"].fillna(False).astype(bool)]
    metadata["n_rows_after"] = len(merged)
    return normalize_panel(merged.set_index(["instrument", "datetime"])), metadata


def _ensure_qlib(provider_uri: str, kernels: int) -> None:
    global _QLIB_PROVIDER
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    resolved = str(Path(provider_uri).resolve())
    if _QLIB_PROVIDER != resolved:
        import qlib

        qlib.init(provider_uri=resolved, region="cn", logging_level="ERROR")
        _QLIB_PROVIDER = resolved
    from qlib.config import C

    C.joblib_backend = "threading"
    C.kernels = int(kernels)


def _ensure_panel(frame: pd.DataFrame, required: list[str]) -> pd.DataFrame:
    frame = normalize_panel(frame)
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"cached panel is missing columns: {missing}")
    return frame


def _factor_cache_paths(
    config: EvaluationConfig,
    validated: ValidatedExpression,
    cache_key: str,
) -> tuple[Path, Path]:
    base = Path(config.cache_dir) if config.cache_dir else DEFAULT_CACHE_DIR
    stem = f"factor_{cache_key}_{validated.expression_hash[:16]}"
    return base / f"{stem}.parquet", base / f"{stem}.json"


def _label_cache_paths(config: EvaluationConfig, cache_key: str) -> tuple[Path, Path]:
    base = Path(config.cache_dir) if config.cache_dir else DEFAULT_CACHE_DIR
    stem = f"labels_{cache_key}"
    return base / f"{stem}.parquet", base / f"{stem}.json"


def _cache_valid(
    cache_path: Path,
    meta_path: Path,
    canonical: str | None,
    cache_key: str,
) -> bool:
    if not cache_path.exists() or not meta_path.exists():
        return False
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get("config_hash") != cache_key:
        return False
    return not (canonical is not None and metadata.get("canonical") != canonical)


def _write_cache(
    frame: pd.DataFrame,
    cache_path: Path,
    meta_path: Path,
    validated: ValidatedExpression | None,
    config: EvaluationConfig,
    cache_key: str,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    frame.to_parquet(tmp)
    tmp.replace(cache_path)
    metadata = {
        "schema_version": "stage3.factor_cache.v1",
        "created_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "config_hash": cache_key,
        "config": config.to_dict(),
        "canonical": validated.canonical if validated else None,
        "expression_hash": validated.expression_hash if validated else None,
        "rows": len(frame),
        "columns": list(frame.columns),
    }
    tmp_meta = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_meta.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    tmp_meta.replace(meta_path)


@lru_cache(maxsize=16)
def _membership_row_count(path: str, start: str, end: str) -> int:
    file_path = Path(path)
    if not file_path.exists():
        return 0
    membership = pd.read_parquet(file_path)
    if "datetime" not in membership.columns:
        return len(membership)
    dates = pd.to_datetime(membership["datetime"])
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    return int(((dates >= lo) & (dates <= hi)).sum())


@lru_cache(maxsize=16)
def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _leaderboard_row(result: EvaluationResult) -> dict[str, Any]:
    metrics = result.metrics
    # The shortest horizon is always present and is the same label used by the
    # Stage 1 baseline.
    shortest = metrics["ic_decay"][0]
    long_short = metrics["quantiles"].get("long_short", {})
    return {
        "name": metrics["name"],
        "expression": metrics["expression"],
        "canonical": metrics["canonical"],
        "expression_hash": metrics["expression_hash"],
        "status": "ok",
        "rank_ic_mean": shortest.get("mean"),
        "icir": shortest.get("icir"),
        "newey_west_t": shortest.get("newey_west_t"),
        "coverage": metrics["data"]["factor_coverage"].get("coverage"),
        "turnover": metrics["turnover"].get("top_quantile_one_way"),
        "long_short_mean": long_short.get("mean"),
        "error": None,
    }


def _count_status(leaderboard: pd.DataFrame, status: str) -> int:
    if leaderboard.empty or "status" not in leaderboard.columns:
        return 0
    return int((leaderboard["status"] == status).sum())


def _error_row(name: str, expression: str, exc: Exception) -> dict[str, Any]:
    return {
        "name": name,
        "expression": expression,
        "canonical": None,
        "expression_hash": None,
        "status": "error",
        "rank_ic_mean": None,
        "icir": None,
        "newey_west_t": None,
        "coverage": None,
        "turnover": None,
        "long_short_mean": None,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA:
        return None
    return value


def _json_default(value: Any) -> Any:
    converted = _jsonable(value)
    if converted is value:
        return str(value)
    return converted
