"""Factor combination for Stage 9.

Three combinations are supported:

* ``equal``: equal-weight standardized factors, with the sign learned on the
  training window only;
* ``icir``: train-window ICIR weights, again with train-only signs;
* ``lgbm``: a small LightGBM model trained on the training window.

No validation or test observation is used to fit signs or weights.  This is a
deliberate change from Stage 8's descriptive ``direction`` field, which is
computed on the full sample for reporting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.factors.evaluator import EvaluationConfig, load_factor
from src.factors.metrics import compute_ic, normalize_panel
from src.factors.preprocess import preprocess_factor


@dataclass(frozen=True)
class FactorSpec:
    name: str
    expression: str
    source: str = ""
    s2_status: str = ""


@dataclass
class ScoreResult:
    method: str
    scores: pd.DataFrame
    diagnostics: pd.DataFrame
    feature_importance: pd.DataFrame | None = None
    model: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "method": self.method,
            "diagnostics": self.diagnostics.to_dict(orient="records"),
            "feature_importance": (
                self.feature_importance.to_dict(orient="records")
                if self.feature_importance is not None
                else None
            ),
        }
        return payload


def load_specs_from_stage8(
    path: str | Path,
    *,
    status: str = "pass",
) -> list[FactorSpec]:
    """Load the Stage 8 pass list (or another status) as factor specs."""
    frame = pd.read_csv(path, keep_default_na=False)
    if "s2_status" not in frame.columns:
        raise ValueError(f"{path} has no s2_status column")
    frame = frame[frame["s2_status"].astype(str) == status]
    specs = []
    for _, row in frame.iterrows():
        canonical = str(row.get("canonical") or "").strip()
        if not canonical:
            continue
        specs.append(
            FactorSpec(
                name=str(row.get("name") or canonical),
                expression=canonical,
                source=str(row.get("source") or ""),
                s2_status=status,
            )
        )
    if not specs:
        raise ValueError(f"no Stage 8 factors with status={status}")
    return specs


def load_factor_panels(
    specs: list[FactorSpec],
    config: EvaluationConfig,
    *,
    winsorize: float | None = 0.01,
    standardize: bool = True,
) -> pd.DataFrame:
    """Load and cross-sectionally preprocess factors into one long panel."""
    frames: list[pd.Series] = []
    names: list[str] = []
    for spec in specs:
        validated, panel = load_factor(spec.expression, config)
        processed = preprocess_factor(panel, winsorize=winsorize, standardize=standardize)
        processed = processed.replace([np.inf, -np.inf], np.nan).rename(spec.name)
        frames.append(processed)
        names.append(spec.name)
        del validated
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate factor names: {names}")
    combined = pd.concat(frames, axis=1)
    combined.index = combined.index.set_names(["instrument", "datetime"])
    return normalize_panel(combined)


def forward_returns(price: pd.DataFrame, *, horizon: int = 1, lag: int = 1) -> pd.Series:
    """Build a tradable forward return series from a wide price panel.

    ``price[t+lag+horizon] / price[t+lag] - 1``: signal at ``t``, trade at
    ``t+lag`` and exit ``horizon`` bars later.
    """
    if horizon < 1 or lag < 1:
        raise ValueError("horizon and lag must be >= 1")
    wide = price.sort_index()
    fwd = wide.shift(-(horizon + lag)) / wide.shift(-lag) - 1.0
    try:
        stacked = fwd.stack(future_stack=True)
    except TypeError:  # pragma: no cover - older pandas fallback
        stacked = fwd.stack()
    stacked.index = stacked.index.set_names(["datetime", "instrument"])
    stacked = stacked.reorder_levels(["instrument", "datetime"]).sort_index()
    return stacked.rename(f"fwd_{horizon}")


# Backwards-compatible, descriptive alias used in the Stage 9 script.
def open_to_open_returns(open_px: pd.DataFrame, *, horizon: int = 1) -> pd.Series:
    return forward_returns(open_px, horizon=horizon, lag=1)


def train_icir_weights(
    factors: pd.DataFrame,
    label: pd.Series,
    *,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    min_names: int = 20,
    min_dates: int = 100,
) -> tuple[pd.DataFrame, pd.Series]:
    """Fit direction and ICIR weights on the training window only."""
    train = factors.join(label, how="inner")
    dates = train.index.get_level_values("datetime")
    train = train[(dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end))]
    rows = []
    raw_weights: dict[str, float] = {}
    signs: dict[str, float] = {}
    for name in factors.columns:
        ic, _ = compute_ic(train, name, label.name, min_names=min_names)
        mean = float(ic.mean()) if len(ic) else 0.0
        std = float(ic.std(ddof=1)) if len(ic) > 1 else 0.0
        icir = mean / std if std > 1e-12 else 0.0
        sign = 1.0 if mean >= 0 else -1.0
        usable = len(ic) >= min_dates and abs(mean) > 1e-12
        weight = abs(icir) if usable else 0.0
        signs[name] = sign
        raw_weights[name] = weight
        rows.append(
            {
                "name": name,
                "n_dates": len(ic),
                "mean_ic": mean,
                "std_ic": std,
                "icir": icir,
                "sign": sign,
                "weight_raw": weight,
            }
        )
    total = sum(raw_weights.values())
    if total <= 1e-12:
        # No factor has a stable train-window sign; fall back to equal weights
        # rather than producing an all-zero score.
        weights = pd.Series(1.0 / max(len(factors.columns), 1), index=factors.columns, dtype=float)
        for row in rows:
            row["weight"] = float(weights[row["name"]])
    else:
        weights = pd.Series({name: value / total for name, value in raw_weights.items()}, dtype=float)
        for row in rows:
            row["weight"] = float(weights[row["name"]])
    diagnostics = pd.DataFrame(rows).set_index("name")
    return diagnostics, weights


def combine_linear(
    factors: pd.DataFrame,
    diagnostics: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """Combine standardized factors with train-only signs and weights."""
    coefficients = {}
    for name in factors.columns:
        if name not in diagnostics.index or name not in weights.index:
            coefficients[name] = 0.0
            continue
        coefficients[name] = float(diagnostics.loc[name, "sign"]) * float(weights[name])
    if not any(abs(value) > 1e-15 for value in coefficients.values()):
        return pd.Series(np.nan, index=factors.index, name="score")
    score = (factors * pd.Series(coefficients)).sum(axis=1, min_count=1)
    return score.rename("score")


def _to_wide(score: pd.Series) -> pd.DataFrame:
    wide = score.unstack("instrument").sort_index()
    return wide.sort_index(axis=1)


def fit_lgbm_combination(
    factors: pd.DataFrame,
    label: pd.Series,
    *,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    params: dict[str, Any] | None = None,
    seed: int = 42,
) -> tuple[pd.Series, pd.DataFrame]:
    """Train a small LightGBM model on the training window and predict all rows."""
    from lightgbm import LGBMRegressor

    merged = factors.join(label, how="inner")
    dates = merged.index.get_level_values("datetime")
    train = merged[(dates >= pd.Timestamp(train_start)) & (dates <= pd.Timestamp(train_end))]
    train = train.dropna(subset=[label.name])
    features = list(factors.columns)
    usable = train[features].notna().any(axis=1)
    train = train[usable]
    if train.empty:
        raise ValueError("no LightGBM training rows")
    model_params: dict[str, Any] = {
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 50,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": seed,
        "n_jobs": 4,
        "verbose": -1,
    }
    model_params.update(params or {})
    model = LGBMRegressor(**model_params)
    model.fit(train[features].to_numpy(dtype=float), train[label.name].to_numpy(dtype=float))
    predict_rows = factors[factors.notna().any(axis=1)]
    prediction = model.predict(predict_rows.to_numpy(dtype=float))
    score = pd.Series(prediction, index=predict_rows.index, name="score")
    importance = pd.DataFrame(
        {
            "name": features,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False, ignore_index=True)
    return score, importance


def build_combination(
    method: str,
    factors: pd.DataFrame,
    label: pd.Series,
    *,
    train_start: str | pd.Timestamp,
    train_end: str | pd.Timestamp,
    min_names: int = 20,
    min_dates: int = 100,
    lgbm_params: dict[str, Any] | None = None,
    seed: int = 42,
) -> ScoreResult:
    """Build one score panel from a combination method."""
    method = method.lower()
    if method == "lgbm":
        score, importance = fit_lgbm_combination(
            factors,
            label,
            train_start=train_start,
            train_end=train_end,
            params=lgbm_params,
            seed=seed,
        )
        diagnostics = pd.DataFrame(
            [
                {
                    "name": name,
                    "sign": np.nan,
                    "mean_ic": np.nan,
                    "icir": np.nan,
                    "weight": np.nan,
                }
                for name in factors.columns
            ]
        ).set_index("name")
        return ScoreResult(
            method=method,
            scores=_to_wide(score),
            diagnostics=diagnostics,
            feature_importance=importance,
        )

    diagnostics, weights = train_icir_weights(
        factors,
        label,
        train_start=train_start,
        train_end=train_end,
        min_names=min_names,
        min_dates=min_dates,
    )
    if method == "equal":
        weights = pd.Series(1.0 / len(factors.columns), index=factors.columns, dtype=float)
        diagnostics["weight"] = weights
    elif method != "icir":
        raise ValueError(f"unknown combination method: {method}")
    score = combine_linear(factors, diagnostics, weights)
    return ScoreResult(
        method=method,
        scores=_to_wide(score),
        diagnostics=diagnostics,
    )
