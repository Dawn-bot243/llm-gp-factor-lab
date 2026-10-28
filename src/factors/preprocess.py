"""Cross-sectional factor preprocessing for Stage 4.

All operations are performed within one trading date, so they cannot use
future information.  Industry/size neutralization is available as an optional
residualization helper, but the current data snapshot does not include
industry classification or market-cap fields; S1 therefore runs without it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_cross_section(
    panel: pd.DataFrame,
    factor_col: str = "factor",
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Clip each date's factor values at the given cross-sectional quantiles."""
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("winsorize bounds must satisfy 0 <= lower < upper <= 1")

    def _clip(group: pd.Series) -> pd.Series:
        if group.notna().sum() == 0:
            return group
        lo, hi = group.quantile([lower, upper])
        if pd.isna(lo) or pd.isna(hi):
            return group
        return group.clip(lo, hi)

    return panel.groupby(level="datetime")[factor_col].transform(_clip).rename(factor_col)


def standardize_cross_section(
    panel: pd.DataFrame,
    factor_col: str = "factor",
) -> pd.Series:
    """Z-score each date's factor values; constant dates become NaN."""
    grouped = panel.groupby(level="datetime")[factor_col]
    mean = grouped.transform("mean")
    std = grouped.transform("std")
    return ((panel[factor_col] - mean) / std.replace(0.0, np.nan)).rename(factor_col)


def neutralize_cross_section(
    panel: pd.DataFrame,
    exposures: pd.DataFrame,
    factor_col: str = "factor",
    add_intercept: bool = True,
) -> pd.Series:
    """Residualize the factor against same-date exposures.

    ``exposures`` must share the panel's ``(instrument, datetime)`` index and
    contain only numeric columns.  This is a hook for industry/size
    neutralization; it is not used by the default S1 run because the current
    snapshot has no industry or market-cap fields.
    """
    if not exposures.index.equals(panel.index):
        exposures = exposures.reindex(panel.index)
    numeric = exposures.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("exposures must contain at least one numeric column")

    residuals = pd.Series(np.nan, index=panel.index, name=factor_col)
    for _, group in panel.groupby(level="datetime", sort=True):
        y = group[factor_col]
        x = numeric.loc[group.index]
        valid = y.notna() & x.notna().all(axis=1)
        if valid.sum() <= numeric.shape[1] + (1 if add_intercept else 0):
            continue
        design = x.loc[valid].to_numpy(dtype=float)
        if add_intercept:
            design = np.column_stack([np.ones(len(design)), design])
        target = y.loc[valid].to_numpy(dtype=float)
        try:
            coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        except np.linalg.LinAlgError:
            continue
        residuals.loc[valid[valid].index] = target - design @ coef
    return residuals


def preprocess_factor(
    panel: pd.DataFrame,
    factor_col: str = "factor",
    *,
    winsorize: float | None = 0.01,
    standardize: bool = True,
    exposures: pd.DataFrame | None = None,
) -> pd.Series:
    """Apply winsorization, optional neutralization and standardization."""
    series = panel[factor_col].astype(float)
    working = pd.DataFrame({factor_col: series}, index=panel.index)
    if winsorize is not None:
        working[factor_col] = winsorize_cross_section(
            working, factor_col=factor_col, lower=winsorize, upper=1.0 - winsorize
        )
    if exposures is not None:
        working[factor_col] = neutralize_cross_section(
            working, exposures, factor_col=factor_col
        )
    if standardize:
        working[factor_col] = standardize_cross_section(working, factor_col=factor_col)
    return working[factor_col].rename(factor_col)
