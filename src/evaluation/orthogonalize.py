"""Cross-sectional residualization against Alpha158 exposures."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.evaluation.dedupe import Candidate, SamplePanels
from src.factors.io import alpha158_specs
from src.factors.metrics import compute_ic, ic_stats


@dataclass
class ExposureConfig:
    alpha158_count: int = 20
    ridge_alpha: float = 1.0
    min_names: int = 20


class Alpha158Exposures:
    """Selected Alpha158 factors, rank-normalized per date."""

    def __init__(self, panels: SamplePanels, config: ExposureConfig) -> None:
        self.config = config
        specs = alpha158_specs(limit=config.alpha158_count)
        self.names = [spec.name for spec in specs]
        from qlib.data import D

        frame = D.features(
            panels.names,
            [spec.expression for spec in specs],
            start_time=panels.start,
            end_time=panels.end,
            freq="day",
        )
        if frame is None or frame.empty:
            raise RuntimeError("could not load Alpha158 exposures")
        frame.columns = [f"a{i}" for i in range(len(specs))]
        frame.index = frame.index.set_names(["instrument", "datetime"])
        self.frame = frame.sort_index()

    def ranked(self) -> pd.DataFrame:
        return self.frame.groupby(level="datetime").rank(pct=True)


def orthogonalize_factor(
    panel: pd.DataFrame,
    exposures: pd.DataFrame,
    *,
    ridge_alpha: float = 1.0,
    min_names: int = 20,
) -> pd.Series:
    """Residualize a factor against same-date Alpha158 ranks.

    Uses a small ridge regression per date to avoid unstable OLS when the
    exposure count is close to the cross-section size.
    """
    if "factor" not in panel.columns:
        raise ValueError("panel needs a factor column")
    common = panel.index.intersection(exposures.index)
    factor = panel.loc[common, "factor"]
    exp = exposures.loc[common]
    output: list[pd.Series] = []
    for date, y in factor.groupby(level="datetime", sort=True):
        x = exp.xs(date, level="datetime")
        y_values = y.droplevel("datetime")
        valid = y_values.notna() & x.notna().all(axis=1)
        if valid.sum() < max(min_names, x.shape[1] + 2):
            continue
        x_valid = x.loc[valid].to_numpy(dtype=float)
        y_valid = y_values.loc[valid].to_numpy(dtype=float)
        x_rank = _rank_rows(x_valid)
        y_rank = pd.Series(y_valid).rank(pct=True).to_numpy(dtype=float)
        design = np.column_stack([np.ones(len(x_rank)), x_rank])
        gram = design.T @ design
        gram.flat[:: gram.shape[0] + 1] += ridge_alpha
        try:
            beta = np.linalg.solve(gram, design.T @ y_rank)
        except np.linalg.LinAlgError:
            continue
        residual = y_rank - design @ beta
        index = pd.MultiIndex.from_product(
            [valid[valid].index, [date]], names=["instrument", "datetime"]
        )
        output.append(pd.Series(residual, index=index))
    if not output:
        return pd.Series(dtype=float, name="factor")
    result = pd.concat(output).sort_index()
    result.name = "factor"
    return result


def _rank_rows(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=float)
    for column in range(values.shape[1]):
        out[:, column] = pd.Series(values[:, column]).rank(pct=True).to_numpy(dtype=float)
    return out


def incremental_ic_table(
    candidates: Iterable[Candidate],
    panels: SamplePanels,
    exposures: Alpha158Exposures,
    config: ExposureConfig,
) -> pd.DataFrame:
    ranked_exposures = exposures.ranked()
    rows = []
    for candidate in candidates:
        try:
            panel = panels.panel(candidate.canonical)
            merged = panel.join(panels.labels, how="inner").sort_index()
            pre_rank_ic, _ = compute_ic(
                merged, "factor", "fwd_1", min_names=config.min_names
            )
            pre_stats = ic_stats(pre_rank_ic)
            residual = orthogonalize_factor(
                panel,
                ranked_exposures,
                ridge_alpha=config.ridge_alpha,
                min_names=config.min_names,
            ).to_frame("factor")
            post_panel = residual.join(panels.labels, how="inner").sort_index()
            post_rank_ic, _ = compute_ic(
                post_panel, "factor", "fwd_1", min_names=config.min_names
            )
            post_stats = ic_stats(post_rank_ic)
            rows.append(
                {
                    "name": candidate.name,
                    "canonical": candidate.canonical,
                    "source": candidate.source,
                    "pre_rank_ic": pre_stats.get("mean"),
                    "pre_icir": pre_stats.get("icir"),
                    "pre_n_dates": pre_stats.get("n_dates"),
                    "orth_rank_ic": post_stats.get("mean"),
                    "orth_icir": post_stats.get("icir"),
                    "orth_n_dates": post_stats.get("n_dates"),
                    "incremental_ic": post_stats.get("mean"),
                    "delta_abs_ic": (
                        abs(post_stats.get("mean") or 0.0)
                        - abs(pre_stats.get("mean") or 0.0)
                    ),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "name": candidate.name,
                    "canonical": candidate.canonical,
                    "source": candidate.source,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)
