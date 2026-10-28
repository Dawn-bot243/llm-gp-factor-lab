"""Candidate loading, semantic proxy similarity and factor-value clustering."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMBERSHIP = PROJECT_ROOT / "data" / "processed" / "pit_csi300_membership.parquet"
_QLIB_PROVIDER: str | None = None


@dataclass
class Candidate:
    name: str
    expression: str
    canonical: str
    source: str
    score: float
    extra: dict[str, Any] = field(default_factory=dict)


def load_candidates(
    stage5_path: str | Path,
    gp_hall_path: str | Path,
    *,
    stage5_statuses: tuple[str, ...] = ("pass", "review"),
) -> list[Candidate]:
    """Load Stage 5 pass/review and Stage 6 hall candidates, exact-deduped."""
    by_canonical: dict[str, Candidate] = {}
    stage5 = Path(stage5_path)
    if stage5.exists():
        frame = pd.read_csv(stage5, keep_default_na=False)
        if "status" in frame.columns:
            frame = frame[frame["status"].isin(stage5_statuses)]
        for _, row in frame.iterrows():
            canonical = str(row.get("canonical") or "").strip()
            if not canonical:
                continue
            score = abs(float(row.get("rank_ic_mean") or 0.0))
            candidate = Candidate(
                name=str(row.get("name") or canonical),
                expression=str(row.get("expression") or canonical),
                canonical=canonical,
                source="stage5",
                score=score,
                extra={"status": row.get("status"), "rank_ic": row.get("rank_ic_mean")},
            )
            _keep(by_canonical, candidate)

    hall = Path(gp_hall_path)
    if hall.exists():
        frame = pd.read_csv(hall, keep_default_na=False)
        for _, row in frame.iterrows():
            canonical = str(row.get("canonical") or "").strip()
            if not canonical:
                continue
            score = float(row.get("fitness") or 0.0)
            candidate = Candidate(
                name=str(row.get("name") or f"GP_{int(row.get('rank', 0))}"),
                expression=str(row.get("expression") or canonical),
                canonical=canonical,
                source="gp_hall",
                score=score,
                extra={"origin": row.get("origin"), "rank_ic": row.get("rank_ic")},
            )
            _keep(by_canonical, candidate)
    return sorted(by_canonical.values(), key=lambda item: item.score, reverse=True)


def _keep(store: dict[str, Candidate], candidate: Candidate) -> None:
    previous = store.get(candidate.canonical)
    if previous is None:
        store[candidate.canonical] = candidate
        return
    if candidate.score > previous.score:
        candidate.source = "both" if previous.source != candidate.source else candidate.source
        store[candidate.canonical] = candidate
    else:
        previous.source = "both" if previous.source != candidate.source else previous.source


class SamplePanels:
    """Load factor values on the fixed Stage 6/7 review sample."""

    def __init__(
        self,
        provider_uri: str,
        membership_path: str | Path = DEFAULT_MEMBERSHIP,
        sample_date: str = "2017-06-30",
        sample_size: int = 100,
        start: str = "2017-01-01",
        end: str = "2018-12-31",
        kernels: int = 4,
    ) -> None:
        self.provider_uri = provider_uri
        self.membership_path = str(membership_path)
        self.sample_date = sample_date
        self.sample_size = sample_size
        self.start = start
        self.end = end
        self.kernels = kernels
        self.names = self._sample_names()
        self._ensure_qlib()
        from qlib.data import D

        labels = D.features(
            self.names,
            ["Ref($close, -2) / Ref($close, -1) - 1"],
            start_time=start,
            end_time=end,
            freq="day",
        )
        if labels is None or labels.empty:
            raise RuntimeError("could not load Stage 7 labels")
        labels.columns = ["fwd_1"]
        labels.index = labels.index.set_names(["instrument", "datetime"])
        self.labels = labels.sort_index()
        self._cache: dict[str, pd.DataFrame] = {}

    def _sample_names(self) -> list[str]:
        frame = pd.read_parquet(self.membership_path)
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        day = frame[frame["datetime"] == pd.Timestamp(self.sample_date)]
        names = sorted(day["instrument"].astype(str).str.upper().unique().tolist())
        if len(names) < self.sample_size:
            raise ValueError(f"only {len(names)} members on {self.sample_date}")
        rng = random.Random(0)
        return sorted(rng.sample(names, self.sample_size))

    def _ensure_qlib(self) -> None:
        global _QLIB_PROVIDER
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
        resolved = str(Path(self.provider_uri).resolve())
        if _QLIB_PROVIDER != resolved:
            import qlib

            qlib.init(provider_uri=resolved, region="cn", logging_level="ERROR")
            _QLIB_PROVIDER = resolved
        from qlib.config import C

        C.joblib_backend = "threading"
        C.kernels = int(self.kernels)

    def panel(self, canonical: str) -> pd.DataFrame:
        if canonical in self._cache:
            return self._cache[canonical]
        from qlib.data import D

        factor = D.features(
            self.names,
            [canonical],
            start_time=self.start,
            end_time=self.end,
            freq="day",
        )
        if factor is None or factor.empty:
            raise RuntimeError(f"empty factor panel: {canonical}")
        factor.columns = ["factor"]
        factor.index = factor.index.set_names(["instrument", "datetime"])
        self._cache[canonical] = factor.sort_index()
        return self._cache[canonical]


def rank_matrix(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a pooled Spearman-like matrix by ranking within each date."""
    series = {name: panel["factor"] for name, panel in panels.items()}
    combined = pd.concat(series, axis=1)
    ranked = combined.groupby(level="datetime").rank(pct=True)
    return ranked.corr()


def semantic_similarity(expressions: list[str]) -> np.ndarray:
    """Text/operator TF-IDF cosine similarity used as a semantic proxy.

    This is intentionally API-free.  It captures expression structure and
    wording, not an LLM embedding; the report calls this out explicitly.
    """
    if not expressions:
        return np.zeros((0, 0))
    if len(expressions) == 1:
        return np.ones((1, 1))
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        lowercase=False,
        min_df=1,
    )
    matrix = vectorizer.fit_transform(expressions)
    return cosine_similarity(matrix)


def cluster_from_similarity(similarity: np.ndarray, threshold: float) -> np.ndarray:
    n = similarity.shape[0]
    if n <= 1:
        return np.ones(n, dtype=int)
    distance = 1.0 - np.clip(similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    return fcluster(linkage_matrix, t=threshold, criterion="distance")


def correlation_clusters(
    correlation: pd.DataFrame,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster on ``1 - |Spearman correlation|``; return labels and linkage."""
    matrix = correlation.abs().to_numpy(dtype=float, copy=True)
    np.fill_diagonal(matrix, 1.0)
    distance = 1.0 - np.clip(matrix, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    if len(correlation) <= 1:
        return np.ones(len(correlation), dtype=int), np.empty((0, 4))
    condensed = squareform(distance, checks=False)
    linkage_matrix = linkage(condensed, method="average")
    labels = fcluster(linkage_matrix, t=threshold, criterion="distance")
    return labels, linkage_matrix


def representative_table(
    candidates: list[Candidate],
    labels: np.ndarray,
    correlation: pd.DataFrame,
) -> pd.DataFrame:
    """Pick the highest-score member of each correlation cluster."""
    names = list(correlation.columns)
    rows = []
    for cluster_id in sorted(set(labels)):
        members = [names[i] for i in range(len(names)) if labels[i] == cluster_id]
        member_candidates = [item for item in candidates if item.canonical in members]
        if not member_candidates:
            continue
        representative = max(member_candidates, key=lambda item: item.score)
        rows.append(
            {
                "cluster": int(cluster_id),
                "size": len(members),
                "representative": representative.name,
                "representative_canonical": representative.canonical,
                "representative_score": representative.score,
                "source": representative.source,
                "members": ";".join(members),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def semantic_vs_correlation_pairs(
    similarity: np.ndarray,
    correlation: pd.DataFrame,
    names: list[str],
    *,
    semantic_high: float = 0.8,
    corr_low: float = 0.3,
    corr_high: float = 0.8,
    semantic_low: float = 0.5,
) -> pd.DataFrame:
    rows = []
    corr_values = correlation.to_numpy(dtype=float)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sem = float(similarity[i, j])
            corr = float(corr_values[i, j])
            if sem >= semantic_high and abs(corr) <= corr_low:
                rows.append(
                    {
                        "kind": "semantic_high_corr_low",
                        "left": names[i],
                        "right": names[j],
                        "semantic": sem,
                        "corr": corr,
                    }
                )
            elif abs(corr) >= corr_high and sem <= semantic_low:
                rows.append(
                    {
                        "kind": "corr_high_semantic_low",
                        "left": names[i],
                        "right": names[j],
                        "semantic": sem,
                        "corr": corr,
                    }
                )
    return pd.DataFrame(rows)
