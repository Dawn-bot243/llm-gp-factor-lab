#!/usr/bin/env python
"""Stage 7: de-duplicate candidates and residualize against Alpha158.

Examples:
    .venv/Scripts/python.exe scripts/dedupe_factors.py
    .venv/Scripts/python.exe scripts/dedupe_factors.py --no-orthogonalize
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.cluster.hierarchy import dendrogram

from src.evaluation.dedupe import (
    SamplePanels,
    cluster_from_similarity,
    correlation_clusters,
    load_candidates,
    rank_matrix,
    representative_table,
    semantic_similarity,
    semantic_vs_correlation_pairs,
)
from src.evaluation.orthogonalize import Alpha158Exposures, ExposureConfig, incremental_ic_table
from src.factors.serialization import json_default, jsonable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "stage7.yaml"))
    parser.add_argument("--baseline-config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument(
        "--stage5-scorecard",
        default=str(PROJECT_ROOT / "data" / "mining" / "stage5_combined_scorecard.csv"),
    )
    parser.add_argument(
        "--gp-hall",
        default=str(PROJECT_ROOT / "data" / "mining" / "gp_run_full_v2" / "hall_of_fame.csv"),
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts" / "stage7"))
    parser.add_argument("--assets-dir", default=str(PROJECT_ROOT / "项目报告" / "assets"))
    parser.add_argument("--corr-threshold", type=float, default=None)
    parser.add_argument("--semantic-threshold", type=float, default=None)
    parser.add_argument("--alpha158-count", type=int, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--no-orthogonalize", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> int:
    args = parse_args()
    payload = _yaml(args.config)
    dedupe_cfg = payload.get("dedupe", {})
    orth_cfg = payload.get("orthogonalize", {})
    baseline = _yaml(args.baseline_config).get("data", {})
    provider = baseline.get("provider_uri")
    if not provider:
        raise SystemExit("provider_uri is missing in the baseline config")
    membership = baseline.get("pit_membership") or str(PROJECT_ROOT / "data" / "processed" / "pit_csi300_membership.parquet")
    if not Path(membership).is_absolute():
        membership = str(PROJECT_ROOT / membership)

    corr_threshold = args.corr_threshold or float(dedupe_cfg.get("corr_threshold", 0.7))
    semantic_threshold = args.semantic_threshold or float(dedupe_cfg.get("semantic_threshold", 0.6))
    sample_size = args.sample_size or int(dedupe_cfg.get("sample_size", 100))
    candidates = load_candidates(args.stage5_scorecard, args.gp_hall)
    if args.limit:
        candidates = candidates[: args.limit]
    if len(candidates) < 2:
        raise SystemExit("need at least two candidates")

    panels = SamplePanels(
        provider_uri=provider,
        membership_path=membership,
        sample_date=str(dedupe_cfg.get("sample_date", "2017-06-30")),
        sample_size=sample_size,
        start=str(dedupe_cfg.get("start", "2017-01-01")),
        end=str(dedupe_cfg.get("end", "2018-12-31")),
    )
    loaded: dict[str, pd.DataFrame] = {}
    usable = []
    for candidate in candidates:
        try:
            loaded[candidate.canonical] = panels.panel(candidate.canonical)
            usable.append(candidate)
        except Exception as exc:
            print(f"skip {candidate.name}: {type(exc).__name__}: {exc}")
    if len(usable) < 2:
        raise SystemExit("fewer than two usable factor panels")
    candidates = usable

    corr = rank_matrix({candidate.canonical: loaded[candidate.canonical] for candidate in candidates})
    labels, linkage_matrix = correlation_clusters(corr, corr_threshold)
    reps = representative_table(candidates, labels, corr)
    similarity = semantic_similarity([candidate.expression for candidate in candidates])
    pairs = semantic_vs_correlation_pairs(
        similarity, corr, [candidate.canonical for candidate in candidates]
    )
    semantic_labels = cluster_from_similarity(similarity, semantic_threshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = Path(args.assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    corr.to_csv(output_dir / "correlation_matrix.csv")
    reps.to_csv(output_dir / "representatives.csv", index=False)
    pairs.to_csv(output_dir / "semantic_vs_correlation_pairs.csv", index=False)
    pd.DataFrame(
        {
            "canonical": [candidate.canonical for candidate in candidates],
            "name": [candidate.name for candidate in candidates],
            "source": [candidate.source for candidate in candidates],
            "score": [candidate.score for candidate in candidates],
            "corr_cluster": labels,
            "semantic_cluster": semantic_labels,
        }
    ).to_csv(output_dir / "clusters.csv", index=False)
    _save_heatmap(corr, assets_dir / "stage7_correlation_heatmap.png")
    _save_dendrogram(
        linkage_matrix, [candidate.name for candidate in candidates], assets_dir / "stage7_dendrogram.png"
    )

    incremental = pd.DataFrame()
    if not args.no_orthogonalize:
        exposure_config = ExposureConfig(
            alpha158_count=args.alpha158_count or int(orth_cfg.get("alpha158_count", 20)),
            ridge_alpha=args.ridge_alpha or float(orth_cfg.get("ridge_alpha", 1.0)),
            min_names=int(dedupe_cfg.get("min_names", 20)),
        )
        exposures = Alpha158Exposures(panels, exposure_config)
        rep_candidates = [
            next(candidate for candidate in candidates if candidate.canonical == canonical)
            for canonical in reps["representative_canonical"]
        ]
        incremental = incremental_ic_table(rep_candidates, panels, exposures, exposure_config)
        incremental.to_csv(output_dir / "incremental_ic.csv", index=False)

    off_diag = corr.abs().where(~np.eye(len(corr), dtype=bool))
    summary = {
        "schema_version": "stage7.dedupe.v1",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "n_candidates": len(candidates),
        "n_correlation_clusters": len(set(labels)),
        "n_representatives": len(reps),
        "n_semantic_clusters": len(set(semantic_labels)),
        "corr_threshold": corr_threshold,
        "semantic_threshold": semantic_threshold,
        "n_semantic_vs_corr_pairs": len(pairs),
        "n_incremental_rows": len(incremental),
        "mean_abs_correlation": float(off_diag.stack().mean()),
        "sources": {
            "stage5": sum(1 for item in candidates if item.source in {"stage5", "both"}),
            "gp_hall": sum(1 for item in candidates if item.source in {"gp_hall", "both"}),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=json_default))
    return 0


def _save_heatmap(corr: pd.DataFrame, path: Path) -> None:
    size = max(8.0, len(corr) * 0.16)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=5)
    ax.set_yticklabels(corr.index, fontsize=5)
    ax.set_title("Stage 7 factor Spearman correlation")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_dendrogram(linkage_matrix: np.ndarray, labels: list[str], path: Path) -> None:
    if len(linkage_matrix) == 0:
        return
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.2), 6))
    dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=6)
    ax.set_title("Stage 7 correlation clustering")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
