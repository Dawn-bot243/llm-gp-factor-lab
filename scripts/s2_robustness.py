#!/usr/bin/env python
"""Stage 8: S2 robustness for Stage 7 representatives and known factors.

Examples:
    .venv/Scripts/python.exe scripts/s2_robustness.py
    .venv/Scripts/python.exe scripts/s2_robustness.py --limit 3
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

import pandas as pd
import yaml

from src.evaluation.dedupe import Candidate
from src.evaluation.robustness import RobustnessEvaluator, S2Config, apply_fdr
from src.factors.evaluator import (
    DEFAULT_CONSTRAINTS,
    DEFAULT_MEMBERSHIP,
    DEFAULT_SNAPSHOT,
    EvaluationConfig,
)
from src.factors.serialization import json_default, jsonable

KNOWN = [
    Candidate("MOM20", "$close / Ref($close, 20) - 1", "$close / Ref($close, 20) - 1", "known", 0.0),
    Candidate("REV5", "$close / Ref($close, 5) - 1", "$close / Ref($close, 5) - 1", "known", 0.0),
    Candidate("LOWVOL20", "-1 * Std($close, 20) / $close", "-1 * Std($close, 20) / $close", "known", 0.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "gates.yaml"))
    parser.add_argument("--baseline-config", default=str(PROJECT_ROOT / "configs" / "baseline.yaml"))
    parser.add_argument(
        "--representatives",
        default=str(PROJECT_ROOT / "artifacts" / "stage7" / "representatives.csv"),
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "artifacts" / "stage8"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-known", action="store_true")
    parser.add_argument("--placebo-iterations", type=int, default=None)
    return parser.parse_args()


def _yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _candidates(path: str) -> list[Candidate]:
    p = Path(path)
    if not p.exists():
        return []
    frame = pd.read_csv(p, keep_default_na=False)
    out = []
    for _, row in frame.iterrows():
        canonical = str(row.get("representative_canonical") or "").strip()
        if not canonical:
            continue
        out.append(
            Candidate(
                name=str(row.get("representative") or canonical),
                expression=canonical,
                canonical=canonical,
                source="stage7_rep",
                score=float(row.get("representative_score") or 0.0),
            )
        )
    return out


def main() -> int:
    args = parse_args()
    payload = _yaml(args.config)
    config = S2Config.from_dict(payload.get("s2", payload))
    if args.placebo_iterations is not None:
        from dataclasses import replace

        config = replace(config, placebo_iterations=args.placebo_iterations)
    baseline = _yaml(args.baseline_config).get("data", {})
    provider = baseline.get("provider_uri")
    if not provider:
        raise SystemExit("provider_uri missing in baseline config")
    membership = baseline.get("pit_membership") or str(DEFAULT_MEMBERSHIP)
    if not Path(membership).is_absolute():
        membership = str(PROJECT_ROOT / membership)
    data_config = EvaluationConfig(
        provider_uri=provider,
        instruments=baseline.get("instruments", "csi300"),
        start="2010-01-01",
        end=baseline.get("end_time", "2020-09-23"),
        membership_path=membership,
        constraints_path=str(DEFAULT_CONSTRAINTS),
        snapshot_file=str(DEFAULT_SNAPSHOT),
    )
    candidates = [] if args.no_known else list(KNOWN)
    candidates.extend(_candidates(args.representatives))
    if args.limit:
        candidates = candidates[: args.limit]
    if not candidates:
        raise SystemExit("no candidates")

    evaluator = RobustnessEvaluator(data_config, config)
    rows = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] {candidate.name}")
        try:
            rows.append(evaluator.evaluate(candidate))
        except Exception as exc:
            rows.append(
                {
                    "name": candidate.name,
                    "canonical": candidate.canonical,
                    "source": candidate.source,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    apply_fdr([row for row in rows if "error" not in row], alpha=config.gates.max_fdr_q)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    flat = [_flatten(row) for row in rows]
    frame = pd.DataFrame(flat)
    frame.to_csv(output_dir / "s2_scorecard.csv", index=False)
    frame.to_parquet(output_dir / "s2_scorecard.parquet", index=False)
    (output_dir / "s2_details.json").write_text(
        json.dumps(jsonable(rows), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    summary = {
        "schema_version": "stage8.s2.v1",
        "generated_at": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
        "config": config.to_dict(),
        "n_candidates": len(rows),
        "n_pass": sum(1 for row in rows if row.get("s2", {}).get("status") == "pass"),
        "n_review": sum(1 for row in rows if row.get("s2", {}).get("status") == "review"),
        "n_fail": sum(1 for row in rows if row.get("s2", {}).get("status") == "fail"),
        "rows": flat,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    print(
        json.dumps(
            jsonable({key: summary[key] for key in ("n_candidates", "n_pass", "n_review", "n_fail")}),
            ensure_ascii=False,
        )
    )
    return 0 if rows else 1


def _flatten(row: dict[str, Any]) -> dict[str, Any]:
    wf = row.get("walk_forward") or {}
    placebo = row.get("placebo") or {}
    sensitivity = row.get("sensitivity") or {}
    s2 = row.get("s2") or {}
    return {
        "name": row.get("name"),
        "canonical": row.get("canonical"),
        "source": row.get("source"),
        "full_rank_ic": row.get("full_rank_ic"),
        "full_icir": row.get("full_icir"),
        "direction": row.get("direction"),
        "wf_folds": wf.get("n_folds"),
        "wf_positive_ratio": wf.get("positive_ratio"),
        "wf_mean_signed_ic": wf.get("mean_signed_ic"),
        "wf_sign_consistency": wf.get("sign_consistency"),
        "placebo_p": placebo.get("p_value"),
        "fdr_q": row.get("fdr_q"),
        "dsr": row.get("dsr"),
        "sensitivity_cv": sensitivity.get("cv"),
        "sensitivity_sign_flip": sensitivity.get("sign_flip"),
        "s2_status": s2.get("status"),
        "s2_reasons": ";".join(s2.get("reasons") or []),
        "error": row.get("error"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
