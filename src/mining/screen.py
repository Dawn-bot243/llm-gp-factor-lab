"""Fast S1 screening for Stage 5 LLM candidates."""
from __future__ import annotations

from typing import Any

from src.factors.evaluator import EvaluationConfig, load_factor
from src.factors.s1 import ForwardReturns, S1Config, S1Result, apply_fdr, evaluate_s1
from src.mining.ledger import CandidateRecord


def screen_records(
    records: list[CandidateRecord],
    data_config: EvaluationConfig,
    s1_config: S1Config,
    *,
    max_candidates: int | None = None,
) -> list[S1Result]:
    """Evaluate valid candidates with S1, updating records in place."""
    valid = [record for record in records if record.validation_status == "valid" and record.canonical]
    if max_candidates is not None:
        valid = valid[:max_candidates]
    if not valid:
        return []

    forward = ForwardReturns.from_config(data_config, s1_config.decay_horizons)
    results: list[S1Result] = []
    for index, record in enumerate(valid, start=1):
        try:
            validated, factor = load_factor(record.raw_expression, data_config)
            result = evaluate_s1(
                record.raw_expression,
                data_config,
                s1_config,
                forward,
                name=record.name or validated.canonical,
                validated=validated,
                factor=factor,
            )
        except Exception as exc:
            record.s1_status = "error"
            record.error = f"S1: {type(exc).__name__}: {exc}"
            print(f"[screen {index}/{len(valid)}] {record.name}: error ({exc})")
            continue
        record.s1_status = result.scorecard["status"]
        record.s1_rank_ic = result.scorecard.get("rank_ic_mean")
        record.s1_icir = result.scorecard.get("icir")
        record.s1_q_value = result.scorecard.get("q_value")
        results.append(result)
        print(
            f"[screen {index}/{len(valid)}] {record.name}: {record.s1_status} "
            f"RankIC={record.s1_rank_ic if record.s1_rank_ic is not None else float('nan'):.4f}"
        )

    apply_fdr(results, alpha=s1_config.thresholds.fdr_alpha)
    by_hash = {result.scorecard["expression_hash"]: result for result in results}
    for record in valid:
        matched = by_hash.get(record.expression_hash or "")
        if matched is None:
            continue
        record.s1_status = matched.scorecard["status"]
        record.s1_rank_ic = matched.scorecard.get("rank_ic_mean")
        record.s1_icir = matched.scorecard.get("icir")
        record.s1_q_value = matched.scorecard.get("q_value")
    return results


def screening_summary(results: list[S1Result]) -> dict[str, Any]:
    counts = {"pass": 0, "review": 0, "fail": 0, "error": 0}
    for result in results:
        status = result.scorecard.get("status", "error")
        counts[status] = counts.get(status, 0) + 1
    return {
        "n_screened": len(results),
        **counts,
        "fdr_alpha": results[0].scorecard.get("thresholds", {}).get("fdr_alpha") if results else None,
    }
