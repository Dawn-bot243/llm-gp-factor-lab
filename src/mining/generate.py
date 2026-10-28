"""Batch generation loop: call an LLM, parse, validate and de-duplicate."""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.factors.expression import ExpressionError, validate_expression
from src.mining.ledger import BatchRecord, CandidateRecord, TrialLedger
from src.mining.llm_client import BaseLLMClient, LLMBudgetExceeded
from src.mining.prompts import CATEGORIES, SYSTEM_PROMPT, build_user_prompt
from src.mining.schemas import FactorCandidate, parse_candidates


@dataclass
class GenerationStats:
    requested: int = 0
    returned: int = 0
    schema_valid: int = 0
    expression_valid: int = 0
    accepted: int = 0
    duplicates: int = 0
    invalid_schema: int = 0
    invalid_expression: int = 0
    invalid_json_batches: int = 0
    client_errors: int = 0
    calls: int = 0
    stopped_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def legal_rate(self) -> float:
        return self.expression_valid / self.returned if self.returned else 0.0

    @property
    def unique_rate(self) -> float:
        return self.accepted / self.returned if self.returned else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "returned": self.returned,
            "schema_valid": self.schema_valid,
            "expression_valid": self.expression_valid,
            "accepted": self.accepted,
            "duplicates": self.duplicates,
            "invalid_schema": self.invalid_schema,
            "invalid_expression": self.invalid_expression,
            "invalid_json_batches": self.invalid_json_batches,
            "client_errors": self.client_errors,
            "calls": self.calls,
            "legal_rate": self.legal_rate,
            "unique_rate": self.unique_rate,
            "stopped_reason": self.stopped_reason,
            "usage": self.usage,
        }


@dataclass
class GenerationResult:
    records: list[CandidateRecord]
    batches: list[BatchRecord]
    stats: GenerationStats


def generate_candidates(
    client: BaseLLMClient,
    *,
    n: int,
    batch_size: int,
    prompt_version: str,
    existing_canonicals: Iterable[str] | None = None,
    ledger: TrialLedger | None = None,
    categories: Iterable[str] = CATEGORIES,
    max_attempts: int | None = None,
    run_id: str | None = None,
) -> GenerationResult:
    """Generate up to ``n`` unique, expression-valid candidates."""
    if n <= 0:
        raise ValueError("n must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    run_id = run_id or datetime.now(UTC).astimezone().strftime("stage5-%Y%m%d-%H%M%S")
    category_list = list(categories) or list(CATEGORIES)
    existing = set(existing_canonicals or set())
    max_attempts = max_attempts or max(1, math.ceil(n / batch_size) * 3)
    records: list[CandidateRecord] = []
    batches: list[BatchRecord] = []
    stats = GenerationStats(requested=n)

    for attempt in range(max_attempts):
        if stats.accepted >= n:
            break
        request_n = min(batch_size, n - stats.accepted)
        focus = category_list[attempt % len(category_list)]
        batch_id = f"{run_id}-{attempt:03d}"
        user_prompt = build_user_prompt(request_n, focus, sorted(existing))
        try:
            response = client.complete(SYSTEM_PROMPT, user_prompt)
        except LLMBudgetExceeded as exc:
            stats.stopped_reason = f"budget: {exc}"
            break
        except Exception as exc:
            stats.calls += 1
            stats.client_errors += 1
            message = f"client: {type(exc).__name__}: {exc}"
            batch = BatchRecord(
                batch_id=batch_id,
                focus=focus,
                prompt_version=prompt_version,
                model=client.config.resolved_model,
                temperature=client.config.temperature,
                response_text="",
                usage=client.total_usage.to_dict(),
                error=message,
            )
            batches.append(batch)
            if ledger:
                ledger.write_batch(batch)
            continue

        candidates, schema_errors = parse_candidates(response.text)
        stats.calls += 1
        stats.returned += len(candidates) + len(schema_errors)
        stats.schema_valid += len(candidates)
        stats.invalid_schema += len(schema_errors)
        if schema_errors and not candidates:
            stats.invalid_json_batches += 1

        batch_records: list[CandidateRecord] = []
        expression_valid_in_batch = 0
        for index, candidate in enumerate(candidates):
            raw = candidate.expression
            try:
                validated = validate_expression(raw)
            except ExpressionError as exc:
                stats.invalid_expression += 1
                batch_records.append(
                    _record_from_candidate(
                        batch_id,
                        index,
                        candidate,
                        schema_status="valid",
                        validation_status="invalid_expression",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            expression_valid_in_batch += 1
            stats.expression_valid += 1
            if validated.canonical in existing:
                stats.duplicates += 1
                batch_records.append(
                    _record_from_candidate(
                        batch_id,
                        index,
                        candidate,
                        schema_status="valid",
                        validation_status="duplicate",
                        canonical=validated.canonical,
                        expression_hash=validated.expression_hash,
                        error="duplicate canonical expression",
                    )
                )
                continue
            existing.add(validated.canonical)
            stats.accepted += 1
            batch_records.append(
                _record_from_candidate(
                    batch_id,
                    index,
                    candidate,
                    schema_status="valid",
                    validation_status="valid",
                    canonical=validated.canonical,
                    expression_hash=validated.expression_hash,
                )
            )

        batch = BatchRecord(
            batch_id=batch_id,
            focus=focus,
            prompt_version=prompt_version,
            model=client.config.resolved_model,
            temperature=client.config.temperature,
            response_text=response.text,
            usage=response.usage.to_dict(),
            n_returned=len(candidates) + len(schema_errors),
            n_schema_valid=len(candidates),
            n_expression_valid=expression_valid_in_batch,
            error="; ".join(schema_errors) if schema_errors else None,
        )
        batches.append(batch)
        records.extend(batch_records)
        if ledger:
            ledger.write_batch(batch)
            ledger.write_trials(batch_records)

    stats.usage = client.total_usage.to_dict()
    if stats.accepted < n and not stats.stopped_reason:
        stats.stopped_reason = f"max_attempts={max_attempts} reached"
    return GenerationResult(records=records, batches=batches, stats=stats)


def _record_from_candidate(
    batch_id: str,
    index: int,
    candidate: FactorCandidate,
    *,
    schema_status: str,
    validation_status: str,
    canonical: str | None = None,
    expression_hash: str | None = None,
    error: str | None = None,
) -> CandidateRecord:
    return CandidateRecord(
        batch_id=batch_id,
        index=index,
        name=candidate.name,
        category=candidate.category,
        hypothesis=candidate.hypothesis,
        raw_expression=candidate.expression,
        expected_sign=candidate.expected_sign,
        holding_period=candidate.holding_period,
        rationale=candidate.rationale,
        schema_status=schema_status,
        validation_status=validation_status,
        canonical=canonical,
        expression_hash=expression_hash,
        error=error,
    )
