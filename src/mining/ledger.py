"""JSONL trial ledger and candidate records for Stage 5."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.factors.serialization import json_default, jsonable


@dataclass
class CandidateRecord:
    batch_id: str
    index: int
    name: str | None
    category: str | None
    hypothesis: str | None
    raw_expression: str
    expected_sign: str | None
    holding_period: int | None
    rationale: str | None
    schema_status: str
    validation_status: str
    canonical: str | None = None
    expression_hash: str | None = None
    error: str | None = None
    s1_status: str | None = None
    s1_rank_ic: float | None = None
    s1_icir: float | None = None
    s1_q_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchRecord:
    batch_id: str
    focus: str
    prompt_version: str
    model: str
    temperature: float
    response_text: str
    usage: dict[str, Any] = field(default_factory=dict)
    n_returned: int = 0
    n_schema_valid: int = 0
    n_expression_valid: int = 0
    error: str | None = None
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).astimezone().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrialLedger:
    """Append-only JSONL ledger; safe to resume from."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.batch_path = self.root / "batches.jsonl"
        self.trial_path = self.root / "trials.jsonl"
        self.screened_path = self.root / "screened_trials.jsonl"

    def write_batch(self, batch: BatchRecord) -> None:
        self._append(self.batch_path, batch.to_dict())

    def write_trials(self, records: list[CandidateRecord]) -> None:
        for record in records:
            self._append(self.trial_path, record.to_dict())

    def write_screened(self, records: list[CandidateRecord]) -> None:
        with self.screened_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(jsonable(record.to_dict()), ensure_ascii=False, default=json_default))
                handle.write("\n")

    def existing_canonicals(self) -> set[str]:
        if not self.trial_path.exists():
            return set()
        canonicals: set[str] = set()
        for line in self.trial_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("canonical"):
                canonicals.add(str(row["canonical"]))
        return canonicals

    @staticmethod
    def _append(path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(jsonable(payload), ensure_ascii=False, default=json_default))
            handle.write("\n")
