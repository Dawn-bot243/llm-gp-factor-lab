"""Pydantic schema and tolerant JSON parsing for LLM factor candidates."""
from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FactorCandidate(BaseModel):
    """One LLM-proposed factor hypothesis."""

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=64)
    category: str = Field(default="other", max_length=64)
    hypothesis: str = Field(min_length=1, max_length=2_000)
    expression: str = Field(min_length=1, max_length=2_000)
    expected_sign: Literal["positive", "negative"]
    holding_period: int
    rationale: str = Field(default="", max_length=2_000)

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        value = re.sub(r"\s+", "_", value.strip())
        return value[:64] or "factor"

    @field_validator("category", mode="before")
    @classmethod
    def _clean_category(cls, value: Any) -> str:
        return str(value or "other").strip().lower()[:64] or "other"

    @field_validator("expected_sign", mode="before")
    @classmethod
    def _normalize_sign(cls, value: Any) -> str:
        text = str(value or "").strip().lower()
        aliases = {
            "+": "positive",
            "long": "positive",
            "up": "positive",
            "-": "negative",
            "short": "negative",
            "down": "negative",
        }
        text = aliases.get(text, text)
        if text not in {"positive", "negative"}:
            raise ValueError("expected_sign must be 'positive' or 'negative'")
        return text

    @field_validator("holding_period", mode="before")
    @classmethod
    def _parse_holding_period(cls, value: Any) -> int:
        if isinstance(value, str):
            match = re.search(r"-?\d+", value)
            if not match:
                raise ValueError("holding_period must contain an integer")
            value = int(match.group())
        # Holding period is metadata for the idea; S1 uses the 1-day label.
        # Clamp instead of discarding an otherwise valid factor when the model
        # ignores the documented 1-20 range.
        return max(1, min(20, int(value)))


class CandidateBatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidates: list[FactorCandidate]


def parse_candidates(text: str) -> tuple[list[FactorCandidate], list[str]]:
    """Parse a model response, keeping valid candidates and error messages."""
    payload, error = _load_json_payload(text)
    if error:
        return [], [error]
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
        rows = payload["candidates"]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        return [], ["response JSON is not an object or array"]

    candidates: list[FactorCandidate] = []
    errors: list[str] = []
    for index, row in enumerate(rows):
        try:
            candidates.append(FactorCandidate.model_validate(row))
        except Exception as exc:
            errors.append(f"candidate[{index}]: {type(exc).__name__}: {exc}")
    return candidates, errors


def _load_json_payload(text: str) -> tuple[Any | None, str | None]:
    if not text or not text.strip():
        return None, "empty response"
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    starts = [pos for pos in (cleaned.find("{"), cleaned.find("[")) if pos >= 0]
    if not starts:
        return None, "no JSON object or array found"
    start = min(starts)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if end <= start:
        return None, "unterminated JSON payload"
    snippet = cleaned[start : end + 1]
    try:
        return json.loads(snippet), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
