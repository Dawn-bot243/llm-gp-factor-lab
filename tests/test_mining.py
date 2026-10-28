from __future__ import annotations

import json

import pytest

from src.mining.generate import generate_candidates
from src.mining.ledger import TrialLedger
from src.mining.llm_client import (
    BaseLLMClient,
    LLMConfig,
    LLMConfigError,
    LLMError,
    LLMResponse,
    LLMUsage,
    MockClient,
    make_client,
)
from src.mining.prompts import SYSTEM_PROMPT, build_user_prompt, prompt_hash
from src.mining.schemas import FactorCandidate, parse_candidates


def _candidate_payload(expression: str = "$close / Ref($close, 5) - 1") -> str:
    return json.dumps(
        {
            "candidates": [
                {
                    "name": "TEST",
                    "category": "momentum",
                    "hypothesis": "test hypothesis",
                    "expression": expression,
                    "expected_sign": "positive",
                    "holding_period": 5,
                    "rationale": "test",
                }
            ]
        }
    )


def test_parse_candidates_accepts_fenced_json_and_normalizes_sign():
    text = "```json\n" + _candidate_payload().replace('"positive"', '"+"') + "\n```"
    candidates, errors = parse_candidates(text)
    assert not errors
    assert len(candidates) == 1
    assert candidates[0].expected_sign == "positive"


def test_parse_candidates_reports_invalid_schema_and_json():
    candidates, errors = parse_candidates("not json")
    assert candidates == []
    assert errors
    bad = _candidate_payload().replace('"positive"', '"sideways"')
    candidates, errors = parse_candidates(bad)
    assert candidates == []
    assert errors


def test_factor_candidate_clamps_holding_period():
    payload = json.loads(_candidate_payload())
    payload["candidates"][0]["holding_period"] = "10d"
    candidate = FactorCandidate.model_validate(payload["candidates"][0])
    assert candidate.holding_period == 10
    payload["candidates"][0]["holding_period"] = 60
    assert FactorCandidate.model_validate(payload["candidates"][0]).holding_period == 20
    payload["candidates"][0]["holding_period"] = 0
    assert FactorCandidate.model_validate(payload["candidates"][0]).holding_period == 1


def test_llm_config_does_not_expose_key(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "secret-value")
    config = LLMConfig(provider="openai", api_key_env="TEST_LLM_KEY")
    assert "secret-value" not in json.dumps(config.to_dict())
    assert config.resolved_api_key_env == "TEST_LLM_KEY"
    assert config.api_key() == "secret-value"


def test_mock_client_is_deterministic_and_valid():
    client = MockClient(LLMConfig(provider="mock", max_calls=10))
    response = client.complete(SYSTEM_PROMPT, build_user_prompt(3, "momentum", []))
    candidates, errors = parse_candidates(response.text)
    assert len(candidates) == 3
    assert not errors
    assert response.usage.n_calls == 1
    client.close()


class _FakeClient(BaseLLMClient):
    def __init__(self, texts: list[str]) -> None:
        super().__init__(LLMConfig(provider="mock", max_calls=10))
        self.texts = texts
        self.index = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        text = self.texts[min(self.index, len(self.texts) - 1)]
        self.index += 1
        return self._record(
            LLMResponse(text=text, usage=LLMUsage(input_tokens=10, output_tokens=10)),
            system,
            user,
        )


def test_generate_records_invalid_expression_and_duplicate():
    valid = _candidate_payload()
    invalid = _candidate_payload("Ref($close, -1)")
    client = _FakeClient([invalid, valid, valid])
    result = generate_candidates(
        client,
        n=2,
        batch_size=1,
        prompt_version="test",
        max_attempts=3,
    )
    assert result.stats.accepted == 1
    assert result.stats.invalid_expression == 1
    assert result.stats.duplicates == 1
    assert result.stats.legal_rate == pytest.approx(2 / 3)
    client.close()


def test_generate_writes_ledger(tmp_path):
    ledger = TrialLedger(tmp_path)
    client = MockClient(LLMConfig(provider="mock", max_calls=10))
    result = generate_candidates(
        client,
        n=4,
        batch_size=2,
        prompt_version="test",
        ledger=ledger,
    )
    assert result.stats.accepted == 4
    assert ledger.batch_path.exists()
    assert ledger.trial_path.exists()
    assert len(ledger.existing_canonicals()) >= 1
    client.close()


def test_make_client_rejects_unknown_provider():
    with pytest.raises(LLMConfigError):
        make_client(LLMConfig(provider="unknown"))


def test_prompt_hash_is_stable():
    assert prompt_hash() == prompt_hash()
    assert "candidates" in SYSTEM_PROMPT


class _FlakyClient(_FakeClient):
    def complete(self, system: str, user: str) -> LLMResponse:
        if self.index == 0:
            self.index += 1
            raise LLMError("temporary no-text response")
        return super().complete(system, user)


def test_generate_continues_after_client_error():
    client = _FlakyClient([_candidate_payload()])
    result = generate_candidates(
        client,
        n=1,
        batch_size=1,
        prompt_version="test",
        max_attempts=3,
    )
    assert result.stats.client_errors == 1
    assert result.stats.accepted == 1
    client.close()
