"""Small OpenAI/Anthropic-compatible LLM client with retry and budget guards.

The client deliberately reads the API key from an environment variable and never
stores or logs it.  ``MockClient`` makes the entire Stage 5 pipeline testable
without network access.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx


class LLMError(RuntimeError):
    """Base class for LLM client errors."""


class LLMBudgetExceeded(LLMError):
    """Raised when the configured call or cost budget would be exceeded."""


class LLMConfigError(LLMError):
    """Raised when provider configuration is incomplete."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "mock"
    base_url: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    auth_style: str = "both"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    max_retries: int = 3
    requests_per_minute: float = 30.0
    max_cost_usd: float | None = 5.0
    max_calls: int | None = 30
    input_price_per_mtok: float | None = None
    output_price_per_mtok: float | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> LLMConfig:
        payload = payload or {}
        return cls(
            provider=str(payload.get("provider", "mock")),
            base_url=payload.get("base_url"),
            model=payload.get("model"),
            api_key_env=payload.get("api_key_env"),
            auth_style=str(payload.get("auth_style", "both")),
            temperature=float(payload.get("temperature", 0.7)),
            max_tokens=int(payload.get("max_tokens", 4096)),
            timeout_seconds=float(payload.get("timeout_seconds", 120.0)),
            max_retries=int(payload.get("max_retries", 3)),
            requests_per_minute=float(payload.get("requests_per_minute", 30.0)),
            max_cost_usd=None if payload.get("max_cost_usd") is None else float(payload["max_cost_usd"]),
            max_calls=None if payload.get("max_calls") is None else int(payload["max_calls"]),
            input_price_per_mtok=(
                None
                if payload.get("input_price_per_mtok") is None
                else float(payload["input_price_per_mtok"])
            ),
            output_price_per_mtok=(
                None
                if payload.get("output_price_per_mtok") is None
                else float(payload["output_price_per_mtok"])
            ),
        )

    @property
    def resolved_base_url(self) -> str:
        if self.base_url:
            return self.base_url
        if self.provider == "anthropic":
            return os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        if self.provider in {"openai", "openai_compatible"}:
            return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return ""

    @property
    def resolved_model(self) -> str:
        if self.model:
            return self.model
        if self.provider == "anthropic":
            return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        if self.provider in {"openai", "openai_compatible"}:
            return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return "mock"

    @property
    def resolved_api_key_env(self) -> str:
        if self.api_key_env:
            return self.api_key_env
        if self.provider == "anthropic":
            for name in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"):
                if os.getenv(name):
                    return name
            return "ANTHROPIC_API_KEY"
        if self.provider in {"openai", "openai_compatible"}:
            return "OPENAI_API_KEY"
        return ""

    def api_key(self) -> str:
        env_name = self.resolved_api_key_env
        value = os.getenv(env_name, "").strip() if env_name else ""
        if not value and self.provider != "mock":
            raise LLMConfigError(
                f"missing API key: set environment variable {env_name or '<provider key>'}"
            )
        return value

    def to_dict(self) -> dict[str, Any]:
        """Configuration without the API key."""
        return {
            "provider": self.provider,
            "base_url": self.resolved_base_url,
            "model": self.resolved_model,
            "api_key_env": self.resolved_api_key_env,
            "auth_style": self.auth_style,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "requests_per_minute": self.requests_per_minute,
            "max_cost_usd": self.max_cost_usd,
            "max_calls": self.max_calls,
            "input_price_per_mtok": self.input_price_per_mtok,
            "output_price_per_mtok": self.output_price_per_mtok,
        }


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    n_calls: int = 0

    def add(self, other: LLMUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.n_calls += other.n_calls
        if other.cost_usd is not None:
            self.cost_usd = (self.cost_usd or 0.0) + other.cost_usd

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "n_calls": self.n_calls,
        }


@dataclass
class LLMResponse:
    text: str
    usage: LLMUsage
    model: str | None = None
    response_id: str | None = None


class BaseLLMClient:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.total_usage = LLMUsage()
        self._last_call = 0.0
        self._client = httpx.Client(timeout=config.timeout_seconds)

    def complete(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError

    def close(self) -> None:
        self._client.close()

    def _before_call(self, system: str, user: str) -> None:
        if self.config.max_calls is not None and self.total_usage.n_calls >= self.config.max_calls:
            raise LLMBudgetExceeded(
                f"max_calls={self.config.max_calls} reached"
            )
        if self.config.requests_per_minute > 0:
            interval = 60.0 / self.config.requests_per_minute
            wait = interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
        self._last_call = time.monotonic()

        if self.config.max_cost_usd is not None and self._has_price():
            estimated_input = max(1, len(system) + len(user)) // 4
            estimated = self._cost(estimated_input, self.config.max_tokens)
            if estimated is not None and (self.total_usage.cost_usd or 0.0) + estimated > self.config.max_cost_usd:
                raise LLMBudgetExceeded(
                    f"estimated call cost would exceed max_cost_usd={self.config.max_cost_usd}"
                )

    def _record(self, response: LLMResponse, system: str, user: str) -> LLMResponse:
        if response.usage.cost_usd is None and self._has_price():
            response.usage.cost_usd = self._cost(
                response.usage.input_tokens, response.usage.output_tokens
            )
        response.usage.n_calls = 1
        self.total_usage.add(response.usage)
        return response

    def _has_price(self) -> bool:
        return (
            self.config.input_price_per_mtok is not None
            and self.config.output_price_per_mtok is not None
        )

    def _cost(self, input_tokens: int, output_tokens: int) -> float | None:
        input_price = self.config.input_price_per_mtok
        output_price = self.config.output_price_per_mtok
        if input_price is None or output_price is None:
            return None
        return (
            input_tokens / 1_000_000 * float(input_price)
            + output_tokens / 1_000_000 * float(output_price)
        )

    def _post_json(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._client.post(url, headers=headers, json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    response.raise_for_status()
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise LLMError("LLM response JSON is not an object")
                return data
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_error = exc
                status = None
                if isinstance(exc, httpx.HTTPStatusError):
                    status = exc.response.status_code
                if status is not None and status in {400, 401, 403, 404}:
                    raise LLMError(f"LLM request failed with HTTP {status}") from exc
                if attempt >= self.config.max_retries:
                    raise LLMError(f"LLM request failed after {attempt + 1} attempts: {exc}") from exc
                time.sleep(min(30.0, 2**attempt))
        raise LLMError(f"LLM request failed: {last_error}")


class AnthropicClient(BaseLLMClient):
    """Anthropic Messages API (also used by Anthropic-compatible gateways)."""

    def complete(self, system: str, user: str) -> LLMResponse:
        self._before_call(system, user)
        key = self.config.api_key()
        base = self.config.resolved_base_url.rstrip("/")
        url = base + ("/messages" if base.endswith("/v1") else "/v1/messages")
        headers = {
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.config.auth_style in {"both", "x-api-key"}:
            headers["x-api-key"] = key
        if self.config.auth_style in {"both", "bearer"}:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": self.config.resolved_model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = self._post_json(url, headers, payload)
        content = data.get("content", [])
        if isinstance(content, str):
            text = content
        else:
            text = "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        usage = data.get("usage") or {}
        response = LLMResponse(
            text=text,
            usage=LLMUsage(
                input_tokens=int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
                output_tokens=int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
            ),
            model=str(data.get("model") or self.config.resolved_model),
            response_id=str(data.get("id") or "") or None,
        )
        self._record(response, system, user)
        if not text.strip():
            raise LLMError(
                "Anthropic response contained no text "
                f"(stop_reason={data.get('stop_reason')})"
            )
        return response


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible chat-completions client."""

    def complete(self, system: str, user: str) -> LLMResponse:
        self._before_call(system, user)
        key = self.config.api_key()
        base = self.config.resolved_base_url.rstrip("/")
        url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        headers = {
            "content-type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        payload = {
            "model": self.config.resolved_model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = self._post_json(url, headers, payload)
        choices = data.get("choices") or []
        if not choices:
            raise LLMError("OpenAI response contained no choices")
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        usage = data.get("usage") or {}
        response = LLMResponse(
            text=text,
            usage=LLMUsage(
                input_tokens=int(usage.get("prompt_tokens", 0) or 0),
                output_tokens=int(usage.get("completion_tokens", 0) or 0),
            ),
            model=str(data.get("model") or self.config.resolved_model),
            response_id=str(data.get("id") or "") or None,
        )
        self._record(response, system, user)
        if not text.strip():
            raise LLMError("OpenAI response contained no text")
        return response


class MockClient(BaseLLMClient):
    """Deterministic offline client used by tests and prompt calibration."""

    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        self._counter = 0

    def complete(self, system: str, user: str) -> LLMResponse:
        self._before_call(system, user)
        import re

        match = re.search(r"Generate\s+(\d+)", user)
        n = int(match.group(1)) if match else 5
        rows = []
        for _ in range(n):
            rows.append(self._candidate(self._counter))
            self._counter += 1
        text = json.dumps({"candidates": rows}, ensure_ascii=False)
        response = LLMResponse(
            text=text,
            usage=LLMUsage(
                input_tokens=max(1, len(system) + len(user)) // 4,
                output_tokens=max(1, len(text)) // 4,
                cost_usd=0.0,
            ),
            model="mock",
            response_id=f"mock-{self._counter}",
        )
        return self._record(response, system, user)

    @staticmethod
    def _candidate(index: int) -> dict[str, Any]:
        windows = (5, 10, 15, 20, 30, 40, 60, 90, 120)
        w = windows[index % len(windows)]
        w2 = windows[(index // len(windows)) % len(windows)]
        variants = [
            ("MOM", "momentum", "$close / Ref($close, {w}) - 1", "positive", w),
            ("REV", "reversal", "Ref($close, {w}) / $close - 1", "positive", w),
            ("MA", "price_pattern", "Mean($close, {w}) / $close - 1", "positive", w),
            ("STD", "volatility", "Std($close, {w}) / $close", "negative", w),
            ("VOLR", "liquidity", "Mean($volume, {w}) / Mean($volume, {w2}) - 1", "positive", w),
            ("CORR", "volume_price", "Corr($close, $volume, {w})", "positive", w),
            ("DELTA", "momentum", "Delta($close, {w}) / $close", "positive", w),
            ("SLOPE", "price_pattern", "Slope($close, {w}) / $close", "positive", w),
            ("VWAP", "price_pattern", "Mean($vwap / $close - 1, {w})", "positive", w),
            ("RANGE", "volatility", "(Mean($high, {w}) - Mean($low, {w})) / $close", "negative", w),
        ]
        prefix, category, template, sign, holding = variants[index % len(variants)]
        expression = template.format(w=w, w2=max(5, w2))
        return {
            "name": f"{prefix}_{index:03d}",
            "category": category,
            "hypothesis": f"Mock hypothesis for {category} variation {index}.",
            "expression": expression,
            "expected_sign": sign,
            "holding_period": min(holding, 20),
            "rationale": "Deterministic mock candidate for pipeline tests.",
        }


def make_client(config: LLMConfig) -> BaseLLMClient:
    provider = config.provider.strip().lower()
    if provider == "mock":
        return MockClient(config)
    if provider == "anthropic":
        return AnthropicClient(config)
    if provider in {"openai", "openai_compatible"}:
        return OpenAIClient(config)
    raise LLMConfigError(f"unsupported provider: {config.provider}")
