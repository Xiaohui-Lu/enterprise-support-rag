"""The one way this repo calls an LLM (D-003): through the cache, with tokens, cost and
latency recorded for every call.

Requests are plain dicts with the Messages API's own field names, so the cache key, the
run record and the API call all see exactly the same thing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

from support_rag import config
from support_rag.llm_cache import LLMCache

GENERATOR_MODEL = "claude-haiku-5-5"  # D-021
JUDGE_MODEL = "claude-sonnet-5-5"  # D-021

# Dollars per million tokens, Anthropic first-party prices, checked 2026-11-02.
# Haiku 5.5 charges more once a prompt goes over 100K tokens.
PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-haiku-5-5>100k": {"input": 0.50, "output": 2.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
}
LONG_PROMPT_TOKENS = 100_000
BATCH_DISCOUNT = 0.5  # the Message Batches API is half price (D-022)

# Only complete answers are cached (D-020). "max_tokens" means cut off, "refusal" means
# declined by a safety classifier: neither is a result to replay.
COMPLETE_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})


@dataclass(frozen=True)
class LLMResult:
    text: str  # the text blocks joined; thinking is billed but not returned
    stop_reason: str
    usage: dict[str, int]  # of the original call, also on a cache hit
    latency_ms: float  # of the original call; never report a cache hit's latency
    cached: bool
    model: str

    @property
    def complete(self) -> bool:
        return self.stop_reason in COMPLETE_STOP_REASONS


class MessagesAPI(Protocol):
    def create(self, **request: Any) -> Any: ...


class LLMClient:
    def __init__(self, cache: LLMCache, messages: MessagesAPI | None = None) -> None:
        self._cache = cache
        self._messages = messages

    def complete(self, request: dict[str, Any]) -> LLMResult:
        hit = self._cache.get(request)
        if hit is not None:
            return _result(hit.response, hit.latency_ms, cached=True)

        start = time.perf_counter()
        response = self._api().create(**request).to_dict()
        latency_ms = round((time.perf_counter() - start) * 1000, 1)
        result = _result(response, latency_ms, cached=False)
        if result.complete:
            self._cache.put(request, response, result.usage, latency_ms)
        return result

    def _api(self) -> MessagesAPI:
        if self._messages is None:  # created on first real call, so cache hits need no key
            import anthropic

            config.load_env()
            self._messages = anthropic.Anthropic().messages
        return self._messages


def _result(response: dict[str, Any], latency_ms: float, *, cached: bool) -> LLMResult:
    text = "".join(block["text"] for block in response["content"] if block["type"] == "text")
    usage = {k: v for k, v in response["usage"].items() if isinstance(v, int)}
    return LLMResult(text, response["stop_reason"], usage, latency_ms, cached, response["model"])


def cost_usd(model: str, usage: dict[str, int], *, batch: bool = False) -> float:
    """What a call cost, from its token usage."""
    prompt = (
        usage.get("input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
    )
    tier = f"{model}>100k" if prompt > LONG_PROMPT_TOKENS and f"{model}>100k" in PRICES else model
    price = PRICES[tier]
    dollars = (
        usage.get("input_tokens", 0) * price["input"]
        + usage.get("output_tokens", 0) * price["output"]
        + usage.get("cache_read_input_tokens", 0) * price.get("cache_read", price["input"])
        + usage.get("cache_creation_input_tokens", 0) * price["input"] * 1.25
    ) / 1_000_000
    return dollars * (BATCH_DISCOUNT if batch else 1.0)
