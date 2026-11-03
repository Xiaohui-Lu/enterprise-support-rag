"""The one way this repo calls an LLM (D-003): through the cache, with tokens, cost and
latency recorded for every call. Real-time calls stream (D-029).

Requests are plain dicts with the Messages API's own field names, so the cache key, the
run record and the API call all see exactly the same thing.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from support_rag import config
from support_rag.llm_cache import LLMCache, cache_key

GENERATOR_MODEL = "claude-haiku-5-5"  # D-021
JUDGE_MODEL = "claude-sonnet-5-5"  # D-021

# Dollars per million tokens, Anthropic first-party prices, checked 2026-11-02.
# Haiku 5.5 charges more once a prompt goes over 100K tokens.
PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-haiku-5-5>100k": {"input": 0.50, "output": 2.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20},
    # The labeling panel (D-028).
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20},
    "claude-fable-5-1": {"input": 10.00, "output": 50.00, "cache_read": 0.25},
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
    batch: bool = False  # answered by the Message Batches API: half price, no latency

    @property
    def complete(self) -> bool:
        return self.stop_reason in COMPLETE_STOP_REASONS


class MessagesAPI(Protocol):
    def stream(self, **request: Any) -> AbstractContextManager[Any]: ...


class LLMClient:
    def __init__(self, cache: LLMCache, messages: MessagesAPI | None = None) -> None:
        self._cache = cache
        self._messages = messages

    def complete(self, request: dict[str, Any]) -> LLMResult:
        hit = self._cache.get(request)
        if hit is not None:
            return _result(hit.response, hit.latency_ms, cached=True, batch=hit.batch)

        # Streamed, so long thinking never meets the SDK's limit on non-streaming calls. The
        # whole message is collected before returning, so latency stays end-to-end. The
        # transport isn't part of the request, so cached calls stay valid.
        start = time.perf_counter()
        with self._api().stream(**request) as stream:
            response = stream.get_final_message().to_dict()
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


def _result(
    response: dict[str, Any], latency_ms: float, *, cached: bool, batch: bool = False
) -> LLMResult:
    text = "".join(block["text"] for block in response["content"] if block["type"] == "text")
    usage = {k: v for k, v in response["usage"].items() if isinstance(v, int)}
    return LLMResult(
        text, response["stop_reason"], usage, latency_ms, cached, response["model"], batch
    )


class BatchesAPI(Protocol):
    def create(self, **kwargs: Any) -> Any: ...
    def retrieve(self, batch_id: str) -> Any: ...
    def results(self, batch_id: str) -> Iterable[Any]: ...


def run_batch(
    requests: Sequence[dict[str, Any]],
    cache: LLMCache,
    *,
    batch_id: str | None = None,
    batches: BatchesAPI | None = None,
    poll_seconds: float = 60,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Send the requests the cache doesn't have as one Message Batch, wait for it, and cache
    the complete results at the batch price (D-022). A run then replays them from the cache.

    Pass `batch_id` to collect a batch submitted earlier, after an interruption, instead of
    paying for the same requests twice. Results that come back incomplete or errored aren't
    cached, so the run that follows calls those again in real time (which pays twice for a
    call cut off at its output limit).
    """
    pending = {cache_key(r): r for r in requests if cache.get(r) is None}
    if not pending and batch_id is None:
        return {"batch_id": None, "submitted": 0, "cached": 0, "not_cached": 0, "cost_usd": 0.0}
    if batches is None:
        import anthropic

        config.load_env()
        batches = anthropic.Anthropic().messages.batches
    if batch_id is None:
        # The cache key (64 hex characters) doubles as the batch's custom_id.
        batch = batches.create(
            requests=[{"custom_id": key, "params": r} for key, r in pending.items()]
        )
        batch_id = batch.id
        log(f"submitted batch {batch_id} ({len(pending)} requests); if interrupted, rerun "
            f"with --batch-id {batch_id}")  # fmt: skip
    while (batch := batches.retrieve(batch_id)).processing_status != "ended":
        log(f"batch {batch_id}: {batch.processing_status}")
        time.sleep(poll_seconds)

    cached = not_cached = 0
    cost = 0.0
    for entry in batches.results(batch_id):
        request = pending.get(entry.custom_id)
        if request is None:
            continue  # already cached, or not one of these requests
        if entry.result.type != "succeeded":
            not_cached += 1
            continue
        response = entry.result.message.to_dict()
        result = _result(response, 0.0, cached=False, batch=True)
        cost += cost_usd(result.model, result.usage, batch=True)
        if result.complete:
            cache.put(request, response, result.usage, 0.0, batch=True)
            cached += 1
        else:
            not_cached += 1
    return {
        "batch_id": batch_id,
        "submitted": len(pending),
        "cached": cached,
        "not_cached": not_cached,
        "cost_usd": round(cost, 4),
    }


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
