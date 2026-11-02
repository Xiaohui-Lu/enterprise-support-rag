import pytest

from support_rag.llm import LLMClient, cost_usd
from support_rag.llm_cache import LLMCache

REQUEST = {"model": "claude-haiku-5-5", "max_tokens": 100, "messages": []}


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return self.data


class FakeMessages:
    """Stands in for the SDK's client.messages; counts calls, never touches the network."""

    def __init__(self, stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.calls = 0

    def create(self, **request):
        self.calls += 1
        return FakeResponse(
            {
                "model": request["model"],
                "stop_reason": self.stop_reason,
                "content": [
                    {"type": "thinking", "thinking": ""},
                    {"type": "text", "text": "Go to Settings [1]."},
                ],
                "usage": {"input_tokens": 4500, "output_tokens": 800, "service_tier": "standard"},
            }
        )


@pytest.fixture
def cache(tmp_path):
    with LLMCache(tmp_path / "cache.sqlite") as cache:
        yield cache


def test_a_complete_answer_is_returned_and_cached(cache):
    api = FakeMessages()
    result = LLMClient(cache, api).complete(REQUEST)
    assert result.text == "Go to Settings [1]."
    assert result.usage == {"input_tokens": 4500, "output_tokens": 800}
    assert result.cached is False
    assert len(cache) == 1


def test_a_repeated_request_is_served_from_the_cache(cache):
    api = FakeMessages()
    client = LLMClient(cache, api)
    first = client.complete(REQUEST)
    second = client.complete(REQUEST)
    assert api.calls == 1
    assert second.cached is True
    assert second.text == first.text
    assert second.latency_ms == first.latency_ms  # the original call's latency


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_a_cut_off_or_refused_answer_is_not_cached(cache, stop_reason):
    api = FakeMessages(stop_reason)
    client = LLMClient(cache, api)
    assert client.complete(REQUEST).complete is False
    client.complete(REQUEST)
    assert api.calls == 2
    assert len(cache) == 0


def test_cost_uses_the_models_prices():
    usage = {"input_tokens": 50_000, "output_tokens": 10_000}
    assert cost_usd("claude-haiku-5-5", usage) == pytest.approx(0.005 + 0.005)
    assert cost_usd("claude-sonnet-5-5", usage) == pytest.approx(0.10 + 0.10)


def test_the_batch_api_halves_the_cost():
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    assert cost_usd("claude-sonnet-5-5", usage, batch=True) == pytest.approx(1.00)


def test_haiku_prompts_over_100k_tokens_cost_more():
    usage = {"input_tokens": 200_000, "output_tokens": 0}
    assert cost_usd("claude-haiku-5-5", usage) == pytest.approx(200_000 * 0.50 / 1_000_000)
