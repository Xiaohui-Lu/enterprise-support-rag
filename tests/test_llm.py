import pytest

from support_rag.llm import LLMClient, cost_usd, run_batch
from support_rag.llm_cache import LLMCache

REQUEST = {"model": "claude-haiku-5-5", "max_tokens": 100, "messages": []}


class FakeResponse:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return self.data


def _message(model, stop_reason):
    return FakeResponse(
        {
            "model": model,
            "stop_reason": stop_reason,
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "Go to Settings [1]."},
            ],
            "usage": {"input_tokens": 4500, "output_tokens": 800, "service_tier": "standard"},
        }
    )


class FakeStream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_final_message(self):
        return self.message


class FakeMessages:
    """Stands in for the SDK's client.messages; counts calls, never touches the network."""

    def __init__(self, stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.calls = 0

    def stream(self, **request):
        self.calls += 1
        return FakeStream(_message(request["model"], self.stop_reason))


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


# run_batch()

SONNET = {"model": "claude-sonnet-5-5", "max_tokens": 100, "messages": []}


class Obj:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class FakeBatches:
    """Stands in for client.messages.batches: answers every request at once."""

    def __init__(self, stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.created = []

    def create(self, requests):
        self.created.append(requests)
        self.requests = requests
        return Obj(id="msgbatch_1")

    def retrieve(self, batch_id):
        return Obj(processing_status="ended")

    def results(self, batch_id):
        for r in self.requests:
            message = _message(r["params"]["model"], self.stop_reason)
            yield Obj(custom_id=r["custom_id"], result=Obj(type="succeeded", message=message))


def _requests(n):
    return [SONNET | {"messages": [{"role": "user", "content": f"Q{i}"}]} for i in range(n)]


def test_a_batch_sends_only_uncached_requests(cache):
    first, second = _requests(2)
    LLMClient(cache, FakeMessages()).complete(first)
    batches = FakeBatches()
    summary = run_batch([first, second], cache, batches=batches, poll_seconds=0, log=print)
    assert len(batches.created[0]) == 1
    assert summary["cached"] == 1
    assert cache.get(second).batch is True


def test_a_replayed_batch_call_costs_half(cache):
    (request,) = _requests(1)
    run_batch([request], cache, batches=FakeBatches(), poll_seconds=0, log=print)
    result = LLMClient(cache, FakeMessages()).complete(request)
    assert result.cached and result.batch
    full = cost_usd(result.model, result.usage)
    assert cost_usd(result.model, result.usage, batch=result.batch) == pytest.approx(full / 2)


def test_collecting_an_earlier_batch_never_submits_again(cache):
    batches = FakeBatches()
    requests = _requests(2)
    run_batch(requests, cache, batches=batches, poll_seconds=0, log=print)
    run_batch(requests, cache, batch_id="msgbatch_1", batches=batches, poll_seconds=0, log=print)
    assert len(batches.created) == 1


def test_an_incomplete_batch_result_is_not_cached(cache):
    summary = run_batch(
        _requests(2), cache, batches=FakeBatches("max_tokens"), poll_seconds=0, log=print
    )
    assert (summary["cached"], summary["not_cached"]) == (0, 2)
    assert len(cache) == 0
