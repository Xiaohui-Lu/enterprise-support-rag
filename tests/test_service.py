import json
import logging
import threading

import pytest
from fastapi.testclient import TestClient

from support_rag.llm import LLMResult
from support_rag.llm_cache import LLMCache
from support_rag.records import Article, Hit
from support_rag.service import Answerer, create_app

ARTICLES = {
    article_id: Article(
        article_id, f"Title {article_id}", f"https://help.example/{article_id}", "article",
        f"Text {article_id}", "",
    )
    for article_id in "ABC"
}  # fmt: skip
QUESTION = "How do I connect my domain?"


class FixedRetriever:
    def search(self, query, k=10):
        return [Hit("B", 0.7), Hit("C", 0.6), Hit("A", 0.5)][:k]


class FakeLLM:
    """Streams its reply in pieces, or fails halfway if told to."""

    def __init__(self, pieces=("Point the DNS here [2], ", "then wait [1]."), fail=False):
        self.pieces = pieces
        self.fail = fail
        self.requests = []

    def stream(self, request):
        self.requests.append(request)
        for piece in self.pieces:
            yield piece
            if self.fail:
                raise RuntimeError("the connection dropped")
        usage = {"input_tokens": 1000, "output_tokens": 200}
        return LLMResult("".join(self.pieces), "end_turn", usage, 900.0, False, request["model"])


def _events(response):
    """The (event, data) pairs in an SSE response body."""
    events = []
    for block in response.text.strip().split("\n\n"):
        name, data = block.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def _client(llm=None):
    answerer = Answerer(ARTICLES, FixedRetriever(), llm or FakeLLM())
    return TestClient(create_app(answerer))


def test_an_answer_streams_sources_then_text_then_done():
    response = _client().post("/api/answer", json={"question": QUESTION})
    assert response.headers["content-type"].startswith("text/event-stream")
    names = [name for name, _ in _events(response)]
    assert names == ["sources", "delta", "delta", "done"]


def test_the_sources_are_numbered_as_the_model_sees_them():
    (_, sources), *_ = _events(_client().post("/api/answer", json={"question": QUESTION}))
    assert [(a["n"], a["title"]) for a in sources["articles"]] == [
        (1, "Title B"), (2, "Title C"), (3, "Title A"),
    ]  # fmt: skip
    assert sources["articles"][0]["url"] == "https://help.example/B"


def test_done_gives_the_cited_numbers_in_the_order_cited():
    *_, (_, done) = _events(_client().post("/api/answer", json={"question": QUESTION}))
    assert done["cited"] == [2, 1]
    assert done["declined"] is False
    assert done["cost_usd"] > 0
    assert set(done["timings_ms"]) == {"embed", "search", "retrieval", "first_text", "total"}
    assert set(done["tokens"]) == {"input", "output"}


def test_a_decline_is_marked_in_done():
    response = _client(FakeLLM(pieces=("NOT_FOUND",))).post("/api/answer", json={"question": "?"})
    *_, (_, done) = _events(response)
    assert done["declined"] is True
    assert done["cited"] == []


def test_the_service_sends_the_request_the_evaluation_sends():
    from support_rag.answer import build_request
    from support_rag.answer_eval import context_for
    from support_rag.records import Question

    llm = FakeLLM()
    _client(llm).post("/api/answer", json={"question": QUESTION})
    question = Question("live", "live", QUESTION, "", ())
    context = context_for("s1", question, articles_by_id=ARTICLES, retriever=FixedRetriever())
    assert llm.requests == [build_request(question, context, effort="medium")]


def test_a_failure_mid_stream_ends_with_an_error_event():
    events = _events(_client(FakeLLM(fail=True)).post("/api/answer", json={"question": QUESTION}))
    assert [name for name, _ in events] == ["sources", "delta", "error"]


@pytest.mark.parametrize("question", ["", "   ", "x" * 8_001])
def test_a_blank_or_huge_question_is_rejected(question):
    assert _client().post("/api/answer", json={"question": question}).status_code == 422


def test_surrounding_whitespace_is_dropped_before_answering():
    llm = FakeLLM()
    _client(llm).post("/api/answer", json={"question": f"  {QUESTION}\n"})
    assert QUESTION in json.dumps(llm.requests[0])
    assert f"  {QUESTION}" not in json.dumps(llm.requests[0])


class TimedRetriever(FixedRetriever):
    """Times its stages the way the Postgres retriever does."""

    last_encode_ms = 41.25
    last_search_ms = 8.31


def _logged(caplog, question, retriever=None):
    answerer = Answerer(ARTICLES, retriever or TimedRetriever(), FakeLLM())
    with caplog.at_level(logging.INFO, logger="support_rag.timing"):
        TestClient(create_app(answerer)).post("/api/answer", json={"question": question})
    return [r.getMessage() for r in caplog.records if r.name == "support_rag.timing"]


def test_each_answer_logs_one_json_line_with_its_stage_timings(caplog):
    (line,) = _logged(caplog, QUESTION)
    record = json.loads(line)
    assert record["event"] == "answered"
    assert (record["embed_ms"], record["search_ms"]) == (41.2, 8.3)
    assert {"retrieval_ms", "first_text_ms", "total_ms", "input_tokens", "output_tokens"} <= set(
        record
    )
    assert (record["cached"], record["declined"]) == (False, False)


def test_a_retriever_that_doesnt_time_its_stages_logs_none_for_them(caplog):
    (line,) = _logged(caplog, QUESTION, FixedRetriever())
    assert json.loads(line)["embed_ms"] is None


def test_the_question_is_never_logged(caplog):
    (line,) = _logged(caplog, "my secret domain question")
    assert "secret" not in line
    assert "secret" not in caplog.text


def test_health():
    assert _client().get("/api/health").json() == {"status": "ok"}


def test_the_call_cache_can_be_shared_between_threads(tmp_path):
    with LLMCache(tmp_path / "cache.sqlite") as cache:
        usage = {"input_tokens": 1}

        def write(i):
            cache.put({"n": i}, {"content": []}, usage, 1.0)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(cache) == 20
        assert cache.get({"n": 7}) is not None
