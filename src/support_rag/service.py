"""The support assistant as a web service: FastAPI streams S1's answer as Server-Sent Events
(D-049, D-050).

`POST /api/answer` with `{"question": "..."}` answers with three kinds of event:

- `sources`: the numbered articles S1 reads, each with its title and URL;
- `delta`: each piece of text as the model writes it;
- `done`: the cited article numbers, whether it declined, the cost, whether the answer came
  from the call cache, and the timings.

An `error` event takes the place of `done` if anything fails once the stream has started.

It runs the functions the evaluation ran (D-008): `context_for("s1")`, `build_request` and
`parse_answer`. The retriever, Postgres and the SDK are synchronous, so FastAPI runs each
answer in its thread pool; retrieval is serialized with a lock, and model calls run side by
side.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import AsyncIterator, Generator, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Protocol

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints

from support_rag.answer import build_request, parse_answer
from support_rag.answer_eval import DEFAULT_EFFORT, DEFAULT_K, DEFAULT_RETRIEVER, context_for
from support_rag.llm import LLMResult, cost_usd, prompt_tokens
from support_rag.records import Article, Question
from support_rag.retrieval_eval import Retriever

log = logging.getLogger("support_rag.service")
# One JSON line per answer, with the time for each stage (D-052). Never the question.
timing_log = logging.getLogger("support_rag.timing")

# Long enough for the longest real question in WixQA, a customer's email.
MAX_QUESTION_CHARS = 8_000
ERROR_MESSAGE = "Something went wrong while answering. Please try again."


class StreamingLLM(Protocol):
    def stream(self, request: dict[str, Any]) -> Generator[str, None, LLMResult]: ...


@dataclass
class Answerer:
    """What answering needs, loaded once: the articles, the retriever and the LLM client."""

    articles_by_id: Mapping[str, Article]
    retriever: Retriever
    llm: StreamingLLM
    effort: str = DEFAULT_EFFORT
    k: int = DEFAULT_K
    _retrieval: threading.Lock = field(default_factory=threading.Lock)

    def events(self, text: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """Answer one question with S1, as (event, data) pairs in the order they're sent."""
        start = time.perf_counter()

        def since_start() -> float:
            return round((time.perf_counter() - start) * 1000, 1)

        question = Question(id="live", subset="live", question=text, answer="", gold_ids=())
        with self._retrieval:  # one embedding model and one Postgres connection
            context = context_for(
                "s1",
                question,
                articles_by_id=self.articles_by_id,
                retriever=self.retriever,
                k=self.k,
            )
            # The Postgres retriever times its two stages; other retrievers don't (D-052).
            embed_ms = getattr(self.retriever, "last_encode_ms", None)
            search_ms = getattr(self.retriever, "last_search_ms", None)
        retrieval_ms = since_start()
        yield (
            "sources",
            {
                "articles": [
                    {"n": n, "title": article.title, "url": article.url}
                    for n, article in enumerate(context, start=1)
                ]
            },
        )

        stream = self.llm.stream(build_request(question, context, effort=self.effort))
        first_text_ms = None
        while True:
            try:
                piece = next(stream)
            except StopIteration as stop:
                result: LLMResult = stop.value
                break
            if first_text_ms is None:
                first_text_ms = since_start()
            yield "delta", {"text": piece}

        context_ids = tuple(article.id for article in context)
        answer = parse_answer(result.text, context_ids)
        yield (
            "done",
            {
                "cited": [context_ids.index(article_id) + 1 for article_id in answer.cited_ids],
                "declined": answer.declined,
                "complete": result.complete,
                "cached": result.cached,
                "cost_usd": round(cost_usd(result.model, result.usage, batch=result.batch), 6),
                "tokens": {
                    "input": prompt_tokens(result.usage),
                    "output": result.usage.get("output_tokens", 0),
                },
                "timings_ms": {
                    "embed": None if embed_ms is None else round(embed_ms, 1),
                    "search": None if search_ms is None else round(search_ms, 1),
                    "retrieval": retrieval_ms,
                    "first_text": first_text_ms,
                    "total": since_start(),
                },
            },
        )


def sse(event: str, data: Mapping[str, Any]) -> str:
    """One Server-Sent Event: its name, then its data as one line of JSON."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def timing_line(done: Mapping[str, Any]) -> str:
    """The timing log's line for one answer: JSON, flat, without the question (D-052).

    {"event": "answered", "embed_ms": 41.2, "search_ms": 8.3, ..., "declined": false}
    """
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "event": "answered",
    }
    record |= {f"{stage}_ms": ms for stage, ms in done["timings_ms"].items()}
    record |= {f"{kind}_tokens": n for kind, n in done["tokens"].items()}
    record |= {key: done[key] for key in ("cost_usd", "cached", "declined", "complete")}
    return json.dumps(record)


def _stream(answerer: Answerer, text: str) -> Iterator[str]:
    """The events as SSE text, and the timing log's line for each answer."""
    try:
        for event, data in answerer.events(text):
            if event == "done":
                timing_log.info(timing_line(data))
            yield sse(event, data)
    except Exception:
        log.exception("answer failed")
        yield sse("error", {"message": ERROR_MESSAGE})


def configure_logging() -> None:
    """Plain lines for uvicorn and errors; bare JSON lines for the timing log, so
    `docker compose logs app` can be piped into jq (D-052)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    timing_log.addHandler(handler)
    timing_log.setLevel(logging.INFO)
    timing_log.propagate = False


class AnswerRequest(BaseModel):
    # Surrounding whitespace is dropped first, so a blank question is rejected, not answered.
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
    ]


def load_answerer() -> Answerer:
    """The real thing: every article, dense search in Postgres, and the cached LLM client."""
    from support_rag.llm import LLMClient
    from support_rag.llm_cache import LLMCache
    from support_rag.retrieval_eval import answer_retriever
    from support_rag.wixqa import load_articles

    articles = load_articles()
    retriever, _ = answer_retriever(DEFAULT_RETRIEVER, "article", articles)
    llm = LLMClient(LLMCache())
    llm.warm_up()  # a cold SDK import took 23 s; pay it at startup, not on the first question
    return Answerer({a.id: a for a in articles}, retriever, llm)


def create_app(answerer: Answerer | None = None, *, static_dir: Path | None = None) -> FastAPI:
    """The app. Tests pass their own `answerer`; otherwise the real one loads at startup.
    A built UI in `static_dir`, if there is one, is served at /."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if answerer is None:
            app.state.answerer = load_answerer()
        yield

    app = FastAPI(title="Support RAG", lifespan=lifespan)
    if answerer is not None:
        app.state.answerer = answerer

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/answer")
    def answer(body: AnswerRequest, request: Request) -> StreamingResponse:
        return StreamingResponse(
            _stream(request.app.state.answerer, body.question),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    if static_dir is not None and static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="ui")
    return app
