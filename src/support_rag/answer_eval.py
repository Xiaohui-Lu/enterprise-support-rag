"""Answer runs: one strategy over a split, every answer saved with its tokens, cost and latency.

Answers are only generated here; grading them is the judge's job. The test set is not
offered: it runs once, at milestone M1 (D-004).
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from support_rag import config
from support_rag.answer import PROMPT_VERSION, build_request, parse_answer
from support_rag.llm import LLMResult, cost_usd
from support_rag.records import Article, Question
from support_rag.retrieval_eval import Retriever, questions_for_split
from support_rag.runs import append_result, start_run, write_summary


class LLM(Protocol):
    def complete(self, request: dict[str, Any]) -> LLMResult: ...


@dataclass(frozen=True)
class AnswerResult:
    """One line of questions.jsonl in an answer run."""

    question_id: str
    subset: str
    multi_article: bool
    gold_ids: tuple[str, ...]
    context_ids: tuple[str, ...]  # what the model saw, in prompt order: [1] is the first
    text: str
    cited_ids: tuple[str, ...]
    declined: bool
    invalid_citations: int
    stop_reason: str
    usage: dict[str, int]
    cost_usd: float  # what this answer cost when it was generated
    latency_ms: float  # of the original call
    cached: bool  # True: replayed from the cache, nothing paid in this run


def context_for(
    strategy: str,
    question: Question,
    *,
    articles_by_id: Mapping[str, Article],
    retriever: Retriever | None = None,
    k: int = 5,
) -> list[Article]:
    """The articles a strategy shows the model: none (S0), BM25's top k (S1) or gold (S*)."""
    match strategy:
        case "s0":
            return []
        case "s1":
            if retriever is None:
                raise ValueError("S1 needs a retriever")
            return [
                articles_by_id[hit.article_id] for hit in retriever.search(question.question, k)
            ]
        case "s_star":
            return [articles_by_id[article_id] for article_id in question.gold_ids]
    raise ValueError(f"unknown strategy {strategy!r}")


def answer_question(
    llm: LLM, question: Question, context: Sequence[Article], *, effort: str
) -> AnswerResult:
    request = build_request(question, context, effort=effort)
    result = llm.complete(request)
    context_ids = tuple(article.id for article in context)
    answer = parse_answer(result.text, context_ids)
    return AnswerResult(
        question_id=question.id,
        subset=question.subset,
        multi_article=question.is_multi_article,
        gold_ids=question.gold_ids,
        context_ids=context_ids,
        text=answer.text,
        cited_ids=answer.cited_ids,
        declined=answer.declined,
        invalid_citations=answer.invalid_citations,
        stop_reason=result.stop_reason,
        usage=result.usage,
        cost_usd=round(cost_usd(result.model, result.usage), 6),
        latency_ms=result.latency_ms,
        cached=result.cached,
    )


def summarize_answers(results: Sequence[AnswerResult]) -> dict[str, Any]:
    """Format and cost figures for a run. Correctness comes later, from the judge."""
    paid = [r for r in results if not r.cached]
    latencies = sorted(r.latency_ms for r in paid)
    return {
        "n": len(results),
        "declined": sum(r.declined for r in results),
        "with_invalid_citations": sum(r.invalid_citations > 0 for r in results),
        "stop_reasons": dict(Counter(r.stop_reason for r in results)),
        "mean_input_tokens": _mean(r.usage.get("input_tokens", 0) for r in results),
        "mean_output_tokens": _mean(r.usage.get("output_tokens", 0) for r in results),
        "cost_of_answers_usd": round(sum(r.cost_usd for r in results), 4),
        "paid_this_run_usd": round(sum(r.cost_usd for r in paid), 4),
        "cached": len(results) - len(paid),
        # Latency only from real calls: a cache hit takes no time (D-020, D-022).
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_max_ms": latencies[-1] if latencies else None,
    }


def _mean(values: Any) -> float:
    values = list(values)
    return round(statistics.fmean(values), 1) if values else 0.0


def pick_evenly(questions: Sequence[Question], limit: int | None) -> list[Question]:
    """`limit` questions spread evenly over the list, so a pilot mixes the subsets."""
    if limit is None or limit >= len(questions):
        return list(questions)
    step = len(questions) / limit
    return [questions[int(i * step)] for i in range(limit)]


def run_answers(
    strategy: str,
    split: str,
    *,
    effort: str = "medium",
    k: int = 5,
    limit: int | None = None,
    runs_dir: Path = config.RUNS_DIR,
) -> Path:
    """Answer a split with one strategy and save the run. Returns the run directory."""
    from support_rag.bm25 import STOPWORDS, BM25Retriever
    from support_rag.llm import GENERATOR_MODEL, LLMClient
    from support_rag.llm_cache import LLMCache
    from support_rag.wixqa import load_articles

    questions = pick_evenly(questions_for_split(split), limit)
    articles = load_articles()
    articles_by_id = {article.id: article for article in articles}
    retriever = BM25Retriever(articles) if strategy == "s1" else None

    run_config: dict[str, Any] = {
        "kind": "answers",
        "strategy": strategy,
        "model": GENERATOR_MODEL,
        "effort": effort,
        "prompt_version": PROMPT_VERSION,
        "split": split,
        "n_questions": len(questions),
        "limit": limit,
    }
    if retriever is not None:
        run_config["k"] = k
        run_config["retriever"] = {"name": "bm25", **retriever.params}
        run_config["tokenizer"] = {"stem": True, "stopwords": f"lucene-en-{len(STOPWORDS)}"}
    name = f"{strategy}{f'-k{k}' if retriever else ''}-{effort}_{split}"
    run_dir = start_run(name, run_config, runs_dir)

    results = []
    with LLMCache() as cache:
        llm = LLMClient(cache)
        for question in questions:
            context = context_for(
                strategy, question, articles_by_id=articles_by_id, retriever=retriever, k=k
            )
            result = answer_question(llm, question, context, effort=effort)
            append_result(run_dir, asdict(result))
            results.append(result)
    write_summary(run_dir, summarize_answers(results))
    return run_dir
