"""Answer runs: one strategy over a split, every answer saved with its tokens, cost and latency.

Answers are only generated here; grading them is the judge's job. The test set runs only at
a named milestone, from a clean commit (D-004, D-034).
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from support_rag import config
from support_rag.agent import (
    AGENT_PROMPT_VERSION,
    MAX_PROMPT_TOKENS,
    MAX_STEPS,
    SEARCH_RESULTS,
    SNIPPET_WORDS,
    ToolBox,
    run_agent,
)
from support_rag.answer import PROMPT_VERSION, SYSTEM_WITH_ARTICLES, build_request, parse_answer
from support_rag.llm import (
    LONG_PROMPT_TOKENS,
    LLMResult,
    cost_usd,
    cost_without_caching,
    prompt_tokens,
)
from support_rag.long_context import (
    CLAUDE_TOKENS_PER_TIKTOKEN,
    PROMPT_BUDGET,
    TokenCounter,
    capped_context,
    estimate_request_tokens,
)
from support_rag.records import Article, Hit, Question
from support_rag.retrieval_eval import Retriever, answer_retriever, questions_for_split
from support_rag.runs import append_result, git_state, start_run, write_summary

# The generator's settings after tuning on dev (D-033): Haiku's default effort, and how
# many articles S1 shows the model. S1 searches with the default retriever (D-042).
DEFAULT_EFFORT = "medium"
DEFAULT_K = 10
DEFAULT_RETRIEVER = "dense"
MILESTONES = ("M1", "M2", "M3")


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
    # What the model read, in prompt order, when it wasn't whole articles (D-042).
    context_sections: tuple[str, ...] | None = None
    # S2 only: how the loop went (D-043).
    agent: dict[str, Any] | None = None
    # S3 only: K, how many articles the token cap let through, and the estimate (D-044).
    long_context: dict[str, Any] | None = None
    # The context-rot test only: the arm, the distractors and where the gold went (D-045).
    rot: dict[str, Any] | None = None


def reading_context(hits: Sequence[Hit], articles_by_id: Mapping[str, Article]) -> list[Article]:
    """What the model reads for these hits, in rank order (D-042).

    A hit that carries a section stands for that section alone, which already starts with
    the article's title. Any other hit is the whole article. The result is a list of
    Articles either way, so the prompt and the citation numbers don't change.
    """
    context = []
    for hit in hits:
        article = articles_by_id[hit.article_id]
        context.append(article if hit.section is None else replace(article, contents=hit.section))
    return context


def context_for(
    strategy: str,
    question: Question,
    *,
    articles_by_id: Mapping[str, Article],
    retriever: Retriever | None = None,
    k: int = DEFAULT_K,
    count: TokenCounter | None = None,
) -> list[Article]:
    """What a strategy shows the model: nothing (S0), the top k hits (S1), the top k hits
    that fit under the token budget (S3), or gold (S*)."""
    match strategy:
        case "s0":
            return []
        case "s1":
            if retriever is None:
                raise ValueError("S1 needs a retriever")
            return reading_context(retriever.search(question.question, k), articles_by_id)
        case "s3":
            if retriever is None or count is None:
                raise ValueError("S3 needs a retriever and a token counter")
            ranked = reading_context(retriever.search(question.question, k), articles_by_id)
            used = count(SYSTEM_WITH_ARTICLES) + count(question.question)
            context = capped_context(ranked, budget=PROMPT_BUDGET, count=count, used=used)
            if not context:  # an empty context would silently become closed book
                raise ValueError(f"{question.id}: the top article alone is over the budget")
            return context
        case "s_star":
            return [articles_by_id[article_id] for article_id in question.gold_ids]
    raise ValueError(f"unknown strategy {strategy!r}")


def answer_question(
    llm: LLM, question: Question, context: Sequence[Article], *, effort: str, sections: bool = False
) -> AnswerResult:
    """Answer one question. With `sections`, the context holds sections rather than whole
    articles, so the result saves what the model read for the judge (D-042)."""
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
        cost_usd=round(cost_usd(result.model, result.usage, batch=result.batch), 6),
        latency_ms=result.latency_ms,
        cached=result.cached,
        context_sections=tuple(article.contents for article in context) if sections else None,
    )


def answer_long_context(
    llm: LLM,
    question: Question,
    context: Sequence[Article],
    *,
    k: int,
    count: TokenCounter,
    effort: str,
) -> AnswerResult:
    """Answer one question with S3, recording what the token cap did and the estimate the
    run checks against the API's real count (D-044)."""
    result = answer_question(llm, question, context, effort=effort)
    estimate = estimate_request_tokens(build_request(question, context, effort=effort), count)
    return replace(
        result,
        long_context={
            "k": k,
            "sent": len(context),
            "capped": len(context) < k,
            "estimated_prompt_tokens": estimate,
        },
    )


def answer_with_agent(
    llm: LLM, question: Question, toolbox: ToolBox, *, effort: str
) -> AnswerResult:
    """Answer one question with S2. Tokens and cost add up every model call, and latency
    also includes the time the tools took (D-043)."""
    run = run_agent(llm, toolbox, question, effort=effort)
    answer = parse_answer(run.final.text, toolbox.context_ids)
    usage: Counter[str] = Counter()
    for call in run.calls:
        usage.update(call.usage)
    return AnswerResult(
        question_id=question.id,
        subset=question.subset,
        multi_article=question.is_multi_article,
        gold_ids=question.gold_ids,
        context_ids=toolbox.context_ids,
        text=answer.text,
        cited_ids=answer.cited_ids,
        declined=answer.declined,
        invalid_citations=answer.invalid_citations,
        stop_reason=run.final.stop_reason,
        usage=dict(usage),
        cost_usd=round(sum(cost_usd(c.model, c.usage, batch=c.batch) for c in run.calls), 6),
        latency_ms=round(sum(c.latency_ms for c in run.calls) + toolbox.tool_ms, 1),
        cached=all(c.cached for c in run.calls),
        context_sections=toolbox.context_sections,
        agent={
            "steps": sum(c.stop_reason == "tool_use" for c in run.calls),
            "model_calls": len(run.calls),
            "stopped_by": run.stopped_by,
            "queries": list(toolbox.queries),
            "read_ids": list(toolbox.read_ids),
            "tool_errors": toolbox.errors,
            "tool_ms": round(toolbox.tool_ms, 1),
            "prompt_tokens": [prompt_tokens(c.usage) for c in run.calls],
            "cost_without_caching_usd": round(
                sum(cost_without_caching(c.model, c.usage) for c in run.calls), 6
            ),
        },
    )


def summarize_answers(results: Sequence[AnswerResult]) -> dict[str, Any]:
    """Format and cost figures for a run. Correctness comes later, from the judge."""
    paid = [r for r in results if not r.cached]
    latencies = sorted(r.latency_ms for r in paid)
    summary: dict[str, Any] = {
        "n": len(results),
        "declined": sum(r.declined for r in results),
        "with_invalid_citations": sum(r.invalid_citations > 0 for r in results),
        "stop_reasons": dict(Counter(r.stop_reason for r in results)),
        # The whole prompt, cached or not; for S2, added up over every model call.
        "mean_input_tokens": _mean(prompt_tokens(r.usage) for r in results),
        "mean_output_tokens": _mean(r.usage.get("output_tokens", 0) for r in results),
        "cost_of_answers_usd": round(sum(r.cost_usd for r in results), 4),
        "paid_this_run_usd": round(sum(r.cost_usd for r in paid), 4),
        "cached": len(results) - len(paid),
        # Latency only from real calls: a cache hit takes no time (D-020, D-022).
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_max_ms": latencies[-1] if latencies else None,
    }
    if any(r.agent is not None for r in results):
        summary["agent"] = _summarize_agent(results)
    if any(r.long_context is not None for r in results):
        summary["long_context"] = _summarize_long_context(results)
    if any(r.rot is not None for r in results):
        summary["rot"] = _summarize_rot(results)
    return summary


def _spread(values: Sequence[float], digits: int | None = None) -> dict[str, float]:
    ordered = sorted(values)
    spread = {"min": ordered[0], "median": statistics.median(ordered), "max": ordered[-1]}
    return spread if digits is None else {k: round(v, digits) for k, v in spread.items()}


def _price_line_check(estimated: Sequence[int], real: Sequence[int]) -> dict[str, Any]:
    """The check on the token cap: the API's real prompt sizes against the 100K price line,
    and how far the local estimate was from them. An error above zero means the estimate
    was low, the side that risks the price line (D-044)."""
    return {
        "max_prompt_tokens": max(real),
        "over_price_line": sum(r > LONG_PROMPT_TOKENS for r in real),
        "estimate_error": _spread([(r - e) / r for e, r in zip(estimated, real, strict=True)], 3),
    }


def _summarize_long_context(results: Sequence[AnswerResult]) -> dict[str, Any]:
    """What S3's token cap did, and the check on it."""
    rows = [(r.long_context, prompt_tokens(r.usage)) for r in results if r.long_context]
    return {
        "k": rows[0][0]["k"],
        "capped": sum(lc["capped"] for lc, _ in rows),
        "articles_sent": _spread([lc["sent"] for lc, _ in rows]),
        **_price_line_check(
            [lc["estimated_prompt_tokens"] for lc, _ in rows], [real for _, real in rows]
        ),
    }


def _summarize_rot(results: Sequence[AnswerResult]) -> dict[str, Any]:
    """One context-rot arm: how many distractors went in, the check on the token cap, and
    how many answers cited a distractor (D-045)."""
    rows = [r for r in results if r.rot]
    return {
        "distractors": rows[0].rot["distractors"],
        "size": rows[0].rot["size"],
        "n_distractors": _spread([r.rot["n_distractors"] for r in rows]),
        **_price_line_check(
            [r.rot["estimated_prompt_tokens"] for r in rows], [prompt_tokens(r.usage) for r in rows]
        ),
        "cited_a_distractor": sum(any(c not in r.gold_ids for c in r.cited_ids) for r in rows),
    }


def _summarize_agent(results: Sequence[AnswerResult]) -> dict[str, Any]:
    """How S2's loops went: what the limits did, what was read, what caching saved (D-043)."""
    agents = [r.agent for r in results if r.agent is not None]
    steps = Counter(a["steps"] for a in agents)
    return {
        "steps": {str(n): steps[n] for n in sorted(steps)},
        "stopped_by": dict(Counter(a["stopped_by"] for a in agents if a["stopped_by"])),
        "mean_searches": _mean(len(a["queries"]) for a in agents),
        "mean_reads": _mean(len(a["read_ids"]) for a in agents),
        "tool_errors": sum(a["tool_errors"] for a in agents),
        "mean_largest_prompt_tokens": _mean(max(a["prompt_tokens"]) for a in agents),
        "mean_cached_input_tokens": _mean(
            r.usage.get("cache_read_input_tokens", 0) for r in results
        ),
        "cost_without_caching_usd": round(sum(a["cost_without_caching_usd"] for a in agents), 4),
    }


def _mean(values: Any) -> float:
    values = list(values)
    return round(statistics.fmean(values), 1) if values else 0.0


def check_test_run(milestone: str | None, git: dict[str, Any] | None = None) -> None:
    """The test set runs only at a named milestone, and only from a clean commit, so the run
    can be reproduced from that commit and nothing is tuned on it (D-004, D-034)."""
    if milestone not in MILESTONES:
        raise ValueError(f"the test set runs only at a milestone: pass --milestone {MILESTONES[0]}")
    if (git or git_state())["dirty"] is not False:
        raise ValueError("commit everything first: a test run must be reproducible from its commit")


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
    effort: str = DEFAULT_EFFORT,
    k: int = DEFAULT_K,
    retriever_name: str = DEFAULT_RETRIEVER,
    unit: str = "article",
    limit: int | None = None,
    milestone: str | None = None,
    runs_dir: Path = config.RUNS_DIR,
) -> Path:
    """Answer a split with one strategy and save the run. Returns the run directory."""
    if split == "test":
        check_test_run(milestone)
    from support_rag.llm import GENERATOR_MODEL, LLMClient
    from support_rag.llm_cache import LLMCache
    from support_rag.wixqa import load_articles

    questions = pick_evenly(questions_for_split(split), limit)
    articles = load_articles()
    articles_by_id = {article.id: article for article in articles}

    run_config: dict[str, Any] = {
        "kind": "answers",
        "strategy": strategy,
        "model": GENERATOR_MODEL,
        "effort": effort,
        "prompt_version": PROMPT_VERSION,
        "split": split,
        "n_questions": len(questions),
        "limit": limit,
        "milestone": milestone,
    }
    retriever = None
    count = None
    name = f"{strategy}-{effort}_{split}"
    if strategy == "s1":
        retriever, settings = answer_retriever(retriever_name, unit, articles)
        run_config |= {"k": k, "unit": unit, **settings}
        reads = "" if unit == "article" else f"-{unit}"
        name = f"s1-{retriever_name}{reads}-k{k}-{effort}_{split}"
    if strategy == "s2":
        retriever, settings = answer_retriever(DEFAULT_RETRIEVER, "article", articles)
        run_config |= {
            "prompt_version": AGENT_PROMPT_VERSION,
            "max_steps": MAX_STEPS,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "search_results": SEARCH_RESULTS,
            "snippet_words": SNIPPET_WORDS,
            **settings,
        }
    if strategy == "s3":
        from support_rag.long_context import claude_token_estimator

        retriever, settings = answer_retriever(DEFAULT_RETRIEVER, "article", articles)
        count = claude_token_estimator()
        run_config |= {
            "k": k,
            "unit": "article",
            "prompt_budget": PROMPT_BUDGET,
            "claude_tokens_per_tiktoken": CLAUDE_TOKENS_PER_TIKTOKEN,
            **settings,
        }
        name = f"s3-k{k}-{effort}_{split}"
    run_dir = start_run(name, run_config, runs_dir)

    results = []
    with LLMCache() as cache:
        llm = LLMClient(cache)
        for question in questions:
            if strategy == "s2":
                toolbox = ToolBox(retriever, articles_by_id)
                result = answer_with_agent(llm, question, toolbox, effort=effort)
            elif strategy == "s3":
                context = context_for(
                    strategy,
                    question,
                    articles_by_id=articles_by_id,
                    retriever=retriever,
                    k=k,
                    count=count,
                )
                result = answer_long_context(
                    llm, question, context, k=k, count=count, effort=effort
                )
            else:
                context = context_for(
                    strategy, question, articles_by_id=articles_by_id, retriever=retriever, k=k
                )
                result = answer_question(
                    llm, question, context, effort=effort, sections=unit != "article"
                )
            append_result(run_dir, asdict(result))
            results.append(result)
    write_summary(run_dir, summarize_answers(results))
    return run_dir
