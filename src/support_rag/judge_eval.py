"""Judge runs: grade every answer of an answer run and save the verdicts as a new run."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from support_rag import config
from support_rag.answer_eval import LLM
from support_rag.judge import JUDGE_PROMPT_VERSION, build_judge_request, parse_verdict
from support_rag.llm import JUDGE_MODEL, cost_usd
from support_rag.records import Article, Question
from support_rag.runs import append_result, load_config, load_results, start_run, write_summary


@dataclass(frozen=True)
class JudgedAnswer:
    """One line of questions.jsonl in a judge run."""

    question_id: str
    subset: str
    multi_article: bool
    correctness: str | None  # None only if the judge failed (cut off or refused)
    grounded: str | None
    reasoning: str
    judged: bool  # False for declines, which are graded incorrect without a judge call
    stop_reason: str | None
    usage: dict[str, int]
    cost_usd: float
    cached: bool


def judge_request(
    answer: Mapping[str, Any],
    question: Question,
    articles_by_id: Mapping[str, Article],
    *,
    effort: str,
    model: str = JUDGE_MODEL,
) -> dict[str, Any]:
    """The judge's request for one saved answer (a line of an answer run's questions.jsonl)."""
    # The answer cites articles by their number in its own prompt: [1] is context_ids[0].
    numbers = {article_id: n for n, article_id in enumerate(answer["context_ids"], start=1)}
    cited = [(numbers[a], articles_by_id[a]) for a in answer["cited_ids"]]
    return build_judge_request(
        question.question, question.answer, answer["text"], cited, effort=effort, model=model
    )


def judge_answer(
    llm: LLM,
    answer: Mapping[str, Any],
    question: Question,
    articles_by_id: Mapping[str, Article],
    *,
    effort: str,
    model: str = JUDGE_MODEL,
) -> JudgedAnswer:
    """Grade one saved answer (a line of an answer run's questions.jsonl)."""
    base = {
        "question_id": question.id,
        "subset": question.subset,
        "multi_article": question.is_multi_article,
    }
    if answer["declined"]:
        return JudgedAnswer(
            **base, correctness="incorrect", grounded="n/a", reasoning="Declined (NOT_FOUND).",
            judged=False, stop_reason=None, usage={}, cost_usd=0.0, cached=False,
        )  # fmt: skip

    request = judge_request(answer, question, articles_by_id, effort=effort, model=model)
    result = llm.complete(request)
    cost = round(cost_usd(result.model, result.usage, batch=result.batch), 6)
    if not result.complete:
        return JudgedAnswer(
            **base, correctness=None, grounded=None, reasoning="", judged=True,
            stop_reason=result.stop_reason, usage=result.usage, cost_usd=cost,
            cached=result.cached,
        )  # fmt: skip
    verdict = parse_verdict(result.text)
    return JudgedAnswer(
        **base,
        correctness=verdict.correctness,
        grounded=verdict.grounded,
        reasoning=verdict.reasoning,
        judged=True,
        stop_reason=result.stop_reason,
        usage=result.usage,
        cost_usd=cost,
        cached=result.cached,
    )


def summarize_verdicts(results: Sequence[JudgedAnswer]) -> dict[str, Any]:
    n = len(results)
    grades = Counter(r.correctness for r in results)
    grounded = Counter(r.grounded for r in results if r.judged and r.grounded != "n/a")
    return {
        "n": n,
        "correct": grades["correct"],
        "partial": grades["partial"],
        "incorrect": grades["incorrect"],
        "judge_failed": grades[None],
        "pct_correct": round(100 * grades["correct"] / n, 1) if n else None,
        "grounded": dict(grounded),
        "judge_calls": sum(r.judged for r in results),
        "cost_usd": round(sum(r.cost_usd for r in results), 4),
        "paid_this_run_usd": round(sum(r.cost_usd for r in results if not r.cached), 4),
    }


def run_judge(answers_run: Path, *, effort: str, runs_dir: Path = config.RUNS_DIR) -> Path:
    from support_rag.llm import LLMClient
    from support_rag.llm_cache import LLMCache
    from support_rag.wixqa import QA_SUBSETS, load_articles, load_questions

    answers_config = load_config(answers_run)
    questions = {q.id: q for subset in QA_SUBSETS for q in load_questions(subset)}
    articles_by_id = {article.id: article for article in load_articles()}

    run_dir = start_run(
        f"judge-{effort}_{answers_config['run_id']}",
        {
            "kind": "judge",
            "answers_run": answers_config["run_id"],
            "strategy": answers_config["strategy"],  # recorded here, never shown to the judge
            "model": JUDGE_MODEL,
            "effort": effort,
            "prompt_version": JUDGE_PROMPT_VERSION,
        },
        runs_dir,
    )
    results = []
    with LLMCache() as cache:
        llm = LLMClient(cache)
        for answer in load_results(answers_run):
            question = questions[answer["question_id"]]
            result = judge_answer(llm, answer, question, articles_by_id, effort=effort)
            append_result(run_dir, asdict(result))
            results.append(result)
    write_summary(run_dir, summarize_verdicts(results))
    return run_dir
