"""The context-rot test (RQ5, D-045): gold articles buried among distractors, sized by tokens.

Each question's gold articles go into prompts of an estimated 10K, 40K and 90K tokens, filled
with hard distractors (dense search's top-ranked non-gold articles) or random ones. Each gold
article sits at one random depth per question, the same in every arm. The gold-only arm is
S*, already run. Answers go through the batch API, since latency isn't what's measured.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from support_rag import config
from support_rag.answer import PROMPT_VERSION, SYSTEM_WITH_ARTICLES, build_request, numbered
from support_rag.answer_eval import (
    DEFAULT_EFFORT,
    DEFAULT_RETRIEVER,
    answer_question,
    check_test_run,
    pick_evenly,
    summarize_answers,
)
from support_rag.long_context import (
    CLAUDE_TOKENS_PER_TIKTOKEN,
    TokenCounter,
    capped_context,
    estimate_request_tokens,
)
from support_rag.records import Article, Question
from support_rag.retrieval_eval import Retriever, answer_retriever, questions_for_split
from support_rag.runs import append_result, start_run, write_summary

SIZES = (10_000, 40_000, 90_000)  # estimated prompt tokens, under the 100K price line
DISTRACTORS = ("hard", "random")
SEED = 20261113
HARD_POOL = 200  # about 180K tokens of look-alike articles, twice the largest size
RANDOM_POOL = 400  # about 290K tokens


def question_rng(question_id: str, purpose: str) -> random.Random:
    """Randomness for one question and one purpose. It doesn't depend on which other
    questions are in the run, so a pilot's contexts are exactly the full run's."""
    return random.Random(f"{SEED}:{question_id}:{purpose}")


def gold_depths(question: Question) -> tuple[float, ...]:
    """One depth per gold article: the share of the distractors that go in front of it."""
    rng = question_rng(question.id, "depths")
    return tuple(rng.random() for _ in question.gold_ids)


def hard_pool(
    question: Question, retriever: Retriever, articles_by_id: Mapping[str, Article]
) -> list[Article]:
    """The articles dense search ranks highest for the question, without the gold, best first."""
    hits = retriever.search(question.question, HARD_POOL + len(question.gold_ids))
    pool = [articles_by_id[h.article_id] for h in hits if h.article_id not in question.gold_ids]
    return pool[:HARD_POOL]


def random_pool(question: Question, articles_by_id: Mapping[str, Article]) -> list[Article]:
    """Non-gold articles drawn at random for this question."""
    candidates = sorted(a for a in articles_by_id if a not in question.gold_ids)
    drawn = question_rng(question.id, "random").sample(candidates, RANDOM_POOL)
    return [articles_by_id[article_id] for article_id in drawn]


def rot_context(
    gold: Sequence[Article],
    depths: Sequence[float],
    pool: Sequence[Article],
    *,
    size: int,
    count: TokenCounter,
    used: int,
    rng: random.Random,
) -> list[Article]:
    """The gold articles buried among distractors, in an estimated prompt of `size` tokens.

    `used` is the prompt before any article: the instructions and the question. Distractors
    come from the start of `pool` until the next one would pass `size`, then get shuffled.
    Each gold article goes in with round(depth × n) distractors in front of it, where n is
    the number of distractors.

    Gold G at depth 0.5 among 4 distractors  ->  2 distractors, G, 2 distractors
    """
    gold_tokens = sum(count(numbered(n, article)) for n, article in enumerate(gold, start=1))
    distractors = capped_context(pool, budget=size, count=count, used=used + gold_tokens)
    rng.shuffle(distractors)
    context = list(distractors)
    # Shallowest first; `placed` counts the gold articles already in front of this one.
    by_depth = sorted(zip(depths, gold, strict=True), key=lambda pair: pair[0])
    for placed, (depth, article) in enumerate(by_depth):
        context.insert(round(depth * len(distractors)) + placed, article)
    return context


def run_context_rot(
    distractors: str,
    size: int,
    split: str,
    *,
    effort: str = DEFAULT_EFFORT,
    limit: int | None = None,
    milestone: str | None = None,
    batch_id: str | None = None,
    runs_dir: Path = config.RUNS_DIR,
) -> Path:
    """Answer one arm of the test through the batch API and save the run (D-045)."""
    if distractors not in DISTRACTORS or size not in SIZES:
        raise ValueError(f"an arm is one of {DISTRACTORS} at one of {SIZES} tokens")
    if split == "test":
        check_test_run(milestone)
    from support_rag.llm import GENERATOR_MODEL, LLMClient, run_batch
    from support_rag.llm_cache import LLMCache
    from support_rag.long_context import claude_token_estimator
    from support_rag.wixqa import load_articles

    questions = pick_evenly(questions_for_split(split), limit)
    articles = load_articles()
    articles_by_id = {article.id: article for article in articles}
    count = claude_token_estimator()
    settings: dict[str, Any] = {}
    retriever = None
    if distractors == "hard":
        retriever, settings = answer_retriever(DEFAULT_RETRIEVER, "article", articles)

    contexts = {}
    for question in questions:
        if retriever is not None:
            pool = hard_pool(question, retriever, articles_by_id)
        else:
            pool = random_pool(question, articles_by_id)
        contexts[question.id] = rot_context(
            [articles_by_id[g] for g in question.gold_ids],
            gold_depths(question),
            pool,
            size=size,
            count=count,
            used=count(SYSTEM_WITH_ARTICLES) + count(question.question),
            rng=question_rng(question.id, f"{distractors}-{size}"),
        )
    requests = [build_request(q, contexts[q.id], effort=effort) for q in questions]

    with LLMCache() as cache:
        batch = run_batch(requests, cache, batch_id=batch_id)
        run_dir = start_run(
            f"rot-{distractors}-{size // 1000}k-{effort}_{split}",
            {
                "kind": "answers",
                "strategy": "rot",
                "distractors": distractors,
                "size": size,
                "model": GENERATOR_MODEL,
                "effort": effort,
                "prompt_version": PROMPT_VERSION,
                "split": split,
                "n_questions": len(questions),
                "limit": limit,
                "milestone": milestone,
                "seed": SEED,
                "pools": {"hard": HARD_POOL, "random": RANDOM_POOL},
                "claude_tokens_per_tiktoken": CLAUDE_TOKENS_PER_TIKTOKEN,
                "batch_id": batch["batch_id"],
                **settings,
            },
            runs_dir,
        )
        llm = LLMClient(cache)  # replays the batch; a result it didn't cache is called again
        results = []
        for question in questions:
            context = contexts[question.id]
            result = answer_question(llm, question, context, effort=effort)
            request = build_request(question, context, effort=effort)
            result = replace(
                result,
                rot={
                    "distractors": distractors,
                    "size": size,
                    "n_distractors": len(context) - len(question.gold_ids),
                    "gold_depths": list(gold_depths(question)),
                    "gold_positions": [result.context_ids.index(g) + 1 for g in question.gold_ids],
                    "estimated_prompt_tokens": estimate_request_tokens(request, count),
                },
            )
            append_result(run_dir, asdict(result))
            results.append(result)
    write_summary(run_dir, summarize_answers(results) | {"batch": batch})
    return run_dir
