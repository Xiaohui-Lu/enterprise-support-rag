"""Retrieval evaluation: run a retriever over a split, score every question, save a run.

No LLM calls, so it's free: dev and the synthetic reference set can be run as often as
needed. The test set is not offered here; it runs only at milestones (D-004).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from statistics import fmean
from typing import Protocol

from support_rag import config
from support_rag.bm25 import STOPWORDS, BM25Retriever, tokenize
from support_rag.metrics import all_gold_at_k, recall_at_k, reciprocal_rank
from support_rag.records import Hit, Question
from support_rag.runs import append_result, start_run, write_summary
from support_rag.splits import load_splits
from support_rag.wixqa import QA_SUBSETS, load_articles, load_questions

KS = (1, 5, 10, 20)  # context sizes the strategies use (D-017)
DEPTH = max(KS)  # retrieved once per question; every metric comes from this list

# Split name on the command line -> lists in splits.json.
SPLITS = {
    "dev": ("dev_real", "dev_synthetic"),
    "synthetic_reference": ("synthetic_reference",),
}


class Retriever(Protocol):
    def search(self, query: str, k: int = 10) -> list[Hit]: ...


@dataclass(frozen=True)
class QuestionResult:
    """One line of questions.jsonl."""

    question_id: str
    subset: str
    multi_article: bool
    gold_ids: tuple[str, ...]
    ranked_ids: tuple[str, ...]  # top DEPTH, best first
    scores: tuple[float, ...]
    metrics: dict[str, float]  # "recall@5", "all_gold@5", "mrr@20" (reciprocal rank), ...
    latency_ms: float


def evaluate_question(retriever: Retriever, question: Question) -> QuestionResult:
    start = time.perf_counter()
    hits = retriever.search(question.question, k=DEPTH)
    latency_ms = (time.perf_counter() - start) * 1000

    ranked = [hit.article_id for hit in hits]
    gold = question.gold_ids
    metrics = {f"recall@{k}": recall_at_k(ranked, gold, k) for k in KS}
    metrics |= {f"all_gold@{k}": float(all_gold_at_k(ranked, gold, k)) for k in KS}
    # Per question this is the reciprocal rank; summarize() averages it into MRR.
    metrics[f"mrr@{DEPTH}"] = float(reciprocal_rank(ranked, gold))

    return QuestionResult(
        question_id=question.id,
        subset=question.subset,
        multi_article=question.is_multi_article,
        gold_ids=gold,
        ranked_ids=tuple(ranked),
        scores=tuple(round(hit.score, 4) for hit in hits),
        metrics=metrics,
        latency_ms=round(latency_ms, 3),
    )


def summarize(results: Sequence[QuestionResult]) -> dict[str, dict[str, float]]:
    """Average every metric over the questions in each group (per question, D-017).

    Groups: "all"; one per subset ("expertwritten", "simulated", "synthetic"); "single"
    and "multi" for single- and multi-article questions. A group with no questions is
    left out. Each group also has "n", its number of questions.

    Two questions with recall@5 of 1.0 and 0.0 -> {"all": {"n": 2, "recall@5": 0.5, ...}}
    """
    groups = {
        "all": list(results),
        **{subset: [r for r in results if r.subset == subset] for subset in QA_SUBSETS},
        "single": [r for r in results if not r.multi_article],
        "multi": [r for r in results if r.multi_article],
    }
    return {name: _average(members) for name, members in groups.items() if members}


def _average(results: Sequence[QuestionResult]) -> dict[str, float]:
    # Every result has the same metric names, so the first one lists them.
    names = results[0].metrics
    return {"n": len(results)} | {name: fmean(r.metrics[name] for r in results) for name in names}


def questions_for_split(split: str, raw_dir: Path = config.RAW_DIR) -> list[Question]:
    wanted = {qid for key in SPLITS[split] for qid in load_splits()[key]}
    every = [q for subset in QA_SUBSETS for q in load_questions(subset, raw_dir)]
    return [q for q in every if q.id in wanted]


def run_bm25_eval(split: str, *, stem: bool = True, runs_dir: Path = config.RUNS_DIR) -> Path:
    """Evaluate BM25 on a split and save the run. Returns the run directory."""
    questions = questions_for_split(split)
    articles = load_articles()

    start = time.perf_counter()
    retriever = BM25Retriever(articles, tokenizer=partial(tokenize, stem=stem))
    index_seconds = time.perf_counter() - start

    run_dir = start_run(
        f"bm25-{'stem' if stem else 'nostem'}_{split}",
        {
            "kind": "retrieval",
            "retriever": {"name": "bm25", **retriever.params},
            "tokenizer": {"stem": stem, "stopwords": f"lucene-en-{len(STOPWORDS)}"},
            "unit": "article",
            "depth": DEPTH,
            "ks": list(KS),
            "split": split,
            "n_questions": len(questions),
            "n_articles": len(articles),
            "index_seconds": round(index_seconds, 2),
        },
        runs_dir,
    )
    results = []
    for question in questions:
        result = evaluate_question(retriever, question)
        append_result(run_dir, asdict(result))
        results.append(result)
    write_summary(run_dir, summarize(results))
    return run_dir


def format_summary(summary: dict[str, dict[str, float]]) -> str:
    """A Markdown table: one row per group, one column per metric."""
    metrics = [name for name in summary["all"] if name != "n"]
    lines = [
        "| Group | n | " + " | ".join(metrics) + " |",
        "|---|---:|" + "---:|" * len(metrics),
    ]
    for group, values in summary.items():
        cells = " | ".join(f"{values[name]:.3f}" for name in metrics)
        lines.append(f"| {group} | {values['n']} | {cells} |")
    return "\n".join(lines)
