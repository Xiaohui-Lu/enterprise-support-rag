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
from typing import TYPE_CHECKING, Any, Protocol

from support_rag import config
from support_rag.bm25 import STOPWORDS, BM25Retriever, tokenize
from support_rag.metrics import all_gold_at_k, recall_at_k, reciprocal_rank
from support_rag.records import Article, Hit, Question
from support_rag.runs import append_result, start_run, write_summary
from support_rag.splits import load_splits
from support_rag.wixqa import QA_SUBSETS, load_articles, load_questions

if TYPE_CHECKING:
    from support_rag.dense import SectionRetriever

KS = (1, 5, 10, 20)  # context sizes the strategies use (D-017)
DEPTH = max(KS)  # retrieved once per question; every metric comes from this list
EMBEDDING_MODEL = "voyage-4-nano"  # the default retriever's model (D-036)

# Split name on the command line -> lists in splits.json.
SPLITS = {
    "dev": ("dev_real", "dev_synthetic"),
    "dev_real": ("dev_real",),  # the 50 real dev questions, for answer runs
    "synthetic_reference": ("synthetic_reference",),
    "test": ("test",),  # milestone runs only (D-004, D-034)
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
    search_ms: float | None = None  # the database query alone, if the retriever reports it


def evaluate_question(retriever: Retriever, question: Question) -> QuestionResult:
    start = time.perf_counter()
    hits = retriever.search(question.question, k=DEPTH)
    latency_ms = (time.perf_counter() - start) * 1000
    # A retriever backed by a database reports its query time on its own (D-041).
    search_ms = getattr(retriever, "last_search_ms", None)

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
        search_ms=None if search_ms is None else round(search_ms, 3),
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


def _bm25_retriever(
    articles: Sequence[Article], *, stem: bool
) -> tuple[BM25Retriever, dict[str, Any]]:
    """The retriever, and its settings for the run record."""
    start = time.perf_counter()
    retriever = BM25Retriever(articles, tokenizer=partial(tokenize, stem=stem))
    return retriever, {
        "retriever": {"name": "bm25", **retriever.params},
        "tokenizer": {"stem": stem, "stopwords": f"lucene-en-{len(STOPWORDS)}"},
        "index_seconds": round(time.perf_counter() - start, 2),
    }


def _dense_retriever(articles: Sequence[Article], model: str) -> tuple[Retriever, dict[str, Any]]:
    """The retriever, and its settings for the run record (D-035)."""
    from support_rag.dense import DenseRetriever
    from support_rag.embeddings import MODELS, SentenceTransformerEncoder, article_vectors

    encoder = SentenceTransformerEncoder(MODELS[model])
    start = time.perf_counter()
    vectors, stored = article_vectors(articles, MODELS[model], encoder)
    return DenseRetriever(articles, encoder, vectors), {
        "retriever": {"name": "dense", **encoder.params},
        # Embedding the articles, or loading their stored vectors.
        "index_seconds": round(time.perf_counter() - start, 2),
        "vectors_stored": stored,
    }


def _section_retriever(
    articles: Sequence[Article], model: str, unit: str, *, title: bool = True
) -> tuple[SectionRetriever, dict[str, Any]]:
    """The retriever, and its settings for the run record (D-039)."""
    from support_rag.dense import SectionRetriever
    from support_rag.embeddings import MODELS, SentenceTransformerEncoder, section_vectors
    from support_rag.sections import cut_articles

    encoder = SentenceTransformerEncoder(MODELS[model])
    start = time.perf_counter()
    sections = cut_articles(articles, unit, title=title)
    vectors, stored = section_vectors(sections, MODELS[model], encoder)
    return SectionRetriever(sections, encoder, vectors), {
        "retriever": {"name": "dense", **encoder.params, "title": title},
        "unit": unit,
        "n_sections": len(sections),
        "index_seconds": round(time.perf_counter() - start, 2),
        "vectors_stored": stored,
    }


def _pg_retriever(
    articles: Sequence[Article], model: str, index: str
) -> tuple[Retriever, dict[str, Any]]:
    """The Postgres retriever, and its settings for the run record (D-041). The connection
    string holds a password, so it isn't recorded."""
    from support_rag.config import database_url
    from support_rag.embeddings import MODELS, SentenceTransformerEncoder
    from support_rag.store import HNSW_SETTINGS, PgVectorRetriever, connect

    encoder = SentenceTransformerEncoder(MODELS[model])
    start = time.perf_counter()
    connection = connect(database_url())
    retriever = PgVectorRetriever(
        connection, encoder, model=MODELS[model], articles=articles, index=index
    )
    postgres, pgvector = connection.execute(
        "SELECT current_setting('server_version'), extversion "
        "FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    store: dict[str, Any] = {
        "name": "postgres",
        "index": index,
        "plan": retriever.plan,
        "postgres": postgres,
        "pgvector": pgvector,
    }
    if index == "hnsw":
        store["hnsw"] = HNSW_SETTINGS
    return retriever, {
        "retriever": {"name": "dense", **encoder.params, "store": store},
        "index_seconds": round(time.perf_counter() - start, 2),  # connecting and checking
        "vectors_stored": True,
    }


def answer_retriever(
    name: str, unit: str, articles: Sequence[Article]
) -> tuple[Retriever, dict[str, Any]]:
    """What S1 searches, and its settings for the run record (D-042): BM25 or dense search.
    Whole articles come from Postgres (D-041); sections stay in memory for dev runs."""
    if name == "bm25":
        return _bm25_retriever(articles, stem=True)
    if unit == "article":
        return _pg_retriever(articles, EMBEDDING_MODEL, "exact")
    return _section_retriever(articles, EMBEDDING_MODEL, unit)


def run_bm25_eval(split: str, *, stem: bool = True, runs_dir: Path = config.RUNS_DIR) -> Path:
    """Evaluate BM25 on a split and save the run. Returns the run directory."""
    articles = load_articles()
    retriever, settings = _bm25_retriever(articles, stem=stem)
    return _run_eval(
        f"bm25-{'stem' if stem else 'nostem'}_{split}",
        retriever,
        {**settings, "n_articles": len(articles)},
        split,
        runs_dir,
    )


def run_dense_eval(
    split: str,
    *,
    model: str,
    unit: str = "article",
    title: bool = True,
    store: str = "numpy",
    index: str = "exact",
    runs_dir: Path = config.RUNS_DIR,
) -> Path:
    """Evaluate dense retrieval with one embedding model and save the run (D-035). A unit
    other than "article" searches that kind of section instead (D-039). The "postgres"
    store searches the vectors loaded by `srag db-load`, exactly or through HNSW (D-041)."""
    articles = load_articles()
    if store == "postgres":
        retriever, settings = _pg_retriever(articles, model, index)
        name = f"dense-pg-{index}-{model}_{split}"
    elif unit == "article":
        retriever, settings = _dense_retriever(articles, model)
        name = f"dense-{model}_{split}"
    else:
        retriever, settings = _section_retriever(articles, model, unit, title=title)
        name = f"dense-{unit}{'' if title else '-notitle'}-{model}_{split}"
    return _run_eval(
        name,
        retriever,
        {**settings, "n_articles": len(articles)},
        split,
        runs_dir,
    )


def run_hybrid_eval(
    split: str, *, model: str, fusion: str, runs_dir: Path = config.RUNS_DIR
) -> Path:
    """Evaluate BM25 plus dense retrieval, merged by one fusion method (D-037)."""
    from support_rag.hybrid import CANDIDATE_DEPTH, FUSION_SETTINGS, FUSIONS, HybridRetriever

    articles = load_articles()
    dense, dense_settings = _dense_retriever(articles, model)
    keyword, keyword_settings = _bm25_retriever(articles, stem=True)
    return _run_eval(
        f"hybrid-{fusion}-{model}_{split}",
        HybridRetriever(dense, keyword, FUSIONS[fusion]),
        {
            "retriever": {
                "name": "hybrid",
                "fusion": fusion,
                **FUSION_SETTINGS[fusion],
                "candidate_depth": CANDIDATE_DEPTH,
                "dense": dense_settings["retriever"],
                "keyword": keyword_settings["retriever"],
            },
            "tokenizer": keyword_settings["tokenizer"],
            "n_articles": len(articles),
            "index_seconds": round(
                dense_settings["index_seconds"] + keyword_settings["index_seconds"], 2
            ),
            "vectors_stored": dense_settings["vectors_stored"],
        },
        split,
        runs_dir,
    )


def run_rerank_eval(
    split: str,
    *,
    model: str,
    reranker: str,
    candidates: str = "dense",
    runs_dir: Path = config.RUNS_DIR,
) -> Path:
    """Evaluate a cross-encoder over dense's top candidates, plus BM25's if asked (D-038)."""
    from support_rag.rerank import (
        CANDIDATE_DEPTH,
        RERANKERS,
        CrossEncoderScorer,
        RerankingRetriever,
    )

    articles = load_articles()
    dense, dense_settings = _dense_retriever(articles, model)
    scorer = CrossEncoderScorer(RERANKERS[reranker])
    retrievers: list[Retriever] = [dense]
    settings: dict[str, Any] = {
        "retriever": {
            "name": "rerank",
            "reranker": scorer.params,
            "candidates": candidates,
            "candidate_depth": CANDIDATE_DEPTH,
            "dense": dense_settings["retriever"],
        },
        "n_articles": len(articles),
        "index_seconds": dense_settings["index_seconds"],
        "vectors_stored": dense_settings["vectors_stored"],
    }
    if candidates == "both":
        keyword, keyword_settings = _bm25_retriever(articles, stem=True)
        retrievers.append(keyword)
        settings["retriever"]["keyword"] = keyword_settings["retriever"]
        settings["tokenizer"] = keyword_settings["tokenizer"]
        settings["index_seconds"] += keyword_settings["index_seconds"]
    return _run_eval(
        f"rerank-{reranker}-{candidates}-{model}_{split}",
        RerankingRetriever(articles, retrievers, scorer),
        settings,
        split,
        runs_dir,
    )


def run_section_rerank_eval(
    split: str, *, model: str, reranker: str, unit: str, runs_dir: Path = config.RUNS_DIR
) -> Path:
    """Evaluate a cross-encoder over the top 50 articles' best sections (D-039)."""
    from support_rag.rerank import (
        CANDIDATE_DEPTH,
        RERANKERS,
        CrossEncoderScorer,
        SectionRerankingRetriever,
    )

    articles = load_articles()
    sections, section_settings = _section_retriever(articles, model, unit)
    scorer = CrossEncoderScorer(RERANKERS[reranker])
    return _run_eval(
        f"rerank-{reranker}-{unit}-{model}_{split}",
        SectionRerankingRetriever(sections, scorer),
        {
            **section_settings,
            "retriever": {
                "name": "rerank",
                "reranker": scorer.params,
                "candidates": "dense",
                "candidate_depth": CANDIDATE_DEPTH,
                "dense": section_settings["retriever"],
            },
            "n_articles": len(articles),
        },
        split,
        runs_dir,
    )


def _run_eval(
    name: str, retriever: Retriever, settings: dict[str, Any], split: str, runs_dir: Path
) -> Path:
    questions = questions_for_split(split)
    run_dir = start_run(
        name,
        {
            "kind": "retrieval",
            "unit": "article",  # the search unit; the metrics always count articles (D-012)
            **settings,
            "depth": DEPTH,
            "ks": list(KS),
            "split": split,
            "n_questions": len(questions),
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
