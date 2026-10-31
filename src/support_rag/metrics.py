"""Retrieval metrics for one question (D-017).

Each function scores one ranked list against one question's gold articles. The harness
averages them per question, so every question counts once, however many gold articles
it has.

`ranked_ids` is best first, with no duplicates (sections are mapped back to their article
and deduplicated before scoring). `gold_ids` is never empty: every WixQA question has at
least one gold article.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence


def recall_at_k(ranked_ids: Sequence[str], gold_ids: Collection[str], k: int) -> float:
    """Share of the gold articles found in the top k: 0.0 to 1.0.

    gold {A, B}, ranked [C, A, D, ...], k=5  ->  0.5
    """
    top_k_ids = ranked_ids[:k]
    recall_ids = [x for x in gold_ids if x in top_k_ids]
    return len(recall_ids) / len(gold_ids)


def all_gold_at_k(ranked_ids: Sequence[str], gold_ids: Collection[str], k: int) -> bool:
    """True only if every gold article is in the top k.

    gold {A, B}, ranked [C, A, D, ...], k=5  ->  False (B is missing)
    """
    top_k_ids = ranked_ids[:k]
    return set(gold_ids).issubset(set(top_k_ids))


def reciprocal_rank(ranked_ids: Sequence[str], gold_ids: Collection[str]) -> float:
    """1 / rank of the first gold article (ranks start at 1); 0.0 if none is in the list.

    The cutoff is the length of the list: the harness retrieves the top 20, so averaging
    this over questions gives MRR@20.

    gold {A, B}, ranked [C, B, D, E, F, A]  ->  0.5 (B at rank 2; A doesn't matter)
    """
    for rank, article_id in enumerate(ranked_ids, start=1):
        if article_id in gold_ids:
            return 1 / rank
    return 0
