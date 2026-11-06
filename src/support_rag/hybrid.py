"""Hybrid retrieval: merge a dense and a keyword ranking into one (D-037).

Their scores can't be added as they are: BM25's have no upper limit, while cosine
similarities run from about 0 to 1. RRF avoids the problem by using only ranks; the
weighted score scales each list to 0-1 first. Neither is tuned.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from support_rag.records import Hit

if TYPE_CHECKING:
    from support_rag.retrieval_eval import Retriever

CANDIDATE_DEPTH = 100  # how many articles each retriever supplies
RRF_K = 60  # the usual default; a larger k flattens the gap between high and low ranks
DENSE_WEIGHT = 0.5

Fusion = Callable[[Sequence[Hit], Sequence[Hit]], list[Hit]]


def rrf(dense: Sequence[Hit], keyword: Sequence[Hit], *, k: int = RRF_K) -> list[Hit]:
    """Reciprocal Rank Fusion: each list gives an article 1 / (k + rank), rank starting at 1.

    An article in both lists collects from both, so agreement counts for a lot: at k=60,
    rank 30 in both lists (2/90) beats rank 1 in one list (1/61). The input scores are
    ignored, only their order counts.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in (dense, keyword):
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.article_id] += 1 / (k + rank)
    return _best_first(scores)


def weighted(
    dense: Sequence[Hit], keyword: Sequence[Hit], *, dense_weight: float = DENSE_WEIGHT
) -> list[Hit]:
    """A weighted sum of each list's scores, scaled to 0-1. An article missing from a list
    gets 0 from it, the same as that list's weakest article."""
    scores: dict[str, float] = defaultdict(float)
    for ranking, weight in ((dense, dense_weight), (keyword, 1 - dense_weight)):
        for article_id, score in _scaled(ranking).items():
            scores[article_id] += weight * score
    return _best_first(scores)


def _scaled(hits: Sequence[Hit]) -> dict[str, float]:
    """Min-max scaling: the list's best score becomes 1 and its lowest 0. If every score is
    the same, including a list of one, there's no range to scale, so each counts as 1."""
    if not hits:
        return {}
    low, high = min(hit.score for hit in hits), max(hit.score for hit in hits)
    if high == low:
        return {hit.article_id: 1.0 for hit in hits}
    return {hit.article_id: (hit.score - low) / (high - low) for hit in hits}


def _best_first(scores: dict[str, float]) -> list[Hit]:
    # Article ID settles equal scores, so the same input always gives the same ranking.
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [Hit(article_id, score) for article_id, score in ranked]


FUSIONS: dict[str, Fusion] = {"rrf": rrf, "weighted": weighted}
# For the run record.
FUSION_SETTINGS: dict[str, dict[str, Any]] = {
    "rrf": {"k": RRF_K},
    "weighted": {"dense_weight": DENSE_WEIGHT, "scaling": "min-max"},
}


class HybridRetriever:
    """Asks both retrievers for their top candidates, then merges the two lists."""

    def __init__(
        self,
        dense: Retriever,
        keyword: Retriever,
        fusion: Fusion,
        *,
        depth: int = CANDIDATE_DEPTH,
    ) -> None:
        self._dense = dense
        self._keyword = keyword
        self._fusion = fusion
        self._depth = depth

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """The top k articles for a query, best first."""
        dense_hits = self._dense.search(query, k=self._depth)
        keyword_hits = self._keyword.search(query, k=self._depth)
        return self._fusion(dense_hits, keyword_hits)[:k]
