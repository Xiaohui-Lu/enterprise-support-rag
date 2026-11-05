"""Dense retrieval: rank articles by how close their vectors are to the question's (D-035).

Search is exact: every article is scored for every question. With 6,221 articles that's
one matrix-vector product, a few milliseconds, so there's no approximate index to tune.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from support_rag.embeddings import Encoder
from support_rag.records import Article, Hit


def top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """The k rows of `matrix` most similar to `query`, best first, as (row, similarity).

    `query` has shape (d,) and `matrix` has shape (n, d); every vector has length 1, so
    the cosine similarity of the query and row i is their dot product. If k is larger
    than n, return all n rows. Similarities are plain Python floats.

    query (1, 0), rows (0, 1), (0.6, 0.8), (1, 0), k=2  ->  [(2, 1.0), (1, 0.6)]

    Equal similarities keep row order, so the result is the same on every run.
    """
    similarities = matrix @ query  # shape (n,): row i's dot product with the query
    # A stable sort of the negated scores: best first, ties in row order. Sorting all n
    # rows is fine at 6,221; at millions, np.argpartition or an index would avoid it.
    best_first = np.argsort(-similarities, kind="stable")[:k]
    # numpy ints and floats don't serialize to JSON, so convert to plain Python ones.
    return [(int(row), float(similarities[row])) for row in best_first]


class DenseRetriever:
    """Whole articles, ranked by cosine similarity to the question."""

    def __init__(self, articles: Sequence[Article], encoder: Encoder, vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(articles):
            raise ValueError(f"{vectors.shape[0]} vectors for {len(articles)} articles")
        self._ids = [article.id for article in articles]
        self._encoder = encoder
        self._vectors = vectors

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """The top k articles for a query, best first."""
        (query_vector,) = self._encoder.encode_queries([query])
        return [Hit(self._ids[row], score) for row, score in top_k(query_vector, self._vectors, k)]
