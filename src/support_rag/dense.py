"""Dense retrieval: rank articles by how close their vectors are to the question's (D-035).

Search is exact: every article is scored for every question. With 6,221 articles that's
one matrix-vector product, a few milliseconds, so there's no approximate index to tune.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from itertools import islice

import numpy as np

from support_rag.embeddings import Encoder
from support_rag.records import Article, Hit, Section


def ranked_rows(query: np.ndarray, matrix: np.ndarray) -> Iterator[tuple[int, float]]:
    """Every row of `matrix`, most similar to `query` first, as (row, similarity). A
    generator, so a caller that stops early doesn't pay to convert the rest.

    Equal similarities keep row order, so the result is the same on every run.
    """
    similarities = matrix @ query  # shape (n,): row i's dot product with the query
    # A stable sort of the negated scores: best first, ties in row order. Sorting all n
    # rows is fine at this size; at millions, np.argpartition or an index would avoid it.
    for row in np.argsort(-similarities, kind="stable"):
        # numpy ints and floats don't serialize to JSON, so convert to plain Python ones.
        yield int(row), float(similarities[row])


def top_k(query: np.ndarray, matrix: np.ndarray, k: int) -> list[tuple[int, float]]:
    """The k rows of `matrix` most similar to `query`, best first, as (row, similarity).

    `query` has shape (d,) and `matrix` has shape (n, d); every vector has length 1, so
    the cosine similarity of the query and row i is their dot product. If k is larger
    than n, return all n rows. Similarities are plain Python floats.

    query (1, 0), rows (0, 1), (0.6, 0.8), (1, 0), k=2  ->  [(2, 1.0), (1, 0.6)]

    Equal similarities keep row order, so the result is the same on every run.
    """
    return list(islice(ranked_rows(query, matrix), k))


def best_per_article(ranked: Iterable[Hit], k: int) -> list[Hit]:
    """Each article once, at its best section, best first, at most k articles.

    `ranked` holds sections, best first, so the first time an article appears is its best
    section, and every later one is ignored. A dict keeps the order its keys were added in.

    A .9, B .8, A .7, C .6, k=2  ->  A .9, B .8

    `ranked` is only read as far as needed, so a generator over tens of thousands of
    sections stops as soon as it has k articles.
    """
    best: dict[str, Hit] = {}
    for hit in ranked:
        if len(best) == k:
            break
        best.setdefault(hit.article_id, hit)
    return list(best.values())


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


class SectionRetriever:
    """Articles ranked by their best section's cosine similarity to the question (D-039)."""

    def __init__(self, sections: Sequence[Section], encoder: Encoder, vectors: np.ndarray) -> None:
        if vectors.shape[0] != len(sections):
            raise ValueError(f"{vectors.shape[0]} vectors for {len(sections)} sections")
        self._sections = sections
        self._encoder = encoder
        self._vectors = vectors

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """The top k articles for a query, best first; each hit carries its best section."""
        (query_vector,) = self._encoder.encode_queries([query])
        ranked = (
            Hit(self._sections[row].article_id, score, section=self._sections[row].text)
            for row, score in ranked_rows(query_vector, self._vectors)
        )
        return best_per_article(ranked, k)
