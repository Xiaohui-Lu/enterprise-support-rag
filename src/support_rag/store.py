"""Postgres with pgvector: the articles and their vectors in one database (D-041).

Exact search scans every vector and sorts, as NumPy does, so it must return NumPy's
ranking. An HNSW index answers approximately, and much faster at scale. `srag db-load`
fills the tables from the stored vectors, so it never runs the embedding model.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from support_rag.embeddings import EmbeddingModel, Encoder, corpus_fingerprint
from support_rag.records import Article, Hit

if TYPE_CHECKING:
    import psycopg

INDEXES = ("exact", "hnsw")
HNSW_INDEX = "article_vectors_hnsw"
HNSW_SETTINGS = {"m": 16, "ef_construction": 64, "ef_search": 40}  # pgvector's defaults

# `position` is each article's row in the pinned file. Exact search sorts by it second,
# so equal scores come out in the same order as NumPy's stable sort.
_SCHEMA = """
DROP TABLE IF EXISTS article_vectors, articles, corpus;
CREATE TABLE corpus (
    model text NOT NULL,
    revision text NOT NULL,
    fingerprint text NOT NULL,
    dimensions integer NOT NULL,
    loaded_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE articles (
    position integer PRIMARY KEY,
    id text NOT NULL UNIQUE,
    title text NOT NULL,
    url text NOT NULL,
    article_type text NOT NULL,
    contents text NOT NULL
);
CREATE TABLE article_vectors (
    position integer PRIMARY KEY REFERENCES articles (position),
    article_id text NOT NULL REFERENCES articles (id),
    embedding vector({dimensions}) NOT NULL
);
-- Keep each vector in its row: values over about 2 KB otherwise move to a side table,
-- and a scan would fetch every vector from there.
ALTER TABLE article_vectors ALTER COLUMN embedding SET STORAGE PLAIN;
"""

# Inner product: the vectors have length 1, so it ranks like cosine. pgvector's <#> is the
# negative inner product, so smaller is closer and the score is its negation.
EXACT_SQL = """
SELECT article_id, -(embedding <#> %(query)s) AS score
FROM article_vectors
ORDER BY embedding <#> %(query)s, position
LIMIT %(k)s
"""
# An index can serve only a single ORDER BY on the distance, so ties here are arbitrary.
HNSW_SQL = """
SELECT article_id, -(embedding <#> %(query)s) AS score
FROM article_vectors
ORDER BY embedding <#> %(query)s
LIMIT %(k)s
"""


def connect(url: str) -> psycopg.Connection:
    """A connection that can send and receive vectors. Creates the extension if needed."""
    import psycopg
    from pgvector.psycopg import register_vector

    connection = psycopg.connect(url, autocommit=True)
    connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(connection)
    return connection


def load(
    connection: psycopg.Connection,
    articles: Sequence[Article],
    vectors: np.ndarray,
    model: EmbeddingModel,
    *,
    hnsw: bool = True,
) -> dict[str, Any]:
    """Replace the tables with these articles and their vectors, in one transaction, then
    build the HNSW index if asked. Returns what was loaded and how long it took."""
    if vectors.shape != (len(articles), model.dimensions):
        raise ValueError(f"vectors of shape {vectors.shape} for {len(articles)} articles")
    start = time.perf_counter()
    with connection.transaction():
        connection.execute(_SCHEMA.format(dimensions=model.dimensions))
        with connection.cursor() as cursor:
            columns = "position, id, title, url, article_type, contents"
            with cursor.copy(f"COPY articles ({columns}) FROM STDIN") as copy:
                for position, a in enumerate(articles):
                    copy.write_row((position, a.id, a.title, a.url, a.article_type, a.contents))
            sql = (
                "COPY article_vectors (position, article_id, embedding) FROM STDIN (FORMAT BINARY)"
            )
            with cursor.copy(sql) as copy:
                copy.set_types(["int4", "text", "vector"])
                for position, (article, vector) in enumerate(zip(articles, vectors, strict=True)):
                    copy.write_row((position, article.id, vector))
        connection.execute(
            "INSERT INTO corpus (model, revision, fingerprint, dimensions) VALUES (%s, %s, %s, %s)",
            (model.name, model.revision, corpus_fingerprint(articles), model.dimensions),
        )
    load_seconds = time.perf_counter() - start

    build_seconds = None
    if hnsw:
        start = time.perf_counter()
        connection.execute(
            f"CREATE INDEX {HNSW_INDEX} ON article_vectors USING hnsw (embedding vector_ip_ops) "
            f"WITH (m = {HNSW_SETTINGS['m']}, ef_construction = {HNSW_SETTINGS['ef_construction']})"
        )
        build_seconds = time.perf_counter() - start
    connection.execute("ANALYZE articles, article_vectors")
    return {
        "articles": len(articles),
        "load_seconds": round(load_seconds, 2),
        "hnsw_build_seconds": None if build_seconds is None else round(build_seconds, 2),
        "table_bytes": _size(connection, "pg_total_relation_size('article_vectors')"),
        "hnsw_bytes": _size(connection, f"pg_relation_size('{HNSW_INDEX}')") if hnsw else None,
    }


def _size(connection: psycopg.Connection, expression: str) -> int:
    return int(connection.execute(f"SELECT {expression}").fetchone()[0])


def plan_nodes(plan: dict[str, Any]) -> Iterator[str]:
    """Every node of an EXPLAIN (FORMAT JSON) plan, top down, as "Node Type" or "Node Type on
    index". Used to check which way a query really runs."""
    node = plan["Node Type"]
    yield f"{node} on {plan['Index Name']}" if "Index Name" in plan else node
    for child in plan.get("Plans", []):
        yield from plan_nodes(child)


class PgVectorRetriever:
    """Whole articles ranked by inner product in Postgres, either exactly or through the
    HNSW index (D-041)."""

    def __init__(
        self,
        connection: psycopg.Connection,
        encoder: Encoder,
        *,
        model: EmbeddingModel,
        articles: Sequence[Article],
        index: str = "exact",
    ) -> None:
        if index not in INDEXES:
            raise ValueError(f"unknown index {index!r}; choose from {INDEXES}")
        self._check_corpus(connection, model, articles)
        if index == "hnsw":
            # The planner may still prefer a scan on a small table, so scans are ruled out.
            connection.execute("SET enable_seqscan = off")
            connection.execute(f"SET hnsw.ef_search = {HNSW_SETTINGS['ef_search']}")
            self._sql = HNSW_SQL
        else:
            connection.execute("SET enable_indexscan = off")
            self._sql = EXACT_SQL
        self._connection = connection
        self._encoder = encoder
        self.last_search_ms: float | None = None  # the SQL alone, for D-041's bar
        self.plan = self._checked_plan(index, np.zeros(model.dimensions, dtype=np.float32))

    @staticmethod
    def _check_corpus(
        connection: psycopg.Connection, model: EmbeddingModel, articles: Sequence[Article]
    ) -> None:
        import psycopg

        try:
            row = connection.execute(
                "SELECT model, revision, fingerprint, dimensions FROM corpus"
            ).fetchone()
        except psycopg.errors.UndefinedTable:
            raise ValueError("the database has no vectors yet; run `srag db-load`") from None
        expected = (model.name, model.revision, corpus_fingerprint(articles), model.dimensions)
        if row != expected:
            raise ValueError(
                "the database holds vectors for another model or corpus; run `srag db-load`"
            )

    def _checked_plan(self, index: str, query: np.ndarray) -> str:
        """The query's plan, refusing one that doesn't search the way it's meant to."""
        explain = f"EXPLAIN (FORMAT JSON) {self._sql}"
        (result,) = self._connection.execute(explain, {"query": query, "k": 20}).fetchone()
        nodes = list(plan_nodes(result[0]["Plan"]))
        uses_index = f"Index Scan on {HNSW_INDEX}" in nodes
        if uses_index != (index == "hnsw"):
            want = "the HNSW index" if index == "hnsw" else "a full scan"
            raise ValueError(f"expected {want}, but the plan is {' > '.join(nodes)}")
        return " > ".join(nodes)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        """The top k articles for a query, best first."""
        (vector,) = self._encoder.encode_queries([query])
        start = time.perf_counter()
        rows = self._connection.execute(self._sql, {"query": vector, "k": k}).fetchall()
        self.last_search_ms = (time.perf_counter() - start) * 1000
        return [Hit(article_id, float(score)) for article_id, score in rows]
