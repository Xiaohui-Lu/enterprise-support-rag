"""The Postgres store. Tests that need a database are skipped unless SRAG_TEST_DATABASE_URL
points at one (D-041): a separate database, since loading replaces its tables."""

import os

import numpy as np
import pytest

from support_rag.dense import top_k
from support_rag.embeddings import EmbeddingModel, unit_rows
from support_rag.records import Article
from support_rag.store import EXACT_SQL, HNSW_SQL, PgVectorRetriever, connect, load, plan_nodes

TEST_URL = os.environ.get("SRAG_TEST_DATABASE_URL")
needs_db = pytest.mark.skipif(not TEST_URL, reason="SRAG_TEST_DATABASE_URL isn't set")

MODEL = EmbeddingModel("tiny", "test/tiny", "0" * 40, 3)


def test_plan_nodes_names_each_node_and_its_index():
    plan = {
        "Node Type": "Limit",
        "Plans": [{"Node Type": "Index Scan", "Index Name": "article_vectors_hnsw"}],
    }
    assert list(plan_nodes(plan)) == ["Limit", "Index Scan on article_vectors_hnsw"]


def test_exact_search_breaks_ties_by_position_and_hnsw_sorts_by_distance_alone():
    assert "ORDER BY embedding <#> %(query)s, position" in EXACT_SQL
    assert "ORDER BY embedding <#> %(query)s\n" in HNSW_SQL


class FixedEncoder:
    def __init__(self, vector):
        self.vector = vector

    def encode_queries(self, texts):
        return np.array([self.vector for _ in texts], dtype=np.float32)


def _corpus(n=40, seed=0):
    rng = np.random.default_rng(seed)
    articles = [Article(f"a{i}", f"T{i}", "", "article", f"text {i}", "") for i in range(n)]
    return articles, unit_rows(rng.normal(size=(n, 3)))


@pytest.fixture
def connection():
    conn = connect(TEST_URL)
    yield conn
    conn.close()


@needs_db
def test_exact_search_returns_numpys_ranking(connection):
    articles, vectors = _corpus()
    load(connection, articles, vectors, MODEL, hnsw=False)
    rng = np.random.default_rng(1)
    for query in unit_rows(rng.normal(size=(20, 3))):
        retriever = PgVectorRetriever(
            connection, FixedEncoder(query), model=MODEL, articles=articles
        )
        hits = retriever.search("q", k=10)
        expected = top_k(query, vectors, 10)
        assert [h.article_id for h in hits] == [articles[row].id for row, _ in expected]
        assert [h.score for h in hits] == pytest.approx([s for _, s in expected], abs=1e-6)


@needs_db
def test_exact_search_keeps_file_order_on_equal_scores(connection):
    articles, _ = _corpus(n=3)
    vectors = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=np.float32)
    load(connection, articles, vectors, MODEL, hnsw=False)
    retriever = PgVectorRetriever(
        connection, FixedEncoder(np.array([1, 0, 0])), model=MODEL, articles=articles
    )
    assert [h.article_id for h in retriever.search("q", k=3)] == ["a0", "a2", "a1"]


@needs_db
def test_hnsw_search_uses_the_index(connection):
    articles, vectors = _corpus()
    stats = load(connection, articles, vectors, MODEL, hnsw=True)
    retriever = PgVectorRetriever(
        connection, FixedEncoder(vectors[0]), model=MODEL, articles=articles, index="hnsw"
    )
    assert "Index Scan on article_vectors_hnsw" in retriever.plan
    assert retriever.search("q", k=1)[0].article_id == "a0"
    assert retriever.last_search_ms is not None
    assert stats["hnsw_bytes"] > 0


@needs_db
def test_hnsw_search_without_the_index_is_refused(connection):
    articles, vectors = _corpus()
    load(connection, articles, vectors, MODEL, hnsw=False)
    with pytest.raises(ValueError, match="HNSW index"):
        PgVectorRetriever(
            connection, FixedEncoder(vectors[0]), model=MODEL, articles=articles, index="hnsw"
        )


@needs_db
def test_vectors_from_another_corpus_are_refused(connection):
    articles, vectors = _corpus()
    load(connection, articles, vectors, MODEL, hnsw=False)
    changed = [*articles[:-1], Article("new", "T", "", "article", "other text", "")]
    with pytest.raises(ValueError, match="another model or corpus"):
        PgVectorRetriever(connection, FixedEncoder(vectors[0]), model=MODEL, articles=changed)


@needs_db
def test_loading_again_replaces_the_tables(connection):
    articles, vectors = _corpus()
    load(connection, articles, vectors, MODEL, hnsw=False)
    load(connection, articles[:5], vectors[:5], MODEL, hnsw=False)
    count = connection.execute("SELECT count(*) FROM article_vectors").fetchone()[0]
    assert count == 5
