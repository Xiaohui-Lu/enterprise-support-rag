import numpy as np
import pytest

from support_rag.dense import DenseRetriever, SectionRetriever, best_per_article, top_k
from support_rag.embeddings import unit_rows
from support_rag.records import Article, Hit, Section

ROWS = np.array([[0.0, 1.0], [0.6, 0.8], [1.0, 0.0]], dtype=np.float32)
QUERY = np.array([1.0, 0.0], dtype=np.float32)


def test_top_k_returns_the_most_similar_rows_best_first():
    assert top_k(QUERY, ROWS, 2) == [(2, pytest.approx(1.0)), (1, pytest.approx(0.6))]


def test_top_k_scores_are_dot_products():
    query = unit_rows(np.array([[1.0, 1.0]]))[0]
    hits = top_k(query, ROWS, 3)
    for row, similarity in hits:
        assert similarity == pytest.approx(float(ROWS[row] @ query))


def test_top_k_with_k_larger_than_the_matrix_returns_every_row():
    hits = top_k(QUERY, ROWS, 10)
    assert [row for row, _ in hits] == [2, 1, 0]


def test_top_k_returns_python_ints_and_floats():
    row, similarity = top_k(QUERY, ROWS, 1)[0]
    assert type(row) is int
    assert type(similarity) is float


def test_top_k_breaks_ties_by_row_order():
    rows = np.array([[0.6, 0.8], [1.0, 0.0], [0.6, 0.8], [1.0, 0.0]], dtype=np.float32)
    assert [row for row, _ in top_k(QUERY, rows, 4)] == [1, 3, 0, 2]


def test_top_k_finds_the_best_rows_in_a_larger_matrix():
    rng = np.random.default_rng(0)
    matrix = unit_rows(rng.normal(size=(1000, 8)))
    query = unit_rows(rng.normal(size=(1, 8)))[0]
    expected = np.argsort(-(matrix @ query))[:5]
    assert [row for row, _ in top_k(query, matrix, 5)] == expected.tolist()


class FixedEncoder:
    def encode_queries(self, texts):
        return np.array([QUERY for _ in texts])


def _article(article_id):
    return Article(article_id, "", "", "article", "", "")


def test_dense_retriever_maps_rows_to_article_ids():
    retriever = DenseRetriever([_article(i) for i in "abc"], FixedEncoder(), ROWS)
    hits = retriever.search("anything", k=2)
    assert [hit.article_id for hit in hits] == ["c", "b"]


def test_dense_retriever_rejects_vectors_that_do_not_match_the_articles():
    with pytest.raises(ValueError):
        DenseRetriever([_article("a")], FixedEncoder(), ROWS)


def test_best_per_article_keeps_each_article_once_at_its_best_section():
    ranked = [Hit("a", 0.9), Hit("b", 0.8), Hit("a", 0.7), Hit("c", 0.6)]
    assert best_per_article(ranked, 10) == [Hit("a", 0.9), Hit("b", 0.8), Hit("c", 0.6)]


def test_best_per_article_counts_articles_not_sections():
    ranked = [Hit("a", 0.9), Hit("a", 0.8), Hit("b", 0.7), Hit("c", 0.6)]
    assert best_per_article(ranked, 2) == [Hit("a", 0.9), Hit("b", 0.7)]


def test_best_per_article_with_k_zero_returns_nothing():
    assert best_per_article([Hit("a", 0.9)], 0) == []


def test_best_per_article_stops_reading_once_it_has_k_articles():
    ranked = iter([Hit("a", 0.9), Hit("b", 0.8), Hit("c", 0.7), Hit("d", 0.6)])
    best_per_article(ranked, 2)
    assert next(ranked) == Hit("d", 0.6)


def test_section_retriever_scores_an_article_by_its_best_section():
    sections = [Section("a", "off topic"), Section("b", "close"), Section("a", "exact")]
    hits = SectionRetriever(sections, FixedEncoder(), ROWS).search("anything", k=10)
    assert [(hit.article_id, hit.section) for hit in hits] == [("a", "exact"), ("b", "close")]
    assert hits[0].score == pytest.approx(1.0)


def test_section_retriever_refuses_vectors_that_do_not_match_its_sections():
    with pytest.raises(ValueError):
        SectionRetriever([Section("a", "only one")], FixedEncoder(), ROWS)
