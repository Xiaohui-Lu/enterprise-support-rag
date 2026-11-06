import math
from pathlib import Path

import pytest

from support_rag.records import Article, Hit
from support_rag.rerank import (
    RERANKERS,
    RerankingRetriever,
    SectionRerankingRetriever,
    best_first,
    checked_scores,
    merged_candidates,
)


def _ids(hits):
    return [hit.article_id for hit in hits]


def _hits(*article_ids):
    return [Hit(a, 1.0) for a in article_ids]


def _articles(*article_ids):
    return [Article(a, a, "", "article", f"text of {a}", "") for a in article_ids]


class FixedRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.asked_for = None

    def search(self, query, k=10):
        self.asked_for = k
        return self.hits[:k]


class TableScorer:
    """Scores each article's text from a table, and remembers what it read."""

    def __init__(self, scores):
        self.scores = scores
        self.read = []

    def score(self, query, documents):
        self.read = list(documents)
        return [self.scores[document.removeprefix("text of ")] for document in documents]


def test_best_first_puts_the_highest_score_first():
    assert best_first(["a", "b", "c"], [0.1, 0.9, -2.0]) == [
        Hit("b", 0.9),
        Hit("a", 0.1),
        Hit("c", -2.0),
    ]


def test_equal_scores_keep_the_order_they_came_in():
    assert _ids(best_first(["z", "a", "m"], [1.0, 1.0, 1.0])) == ["z", "a", "m"]


def test_best_first_refuses_a_different_number_of_scores():
    with pytest.raises(ValueError):
        best_first(["a", "b"], [1.0])


def test_candidates_keep_their_order_and_appear_once():
    rankings = [_hits("c", "a"), _hits("a", "d", "b")]
    assert merged_candidates(rankings) == ["c", "a", "d", "b"]


def test_a_candidate_on_both_lists_is_read_once():
    scorer = TableScorer({"a": 1.0, "b": 2.0, "c": 3.0})
    retrievers = [FixedRetriever(_hits("a", "b")), FixedRetriever(_hits("b", "c"))]
    RerankingRetriever(_articles("a", "b", "c"), retrievers, scorer).search("question")
    assert scorer.read == ["text of a", "text of b", "text of c"]


def test_the_last_candidate_can_move_to_the_top():
    scorer = TableScorer({"a": -1.0, "b": 0.0, "c": 5.0})
    retriever = RerankingRetriever(
        _articles("a", "b", "c"), [FixedRetriever(_hits("a", "b", "c"))], scorer
    )
    assert _ids(retriever.search("question")) == ["c", "b", "a"]


def test_an_article_off_the_shortlist_is_never_returned():
    scorer = TableScorer({"a": 1.0, "b": 99.0})
    retriever = RerankingRetriever(_articles("a", "b"), [FixedRetriever(_hits("a"))], scorer)
    assert _ids(retriever.search("question")) == ["a"]


def test_it_asks_each_retriever_for_the_depth_and_returns_only_k():
    scorer = TableScorer({"a": 1.0, "b": 2.0, "c": 3.0})
    dense, keyword = FixedRetriever(_hits("a", "b")), FixedRetriever(_hits("c"))
    retriever = RerankingRetriever(_articles("a", "b", "c"), [dense, keyword], scorer, depth=50)
    assert _ids(retriever.search("question", k=2)) == ["c", "b"]
    assert (dense.asked_for, keyword.asked_for) == (50, 50)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_scores_that_are_not_finite_are_refused(bad):
    with pytest.raises(ValueError):
        checked_scores([1.0, bad])


def test_every_reranker_is_pinned_to_a_full_revision():
    for model in RERANKERS.values():
        assert len(model.revision) == 40


def test_the_cli_offers_every_reranker():
    from support_rag import cli

    source = Path(cli.__file__).read_text(encoding="utf-8")
    for name in RERANKERS:
        assert f'"{name}"' in source


def test_section_reranker_reads_each_articles_best_section_not_the_article():
    hits = [Hit("a", 0.9, section="text of a1"), Hit("b", 0.8, section="text of b1")]
    scorer = TableScorer({"a1": 1.0, "b1": 2.0})
    found = SectionRerankingRetriever(FixedRetriever(hits), scorer).search("q", k=2)
    assert scorer.read == ["text of a1", "text of b1"]
    assert [(hit.article_id, hit.score) for hit in found] == [("b", 2.0), ("a", 1.0)]


def test_section_reranker_asks_for_its_depth_of_articles_and_returns_k():
    hits = [Hit(a, 1.0, section=f"text of {a}") for a in "abcd"]
    retriever = FixedRetriever(hits)
    scorer = TableScorer(dict.fromkeys("abcd", 1.0))
    found = SectionRerankingRetriever(retriever, scorer, depth=3).search("q", k=2)
    assert retriever.asked_for == 3
    assert _ids(found) == ["a", "b"]
