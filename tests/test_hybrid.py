from pathlib import Path

import pytest

from support_rag.hybrid import FUSION_SETTINGS, FUSIONS, HybridRetriever, rrf, weighted
from support_rag.records import Hit


def _hits(*article_ids, scores=None):
    scores = scores or [1.0 - i / 100 for i in range(len(article_ids))]
    return [Hit(a, s) for a, s in zip(article_ids, scores, strict=True)]


def _ids(hits):
    return [hit.article_id for hit in hits]


def test_rrf_adds_one_over_k_plus_rank_with_rank_starting_at_one():
    hits = {hit.article_id: hit.score for hit in rrf(_hits("a", "b"), _hits("b", "c"))}
    assert hits == {
        "a": pytest.approx(1 / 61),
        "b": pytest.approx(1 / 62 + 1 / 61),
        "c": pytest.approx(1 / 62),
    }


def test_rrf_returns_the_best_article_first():
    assert _ids(rrf(_hits("a", "b"), _hits("b", "c"))) == ["b", "a", "c"]


def test_rrf_ranks_agreement_above_one_strong_list():
    # y is rank 30 in both lists, and x is rank 1 in dense and missing from keyword.
    fillers = [f"f{i:02}" for i in range(28)]
    dense = _hits("x", *fillers, "y")
    keyword = _hits(*[f"g{i:02}" for i in range(29)], "y")
    scores = {hit.article_id: hit.score for hit in rrf(dense, keyword)}
    assert scores["y"] == pytest.approx(2 / 90)
    assert scores["x"] == pytest.approx(1 / 61)
    assert scores["y"] > scores["x"]


def test_rrf_ignores_how_large_the_scores_are():
    small = rrf(_hits("a", "b", scores=[0.9, 0.1]), _hits("b", "c", scores=[0.9, 0.1]))
    large = rrf(_hits("a", "b", scores=[900.0, 1.0]), _hits("b", "c", scores=[50.0, 49.0]))
    assert small == large


def test_equal_scores_are_ordered_by_article_id():
    # Each is rank 1 in one list, so they tie at 1/61.
    assert _ids(rrf(_hits("z"), _hits("a"))) == ["a", "z"]
    assert _ids(weighted(_hits("z"), _hits("a"))) == ["a", "z"]


def test_rrf_with_one_empty_list_keeps_the_other_lists_order():
    assert _ids(rrf(_hits("a", "b"), [])) == ["a", "b"]


def test_weighted_scales_each_list_before_combining():
    # BM25's larger numbers must not decide: after scaling, a and c tie at 0.5.
    keyword = _hits("a", "b", "low1", scores=[23.4, 11.0, 5.0])
    dense = _hits("c", "b", "low2", scores=[0.74, 0.61, 0.40])
    scores = {hit.article_id: hit.score for hit in weighted(dense, keyword)}
    assert scores["a"] == pytest.approx(0.5)
    assert scores["c"] == pytest.approx(0.5)
    assert scores["b"] == pytest.approx(0.5 * (0.21 / 0.34) + 0.5 * (6.0 / 18.4))


def test_weighted_counts_a_list_with_no_range_as_one():
    hits = weighted(_hits("a", scores=[0.3]), _hits("b", "c", scores=[2.0, 2.0]))
    assert {hit.article_id: hit.score for hit in hits} == {"a": 0.5, "b": 0.5, "c": 0.5}


def test_weighted_with_no_keyword_hits_still_ranks_the_dense_list():
    assert _ids(weighted(_hits("a", "b", scores=[0.9, 0.1]), [])) == ["a", "b"]


class FixedRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.asked_for = None

    def search(self, query, k=10):
        self.asked_for = k
        return self.hits[:k]


def test_hybrid_asks_each_retriever_for_the_depth_and_returns_only_k():
    dense = FixedRetriever(_hits("a", "b", "c"))
    keyword = FixedRetriever(_hits("c", "d"))
    hits = HybridRetriever(dense, keyword, rrf, depth=50).search("question", k=2)
    assert (dense.asked_for, keyword.asked_for) == (50, 50)
    assert _ids(hits) == ["c", "a"]


def test_every_fusion_has_settings_for_the_run_record():
    assert FUSION_SETTINGS.keys() == FUSIONS.keys()


def test_the_cli_offers_every_fusion():
    from support_rag import cli

    source = Path(cli.__file__).read_text(encoding="utf-8")
    for name in FUSIONS:
        assert f'"{name}"' in source
