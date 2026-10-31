import pytest

from support_rag.metrics import all_gold_at_k, recall_at_k, reciprocal_rank

RANKED = ["C", "B", "D", "E", "F", "A"]
GOLD = ("A", "B")  # a tuple, like Question.gold_ids


def test_recall_counts_the_share_of_gold_articles_in_the_top_k():
    assert recall_at_k(RANKED, GOLD, k=1) == 0.0
    assert recall_at_k(RANKED, GOLD, k=5) == 0.5
    assert recall_at_k(RANKED, GOLD, k=6) == 1.0


def test_recall_handles_a_list_shorter_than_k():
    assert recall_at_k(["A"], GOLD, k=5) == 0.5


def test_recall_of_a_single_gold_article_is_all_or_nothing():
    assert recall_at_k(RANKED, ("D",), k=2) == 0.0
    assert recall_at_k(RANKED, ("D",), k=3) == 1.0


def test_all_gold_needs_every_gold_article_in_the_top_k():
    assert all_gold_at_k(RANKED, GOLD, k=5) is False
    assert all_gold_at_k(RANKED, GOLD, k=6) is True


def test_reciprocal_rank_uses_only_the_first_gold_article():
    assert reciprocal_rank(RANKED, GOLD) == 0.5


def test_reciprocal_rank_of_a_gold_article_at_the_top_is_one():
    assert reciprocal_rank(["A", "C"], GOLD) == 1.0


def test_reciprocal_rank_is_zero_when_no_gold_article_is_retrieved():
    assert reciprocal_rank(["C", "D"], GOLD) == 0.0
    assert reciprocal_rank([], GOLD) == 0.0


def test_reciprocal_rank_at_a_deep_rank():
    ranked = [f"x{i}" for i in range(19)] + ["A"]
    assert reciprocal_rank(ranked, GOLD) == pytest.approx(1 / 20)
