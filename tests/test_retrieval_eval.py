import pytest

from support_rag.records import Hit, Question
from support_rag.retrieval_eval import QuestionResult, evaluate_question, summarize


class FixedRetriever:
    """Returns the same ranking for every query."""

    def __init__(self, ranked_ids):
        self.ranked_ids = ranked_ids

    def search(self, query, k=10):
        return [Hit(article_id, 1.0) for article_id in self.ranked_ids[:k]]


def _question(gold_ids, subset="expertwritten"):
    return Question("ew-000", subset, "How do I connect a domain?", "", tuple(gold_ids))


def _result(question_id, subset, multi, recall_at_5, mrr):
    return QuestionResult(
        question_id=question_id,
        subset=subset,
        multi_article=multi,
        gold_ids=("A",),
        ranked_ids=(),
        scores=(),
        metrics={"recall@5": recall_at_5, "mrr@20": mrr},
        latency_ms=0.1,
    )


# evaluate_question()


def test_evaluate_question_scores_the_ranking_at_every_k():
    retriever = FixedRetriever(["C", "B", "D", "E", "F", "A"])
    result = evaluate_question(retriever, _question(["A", "B"]))
    assert result.ranked_ids == ("C", "B", "D", "E", "F", "A")
    assert result.multi_article is True
    assert result.metrics["recall@1"] == 0.0
    assert result.metrics["recall@5"] == 0.5
    assert result.metrics["recall@10"] == 1.0
    assert result.metrics["all_gold@5"] == 0.0
    assert result.metrics["all_gold@10"] == 1.0
    assert result.metrics["mrr@20"] == 0.5


class TimedRetriever(FixedRetriever):
    last_search_ms = 1.23456


def test_evaluate_question_keeps_the_search_time_a_retriever_reports():
    result = evaluate_question(TimedRetriever(["A"]), _question(["A"]))
    assert result.search_ms == 1.235


def test_evaluate_question_leaves_search_time_empty_when_not_reported():
    assert evaluate_question(FixedRetriever(["A"]), _question(["A"])).search_ms is None


def test_evaluate_question_asks_the_retriever_for_the_top_20():
    retriever = FixedRetriever([f"x{i}" for i in range(30)])
    result = evaluate_question(retriever, _question(["A"]))
    assert len(result.ranked_ids) == 20


# summarize()

RESULTS = [
    _result("ew-000", "expertwritten", False, recall_at_5=1.0, mrr=1.0),
    _result("ew-001", "expertwritten", True, recall_at_5=0.5, mrr=0.5),
    _result("sim-000", "simulated", False, recall_at_5=0.0, mrr=0.0),
    _result("syn-0000", "synthetic", False, recall_at_5=1.0, mrr=0.25),
]


def test_summarize_averages_each_metric_over_all_questions():
    summary = summarize(RESULTS)
    assert summary["all"]["n"] == 4
    assert summary["all"]["recall@5"] == pytest.approx((1.0 + 0.5 + 0.0 + 1.0) / 4)
    assert summary["all"]["mrr@20"] == pytest.approx((1.0 + 0.5 + 0.0 + 0.25) / 4)


def test_summarize_groups_by_subset():
    summary = summarize(RESULTS)
    assert summary["expertwritten"] == {"n": 2, "recall@5": 0.75, "mrr@20": 0.75}
    assert summary["simulated"] == {"n": 1, "recall@5": 0.0, "mrr@20": 0.0}
    assert summary["synthetic"] == {"n": 1, "recall@5": 1.0, "mrr@20": 0.25}


def test_summarize_groups_by_single_and_multi_article():
    summary = summarize(RESULTS)
    assert summary["multi"] == {"n": 1, "recall@5": 0.5, "mrr@20": 0.5}
    assert summary["single"]["n"] == 3
    assert summary["single"]["recall@5"] == pytest.approx(2.0 / 3)


def test_summarize_leaves_out_groups_without_questions():
    synthetic_only = [r for r in RESULTS if r.subset == "synthetic"]
    assert set(summarize(synthetic_only)) == {"all", "synthetic", "single"}
