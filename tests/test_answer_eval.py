import pytest

from support_rag.answer_eval import (
    answer_question,
    context_for,
    pick_evenly,
    summarize_answers,
)
from support_rag.llm import LLMResult
from support_rag.records import Article, Hit, Question

ARTICLES = {
    article_id: Article(article_id, f"Title {article_id}", "", "article", f"Text {article_id}", "")
    for article_id in ("A", "B", "C")
}
QUESTION = Question("ew-000", "expertwritten", "How do I connect a domain?", "", ("C", "A"))


class FixedRetriever:
    def search(self, query, k=10):
        return [Hit("B", 2.0), Hit("C", 1.0)][:k]


class FakeLLM:
    def __init__(self, text, cached=False):
        self.text = text
        self.cached = cached
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        usage = {"input_tokens": 1000, "output_tokens": 200}
        return LLMResult(self.text, "end_turn", usage, 1500.0, self.cached, request["model"])


def test_each_strategy_shows_the_right_articles():
    def ids(strategy):
        context = context_for(
            strategy, QUESTION, articles_by_id=ARTICLES, retriever=FixedRetriever(), k=2
        )
        return [article.id for article in context]

    assert ids("s0") == []
    assert ids("s1") == ["B", "C"]
    assert ids("s_star") == ["C", "A"]


def test_s1_without_a_retriever_is_an_error():
    with pytest.raises(ValueError):
        context_for("s1", QUESTION, articles_by_id=ARTICLES)


def test_answer_maps_citations_through_the_context_order():
    llm = FakeLLM("Point the DNS here [2], then wait [1].")
    context = [ARTICLES["B"], ARTICLES["C"]]
    result = answer_question(llm, QUESTION, context, effort="medium")
    assert result.context_ids == ("B", "C")
    assert result.cited_ids == ("C", "B")
    assert result.cost_usd == pytest.approx((1000 * 0.10 + 200 * 0.50) / 1_000_000)
    assert llm.requests[0]["output_config"] == {"effort": "medium"}


def test_summary_counts_paid_calls_and_latency_from_real_calls_only():
    paid = answer_question(FakeLLM("Yes [1]."), QUESTION, [ARTICLES["A"]], effort="medium")
    replayed = answer_question(
        FakeLLM("NOT_FOUND", cached=True), QUESTION, [ARTICLES["A"]], effort="medium"
    )
    summary = summarize_answers([paid, replayed])
    assert summary["n"] == 2
    assert summary["declined"] == 1
    assert summary["cached"] == 1
    assert summary["paid_this_run_usd"] == pytest.approx(paid.cost_usd, abs=1e-4)
    assert summary["cost_of_answers_usd"] == pytest.approx(2 * paid.cost_usd, abs=1e-4)
    assert summary["latency_p50_ms"] == 1500.0


def test_a_pilot_spreads_its_questions_over_the_list():
    questions = [Question(f"q{i}", "s", "", "", ("A",)) for i in range(50)]
    assert [q.id for q in pick_evenly(questions, 5)] == ["q0", "q10", "q20", "q30", "q40"]
    assert len(pick_evenly(questions, None)) == 50
